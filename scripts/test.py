"""
Real-world ZeroMQ inference server for the Qwen3-VL MoT VLA model
with cascaded flow matching and flare visual prediction tokens.

Stateful slow/fast protocol:
  slow request → forward_flow_action_partial → cache (latent + action) KV
                 at τ=τ_split, return [] (no usable action without a fast tick).
  fast request → tactile_flow_continue on cached KV with fresh tactile,
                 returning the final action chunk directly (no Â + Δa add).
  slow_and_fast → run both in sequence (typical at chunk start).

The client orchestrates cadence (e.g. slow every 16 robot steps, fast at
offsets 0, 4, 8, 12).  ZMQ REP is single-threaded so a fast request
arriving mid-slow naturally waits until slow finishes — the "if a
refinement has not finished, wait until it finishes" guarantee.
"""

import os
import sys
import threading

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

import argparse
import hashlib
import json
import io
import pickle
import time
import traceback

import numpy as np
import torch
from PIL import Image
import zmq
from transformers import AutoProcessor
from qwen_vla import Qwen3VLVLAModel, extend_position_ids_for_flare, split_slow_fast_embeds


OFFICIAL_SINGLE_VIEW_PROFILE = "official_single_view"
REVO3_FULL_CENTER_PROFILE = "revo3_full_center_v1"
CAMERA_PROFILES = (OFFICIAL_SINGLE_VIEW_PROFILE, REVO3_FULL_CENTER_PROFILE)


def _validated_camera_profile(args):
    profile = str(getattr(args, "camera_profile", OFFICIAL_SINGLE_VIEW_PROFILE))
    if profile not in CAMERA_PROFILES:
        raise ValueError(f"unsupported camera_profile={profile!r}")
    if profile == REVO3_FULL_CENTER_PROFILE:
        if tuple(getattr(args, "image_size", ()) or ()) != (384, 288):
            raise ValueError(
                "revo3_full_center_v1 requires --image_size 384 288; refusing "
                "to silently change the trained visual contract."
            )
    return profile


def _normalize(values, mask, vmin, vmax):
    return np.where(
        mask,
        np.clip(2.0 * (values - vmin) / (vmax - vmin + 1e-8) - 1.0, -1.0, 1.0),
        values,
    )

def _denormalize(norm_values, mask, vmin, vmax):
    return np.where(
        mask,
        0.5 * (norm_values + 1.0) * (vmax - vmin) + vmin,
        norm_values,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Build model from config.json
# ─────────────────────────────────────────────────────────────────────────────

def _build_qwen3vl_from_config(config_path, args):
    with open(config_path) as f:
        full_cfg = json.load(f)

    image_token_id = full_cfg.get("image_token_id", 151655)
    model_type = full_cfg.get("model_type", "qwen2_vl")

    try:
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
        vl_config = Qwen3VLConfig(**{k: v for k, v in full_cfg.items()
                                     if k not in ("architectures", "transformers_version")})
        text_config = vl_config.text_config
    except Exception:
        from transformers import AutoConfig
        vl_config = AutoConfig.from_pretrained(
            os.path.dirname(config_path), trust_remote_code=True)
        text_config = getattr(vl_config, "text_config", vl_config)

    tac_isize = getattr(args, "tactile_intermediate_size", 0)
    tac_isize = tac_isize if tac_isize > 0 else None
    n_flare_tpf = getattr(args, "n_flare_tokens_per_frame", 0)
    n_flare_steps = getattr(args, "n_flare_steps", 0)

    model = Qwen3VLVLAModel(
        config             = text_config,
        action_dim         = args.action_dim,
        action_chunk       = args.action_chunk,
        tactile_num_fingers= args.tactile_num_fingers,
        use_tactile_deform = bool(args.use_tactile_deform),
        use_robot_state    = bool(args.use_robot_state),
        image_token_id     = image_token_id,
        tactile_intermediate_size = tac_isize,
        n_flare_tokens_per_frame = n_flare_tpf,
        n_flare_steps            = n_flare_steps,
        use_tactile_code         = bool(getattr(args, "use_tactile_code", 0)),
        vqvae_codebook_size      = getattr(args, "vqvae_codebook_size", 64),
        use_tactile_vqvae        = bool(getattr(args, "use_tactile_vqvae", 0)),
        vqvae_config             = getattr(args, "vqvae_config", None),
    )

    vis_cfg_dict = full_cfg.get("vision_config", {})
    try:
        if model_type == "qwen3_vl":
            from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
            vis_cfg = Qwen3VLVisionConfig(**{k: v for k, v in vis_cfg_dict.items()
                                             if k != "model_type"})
            model.visual = Qwen3VLVisionModel(vis_cfg)
        else:
            from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig
            from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLVisionModel
            vis_cfg = Qwen2VLVisionConfig(**{k: v for k, v in vis_cfg_dict.items()
                                             if k != "model_type"})
            model.visual = Qwen2VLVisionModel(vis_cfg)
        print(f"  Visual tower created from config")
    except Exception as e:
        print(f"  Warning: visual tower creation failed: {e}")
        model.visual = None

    try:
        if model_type == "qwen3_vl":
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel as _VLModel
        else:
            from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLModel as _VLModel

        class _RopeStub:
            def __init__(self, cfg):
                self.config = cfg
            def get_rope_index(self, input_ids, image_grid_thw=None, attention_mask=None):
                return _VLModel.get_rope_index(
                    self, input_ids=input_ids,
                    image_grid_thw=image_grid_thw, attention_mask=attention_mask)

        object.__setattr__(model, '_rope_index_fn', _RopeStub(vl_config).get_rope_index)
        print("  M-RoPE helper ready.")
    except Exception as e:
        print(f"  Warning: rope index setup failed ({e}).")

    return model


def _has_hf_weights(path):
    import glob as _glob
    for pattern in ("*.safetensors", "pytorch_model*.bin"):
        if _glob.glob(os.path.join(path, pattern)):
            return True
    return False


def _load_policy_statistics(stats_path, args):
    """Load exactly the normalization blocks required by this checkpoint.

    Profile B is DIFF-only and its frozen MIDTRAIN_TRAIN artifact deliberately
    contains no Force6D block.  Conversely, any checkpoint which consumes a
    Force6D vector/history/code must fail closed when that block is absent.
    """
    with open(stats_path) as f:
        stats_raw = json.load(f)
    if not isinstance(stats_raw, dict) or not stats_raw:
        raise ValueError("normalization statistics must contain a dataset block")
    ds = (
        args.dataset_name
        if args.dataset_name and args.dataset_name in stats_raw
        else next(iter(stats_raw))
    )
    block = stats_raw[ds]
    if not isinstance(block, dict):
        raise ValueError("normalization dataset block must be an object")

    def _arr(key, sub):
        try:
            return np.array(block[key][sub])
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"normalization statistics are missing required {key}.{sub}"
            ) from exc

    statistic = {
        "action_mask": _arr("action", "mask"),
        "action_min": _arr("action", "q01"),
        "action_max": _arr("action", "q99"),
    }
    force_required = bool(
        getattr(args, "use_tactile_vec", 0)
        or getattr(args, "use_tactile_vqvae", 0)
        or getattr(args, "use_tactile_code", 0)
    )
    if force_required:
        statistic.update({
            "tacf6_mask": _arr("tactile_f6", "mask"),
            "tacf6_min": _arr("tactile_f6", "q01"),
            "tacf6_max": _arr("tactile_f6", "q99"),
        })
        expected_tactile_dim = int(args.tactile_num_fingers) * 6
        if any(np.asarray(statistic[key]).size != expected_tactile_dim for key in (
            "tacf6_mask", "tacf6_min", "tacf6_max"
        )):
            raise ValueError(
                "Checkpoint tactile statistics do not match "
                f"tactile_num_fingers={args.tactile_num_fingers}; "
                "implicit hand padding is forbidden"
            )
    if args.use_robot_state:
        statistic["state_mask"] = _arr("state", "mask")
        statistic["state_min"] = _arr("state", "q01")
        statistic["state_max"] = _arr("state", "q99")
    return statistic


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_nonproduction_server_identity(args, training_args, checkpoint_path, stats_path):
    """Best-effort identity for the upstream single-view compatibility path.

    This path is not accepted by the Revo production client.  It still emits
    a server-owned checkpoint/config/statistics identity on every reply so a
    diagnostic client never has to mistake request echoes for model identity.
    """
    files = {
        "checkpoint_sha256": os.path.join(checkpoint_path, "model.pt"),
        "model_config_sha256": os.path.join(checkpoint_path, "config.json"),
        "training_args_sha256": os.path.join(checkpoint_path, "training_args.json"),
        "normalization_statistics_sha256": stats_path,
    }
    identity = {
        "schema_version": "trex-server-identity-nonproduction-v1",
        **{
            name: _sha256_file(path) if os.path.isfile(path) else "unavailable"
            for name, path in files.items()
        },
        "checkpoint_lineage_sha256": "unavailable",
        "normalization_artifact_sha256": "unavailable",
        "checkpoint_family_id": str(training_args.get("checkpoint_family_id", "untracked")),
        "normalization_family_id": str(training_args.get("normalization_family_id", "untracked")),
        "tactile_profile": str(getattr(args, "tactile_profile", "legacy_force6d")),
        "camera_profile": str(getattr(args, "camera_profile", OFFICIAL_SINGLE_VIEW_PROFILE)),
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    identity["identity_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return identity


def _validate_revo_normalization_runtime(args, training_args, checkpoint_path, stats_path):
    """Re-hash the frozen artifact inside the actual server process.

    The launcher performs the same checks before spawning us; repeating them
    here closes both direct-entrypoint bypass and mutation-between-check/use.
    Artifact ``statistics_path`` is provenance only, so a byte-identical
    bundle remains relocatable between the collection and GPU hosts.
    """
    if getattr(args, "camera_profile", OFFICIAL_SINGLE_VIEW_PROFILE) != REVO3_FULL_CENTER_PROFILE:
        return
    from revo3_v1.revo.contracts import JOINT_ORDER_HASH

    artifact_path = str(getattr(args, "stats_artifact_path", "") or "")
    if not artifact_path or not os.path.isfile(artifact_path):
        raise FileNotFoundError(
            "Revo3 serving requires --stats_artifact_path for frozen normalization"
        )
    with open(artifact_path, encoding="utf-8") as handle:
        artifact = json.load(handle)
    lineage_path = os.path.join(checkpoint_path, "checkpoint_lineage.json")
    if not os.path.isfile(lineage_path):
        raise FileNotFoundError("Revo3 serving requires checkpoint_lineage.json")
    with open(lineage_path, encoding="utf-8") as handle:
        lineage = json.load(handle)

    stats_sha = _sha256_file(stats_path)
    artifact_sha = _sha256_file(artifact_path)
    checkpoint_sha = _sha256_file(os.path.join(checkpoint_path, "model.pt"))
    expected_cli = {
        "normalization_statistics_sha256": stats_sha,
        "normalization_artifact_sha256": artifact_sha,
    }
    mismatches = {}
    for key, expected in expected_cli.items():
        observed = str(getattr(args, key, "") or "")
        if observed != expected:
            mismatches[f"cli.{key}"] = (expected, observed)
        if training_args.get(key) != expected:
            mismatches[f"training_args.{key}"] = (expected, training_args.get(key))
        if lineage.get(key) != expected:
            mismatches[f"lineage.{key}"] = (expected, lineage.get(key))
    expected_artifact = {
        "schema_version": "revo3-normalization-artifact-v1",
        "statistics_sha256": stats_sha,
        "source_split": "midtrain_train",
        "split_manifest_sha256": training_args.get("split_manifest_sha256"),
        "joint_order_hash": JOINT_ORDER_HASH,
        "tactile_profile": training_args.get("tactile_profile"),
        "checkpoint_family_id": training_args.get("checkpoint_family_id"),
        "normalization_family_id": training_args.get("normalization_family_id"),
        "capability_manifest_sha256": training_args.get("capability_manifest_sha256"),
    }
    for key, expected in expected_artifact.items():
        if artifact.get(key) != expected:
            mismatches[f"artifact.{key}"] = (expected, artifact.get(key))
    if not isinstance(artifact.get("statistics_path"), str) or not artifact["statistics_path"]:
        mismatches["artifact.statistics_path"] = ("non-empty provenance path", artifact.get("statistics_path"))
    episode_ids = artifact.get("stats_episode_ids")
    if not isinstance(episode_ids, list) or not episode_ids:
        mismatches["artifact.stats_episode_ids"] = ("non-empty list", episode_ids)
    expected_lineage = {
        "schema_version": "revo3-checkpoint-lineage-v1",
        "checkpoint_sha256": checkpoint_sha,
        "split_manifest_sha256": training_args.get("split_manifest_sha256"),
        "capability_manifest_sha256": training_args.get("capability_manifest_sha256"),
        "tactile_profile": training_args.get("tactile_profile"),
        "checkpoint_family_id": training_args.get("checkpoint_family_id"),
        "normalization_family_id": training_args.get("normalization_family_id"),
        "tactile_profile_manifest_sha256": training_args.get(
            "tactile_profile_manifest_sha256"
        ),
        "joint_order_hash": JOINT_ORDER_HASH,
        "camera_profile": REVO3_FULL_CENTER_PROFILE,
        "view_slots": {"slow": "full", "fast": "fixed_center"},
    }
    for key, expected in expected_lineage.items():
        if lineage.get(key) != expected:
            mismatches[f"lineage.{key}"] = (expected, lineage.get(key))
    stage = training_args.get("revo_training_stage")
    if stage not in {"w0", "w1", "midtrain", "sft"} or lineage.get("stage") != stage:
        mismatches["lineage.stage"] = (stage, lineage.get("stage"))
    if mismatches:
        raise ValueError(f"Revo3 frozen normalization lineage failed: {mismatches}")


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def model_load(args):
    ckpt = args.checkpoint_path

    ta_path = os.path.join(ckpt, "training_args.json")
    ta = {}
    if os.path.exists(ta_path):
        with open(ta_path) as f:
            ta = json.load(f)
        for key, default in [("action_dim", 31),
                             ("action_chunk", 8),
                             ("use_robot_state", 0),
                             ("use_tactile_deform", 1),
                             ("use_tactile_vec", 0),
                             ("tactile_intermediate_size", 0),
                             ("tactile_num_fingers", 10),
                             ("n_flare_tokens_per_frame", 0),
                             ("n_flare_steps", 0),
                             ("use_tactile_code", 0),
                             ("vqvae_codebook_size", 64),
                             ("use_tactile_vqvae", 0),
                             ("cascaded_total_steps", 10),
                             ("cascaded_split_step", 6)]:
            saved = ta.get(key, default)
            cli_val = getattr(args, key, default)
            if key in ta and cli_val == default:
                setattr(args, key, saved)
                print(f"Auto-detected {key}={saved} from training_args.json")
        # vqvae_config is a dict — restore it verbatim so the embedded VQ-VAE
        # submodule is rebuilt with the right architecture before weights load.
        if ta.get("vqvae_config") is not None and getattr(args, "vqvae_config", None) is None:
            args.vqvae_config = ta["vqvae_config"]

    stats_path = args.stats_path or ""
    if not stats_path:
        candidate = os.path.join(ckpt, "stats_data.json")
        if os.path.exists(candidate):
            stats_path = candidate
    if not stats_path or not os.path.exists(stats_path):
        raise FileNotFoundError("Cannot find stats JSON.")
    _validate_revo_normalization_runtime(args, ta, ckpt, stats_path)
    if _validated_camera_profile(args) == REVO3_FULL_CENTER_PROFILE:
        from revo3_v1.policy.server_identity import build_revo_server_identity
        from revo3_v1.revo.contracts import JOINT_ORDER_HASH

        args._server_identity = build_revo_server_identity(
            checkpoint_path=ckpt,
            normalization_statistics_path=stats_path,
            normalization_artifact_path=args.stats_artifact_path,
            model_config_path=(
                os.path.join(str(args.base_model_path), "config.json")
                if (
                    os.path.isdir(str(args.base_model_path))
                    and _has_hf_weights(str(args.base_model_path))
                )
                else os.path.join(ckpt, "config.json")
            ),
            camera_profile=REVO3_FULL_CENTER_PROFILE,
            tactile_profile=str(args.tactile_profile),
            joint_order_hash=JOINT_ORDER_HASH,
        ).as_mapping()
    else:
        args._server_identity = _build_nonproduction_server_identity(
            args, ta, ckpt, stats_path
        )

    tac_isize = args.tactile_intermediate_size if args.tactile_intermediate_size > 0 else None
    n_flare_tpf = getattr(args, "n_flare_tokens_per_frame", 0)
    n_flare_steps = getattr(args, "n_flare_steps", 0)

    proc_dir = os.path.join(ckpt, "processor")
    if not os.path.isdir(proc_dir):
        raise FileNotFoundError(f"processor/ not found in checkpoint: {ckpt}")
    processor = AutoProcessor.from_pretrained(proc_dir, trust_remote_code=True)
    print(f"Processor loaded from: {proc_dir}")

    base_model_path = getattr(args, "base_model_path", "")
    ckpt_config = os.path.join(ckpt, "config.json")

    if base_model_path and os.path.isdir(base_model_path) and _has_hf_weights(base_model_path):
        model = Qwen3VLVLAModel.from_pretrained_qwen3vl(
            pretrained_path=base_model_path,
            action_dim=args.action_dim, action_chunk=args.action_chunk,
            tactile_num_fingers=args.tactile_num_fingers,
            use_tactile_deform=bool(args.use_tactile_deform),
            use_robot_state=bool(args.use_robot_state),
            torch_dtype=torch.bfloat16,
            tactile_intermediate_size=tac_isize,
            n_flare_tokens_per_frame=n_flare_tpf,
            n_flare_steps=n_flare_steps,
            use_tactile_code=bool(getattr(args, "use_tactile_code", 0)),
            vqvae_codebook_size=getattr(args, "vqvae_codebook_size", 64),
            use_tactile_vqvae=bool(getattr(args, "use_tactile_vqvae", 0)),
            vqvae_config=getattr(args, "vqvae_config", None),
        )
    elif os.path.exists(ckpt_config):
        model = _build_qwen3vl_from_config(ckpt_config, args)
    else:
        pretrained_path = None
        if ta:
            mp = ta.get("model_path", "")
            if mp and os.path.isdir(mp) and _has_hf_weights(mp):
                pretrained_path = mp
        if pretrained_path is None:
            raise FileNotFoundError(f"Cannot reconstruct model from {ckpt}")
        model = Qwen3VLVLAModel.from_pretrained_qwen3vl(
            pretrained_path=pretrained_path,
            action_dim=args.action_dim, action_chunk=args.action_chunk,
            tactile_num_fingers=args.tactile_num_fingers,
            use_tactile_deform=bool(args.use_tactile_deform),
            use_robot_state=bool(args.use_robot_state),
            torch_dtype=torch.bfloat16,
            tactile_intermediate_size=tac_isize,
            n_flare_tokens_per_frame=n_flare_tpf,
            n_flare_steps=n_flare_steps,
            use_tactile_code=bool(getattr(args, "use_tactile_code", 0)),
            vqvae_codebook_size=getattr(args, "vqvae_codebook_size", 64),
            use_tactile_vqvae=bool(getattr(args, "use_tactile_vqvae", 0)),
            vqvae_config=getattr(args, "vqvae_config", None),
        )

    ckpt_file = os.path.join(ckpt, "model.pt")
    sd = torch.load(ckpt_file, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if getattr(args, "camera_profile", OFFICIAL_SINGLE_VIEW_PROFILE) == REVO3_FULL_CENTER_PROFILE:
        if missing or unexpected:
            raise RuntimeError(
                "Revo3 migrated checkpoint must load exactly; "
                f"missing={missing[:10]}, unexpected={unexpected[:10]}"
            )
    print(f"Checkpoint loaded: missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"  missing (first 10): {missing[:10]}")
    model = model.to(torch.bfloat16)

    # Keep the embedded VQ-VAE + its F6 stats in fp32 so on-the-fly codes match
    # the standalone tokenizer (the bf16 cast above would otherwise downcast the
    # codebook and normalization buffers).
    if getattr(model, "tactile_vqvae", None) is not None:
        model.tactile_vqvae.float().eval()
        model.tacf6_vqvae_min = model.tacf6_vqvae_min.float()
        model.tacf6_vqvae_max = model.tacf6_vqvae_max.float()

    n_flare_total = n_flare_tpf * n_flare_steps
    if n_flare_total > 0:
        print(f"Flare prediction: {n_flare_steps} steps × {n_flare_tpf} tok/frame = {n_flare_total} total tokens")

    statistic = _load_policy_statistics(stats_path, args)

    return model, processor, statistic


# ─────────────────────────────────────────────────────────────────────────────
# Tactile encoding helpers
# ─────────────────────────────────────────────────────────────────────────────

def _encode_tactile_f6(tactile_f6_input, statistic, device):
    """Encode F6 for the single-frame tacf6_embedder.

    Accepts either:
      * `[n_fingers, 6]`           — single current frame (legacy clients).
      * `[T, n_fingers, 6]`        — dense rolling window (new clients send
                                     the full VQ-VAE window).  We take the
                                     last frame here for the per-frame
                                     embedder; the full window is consumed
                                     separately by `_push_f6_and_encode`.
    """
    if tactile_f6_input is None:
        return None
    arr = np.asarray(tactile_f6_input, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[-1]                    # most recent frame
    tacf6 = arr.reshape(-1)
    norm_tacf6 = _normalize(tacf6, statistic["tacf6_mask"],
                            statistic["tacf6_min"], statistic["tacf6_max"])
    return (torch.tensor(norm_tacf6.reshape(-1, 6), dtype=torch.bfloat16)
            .unsqueeze(0).to(device))


def _encode_tactile_deform(tactile_deform_input, device):
    if tactile_deform_input is None:
        return None
    arr = np.array(tactile_deform_input, dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    if arr.ndim == 3:
        return (torch.tensor(arr).unsqueeze(0).unsqueeze(2)
                .to(device, dtype=torch.bfloat16))
    elif arr.ndim == 4:
        return (torch.tensor(arr).unsqueeze(0)
                .to(device, dtype=torch.bfloat16))
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Cascaded slow/fast inference
# ─────────────────────────────────────────────────────────────────────────────

class CascadedServer:
    """
    Stateful server for cascaded flow-matching inference.

    Holds the slow-tick snapshot (cached latent + action KV at τ_split, the
    partially-denoised intermediate state x_split, encoded fast cameras +
    state, latent position ids) between requests so a subsequent fast tick
    can run the tactile expert without re-encoding the visual tower.

    A single `lock` serializes all model calls so:
      • only one inference runs on the GPU at a time, and
      • a fast request that arrives while a slow inference is still in flight
        blocks until the slow one finishes ("wait until refinement finishes"
        semantic).  ZMQ REP is already single-threaded; the lock is belt-and-
        braces for any future move to a multi-threaded transport.
    """

    def __init__(self, args, model, processor, statistic):
        self.args      = args
        self.tactile_num_fingers = int(args.tactile_num_fingers)
        self.model     = model
        self.processor = processor
        self.statistic = statistic
        self.device    = f"cuda:{args.cuda}"
        self.lock      = threading.Lock()
        self.camera_profile = _validated_camera_profile(args)
        self.tactile_profile = str(getattr(args, "tactile_profile", "legacy_force6d"))
        self.cached_request_identity = None
        self.cached_observation_timestamp_ns = None
        server_identity = getattr(args, "_server_identity", None)
        if not isinstance(server_identity, dict) or not server_identity:
            raise ValueError("model_load must establish server-owned identity before serving")
        # Never update this object from a request.  A fresh copy is returned
        # in every reply so callers cannot mutate the server's pinned value.
        self.server_identity = dict(server_identity)

        # Ablation: when True, skip the cascaded split entirely and let the
        # action expert integrate the full τ ∈ [0, 1] flow alone.  The tactile
        # expert is never invoked; fast ticks return the cached full-flow
        # chunk unchanged.
        self.disable_tactile = bool(getattr(args, "disable_tactile", 0))

        # Slow-tick snapshot
        self.cached_kv          = None
        self.x_split            = None             # [B, n_chunk, action_dim] bf16,
                                                   # action-expert intermediate at τ=τ_split
        self.tau_split          = None
        self.position_ids       = None
        self.attention_mask     = None
        self.n_action_in_cache  = 0
        self.chunk_id           = -1               # incremented per slow
        self.last_actions       = None             # cached denormalized chunk for
                                                   # disable_tactile fast-tick passthrough

        # VQ-VAE tactile-code encoder.  Two modes:
        #   • embedded  — the model carries `tactile_vqvae`; the server only
        #                 builds the raw F6 history window and the model encodes
        #                 it internally (codes never leave the model).
        #   • external  — legacy: load a standalone VQ-VAE here and feed
        #                 pre-computed codes to the model.
        # Both keep a rolling F6 buffer so each fast tick sees the historical
        # `window` frames — same alignment as offline JSON encoding.
        self.vqvae_model    = None
        self.vqvae_stats    = None
        self.vqvae_window   = 16
        self.f6_buffer: list = []                  # list of [F, 6] np arrays
        self.use_embedded_vqvae = bool(
            getattr(model, "use_tactile_vqvae", False)
            and getattr(model, "tactile_vqvae", None) is not None)
        if self.use_embedded_vqvae:
            if str(getattr(args, "vqvae_mode", "embedded")) != "embedded":
                raise RuntimeError(
                    "checkpoint contains an embedded VQ-VAE but server vqvae_mode is not embedded"
                )
            self.vqvae_window = int(model.tactile_vqvae.cfg.window)
            print(f">>> embedded VQ-VAE in model — F6 encoded on-the-fly "
                  f"(K={model.tactile_vqvae.cfg.codebook_size}, W={self.vqvae_window})")
        elif bool(getattr(args, "use_tactile_code", 0)):
            if str(getattr(args, "vqvae_mode", "embedded")) != "external":
                raise RuntimeError(
                    "tactile codes requested but checkpoint has no embedded VQ-VAE; "
                    "an explicit external mode plus --vqvae_ckpt is required"
                )
            if not str(getattr(args, "vqvae_ckpt", "")):
                raise RuntimeError("external VQ-VAE mode requires --vqvae_ckpt")
            from tactile_vqvae.models.tactile_vqvae import (
                TactileVQVAE, TactileVQVAEConfig)
            from tactile_vqvae.data.stats import TacF6Stats
            blob = torch.load(args.vqvae_ckpt, map_location="cpu",
                              weights_only=False)
            cfg = TactileVQVAEConfig.from_dict(blob["config"])
            self.vqvae_model = TactileVQVAE(cfg)
            self.vqvae_model.load_state_dict(blob["model_state"])
            self.vqvae_model.eval().to(self.device)
            self.vqvae_stats  = TacF6Stats.from_dict(blob["stats"])
            self.vqvae_window = int(cfg.window)
            print(f">>> external VQ-VAE loaded for tactile codes "
                  f"(K={cfg.codebook_size}, W={self.vqvae_window})")

    def _rolling_f6_window(self, tactile_f6_input):
        """Build the raw [window, 10, 6] F6 history from either a dense client
        window ([T,10,6]) or a single frame ([10,6]) via the rolling buffer."""
        if tactile_f6_input is None:
            return None
        arr = np.asarray(tactile_f6_input, dtype=np.float32)
        w = self.vqvae_window
        if arr.ndim == 3:
            if arr.shape[0] >= w:
                arr = arr[-w:]
            else:
                head = np.repeat(arr[:1], w - arr.shape[0], axis=0)
                arr = np.concatenate([head, arr], axis=0)
        else:
            f6 = arr.reshape(self.tactile_num_fingers, 6)
            self.f6_buffer.append(f6)
            if len(self.f6_buffer) > w:
                self.f6_buffer = self.f6_buffer[-w:]
            if len(self.f6_buffer) < w:
                head = [self.f6_buffer[0]] * (w - len(self.f6_buffer))
                arr = np.stack(head + self.f6_buffer, axis=0)
            else:
                arr = np.stack(self.f6_buffer, axis=0)
        return arr                                  # [W, 10, 6] raw

    def _f6_history_window(self, tactile_f6_input):
        """Raw F6 history tensor [1, window, 10, 6] for the embedded VQ-VAE."""
        arr = self._rolling_f6_window(tactile_f6_input)
        if arr is None:
            return None
        return (torch.from_numpy(arr.astype(np.float32))
                .unsqueeze(0).to(self.device))

    def _push_f6_and_encode(self, tactile_f6_input):
        """Encode an F6 history window into per-hand VQ-VAE codes.

        Accepted inputs:
          * `[10, 6]`         — single current frame.  We append it to a
                                server-side rolling buffer and encode the
                                last `window` frames.  Used by legacy clients
                                that don't track tactile history themselves.
          * `[T, 10, 6]`      — dense rolling window from a client that
                                already maintains the F6 history at fetch-
                                rate (e.g. eval_trex_async.py with the F6_HISTORY
                                deque).  We use the window directly — this
                                matches VQ-VAE training-time temporal density.

        Returns `[1, K]` int64 codes (K=2 hand or K=10 finger) on `self.device`,
        or None when VQ-VAE is disabled / input is missing.
        """
        if self.vqvae_model is None or tactile_f6_input is None:
            return None
        arr = self._rolling_f6_window(tactile_f6_input)        # [W, 10, 6] raw
        arr_n = self.vqvae_stats.normalize(arr).astype(np.float32, copy=False)

        is_per_finger = getattr(self.vqvae_model.cfg, "granularity", "hand") == "finger"
        n_fingers = int(getattr(self.vqvae_model.cfg, "n_fingers", 5)) if is_per_finger else 1

        if self.tactile_num_fingers % n_fingers:
            raise ValueError(
                f"tactile_num_fingers={self.tactile_num_fingers} is not divisible "
                f"by VQ-VAE n_fingers={n_fingers}"
            )
        n_groups = self.tactile_num_fingers // n_fingers
        if is_per_finger:
            codes = np.zeros((n_groups, n_fingers), dtype=np.int64)
        else:
            codes = np.zeros(n_groups, dtype=np.int64)

        for hand in range(n_groups):
            wh = arr_n[:, hand * n_fingers: (hand + 1) * n_fingers, :]
            t = torch.from_numpy(wh).unsqueeze(0).to(self.device)
            with torch.no_grad():
                idx = self.vqvae_model.encode(t).cpu().numpy()   # [1] or [1, 5]
            if is_per_finger:
                codes[hand] = idx.reshape(-1)
            else:
                codes[hand] = int(idx.item())

        # Flatten and add batch dim → [1, 2] (hand) or [1, 10] (finger).
        flat = codes.reshape(-1)
        return torch.tensor(flat, dtype=torch.long,
                            device=self.device).unsqueeze(0)

    # -- internal: build slow embeddings, run action-only flow, cache state --
    def _run_slow(
        self, task_description, slow_images, fast_images,
        tactile_f6_input=None, tactile_deform_input=None, state_fast=None,
    ):
        args, model, processor, statistic = (
            self.args, self.model, self.processor, self.statistic)
        device = self.device

        if args.image_size:
            _sz = tuple(args.image_size)
            slow_images = [img.resize(_sz, Image.LANCZOS) for img in slow_images]
            fast_images = [img.resize(_sz, Image.LANCZOS) for img in fast_images]

        state_embeds = None
        if args.use_robot_state and state_fast is not None:
            norm_state = _normalize(
                np.array(state_fast, dtype=np.float32),
                statistic["state_mask"], statistic["state_min"], statistic["state_max"])
            state_vec = torch.tensor(norm_state, dtype=torch.bfloat16).unsqueeze(0).to(device)
            state_embeds = model.state_embedder(state_vec).unsqueeze(1)

        n_slow = len(slow_images)
        all_pil = slow_images + fast_images
        content = [{"type": "image"} for _ in slow_images]
        content.append({"type": "text", "text": task_description})
        content += [{"type": "image"} for _ in fast_images]
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inp = processor(text=text, images=all_pil if all_pil else None,
                        return_tensors="pt", padding=False)

        input_ids = inp.input_ids.to(device)
        attention_mask = inp.attention_mask.to(device)
        pixel_values = (inp.pixel_values.to(device, dtype=torch.bfloat16)
                        if getattr(inp, "pixel_values", None) is not None else None)
        image_grid_thw = (inp.image_grid_thw.to(device)
                          if getattr(inp, "image_grid_thw", None) is not None else None)

        inputs_embeds = model.prepare_inputs_embeds(
            input_ids=input_ids, pixel_values=pixel_values,
            image_grid_thw=image_grid_thw)

        fast_embeds = None
        if image_grid_thw is not None and fast_images:
            merge = getattr(model.visual, "spatial_merge_size",
                            getattr(processor.image_processor, "merge_size", 2))
            n_slow_img_tokens = sum(
                int(g[0] * (g[1] // merge) * (g[2] // merge))
                for g in image_grid_thw[:n_slow])
            slow_embeds, fast_embeds = split_slow_fast_embeds(
                inputs_embeds, input_ids,
                model.image_token_id, n_slow_img_tokens)
        else:
            slow_embeds = inputs_embeds

        position_ids, _ = model.get_rope_index(
            input_ids=input_ids, image_grid_thw=image_grid_thw,
            attention_mask=attention_mask)
        position_ids = position_ids[:, :, :slow_embeds.shape[1]]

        if model.n_flare_tokens > 0:
            flare_q = model.flare_queries.to(
                device=slow_embeds.device, dtype=slow_embeds.dtype)
            slow_embeds = torch.cat([slow_embeds, flare_q.expand(1, -1, -1)], dim=1)
            position_ids = extend_position_ids_for_flare(
                position_ids, model.n_flare_tokens)

        noise = torch.randn(1, args.action_chunk, args.action_dim,
                            dtype=torch.bfloat16, device=device)

        if self.disable_tactile:
            # Action-expert-only ablation: integrate the full τ ∈ [0, 1] flow
            # and return the resulting chunk directly.  No tactile expert is
            # invoked; subsequent fast ticks reuse the same chunk.
            full_chunk = model.forward_flow_action_full(
                inputs_embeds=slow_embeds,
                position_ids=position_ids,
                attention_mask=attention_mask,
                noise=noise,
                state_embeds=state_embeds,
                fast_embeds=fast_embeds,
                num_steps=args.cascaded_total_steps,
            )
            self.x_split           = None
            self.tau_split         = None
            self.cached_kv         = None
            self.position_ids      = position_ids
            self.attention_mask    = attention_mask
            self.n_action_in_cache = 0
            self.chunk_id         += 1
            a_full = _denormalize(
                full_chunk[0].float().cpu().numpy(),
                statistic["action_mask"],
                statistic["action_min"], statistic["action_max"])
            self.last_actions = list(a_full)
            return self.last_actions, self.chunk_id

        x_split, cached_kv, n_action_in_cache, tau_split = (
            model.forward_flow_action_partial(
                inputs_embeds=slow_embeds,
                position_ids=position_ids,
                attention_mask=attention_mask,
                noise=noise,
                state_embeds=state_embeds,
                fast_embeds=fast_embeds,
                num_steps_total=args.cascaded_total_steps,
                split_step=args.cascaded_split_step,
                refresh_clean_kv=True,
            ))
        self.x_split    = x_split           # action expert's intermediate at τ=τ_split
        self.tau_split  = tau_split
        self.cached_kv         = cached_kv
        self.position_ids      = position_ids
        self.attention_mask    = attention_mask
        self.n_action_in_cache = n_action_in_cache
        self.chunk_id         += 1

        # Cascaded mode produces no usable action from the slow tick alone;
        # caller should always send 'slow_and_fast' so a_refined comes back.
        return [], self.chunk_id

    # -- internal: run tactile expert flow continuation on cached state --
    def _run_fast(self, tactile_f6_input=None, tactile_deform_input=None):
        args, model, statistic = self.args, self.model, self.statistic
        device = self.device

        if self.disable_tactile:
            # Tactile expert is disabled — fast ticks just replay the chunk
            # computed by the last slow tick.
            if self.last_actions is None:
                raise RuntimeError(
                    "fast request received before any slow request — server "
                    "has no cached action.  Send mode='slow_and_fast' first.")
            return self.last_actions, self.chunk_id

        if self.cached_kv is None or self.x_split is None:
            raise RuntimeError(
                "fast request received before any slow request — server has "
                "no cached state. Send mode='slow' or 'slow_and_fast' first.")

        tac_f6_tensor     = _encode_tactile_f6(
            tactile_f6_input if args.use_tactile_vec else None,
            statistic, device)
        tac_deform_tensor = _encode_tactile_deform(
            tactile_deform_input if args.use_tactile_deform else None, device)
        # Embedded VQ-VAE: hand the model the raw F6 history and let it encode.
        # External / legacy: encode here and pass pre-computed codes.
        if self.use_embedded_vqvae:
            tac_codes_tensor   = None
            tac_hist_tensor    = self._f6_history_window(tactile_f6_input)
        else:
            tac_codes_tensor   = self._push_f6_and_encode(tactile_f6_input)
            tac_hist_tensor    = None

        # Continue the action expert's flow with the tactile expert from
        # x_split → τ=0; the result IS the clean action (no Â + Δa add).
        refined = model.tactile_flow_continue(
            cached_kv          = self.cached_kv,
            latent_position_ids= self.position_ids,
            n_action_in_cache  = self.n_action_in_cache,
            x_split            = self.x_split,
            tau_split          = self.tau_split,
            tactile_f6         = tac_f6_tensor,
            tactile_deform     = tac_deform_tensor,
            tactile_codes      = tac_codes_tensor,
            tactile_f6_history = tac_hist_tensor,
            num_steps_total    = args.cascaded_total_steps,
            split_step         = args.cascaded_split_step,
        )
        a_refined_norm = refined[0].float().cpu().numpy()
        a_refined = _denormalize(
            a_refined_norm, statistic["action_mask"],
            statistic["action_min"], statistic["action_max"])
        return list(a_refined), self.chunk_id

    def predict(self, mode, payload):
        """Top-level dispatch.  Returns dict suitable for pickling back to
        the client.  Any exception inside a mode's body is propagated to the
        caller, which logs and replies with status='error'."""
        if mode == "identity":
            return {
                "status": "success",
                "mode": "identity",
                "server_identity": dict(self.server_identity),
            }
        payload_profile = payload.get("camera_profile", OFFICIAL_SINGLE_VIEW_PROFILE)
        if payload_profile != self.camera_profile:
            raise ValueError(
                f"request camera_profile={payload_profile!r} does not match "
                f"server camera_profile={self.camera_profile!r}"
            )
        strict_revo = self.camera_profile == REVO3_FULL_CENTER_PROFILE
        identity = None
        if strict_revo:
            payload_tactile_profile = str(payload.get("tactile_profile", ""))
            if payload_tactile_profile != self.tactile_profile:
                raise ValueError(
                    "Revo3 request tactile_profile does not match server checkpoint profile"
                )
            identity_names = (
                "task_id", "task_version", "instruction_hash", "lease_id",
                "version_fingerprint",
            )
            missing_identity = [name for name in identity_names if payload.get(name) in (None, "")]
            if missing_identity:
                raise ValueError(f"Revo3 request missing identity fields: {missing_identity}")
            identity = tuple(payload[name] for name in identity_names)
            observation_timestamp_ns = payload.get("observation_timestamp_ns")
            request_sent_at_ns = payload.get("request_sent_at_ns")
            lease_expires_at_ns = payload.get("lease_expires_at_ns")
            for name, value in (
                ("observation_timestamp_ns", observation_timestamp_ns),
                ("request_sent_at_ns", request_sent_at_ns),
                ("lease_expires_at_ns", lease_expires_at_ns),
            ):
                if not isinstance(value, (int, np.integer)) or value < 0:
                    raise ValueError(f"Revo3 request requires non-negative integer {name}")
            if not observation_timestamp_ns <= request_sent_at_ns < lease_expires_at_ns:
                raise ValueError("Revo3 observation/request/lease timestamps are inconsistent")
            requires_force = self.tactile_profile in {
                "profile_a_force6d_diff", "ablation_force6d_only"
            }
            if requires_force:
                history = np.asarray(payload.get("tactile_f6_history"), dtype=np.float32)
                history_ts = np.asarray(
                    payload.get("tactile_f6_history_timestamps_ns"), dtype=np.int64
                )
                history_seq = np.asarray(
                    payload.get("tactile_f6_history_sequences"), dtype=np.int64
                )
                expected_history = (16, self.tactile_num_fingers, 6)
                if history.shape != expected_history or not np.isfinite(history).all():
                    raise ValueError(f"Revo3 tactile history must be finite {expected_history}")
                if history_ts.shape != (16,) or history_seq.shape != (16,):
                    raise ValueError("Revo3 tactile history timestamps/sequences must be [16]")
                if np.any(np.diff(history_ts) <= 0) or np.any(np.diff(history_seq) <= 0):
                    raise ValueError("Revo3 tactile history timestamps/sequences must be strictly increasing")
                if history_ts[-1] != payload.get("tactile_timestamp_ns"):
                    raise ValueError("Revo3 tactile history latest timestamp mismatch")
                current_f6 = np.asarray(payload.get("tactile_f6"), dtype=np.float32)
                if current_f6.shape != (self.tactile_num_fingers, 6) or not np.array_equal(
                    history[-1], current_f6
                ):
                    raise ValueError("Revo3 tactile history latest value must equal tactile_f6")
            elif any(
                name in payload for name in (
                    "tactile_f6", "tactile_f6_history",
                    "tactile_f6_history_timestamps_ns", "tactile_f6_history_sequences",
                )
            ):
                raise ValueError("profile_b_diff_only forbids Force6D payload fields")
            if self.tactile_profile in {"profile_a_force6d_diff", "profile_b_diff_only"}:
                deform = np.asarray(payload.get("tactile_deform"))
                deform_ts = np.asarray(payload.get("tactile_deform_timestamp_ns"))
                if deform.shape != (5, 240, 240) or deform_ts.shape != (5,):
                    raise ValueError("Revo3 current DIFF must be [5,240,240] with [5] timestamps")
                if deform.dtype != np.uint8:
                    raise ValueError("Revo3 current DIFF payload must be uint8")
                diff_age_ns = observation_timestamp_ns - deform_ts.astype(np.int64)
                if np.any(diff_age_ns < 0):
                    raise ValueError("Revo3 current DIFF payload contains future samples")
                if np.any(diff_age_ns > 150_000_000):
                    raise ValueError("Revo3 current DIFF payload is stale")
            if mode == "fast":
                if self.cached_request_identity is None or identity != self.cached_request_identity:
                    raise ValueError("fast request identity does not match cached slow task")
                if payload.get("chunk_id") != self.chunk_id:
                    raise ValueError("fast request chunk_id does not match cached slow chunk")
                if observation_timestamp_ns < self.cached_observation_timestamp_ns:
                    raise ValueError("fast request observation timestamp regressed")
        if self.camera_profile == REVO3_FULL_CENTER_PROFILE and mode != "fast":
            if "image_head" not in payload or "image_wrist_right" not in payload:
                raise ValueError(
                    "revo3_full_center_v1 requires image_head=full and "
                    "image_wrist_right=fixed_center on every slow request"
                )
            if "image_wrist_left" in payload:
                raise ValueError("Revo3 single-camera profile forbids a third image slot")
            if tuple(payload.get("image_view_names", ())) != ("full", "fixed_center"):
                raise ValueError("Revo3 request is missing the frozen dual-view mapping")
            capture_timestamp_ns = payload.get("capture_timestamp_ns")
            if not isinstance(capture_timestamp_ns, (int, np.integer)) or capture_timestamp_ns < 0:
                raise ValueError("Revo3 dual views require one valid capture timestamp")

        slow_img = (Image.open(io.BytesIO(payload["image_head"])).convert("RGB")
                    if "image_head" in payload else None)
        fast_list = []
        if "image_wrist_right" in payload:
            fast_list.append(Image.open(io.BytesIO(payload["image_wrist_right"])).convert("RGB"))
        if "image_wrist_left" in payload:
            fast_list.append(Image.open(io.BytesIO(payload["image_wrist_left"])).convert("RGB"))

        tac_f6 = (
            payload.get("tactile_f6_history")
            if strict_revo and self.tactile_profile != "profile_b_diff_only"
            else payload.get("tactile_f6")
        )
        tac_deform = payload.get("tactile_deform", payload.get("tactile_image_deform"))
        state_fast = payload.get("state_fast")
        task_desc  = payload.get("task_description", "")

        with self.lock, torch.inference_mode():
            self.model = self.model.to(self.device).eval()
            t0 = time.time()
            inference_started_ns = time.perf_counter_ns()
            if mode == "slow":
                if slow_img is None:
                    raise ValueError("slow request requires image_head")
                actions, cid = self._run_slow(
                    task_desc, [slow_img], fast_list,
                    tac_f6, tac_deform, state_fast)
                latency_ms = (time.time() - t0) * 1000.0
                if strict_revo:
                    self.cached_request_identity = identity
                    self.cached_observation_timestamp_ns = observation_timestamp_ns
                result = {"status": "success", "mode": "slow",
                        "actions": actions, "chunk_id": cid,
                        "latency_ms": latency_ms}
            elif mode == "fast":
                actions, cid = self._run_fast(tac_f6, tac_deform)
                latency_ms = (time.time() - t0) * 1000.0
                if strict_revo:
                    self.cached_observation_timestamp_ns = observation_timestamp_ns
                result = {"status": "success", "mode": "fast",
                        "actions": actions, "chunk_id": cid,
                        "latency_ms": latency_ms}
            elif mode == "slow_and_fast":
                if slow_img is None:
                    raise ValueError("slow_and_fast request requires image_head")
                self._run_slow(task_desc, [slow_img], fast_list,
                               tac_f6, tac_deform, state_fast)
                actions, cid = self._run_fast(tac_f6, tac_deform)
                latency_ms = (time.time() - t0) * 1000.0
                if strict_revo:
                    self.cached_request_identity = identity
                    self.cached_observation_timestamp_ns = observation_timestamp_ns
                result = {"status": "success", "mode": "slow_and_fast",
                        "actions": actions, "chunk_id": cid,
                        "latency_ms": latency_ms}
            else:
                raise ValueError(f"unknown mode: {mode}")
            if strict_revo:
                # Server and controller monotonic clocks are unrelated across
                # hosts.  Express production time in the client's declared
                # clock domain by adding only the server-measured duration to
                # request_sent_at_ns.  The client can then reject stale/future
                # responses without comparing two arbitrary monotonic epochs.
                produced_at_ns = int(request_sent_at_ns) + (
                    time.perf_counter_ns() - inference_started_ns
                )
                result.update({
                    "task_id": payload["task_id"],
                    "task_version": payload["task_version"],
                    "instruction_hash": payload["instruction_hash"],
                    "lease_id": payload["lease_id"],
                    "version_fingerprint": payload["version_fingerprint"],
                    "observation_timestamp_ns": payload["observation_timestamp_ns"],
                    "produced_at_ns": produced_at_ns,
                })
            result["server_identity"] = dict(self.server_identity)
            return result


def main(args):
    print(f"Loading VLA model from checkpoint: {args.checkpoint_path}")
    model, processor, statistic = model_load(args)
    print("Model loaded successfully!")

    camera_profile = _validated_camera_profile(args)
    # Warm-up (use 2 fast images for bimanual / dual-arm tasks)
    print("Warming up model...")
    dummy_slow  = [Image.new("RGB", (224, 224), color="black")]
    n_fast_cams = 2 if args.action_dim > 31 else 1
    dummy_fast  = [Image.new("RGB", (224, 224), color="black") for _ in range(n_fast_cams)]
    dummy_state = np.zeros(args.action_dim, dtype=np.float32) if args.use_robot_state else None
    dummy_f6    = np.zeros((args.tactile_num_fingers, 6), dtype=np.float32) if args.use_tactile_vec else None
    dummy_deform = np.zeros((args.tactile_num_fingers, 240, 240), dtype=np.float32) if args.use_tactile_deform else None

    server = CascadedServer(args, model, processor, statistic)
    # Warm-up: run one slow_and_fast and discard
    dummy_payload = {
        "image_head":         _pil_to_bytes(dummy_slow[0]),
        "image_wrist_right":  _pil_to_bytes(dummy_fast[0]),
        "task_description":   "dummy task",
        "tactile_f6":         dummy_f6,
        "tactile_deform":     dummy_deform,
        "state_fast":         dummy_state,
    }
    if camera_profile == REVO3_FULL_CENTER_PROFILE:
        warm_now = time.monotonic_ns()
        warm_history_ts = np.arange(warm_now - 15, warm_now + 1, dtype=np.int64)
        warm_f6 = (
            dummy_f6
            if dummy_f6 is not None
            else np.zeros((args.tactile_num_fingers, 6), dtype=np.float32)
        )
        dummy_payload.update({
            "camera_profile": REVO3_FULL_CENTER_PROFILE,
            "capture_timestamp_ns": 0,
            "image_view_names": ("full", "fixed_center"),
            "task_id": "warmup",
            "task_version": 1,
            "instruction_hash": "warmup-sha256",
            "lease_id": "warmup-lease",
            "version_fingerprint": "warmup-version",
            "observation_timestamp_ns": warm_now,
            "state_timestamp_ns": warm_now,
            "rgb_timestamp_ns": warm_now,
            "tactile_timestamp_ns": warm_now,
            "request_sent_at_ns": warm_now,
            "lease_expires_at_ns": warm_now + 10_000_000_000,
            "tactile_profile": args.tactile_profile,
        })
        if args.tactile_profile != "profile_b_diff_only":
            dummy_payload.update({
                "tactile_f6": warm_f6,
                "tactile_f6_history": np.repeat(warm_f6[None, ...], 16, axis=0),
                "tactile_f6_history_timestamps_ns": warm_history_ts,
                "tactile_f6_history_sequences": np.arange(16, dtype=np.int64),
            })
        else:
            dummy_payload.pop("tactile_f6", None)
        if bool(args.use_tactile_deform):
            dummy_payload.update({
                "tactile_deform": dummy_deform.astype(np.uint8),
                "tactile_deform_timestamp_ns": np.full(5, warm_now, dtype=np.int64),
            })
    elif len(dummy_fast) > 1:
        dummy_payload["image_wrist_left"] = _pil_to_bytes(dummy_fast[1])
    result = server.predict("slow_and_fast", dummy_payload)
    print(f"Warm-up output shape: "
          f"{np.array(result['actions']).shape}, "
          f"latency {result['latency_ms']:.1f} ms")

    # A bounded model-loading/inference gate for CI and remote GPU bring-up.
    # This intentionally exercises the same model_load + CascadedServer path as
    # production serving, then exits before opening a network listener.
    if bool(getattr(args, "smoke_only", 0)):
        print("Smoke-only inference completed; ZMQ listener was not started.")
        return result

    # ZMQ Server
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://0.0.0.0:{args.port}")
    print(f"VLA Server listening on port {args.port} "
          f"(cascaded slow/fast, single-threaded REP)")

    step_counter = 0
    n_slow = n_fast = 0
    try:
        while True:
            try:
                payload = pickle.loads(socket.recv())

                # Default to slow_and_fast for first request; clients can override
                # with mode='slow' or mode='fast'.
                mode = payload.get("mode", "slow_and_fast")
                result = server.predict(mode, payload)
                if mode == "fast":
                    n_fast += 1
                else:
                    n_slow += 1

                socket.send(pickle.dumps(result))
                step_counter += 1
                if step_counter % 10 == 0:
                    print(f"Processed {step_counter} requests "
                          f"(slow={n_slow}, fast={n_fast}, "
                          f"chunk_id={server.chunk_id}). "
                          f"Task: {payload.get('task_description', '')}")
                max_requests = int(getattr(args, "max_requests", 0))
                if max_requests > 0 and step_counter >= max_requests:
                    print(f"Reached max_requests={max_requests}; shutting down cleanly.")
                    break

            except Exception as e:
                traceback.print_exc()
                socket.send(pickle.dumps({
                    "status": "error",
                    "message": str(e),
                    "server_identity": dict(server.server_identity),
                }))
    finally:
        socket.close(linger=0)
        context.term()
    return result


def _pil_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def build_server_argument_parser():
    parser = argparse.ArgumentParser(description="Real-world ZMQ server (with flare prediction)")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--base_model_path", type=str, default="")
    parser.add_argument("--stats_path", type=str, default="")
    parser.add_argument("--stats_artifact_path", type=str, default="")
    parser.add_argument("--normalization_statistics_sha256", type=str, default="")
    parser.add_argument("--normalization_artifact_sha256", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--action_dim", type=int, default=31)
    parser.add_argument("--action_chunk", type=int, default=8)
    parser.add_argument(
        "--tactile_num_fingers", type=int, default=10,
        help="Number of tactile feature streams; set 5 for a Revo3 single hand.",
    )
    parser.add_argument("--use_robot_state", type=int, default=0)
    parser.add_argument("--use_tactile_deform", type=int, default=1)
    parser.add_argument("--use_tactile_vec", type=int, default=0)
    parser.add_argument(
        "--tactile_profile",
        choices=("legacy_force6d", "profile_a_force6d_diff", "profile_b_diff_only", "ablation_force6d_only"),
        default="legacy_force6d",
    )
    parser.add_argument("--tactile_intermediate_size", type=int, default=0)
    parser.add_argument("--n_flare_tokens_per_frame", type=int, default=0,
                        help="0 = auto-detect from training_args.json")
    parser.add_argument("--n_flare_steps", type=int, default=0,
                        help="0 = auto-detect from training_args.json")
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--image_size", type=int, nargs=2, default=None, metavar=("W", "H"))
    parser.add_argument(
        "--camera_profile",
        choices=CAMERA_PROFILES,
        default=OFFICIAL_SINGLE_VIEW_PROFILE,
        help="Explicit visual slot contract. Revo3 maps one capture to "
             "image_head=full and image_wrist_right=fixed_center.",
    )
    parser.add_argument(
        "--smoke_only", type=int, choices=(0, 1), default=0,
        help="Run the production model-load and one slow+fast warm-up, then exit "
             "before binding a ZMQ port.",
    )
    parser.add_argument(
        "--max_requests", type=int, default=0,
        help="Exit cleanly after this many successful ZMQ requests; 0 serves forever.",
    )

    # Cascaded flow matching schedule (auto-detected from training_args.json
    # when available).  The client sends payloads with mode='slow' once per
    # action chunk and mode='fast' multiple times within the chunk window;
    # the first request must be 'slow' or 'slow_and_fast'.
    parser.add_argument("--cascaded_total_steps", type=int, default=10)
    parser.add_argument("--cascaded_split_step",  type=int, default=6)

    # Ablation: action-expert-only inference (no tactile expert ever invoked).
    # The action expert integrates the full τ ∈ [0, 1] flow for
    # `cascaded_total_steps` Euler steps; fast ticks return the cached chunk.
    parser.add_argument("--disable_tactile", type=int, default=0,
                        help="1: skip the cascaded split.  Action expert "
                             "integrates the full flow alone; tactile expert "
                             "is never called.  Useful for ablating the "
                             "tactile-expert contribution at test time.")

    # VQ-VAE tactile code tokens (fast-path only).  When 0 (default) the model
    # graph and behavior are identical to the pre-feature version.  Revo
    # checkpoints carry the VQ-VAE inside model.pt; a standalone checkpoint is
    # required only for the explicitly selected legacy external mode.
    parser.add_argument("--use_tactile_code", type=int, default=0,
                        help="1: server-side VQ-VAE encodes a rolling F6 "
                             "window into 2 codes per fast tick.")
    parser.add_argument("--vqvae_codebook_size", type=int, default=64,
                        help="Codebook size of the VQ-VAE that produces the codes.")
    parser.add_argument("--vqvae_ckpt", type=str, default="",
                        help="Path to TactileVQVAE checkpoint (latest.pt). "
                             "Required only with --vqvae_mode external.")
    parser.add_argument(
        "--vqvae_mode",
        choices=("embedded", "external"),
        default="embedded",
        help="Revo checkpoints use embedded; external is a legacy explicit adapter.",
    )

    return parser


if __name__ == "__main__":
    parser = build_server_argument_parser()
    args = parser.parse_args()
    if bool(args.use_tactile_code) and args.vqvae_mode == "external" and not args.vqvae_ckpt:
        parser.error("--vqvae_ckpt must be set when --vqvae_mode external")
    main(args)

