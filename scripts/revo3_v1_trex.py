#!/usr/bin/env python3
"""Auditable train/serve launcher for the Revo3 single-hand T-Rex path.

The launcher deliberately separates command construction from execution.  It
rejects synthetic data, EMG-bearing VLA records, unapproved action labels, and
shape-incompatible datasets before it can spawn a multi-GPU process.  The
primary route resumes the released T-Rex *pretrain* checkpoint.  A raw
released midtrain checkpoint is never launchable for Revo3: the optional
heterogeneous-hand ablation requires both an explicit acknowledgement and a
separately verified Revo-compatible migration artifact.

This wrapper invokes the locally adapted ``scripts/train.py`` cascaded stage-2
path.  Upstream ``main`` calls that file post-training code, while upstream's
paper-scale midtraining loader lives on ``full-pipeline`` and assumes a
bimanual 62-D robot.  Consequently this is accurately described as a
Revo-specific midtrain-like run, not a reproduction of T-Rex's paper midtrain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from revo3_v1.policy.artifacts import (
    validate_revo_deform_artifact,
    validate_revo_vqvae_artifact,
)
from revo3_v1.policy.server_identity import build_revo_server_identity
from revo3_v1.revo.contracts import JOINT_ORDER_HASH


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "revo3_v1_trex.json"

EXPECTED_CONTRACT: Dict[str, Any] = {
    "action_dim": 21,
    "action_chunk": 16,
    "tactile_num_fingers": 5,
    "use_robot_state": 1,
    "emg_in_vla": False,
}


class LaunchContractError(ValueError):
    """Raised before any process is launched when a reviewed contract fails."""


def _read_json(path: Path, label: str) -> Any:
    if not path.is_file():
        raise LaunchContractError(f"{label} does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise LaunchContractError(f"cannot read {label} {path}: {exc}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_launch_config(path: Path) -> Dict[str, Any]:
    config = _read_json(path, "launch config")
    if config.get("schema_version") != "revo3-trex-launch-v1":
        raise LaunchContractError("unsupported Revo3 launch-config schema")
    contract = config.get("fixed_contract")
    if contract != EXPECTED_CONTRACT:
        raise LaunchContractError(
            f"fixed Revo3 contract changed: expected {EXPECTED_CONTRACT}, got {contract}"
        )
    profiles = config.get("tactile_profiles")
    if not isinstance(profiles, dict) or config.get("default_tactile_profile") not in profiles:
        raise LaunchContractError("launch config must declare a valid default tactile profile")
    inference = config.get("inference", {})
    frozen_runtime = {
        "camera_profile": "revo3_full_center_v1",
        "image_size": [384, 288],
        "policy_hz": 30,
        "chunk_size": 16,
        "execute_steps_per_chunk": 16,
        "initial_request": "slow_and_fast",
        "refine_offsets": [4, 8, 12],
        "temporal_aggregation": True,
        "temporal_agg_k": 0.0,
    }
    runtime_mismatches = {
        key: (expected, inference.get(key))
        for key, expected in frozen_runtime.items()
        if inference.get(key) != expected
    }
    if runtime_mismatches:
        raise LaunchContractError(
            f"frozen Revo3 main runtime changed: {runtime_mismatches}"
        )
    return config


def _validate_profile_manifest(
    path: Path,
    *,
    profile: str,
    config: Mapping[str, Any],
    ablation_ack: bool,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    profiles = config["tactile_profiles"]
    if profile not in profiles:
        raise LaunchContractError(f"unknown tactile profile: {profile}")
    profile_config = profiles[profile]
    if profile_config.get("launchable_by_current_trainer") is False:
        raise LaunchContractError(
            f"{profile} has no reviewed native trainer; pressure/matrix cannot be reshaped to Force6D"
        )
    if profile_config.get("ablation_only") and not ablation_ack:
        raise LaunchContractError(
            f"{profile} is an ablation only; pass --ack-tactile-ablation"
        )
    manifest = _read_json(path, "tactile profile manifest")
    required = {
        "schema_version": "revo3-tactile-profile-v1",
        "profile": profile,
        "approved_for_training": True,
    }
    mismatches = {
        key: (expected, manifest.get(key))
        for key, expected in required.items()
        if manifest.get(key) != expected
    }
    for key in ("capability_manifest_sha256", "checkpoint_family_id", "normalization_family_id"):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            mismatches[key] = ("non-empty string", manifest.get(key))
    if len(str(manifest.get("capability_manifest_sha256", ""))) != 64:
        mismatches["capability_manifest_sha256"] = (
            "64-character SHA-256", manifest.get("capability_manifest_sha256")
        )
    if mismatches:
        raise LaunchContractError(f"tactile profile manifest failed: {mismatches}")
    return profile_config, manifest


def _require_directory(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_dir():
        raise LaunchContractError(f"{label} must be an existing directory: {resolved}")
    return resolved


def _validate_base_model(path: Path) -> Path:
    path = _require_directory(path, "Qwen3-VL base model")
    required_metadata = ("config.json", "tokenizer_config.json", "preprocessor_config.json")
    missing = [name for name in required_metadata if not (path / name).is_file()]
    weight_files = list(path.glob("*.safetensors")) + list(path.glob("pytorch_model*.bin"))
    if missing or not weight_files:
        raise LaunchContractError(
            "Qwen3-VL base model is incomplete: "
            f"missing_metadata={missing}, weight_files={len(weight_files)}, path={path}"
        )
    return path


def _checkpoint_model_path(checkpoint: Path) -> Path:
    checkpoint = _require_directory(checkpoint, "checkpoint")
    model = checkpoint / "model.pt"
    if not model.is_file():
        raise LaunchContractError(
            f"checkpoint must contain model.pt because scripts/train.py loads that exact file: {checkpoint}"
        )
    return checkpoint


def _checkpoint_training_args(checkpoint: Path) -> Mapping[str, Any]:
    path = checkpoint / "training_args.json"
    if not path.is_file():
        return {}
    payload = _read_json(path, "checkpoint training_args.json")
    if not isinstance(payload, dict):
        raise LaunchContractError("checkpoint training_args.json must be a JSON object")
    return payload


def validate_checkpoint(
    checkpoint: Path,
    checkpoint_id: str,
    mode: str,
    config: Mapping[str, Any],
    ablation_ack: bool,
) -> Path:
    checkpoint = _checkpoint_model_path(checkpoint)
    sources = config["official_sources"]
    if mode == "main":
        expected_id = sources["primary_pretrain_checkpoint_id"]
        if checkpoint_id != expected_id:
            raise LaunchContractError(
                f"main route requires checkpoint id {expected_id!r}, got {checkpoint_id!r}"
            )
        training_args = _checkpoint_training_args(checkpoint)
        if training_args.get("use_tactile_vqvae"):
            raise LaunchContractError(
                "main route received a checkpoint declaring embedded tactile VQ-VAE; "
                "this is not the reviewed pretrain start"
            )
    elif mode == "midtrain_ablation":
        expected_id = sources["midtrain_ablation_checkpoint_id"]
        if checkpoint_id != expected_id:
            raise LaunchContractError(
                f"midtrain ablation requires checkpoint id {expected_id!r}, got {checkpoint_id!r}"
            )
        if not ablation_ack:
            raise LaunchContractError(
                "midtrain checkpoint is a heterogeneous 10-finger/bimanual ablation only; "
                "pass --ack-heterogeneous-midtrain-ablation to construct this command"
            )
        # train.py auto-enables an embedded VQ-VAE when training_args.json says
        # it exists.  The released midtrain checkpoint therefore cannot be
        # made Revo-compatible merely by passing CLI flags: its 10-finger VQ
        # path would silently override our fixed 5-finger/use_vqvae=0 contract.
        # Only an explicit, co-located migration artifact is accepted.
        training_args = _checkpoint_training_args(checkpoint)
        migrated_required = {
            "action_dim": 21,
            "action_chunk": 16,
            "tactile_num_fingers": 5,
            "use_robot_state": 1,
            "use_tactile_vec": 1,
            "use_tactile_deform": 0,
            "use_tactile_vqvae": 0,
            "use_tactile_code": 0,
        }
        migrated_mismatches = {
            key: (expected, training_args.get(key))
            for key, expected in migrated_required.items()
            if training_args.get(key) != expected
        }
        migration_path = checkpoint / "revo3_migration.json"
        if migrated_mismatches or not migration_path.is_file():
            raise LaunchContractError(
                "raw/incompatible official midtrain is not launchable for Revo3; "
                "a verified Revo3 migration artifact is required before this ablation. "
                f"training_args mismatches={migrated_mismatches}, "
                f"migration_manifest={migration_path}"
            )
        migration = _read_json(migration_path, "Revo3 midtrain migration manifest")
        migration_required = {
            "schema_version": "revo3-trex-midtrain-migration-v1",
            "source_checkpoint_id": expected_id,
            "source_tactile_num_fingers": 10,
            "target_tactile_num_fingers": 5,
            "target_action_dim": 21,
            "removed_embedded_vqvae": True,
            "reinitialized_shape_mismatched_heads": True,
            "migration_test_passed": True,
        }
        migration_mismatches = {
            key: (expected, migration.get(key))
            for key, expected in migration_required.items()
            if migration.get(key) != expected
        }
        if migration_mismatches:
            raise LaunchContractError(
                f"Revo3 midtrain migration manifest failed: {migration_mismatches}"
            )
    else:  # protected even when called directly by tests/other Python code
        raise LaunchContractError(f"unsupported training mode: {mode}")
    return checkpoint


def validate_stage_checkpoint(
    checkpoint: Path,
    checkpoint_id: str,
    *,
    mode: str,
    stage: str,
    resume_kind: str,
    config: Mapping[str, Any],
    profile_name: str,
    profile_manifest: Mapping[str, Any],
    ablation_ack: bool,
) -> tuple[Path, str]:
    """Validate the only reviewed predecessor for each cascaded Revo stage."""

    predecessors = {
        "w0": "official_pretrain",
        "w1": "revo_w0",
        "midtrain": "revo_w1",
        "sft": "revo_midtrain",
    }
    if stage not in predecessors:
        raise LaunchContractError(f"unknown Revo training stage: {stage}")
    if resume_kind == "official_midtrain_ablation":
        if stage != "midtrain" or mode != "midtrain_ablation":
            raise LaunchContractError(
                "the migrated official midtrain is only an explicit midtrain ablation"
            )
        checked = validate_checkpoint(
            checkpoint, checkpoint_id, "midtrain_ablation", config, ablation_ack
        )
        return checked, _sha256_file(checked / "model.pt")
    expected = predecessors[stage]
    if resume_kind != expected:
        raise LaunchContractError(
            f"stage {stage} must resume from {expected}; got {resume_kind}. "
            "W0->W1->midtrain->SFT lineage cannot be skipped silently."
        )
    if mode != "main":
        raise LaunchContractError("non-ablation Revo lineage requires --mode main")
    if resume_kind == "official_pretrain":
        checked = validate_checkpoint(
            checkpoint, checkpoint_id, "main", config, ablation_ack=False
        )
        return checked, _sha256_file(checked / "model.pt")

    checked = _checkpoint_model_path(checkpoint)
    source_stage = resume_kind.removeprefix("revo_")
    training_args = _checkpoint_training_args(checked)
    required_training = {
        "revo_training_stage": source_stage,
        "action_dim": 21,
        "action_chunk": 16,
        "tactile_num_fingers": 5,
        "use_robot_state": 1,
        "tactile_profile": profile_name,
        "checkpoint_family_id": profile_manifest["checkpoint_family_id"],
        "normalization_family_id": profile_manifest["normalization_family_id"],
        "capability_manifest_sha256": profile_manifest[
            "capability_manifest_sha256"
        ],
        "camera_profile": "revo3_full_center_v1",
        "view_slots": {"slow": "full", "fast": "fixed_center"},
    }
    mismatches = {
        key: (expected_value, training_args.get(key))
        for key, expected_value in required_training.items()
        if training_args.get(key) != expected_value
    }
    lineage_path = checked / "checkpoint_lineage.json"
    lineage = _read_json(lineage_path, "Revo checkpoint lineage")
    checkpoint_sha256 = _sha256_file(checked / "model.pt")
    required_lineage = {
        "schema_version": "revo3-checkpoint-lineage-v1",
        "checkpoint_sha256": checkpoint_sha256,
        "stage": source_stage,
        "action_dim": 21,
        "action_chunk": 16,
        "joint_order_hash": JOINT_ORDER_HASH,
        "camera_profile": "revo3_full_center_v1",
        "view_slots": {"slow": "full", "fast": "fixed_center"},
        "tactile_profile": profile_name,
        "checkpoint_family_id": profile_manifest["checkpoint_family_id"],
        "normalization_family_id": profile_manifest["normalization_family_id"],
        "capability_manifest_sha256": profile_manifest[
            "capability_manifest_sha256"
        ],
    }
    for key, expected_value in required_lineage.items():
        if lineage.get(key) != expected_value:
            mismatches[f"lineage.{key}"] = (expected_value, lineage.get(key))
    for key in (
        "parent_checkpoint_sha256",
        "split_manifest_sha256",
        "tactile_profile_manifest_sha256",
    ):
        value = lineage.get(key)
        if not isinstance(value, str) or len(value) != 64:
            mismatches[f"lineage.{key}"] = ("64-char SHA-256", value)
    for key in (
        "normalization_statistics_sha256",
        "normalization_artifact_sha256",
    ):
        value = training_args.get(key)
        if not isinstance(value, str) or len(value) != 64:
            mismatches[key] = ("64-char SHA-256", value)
        if lineage.get(key) != value:
            mismatches[f"lineage.{key}"] = (value, lineage.get(key))
    if source_stage != "w0":
        profile = config["tactile_profiles"][profile_name]
        artifact_hash_fields = []
        if profile["use_tactile_vqvae"]:
            artifact_hash_fields.append("vqvae_artifact_sha256")
        if profile["use_tactile_deform"]:
            artifact_hash_fields.append("deform_encoder_artifact_sha256")
        for key in artifact_hash_fields:
            if not isinstance(training_args.get(key), str) or len(training_args[key]) != 64:
                mismatches[key] = ("64-char SHA-256", training_args.get(key))
            if lineage.get(key) != training_args.get(key):
                mismatches[f"lineage.{key}"] = (training_args.get(key), lineage.get(key))
    if mismatches:
        raise LaunchContractError(f"Revo checkpoint lineage failed: {mismatches}")
    return checked, checkpoint_sha256


def _validate_readiness(path: Path) -> Mapping[str, Any]:
    readiness = _read_json(path, "dataset readiness manifest")
    required = {
        "schema_version": "revo3-vla-readiness-v1",
        "dataset_kind": "real_robot",
        "ready_for_training": True,
        "synthetic_fixture": False,
        "contains_emg": False,
        "action_label_source": "controller_target",
        "action_semantics": "accepted_exact_sent_teleop_target",
        "contains_cair_residual": False,
        "timestamp_alignment_verified": True,
        "split_before_statistics_verified": True,
        "duration_targets_reviewed": True,
        "replay_gate_passed": True,
    }
    mismatches = {
        key: (expected, readiness.get(key))
        for key, expected in required.items()
        if readiness.get(key) != expected
    }
    if mismatches:
        raise LaunchContractError(f"dataset readiness gate failed: {mismatches}")
    return readiness


def _validate_conversion_manifest(
    path: Path,
    *,
    tactile_profile: str | None = None,
    profile_manifest: Mapping[str, Any] | None = None,
    expected_split: str | None = None,
) -> Mapping[str, Any]:
    manifest = _read_json(path, "Revo3 conversion manifest")
    required = {
        "schema_version": "revo3-trex-conversion-v1",
        "action_shape": [16, 21],
        "state_shape": [21],
        "tactile_delay_offsets": [0, 4, 8, 12],
        "tactile_temporal_jitter_samples": [-1, 0, 1],
        "tactile_temporal_jitter_native_offsets": [-2, -1, 0],
        "action_grid_hz": 30,
        "training_anchor_hz": 10,
        "flare_steps": 8,
        "flare_frame_stride": 4,
        "flare_padding": "forbidden",
        "view_slots": {"slow": "full", "fast": "fixed_center"},
        "view_shape_hwc": [288, 384, 3],
        "views_share_timestamp": True,
        "terminal_padding": "forbidden",
        "action_label_source": "controller_target",
        "action_semantics": "accepted_exact_sent_teleop_target",
        "contains_cair_residual": False,
        "contains_emg": False,
        "split_before_statistics": True,
        "duration_targets_enforced": True,
        "normalization_frozen": True,
        "statistics_source_split": "midtrain_train",
    }
    mismatches = {
        key: (expected, manifest.get(key))
        for key, expected in required.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise LaunchContractError(f"Revo3 conversion contract failed: {mismatches}")
    if tactile_profile is not None:
        profile_required = {
            "tactile_profile": tactile_profile,
            "checkpoint_family_id": profile_manifest.get("checkpoint_family_id") if profile_manifest else None,
            "normalization_family_id": profile_manifest.get("normalization_family_id") if profile_manifest else None,
            "capability_manifest_sha256": profile_manifest.get("capability_manifest_sha256") if profile_manifest else None,
        }
        profile_mismatches = {
            key: (expected, manifest.get(key))
            for key, expected in profile_required.items()
            if manifest.get(key) != expected
        }
        if profile_mismatches:
            raise LaunchContractError(
                f"dataset/checkpoint tactile profile family mismatch: {profile_mismatches}"
            )
        requires_force = tactile_profile in {
            "profile_a_force6d_diff", "ablation_force6d_only"
        }
        requires_diff = tactile_profile in {
            "profile_a_force6d_diff", "profile_b_diff_only"
        }
        shape_required = {
            "tactile_shape": [5, 6] if requires_force else None,
            "tactile_history_shape": [16, 5, 6] if requires_force else None,
            "tactile_deform_shape": [5, 1, 240, 240] if requires_diff else None,
        }
        shape_mismatches = {
            key: (expected, manifest.get(key))
            for key, expected in shape_required.items()
            if manifest.get(key) != expected
        }
        if shape_mismatches:
            raise LaunchContractError(f"tactile profile schema mismatch: {shape_mismatches}")
    if expected_split is not None and manifest.get("dataset_split") != expected_split:
        raise LaunchContractError(
            f"expected conversion split {expected_split}, got {manifest.get('dataset_split')}"
        )
    if expected_split == "sft_train":
        counts = manifest.get("instruction_source_counts")
        if not isinstance(counts, dict):
            raise LaunchContractError("SFT conversion lacks instruction-source provenance")
        manual = int(counts.get("manual_canonical", 0))
        planner = int(counts.get("frozen_planner", 0))
        planner_ratio = planner / max(manual + planner, 1)
        if not 0.4 <= planner_ratio <= 0.6:
            raise LaunchContractError(
                f"SFT Planner/manual language ratio is outside 40-60%: {planner_ratio:.3f}"
            )
    split_hash = manifest.get("split_manifest_sha256")
    if not isinstance(split_hash, str) or len(split_hash) != 64:
        raise LaunchContractError("real conversion must carry a split manifest SHA-256")
    episode_ids = manifest.get("episode_ids", [])
    if not isinstance(episode_ids, list) or len(set(episode_ids)) < 2:
        raise LaunchContractError("episode-grouped validation needs at least two episodes")
    source_root_raw = manifest.get("source_root")
    if not source_root_raw:
        raise LaunchContractError("conversion manifest must record its source_root")
    source_root = _require_directory(Path(source_root_raw), "converted episode source_root")
    corpus_meta = source_root / "corpus_meta.json"
    if corpus_meta.is_file() and _read_json(corpus_meta, "corpus metadata").get(
        "synthetic_fixture"
    ):
        raise LaunchContractError("synthetic Revo corpus is smoke-only and cannot launch training")
    episode_meta_paths = sorted(source_root.glob("*/meta.json"))
    if len(episode_meta_paths) < 2:
        raise LaunchContractError("source_root must contain at least two real episode meta.json files")
    observed_episode_ids = set()
    for episode_meta_path in episode_meta_paths:
        episode = _read_json(episode_meta_path, "episode metadata")
        observed_episode_ids.add(episode.get("episode_id"))
        if episode.get("synthetic_fixture") is not False:
            raise LaunchContractError(
                f"episode is not explicitly marked real (synthetic_fixture=false): {episode_meta_path}"
            )
        if episode.get("contains_emg") is not False:
            raise LaunchContractError(f"episode contains/omits EMG exclusion: {episode_meta_path}")
        if episode.get("action_label_source") != "controller_target":
            raise LaunchContractError(f"episode action labels are not controller targets: {episode_meta_path}")
        if episode.get("action_semantics") != "accepted_exact_sent_teleop_target":
            raise LaunchContractError(f"episode lacks exact-sent action semantics: {episode_meta_path}")
        if episode.get("contains_cair_residual") is not False:
            raise LaunchContractError(f"episode contains/omits CAIR exclusion: {episode_meta_path}")
        for key in (
            "task_id", "object_id", "object_instance", "operator", "collection_day",
            "grasp_primitive", "instruction_sha256", "camera_profile_id",
            "camera_calibration_sha256", "capability_manifest_sha256",
        ):
            if not episode.get(key):
                raise LaunchContractError(f"episode provenance missing {key}: {episode_meta_path}")
    if not set(episode_ids).issubset(observed_episode_ids):
        raise LaunchContractError("conversion episode_ids do not match source episode metadata")
    return manifest


def _first_array_record(path: Path, max_bytes: int = 2 * 1024 * 1024) -> Mapping[str, Any]:
    """Read only the first record from a large top-level JSON array."""
    if not path.is_file():
        raise LaunchContractError(f"training JSON does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        prefix = handle.read(max_bytes)
    start = prefix.find("[")
    if start < 0:
        raise LaunchContractError("training JSON must be a top-level array")
    payload = prefix[start + 1 :].lstrip()
    if not payload or payload.startswith("]"):
        raise LaunchContractError("training JSON contains no records")
    try:
        record, _ = json.JSONDecoder().raw_decode(payload)
    except json.JSONDecodeError as exc:
        raise LaunchContractError(
            "could not inspect the first training record within 2 MiB"
        ) from exc
    if not isinstance(record, dict):
        raise LaunchContractError("training JSON records must be objects")
    return record


def _flattened_size(value: Any) -> int:
    if not isinstance(value, list):
        return -1
    if value and isinstance(value[0], list):
        return sum(_flattened_size(item) for item in value)
    return len(value)


def _validate_stats(
    conversion: Mapping[str, Any],
    *,
    tactile_profile: str | None = None,
    profile_manifest: Mapping[str, Any] | None = None,
) -> Path:
    stats_path = Path(str(conversion.get("stats_path", ""))).resolve()
    stats = _read_json(stats_path, "normalization statistics")
    artifact_path = Path(str(conversion.get("stats_artifact_path", ""))).resolve()
    artifact = _read_json(artifact_path, "normalization artifact")
    required_artifact = {
        "schema_version": "revo3-normalization-artifact-v1",
        "statistics_path": str(stats_path),
        "statistics_sha256": _sha256_file(stats_path),
        "source_split": "midtrain_train",
        "split_manifest_sha256": conversion.get("split_manifest_sha256"),
        "stats_episode_ids": conversion.get("stats_episode_ids"),
        "joint_order_hash": JOINT_ORDER_HASH,
        "tactile_profile": tactile_profile,
        "checkpoint_family_id": (profile_manifest or {}).get("checkpoint_family_id"),
        "normalization_family_id": (profile_manifest or {}).get("normalization_family_id"),
        "capability_manifest_sha256": (profile_manifest or {}).get(
            "capability_manifest_sha256"
        ),
    }
    artifact_mismatches = {
        key: (expected, artifact.get(key))
        for key, expected in required_artifact.items()
        if artifact.get(key) != expected
    }
    if conversion.get("statistics_sha256") != _sha256_file(stats_path):
        artifact_mismatches["conversion.statistics_sha256"] = (
            _sha256_file(stats_path), conversion.get("statistics_sha256")
        )
    if conversion.get("statistics_artifact_sha256") != _sha256_file(artifact_path):
        artifact_mismatches["conversion.statistics_artifact_sha256"] = (
            _sha256_file(artifact_path),
            conversion.get("statistics_artifact_sha256"),
        )
    if artifact_mismatches:
        raise LaunchContractError(
            f"frozen normalization artifact failed: {artifact_mismatches}"
        )
    if not isinstance(stats, dict) or len(stats) != 1:
        raise LaunchContractError("statistics must contain exactly one dataset block")
    block = next(iter(stats.values()))
    expected_shapes = {"action": 16 * 21, "state": 21}
    force_required = tactile_profile in {
        "profile_a_force6d_diff", "ablation_force6d_only"
    }
    if force_required:
        expected_shapes["tactile_f6"] = 5 * 6
    for name, expected in expected_shapes.items():
        entry = block.get(name, {}) if isinstance(block, dict) else {}
        for statistic in ("q01", "q99", "mask"):
            observed = _flattened_size(entry.get(statistic))
            if observed != expected:
                raise LaunchContractError(
                    f"{name}.{statistic} has {observed} values; expected {expected}"
                )
    tracking = block.get("tracking_error", {}) if isinstance(block, dict) else {}
    for statistic in ("mean", "std"):
        if _flattened_size(tracking.get(statistic)) != 21:
            raise LaunchContractError(f"tracking_error.{statistic} must have 21 values")
    if force_required:
        noise = block.get("tactile_no_contact_noise", {})
        expected_family = profile_manifest or {}
        if (
            _flattened_size(noise.get("robust_center")) != 30
            or _flattened_size(noise.get("robust_scale")) != 30
            or _flattened_size(noise.get("covariance")) != 30 * 30
            or noise.get("normalization_family_id")
            != expected_family.get("normalization_family_id")
            or noise.get("checkpoint_family_id")
            != expected_family.get("checkpoint_family_id")
        ):
            raise LaunchContractError("no-contact Force6D noise stats/profile family mismatch")
    return stats_path


def validate_json_dataset(
    data_json: Path,
    conversion_manifest: Path,
    readiness_manifest: Path,
    *,
    tactile_profile: str | None = None,
    profile_manifest: Mapping[str, Any] | None = None,
    expected_split: str | None = None,
) -> tuple[Path, Mapping[str, Any]]:
    data_json = data_json.resolve()
    _validate_readiness(readiness_manifest)
    manifest = _validate_conversion_manifest(
        conversion_manifest,
        tactile_profile=tactile_profile,
        profile_manifest=profile_manifest,
        expected_split=expected_split,
    )
    record = _first_array_record(data_json)
    def _emg_paths(value: Any, prefix: str = "") -> list[str]:
        if isinstance(value, dict):
            result = []
            for key, item in value.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                if "emg" in str(key).lower() and str(key).lower() != "contains_emg":
                    result.append(path)
                result.extend(_emg_paths(item, path))
            return result
        if isinstance(value, list):
            return [path for index, item in enumerate(value) for path in _emg_paths(item, f"{prefix}[{index}]")]
        return []
    emg_keys = _emg_paths(record)
    if emg_keys or record.get("contains_emg") is not False:
        raise LaunchContractError(
            f"EMG is forbidden in T-Rex records; offending keys={emg_keys}"
        )
    expected_sizes = {"action": 16 * 21, "state_fast": 21}
    force_required = tactile_profile in {
        "profile_a_force6d_diff", "ablation_force6d_only"
    }
    diff_required = tactile_profile in {
        "profile_a_force6d_diff", "profile_b_diff_only"
    }
    if force_required:
        expected_sizes["tactile_f6"] = 5 * 6
    for key, expected in expected_sizes.items():
        observed = _flattened_size(record.get(key))
        if observed != expected:
            raise LaunchContractError(f"first record {key} size={observed}; expected {expected}")
    if record.get("action_label_source") != "controller_target":
        raise LaunchContractError("action labels must be recorded controller targets")
    if (
        record.get("action_semantics") != "accepted_exact_sent_teleop_target"
        or record.get("contains_cair_residual") is not False
    ):
        raise LaunchContractError("action labels lack exact-sent/no-CAIR provenance")
    if record.get("policy_loss_eligible") is not True:
        raise LaunchContractError("pre-roll/ineligible record reached the policy JSON")
    if (
        len(record.get("action_write_timestamp_ns", [])) != 16
        or len(record.get("action_controller_sequence", [])) != 16
        or len(record.get("action_request_id_hash", [])) != 16
        or len(set(record.get("action_request_id_hash", []))) != 16
    ):
        raise LaunchContractError("record lacks 16 unique accepted controller receipts")
    slow_views = record.get("input_image_slow")
    fast_views = record.get("input_image_fast")
    if not (
        isinstance(slow_views, list)
        and len(slow_views) == 1
        and isinstance(slow_views[0], str)
        and slow_views[0]
        and isinstance(fast_views, list)
        and len(fast_views) == 1
        and isinstance(fast_views[0], str)
        and fast_views[0]
    ):
        raise LaunchContractError(
            "each Revo record must map full->slow and fixed_center->fast"
        )
    if record.get("rgb_full_timestamp_ns") != record.get("rgb_center_timestamp_ns"):
        raise LaunchContractError("full and fixed-center views must share one source timestamp")
    if record.get("rgb_receive_timestamp_ns", 2**63 - 1) > record.get(
        "action_decision_timestamp_ns", -1
    ):
        raise LaunchContractError("RGB receive timestamp is after the policy decision")
    image_root = data_json.parent
    all_model_paths = list(slow_views) + list(fast_views)
    flare_paths = record.get("flare_image_full")
    if not isinstance(flare_paths, list) or len(flare_paths) != 8:
        raise LaunchContractError("Revo FLARE requires eight explicit future full frames")
    all_model_paths.extend(flare_paths)
    missing_images = [path for path in all_model_paths if not (image_root / path).is_file()]
    if missing_images:
        raise LaunchContractError(f"training record image is missing: {missing_images[0]}")
    if record.get("tactile_delay_offsets") != [0, 4, 8, 12]:
        raise LaunchContractError("record tactile delay offsets must be [0,4,8,12]")
    if record.get("tactile_temporal_jitter_samples") != [-1, 0, 1] or record.get(
        "tactile_temporal_jitter_native_offsets"
    ) != [-2, -1, 0]:
        raise LaunchContractError(
            "tactile jitter must use the reviewed causal native-sample form"
        )
    if force_required:
        required_sizes = {
            "tactile_f6_delayed": 4 * 5 * 6,
            "tactile_f6_history_delayed": 4 * 16 * 5 * 6,
            "tactile_f6_delayed_jitter": 4 * 3 * 5 * 6,
            "tactile_f6_history_delayed_jitter": 4 * 3 * 16 * 5 * 6,
            "tactile_f6_history_timestamp_ns_delayed_jitter": 4 * 3 * 16,
            "tactile_f6_history_sequence_delayed_jitter": 4 * 3 * 16,
        }
        for key, expected in required_sizes.items():
            if _flattened_size(record.get(key)) != expected:
                raise LaunchContractError(f"record {key} has invalid native-history shape")
        touch_ts = record.get("touch_timestamp_ns_delayed_jitter")
        decisions = record.get("tactile_decision_timestamp_ns_delayed")
        if _flattened_size(touch_ts) != 4 * 3 or _flattened_size(decisions) != 4:
            raise LaunchContractError("Force6D delay decision/touch timestamps are incomplete")
        if any(
            touch_ts[delay][jitter] > decisions[delay]
            for delay in range(4)
            for jitter in range(3)
        ):
            raise LaunchContractError("Force6D causal jitter includes a future observation")
        history_ts = record["tactile_f6_history_timestamp_ns_delayed_jitter"]
        history_sequence = record["tactile_f6_history_sequence_delayed_jitter"]
        for delay_index in range(4):
            for jitter_index in range(3):
                if any(
                    later <= earlier
                    for earlier, later in zip(
                        history_ts[delay_index][jitter_index],
                        history_ts[delay_index][jitter_index][1:],
                    )
                ) or any(
                    later <= earlier
                    for earlier, later in zip(
                        history_sequence[delay_index][jitter_index],
                        history_sequence[delay_index][jitter_index][1:],
                    )
                ):
                    raise LaunchContractError("Force6D history contains duplicated native samples")
    elif record.get("tactile_f6") not in (None, []):
        raise LaunchContractError("DIFF-only Profile B must not carry fake Force6D")
    if diff_required:
        current_diff = record.get("tactile_image_deform")
        delayed_diff = record.get("tactile_image_deform_delayed")
        jittered_diff = record.get("tactile_image_deform_delayed_jitter")
        if not (
            isinstance(current_diff, list) and len(current_diff) == 5
            and isinstance(delayed_diff, list) and len(delayed_diff) == 4
            and all(isinstance(paths, list) and len(paths) == 5 for paths in delayed_diff)
            and isinstance(jittered_diff, list) and len(jittered_diff) == 4
            and all(
                isinstance(options, list) and len(options) == 3
                and all(isinstance(paths, list) and len(paths) == 5 for paths in options)
                for options in jittered_diff
            )
            and _flattened_size(record.get("tactile_deform_timestamp_ns_delayed_jitter"))
            == 4 * 3 * 5
        ):
            raise LaunchContractError("Profile A/B requires DIFF paths/timestamps [4,3,5]")
        diff_ts = record["tactile_deform_timestamp_ns_delayed_jitter"]
        decisions = record.get("tactile_decision_timestamp_ns_delayed")
        if any(
            timestamp > decisions[delay]
            for delay in range(4)
            for option in diff_ts[delay]
            for timestamp in option
        ):
            raise LaunchContractError("DIFF causal jitter includes a future observation")
        diff_paths = current_diff + [path for delay in jittered_diff for option in delay for path in option]
        missing_diff = [path for path in diff_paths if not (image_root / path).is_file()]
        if missing_diff:
            raise LaunchContractError(f"DIFF training image is missing: {missing_diff[0]}")
    elif record.get("tactile_image_deform") not in (None, []):
        raise LaunchContractError("Force6D-only ablation cannot carry DIFF")
    if record.get("schema_version") != "revo3-trex-json-v1":
        raise LaunchContractError("unsupported Revo3 training-record schema")
    validated_stats_path = _validate_stats(
        manifest,
        tactile_profile=tactile_profile,
        profile_manifest=profile_manifest,
    )
    if validated_stats_path != Path(manifest["stats_path"]).resolve():
        raise LaunchContractError("conversion manifest points to a different statistics file")
    return data_json, manifest


def _option(command: List[str], name: str, value: Any) -> None:
    command.extend((f"--{name}", str(value)))


def build_train_command(args: argparse.Namespace, config: Mapping[str, Any]) -> List[str]:
    base_model = _validate_base_model(args.base_model)
    accelerate_config = args.accelerate_config.resolve()
    if not accelerate_config.is_file():
        raise LaunchContractError(f"accelerate config does not exist: {accelerate_config}")
    if args.stage not in config["training"].get("stages", {}):
        raise LaunchContractError(f"training stage is not configured: {args.stage}")
    profile_name = args.tactile_profile or config["default_tactile_profile"]
    profile, profile_manifest = _validate_profile_manifest(
        args.tactile_profile_manifest.resolve(),
        profile=profile_name,
        config=config,
        ablation_ack=args.ack_tactile_ablation,
    )
    checkpoint, source_checkpoint_sha256 = validate_stage_checkpoint(
        args.checkpoint,
        args.checkpoint_id,
        mode=args.mode,
        stage=args.stage,
        resume_kind=args.resume_kind,
        config=config,
        profile_name=profile_name,
        profile_manifest=profile_manifest,
        ablation_ack=args.ack_heterogeneous_midtrain_ablation,
    )
    vqvae_checkpoint = None
    vqvae_artifact = None
    if args.stage != "w0" and profile["use_tactile_vqvae"]:
        if args.vqvae_checkpoint is None or not args.vqvae_checkpoint.resolve().is_file():
            raise LaunchContractError(f"{profile_name} requires a Revo VQ-VAE checkpoint")
        if args.vqvae_artifact is None or not args.vqvae_artifact.resolve().is_file():
            raise LaunchContractError(f"{profile_name} requires a Revo VQ-VAE artifact")
        vqvae_checkpoint = args.vqvae_checkpoint.resolve()
        vqvae_artifact = args.vqvae_artifact.resolve()
    deform_checkpoint = None
    deform_artifact = None
    if args.stage != "w0" and profile["use_tactile_deform"]:
        if args.deform_encoder_checkpoint is None or not args.deform_encoder_checkpoint.resolve().is_file():
            raise LaunchContractError(f"{profile_name} requires a Revo DIFF encoder checkpoint")
        if args.deform_encoder_artifact is None or not args.deform_encoder_artifact.resolve().is_file():
            raise LaunchContractError(f"{profile_name} requires a Revo DIFF encoder artifact")
        deform_checkpoint = args.deform_encoder_checkpoint.resolve()
        deform_artifact = args.deform_encoder_artifact.resolve()
    data_json, conversion = validate_json_dataset(
        args.data_json,
        args.conversion_manifest,
        args.readiness_manifest,
        tactile_profile=profile_name,
        profile_manifest=profile_manifest,
        expected_split="sft_train" if args.stage == "sft" else "midtrain_train",
    )
    development_json, development_conversion = validate_json_dataset(
        args.development_data_json,
        args.development_conversion_manifest,
        args.development_readiness_manifest,
        tactile_profile=profile_name,
        profile_manifest=profile_manifest,
        expected_split="development",
    )
    if development_conversion.get("split_manifest_sha256") != conversion.get(
        "split_manifest_sha256"
    ):
        raise LaunchContractError("train and development JSONs come from different split manifests")
    for field in (
        "stats_path",
        "stats_artifact_path",
        "statistics_sha256",
        "statistics_artifact_sha256",
    ):
        if development_conversion.get(field) != conversion.get(field):
            raise LaunchContractError(
                f"train/development must share one frozen normalization artifact: {field}"
            )
    if args.resume_kind.startswith("revo_"):
        parent_lineage = _read_json(
            checkpoint / "checkpoint_lineage.json", "Revo checkpoint lineage"
        )
        if parent_lineage.get("split_manifest_sha256") != conversion.get(
            "split_manifest_sha256"
        ):
            raise LaunchContractError(
                "source checkpoint and current data use different split manifests"
            )
        if parent_lineage.get("normalization_statistics_sha256") != conversion.get(
            "statistics_sha256"
        ) or parent_lineage.get("normalization_artifact_sha256") != conversion.get(
            "statistics_artifact_sha256"
        ):
            raise LaunchContractError(
                "source checkpoint and current data use different frozen normalization"
            )
    artifact_expected = {
        "tactile_profile": profile_name,
        "checkpoint_family_id": profile_manifest["checkpoint_family_id"],
        "normalization_family_id": profile_manifest["normalization_family_id"],
        "capability_manifest_sha256": profile_manifest[
            "capability_manifest_sha256"
        ],
        "split_manifest_sha256": conversion["split_manifest_sha256"],
    }
    try:
        if vqvae_checkpoint is not None:
            assert vqvae_artifact is not None
            validate_revo_vqvae_artifact(
                vqvae_checkpoint, vqvae_artifact, **artifact_expected
            )
        if deform_checkpoint is not None:
            assert deform_artifact is not None
            validate_revo_deform_artifact(
                deform_checkpoint, deform_artifact, **artifact_expected
            )
    except ValueError as exc:
        raise LaunchContractError(str(exc)) from exc
    output_dir = args.output_dir.resolve()
    if args.num_processes < 1:
        raise LaunchContractError("--num-processes must be positive")

    contract = config["fixed_contract"]
    train = dict(config["training"])
    train.update(train.pop("stages")[args.stage])
    command = [
        "accelerate",
        "launch",
        "--config_file",
        str(accelerate_config),
        "--num_processes",
        str(args.num_processes),
        str(REPO_ROOT / "scripts" / "train.py"),
        "--model_path",
        str(base_model),
        "--data_format",
        "json",
        "--data_path",
        str(data_json),
        "--val_data_path",
        str(development_json),
        "--stats_path",
        str(conversion["stats_path"]),
        "--val_stats_path",
        str(conversion["stats_path"]),
        "--output_dir",
        str(output_dir),
        "--log_dir",
        str(output_dir),
        "--experiment_name",
        "revo3_trex_midtrain_like" if args.mode == "main" else "revo3_trex_midtrain_ablation",
        "--run_name",
        args.run_name,
        "--resume_checkpoint",
        str(checkpoint),
        "--resume_source",
        args.resume_kind,
    ]
    fixed_names = (
        "action_dim",
        "action_chunk",
        "tactile_num_fingers",
        "use_robot_state",
    )
    for name in fixed_names:
        _option(command, name, contract[name])
    profile_flags = dict(profile)
    if args.stage == "w0":
        profile_flags.update(
            use_tactile_vec=0,
            use_tactile_deform=0,
            use_tactile_vqvae=0,
            use_tactile_code=0,
        )
    for name in (
        "use_tactile_vec",
        "use_tactile_deform",
        "use_tactile_vqvae",
        "use_tactile_code",
    ):
        _option(command, name, profile_flags[name])
    _option(command, "revo_training_stage", args.stage)
    _option(command, "tactile_profile", profile_name)
    _option(command, "checkpoint_family_id", profile_manifest["checkpoint_family_id"])
    _option(command, "normalization_family_id", profile_manifest["normalization_family_id"])
    _option(command, "source_checkpoint_sha256", source_checkpoint_sha256)
    _option(command, "split_manifest_sha256", conversion["split_manifest_sha256"])
    _option(command, "normalization_statistics_sha256", conversion["statistics_sha256"])
    _option(
        command,
        "normalization_artifact_sha256",
        conversion["statistics_artifact_sha256"],
    )
    _option(
        command,
        "tactile_profile_manifest_sha256",
        _sha256_file(args.tactile_profile_manifest.resolve()),
    )
    _option(
        command,
        "capability_manifest_sha256",
        profile_manifest["capability_manifest_sha256"],
    )
    if vqvae_checkpoint is not None:
        _option(command, "vqvae_ckpt", vqvae_checkpoint)
        _option(command, "vqvae_artifact", vqvae_artifact)
    if deform_checkpoint is not None:
        _option(command, "deform_encoder_ckpt", deform_checkpoint)
        _option(command, "deform_encoder_artifact", deform_artifact)
    effective_batch = int(train["effective_batch"])
    denominator = int(train["train_bsz_per_gpu"]) * int(args.num_processes)
    if effective_batch % denominator:
        raise LaunchContractError(
            "effective batch is not divisible by per-GPU batch * num_processes"
        )
    gradient_accumulation = effective_batch // denominator
    for name in (
        "n_epochs",
        "save_freq",
        "train_bsz_per_gpu",
        "learning_rate",
        "min_lr_ratio",
        "warmup_rates",
        "weight_decay",
        "max_grad_norm",
        "seed",
        "training_stage",
        "tactile_intermediate_size",
        "cascaded_total_steps",
        "cascaded_split_step",
        "cascaded_tactile_dropout",
        "cascaded_loss_weight",
        "tactile_loss_weight",
        "use_flare",
        "n_flare_tokens_per_frame",
        "n_flare_steps",
        "flare_loss_weight",
        "flare_frame_stride",
        "flare_layer_index",
        "val_ratio",
        "val_freq",
        "max_val_batches",
        "max_steps",
        "state_dropout",
        "tracking_error_clip_rad",
        "revo_image_augmentation",
        "revo_tactile_noise_augmentation",
        "augmentation_seed",
    ):
        _option(command, name, train[name])
    _option(command, "gradient_accumulation_steps", gradient_accumulation)
    command.extend(("--image_size", *(str(x) for x in train["image_size"])))
    return command


def validate_serve_checkpoint(
    checkpoint: Path, config: Mapping[str, Any] | None = None
) -> Path:
    checkpoint = _checkpoint_model_path(checkpoint)
    training_args = _checkpoint_training_args(checkpoint)
    if not training_args:
        raise LaunchContractError("serving requires checkpoint/training_args.json")
    if not (checkpoint / "processor").is_dir():
        raise LaunchContractError("serving requires the processor/ saved with the checkpoint")
    required = {
        "action_dim": 21,
        "action_chunk": 16,
        "tactile_num_fingers": 5,
        "use_robot_state": 1,
        "camera_profile": "revo3_full_center_v1",
        "view_slots": {"slow": "full", "fast": "fixed_center"},
    }
    mismatches = {
        key: (value, training_args.get(key))
        for key, value in required.items()
        if training_args.get(key) != value
    }
    profile_name = training_args.get("tactile_profile")
    profiles = None if config is None else config.get("tactile_profiles")
    if not isinstance(profiles, dict) or profile_name not in profiles:
        mismatches["tactile_profile"] = ("known profile", profile_name)
    else:
        for key in (
            "use_tactile_vec",
            "use_tactile_deform",
            "use_tactile_vqvae",
            "use_tactile_code",
        ):
            expected = profiles[profile_name][key]
            if training_args.get(key) != expected:
                mismatches[key] = (expected, training_args.get(key))
    for family_key in ("checkpoint_family_id", "normalization_family_id"):
        if not isinstance(training_args.get(family_key), str) or not training_args[family_key]:
            mismatches[family_key] = ("non-empty string", training_args.get(family_key))
    if any("emg" in key.lower() for key in training_args) or mismatches:
        raise LaunchContractError(
            f"checkpoint is not the reviewed EMG-free Revo3 contract: {mismatches}"
        )
    return checkpoint


def _validate_serve_normalization(
    *,
    checkpoint: Path,
    stats_path: Path,
    artifact_path: Path,
    training_args: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Bind serve-time statistics to the frozen MIDTRAIN_TRAIN lineage."""

    stats_path = stats_path.resolve()
    artifact_path = artifact_path.resolve()
    stats = _read_json(stats_path, "normalization statistics")
    artifact = _read_json(artifact_path, "normalization companion artifact")
    lineage = _read_json(
        checkpoint / "checkpoint_lineage.json", "Revo checkpoint lineage"
    )
    stats_sha = _sha256_file(stats_path)
    artifact_sha = _sha256_file(artifact_path)
    checkpoint_sha = _sha256_file(checkpoint / "model.pt")

    required_training_hashes = (
        "normalization_statistics_sha256",
        "normalization_artifact_sha256",
        "split_manifest_sha256",
        "tactile_profile_manifest_sha256",
        "capability_manifest_sha256",
    )
    mismatches: dict[str, tuple[object, object]] = {}
    for key in required_training_hashes:
        value = training_args.get(key)
        if not isinstance(value, str) or len(value) != 64:
            mismatches[f"training_args.{key}"] = ("64-char SHA-256", value)

    expected_artifact = {
        "schema_version": "revo3-normalization-artifact-v1",
        "statistics_sha256": stats_sha,
        "source_split": "midtrain_train",
        "split_manifest_sha256": training_args.get("split_manifest_sha256"),
        "joint_order_hash": JOINT_ORDER_HASH,
        "tactile_profile": training_args.get("tactile_profile"),
        "checkpoint_family_id": training_args.get("checkpoint_family_id"),
        "normalization_family_id": training_args.get("normalization_family_id"),
        "capability_manifest_sha256": training_args.get(
            "capability_manifest_sha256"
        ),
    }
    for key, expected in expected_artifact.items():
        if artifact.get(key) != expected:
            mismatches[f"artifact.{key}"] = (expected, artifact.get(key))
    if not isinstance(artifact.get("statistics_path"), str) or not artifact[
        "statistics_path"
    ]:
        mismatches["artifact.statistics_path"] = (
            "non-empty provenance path",
            artifact.get("statistics_path"),
        )
    episode_ids = artifact.get("stats_episode_ids")
    if (
        not isinstance(episode_ids, list)
        or not episode_ids
        or any(not isinstance(item, str) or not item for item in episode_ids)
    ):
        mismatches["artifact.stats_episode_ids"] = (
            "non-empty MIDTRAIN_TRAIN episode-id list",
            episode_ids,
        )
    if training_args.get("normalization_statistics_sha256") != stats_sha:
        mismatches["training_args.normalization_statistics_sha256"] = (
            stats_sha,
            training_args.get("normalization_statistics_sha256"),
        )
    if training_args.get("normalization_artifact_sha256") != artifact_sha:
        mismatches["training_args.normalization_artifact_sha256"] = (
            artifact_sha,
            training_args.get("normalization_artifact_sha256"),
        )

    expected_lineage = {
        "schema_version": "revo3-checkpoint-lineage-v1",
        "checkpoint_sha256": checkpoint_sha,
        "split_manifest_sha256": training_args.get("split_manifest_sha256"),
        "normalization_statistics_sha256": stats_sha,
        "normalization_artifact_sha256": artifact_sha,
        "capability_manifest_sha256": training_args.get(
            "capability_manifest_sha256"
        ),
        "tactile_profile": training_args.get("tactile_profile"),
        "checkpoint_family_id": training_args.get("checkpoint_family_id"),
        "normalization_family_id": training_args.get("normalization_family_id"),
        "tactile_profile_manifest_sha256": training_args.get(
            "tactile_profile_manifest_sha256"
        ),
        "joint_order_hash": JOINT_ORDER_HASH,
        "camera_profile": "revo3_full_center_v1",
        "view_slots": {"slow": "full", "fast": "fixed_center"},
    }
    for key, expected in expected_lineage.items():
        if lineage.get(key) != expected:
            mismatches[f"lineage.{key}"] = (expected, lineage.get(key))
    training_stage = training_args.get("revo_training_stage")
    if training_stage not in {"w0", "w1", "midtrain", "sft"}:
        mismatches["training_args.revo_training_stage"] = (
            "w0|w1|midtrain|sft",
            training_stage,
        )
    elif lineage.get("stage") != training_stage:
        mismatches["lineage.stage"] = (training_stage, lineage.get("stage"))

    if not isinstance(stats, dict) or len(stats) != 1:
        mismatches["statistics.schema"] = ("exactly one dataset block", type(stats))
    if mismatches:
        raise LaunchContractError(
            f"serving frozen normalization lineage failed: {mismatches}"
        )
    return artifact


def build_serve_command(args: argparse.Namespace, config: Mapping[str, Any]) -> List[str]:
    base_model = _validate_base_model(args.base_model)
    checkpoint = validate_serve_checkpoint(args.checkpoint, config)
    stats_path = args.stats_path.resolve()
    artifact_path_arg = getattr(args, "stats_artifact_path", None)
    if artifact_path_arg is None:
        raise LaunchContractError(
            "serving requires the frozen normalization companion artifact"
        )
    artifact_path = artifact_path_arg.resolve()
    contract = config["fixed_contract"]
    infer = config["inference"]
    training_args = _checkpoint_training_args(checkpoint)
    _validate_serve_normalization(
        checkpoint=checkpoint,
        stats_path=stats_path,
        artifact_path=artifact_path,
        training_args=training_args,
    )
    tactile_profile = str(training_args["tactile_profile"])
    identity_manifest_out = getattr(args, "identity_manifest_out", None)
    if identity_manifest_out is not None:
        identity = build_revo_server_identity(
            checkpoint_path=checkpoint,
            normalization_statistics_path=stats_path,
            normalization_artifact_path=artifact_path,
            model_config_path=base_model / "config.json",
            camera_profile=str(infer["camera_profile"]),
            tactile_profile=tactile_profile,
            joint_order_hash=JOINT_ORDER_HASH,
        )
        identity_path = Path(identity_manifest_out).resolve()
        identity_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = identity_path.with_suffix(identity_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(identity.as_mapping(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(identity_path)
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "test.py"),
        "--checkpoint_path",
        str(checkpoint),
        "--base_model_path",
        str(base_model),
        "--stats_path",
        str(stats_path),
        "--stats_artifact_path",
        str(artifact_path),
        "--normalization_statistics_sha256",
        str(training_args["normalization_statistics_sha256"]),
        "--normalization_artifact_sha256",
        str(training_args["normalization_artifact_sha256"]),
        "--cuda",
        args.cuda,
        "--port",
        str(args.port),
        "--camera_profile",
        str(infer["camera_profile"]),
        "--tactile_profile",
        tactile_profile,
    ]
    for name in (
        "action_dim",
        "action_chunk",
        "tactile_num_fingers",
        "use_robot_state",
    ):
        _option(command, name, contract[name])
    # Tactile flags are profile-specific.  Never read them from fixed_contract
    # (which deliberately contains only embodiment-invariant values), and
    # never let the server silently fall back to the legacy Force6D profile.
    profile = config["tactile_profiles"][tactile_profile]
    for name in (
        "use_tactile_vec",
        "use_tactile_deform",
        "use_tactile_code",
    ):
        _option(command, name, profile[name])
    if profile["use_tactile_code"]:
        command.extend(("--vqvae_mode", "embedded"))
    for name in ("cascaded_total_steps", "cascaded_split_step", "disable_tactile"):
        _option(command, name, infer[name])
    command.extend(("--image_size", *(str(x) for x in infer["image_size"])))
    return command


def format_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="operation", required=True)

    train = subparsers.add_parser("train", help="validate and launch a real-data Revo run")
    train.add_argument("--base-model", type=Path, required=True)
    train.add_argument("--checkpoint", type=Path, required=True)
    train.add_argument("--checkpoint-id", required=True)
    train.add_argument(
        "--resume-kind",
        choices=(
            "official_pretrain",
            "official_midtrain_ablation",
            "revo_w0",
            "revo_w1",
            "revo_midtrain",
            "revo_sft",
        ),
        required=True,
    )
    train.add_argument("--data-json", type=Path, required=True)
    train.add_argument("--conversion-manifest", type=Path, required=True)
    train.add_argument("--readiness-manifest", type=Path, required=True)
    train.add_argument("--development-data-json", type=Path, required=True)
    train.add_argument("--development-conversion-manifest", type=Path, required=True)
    train.add_argument("--development-readiness-manifest", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--run-name", default="revo3_v1")
    train.add_argument("--num-processes", type=int, default=1)
    train.add_argument(
        "--accelerate-config", type=Path, default=REPO_ROOT / "config" / "sft_qwen.yaml"
    )
    train.add_argument("--mode", choices=("main", "midtrain_ablation"), default="main")
    train.add_argument(
        "--stage", choices=("w0", "w1", "midtrain", "sft"), default="midtrain"
    )
    train.add_argument("--tactile-profile", default=None)
    train.add_argument("--tactile-profile-manifest", type=Path, required=True)
    train.add_argument("--vqvae-checkpoint", type=Path, default=None)
    train.add_argument("--vqvae-artifact", type=Path, default=None)
    train.add_argument("--deform-encoder-checkpoint", type=Path, default=None)
    train.add_argument("--deform-encoder-artifact", type=Path, default=None)
    train.add_argument("--ack-tactile-ablation", action="store_true")
    train.add_argument("--ack-heterogeneous-midtrain-ablation", action="store_true")
    train.add_argument("--execute", action="store_true", help="run after all gates pass")

    serve = subparsers.add_parser("serve", help="validate and launch the T-Rex ZMQ server")
    serve.add_argument("--base-model", type=Path, required=True)
    serve.add_argument("--checkpoint", type=Path, required=True)
    serve.add_argument("--stats-path", type=Path, required=True)
    serve.add_argument("--stats-artifact-path", type=Path, required=True)
    serve.add_argument("--cuda", default="0")
    serve.add_argument("--port", type=int, default=5555)
    serve.add_argument(
        "--identity-manifest-out",
        type=Path,
        default=None,
        help=(
            "Write the locally verified server identity used as the controller's "
            "production trust anchor."
        ),
    )
    serve.add_argument("--execute", action="store_true", help="run after all gates pass")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = load_launch_config(args.config.resolve())
        command = (
            build_train_command(args, config)
            if args.operation == "train"
            else build_serve_command(args, config)
        )
    except LaunchContractError as exc:
        parser.error(str(exc))
    print(format_command(command))
    if args.execute:
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    else:
        print("DRY RUN ONLY: pass --execute after reviewing the command and artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
