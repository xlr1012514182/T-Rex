#!/usr/bin/env python3
"""Auditable train/serve launcher for the Revo3 single-hand T-Rex path.

The launcher deliberately separates command construction from execution.  It
rejects synthetic data, EMG-bearing VLA records, unapproved action labels, and
shape-incompatible datasets before it can spawn a multi-GPU process.  The
primary route resumes the released T-Rex *pretrain* checkpoint; the released
midtrain checkpoint is available only behind an explicit heterogeneous-hand
ablation acknowledgement.

This wrapper invokes the locally adapted ``scripts/train.py`` cascaded stage-2
path.  Upstream ``main`` calls that file post-training code, while upstream's
paper-scale midtraining loader lives on ``full-pipeline`` and assumes a
bimanual 62-D robot.  Consequently this is accurately described as a
Revo-specific midtrain-like run, not a reproduction of T-Rex's paper midtrain.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "revo3_v1_trex.json"

EXPECTED_CONTRACT: Dict[str, Any] = {
    "action_dim": 21,
    "action_chunk": 16,
    "tactile_num_fingers": 5,
    "use_robot_state": 1,
    "use_tactile_vec": 1,
    "use_tactile_deform": 0,
    "use_tactile_vqvae": 0,
    "use_tactile_code": 0,
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


def load_launch_config(path: Path) -> Dict[str, Any]:
    config = _read_json(path, "launch config")
    if config.get("schema_version") != "revo3-trex-launch-v1":
        raise LaunchContractError("unsupported Revo3 launch-config schema")
    contract = config.get("fixed_contract")
    if contract != EXPECTED_CONTRACT:
        raise LaunchContractError(
            f"fixed Revo3 contract changed: expected {EXPECTED_CONTRACT}, got {contract}"
        )
    return config


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


def _validate_readiness(path: Path) -> Mapping[str, Any]:
    readiness = _read_json(path, "dataset readiness manifest")
    required = {
        "schema_version": "revo3-vla-readiness-v1",
        "dataset_kind": "real_robot",
        "ready_for_training": True,
        "synthetic_fixture": False,
        "contains_emg": False,
        "action_label_source": "controller_target",
        "timestamp_alignment_verified": True,
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


def _validate_conversion_manifest(path: Path) -> Mapping[str, Any]:
    manifest = _read_json(path, "Revo3 conversion manifest")
    required = {
        "schema_version": "revo3-trex-conversion-v1",
        "action_shape": [16, 21],
        "state_shape": [21],
        "tactile_shape": [5, 6],
        "action_label_source": "controller_target",
        "contains_emg": False,
    }
    mismatches = {
        key: (expected, manifest.get(key))
        for key, expected in required.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise LaunchContractError(f"Revo3 conversion contract failed: {mismatches}")
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


def _validate_stats(data_json: Path) -> Path:
    stats_path = Path(str(data_json).replace(".json", "_statistics.json"))
    stats = _read_json(stats_path, "normalization statistics")
    if not isinstance(stats, dict) or len(stats) != 1:
        raise LaunchContractError("statistics must contain exactly one dataset block")
    block = next(iter(stats.values()))
    expected_shapes = {"action": 16 * 21, "state": 21, "tactile_f6": 5 * 6}
    for name, expected in expected_shapes.items():
        entry = block.get(name, {}) if isinstance(block, dict) else {}
        for statistic in ("q01", "q99", "mask"):
            observed = _flattened_size(entry.get(statistic))
            if observed != expected:
                raise LaunchContractError(
                    f"{name}.{statistic} has {observed} values; expected {expected}"
                )
    return stats_path.resolve()


def validate_json_dataset(
    data_json: Path, conversion_manifest: Path, readiness_manifest: Path
) -> Path:
    data_json = data_json.resolve()
    _validate_readiness(readiness_manifest)
    manifest = _validate_conversion_manifest(conversion_manifest)
    record = _first_array_record(data_json)
    emg_keys = [
        key for key in record
        if "emg" in key.lower() and key.lower() != "contains_emg"
    ]
    if emg_keys or record.get("contains_emg") is not False:
        raise LaunchContractError(
            f"EMG is forbidden in T-Rex records; offending keys={emg_keys}"
        )
    expected_sizes = {"action": 16 * 21, "state_fast": 21, "tactile_f6": 5 * 6}
    for key, expected in expected_sizes.items():
        observed = _flattened_size(record.get(key))
        if observed != expected:
            raise LaunchContractError(f"first record {key} size={observed}; expected {expected}")
    if record.get("action_label_source") != "controller_target":
        raise LaunchContractError("action labels must be recorded controller targets")
    if record.get("schema_version") != "revo3-trex-json-v1":
        raise LaunchContractError("unsupported Revo3 training-record schema")
    if Path(manifest.get("stats_path", "")).name != _validate_stats(data_json).name:
        raise LaunchContractError("conversion manifest points to a different statistics file")
    return data_json


def _option(command: List[str], name: str, value: Any) -> None:
    command.extend((f"--{name}", str(value)))


def build_train_command(args: argparse.Namespace, config: Mapping[str, Any]) -> List[str]:
    base_model = _validate_base_model(args.base_model)
    accelerate_config = args.accelerate_config.resolve()
    if not accelerate_config.is_file():
        raise LaunchContractError(f"accelerate config does not exist: {accelerate_config}")
    checkpoint = validate_checkpoint(
        args.checkpoint,
        args.checkpoint_id,
        args.mode,
        config,
        args.ack_heterogeneous_midtrain_ablation,
    )
    data_json = validate_json_dataset(
        args.data_json, args.conversion_manifest, args.readiness_manifest
    )
    output_dir = args.output_dir.resolve()
    if args.num_processes < 1:
        raise LaunchContractError("--num-processes must be positive")

    contract = config["fixed_contract"]
    train = config["training"]
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
        "pretrain" if args.mode == "main" else "midtrain",
    ]
    fixed_names = (
        "action_dim",
        "action_chunk",
        "tactile_num_fingers",
        "use_robot_state",
        "use_tactile_vec",
        "use_tactile_deform",
        "use_tactile_vqvae",
        "use_tactile_code",
    )
    for name in fixed_names:
        _option(command, name, contract[name])
    for name in (
        "n_epochs",
        "save_freq",
        "train_bsz_per_gpu",
        "gradient_accumulation_steps",
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
    ):
        _option(command, name, train[name])
    command.extend(("--image_size", *(str(x) for x in train["image_size"])))
    return command


def validate_serve_checkpoint(checkpoint: Path) -> Path:
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
        "use_tactile_vec": 1,
        "use_tactile_deform": 0,
        "use_tactile_vqvae": 0,
        "use_tactile_code": 0,
    }
    mismatches = {
        key: (value, training_args.get(key))
        for key, value in required.items()
        if training_args.get(key) != value
    }
    if any("emg" in key.lower() for key in training_args) or mismatches:
        raise LaunchContractError(
            f"checkpoint is not the reviewed EMG-free Revo3 contract: {mismatches}"
        )
    return checkpoint


def build_serve_command(args: argparse.Namespace, config: Mapping[str, Any]) -> List[str]:
    base_model = _validate_base_model(args.base_model)
    checkpoint = validate_serve_checkpoint(args.checkpoint)
    stats_path = args.stats_path.resolve()
    _read_json(stats_path, "normalization statistics")
    contract = config["fixed_contract"]
    infer = config["inference"]
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "test.py"),
        "--checkpoint_path",
        str(checkpoint),
        "--base_model_path",
        str(base_model),
        "--stats_path",
        str(stats_path),
        "--cuda",
        args.cuda,
        "--port",
        str(args.port),
    ]
    for name in (
        "action_dim",
        "action_chunk",
        "tactile_num_fingers",
        "use_robot_state",
        "use_tactile_vec",
        "use_tactile_deform",
        "use_tactile_code",
    ):
        _option(command, name, contract[name])
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
    train.add_argument("--data-json", type=Path, required=True)
    train.add_argument("--conversion-manifest", type=Path, required=True)
    train.add_argument("--readiness-manifest", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--run-name", default="revo3_v1")
    train.add_argument("--num-processes", type=int, default=1)
    train.add_argument(
        "--accelerate-config", type=Path, default=REPO_ROOT / "config" / "sft_qwen.yaml"
    )
    train.add_argument("--mode", choices=("main", "midtrain_ablation"), default="main")
    train.add_argument("--ack-heterogeneous-midtrain-ablation", action="store_true")
    train.add_argument("--execute", action="store_true", help="run after all gates pass")

    serve = subparsers.add_parser("serve", help="validate and launch the T-Rex ZMQ server")
    serve.add_argument("--base-model", type=Path, required=True)
    serve.add_argument("--checkpoint", type=Path, required=True)
    serve.add_argument("--stats-path", type=Path, required=True)
    serve.add_argument("--cuda", default="0")
    serve.add_argument("--port", type=int, default=5555)
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
