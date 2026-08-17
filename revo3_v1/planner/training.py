"""Fail-closed Planner LoRA data audit and deterministic command builder.

This module deliberately performs no download and no training on import.  It
freezes the V1 LoRA hyperparameters, verifies real-image split manifests, and
constructs an auditable command for a caller-supplied trainer entry point.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import hashlib
import json
from pathlib import Path
import shlex
from typing import Any, Dict, Mapping, Sequence, Tuple

from .schema import PlannerDecision
from .artifacts import QWEN3_VL_MODEL_ID, QWEN3_VL_REVISION
from revo3_v1.emg.primitives import normalize_emg_primitive


@dataclass(frozen=True)
class PlannerLoRAConfig:
    base_model: str = QWEN3_VL_MODEL_ID
    base_revision: str = QWEN3_VL_REVISION
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    learning_rate: float = 5e-5
    epochs: int = 3
    effective_batch: int = 32
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_sequence_length: int = 1024
    max_new_tokens: int = 384
    precision: str = "bf16"
    target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

    def validate(self) -> None:
        if self.base_model != QWEN3_VL_MODEL_ID or self.base_revision != QWEN3_VL_REVISION:
            raise ValueError("Planner V1 freezes the exact Qwen3-VL model ID and revision")
        frozen = {
            "rank": (self.rank, 16),
            "alpha": (self.alpha, 32),
            "dropout": (self.dropout, 0.05),
            "learning_rate": (self.learning_rate, 5e-5),
        }
        for name, (actual, expected) in frozen.items():
            if actual != expected:
                raise ValueError(f"V1 freezes {name}={expected}, got {actual}")
        if not 1 <= self.epochs <= 3:
            raise ValueError("Planner LoRA epochs must be in [1, 3]")
        if self.effective_batch not in range(32, 65):
            raise ValueError("Planner effective batch must be in [32, 64]")
        if self.precision != "bf16":
            raise ValueError("Planner V1 training precision is bf16")
        if self.max_new_tokens != 384:
            raise ValueError("Planner V1 production schema budget is frozen to 384 tokens")


@dataclass(frozen=True)
class PlannerDatasetAudit:
    root: str
    split_counts: Mapping[str, int]
    negative_fraction: float
    real_data_verified: bool
    dataset_sha256: str
    ood_isolation_dimensions: Tuple[str, ...]
    schema_version: str = "planner-lora-audit-v1"


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"Planner split is empty: {path}")
    return rows


def validate_planner_dataset(root: str | Path) -> PlannerDatasetAudit:
    """Verify real-image records, causal four-frame context and split isolation."""

    dataset_root = Path(root).resolve()
    manifest_path = dataset_root / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Planner dataset_manifest.json is required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("data_origin") != "real_camera":
        raise ValueError("Planner LoRA refuses synthetic/mock data; data_origin must be real_camera")
    protocol = manifest.get("evaluation_protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("Planner manifest must freeze evaluation_protocol")
    ood_isolation = protocol.get("ood_test_isolation")
    required_ood_isolation = {"object_instance_id", "day_id"}
    if not isinstance(ood_isolation, list) or set(ood_isolation) != required_ood_isolation:
        raise ValueError(
            "Planner evaluation_protocol.ood_test_isolation must freeze exactly "
            "object_instance_id and day_id"
        )
    split_names = ("train", "val", "id_test", "ood_test")
    split_counts: Dict[str, int] = {}
    group_owner: Dict[tuple[str, str, str], str] = {}
    record_ids = set()
    train_negative = 0
    train_total = 0
    reference_object_instances: set[str] = set()
    reference_days: set[str] = set()
    ood_object_instances: set[str] = set()
    ood_days: set[str] = set()
    for split in split_names:
        rows = _read_jsonl(dataset_root / f"{split}.jsonl")
        split_counts[split] = len(rows)
        for index, row in enumerate(rows):
            prefix = f"{split}[{index}]"
            record_id = str(row.get("record_id", ""))
            if not record_id or record_id in record_ids:
                raise ValueError(f"{prefix}: missing or duplicate record_id")
            record_ids.add(record_id)
            group = tuple(str(row.get(key, "")) for key in ("object_instance_id", "day_id", "scene_id"))
            if not all(group):
                raise ValueError(f"{prefix}: object_instance_id/day_id/scene_id are required")
            owner = group_owner.setdefault(group, split)
            if owner != split:
                raise ValueError(f"Planner split leakage for group {group}: {owner} vs {split}")
            object_instance_id, day_id, _ = group
            if split == "ood_test":
                ood_object_instances.add(object_instance_id)
                ood_days.add(day_id)
            else:
                reference_object_instances.add(object_instance_id)
                reference_days.add(day_id)
            primitive = normalize_emg_primitive(row.get("primitive"))
            if primitive is None or not primitive.starts_task:
                raise ValueError(f"{prefix}: invalid start primitive")
            full_paths = tuple(row.get("full_view_paths", ()))
            center_path = str(row.get("center_view_path", ""))
            timestamps = tuple(int(value) for value in row.get("frame_timestamps_ns", ()))
            if len(full_paths) != 3 or not center_path or len(timestamps) != 4:
                raise ValueError(f"{prefix}: requires 3 full views, 1 center view, 4 timestamps")
            if tuple(sorted(timestamps[:3])) != timestamps[:3] or timestamps[2] != timestamps[3]:
                raise ValueError(f"{prefix}: invalid causal/aligned frame timestamps")
            for relative in (*full_paths, center_path):
                image_path = (dataset_root / str(relative)).resolve()
                try:
                    image_path.relative_to(dataset_root)
                except ValueError as exc:
                    raise ValueError(f"{prefix}: image escapes dataset root") from exc
                if not image_path.is_file():
                    raise FileNotFoundError(f"{prefix}: missing image {relative}")
            output = row.get("output")
            if not isinstance(output, Mapping):
                raise ValueError(f"{prefix}: structured output is required")
            PlannerDecision.from_mapping(
                output,
                timestamp_ns=timestamps[-1],
                expected_primitive=primitive,
            )
            if split == "train":
                train_total += 1
                if str(output.get("status", "")).upper() != "READY":
                    train_negative += 1
    object_overlap = ood_object_instances & reference_object_instances
    day_overlap = ood_days & reference_days
    if object_overlap or day_overlap:
        raise ValueError(
            "Planner ood_test must be disjoint from train/val/id_test on both "
            "object_instance_id and day_id; "
            f"object_overlap={sorted(object_overlap)}, day_overlap={sorted(day_overlap)}"
        )
    negative_fraction = train_negative / train_total
    if not 0.30 <= negative_fraction <= 0.40:
        raise ValueError(
            f"train negative/ambiguous fraction must be 0.30..0.40, got {negative_fraction:.3f}"
        )
    digest = hashlib.sha256()
    for path in [manifest_path, *(dataset_root / f"{name}.jsonl" for name in split_names)]:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return PlannerDatasetAudit(
        root=str(dataset_root),
        split_counts=split_counts,
        negative_fraction=negative_fraction,
        real_data_verified=True,
        dataset_sha256=digest.hexdigest(),
        ood_isolation_dimensions=tuple(ood_isolation),
    )


def build_lora_training_command(
    *,
    trainer_entry: str | Path | None = None,
    dataset_root: str | Path,
    output_dir: str | Path,
    config: PlannerLoRAConfig = PlannerLoRAConfig(),
) -> tuple[str, PlannerDatasetAudit]:
    """Validate first, then build a shell-safe argument vector as one string."""

    config.validate()
    audit = validate_planner_dataset(dataset_root)
    if trainer_entry is None:
        args = ["python", "-m", "revo3_v1.planner.lora_sft"]
    else:
        entry = Path(trainer_entry).resolve()
        if not entry.is_file():
            raise FileNotFoundError(f"Caller-supplied Planner trainer does not exist: {entry}")
        args = ["python", str(entry)]
    args += [
        "--model", config.base_model,
        "--model-revision", config.base_revision,
        "--dataset", audit.root,
        "--output", str(Path(output_dir).resolve()),
        "--lora-rank", str(config.rank),
        "--lora-alpha", str(config.alpha),
        "--lora-dropout", str(config.dropout),
        "--learning-rate", str(config.learning_rate),
        "--epochs", str(config.epochs),
        "--effective-batch", str(config.effective_batch),
        "--warmup-ratio", str(config.warmup_ratio),
        "--weight-decay", str(config.weight_decay),
        "--max-sequence-length", str(config.max_sequence_length),
        "--precision", config.precision,
        "--target-modules", ",".join(config.target_modules),
    ]
    return " ".join(shlex.quote(value) for value in args), audit


def write_training_audit(
    path: str | Path,
    *,
    config: PlannerLoRAConfig,
    audit: PlannerDatasetAudit,
    command: str,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {"config": asdict(config), "dataset": asdict(audit), "command": command},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--trainer", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-json", type=Path)
    args = parser.parse_args(argv)
    if args.trainer is not None and args.output is None:
        parser.error("--trainer requires --output")
    config = PlannerLoRAConfig()
    if args.output is None:
        audit = validate_planner_dataset(args.dataset)
        payload = {"config": asdict(config), "dataset": asdict(audit), "command": None}
    else:
        command, audit = build_lora_training_command(
            trainer_entry=args.trainer,
            dataset_root=args.dataset,
            output_dir=args.output,
            config=config,
        )
        payload = {"config": asdict(config), "dataset": asdict(audit), "command": command}
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.audit_json:
        args.audit_json.parent.mkdir(parents=True, exist_ok=True)
        args.audit_json.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
