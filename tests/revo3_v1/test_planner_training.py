import json

import pytest

from revo3_v1.planner import (
    PlannerLoRAConfig,
    build_lora_training_command,
    validate_planner_dataset,
)
from revo3_v1.planner.lora_sft import main as lora_main
from revo3_v1.planner.artifacts import (
    ADAPTER_MANIFEST_NAME,
    PLANNER_ADAPTER_SCHEMA,
    PROCESSOR_SCHEMA,
    QWEN3_VL_MODEL_ID,
    QWEN3_VL_REVISION,
    adapter_files_sha256,
    load_and_validate_adapter_manifest,
)
from revo3_v1.planner import Qwen3VLBackend


def output(ready=True):
    return {
        "schema_version": "planner_v1",
        "status": "READY" if ready else "NOT_READY",
        "primitive": "POWER_GRASP",
        "target_category": "bottle" if ready else None,
        "target_part": "body" if ready else "",
        "grasp_style": "power",
        "target_present": ready,
        "near_ready": ready,
        "center_ready": ready,
        "compatible": ready,
        "ambiguous": False,
        "target_region": [0.5, 0.5, 0.5, 0.5] if ready else None,
        "ready_frame_count": 3 if ready else 0,
        "confidence": 0.9,
        "instruction": "Grasp the centered bottle using a power grasp and hold it securely." if ready else "",
        "reason_code": "READY" if ready else "TOO_FAR",
    }


def make_dataset(root, origin="real_camera"):
    (root / "images").mkdir()
    (root / "dataset_manifest.json").write_text(
        json.dumps({
            "data_origin": origin,
            "evaluation_protocol": {
                "ood_test_isolation": ["object_instance_id", "day_id"]
            },
        }),
        encoding="utf-8",
    )
    for split, count in (("train", 10), ("val", 1), ("id_test", 1), ("ood_test", 1)):
        rows = []
        for index in range(count):
            paths = []
            for view in range(4):
                path = root / "images" / f"{split}-{index}-{view}.png"
                path.write_bytes(b"fixture")
                paths.append(str(path.relative_to(root)))
            rows.append({
                "record_id": f"{split}-{index}",
                "object_instance_id": f"object-{split}-{index}",
                "day_id": f"day-{split}",
                "scene_id": f"scene-{index}",
                "primitive": "POWER_GRASP",
                "full_view_paths": paths[:3],
                "center_view_path": paths[3],
                "frame_timestamps_ns": [1, 2, 3, 3],
                "output": output(ready=not (split == "train" and index < 3)),
            })
        (root / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )


def test_real_dataset_validation_and_frozen_command(tmp_path):
    make_dataset(tmp_path)
    audit = validate_planner_dataset(tmp_path)
    assert audit.real_data_verified
    assert audit.negative_fraction == pytest.approx(0.3)
    trainer = tmp_path / "trainer.py"
    trainer.write_text("# caller supplied trainer\n", encoding="utf-8")
    command, _ = build_lora_training_command(
        trainer_entry=trainer,
        dataset_root=tmp_path,
        output_dir=tmp_path / "run",
    )
    assert "--lora-rank 16" in command
    assert "--lora-alpha 32" in command
    assert "--learning-rate 5e-05" in command
    assert f"--model-revision {QWEN3_VL_REVISION}" in command
    builtin_command, _ = build_lora_training_command(
        dataset_root=tmp_path,
        output_dir=tmp_path / "builtin-run",
    )
    assert "-m revo3_v1.planner.lora_sft" in builtin_command
    assert lora_main([
        "--dataset", str(tmp_path),
        "--output", str(tmp_path / "dry-run"),
        "--dry-run",
    ]) == 0
    assert (tmp_path / "dry-run" / "run_contract.json").is_file()


def test_synthetic_data_and_hyperparameter_drift_fail_closed(tmp_path):
    make_dataset(tmp_path, origin="synthetic")
    with pytest.raises(ValueError, match="refuses synthetic"):
        validate_planner_dataset(tmp_path)
    with pytest.raises(ValueError, match="rank=16"):
        PlannerLoRAConfig(rank=8).validate()


@pytest.mark.parametrize("overlap_field", ["object_instance_id", "day_id"])
def test_ood_requires_independent_object_instance_and_day(tmp_path, overlap_field):
    make_dataset(tmp_path)
    train = json.loads((tmp_path / "train.jsonl").read_text().splitlines()[0])
    ood_path = tmp_path / "ood_test.jsonl"
    ood = json.loads(ood_path.read_text().splitlines()[0])
    ood[overlap_field] = train[overlap_field]
    # Keep the full tuple unique so this specifically catches single-dimension
    # leakage rather than the older exact-group check.
    ood["scene_id"] = "different-scene"
    ood_path.write_text(json.dumps(ood) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ood_test must be disjoint"):
        validate_planner_dataset(tmp_path)


def test_ood_isolation_dimensions_are_frozen_in_manifest(tmp_path):
    make_dataset(tmp_path)
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["evaluation_protocol"]["ood_test_isolation"] = ["day_id"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="must freeze exactly"):
        validate_planner_dataset(tmp_path)


def test_adapter_manifest_binds_base_revision_processor_and_files(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights-fixture")
    manifest = {
        "schema_version": PLANNER_ADAPTER_SCHEMA,
        "base_model": QWEN3_VL_MODEL_ID,
        "base_revision": QWEN3_VL_REVISION,
        "processor_schema": PROCESSOR_SCHEMA,
        "processor_sha256": "a" * 64,
        "adapter_files_sha256": adapter_files_sha256(adapter),
    }
    (adapter / ADAPTER_MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    loaded = load_and_validate_adapter_manifest(
        adapter,
        expected_model_id=QWEN3_VL_MODEL_ID,
        expected_revision=QWEN3_VL_REVISION,
        expected_processor_sha256="a" * 64,
    )
    assert loaded["adapter_files_sha256"] == manifest["adapter_files_sha256"]
    assert Qwen3VLBackend(adapter_path=adapter).deployment_mode == "planner_lora"
    with pytest.raises(ValueError, match="requires a validated LoRA"):
        Qwen3VLBackend(production=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        load_and_validate_adapter_manifest(
            adapter,
            expected_model_id=QWEN3_VL_MODEL_ID,
            expected_revision=QWEN3_VL_REVISION,
        )
