"""Fail-closed provenance checks for Revo tactile pretrained components."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: str | Path, label: str) -> Mapping[str, Any]:
    resolved = Path(path)
    if not resolved.is_file():
        raise ValueError(f"{label} does not exist: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _validate_common(
    checkpoint: str | Path,
    artifact: str | Path,
    *,
    schema_version: str,
    source_sensor_family: str,
    tactile_profile: str,
    checkpoint_family_id: str,
    normalization_family_id: str,
    capability_manifest_sha256: str,
    split_manifest_sha256: str,
) -> Mapping[str, Any]:
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise ValueError(f"tactile component checkpoint does not exist: {checkpoint_path}")
    value = _load_json(artifact, "tactile component artifact")
    required = {
        "schema_version": schema_version,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "trained_from_scratch": True,
        "source_sensor_family": source_sensor_family,
        "tactile_profile": tactile_profile,
        "checkpoint_family_id": checkpoint_family_id,
        "normalization_family_id": normalization_family_id,
        "capability_manifest_sha256": capability_manifest_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "locked_test_opened": False,
    }
    mismatches = {
        key: {"expected": expected, "observed": value.get(key)}
        for key, expected in required.items()
        if value.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"tactile component provenance mismatch: {mismatches}")
    return value


def validate_revo_vqvae_artifact(
    checkpoint: str | Path,
    artifact: str | Path,
    **expected: str,
) -> Mapping[str, Any]:
    value = _validate_common(
        checkpoint,
        artifact,
        schema_version="revo3-force6d-vqvae-artifact-v1",
        source_sensor_family="revo3_u21vt_force6d",
        **expected,
    )
    frozen = {
        "num_fingers": 5,
        "window": 16,
        "stride": 4,
        "codebook_size": 64,
        "embed_dim": 256,
        "ema_decay": 0.99,
        "commitment_beta": 0.25,
        "granularity": "finger",
        "source_splits": ["midtrain_train"],
        "validation_split": "development",
    }
    mismatches = {
        key: {"expected": expected_value, "observed": value.get(key)}
        for key, expected_value in frozen.items()
        if value.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"Revo VQ-VAE frozen contract mismatch: {mismatches}")
    _validate_episode_lineage(value, "VQ-VAE")
    return value


def _validate_episode_lineage(value: Mapping[str, Any], label: str) -> None:
    for prefix in ("train", "validation"):
        episode_ids = value.get(f"{prefix}_episode_ids")
        if (
            not isinstance(episode_ids, list)
            or not episode_ids
            or any(not isinstance(item, str) or not item for item in episode_ids)
        ):
            raise ValueError(f"Revo {label} artifact requires non-empty {prefix}_episode_ids")
        payload = json.dumps(
            episode_ids, ensure_ascii=True, separators=(",", ":")
        ).encode("utf-8")
        expected_hash = hashlib.sha256(payload).hexdigest()
        if value.get(f"{prefix}_episode_ids_sha256") != expected_hash:
            raise ValueError(f"Revo {label} artifact {prefix} episode-id hash mismatch")


def validate_revo_deform_artifact(
    checkpoint: str | Path,
    artifact: str | Path,
    **expected: str,
) -> Mapping[str, Any]:
    value = _validate_common(
        checkpoint,
        artifact,
        schema_version="revo3-deform-encoder-artifact-v1",
        source_sensor_family="revo3_visiontouch_diff",
        **expected,
    )
    frozen = {
        "input_shape": [5, 1, 240, 240],
        "num_fingers": 5,
        "encoder_state_complete": True,
        "source_splits": ["midtrain_train"],
        "validation_split": "development",
    }
    mismatches = {
        key: {"expected": expected_value, "observed": value.get(key)}
        for key, expected_value in frozen.items()
        if value.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"Revo DIFF encoder frozen contract mismatch: {mismatches}")
    _validate_episode_lineage(value, "DIFF")
    return value
