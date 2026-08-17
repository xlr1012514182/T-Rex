"""Immutable, server-owned identity for a production Revo3 T-Rex model.

The task/version lease identifies *what the controller asked for*.  It does
not prove which checkpoint answered the request.  This module defines the
separate model-serving identity that is calculated from server-local files,
pinned by the controller, and repeated on every response.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SERVER_IDENTITY_SCHEMA = "revo3-trex-server-identity-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_sha256(value: object, *, name: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{name} must be a 64-character SHA-256 digest")
    return text


def _required_text(value: object, *, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


@dataclass(frozen=True)
class TReXServerIdentity:
    """All server/model artifacts that are immutable within a task lease."""

    checkpoint_sha256: str
    model_config_sha256: str
    training_args_sha256: str
    checkpoint_lineage_sha256: str
    normalization_statistics_sha256: str
    normalization_artifact_sha256: str
    checkpoint_family_id: str
    normalization_family_id: str
    tactile_profile_manifest_sha256: str
    capability_manifest_sha256: str
    split_manifest_sha256: str
    tactile_profile: str
    camera_profile: str
    joint_order_hash: str
    training_stage: str
    schema_version: str = SERVER_IDENTITY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != SERVER_IDENTITY_SCHEMA:
            raise ValueError("unsupported T-Rex server identity schema")
        for name in (
            "checkpoint_sha256",
            "model_config_sha256",
            "training_args_sha256",
            "checkpoint_lineage_sha256",
            "normalization_statistics_sha256",
            "normalization_artifact_sha256",
            "tactile_profile_manifest_sha256",
            "capability_manifest_sha256",
            "split_manifest_sha256",
            "joint_order_hash",
        ):
            object.__setattr__(
                self,
                name,
                _required_sha256(getattr(self, name), name=name),
            )
        for name in (
            "checkpoint_family_id",
            "normalization_family_id",
            "tactile_profile",
            "camera_profile",
            "training_stage",
        ):
            object.__setattr__(
                self,
                name,
                _required_text(getattr(self, name), name=name),
            )

    @property
    def identity_sha256(self) -> str:
        encoded = json.dumps(
            self._core_mapping(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _core_mapping(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "checkpoint_sha256": self.checkpoint_sha256,
            "model_config_sha256": self.model_config_sha256,
            "training_args_sha256": self.training_args_sha256,
            "checkpoint_lineage_sha256": self.checkpoint_lineage_sha256,
            "normalization_statistics_sha256": self.normalization_statistics_sha256,
            "normalization_artifact_sha256": self.normalization_artifact_sha256,
            "checkpoint_family_id": self.checkpoint_family_id,
            "normalization_family_id": self.normalization_family_id,
            "tactile_profile_manifest_sha256": self.tactile_profile_manifest_sha256,
            "capability_manifest_sha256": self.capability_manifest_sha256,
            "split_manifest_sha256": self.split_manifest_sha256,
            "tactile_profile": self.tactile_profile,
            "camera_profile": self.camera_profile,
            "joint_order_hash": self.joint_order_hash,
            "training_stage": self.training_stage,
        }

    def as_mapping(self) -> dict[str, str]:
        return {**self._core_mapping(), "identity_sha256": self.identity_sha256}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TReXServerIdentity":
        if not isinstance(value, Mapping):
            raise ValueError("server_identity must be a mapping")
        names = {
            "checkpoint_sha256",
            "model_config_sha256",
            "training_args_sha256",
            "checkpoint_lineage_sha256",
            "normalization_statistics_sha256",
            "normalization_artifact_sha256",
            "checkpoint_family_id",
            "normalization_family_id",
            "tactile_profile_manifest_sha256",
            "capability_manifest_sha256",
            "split_manifest_sha256",
            "tactile_profile",
            "camera_profile",
            "joint_order_hash",
            "training_stage",
            "schema_version",
        }
        missing = sorted(name for name in names if name not in value)
        if missing:
            raise ValueError(f"server_identity is missing fields: {missing}")
        identity = cls(**{name: value[name] for name in names})
        supplied = value.get("identity_sha256")
        if supplied != identity.identity_sha256:
            raise ValueError("server_identity identity_sha256 mismatch")
        return identity


def build_revo_server_identity(
    *,
    checkpoint_path: str | Path,
    normalization_statistics_path: str | Path,
    normalization_artifact_path: str | Path,
    model_config_path: str | Path | None = None,
    camera_profile: str,
    tactile_profile: str,
    joint_order_hash: str,
) -> TReXServerIdentity:
    """Hash a previously validated Revo serving bundle.

    Cross-file semantic validation remains the launcher's/server's job.  This
    helper deliberately derives every byte identity from the files on the
    machine that will serve the model; no value is accepted from a request.
    """

    checkpoint = Path(checkpoint_path).resolve()
    paths = {
        "model": checkpoint / "model.pt",
        "model_config": (
            Path(model_config_path).resolve()
            if model_config_path is not None
            else checkpoint / "config.json"
        ),
        "training_args": checkpoint / "training_args.json",
        "checkpoint_lineage": checkpoint / "checkpoint_lineage.json",
        "normalization_statistics": Path(normalization_statistics_path).resolve(),
        "normalization_artifact": Path(normalization_artifact_path).resolve(),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"server identity files are missing: {missing}")
    training_args = json.loads(paths["training_args"].read_text(encoding="utf-8"))
    if not isinstance(training_args, Mapping):
        raise ValueError("training_args.json must be an object")
    if training_args.get("tactile_profile") != tactile_profile:
        raise ValueError("server tactile profile differs from training_args.json")
    if training_args.get("camera_profile") != camera_profile:
        raise ValueError("server camera profile differs from training_args.json")
    return TReXServerIdentity(
        checkpoint_sha256=_sha256_file(paths["model"]),
        model_config_sha256=_sha256_file(paths["model_config"]),
        training_args_sha256=_sha256_file(paths["training_args"]),
        checkpoint_lineage_sha256=_sha256_file(paths["checkpoint_lineage"]),
        normalization_statistics_sha256=_sha256_file(paths["normalization_statistics"]),
        normalization_artifact_sha256=_sha256_file(paths["normalization_artifact"]),
        checkpoint_family_id=training_args.get("checkpoint_family_id"),
        normalization_family_id=training_args.get("normalization_family_id"),
        tactile_profile_manifest_sha256=training_args.get(
            "tactile_profile_manifest_sha256"
        ),
        capability_manifest_sha256=training_args.get("capability_manifest_sha256"),
        split_manifest_sha256=training_args.get("split_manifest_sha256"),
        tactile_profile=tactile_profile,
        camera_profile=camera_profile,
        joint_order_hash=joint_order_hash,
        training_stage=training_args.get("revo_training_stage"),
    )


__all__ = [
    "SERVER_IDENTITY_SCHEMA",
    "TReXServerIdentity",
    "build_revo_server_identity",
]
