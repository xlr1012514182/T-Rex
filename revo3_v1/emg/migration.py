"""Strict migration of the official GNI discrete-gesture encoder.

The BrainCo acquisition domain is 8ch/250 Hz rather than the official
16ch/2 kHz domain.  Therefore only the shape-compatible temporal encoder is
inherited.  The rescaled input convolution and five-class projection are
deliberately left at their newly initialized values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence


GNI_SOURCE_REPOSITORY = "https://github.com/facebookresearch/generic-neuromotor-interface"
GNI_SOURCE_COMMIT = "b6bf250e2be5a67b23488104335373cdb87a15c9"
MIGRATION_SCHEMA = "revo3-gni-encoder-migration-v1"

_INHERITED_PREFIXES = (
    "post_conv_layer_norm.",
    "lstm.",
    "post_lstm_layer_norm.",
)
_REINITIALIZED_PREFIXES = ("conv_layer.", "projection.")
_IGNORED_SOURCE_PREFIXES = (
    "val_accuracy.",
    "test_cler.",
    "mask_generator.",
    "loss_fn.",
)
_SOURCE_PREFIXES = (
    "model.network.",
    "module.network.",
    "network.",
    "model.",
    "module.",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_key(key: str) -> str:
    value = str(key)
    for prefix in _SOURCE_PREFIXES:
        if value.startswith(prefix):
            return value[len(prefix) :]
    return value


@dataclass(frozen=True)
class GNIMigrationReport:
    schema_version: str
    source_repository: str
    source_commit: str
    source_checkpoint: str
    source_checkpoint_sha256: str
    inherited_keys: tuple[str, ...]
    reinitialized_keys: tuple[str, ...]
    ignored_allowlisted_keys: tuple[str, ...]
    ignored_non_parameter_keys: tuple[str, ...]

    def to_mapping(self) -> Mapping[str, Any]:
        return asdict(self)


def migrate_gni_encoder(
    model: Any,
    checkpoint: str | Path,
    *,
    source_commit: str,
    map_location: str = "cpu",
) -> GNIMigrationReport:
    """Load only the official GNI temporal encoder with a closed allowlist.

    Any missing or shape-incompatible encoder tensor fails.  Any unexpected
    tensor that does not belong to the known input stem/head allowlist also
    fails.  Lightning bookkeeping values that are not tensors are reported but
    ignored.
    """

    if not source_commit or len(source_commit) < 7:
        raise ValueError("An explicit official GNI source commit is required")
    import torch

    source_path = Path(checkpoint).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"GNI checkpoint does not exist: {source_path}")
    try:
        payload = torch.load(source_path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(source_path, map_location=map_location)
    if not isinstance(payload, Mapping):
        raise ValueError("GNI checkpoint root must be a mapping")
    raw_state = payload.get("state_dict", payload)
    if not isinstance(raw_state, Mapping):
        raise ValueError("GNI checkpoint state_dict must be a mapping")

    canonical: dict[str, Any] = {}
    ignored_non_tensors = []
    for raw_key, value in raw_state.items():
        key = _canonical_key(str(raw_key))
        if not torch.is_tensor(value):
            ignored_non_tensors.append(str(raw_key))
            continue
        if key in canonical:
            raise ValueError(f"GNI checkpoint key collision after prefix mapping: {key}")
        canonical[key] = value

    target_state = model.state_dict()
    inherited = []
    ignored_allowlisted = []
    unknown = []
    for key in canonical:
        if key.startswith(_REINITIALIZED_PREFIXES):
            continue
        elif key.startswith(_IGNORED_SOURCE_PREFIXES):
            ignored_allowlisted.append(key)
        elif not key.startswith(_INHERITED_PREFIXES):
            unknown.append(key)
    if unknown:
        raise ValueError("Unexpected GNI checkpoint tensor keys: " + ",".join(sorted(unknown)))

    required = [key for key in target_state if key.startswith(_INHERITED_PREFIXES)]
    missing = [key for key in required if key not in canonical]
    if missing:
        raise ValueError("GNI checkpoint lacks required temporal encoder keys: " + ",".join(missing))
    for key in required:
        source_value = canonical[key]
        if tuple(source_value.shape) != tuple(target_state[key].shape):
            raise ValueError(
                f"GNI encoder shape mismatch for {key}: "
                f"{tuple(source_value.shape)} != {tuple(target_state[key].shape)}"
            )
        target_state[key] = source_value.to(dtype=target_state[key].dtype)
        inherited.append(key)
    model.load_state_dict(target_state, strict=True)

    # Report every target stem/head tensor as intentionally reinitialized,
    # whether or not the source checkpoint happened to contain that key.
    target_reinitialized = tuple(
        sorted(key for key in target_state if key.startswith(_REINITIALIZED_PREFIXES))
    )
    return GNIMigrationReport(
        schema_version=MIGRATION_SCHEMA,
        source_repository=GNI_SOURCE_REPOSITORY,
        source_commit=str(source_commit),
        source_checkpoint=str(source_path),
        source_checkpoint_sha256=_sha256(source_path),
        inherited_keys=tuple(sorted(inherited)),
        reinitialized_keys=target_reinitialized,
        ignored_allowlisted_keys=tuple(sorted(ignored_allowlisted)),
        ignored_non_parameter_keys=tuple(sorted(ignored_non_tensors)),
    )
