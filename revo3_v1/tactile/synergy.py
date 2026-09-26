"""Versioned, offline calibration of one bounded Revo closing synergy.

The fitter consumes only operator-authorized grasp demonstration deltas.  It
does not infer a direction from executed CAIR corrections and it never writes
hardware.  A held-out cosine gate prevents publishing an unstable direction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from revo3_v1.revo.contracts import JOINT_COUNT, JOINT_ORDER_HASH, assert_joint_vector


SYNERGY_ARTIFACT_SCHEMA = "revo3-cair-synergy-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _stable_hash(value: dict[str, object]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ClosingSynergyArtifact:
    schema_version: str
    joint_order_hash: str
    source_manifest_sha256: str
    sample_count: int
    train_count: int
    heldout_count: int
    closing_synergy: tuple[float, ...]
    heldout_median_cosine: float
    heldout_min_cosine: float
    fit_method: str = "median_of_l2_normalized_authorized_grasp_deltas"
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != SYNERGY_ARTIFACT_SCHEMA:
            raise ValueError("unsupported CAIR synergy artifact schema")
        if self.joint_order_hash != JOINT_ORDER_HASH:
            raise ValueError("CAIR synergy joint order mismatch")
        if _SHA256.fullmatch(self.source_manifest_sha256) is None:
            raise ValueError("source_manifest_sha256 must be 64 lowercase hex")
        vector = assert_joint_vector(self.closing_synergy, name="closing_synergy")
        if not np.isclose(np.max(np.abs(vector)), 1.0, atol=1e-5):
            raise ValueError("closing synergy must be max-absolute normalized")
        if self.sample_count != self.train_count + self.heldout_count:
            raise ValueError("synergy artifact sample counts are inconsistent")
        if self.train_count < 1 or self.heldout_count < 1:
            raise ValueError("synergy artifact requires train and held-out samples")
        unsigned = asdict(self)
        unsigned["artifact_sha256"] = ""
        expected = _stable_hash(unsigned)
        if self.artifact_sha256 and self.artifact_sha256 != expected:
            raise ValueError("CAIR synergy artifact hash mismatch")
        object.__setattr__(self, "closing_synergy", tuple(float(x) for x in vector))
        object.__setattr__(self, "artifact_sha256", expected)

    @property
    def vector(self) -> np.ndarray:
        return np.asarray(self.closing_synergy, dtype=np.float32)

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(asdict(self), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "ClosingSynergyArtifact":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("CAIR synergy artifact must be a JSON object")
        return cls(**value)


def fit_closing_synergy(
    authorized_grasp_deltas_rad: np.ndarray,
    *,
    source_manifest_sha256: str,
    min_samples: int = 20,
    heldout_fraction: float = 0.2,
    min_delta_norm_rad: float = 0.02,
    min_heldout_median_cosine: float = 0.8,
    min_heldout_cosine: float = 0.5,
) -> ClosingSynergyArtifact:
    """Fit and validate a single synergy from authorized teacher deltas.

    Samples must already be in canonical Revo order and represent
    ``authorized_grasp_target - pregrasp_state``.  Ordering is deterministic:
    the final fraction is held out, so the caller must provide a manifest-bound
    stable order rather than relying on a random split hidden in this API.
    """

    if _SHA256.fullmatch(str(source_manifest_sha256)) is None:
        raise ValueError("source_manifest_sha256 must be 64 lowercase hex")
    values = np.asarray(authorized_grasp_deltas_rad, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != JOINT_COUNT:
        raise ValueError(f"authorized_grasp_deltas_rad must be [N,{JOINT_COUNT}]")
    if values.shape[0] < int(min_samples) or int(min_samples) < 4:
        raise ValueError("insufficient authorized grasp samples for synergy fit")
    if not np.isfinite(values).all():
        raise ValueError("authorized grasp deltas contain NaN or infinity")
    if not 0.1 <= float(heldout_fraction) <= 0.5:
        raise ValueError("heldout_fraction must be in [0.1,0.5]")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms < float(min_delta_norm_rad)):
        raise ValueError("authorized grasp delta is too small to define direction")
    normalized = values / norms[:, None]
    heldout_count = max(2, int(np.ceil(values.shape[0] * heldout_fraction)))
    train = normalized[:-heldout_count]
    heldout = normalized[-heldout_count:]
    direction = np.median(train, axis=0)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm < 1e-6:
        raise ValueError("authorized demonstrations do not define one closing direction")
    unit = direction / direction_norm
    train_cosine = train @ unit
    if float(np.median(train_cosine)) < 0:
        unit = -unit
    heldout_cosine = heldout @ unit
    median_cos = float(np.median(heldout_cosine))
    min_cos = float(np.min(heldout_cosine))
    if median_cos < float(min_heldout_median_cosine) or min_cos < float(min_heldout_cosine):
        raise ValueError(
            "held-out authorized grasps do not validate a stable closing synergy"
        )
    max_abs = float(np.max(np.abs(unit)))
    if max_abs < 1e-6:
        raise ValueError("fitted synergy is degenerate")
    synergy = (unit / max_abs).astype(np.float32)
    return ClosingSynergyArtifact(
        schema_version=SYNERGY_ARTIFACT_SCHEMA,
        joint_order_hash=JOINT_ORDER_HASH,
        source_manifest_sha256=str(source_manifest_sha256),
        sample_count=int(values.shape[0]),
        train_count=int(train.shape[0]),
        heldout_count=int(heldout.shape[0]),
        closing_synergy=tuple(float(x) for x in synergy),
        heldout_median_cosine=median_cos,
        heldout_min_cosine=min_cos,
    )


__all__ = [
    "ClosingSynergyArtifact",
    "SYNERGY_ARTIFACT_SCHEMA",
    "fit_closing_synergy",
]
