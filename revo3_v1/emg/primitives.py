"""Canonical EMG intent vocabulary shared by the V1 control plane.

Only the three grasp primitives may start a task.  ``RELEASE`` is a distinct
edge-triggered command; ``REST`` means no new command and ``UNKNOWN`` /
``BAD_SIGNAL`` are fail-closed observations.  None of these values belongs in
the T-Rex observation contract.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional, Tuple


class EMGPrimitive(str, Enum):
    POWER_GRASP = "POWER_GRASP"
    PRECISION_GRASP = "PRECISION_GRASP"
    LATERAL_GRASP = "LATERAL_GRASP"
    RELEASE = "RELEASE"
    REST = "REST"
    UNKNOWN = "UNKNOWN"
    BAD_SIGNAL = "BAD_SIGNAL"

    @property
    def starts_task(self) -> bool:
        return self in START_PRIMITIVES


START_PRIMITIVES: Tuple[EMGPrimitive, ...] = (
    EMGPrimitive.POWER_GRASP,
    EMGPrimitive.PRECISION_GRASP,
    EMGPrimitive.LATERAL_GRASP,
)

# These are the labels learned by the mainline GNI-derived head.  UNKNOWN and
# BAD_SIGNAL are produced by confidence/margin and acquisition-quality gates.
MAINLINE_CLASS_LABELS: Tuple[str, ...] = (
    EMGPrimitive.POWER_GRASP.value,
    EMGPrimitive.PRECISION_GRASP.value,
    EMGPrimitive.LATERAL_GRASP.value,
    EMGPrimitive.RELEASE.value,
    EMGPrimitive.REST.value,
)


_ALIASES = {
    "CLOSE": EMGPrimitive.POWER_GRASP,
    "GRASP": EMGPrimitive.POWER_GRASP,
    "POWER": EMGPrimitive.POWER_GRASP,
    "PRECISION": EMGPrimitive.PRECISION_GRASP,
    "PINCH": EMGPrimitive.PRECISION_GRASP,
    "LATERAL": EMGPrimitive.LATERAL_GRASP,
    "KEY_GRASP": EMGPrimitive.LATERAL_GRASP,
    "OPEN": EMGPrimitive.RELEASE,
}


def normalize_emg_primitive(value: Any) -> Optional[EMGPrimitive]:
    """Normalize a protocol value without silently inventing a new class."""

    if isinstance(value, EMGPrimitive):
        return value
    if value is None:
        return None
    normalized = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    try:
        return EMGPrimitive(normalized)
    except ValueError:
        return _ALIASES.get(normalized)
