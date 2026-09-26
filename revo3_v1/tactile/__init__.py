"""Single-hand tactile contracts, dense history and bounded reflex plugin."""

from .buffer import DenseTactileBuffer, TactileBufferError, TactileNotReady
from .contracts import (
    F6_DIM,
    FINGER_COUNT,
    FINGER_ORDER,
    HISTORY_LENGTH,
    TactileFrame,
    TactileWindow,
)
from .reflex import (
    ReflexConfig,
    ReflexPhase,
    ReflexResult,
    TactileReflexPlugin,
    demo_closing_synergy,
)
from .synergy import (
    ClosingSynergyArtifact,
    SYNERGY_ARTIFACT_SCHEMA,
    fit_closing_synergy,
)

__all__ = [
    "DenseTactileBuffer",
    "ClosingSynergyArtifact",
    "F6_DIM",
    "FINGER_COUNT",
    "FINGER_ORDER",
    "HISTORY_LENGTH",
    "ReflexConfig",
    "ReflexPhase",
    "ReflexResult",
    "TactileBufferError",
    "TactileFrame",
    "TactileNotReady",
    "TactileReflexPlugin",
    "TactileWindow",
    "SYNERGY_ARTIFACT_SCHEMA",
    "demo_closing_synergy",
    "fit_closing_synergy",
]
