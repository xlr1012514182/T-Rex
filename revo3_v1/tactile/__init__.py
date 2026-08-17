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

__all__ = [
    "DenseTactileBuffer",
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
    "demo_closing_synergy",
]
