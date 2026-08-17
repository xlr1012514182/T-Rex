"""Binary EMG intent classification for the Revo3 V1 demo.

The data generator is deliberately NumPy-only.  Torch-backed model and
training symbols are imported lazily so synthetic-data tooling remains usable
on acquisition machines without PyTorch.
"""

from .synthetic import LABEL_NAMES, SyntheticEMGConfig, generate_synthetic_dataset
from .streaming import BinaryIntentGate, IntentEvent, IntentGateConfig

__all__ = [
    "LABEL_NAMES",
    "SyntheticEMGConfig",
    "generate_synthetic_dataset",
    "BinaryIntentGate",
    "IntentEvent",
    "IntentGateConfig",
]

