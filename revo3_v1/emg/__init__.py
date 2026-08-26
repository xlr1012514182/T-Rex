"""GNI-derived EMG primitive classification for the Revo3 V1 control plane.

The data generator is deliberately NumPy-only.  Torch-backed model and
training symbols are imported lazily so synthetic-data tooling remains usable
on acquisition machines without PyTorch.
"""

from .synthetic import LABEL_NAMES, SyntheticEMGConfig, generate_synthetic_dataset
from .primitives import EMGPrimitive, MAINLINE_CLASS_LABELS, START_PRIMITIVES
from .streaming import (
    BinaryIntentGate,
    IntentEvent,
    IntentGateConfig,
    IntentObservation,
    MulticlassIntentGate,
)
from .calibration import (
    CALIBRATION_SCHEMA,
    LoadedEMGCalibration,
    calibrate_emg_checkpoint,
    load_emg_calibration,
    validate_calibration_manifest,
)
from .preprocessing import (
    BRAINCO_EDU_8CH_250HZ,
    PREPROCESSING_SCHEMA,
    CausalEMGPreprocessor,
    EmgPreprocessingProfile,
    normalization_sha256,
)
from .migration import (
    GNI_SOURCE_COMMIT,
    GNI_SOURCE_REPOSITORY,
    GNIMigrationReport,
    migrate_gni_encoder,
)

__all__ = [
    "LABEL_NAMES",
    "SyntheticEMGConfig",
    "generate_synthetic_dataset",
    "BinaryIntentGate",
    "EMGPrimitive",
    "IntentEvent",
    "IntentGateConfig",
    "IntentObservation",
    "MAINLINE_CLASS_LABELS",
    "MulticlassIntentGate",
    "START_PRIMITIVES",
    "CALIBRATION_SCHEMA",
    "LoadedEMGCalibration",
    "calibrate_emg_checkpoint",
    "load_emg_calibration",
    "validate_calibration_manifest",
    "BRAINCO_EDU_8CH_250HZ",
    "PREPROCESSING_SCHEMA",
    "CausalEMGPreprocessor",
    "EmgPreprocessingProfile",
    "normalization_sha256",
    "GNI_SOURCE_COMMIT",
    "GNI_SOURCE_REPOSITORY",
    "GNIMigrationReport",
    "migrate_gni_encoder",
]
