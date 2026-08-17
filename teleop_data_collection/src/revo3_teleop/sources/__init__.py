"""Opt-in sensor source adapters for teleoperation data collection."""

from .brainco_emg import (
    EMG_CHANNELS,
    EMG_SAMPLE_RATE_HZ,
    EMG_SAMPLES_PER_PACKET,
    BrainCoEduEMGParser,
    BrainCoEduEMGSource,
)
from .brainco_glove import BrainCoGloveParser, BrainCoGloveSource
from .camera import (
    CAMERA_CLOCK_DOMAIN,
    OUTPUT_COLOUR_ORDER,
    CameraCalibration,
    CameraClient,
    CameraRead,
    CameraSourceConfig,
    OpenCvCameraClient,
    RgbCameraSource,
    RgbFrameParser,
)
from .manus_ros import (
    DEFAULT_MANUS_TOPICS,
    MANUS_OFFICIAL_MESSAGE_HAS_DEVICE_TIMESTAMP,
    ManusRosFrame,
    ManusRosMessageParser,
    ManusRosSource,
)

__all__ = [
    "BrainCoEduEMGParser",
    "BrainCoEduEMGSource",
    "BrainCoGloveParser",
    "BrainCoGloveSource",
    "CAMERA_CLOCK_DOMAIN",
    "CameraCalibration",
    "CameraClient",
    "CameraRead",
    "CameraSourceConfig",
    "DEFAULT_MANUS_TOPICS",
    "EMG_CHANNELS",
    "EMG_SAMPLE_RATE_HZ",
    "EMG_SAMPLES_PER_PACKET",
    "MANUS_OFFICIAL_MESSAGE_HAS_DEVICE_TIMESTAMP",
    "ManusRosFrame",
    "ManusRosMessageParser",
    "ManusRosSource",
    "OUTPUT_COLOUR_ORDER",
    "OpenCvCameraClient",
    "RgbCameraSource",
    "RgbFrameParser",
]
