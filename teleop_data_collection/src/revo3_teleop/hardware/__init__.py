"""Opt-in assembly boundaries for real Revo 3 collection hardware.

Importing this package never scans a bus, opens a camera, starts an EDU
stream, or writes an actuator.  Every concrete client requires a separately
confirmed capability/discovery result before hardware lifecycle methods are
available.
"""

from .brainco_edu import (
    BrainCoEduArmbandConfig,
    BrainCoEduArmbandDiscovery,
    BrainCoEduSdkEMGClient,
)
from .brainco_glove import (
    BrainCoEduGloveConfig,
    BrainCoEduGloveDiscovery,
    BrainCoEduSdkGloveClient,
)
from .brainco_edu_callbacks import (
    BRAINCO_EDU_CALLBACK_NAMESPACE,
    BrainCoEduCallbackNamespaceStatus,
    brainco_edu_callback_namespace_status,
)
from .camera import (
    CameraProbeConfig,
    FisheyeRectificationConfig,
    OpenCvCameraCapability,
    OpenCvFisheyeRectifier,
    ProbedOpenCvCameraClient,
    probe_opencv_camera,
)
from .collection import (
    AuxiliarySourceBinding,
    CollectionControlCycle,
    HARDWARE_COLLECTION_SCHEMA_VERSION,
    HardwareCollectionConfig,
    HardwareCollectionDependencies,
    HardwareCollectionOrchestrator,
    HardwareCollectionReadiness,
    HardwareCollectionResult,
    assess_hardware_collection_config,
    load_hardware_collection_dependencies,
)
from .revo3_sdk import (
    BrainCoRevo3SdkAssembly,
    Revo3BenchApproval,
    Revo3ProbeConfig,
    Revo3ProbeReport,
    Revo3ProbedConnection,
    Revo3TelemetrySource,
    U21VTPressureZoneProjection,
    U21VTSummaryProjection,
)
from .runner import RealSensorRunner, RealSensorRunnerConfig
from .tianji import (
    HARDWARE_SCHEMA_VERSION,
    HardwareConfig,
    HardwareReadinessReport,
    TianjiHardwareAssembly,
    assemble_tianji_hardware,
    assess_hardware_config,
)
from .visiontouch import (
    VISIONTOUCH_DIFF_SHAPE,
    VISIONTOUCH_PROFILE_DIFF_ONLY,
    VISIONTOUCH_PROFILE_FORCE6D,
    VISIONTOUCH_PROFILE_FORCE6D_DIFF,
    VisionTouchForce6DConfig,
    VisionTouchForce6DSource,
    VisionTouchProbeReport,
)

__all__ = [
    "BRAINCO_EDU_CALLBACK_NAMESPACE",
    "BrainCoEduArmbandConfig",
    "BrainCoEduArmbandDiscovery",
    "BrainCoEduGloveConfig",
    "BrainCoEduGloveDiscovery",
    "BrainCoEduSdkEMGClient",
    "BrainCoEduSdkGloveClient",
    "BrainCoEduCallbackNamespaceStatus",
    "BrainCoRevo3SdkAssembly",
    "CameraProbeConfig",
    "AuxiliarySourceBinding",
    "CollectionControlCycle",
    "FisheyeRectificationConfig",
    "HARDWARE_COLLECTION_SCHEMA_VERSION",
    "HARDWARE_SCHEMA_VERSION",
    "HardwareConfig",
    "HardwareReadinessReport",
    "HardwareCollectionConfig",
    "HardwareCollectionDependencies",
    "HardwareCollectionOrchestrator",
    "HardwareCollectionReadiness",
    "HardwareCollectionResult",
    "OpenCvCameraCapability",
    "OpenCvFisheyeRectifier",
    "ProbedOpenCvCameraClient",
    "Revo3BenchApproval",
    "Revo3ProbeConfig",
    "Revo3ProbeReport",
    "Revo3ProbedConnection",
    "Revo3TelemetrySource",
    "RealSensorRunner",
    "RealSensorRunnerConfig",
    "TianjiHardwareAssembly",
    "U21VTPressureZoneProjection",
    "U21VTSummaryProjection",
    "VisionTouchForce6DConfig",
    "VisionTouchForce6DSource",
    "VisionTouchProbeReport",
    "VISIONTOUCH_DIFF_SHAPE",
    "VISIONTOUCH_PROFILE_DIFF_ONLY",
    "VISIONTOUCH_PROFILE_FORCE6D",
    "VISIONTOUCH_PROFILE_FORCE6D_DIFF",
    "probe_opencv_camera",
    "assemble_tianji_hardware",
    "assess_hardware_config",
    "assess_hardware_collection_config",
    "brainco_edu_callback_namespace_status",
    "load_hardware_collection_dependencies",
]
