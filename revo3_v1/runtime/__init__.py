"""Injectable, hardware-neutral V1 online orchestration."""

from .emg_bridge import (
    StreamingEMGEventBridge,
    StreamingEMGSource,
    streaming_result_to_emg_event,
)
from .orchestrator import (
    ClarificationToken,
    OnlineV1Coordinator,
    Runtime30HzResult,
    RuntimeSynchronizedInput,
)
from .factory import (
    ProductionBindings,
    RuntimeAssembly,
    RuntimeAssemblyError,
    SyntheticRuntimeIO,
    build_production_runtime,
    build_simulation_runtime,
    load_control_config,
)
from .service import (
    DoubleRateRuntimeService,
    EMGPacket,
    RuntimeEvidence,
    RuntimeIO,
    RuntimeShutdown,
)

__all__ = [
    "ClarificationToken",
    "OnlineV1Coordinator",
    "Runtime30HzResult",
    "RuntimeSynchronizedInput",
    "StreamingEMGEventBridge",
    "StreamingEMGSource",
    "streaming_result_to_emg_event",
    "DoubleRateRuntimeService",
    "EMGPacket",
    "ProductionBindings",
    "RuntimeAssembly",
    "RuntimeAssemblyError",
    "RuntimeIO",
    "RuntimeEvidence",
    "RuntimeShutdown",
    "SyntheticRuntimeIO",
    "build_production_runtime",
    "build_simulation_runtime",
    "load_control_config",
]
