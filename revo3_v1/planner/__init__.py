"""RGB + EMG-action instruction planning for the Revo 3 V1 demo."""

from .backends import MockPlannerBackend, PlannerBackend, Qwen3VLBackend
from .planner import AskToClarifyPlanner
from .schema import (
    Ambiguity,
    DEFAULT_INSTRUCTIONS,
    NormalizedBBox,
    PlannerDecision,
    PlannerRequest,
    PlannerStatus,
    SupportedTask,
    TASK_GRASP_PRIMITIVES,
)
from .visual_gate import VisualGate, VisualGateConfig, VisualGateResult
from .training import (
    PlannerDatasetAudit,
    PlannerLoRAConfig,
    build_lora_training_command,
    validate_planner_dataset,
)
from .artifacts import (
    PLANNER_ADAPTER_SCHEMA,
    PROCESSOR_SCHEMA,
    QWEN3_VL_MODEL_ID,
    QWEN3_VL_REVISION,
    load_and_validate_adapter_manifest,
)
from .async_worker import (
    AsyncPlannerWorker,
    PlannerSceneSignature,
    PlannerContextBuffer,
    PlannerContextFrame,
    PlannerWorkItem,
    PlannerWorkResult,
    planner_scene_signature,
)

__all__ = [
    "Ambiguity",
    "DEFAULT_INSTRUCTIONS",
    "AskToClarifyPlanner",
    "MockPlannerBackend",
    "NormalizedBBox",
    "PlannerBackend",
    "PlannerDecision",
    "PlannerDatasetAudit",
    "PlannerLoRAConfig",
    "PlannerRequest",
    "PlannerStatus",
    "Qwen3VLBackend",
    "SupportedTask",
    "TASK_GRASP_PRIMITIVES",
    "VisualGate",
    "VisualGateConfig",
    "VisualGateResult",
    "build_lora_training_command",
    "validate_planner_dataset",
    "PLANNER_ADAPTER_SCHEMA",
    "PROCESSOR_SCHEMA",
    "QWEN3_VL_MODEL_ID",
    "QWEN3_VL_REVISION",
    "load_and_validate_adapter_manifest",
    "AsyncPlannerWorker",
    "PlannerSceneSignature",
    "PlannerContextBuffer",
    "PlannerContextFrame",
    "PlannerWorkItem",
    "PlannerWorkResult",
    "planner_scene_signature",
]
