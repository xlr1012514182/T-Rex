"""Data contracts for the single-authority Revo 3 Task Executive."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Optional, Tuple

from revo3_v1.planner import PlannerDecision, VisualGateResult


class EmgIntent(str, Enum):
    CLOSE = "CLOSE"
    OPEN = "OPEN"
    RELEASE = "RELEASE"
    REST = "REST"
    UNKNOWN = "UNKNOWN"


class ExecutiveOutput(str, Enum):
    WAIT = "WAIT"
    START = "START"
    CONTINUE = "CONTINUE"
    HOLD = "HOLD"
    REPLAN = "REPLAN"
    COMPLETE = "COMPLETE"
    ABORT = "ABORT"


class MotionDirective(str, Enum):
    NONE = "NONE"
    POLICY = "POLICY"
    HOLD_POSITION = "HOLD_POSITION"
    CONTROLLED_OPEN = "CONTROLLED_OPEN"
    SAFE_STOP = "SAFE_STOP"


class ExecutivePhase(str, Enum):
    IDLE_WAIT = "IDLE_WAIT"
    CONTEXT_WAIT = "CONTEXT_WAIT"
    ACTIVE = "ACTIVE"
    STABLE_HOLD = "STABLE_HOLD"
    REPLAN_WAIT = "REPLAN_WAIT"
    CONTROLLED_RELEASE = "CONTROLLED_RELEASE"
    COMPLETE = "COMPLETE"
    FAULT_LATCHED = "FAULT_LATCHED"


class CompletionState(str, Enum):
    IN_PROGRESS = "IN_PROGRESS"
    GRASP_STABLE = "GRASP_STABLE"
    TASK_SUCCESS = "TASK_SUCCESS"
    NO_PROGRESS = "NO_PROGRESS"
    FAILED = "FAILED"
    RELEASED = "RELEASED"


class SafetyLevel(str, Enum):
    SAFE = "SAFE"
    HOLD = "HOLD"
    ABORT = "ABORT"
    EMERGENCY = "EMERGENCY"


@dataclass(frozen=True)
class EmgEvent:
    intent: EmgIntent
    timestamp_ns: int
    confidence: float = 1.0
    event_id: str = ""

    def __post_init__(self) -> None:
        if self.timestamp_ns < 0:
            raise ValueError("EMG timestamp must be non-negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("EMG confidence must be in [0,1]")


@dataclass(frozen=True)
class SafetySignal:
    level: SafetyLevel = SafetyLevel.SAFE
    reason: str = ""


@dataclass(frozen=True)
class RuntimeVersions:
    """Immutable identities that must not change inside a task lease."""

    schema_version: str
    planner_revision: str
    policy_revision: str
    hardware_manifest_hash: str
    joint_order_hash: str
    tactile_profile_hash: str

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "schema_version": self.schema_version,
                "planner_revision": self.planner_revision,
                "policy_revision": self.policy_revision,
                "hardware_manifest_hash": self.hardware_manifest_hash,
                "joint_order_hash": self.joint_order_hash,
                "tactile_profile_hash": self.tactile_profile_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ModalityTimestamps:
    camera_ns: int
    state_ns: int
    touch_ns: int
    policy_ns: Optional[int] = None

    def as_mapping(self) -> Mapping[str, Optional[int]]:
        return {
            "camera": self.camera_ns,
            "state": self.state_ns,
            "touch": self.touch_ns,
            "policy": self.policy_ns,
        }


@dataclass(frozen=True)
class TaskLease:
    task_id: str
    task_version: int
    lease_id: str
    instruction: str
    instruction_hash: str
    version_fingerprint: str
    issued_at_ns: int
    expires_at_ns: int


@dataclass(frozen=True)
class PolicyResponseEnvelope:
    task_id: str
    task_version: int
    lease_id: str
    instruction_hash: str
    version_fingerprint: str
    observation_timestamp_ns: int
    produced_at_ns: int


@dataclass(frozen=True)
class ExecutiveTick:
    now_ns: int
    emg: EmgEvent
    visual: Optional[VisualGateResult]
    planner: Optional[PlannerDecision]
    timestamps: ModalityTimestamps
    versions: RuntimeVersions
    safety: SafetySignal = SafetySignal()
    completion: CompletionState = CompletionState.IN_PROGRESS

    def __post_init__(self) -> None:
        if self.now_ns < 0:
            raise ValueError("now_ns must be non-negative")


@dataclass(frozen=True)
class ExecutiveDecision:
    output: ExecutiveOutput
    directive: MotionDirective
    phase: ExecutivePhase
    reason: str
    lease: Optional[TaskLease]
    latched_instruction: str = ""
    clear_policy_cache: bool = False
    accept_policy_response: bool = False
    stale_modalities: Tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskExecutiveConfig:
    emg_start_ttl_ns: int = 3_000_000_000
    pending_intent_ttl_ns: int = 3_000_000_000
    camera_ttl_ns: int = 100_000_000
    state_ttl_ns: int = 50_000_000
    touch_ttl_ns: int = 150_000_000
    policy_ttl_ns: int = 750_000_000
    planner_ttl_ns: int = 1_000_000_000
    lease_ttl_ns: int = 1_000_000_000
    release_timeout_ns: int = 3_000_000_000
    allowed_future_skew_ns: int = 5_000_000
    min_start_confidence: float = 0.80
    max_replans: int = 1

    def __post_init__(self) -> None:
        non_negative = (
            self.emg_start_ttl_ns,
            self.pending_intent_ttl_ns,
            self.camera_ttl_ns,
            self.state_ttl_ns,
            self.touch_ttl_ns,
            self.policy_ttl_ns,
            self.planner_ttl_ns,
            self.lease_ttl_ns,
            self.release_timeout_ns,
            self.allowed_future_skew_ns,
        )
        if any(value < 0 for value in non_negative):
            raise ValueError("Task Executive timeouts must be non-negative")
        if self.max_replans < 0:
            raise ValueError("max_replans must be non-negative")

