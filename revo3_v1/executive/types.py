"""Data contracts for the single-authority Revo 3 Task Executive."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Optional, Tuple

from revo3_v1.planner import PlannerDecision, VisualGateResult


class EmgIntent(str, Enum):
    POWER_GRASP = "POWER_GRASP"
    PRECISION_GRASP = "PRECISION_GRASP"
    LATERAL_GRASP = "LATERAL_GRASP"
    RELEASE = "RELEASE"
    REST = "REST"
    UNKNOWN = "UNKNOWN"
    BAD_SIGNAL = "BAD_SIGNAL"

    # Source-compatible names for the first binary mock.  They are aliases,
    # not extra protocol classes.
    CLOSE = "POWER_GRASP"
    OPEN = "RELEASE"

    @property
    def starts_task(self) -> bool:
        return self in {
            EmgIntent.POWER_GRASP,
            EmgIntent.PRECISION_GRASP,
            EmgIntent.LATERAL_GRASP,
        }


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
    ATOMIC_COMMIT = "ATOMIC_COMMIT"
    PRECONTACT_RUN = "PRECONTACT_RUN"
    CONTACT_BUILD = "CONTACT_BUILD"
    ACTIVE = "PRECONTACT_RUN"  # compatibility alias
    STABLE_HOLD = "STABLE_HOLD"
    REPLAN_WAIT = "REPLAN_WAIT"
    CONTROLLED_RELEASE = "CONTROLLED_RELEASE"
    COMPLETE = "COMPLETE"
    FAULT_LATCHED = "FAULT_LATCHED"


class CompletionState(str, Enum):
    IN_PROGRESS = "IN_PROGRESS"
    CONTACT_ESTABLISHED = "CONTACT_ESTABLISHED"
    GRASP_STABLE = "GRASP_STABLE"
    TASK_SUCCESS = "TASK_SUCCESS"
    NO_PROGRESS = "NO_PROGRESS"
    FAILED = "FAILED"
    RELEASED = "RELEASED"


def completion_state_from_status(value: object) -> CompletionState:
    """Map the sole low-level ``revo.CompletionStatus`` to Executive input.

    Kept string-based to avoid making the control-plane types depend on the
    Revo hardware package (and to avoid introducing a second monitor).
    """

    raw = getattr(value, "value", value)
    try:
        return CompletionState(str(raw))
    except ValueError as exc:
        raise ValueError(f"Unknown completion status: {raw}") from exc


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
    margin: float = 1.0
    signal_quality: float = 1.0

    def __post_init__(self) -> None:
        if self.timestamp_ns < 0:
            raise ValueError("EMG timestamp must be non-negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("EMG confidence must be in [0,1]")
        if not 0.0 <= self.margin <= 1.0:
            raise ValueError("EMG margin must be in [0,1]")
        if not 0.0 <= self.signal_quality <= 1.0:
            raise ValueError("EMG signal_quality must be in [0,1]")


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
class TactileProfileReadiness:
    """Explicit profile-specific evidence required before START."""

    profile_kind: str
    profile_hash: str
    ready: bool
    reason: str
    force6d_history_frames: int = 0
    diff_valid_fingers: int = 0
    pressure_present: bool = False
    pressure_valid_mask_present: bool = False
    policy_adapter_ready: bool = False

    @classmethod
    def evaluate(
        cls,
        *,
        profile_kind: str,
        profile_hash: str,
        force6d_history_frames: int = 0,
        diff_valid_fingers: int = 0,
        pressure_present: bool = False,
        pressure_valid_mask_present: bool = False,
        policy_adapter_ready: bool = False,
    ) -> "TactileProfileReadiness":
        kind = str(profile_kind).upper()
        if kind == "A":
            force_ready = force6d_history_frames >= 16
            diff_ready = diff_valid_fingers == 5
            ready = force_ready and diff_ready
            if not force_ready:
                reason = "force6d_history_not_full"
            elif not diff_ready:
                reason = "diff_five_fingers_not_ready"
            else:
                reason = "ready"
        elif kind == "B":
            ready = diff_valid_fingers == 5
            reason = "ready" if ready else "diff_five_fingers_not_ready"
        elif kind == "C":
            sensory_ready = bool(pressure_present and pressure_valid_mask_present)
            ready = bool(sensory_ready and policy_adapter_ready)
            if not sensory_ready:
                reason = "pressure_or_valid_mask_missing"
            elif not policy_adapter_ready:
                reason = "profile_c_policy_adapter_not_validated"
            else:
                reason = "ready"
        else:
            raise ValueError("profile_kind must be A, B, or C")
        if not profile_hash:
            ready, reason = False, "profile_hash_missing"
        return cls(
            profile_kind=kind,
            profile_hash=str(profile_hash),
            ready=ready,
            reason=reason,
            force6d_history_frames=int(force6d_history_frames),
            diff_valid_fingers=int(diff_valid_fingers),
            pressure_present=bool(pressure_present),
            pressure_valid_mask_present=bool(pressure_valid_mask_present),
            policy_adapter_ready=bool(policy_adapter_ready),
        )

    @classmethod
    def not_ready(cls) -> "TactileProfileReadiness":
        return cls("", "", False, "profile_readiness_missing")


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
    primitive: str = ""


@dataclass(frozen=True)
class PolicyResponseEnvelope:
    task_id: str
    task_version: int
    lease_id: str
    instruction_hash: str
    version_fingerprint: str
    observation_timestamp_ns: int
    produced_at_ns: int
    inference_mode: str = "slow_and_fast"


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
    tactile_readiness: TactileProfileReadiness = field(
        default_factory=TactileProfileReadiness.not_ready
    )

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
    pending_intent_ttl_ns: int = 22_000_000_000
    camera_ttl_ns: int = 100_000_000
    state_ttl_ns: int = 50_000_000
    touch_ttl_ns: int = 150_000_000
    policy_ttl_ns: int = 750_000_000
    slow_response_observation_budget_ns: int = 1_500_000_000
    fast_response_observation_budget_ns: int = 500_000_000
    planner_ttl_ns: int = 1_000_000_000
    planner_sla_ns: int = 20_000_000_000
    planner_source_max_age_ns: int = 18_000_000_000
    lease_ttl_ns: int = 3_000_000_000
    release_timeout_ns: int = 3_000_000_000
    allowed_future_skew_ns: int = 5_000_000
    min_start_confidence: float = 0.80
    min_start_margin: float = 0.20
    min_signal_quality: float = 0.80
    max_replans: int = 1
    commit_stability_ns: int = 150_000_000
    touch_stale_abort_ns: int = 500_000_000
    camera_stale_abort_ns: int = 1_000_000_000
    policy_stale_abort_ns: int = 1_000_000_000
    policy_startup_abort_ns: int = 3_000_000_000

    def __post_init__(self) -> None:
        non_negative = (
            self.emg_start_ttl_ns,
            self.pending_intent_ttl_ns,
            self.camera_ttl_ns,
            self.state_ttl_ns,
            self.touch_ttl_ns,
            self.policy_ttl_ns,
            self.slow_response_observation_budget_ns,
            self.fast_response_observation_budget_ns,
            self.planner_ttl_ns,
            self.planner_sla_ns,
            self.planner_source_max_age_ns,
            self.lease_ttl_ns,
            self.release_timeout_ns,
            self.allowed_future_skew_ns,
            self.commit_stability_ns,
            self.touch_stale_abort_ns,
            self.camera_stale_abort_ns,
            self.policy_stale_abort_ns,
            self.policy_startup_abort_ns,
        )
        if any(value < 0 for value in non_negative):
            raise ValueError("Task Executive timeouts must be non-negative")
        if self.slow_response_observation_budget_ns <= 0:
            raise ValueError("slow response-observation budget must be positive")
        if self.planner_sla_ns <= 0 or self.planner_ttl_ns <= 0:
            raise ValueError("planner SLA and result TTL must be positive")
        if not 0 < self.planner_source_max_age_ns <= self.planner_sla_ns:
            raise ValueError("planner source max age must be positive and no greater than SLA")
        if self.pending_intent_ttl_ns < self.planner_sla_ns + self.commit_stability_ns:
            raise ValueError(
                "pending_intent_ttl_ns must cover planner_sla_ns plus atomic commit"
            )
        if self.fast_response_observation_budget_ns <= 0:
            raise ValueError("fast response-observation budget must be positive")
        if (
            self.fast_response_observation_budget_ns
            > self.slow_response_observation_budget_ns
        ):
            raise ValueError("fast policy latency budget cannot exceed slow budget")
        if self.max_replans < 0:
            raise ValueError("max_replans must be non-negative")
        for name, value in (
            ("min_start_confidence", self.min_start_confidence),
            ("min_start_margin", self.min_start_margin),
            ("min_signal_quality", self.min_signal_quality),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
