"""100 Hz Revo single-writer executor independent of GPU latency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from revo3_v1.executive import MotionDirective, TaskLease
from revo3_v1.policy import AsyncTReXPolicyRunner, TaskKey, TemporalAggregationError
from revo3_v1.tactile import ReflexPhase, TactileFrame, TactileReflexPlugin

from .contracts import RevoState, assert_joint_vector
from .pipeline import RevoCommandPipeline
from .safety import SafetyContext, SafetyResult


@dataclass(frozen=True)
class RevoServoConfig:
    safe_open_q_rad: np.ndarray
    executor_hz: int = 100
    max_tick_jitter_ns: int = 2_000_000
    hardware_mode: bool = False
    calibration_id: str = ""

    def __post_init__(self) -> None:
        safe_open = assert_joint_vector(self.safe_open_q_rad, name="safe_open_q_rad")
        if self.executor_hz != 100:
            raise ValueError("V1 Revo servo is frozen at 100 Hz")
        if self.max_tick_jitter_ns < 0:
            raise ValueError("max_tick_jitter_ns must be non-negative")
        if self.hardware_mode and not self.calibration_id.strip():
            raise ValueError("hardware servo requires a calibrated safe-open artifact")
        object.__setattr__(self, "safe_open_q_rad", safe_open)

    @classmethod
    def demo(cls) -> "RevoServoConfig":
        # Desktop/CI simulation is a wiring smoke, not a 2 ms real-time
        # scheduling claim.  Use a wider host jitter envelope so the new
        # fail-fast hard-fault path is deterministic under non-RT Windows;
        # production still loads its separately calibrated hardware value.
        return cls(
            np.zeros(21, np.float32),
            max_tick_jitter_ns=250_000_000,
            hardware_mode=False,
            calibration_id="simulation",
        )


@dataclass(frozen=True)
class RevoServoResult:
    directive: MotionDirective
    nominal_q_rad: np.ndarray
    residual_q_rad: np.ndarray
    safety: Optional[SafetyResult]
    wrote_command: bool
    reason: str


class RevoServoExecutor:
    """Lift 30 Hz chunks to 100 Hz and route every write through Safety."""

    def __init__(
        self,
        pipeline: RevoCommandPipeline,
        policy: AsyncTReXPolicyRunner,
        cair: TactileReflexPlugin,
        config: RevoServoConfig,
    ) -> None:
        self.pipeline = pipeline
        self.policy = policy
        self.cair = cair
        self.config = config
        self._last_tick_ns: int | None = None
        self._hold_q_rad: np.ndarray | None = None
        self._last_authorized_q_rad: np.ndarray | None = None

    def reset_task(self) -> None:
        self._last_tick_ns = None
        self._hold_q_rad = None
        self._last_authorized_q_rad = None
        self.cair.reset()

    def _pipeline_state(self, state: RevoState | None) -> RevoState | None:
        """Never let aligned caller telemetry stand in for a hardware read."""

        return None if bool(getattr(self.pipeline.backend, "is_hardware", True)) else state

    async def _observed_for_hold(self, state: RevoState | None) -> np.ndarray:
        if state is not None and not bool(
            getattr(self.pipeline.backend, "is_hardware", True)
        ):
            return state.q_rad.copy()
        observed = await self.pipeline.backend.read_state()
        return observed.q_rad.copy()

    async def step(
        self,
        *,
        now_ns: int,
        directive: MotionDirective,
        lease: TaskLease | None,
        action_epoch_ns: int | None,
        tactile: TactileFrame | None,
        reflex_phase: ReflexPhase,
        safety_context: SafetyContext = SafetyContext(),
        emg_requests_close: bool = False,
        state: RevoState | None = None,
        source_chunk_id: str | None = None,
    ) -> RevoServoResult:
        now = int(now_ns)
        if now < 0:
            raise ValueError("now_ns must be non-negative")
        # IDLE/WAIT has no actuator command or active 100 Hz control contract.
        # Keep the cadence anchor current, but do not hard-latch a motor timing
        # fault before motion has ever been authorized.  START calls
        # ``reset_task`` and establishes a fresh active-writer epoch.
        if directive is MotionDirective.NONE:
            self._last_tick_ns = now
            nominal = (
                self._last_authorized_q_rad.copy()
                if self._last_authorized_q_rad is not None
                else (state.q_rad.copy() if state is not None else self.config.safe_open_q_rad.copy())
            )
            return RevoServoResult(
                directive, nominal, np.zeros(21, np.float32), None, False, "no_motion"
            )
        if self._last_tick_ns is not None:
            observed_period = now - self._last_tick_ns
            expected = 1_000_000_000 // self.config.executor_hz
            if observed_period <= 0 or abs(observed_period - expected) > self.config.max_tick_jitter_ns:
                safety = await self.pipeline.abort(
                    "servo_tick_timing_fault", state=self._pipeline_state(state)
                )
                self.policy.reset("servo_tick_timing_fault")
                return RevoServoResult(
                    MotionDirective.SAFE_STOP,
                    safety.q_authorized_rad.copy(),
                    np.zeros(21, np.float32),
                    safety,
                    False,
                    "servo_tick_timing_fault",
                )
        self._last_tick_ns = now

        if directive is MotionDirective.SAFE_STOP:
            safety = await self.pipeline.abort(
                "task_executive_abort", state=self._pipeline_state(state)
            )
            self.policy.reset("task_executive_abort")
            self.cair.reset()
            return RevoServoResult(
                directive,
                safety.q_authorized_rad.copy(),
                np.zeros(21, np.float32),
                safety,
                False,
                "soft_stop_latched_no_auto_open",
            )
        if lease is None:
            safety = await self.pipeline.abort(
                "motion_without_task_lease", state=self._pipeline_state(state)
            )
            self.policy.reset("motion_without_task_lease")
            return RevoServoResult(
                MotionDirective.SAFE_STOP,
                safety.q_authorized_rad.copy(),
                np.zeros(21, np.float32),
                safety,
                False,
                "motion_without_task_lease",
            )

        task_key = TaskKey(
            lease.task_id,
            lease.task_version,
            lease.instruction_hash,
            lease.lease_id,
            lease.version_fingerprint,
        )
        reason = "policy"
        if directive is MotionDirective.POLICY:
            if action_epoch_ns is None:
                raise ValueError("POLICY directive requires action_epoch_ns")
            try:
                nominal = self.policy.interpolated_target_for_tick(
                    executor_timestamp_ns=now,
                    action_epoch_ns=action_epoch_ns,
                    task_key=task_key,
                    now_ns=now,
                )
                self._hold_q_rad = None
            except TemporalAggregationError:
                if self._hold_q_rad is None:
                    self._hold_q_rad = await self._observed_for_hold(state)
                nominal = self._hold_q_rad.copy()
                reason = "policy_chunk_unavailable_hold"
                directive = MotionDirective.HOLD_POSITION
        elif directive is MotionDirective.HOLD_POSITION:
            if self._hold_q_rad is None:
                self._hold_q_rad = await self._observed_for_hold(state)
            nominal = self._hold_q_rad.copy()
            reason = "hold_latched"
        elif directive is MotionDirective.CONTROLLED_OPEN:
            nominal = self.config.safe_open_q_rad.copy()
            self._hold_q_rad = None
            reason = "controlled_open"
        else:  # pragma: no cover - enum exhaustiveness guard
            raise ValueError(f"unsupported directive: {directive}")

        residual = np.zeros(21, np.float32)
        hard_overload = False
        if tactile is not None:
            reflex = self.cair.update(
                tactile, phase=reflex_phase, now_ns=now
            )
            residual = reflex.residual_q_rad
            hard_overload = reflex.hard_overload
        elif reflex_phase in {ReflexPhase.CONTACT_BUILD, ReflexPhase.HOLD}:
            # Missing touch in contact/hold never causes residual decay/opening.
            residual = self.cair.residual_q_rad
            reason += ":touch_missing_cair_frozen"
        else:
            self.cair.reset()

        context = SafetyContext(
            collision_active=safety_context.collision_active,
            tactile_overload=safety_context.tactile_overload or hard_overload,
            emergency_stop=safety_context.emergency_stop,
            fault_latched=safety_context.fault_latched,
            command_lease_valid=(
                safety_context.command_lease_valid and now < lease.expires_at_ns
            ),
            policy_aborted=safety_context.policy_aborted,
            holding_object=safety_context.holding_object,
            reason=safety_context.reason,
        )
        safety = await self.pipeline.execute(
            nominal_q_rad=nominal,
            residual_q_rad=residual,
            task_id=lease.task_id,
            task_version=lease.task_version,
            source_chunk_id=source_chunk_id,
            safety_context=context,
            emg_requests_close=emg_requests_close,
            now_ns=now,
            state=self._pipeline_state(state),
        )
        if safety.clear_policy_cache:
            self.policy.reset("safety_clear_policy_cache")
        wrote = not safety.vetoed
        if wrote:
            self._last_authorized_q_rad = safety.q_authorized_rad.copy()
        return RevoServoResult(
            directive,
            nominal.copy(),
            residual.copy(),
            safety,
            wrote,
            reason if not safety.vetoed else f"{reason}:{safety.reason}",
        )


__all__ = ["RevoServoConfig", "RevoServoExecutor", "RevoServoResult"]
