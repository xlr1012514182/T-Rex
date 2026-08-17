"""Unified Revo 3 task state machine.

The Task Executive is the sole authority for task lifecycle state.  A CLOSE
event has priority over learned-policy behavior and is latched at start, but it
never bypasses the final hard-safety veto.  REST means no new user command and
does not cancel an active task.  OPEN/RELEASE request deterministic controlled
opening rather than another VLA chunk.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import replace
from typing import Callable, Optional, Tuple

from revo3_v1.planner import PlannerStatus

from .types import (
    CompletionState,
    EmgIntent,
    ExecutiveDecision,
    ExecutiveOutput,
    ExecutivePhase,
    ExecutiveTick,
    MotionDirective,
    PolicyResponseEnvelope,
    SafetyLevel,
    TaskExecutiveConfig,
    TaskLease,
)


class TaskExecutive:
    """Deterministic lifecycle state machine with versioned task leases."""

    def __init__(
        self,
        config: TaskExecutiveConfig = TaskExecutiveConfig(),
        *,
        id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        self.config = config
        self._id_factory = id_factory
        self.phase = ExecutivePhase.IDLE_WAIT
        self.lease: Optional[TaskLease] = None
        self._latched_close_ns: Optional[int] = None
        self._latched_event_id = ""
        self._replans = 0
        self._replan_started_ns: Optional[int] = None
        self._release_started_ns: Optional[int] = None

    @property
    def active(self) -> bool:
        return self.phase in {
            ExecutivePhase.ACTIVE,
            ExecutivePhase.STABLE_HOLD,
            ExecutivePhase.REPLAN_WAIT,
            ExecutivePhase.CONTROLLED_RELEASE,
        }

    @property
    def latched_instruction(self) -> str:
        return self.lease.instruction if self.lease else ""

    def _decision(
        self,
        output: ExecutiveOutput,
        directive: MotionDirective,
        reason: str,
        *,
        clear_policy_cache: bool = False,
        accept_policy_response: bool = False,
        stale_modalities: Tuple[str, ...] = (),
    ) -> ExecutiveDecision:
        return ExecutiveDecision(
            output=output,
            directive=directive,
            phase=self.phase,
            reason=reason,
            lease=self.lease,
            latched_instruction=self.latched_instruction,
            clear_policy_cache=clear_policy_cache,
            accept_policy_response=accept_policy_response,
            stale_modalities=stale_modalities,
        )

    def _clear_task(self) -> None:
        self.lease = None
        self._latched_close_ns = None
        self._latched_event_id = ""
        self._replans = 0
        self._replan_started_ns = None
        self._release_started_ns = None

    def reset_terminal(self) -> None:
        """Acknowledge COMPLETE/ABORT after the actuator reached a safe state."""

        if self.phase not in {ExecutivePhase.COMPLETE, ExecutivePhase.FAULT_LATCHED}:
            raise RuntimeError("reset_terminal is valid only after COMPLETE or ABORT")
        self._clear_task()
        self.phase = ExecutivePhase.IDLE_WAIT

    def _abort(self, reason: str) -> ExecutiveDecision:
        self.phase = ExecutivePhase.FAULT_LATCHED
        return self._decision(
            ExecutiveOutput.ABORT,
            MotionDirective.SAFE_STOP,
            reason,
            clear_policy_cache=True,
        )

    def _ages(self, tick: ExecutiveTick) -> tuple[Tuple[str, ...], Tuple[str, ...]]:
        stale = []
        future = []
        limits = {
            "camera": self.config.camera_ttl_ns,
            "state": self.config.state_ttl_ns,
            "touch": self.config.touch_ttl_ns,
            "policy": self.config.policy_ttl_ns,
        }
        for name, timestamp in tick.timestamps.as_mapping().items():
            if timestamp is None:
                stale.append(name)
                continue
            if timestamp > tick.now_ns + self.config.allowed_future_skew_ns:
                future.append(name)
                continue
            if tick.now_ns - timestamp > limits[name]:
                stale.append(name)
        return tuple(sorted(stale)), tuple(sorted(future))

    def _hard_checks(self, tick: ExecutiveTick) -> Optional[ExecutiveDecision]:
        if tick.safety.level in {SafetyLevel.ABORT, SafetyLevel.EMERGENCY}:
            return self._abort("hard_safety_veto:" + (tick.safety.reason or tick.safety.level.value))
        if tick.emg.timestamp_ns > tick.now_ns + self.config.allowed_future_skew_ns:
            return self._abort("future_emg_event")
        _, future = self._ages(tick)
        if future:
            return self._abort("future_timestamp:" + ",".join(future))
        if self.lease and tick.versions.fingerprint != self.lease.version_fingerprint:
            return self._abort("runtime_version_changed")
        return None

    def _new_lease(self, tick: ExecutiveTick, *, task_version: int) -> TaskLease:
        if tick.planner is None:
            raise ValueError("planner decision required to create lease")
        instruction = tick.planner.instruction
        instruction_hash = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
        return TaskLease(
            task_id=self.lease.task_id if self.lease else self._id_factory(),
            task_version=task_version,
            lease_id=self._id_factory(),
            instruction=instruction,
            instruction_hash=instruction_hash,
            version_fingerprint=tick.versions.fingerprint,
            issued_at_ns=tick.now_ns,
            expires_at_ns=tick.now_ns + self.config.lease_ttl_ns,
        )

    def _renew_lease(self, now_ns: int) -> None:
        if self.lease:
            self.lease = replace(
                self.lease,
                expires_at_ns=now_ns + self.config.lease_ttl_ns,
            )

    def _context_ready(self, tick: ExecutiveTick) -> tuple[bool, str, Tuple[str, ...]]:
        stale, _ = self._ages(tick)
        start_stale = tuple(name for name in stale if name in {"camera", "state", "touch"})
        if start_stale:
            return False, "stale_start_context", start_stale
        if tick.planner is None or tick.visual is None:
            return False, "planner_or_visual_missing", ()
        if tick.planner.timestamp_ns > tick.now_ns + self.config.allowed_future_skew_ns:
            return False, "future_planner_decision", ()
        if tick.now_ns - tick.planner.timestamp_ns > self.config.planner_ttl_ns:
            return False, "stale_planner_decision", ()
        if tick.visual.timestamp_ns > tick.now_ns + self.config.allowed_future_skew_ns:
            return False, "future_visual_gate_result", ()
        if tick.now_ns - tick.visual.timestamp_ns > self.config.planner_ttl_ns:
            return False, "stale_visual_gate_result", ()
        if tick.visual.timestamp_ns != tick.planner.timestamp_ns:
            return False, "planner_visual_timestamp_mismatch", ()
        if tick.planner.status != PlannerStatus.READY:
            return False, "planner_not_ready", ()
        if not tick.visual.ready:
            return False, "visual_not_ready:" + tick.visual.reason, ()
        if tick.visual.task != tick.planner.task:
            return False, "planner_visual_task_mismatch", ()
        return True, "ready", ()

    def _enter_release(self, tick: ExecutiveTick) -> ExecutiveDecision:
        self.phase = ExecutivePhase.CONTROLLED_RELEASE
        self._release_started_ns = tick.now_ns
        if self.lease:
            # A release invalidates every outstanding policy response.
            self.lease = replace(
                self.lease,
                task_version=self.lease.task_version + 1,
                lease_id=self._id_factory(),
                issued_at_ns=tick.now_ns,
                expires_at_ns=tick.now_ns + self.config.lease_ttl_ns,
            )
        return self._decision(
            ExecutiveOutput.CONTINUE,
            MotionDirective.CONTROLLED_OPEN,
            "explicit_release",
            clear_policy_cache=True,
        )

    def step(self, tick: ExecutiveTick) -> ExecutiveDecision:
        """Advance the state machine by one aligned observation tick."""

        hard_result = self._hard_checks(tick)
        if hard_result:
            return hard_result

        if self.phase == ExecutivePhase.COMPLETE:
            return self._decision(
                ExecutiveOutput.COMPLETE,
                MotionDirective.NONE,
                "terminal_complete_waiting_for_ack",
            )
        if self.phase == ExecutivePhase.FAULT_LATCHED:
            return self._decision(
                ExecutiveOutput.ABORT,
                MotionDirective.SAFE_STOP,
                "fault_latched_waiting_for_ack",
            )

        # OPEN and RELEASE are explicit controlled-open requests.  They override
        # a latched CLOSE task but still cannot bypass the hard-safety checks.
        release_event_fresh = (
            tick.now_ns - tick.emg.timestamp_ns <= self.config.emg_start_ttl_ns
        )
        if tick.emg.intent in {EmgIntent.OPEN, EmgIntent.RELEASE} and release_event_fresh:
            if self.phase == ExecutivePhase.CONTROLLED_RELEASE:
                pass
            elif self.active:
                return self._enter_release(tick)
            else:
                # An explicit open before START cancels a pending close latch.
                self._clear_task()
                self.phase = ExecutivePhase.IDLE_WAIT
                return self._decision(
                    ExecutiveOutput.WAIT,
                    MotionDirective.CONTROLLED_OPEN,
                    "open_request_without_active_task",
                )

        if self.phase == ExecutivePhase.IDLE_WAIT:
            if tick.emg.intent != EmgIntent.CLOSE:
                return self._decision(
                    ExecutiveOutput.WAIT,
                    MotionDirective.NONE,
                    "waiting_for_close",
                )
            if tick.emg.confidence < self.config.min_start_confidence:
                return self._decision(
                    ExecutiveOutput.WAIT,
                    MotionDirective.NONE,
                    "close_confidence_too_low",
                )
            if tick.now_ns - tick.emg.timestamp_ns > self.config.emg_start_ttl_ns:
                return self._decision(
                    ExecutiveOutput.WAIT,
                    MotionDirective.NONE,
                    "stale_close_event",
                )
            self._latched_close_ns = tick.now_ns
            self._latched_event_id = tick.emg.event_id
            self.phase = ExecutivePhase.CONTEXT_WAIT

        if self.phase == ExecutivePhase.CONTEXT_WAIT:
            assert self._latched_close_ns is not None
            if tick.now_ns - self._latched_close_ns > self.config.pending_intent_ttl_ns:
                self._clear_task()
                self.phase = ExecutivePhase.IDLE_WAIT
                return self._decision(
                    ExecutiveOutput.WAIT,
                    MotionDirective.NONE,
                    "pending_close_expired",
                )
            if tick.safety.level == SafetyLevel.HOLD:
                return self._decision(
                    ExecutiveOutput.WAIT,
                    MotionDirective.NONE,
                    "safety_not_ready:" + tick.safety.reason,
                )
            ready, reason, stale = self._context_ready(tick)
            if not ready:
                return self._decision(
                    ExecutiveOutput.WAIT,
                    MotionDirective.NONE,
                    reason,
                    stale_modalities=stale,
                )
            self.lease = self._new_lease(tick, task_version=1)
            self.phase = ExecutivePhase.ACTIVE
            return self._decision(
                ExecutiveOutput.START,
                MotionDirective.POLICY,
                "close_and_visual_ready",
                clear_policy_cache=True,
                accept_policy_response=True,
            )

        if self.phase == ExecutivePhase.CONTROLLED_RELEASE:
            release_stale, _ = self._ages(tick)
            if "state" in release_stale:
                return self._abort("stale_state_during_release")
            if tick.safety.level == SafetyLevel.HOLD:
                return self._decision(
                    ExecutiveOutput.HOLD,
                    MotionDirective.HOLD_POSITION,
                    "safety_hold_during_release:" + tick.safety.reason,
                )
            if tick.completion == CompletionState.RELEASED:
                self.phase = ExecutivePhase.COMPLETE
                return self._decision(
                    ExecutiveOutput.COMPLETE,
                    MotionDirective.NONE,
                    "controlled_release_complete",
                    clear_policy_cache=True,
                )
            assert self._release_started_ns is not None
            if tick.now_ns - self._release_started_ns > self.config.release_timeout_ns:
                return self._abort("controlled_release_timeout")
            self._renew_lease(tick.now_ns)
            return self._decision(
                ExecutiveOutput.CONTINUE,
                MotionDirective.CONTROLLED_OPEN,
                "controlled_release_in_progress",
            )

        if tick.completion == CompletionState.FAILED:
            return self._abort("completion_monitor_failed")

        stale, _ = self._ages(tick)
        if "state" in stale:
            return self._abort("stale_state")
        recoverable_stale = tuple(name for name in stale if name in {"camera", "touch", "policy"})
        if tick.safety.level == SafetyLevel.HOLD or recoverable_stale:
            self._renew_lease(tick.now_ns)
            reason = "safety_hold:" + tick.safety.reason if tick.safety.level == SafetyLevel.HOLD else "recoverable_stale"
            return self._decision(
                ExecutiveOutput.HOLD,
                MotionDirective.HOLD_POSITION,
                reason,
                stale_modalities=recoverable_stale,
            )

        if tick.completion in {CompletionState.GRASP_STABLE, CompletionState.TASK_SUCCESS}:
            self.phase = ExecutivePhase.STABLE_HOLD
            self._renew_lease(tick.now_ns)
            return self._decision(
                ExecutiveOutput.HOLD,
                MotionDirective.HOLD_POSITION,
                "stable_grasp",
            )

        if tick.completion == CompletionState.NO_PROGRESS and self.phase != ExecutivePhase.REPLAN_WAIT:
            if self._replans >= self.config.max_replans:
                return self._abort("no_progress_after_replan")
            self._replans += 1
            self._replan_started_ns = tick.now_ns
            self.phase = ExecutivePhase.REPLAN_WAIT
            if self.lease:
                self.lease = replace(
                    self.lease,
                    task_version=self.lease.task_version + 1,
                    lease_id=self._id_factory(),
                    issued_at_ns=tick.now_ns,
                    expires_at_ns=tick.now_ns + self.config.lease_ttl_ns,
                )
            return self._decision(
                ExecutiveOutput.REPLAN,
                MotionDirective.HOLD_POSITION,
                "no_progress",
                clear_policy_cache=True,
            )

        if self.phase == ExecutivePhase.REPLAN_WAIT:
            ready, reason, stale_context = self._context_ready(tick)
            is_new = bool(
                tick.planner
                and self._replan_started_ns is not None
                and tick.planner.timestamp_ns >= self._replan_started_ns
            )
            if not ready or not is_new:
                return self._decision(
                    ExecutiveOutput.HOLD,
                    MotionDirective.HOLD_POSITION,
                    "awaiting_replan:" + (reason if not ready else "old_planner_result"),
                    stale_modalities=stale_context,
                )
            assert self.lease is not None
            self.lease = self._new_lease(tick, task_version=self.lease.task_version)
            self.phase = ExecutivePhase.ACTIVE
            return self._decision(
                ExecutiveOutput.CONTINUE,
                MotionDirective.POLICY,
                "replan_committed",
                clear_policy_cache=True,
                accept_policy_response=True,
            )

        if self.phase == ExecutivePhase.STABLE_HOLD:
            self._renew_lease(tick.now_ns)
            return self._decision(
                ExecutiveOutput.HOLD,
                MotionDirective.HOLD_POSITION,
                "holding_until_explicit_release",
            )

        self._renew_lease(tick.now_ns)
        return self._decision(
            ExecutiveOutput.CONTINUE,
            MotionDirective.POLICY,
            "active_task",
            accept_policy_response=True,
        )

    def validate_policy_response(
        self,
        response: PolicyResponseEnvelope,
        *,
        now_ns: int,
    ) -> tuple[bool, str]:
        """Reject stale/cross-task/cross-version policy chunks before execution."""

        if self.phase != ExecutivePhase.ACTIVE or self.lease is None:
            return False, "policy_not_allowed_in_current_phase"
        lease = self.lease
        expected = (
            (response.task_id, lease.task_id, "task_id_mismatch"),
            (response.task_version, lease.task_version, "task_version_mismatch"),
            (response.lease_id, lease.lease_id, "lease_id_mismatch"),
            (response.instruction_hash, lease.instruction_hash, "instruction_hash_mismatch"),
            (
                response.version_fingerprint,
                lease.version_fingerprint,
                "version_fingerprint_mismatch",
            ),
        )
        for observed, required, reason in expected:
            if observed != required:
                return False, reason
        if now_ns > lease.expires_at_ns:
            return False, "lease_expired"
        if response.observation_timestamp_ns > now_ns + self.config.allowed_future_skew_ns:
            return False, "future_observation_timestamp"
        if now_ns - response.observation_timestamp_ns > self.config.policy_ttl_ns:
            return False, "stale_policy_observation"
        if response.produced_at_ns > now_ns + self.config.allowed_future_skew_ns:
            return False, "future_policy_response"
        if now_ns - response.produced_at_ns > self.config.policy_ttl_ns:
            return False, "stale_policy_response"
        if response.produced_at_ns < response.observation_timestamp_ns:
            return False, "policy_produced_before_observation"
        return True, "accepted"
