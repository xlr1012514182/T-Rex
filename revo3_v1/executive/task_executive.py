"""Unified Revo 3 task state machine.

The Task Executive is the sole authority for task lifecycle state.  A grasp
primitive is latched at start but never bypasses the final hard-safety veto.
REST/BAD_SIGNAL mean no new command and cannot cancel an active task.  RELEASE
requests deterministic controlled opening rather than another VLA chunk.
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
        self._latched_event_timestamp_ns: Optional[int] = None
        self._latched_event_id = ""
        self._latched_primitive = ""
        self._replans = 0
        self._replan_started_ns: Optional[int] = None
        self._release_started_ns: Optional[int] = None
        self._commit_started_ns: Optional[int] = None
        self._commit_signature: Optional[tuple[object, ...]] = None
        self._commit_timestamps: Optional[tuple[int, ...]] = None
        self._stale_since_ns: dict[str, int] = {}
        self._policy_seen_for_lease = False

    @property
    def active(self) -> bool:
        return self.phase in {
            ExecutivePhase.ACTIVE,
            ExecutivePhase.CONTACT_BUILD,
            ExecutivePhase.STABLE_HOLD,
            ExecutivePhase.REPLAN_WAIT,
            ExecutivePhase.CONTROLLED_RELEASE,
        }

    @property
    def latched_instruction(self) -> str:
        return self.lease.instruction if self.lease else ""

    @property
    def pending_intent_started_ns(self) -> Optional[int]:
        return self._latched_close_ns

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
        self._latched_event_timestamp_ns = None
        self._latched_event_id = ""
        self._latched_primitive = ""
        self._replans = 0
        self._replan_started_ns = None
        self._release_started_ns = None
        self._stale_since_ns.clear()
        self._policy_seen_for_lease = False
        self._clear_commit_candidate()

    def _clear_commit_candidate(self) -> None:
        self._commit_started_ns = None
        self._commit_signature = None
        self._commit_timestamps = None

    def _candidate_signature(self, tick: ExecutiveTick) -> tuple[object, ...]:
        assert tick.planner is not None and tick.visual is not None
        instruction_hash = hashlib.sha256(tick.planner.instruction.encode("utf-8")).hexdigest()
        readiness = tick.tactile_readiness
        return (
            self._latched_primitive,
            tick.planner.primitive.value if tick.planner.primitive else "",
            instruction_hash,
            tick.planner.task.value if tick.planner.task else "",
            tick.planner.status.value,
            round(tick.planner.confidence, 6),
            tick.planner.compatible,
            tick.planner.center_ready,
            tick.visual.task.value if tick.visual.task else "",
            tick.visual.ready,
            tick.visual.reason,
            round(tick.visual.area, 6),
            round(tick.visual.center_distance, 6),
            tick.versions.fingerprint,
            tick.safety.level.value,
            tick.safety.reason,
            readiness.profile_kind,
            readiness.profile_hash,
            readiness.ready,
            min(readiness.force6d_history_frames, 16),
            readiness.diff_valid_fingers,
            readiness.pressure_present,
            readiness.pressure_valid_mask_present,
            readiness.policy_adapter_ready,
        )

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

    def abort_runtime_fault(self, reason: str) -> ExecutiveDecision:
        """Latch a lower-layer fatal fault through the sole lifecycle authority."""

        text = str(reason).strip()
        if not text:
            raise ValueError("runtime fault reason must be non-empty")
        return self._abort("runtime_fault:" + text)

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
        if (
            self.active
            and self.lease is not None
            and tick.now_ns >= self.lease.expires_at_ns
        ):
            # A missed control heartbeat cannot revive an expired motion
            # authority merely by reaching a later renewal branch.  Release
            # also fails to SoftStop here; it is never allowed to auto-open
            # under an expired lease.
            return self._abort("active_task_lease_expired")
        if self.lease and tick.versions.fingerprint != self.lease.version_fingerprint:
            return self._abort("runtime_version_changed")
        return None

    def _new_lease(self, tick: ExecutiveTick, *, task_version: int) -> TaskLease:
        if tick.planner is None:
            raise ValueError("planner decision required to create lease")
        if tick.planner.primitive is None or tick.planner.primitive.value != self._latched_primitive:
            raise ValueError("planner primitive does not match latched EMG intent")
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
            primitive=self._latched_primitive,
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
        if self._latched_event_timestamp_ns is None:
            return False, "start_event_not_latched", ()
        if tick.planner.timestamp_ns < self._latched_event_timestamp_ns:
            return False, "planner_precedes_latched_start_event", ()
        if tick.visual.timestamp_ns < self._latched_event_timestamp_ns:
            return False, "visual_precedes_latched_start_event", ()
        if tick.planner.timestamp_ns > tick.now_ns + self.config.allowed_future_skew_ns:
            return False, "future_planner_decision", ()
        if tick.now_ns - tick.planner.timestamp_ns > self.config.planner_source_max_age_ns:
            return False, "planner_source_expired", ()
        if tick.planner.produced_at_ns > tick.now_ns + self.config.allowed_future_skew_ns:
            return False, "future_planner_result", ()
        if tick.now_ns - tick.planner.produced_at_ns > self.config.planner_ttl_ns:
            return False, "stale_planner_result", ()
        if tick.visual.timestamp_ns > tick.now_ns + self.config.allowed_future_skew_ns:
            return False, "future_visual_gate_result", ()
        if tick.visual.produced_at_ns > tick.now_ns + self.config.allowed_future_skew_ns:
            return False, "future_visual_gate_result_produced_at", ()
        if tick.now_ns - tick.visual.produced_at_ns > self.config.planner_ttl_ns:
            return False, "stale_visual_gate_result", ()
        if tick.visual.timestamp_ns != tick.planner.timestamp_ns:
            return False, "planner_visual_timestamp_mismatch", ()
        if tick.visual.produced_at_ns != tick.planner.produced_at_ns:
            return False, "planner_visual_produced_timestamp_mismatch", ()
        if tick.planner.status != PlannerStatus.READY:
            return False, "planner_not_ready", ()
        if not tick.visual.ready:
            return False, "visual_not_ready:" + tick.visual.reason, ()
        if tick.visual.task != tick.planner.task:
            return False, "planner_visual_task_mismatch", ()
        if tick.planner.primitive is None or tick.planner.primitive.value != self._latched_primitive:
            return False, "planner_primitive_mismatch", ()
        readiness = tick.tactile_readiness
        if not readiness.ready:
            return False, "tactile_profile_not_ready:" + readiness.reason, ()
        if readiness.profile_hash != tick.versions.tactile_profile_hash:
            return False, "tactile_profile_readiness_hash_mismatch", ()
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

    def _recoverable_stale_decision(
        self,
        tick: ExecutiveTick,
        stale_modalities: Tuple[str, ...],
    ) -> Optional[ExecutiveDecision]:
        """HOLD first, then fail closed after a bounded stale interval."""

        limits = {
            "touch": self.config.touch_stale_abort_ns,
            "camera": self.config.camera_stale_abort_ns,
            "policy": self.config.policy_stale_abort_ns,
        }
        present = set(stale_modalities)
        for name in tuple(self._stale_since_ns):
            if name not in present:
                del self._stale_since_ns[name]
        for name in stale_modalities:
            if name == "policy" and not self._policy_seen_for_lease and self.lease is not None:
                stale_since = self.lease.issued_at_ns
                timeout = self.config.policy_startup_abort_ns
            else:
                self._stale_since_ns.setdefault(name, tick.now_ns)
                stale_since = self._stale_since_ns[name]
                timeout = limits[name]
            if tick.now_ns - stale_since >= timeout:
                return self._abort(f"{name}_stale_timeout")
        return None

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

        # RELEASE is an explicit controlled-open request.  It overrides a
        # latched grasp task but still cannot bypass hard-safety checks.
        release_event_fresh = (
            tick.now_ns - tick.emg.timestamp_ns <= self.config.emg_start_ttl_ns
        )
        if tick.emg.intent == EmgIntent.RELEASE and release_event_fresh:
            if self.phase == ExecutivePhase.CONTROLLED_RELEASE:
                pass
            elif self.active:
                return self._enter_release(tick)
            else:
                # RELEASE without an active task may cancel a pending intent,
                # but V1 does not authorize an idle actuator-open command.
                self._clear_task()
                self.phase = ExecutivePhase.IDLE_WAIT
                return self._decision(
                    ExecutiveOutput.WAIT,
                    MotionDirective.NONE,
                    "release_without_active_task_ignored",
                )

        if self.phase == ExecutivePhase.IDLE_WAIT:
            if not tick.emg.intent.starts_task:
                return self._decision(
                    ExecutiveOutput.WAIT,
                    MotionDirective.NONE,
                    "waiting_for_start_primitive",
                )
            if tick.emg.confidence < self.config.min_start_confidence:
                return self._decision(
                    ExecutiveOutput.WAIT,
                    MotionDirective.NONE,
                    "start_confidence_too_low",
                )
            if tick.emg.margin < self.config.min_start_margin:
                return self._decision(
                    ExecutiveOutput.WAIT,
                    MotionDirective.NONE,
                    "start_margin_too_low",
                )
            if tick.emg.signal_quality < self.config.min_signal_quality:
                return self._decision(
                    ExecutiveOutput.WAIT,
                    MotionDirective.NONE,
                    "start_signal_quality_too_low",
                )
            if tick.now_ns - tick.emg.timestamp_ns > self.config.emg_start_ttl_ns:
                return self._decision(
                    ExecutiveOutput.WAIT,
                    MotionDirective.NONE,
                    "stale_start_event",
                )
            self._latched_close_ns = tick.now_ns
            self._latched_event_timestamp_ns = tick.emg.timestamp_ns
            self._latched_event_id = tick.emg.event_id
            self._latched_primitive = tick.emg.intent.value
            self.phase = ExecutivePhase.CONTEXT_WAIT

        if self.phase in {ExecutivePhase.CONTEXT_WAIT, ExecutivePhase.ATOMIC_COMMIT}:
            assert self._latched_close_ns is not None
            if tick.now_ns - self._latched_close_ns > self.config.pending_intent_ttl_ns:
                self._clear_task()
                self.phase = ExecutivePhase.IDLE_WAIT
                return self._decision(
                    ExecutiveOutput.WAIT,
                    MotionDirective.NONE,
                    "pending_intent_expired",
                )
            if tick.safety.level == SafetyLevel.HOLD:
                self._clear_commit_candidate()
                self.phase = ExecutivePhase.CONTEXT_WAIT
                return self._decision(
                    ExecutiveOutput.WAIT,
                    MotionDirective.NONE,
                    "safety_not_ready:" + tick.safety.reason,
                )
            ready, reason, stale = self._context_ready(tick)
            if not ready:
                self._clear_commit_candidate()
                self.phase = ExecutivePhase.CONTEXT_WAIT
                return self._decision(
                    ExecutiveOutput.WAIT,
                    MotionDirective.NONE,
                    reason,
                    stale_modalities=stale,
                )
            signature = self._candidate_signature(tick)
            current_timestamps = (
                tick.timestamps.camera_ns,
                tick.timestamps.state_ns,
                tick.timestamps.touch_ns,
                tick.planner.timestamp_ns,
                int(tick.planner.produced_at_ns),
                tick.visual.timestamp_ns,
                tick.visual.produced_at_ns,
            )
            if self._commit_signature is None:
                self._commit_signature = signature
                self._commit_timestamps = current_timestamps
                self._commit_started_ns = tick.now_ns
                self.phase = ExecutivePhase.ATOMIC_COMMIT
            elif signature != self._commit_signature:
                self._clear_commit_candidate()
                self._commit_signature = signature
                self._commit_timestamps = current_timestamps
                self._commit_started_ns = tick.now_ns
                self.phase = ExecutivePhase.ATOMIC_COMMIT
                return self._decision(
                    ExecutiveOutput.WAIT,
                    MotionDirective.NONE,
                    "atomic_commit_candidate_changed",
                )
            assert self._commit_timestamps is not None and self._commit_started_ns is not None
            if any(current < frozen for current, frozen in zip(current_timestamps, self._commit_timestamps)):
                self._clear_commit_candidate()
                self.phase = ExecutivePhase.CONTEXT_WAIT
                return self._decision(
                    ExecutiveOutput.WAIT,
                    MotionDirective.NONE,
                    "atomic_commit_timestamp_regressed",
                )
            if tick.now_ns - self._commit_started_ns < self.config.commit_stability_ns:
                return self._decision(
                    ExecutiveOutput.WAIT,
                    MotionDirective.NONE,
                    "atomic_commit_pending",
                )
            self.lease = self._new_lease(tick, task_version=1)
            self._policy_seen_for_lease = bool(
                tick.timestamps.policy_ns is not None
                and tick.now_ns - tick.timestamps.policy_ns <= self.config.policy_ttl_ns
            )
            self._clear_commit_candidate()
            self.phase = ExecutivePhase.PRECONTACT_RUN
            return self._decision(
                ExecutiveOutput.START,
                MotionDirective.POLICY,
                "intent_and_visual_ready",
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

        if tick.completion == CompletionState.CONTACT_ESTABLISHED:
            self.phase = ExecutivePhase.CONTACT_BUILD

        stale, _ = self._ages(tick)
        if "state" in stale:
            return self._abort("stale_state")
        recoverable_stale = tuple(name for name in stale if name in {"camera", "touch", "policy"})
        if "policy" not in stale and tick.timestamps.policy_ns is not None:
            self._policy_seen_for_lease = True
        stale_escalation = self._recoverable_stale_decision(tick, recoverable_stale)
        if stale_escalation is not None:
            return stale_escalation
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

        if (
            tick.completion == CompletionState.NO_PROGRESS
            and self.phase == ExecutivePhase.PRECONTACT_RUN
        ):
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

        if (
            tick.completion == CompletionState.NO_PROGRESS
            and self.phase == ExecutivePhase.CONTACT_BUILD
        ):
            # Replanning a grounded target after contact can command a
            # discontinuous grasp.  Freeze actuation and require operator/safety
            # acknowledgement; never auto-open here.
            return self._abort("no_progress_after_contact")

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
            self._policy_seen_for_lease = False
            self.phase = ExecutivePhase.PRECONTACT_RUN
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

        if self.phase not in {
            ExecutivePhase.PRECONTACT_RUN,
            ExecutivePhase.CONTACT_BUILD,
        } or self.lease is None:
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
        if response.produced_at_ns > now_ns + self.config.allowed_future_skew_ns:
            return False, "future_policy_response"
        if now_ns - response.produced_at_ns > self.config.policy_ttl_ns:
            return False, "stale_policy_response"
        if response.produced_at_ns < response.observation_timestamp_ns:
            return False, "policy_produced_before_observation"
        mode = str(response.inference_mode)
        if mode in {"slow", "slow_and_fast"}:
            latency_budget_ns = self.config.slow_response_observation_budget_ns
            latency_reason = "slow_response_latency_budget_exceeded"
        elif mode == "fast":
            latency_budget_ns = self.config.fast_response_observation_budget_ns
            latency_reason = "fast_response_latency_budget_exceeded"
        else:
            return False, "unknown_policy_inference_mode"
        if (
            response.produced_at_ns - response.observation_timestamp_ns
            > latency_budget_ns
        ):
            return False, latency_reason
        return True, "accepted"
