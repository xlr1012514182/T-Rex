"""Production-contract coordinator; imports no SDK and opens no hardware.

All model/hardware implementations are injected.  Planner and T-Rex inference
run in their dedicated bounded workers; this coordinator only polls them.  It
is deliberately not a second lifecycle authority: every task transition comes
from ``TaskExecutive`` and every motor write comes from ``RevoServoExecutor``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from revo3_v1.emg import EMGPrimitive
from revo3_v1.emg.primitives import normalize_emg_primitive
from revo3_v1.executive import (
    CompletionState,
    EmgEvent,
    ExecutiveDecision,
    ExecutiveOutput,
    ExecutivePhase,
    ExecutiveTick,
    ModalityTimestamps,
    MotionDirective,
    PolicyResponseEnvelope,
    RuntimeVersions,
    SafetyLevel,
    SafetySignal,
    TaskExecutive,
    TactileProfileReadiness,
    completion_state_from_status,
)
from revo3_v1.planner import (
    AsyncPlannerWorker,
    PlannerContextBuffer,
    PlannerContextFrame,
    PlannerDecision,
    PlannerSceneSignature,
    PlannerStatus,
    PlannerWorkItem,
    VisualGate,
    VisualGateResult,
    planner_scene_signature,
)
from revo3_v1.policy import (
    AsyncPolicyPoll,
    AsyncPolicyState,
    AsyncTReXPolicyRunner,
    InferenceMode,
    PolicyObservation,
    TaskKey,
)
from revo3_v1.revo import (
    CompletionMonitor,
    CompletionPhase,
    CompletionResult,
    CompletionStatus,
    RevoState,
    SafetyContext,
)
from revo3_v1.revo.servo import RevoServoExecutor, RevoServoResult
from revo3_v1.tactile import ReflexPhase, TactileFrame, TactileWindow
from revo3_v1.vision import CameraHealthMonitor, CameraHealthResult, DerivedCameraViews

from .emg_bridge import StreamingEMGEventBridge


@dataclass(frozen=True)
class RuntimeSynchronizedInput:
    now_ns: int
    emg: EmgEvent | None
    views: DerivedCameraViews
    state: RevoState
    tactile: TactileFrame
    tactile_window: TactileWindow
    tactile_deform: np.ndarray
    tactile_deform_timestamp_ns: np.ndarray
    tactile_deform_delayed: np.ndarray | None = None
    tactile_deform_delayed_timestamps_ns: np.ndarray | None = None
    safety: SafetySignal = SafetySignal()
    task_postcondition_met: bool = False

    def __post_init__(self) -> None:
        if self.now_ns < 0:
            raise ValueError("now_ns must be non-negative")
        if self.views.capture_timestamp_ns > self.now_ns:
            raise ValueError("camera capture cannot be from the future")
        if self.state.timestamp_ns > self.now_ns:
            raise ValueError("Revo state cannot be from the future")
        if self.tactile.timestamp_ns > self.now_ns:
            raise ValueError("tactile sample cannot be from the future")
        if self.tactile.timestamp_ns != self.tactile_window.newest_timestamp_ns:
            raise ValueError("current tactile timestamp must match native history tail")
        if not np.array_equal(self.tactile.f6, self.tactile_window.current):
            raise ValueError("current tactile value must match native history tail")
        deform = np.asarray(self.tactile_deform)
        deform_ts = np.asarray(self.tactile_deform_timestamp_ns, dtype=np.int64)
        delayed_value = self.tactile_deform_delayed
        delayed_ts_value = self.tactile_deform_delayed_timestamps_ns
        if deform.shape != (5, 240, 240) or deform.dtype != np.uint8:
            raise ValueError("current DIFF must be uint8 with shape (5,240,240)")
        if deform_ts.shape != (5,):
            raise ValueError("current DIFF timestamps must have shape (5,)")
        if (delayed_value is None) != (delayed_ts_value is None):
            raise ValueError("legacy delayed DIFF values/timestamps must be paired")
        if (
            np.any(deform_ts < 0)
            or np.any(deform_ts > self.now_ns)
        ):
            raise ValueError("current DIFF timestamps must be causal and non-negative")
        object.__setattr__(self, "tactile_deform", deform.copy())
        object.__setattr__(self, "tactile_deform_timestamp_ns", deform_ts.copy())
        if delayed_value is not None:
            delayed = np.asarray(delayed_value)
            delayed_ts = np.asarray(delayed_ts_value, dtype=np.int64)
            if delayed.shape != (4, 5, 240, 240) or delayed.dtype != np.uint8:
                raise ValueError("legacy delayed DIFF must be uint8 (4,5,240,240)")
            if delayed_ts.shape != (4, 5):
                raise ValueError("legacy delayed DIFF timestamps must be (4,5)")
            if np.any(delayed_ts < 0) or np.any(delayed_ts > self.now_ns):
                raise ValueError("legacy delayed DIFF timestamps must be causal")
            object.__setattr__(self, "tactile_deform_delayed", delayed.copy())
            object.__setattr__(
                self, "tactile_deform_delayed_timestamps_ns", delayed_ts.copy()
            )


@dataclass(frozen=True)
class Runtime30HzResult:
    executive: ExecutiveDecision
    camera_health: CameraHealthResult
    planner: PlannerDecision | None
    visual: VisualGateResult | None
    completion: CompletionResult | None
    policy: AsyncPolicyPoll | None
    emg: EmgEvent


@dataclass(frozen=True)
class ClarificationToken:
    """Identity required to answer the one currently pending Planner question."""

    event_id: str
    planner_generation: int
    task_version: int
    question: str


@dataclass(frozen=True)
class _PendingClarification:
    answer: str
    token: ClarificationToken
    received_at_ns: int


class OnlineV1Coordinator:
    """Connect the frozen V1 modules without acquiring hardware itself."""

    def __init__(
        self,
        *,
        planner_worker: AsyncPlannerWorker,
        camera_health: CameraHealthMonitor,
        visual_gate: VisualGate,
        executive: TaskExecutive,
        policy_worker: AsyncTReXPolicyRunner,
        completion: CompletionMonitor,
        servo: RevoServoExecutor,
        versions: RuntimeVersions,
        emg_bridge: StreamingEMGEventBridge | None = None,
        tactile_profile: str = "profile_a_force6d_diff",
        planner_result_ttl_ns: int | None = None,
        planner_sla_ns: int | None = None,
        planner_scene_max_distance: float = 0.08,
    ) -> None:
        if tactile_profile != "profile_a_force6d_diff":
            raise RuntimeError(
                "V1 full online runtime currently requires Profile A; Profile B "
                "has no calibrated DIFF-only CompletionMonitor and Profile C has "
                "no validated T-Rex adapter.  Training/inference components remain separate."
            )
        result_ttl = executive.config.planner_ttl_ns if planner_result_ttl_ns is None else int(planner_result_ttl_ns)
        sla = executive.config.planner_sla_ns if planner_sla_ns is None else int(planner_sla_ns)
        if result_ttl <= 0 or sla <= 0:
            raise ValueError("planner result TTL/SLA must be positive")
        if not 0.0 < float(planner_scene_max_distance) <= 1.0:
            raise ValueError("planner_scene_max_distance must be in (0,1]")
        self.planner_worker = planner_worker
        self.camera_health = camera_health
        self.visual_gate = visual_gate
        self.executive = executive
        self.policy_worker = policy_worker
        self.completion = completion
        self.servo = servo
        self.versions = versions
        self.emg_bridge = emg_bridge
        self.tactile_profile = tactile_profile
        self.planner_result_ttl_ns = result_ttl
        self.planner_sla_ns = sla
        self.planner_scene_max_distance = float(planner_scene_max_distance)
        self.context = PlannerContextBuffer()
        self._planner_generation = 0
        self._latched_event_id = ""
        self._latched_primitive: EMGPrimitive | None = None
        self._planner_decision: PlannerDecision | None = None
        self._visual_result: VisualGateResult | None = None
        self._planner_scene_signature: PlannerSceneSignature | None = None
        self._last_policy_timestamp_ns: int | None = None
        self._last_nominal_q_rad: np.ndarray | None = None
        self._action_epoch_ns: int | None = None
        self._awaiting_first_chunk = False
        self._last_rebased_slow_start_step: int | None = None
        self._last_executive: ExecutiveDecision | None = None
        self._pending_clarification: _PendingClarification | None = None

    def ingest_emg_packet(
        self,
        samples: np.ndarray,
        sample_timestamps_ns: Sequence[int] | np.ndarray,
        *,
        signal_quality: float = 1.0,
    ) -> tuple[EmgEvent, ...]:
        """Run every due streaming EMG inference without touching T-Rex."""

        if self.emg_bridge is None:
            raise RuntimeError("no StreamingEMGEventBridge was injected")
        return self.emg_bridge.push_many(
            samples,
            sample_timestamps_ns,
            signal_quality=signal_quality,
        )

    def _resolve_emg(self, tick: RuntimeSynchronizedInput) -> RuntimeSynchronizedInput:
        if tick.emg is not None:
            return tick
        if self.emg_bridge is None:
            raise RuntimeError(
                "RuntimeSynchronizedInput.emg=None requires an injected streaming EMG bridge"
            )
        return replace(tick, emg=self.emg_bridge.event_for_tick(now_ns=tick.now_ns))

    def _submit_planner(
        self,
        tick: RuntimeSynchronizedInput,
        *,
        clarification_answer: str = "",
    ) -> bool:
        if self._latched_primitive is None or not self.context.ready:
            return False
        request = self.context.build_request(
            primitive=self._latched_primitive,
            now_ns=tick.now_ns,
            metadata={"camera_schema": "revo3_full_center_v1"},
            clarification_answer=clarification_answer,
        )
        return self.planner_worker.submit(
            PlannerWorkItem(
                generation=self._planner_generation,
                event_id=self._latched_event_id,
                primitive=self._latched_primitive,
                task_version=(0 if self.executive.lease is None else self.executive.lease.task_version),
                request=request,
                submitted_at_ns=tick.now_ns,
            )
        )

    @property
    def clarification_token(self) -> ClarificationToken | None:
        decision = self._planner_decision
        if decision is None or decision.status is not PlannerStatus.ASK_CLARIFY:
            return None
        if self._latched_primitive is None or not self._latched_event_id:
            return None
        return ClarificationToken(
            event_id=self._latched_event_id,
            planner_generation=self._planner_generation,
            task_version=(0 if self.executive.lease is None else self.executive.lease.task_version),
            question=decision.ambiguity.question,
        )

    def submit_clarification(
        self,
        answer: str,
        *,
        event_id: str,
        planner_generation: int,
        task_version: int,
        received_at_ns: int,
    ) -> None:
        """Queue a bounded answer for the next fresh 30 Hz camera context.

        The answer has no motion authority.  It is accepted only for the exact
        currently pending ASK_CLARIFY identity and the Executive remains
        WAIT/NONE until a later READY result passes all ordinary start gates.
        """

        normalized = " ".join(str(answer).split())
        if not normalized:
            raise ValueError("clarification answer must not be empty")
        if len(normalized) > 256:
            raise ValueError("clarification answer exceeds 256 characters")
        now = int(received_at_ns)
        if now < 0:
            raise ValueError("clarification timestamp must be non-negative")
        token = self.clarification_token
        if token is None:
            raise RuntimeError("no clarification is currently pending")
        supplied = ClarificationToken(
            str(event_id), int(planner_generation), int(task_version), token.question
        )
        if supplied != token:
            raise RuntimeError("stale_or_wrong_clarification_identity")
        latched_at = self.executive.pending_intent_started_ns
        if (
            latched_at is None
            or now - latched_at > self.executive.config.pending_intent_ttl_ns
        ):
            raise RuntimeError("clarification_arrived_after_pending_intent_ttl")
        if self._pending_clarification is not None:
            raise RuntimeError("a clarification answer is already queued")
        self._pending_clarification = _PendingClarification(normalized, token, now)

    def _reset_prestart_attempt(self) -> None:
        """Atomically invalidate a cancelled/expired intent before START.

        This differs from terminal acknowledgement: no task lease or actuator
        lifecycle existed yet.  Resetting the streaming gate inactive allows a
        genuinely new StartIntentEvent without forcing a synthetic RELEASE.
        """

        self._planner_generation += 1
        self._latched_event_id = ""
        self._latched_primitive = None
        self._planner_decision = None
        self._visual_result = None
        self._planner_scene_signature = None
        self._pending_clarification = None
        self.visual_gate.reset()
        self._last_policy_timestamp_ns = None
        self._action_epoch_ns = None
        self._awaiting_first_chunk = False
        self._last_rebased_slow_start_step = None
        if self.emg_bridge is not None:
            self.emg_bridge.reset(active=False)

    def _poll_planner(self, tick: RuntimeSynchronizedInput) -> None:
        if self._latched_primitive is None:
            return
        result = self.planner_worker.poll(
            expected_generation=self._planner_generation,
            expected_event_id=self._latched_event_id,
            expected_primitive=self._latched_primitive,
            expected_task_version=(0 if self.executive.lease is None else self.executive.lease.task_version),
            now_ns=tick.now_ns,
            result_ttl_ns=self.planner_result_ttl_ns,
            planner_sla_ns=self.planner_sla_ns,
            allowed_future_skew_ns=self.executive.config.allowed_future_skew_ns,
            current_scene_signature=planner_scene_signature(
                tick.views.full, tick.views.fixed_center
            ),
            max_scene_distance=self.planner_scene_max_distance,
        )
        if result is None:
            return
        if result.error or result.decision is None:
            return
        self._planner_decision = result.decision
        self._planner_scene_signature = result.scene_signature
        self._visual_result = self.visual_gate.update(
            result.decision, now_ns=tick.now_ns
        )

    def _planner_has_pending(self) -> bool:
        # Production worker exposes ``pending`` (busy or an unread result).
        # Structural test doubles written before that property remain usable.
        return bool(getattr(self.planner_worker, "pending", self.planner_worker.busy))

    @staticmethod
    def _completion_phase(phase: ExecutivePhase) -> CompletionPhase:
        if phase is ExecutivePhase.CONTACT_BUILD:
            return CompletionPhase.CONTACT_BUILD
        if phase is ExecutivePhase.STABLE_HOLD:
            return CompletionPhase.STABLE_HOLD
        if phase is ExecutivePhase.CONTROLLED_RELEASE:
            return CompletionPhase.CONTROLLED_RELEASE
        if phase is ExecutivePhase.FAULT_LATCHED:
            return CompletionPhase.ABORTED
        return CompletionPhase.PRECONTACT_RUN

    @staticmethod
    def _reflex_phase(phase: ExecutivePhase) -> ReflexPhase:
        if phase is ExecutivePhase.CONTACT_BUILD:
            return ReflexPhase.CONTACT_BUILD
        if phase is ExecutivePhase.STABLE_HOLD:
            return ReflexPhase.HOLD
        if phase is ExecutivePhase.CONTROLLED_RELEASE:
            return ReflexPhase.RELEASE
        return ReflexPhase.PRECONTACT

    def _completion_result(self, tick: RuntimeSynchronizedInput) -> CompletionResult | None:
        if not self.executive.active:
            return None
        nominal = (
            tick.state.q_rad
            if self._last_nominal_q_rad is None
            else self._last_nominal_q_rad
        )
        return self.completion.update(
            phase=self._completion_phase(self.executive.phase),
            state=tick.state,
            tactile=tick.tactile,
            nominal_q_rad=nominal,
            now_ns=tick.now_ns,
            safety_failed=tick.safety.level in {SafetyLevel.ABORT, SafetyLevel.EMERGENCY},
            task_postcondition_met=tick.task_postcondition_met,
        )

    def _policy_observation(
        self, tick: RuntimeSynchronizedInput, decision: ExecutiveDecision
    ) -> PolicyObservation:
        assert decision.lease is not None
        lease = decision.lease
        key = TaskKey(
            lease.task_id,
            lease.task_version,
            lease.instruction_hash,
            lease.lease_id,
            lease.version_fingerprint,
        )
        return PolicyObservation(
            timestamp_ns=tick.now_ns,
            state_timestamp_ns=tick.state.timestamp_ns,
            rgb_timestamp_ns=tick.views.capture_timestamp_ns,
            tactile_timestamp_ns=tick.tactile.timestamp_ns,
            q_rad=tick.state.q_rad,
            tactile_f6=tick.tactile.f6,
            tactile_history_f6=tick.tactile_window.f6,
            tactile_history_timestamps_ns=tick.tactile_window.timestamps_ns,
            tactile_history_sequences=tick.tactile_window.sequences,
            tactile_deform=tick.tactile_deform,
            tactile_deform_timestamp_ns=tick.tactile_deform_timestamp_ns,
            tactile_profile=self.tactile_profile,
            instruction=lease.instruction,
            task_key=key,
            images=tick.views.policy_images,
            lease_expires_at_ns=lease.expires_at_ns,
        )

    def step_30hz(
        self,
        tick: RuntimeSynchronizedInput,
        *,
        camera_health_result: CameraHealthResult | None = None,
    ) -> Runtime30HzResult:
        tick = self._resolve_emg(tick)
        assert tick.emg is not None
        health = (
            self.camera_health.evaluate(tick.views, now_ns=tick.now_ns)
            if camera_health_result is None
            else camera_health_result
        )
        if health.healthy:
            self.context.append(
                PlannerContextFrame(
                    tick.views.sequence,
                    tick.views.capture_timestamp_ns,
                    tick.views.full,
                    tick.views.fixed_center,
                ),
                now_ns=tick.now_ns,
            )

        # Consume the one prior-generation outcome before any submit decision.
        # This closes the fast-worker race where busy=False while a result is
        # already queued but not yet observed by the coordinator.
        self._poll_planner(tick)

        # A fresh completion timestamp cannot make a visibly obsolete source
        # frame actionable.  This cheap deterministic signature uses only the
        # already-rectified full/center pixels; it adds no detector/tracker.
        if self._planner_decision is not None:
            current_signature = planner_scene_signature(
                tick.views.full, tick.views.fixed_center
            )
            source_expired = (
                tick.now_ns - self._planner_decision.timestamp_ns
                > self.executive.config.planner_source_max_age_ns
            )
            scene_changed = (
                self._planner_scene_signature is None
                or self._planner_scene_signature.distance(current_signature)
                > self.planner_scene_max_distance
            )
            if source_expired or scene_changed:
                self._planner_generation += 1
                self._planner_decision = None
                self._visual_result = None
                self._planner_scene_signature = None
                self.visual_gate.reset()

        if self._pending_clarification is not None:
            pending = self._pending_clarification
            current = self.clarification_token
            if current != pending.token:
                self._pending_clarification = None
            elif not self._planner_has_pending() and self.context.ready:
                # Rebase to a new generation and the latest causal RGB.  Old
                # in-flight/result work is thereby rejected by identity.
                self._planner_generation += 1
                self._planner_decision = None
                self._visual_result = None
                self._planner_scene_signature = None
                self.visual_gate.reset()
                submitted = self._submit_planner(
                    tick, clarification_answer=pending.answer
                )
                if submitted:
                    self._pending_clarification = None

        if tick.emg.intent.starts_task and self.executive.phase is ExecutivePhase.IDLE_WAIT:
            primitive = normalize_emg_primitive(tick.emg.intent.value)
            assert primitive is not None
            self._latched_primitive = primitive
            self._latched_event_id = tick.emg.event_id or f"emg-{tick.emg.timestamp_ns}"
            self._planner_generation += 1
            self._planner_decision = None
            self._visual_result = None
            self._planner_scene_signature = None
            self.visual_gate.reset()
            self._submit_planner(tick)
        elif (
            self.executive.phase in {ExecutivePhase.CONTEXT_WAIT, ExecutivePhase.REPLAN_WAIT}
            and self._planner_decision is None
            and not self._planner_has_pending()
        ):
            self._submit_planner(tick)

        policy_poll: AsyncPolicyPoll | None = None
        policy_safety = tick.safety
        policy_phase = self.executive.phase in {
            ExecutivePhase.PRECONTACT_RUN,
            ExecutivePhase.CONTACT_BUILD,
        }
        accepted_slow_start_step: int | None = None
        if self.executive.lease is not None:
            lease = self.executive.lease
            expected_key = TaskKey(
                lease.task_id,
                lease.task_version,
                lease.instruction_hash,
                lease.lease_id,
                lease.version_fingerprint,
            )
            policy_poll = self.policy_worker.poll(
                now_ns=tick.now_ns, expected_task_key=expected_key
            )
            if policy_poll.state is AsyncPolicyState.READY and policy_poll.chunk is not None:
                if policy_phase:
                    envelope = PolicyResponseEnvelope(
                        task_id=policy_poll.chunk.task_key.task_id,
                        task_version=policy_poll.chunk.task_key.task_version,
                        lease_id=policy_poll.chunk.task_key.lease_id,
                        instruction_hash=policy_poll.chunk.task_key.instruction_hash,
                        version_fingerprint=policy_poll.chunk.task_key.version_fingerprint,
                        observation_timestamp_ns=policy_poll.chunk.observation_timestamp_ns,
                        produced_at_ns=policy_poll.chunk.generated_ns,
                        inference_mode=policy_poll.chunk.mode.value,
                    )
                    accepted, reason = self.executive.validate_policy_response(
                        envelope, now_ns=tick.now_ns
                    )
                    if accepted:
                        self._last_policy_timestamp_ns = tick.now_ns
                        if policy_poll.chunk.mode in {
                            InferenceMode.SLOW,
                            InferenceMode.SLOW_AND_FAST,
                        }:
                            accepted_slow_start_step = policy_poll.chunk.start_step
                            if (
                                self._last_rebased_slow_start_step is not None
                                and accepted_slow_start_step
                                < self._last_rebased_slow_start_step
                            ):
                                policy_safety = SafetySignal(
                                    SafetyLevel.ABORT,
                                    "slow_chunk_start_step_regressed",
                                )
                                accepted_slow_start_step = None
                    else:
                        if reason in {
                            "slow_response_latency_budget_exceeded",
                            "fast_response_latency_budget_exceeded",
                        }:
                            # The reply is same-task but too late to execute.
                            # Drop its already-aggregated chunk, HOLD, and let
                            # the Executive's bounded stale timer decide the
                            # eventual abort instead of misclassifying latency
                            # as a cross-task safety violation.
                            self.policy_worker.reset(reason)
                            policy_safety = SafetySignal(SafetyLevel.HOLD, reason)
                        else:
                            policy_safety = SafetySignal(SafetyLevel.ABORT, reason)
                else:
                    # A late result after STABLE_HOLD/RELEASE/terminal state is
                    # obsolete work, not a new lifecycle fault.
                    self.policy_worker.reset("policy_result_outside_policy_phase")
            elif policy_poll.state is AsyncPolicyState.ABORT and policy_phase:
                policy_safety = SafetySignal(SafetyLevel.ABORT, policy_poll.reason)
            elif policy_phase and policy_poll.reason in {
                "policy_request_timeout",
                "policy_timeout_waiting_for_worker_unwind",
                "policy_backend_error",
            }:
                policy_safety = SafetySignal(SafetyLevel.HOLD, policy_poll.reason)
        if not health.healthy and policy_safety.level is SafetyLevel.SAFE:
            policy_safety = SafetySignal(SafetyLevel.HOLD, "camera_health:" + health.reason)

        completion = self._completion_result(tick)
        completion_state = (
            CompletionState.IN_PROGRESS
            if completion is None
            else completion_state_from_status(completion.status)
        )
        executive_tick = ExecutiveTick(
            now_ns=tick.now_ns,
            emg=tick.emg,
            visual=self._visual_result,
            planner=self._planner_decision,
            timestamps=ModalityTimestamps(
                camera_ns=tick.views.capture_timestamp_ns,
                state_ns=tick.state.timestamp_ns,
                touch_ns=tick.tactile.timestamp_ns,
                policy_ns=self._last_policy_timestamp_ns,
            ),
            versions=self.versions,
            safety=policy_safety,
            completion=completion_state,
            tactile_readiness=TactileProfileReadiness.evaluate(
                profile_kind="A",
                profile_hash=self.versions.tactile_profile_hash,
                force6d_history_frames=int(
                    np.sum(np.all(tick.tactile_window.valid_fingers, axis=1))
                ),
                diff_valid_fingers=int(
                    np.sum(
                        (tick.tactile_deform_timestamp_ns <= tick.now_ns)
                        & (
                            tick.now_ns - tick.tactile_deform_timestamp_ns
                            <= self.executive.config.touch_ttl_ns
                        )
                    )
                ),
            ),
        )
        previous_phase = self.executive.phase
        decision = self.executive.step(executive_tick)
        self._last_executive = decision
        if (
            previous_phase in {
                ExecutivePhase.CONTEXT_WAIT,
                ExecutivePhase.ATOMIC_COMMIT,
            }
            and decision.phase is ExecutivePhase.IDLE_WAIT
            and decision.reason in {
                "pending_intent_expired",
                "release_without_active_task_ignored",
            }
        ):
            self._reset_prestart_attempt()
        if decision.clear_policy_cache:
            self.policy_worker.reset(decision.reason)
            self._last_policy_timestamp_ns = None
        if decision.output is ExecutiveOutput.REPLAN:
            self._action_epoch_ns = None
            self._awaiting_first_chunk = False
            self._last_rebased_slow_start_step = None
            self._planner_generation += 1
            self._planner_decision = None
            self._visual_result = None
            self._planner_scene_signature = None
            self.visual_gate.reset()
        entered_policy_epoch = decision.output is ExecutiveOutput.START or (
            previous_phase is ExecutivePhase.REPLAN_WAIT
            and decision.phase is ExecutivePhase.PRECONTACT_RUN
            and decision.directive is MotionDirective.POLICY
        )
        if entered_policy_epoch:
            # The execution clock starts only when the first accepted chunk
            # arrives.  Otherwise model latency would skip its early actions.
            self._action_epoch_ns = None
            self._awaiting_first_chunk = True
            self._last_rebased_slow_start_step = None
            self.completion.reset()
            self.servo.reset_task()
        elif (
            decision.phase is ExecutivePhase.STABLE_HOLD
            and previous_phase is not ExecutivePhase.STABLE_HOLD
        ):
            self.policy_worker.reset("stable_hold")
            self._last_policy_timestamp_ns = None
            self._awaiting_first_chunk = False
            self._last_rebased_slow_start_step = None
        if (
            accepted_slow_start_step is not None
            and decision.directive is MotionDirective.POLICY
            and accepted_slow_start_step != self._last_rebased_slow_start_step
        ):
            # Every new slow chunk defines a fresh safe execution epoch.  GPU
            # latency must not skip its early actions: map the acceptance tick
            # to that chunk's start_step (chunk[0]), preserving global indices
            # for subsequent 4/8/12 fast refinements.
            self._action_epoch_ns = (
                tick.now_ns
                - accepted_slow_start_step
                * self.policy_worker.schedule.action_period_ns
            )
            self._awaiting_first_chunk = False
            self._last_rebased_slow_start_step = accepted_slow_start_step

        if (
            decision.directive is MotionDirective.POLICY
            and decision.lease is not None
        ):
            observation = self._policy_observation(tick, decision)
            if self._action_epoch_ns is None:
                global_step = 0
            else:
                global_step = max(
                    0,
                    (tick.now_ns - self._action_epoch_ns)
                    // self.policy_worker.schedule.action_period_ns,
                )
            # Do not immediately replace a just-accepted first chunk with a
            # duplicate offset-zero request on the same control tick.
            if policy_poll is None or policy_poll.state is not AsyncPolicyState.READY:
                submitted = self.policy_worker.submit_if_due(
                    global_step=int(global_step),
                    observation=observation,
                    now_ns=tick.now_ns,
                )
                if policy_poll is None or policy_poll.state in {
                    AsyncPolicyState.IDLE,
                    AsyncPolicyState.PENDING,
                }:
                    policy_poll = submitted
        return Runtime30HzResult(
            decision, health, self._planner_decision, self._visual_result,
            completion, policy_poll, tick.emg
        )

    async def step_servo(
        self,
        tick: RuntimeSynchronizedInput,
        *,
        source_chunk_id: str | None = None,
    ) -> RevoServoResult:
        if self._last_executive is None:
            raise RuntimeError("step_30hz must establish a TaskExecutive decision first")
        directive = self._last_executive.directive
        if directive is MotionDirective.POLICY and self._action_epoch_ns is None:
            directive = MotionDirective.HOLD_POSITION
        result = await self.servo.step(
            now_ns=tick.now_ns,
            directive=directive,
            lease=self._last_executive.lease,
            action_epoch_ns=self._action_epoch_ns,
            tactile=tick.tactile,
            reflex_phase=self._reflex_phase(self._last_executive.phase),
            safety_context=SafetyContext(
                command_lease_valid=(
                    self._last_executive.lease is not None
                    and tick.now_ns < self._last_executive.lease.expires_at_ns
                ),
                holding_object=self._last_executive.phase is ExecutivePhase.STABLE_HOLD,
            ),
            emg_requests_close=bool(
                tick.emg is not None and tick.emg.intent.starts_task
            ),
            state=tick.state,
            source_chunk_id=source_chunk_id,
        )
        self._last_nominal_q_rad = result.nominal_q_rad.copy()
        return result

    def latch_servo_fault(self, reason: str) -> ExecutiveDecision:
        """Expose a fatal writer result as the TaskExecutive ABORT terminal."""

        decision = self.executive.abort_runtime_fault(reason)
        self._last_executive = decision
        self.policy_worker.reset("servo_runtime_fault")
        self._last_policy_timestamp_ns = None
        self._action_epoch_ns = None
        self._awaiting_first_chunk = False
        self._last_rebased_slow_start_step = None
        return decision

    def reset_terminal(
        self,
        *,
        safe_state_confirmed: bool,
        operator_reset_confirmed: bool = False,
    ) -> None:
        """Acknowledge COMPLETE/ABORT and atomically clear task-local state.

        This is deliberately explicit: an ABORT is never converted into an
        automatic open/reset.  The caller may invoke this method only after
        the actuator and supervising application have established the safe
        state required by :meth:`TaskExecutive.reset_terminal`.
        """

        if not safe_state_confirmed:
            raise ValueError("terminal reset requires safe_state_confirmed")
        phase = self.executive.phase
        pipeline = self.servo.pipeline
        if phase is ExecutivePhase.FAULT_LATCHED:
            if not operator_reset_confirmed:
                raise ValueError("ABORT reset requires operator_reset_confirmed")
            if not pipeline.soft_stop_confirmed:
                raise RuntimeError("ABORT reset requires a confirmed SoftStop")
            pipeline.clear_fault_latch_after_operator_reset()
        elif phase is ExecutivePhase.COMPLETE:
            if pipeline.hard_fault_latched:
                raise RuntimeError("COMPLETE cannot reset a latched hardware fault")
        else:
            raise RuntimeError("terminal reset requires COMPLETE or ABORT")

        self.executive.reset_terminal()
        self._planner_generation += 1
        self._latched_event_id = ""
        self._latched_primitive = None
        self._planner_decision = None
        self._visual_result = None
        self._planner_scene_signature = None
        self._pending_clarification = None
        self.visual_gate.reset()
        self.context.clear()
        self.policy_worker.reset("terminal_acknowledged")
        self.completion.reset()
        self.servo.reset_task()
        self._last_policy_timestamp_ns = None
        self._last_nominal_q_rad = None
        self._action_epoch_ns = None
        self._awaiting_first_chunk = False
        self._last_rebased_slow_start_step = None
        self._last_executive = None
        if self.emg_bridge is not None:
            self.emg_bridge.reset(active=False)

    def close(
        self,
        *,
        planner_timeout_s: float = 1.0,
        policy_timeout_s: float | None = None,
    ) -> tuple[bool, bool]:
        """Boundedly join both model owners and report actual thread exit.

        ``busy`` is not shutdown evidence: a request can finish while its
        worker remains alive, and an uninterruptible Qwen call can outlive a
        close request.  The Planner worker therefore returns its real thread
        state.  Policy gets at least its request/transport timeout budget so a
        normal bounded GPU/ZMQ response is not mislabeled unclean.
        """

        planner_clean = self.planner_worker.close(timeout_s=planner_timeout_s)
        policy_budget = (
            self.policy_worker.recommended_close_timeout_s
            if policy_timeout_s is None
            else float(policy_timeout_s)
        )
        policy_clean = self.policy_worker.close(timeout_s=policy_budget)
        return planner_clean, policy_clean


__all__ = [
    "ClarificationToken",
    "OnlineV1Coordinator",
    "Runtime30HzResult",
    "RuntimeSynchronizedInput",
]
