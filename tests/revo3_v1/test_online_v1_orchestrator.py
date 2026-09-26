from __future__ import annotations

import asyncio
from dataclasses import replace
import threading
import time

import numpy as np
import pytest

from revo3_v1.emg import EMGPrimitive
from revo3_v1.executive import (
    EmgEvent,
    EmgIntent,
    ExecutiveOutput,
    MotionDirective,
    RuntimeVersions,
    SafetyLevel,
    SafetySignal,
    TaskExecutive,
    TaskExecutiveConfig,
)
from revo3_v1.planner import (
    AskToClarifyPlanner,
    AsyncPlannerWorker,
    MockPlannerBackend,
    PlannerStatus,
    SupportedTask,
    TASK_GRASP_PRIMITIVES,
    VisualGate,
)
from revo3_v1.policy import AsyncPolicyState, AsyncTReXPolicyRunner, MockTReXBackend, TReXRevoPolicyAdapter
from revo3_v1.revo import (
    CompletionConfig,
    CompletionMonitor,
    CompletionResult,
    CompletionStatus,
    MockRevoBackend,
    RevoCommandPipeline,
    RevoState,
    SafetyEnvelope,
    SafetySupervisor,
)
from revo3_v1.revo.servo import RevoServoConfig, RevoServoExecutor
from revo3_v1.runtime import (
    OnlineV1Coordinator,
    RuntimeSynchronizedInput,
    StreamingEMGEventBridge,
)
from revo3_v1.tactile import ReflexConfig, TactileFrame, TactileReflexPlugin, TactileWindow
from revo3_v1.vision import (
    CameraHealthConfig,
    CameraHealthMonitor,
    SingleCameraViewConfig,
    SingleCameraViewDeriver,
)


CALIBRATION = "unit-test-camera-calibration"


def versions() -> RuntimeVersions:
    return RuntimeVersions("v1", "planner", "policy", "hardware", "joints", "touch")


def build(
    task: SupportedTask,
    *,
    policy_backend=None,
    completion=None,
    emg_bridge=None,
    planner_worker=None,
    executive_config=None,
):
    planner = planner_worker or AsyncPlannerWorker(
        AskToClarifyPlanner(MockPlannerBackend(default_task=task))
    )
    policy = AsyncTReXPolicyRunner(
        TReXRevoPolicyAdapter(policy_backend or MockTReXBackend())
    )
    backend = MockRevoBackend()
    pipeline = RevoCommandPipeline(backend, SafetySupervisor(SafetyEnvelope.demo()))
    servo = RevoServoExecutor(
        pipeline,
        policy,
        TactileReflexPlugin(ReflexConfig(enabled=False)),
        RevoServoConfig.demo(),
    )
    coordinator = OnlineV1Coordinator(
        planner_worker=planner,
        camera_health=CameraHealthMonitor(CameraHealthConfig(CALIBRATION)),
        visual_gate=VisualGate(),
        executive=TaskExecutive(
            executive_config or TaskExecutiveConfig(commit_stability_ns=0),
            id_factory=iter(f"id-{i}" for i in range(100)).__next__,
        ),
        policy_worker=policy,
        completion=completion or CompletionMonitor(CompletionConfig.demo()),
        servo=servo,
        versions=versions(),
        emg_bridge=emg_bridge,
    )
    return coordinator, backend


def synchronized(now: int, sequence: int, intent: EmgIntent) -> RuntimeSynchronizedInput:
    yy, xx = np.indices((120, 160))
    rgb = np.stack(
        ((xx + sequence) % 255, (yy * 2 + sequence) % 255, (xx + yy) % 255),
        axis=-1,
    ).astype(np.uint8)
    views = SingleCameraViewDeriver(SingleCameraViewConfig(CALIBRATION)).derive(
        rgb,
        capture_timestamp_ns=now,
        sequence=sequence,
        calibration_hash=CALIBRATION,
    )
    history_ts = np.arange(now - 15_000_000, now + 1, 1_000_000, dtype=np.int64)
    history = np.zeros((16, 5, 6), np.float32)
    tactile = TactileFrame(now, history[-1], sequence)
    window = TactileWindow(
        history,
        history_ts,
        np.arange(sequence * 16, sequence * 16 + 16, dtype=np.int64),
        np.ones((16, 5), bool),
    )
    delayed_ts = np.asarray(
        [[now - 12_000_000] * 5, [now - 8_000_000] * 5,
         [now - 4_000_000] * 5, [now] * 5],
        dtype=np.int64,
    )
    return RuntimeSynchronizedInput(
        now_ns=now,
        emg=EmgEvent(intent, now, confidence=0.99, margin=0.9, signal_quality=0.99, event_id=f"event-{sequence}"),
        views=views,
        state=RevoState(now, np.zeros(21, np.float32), sequence=sequence, temperature_c=np.full(21, 25.0, np.float32)),
        tactile=tactile,
        tactile_window=window,
        tactile_deform=np.zeros((5, 240, 240), np.uint8),
        tactile_deform_timestamp_ns=np.full(5, now, np.int64),
        tactile_deform_delayed=np.zeros((4, 5, 240, 240), np.uint8),
        tactile_deform_delayed_timestamps_ns=delayed_ts,
    )


class ScriptedCompletion:
    def __init__(self) -> None:
        self.status = CompletionStatus.IN_PROGRESS
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1
        self.status = CompletionStatus.IN_PROGRESS

    def update(self, *, now_ns: int, **kwargs) -> CompletionResult:
        del kwargs
        return CompletionResult(
            self.status,
            "scripted",
            np.zeros(5, bool),
            0,
            now_ns,
        )


class ResetRecordingEMGBridge:
    def __init__(self) -> None:
        self.reset_active: list[bool] = []

    def reset(self, *, active: bool = False) -> None:
        self.reset_active.append(bool(active))


class NeverResultPlannerWorker:
    """Deterministic pending Planner used to exercise pre-START expiry."""

    def __init__(self) -> None:
        self.busy = False
        self.submit_count = 0

    def submit(self, item) -> bool:
        del item
        if self.busy:
            return False
        self.busy = True
        self.submit_count += 1
        return True

    def poll(self, **kwargs):
        del kwargs
        return None

    def close(self, *, timeout_s: float = 1.0) -> bool:
        del timeout_s
        return True


def drive_to_first_chunk(coordinator, task: SupportedTask, *, base_ns: int):
    now = base_ns
    sequence = 0
    for _ in range(3):
        coordinator.step_30hz(synchronized(now, sequence, EmgIntent.REST))
        now += 33_333_333
        sequence += 1
    primitive = TASK_GRASP_PRIMITIVES[task]
    coordinator.step_30hz(synchronized(now, sequence, EmgIntent(primitive.value)))
    now += 33_333_333
    sequence += 1
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        tick = synchronized(now, sequence, EmgIntent.REST)
        result = coordinator.step_30hz(tick)
        if result.policy and result.policy.state is AsyncPolicyState.READY:
            return result, tick, sequence
        now += 33_333_333
        sequence += 1
        time.sleep(0.001)
    raise AssertionError("first policy chunk was not accepted")


@pytest.mark.parametrize("task", list(SupportedTask))
def test_four_frozen_tasks_run_planner_executive_async_policy_and_single_writer(
    task: SupportedTask,
) -> None:
    coordinator, backend = build(task)
    primitive: EMGPrimitive = TASK_GRASP_PRIMITIVES[task]
    now = 1_000_000_000
    sequence = 0
    for _ in range(3):
        coordinator.step_30hz(synchronized(now, sequence, EmgIntent.REST))
        now += 33_333_333
        sequence += 1
    started = coordinator.step_30hz(
        synchronized(now, sequence, EmgIntent(primitive.value))
    )
    sequence += 1
    now += 33_333_333

    deadline = time.monotonic() + 2.0
    accepted = None
    latest_tick = None
    while time.monotonic() < deadline:
        latest_tick = synchronized(now, sequence, EmgIntent.REST)
        accepted = coordinator.step_30hz(latest_tick)
        if (
            accepted.executive.output in {ExecutiveOutput.START, ExecutiveOutput.CONTINUE}
            and accepted.policy is not None
            and accepted.policy.state is AsyncPolicyState.READY
        ):
            break
        now += 33_333_333
        sequence += 1
        time.sleep(0.001)
    assert accepted is not None and accepted.planner is not None
    assert accepted.planner.task is task
    assert accepted.planner.primitive is primitive
    assert accepted.executive.lease is not None
    assert accepted.policy is not None and accepted.policy.state is AsyncPolicyState.READY
    assert latest_tick is not None
    servo = asyncio.run(
        coordinator.step_servo(latest_tick, source_chunk_id=accepted.policy.chunk.chunk_id)
    )
    assert servo.wrote_command
    assert backend.commands
    coordinator.close()


def test_stable_hold_release_and_complete_clear_policy_without_a_second_authority() -> None:
    completion = ScriptedCompletion()
    coordinator, backend = build(
        SupportedTask.BOTTLE, completion=completion
    )
    try:
        accepted, tick, sequence = drive_to_first_chunk(
            coordinator, SupportedTask.BOTTLE, base_ns=5_000_000_000
        )
        first = asyncio.run(
            coordinator.step_servo(
                tick, source_chunk_id=accepted.policy.chunk.chunk_id
            )
        )
        assert first.wrote_command
        generation = coordinator.policy_worker.generation

        completion.status = CompletionStatus.GRASP_STABLE
        hold_tick = synchronized(tick.now_ns + 10_000_000, sequence + 1, EmgIntent.REST)
        hold = coordinator.step_30hz(hold_tick)
        assert hold.executive.output is ExecutiveOutput.HOLD
        assert hold.executive.directive is MotionDirective.HOLD_POSITION
        assert coordinator.policy_worker.generation > generation
        held = asyncio.run(coordinator.step_servo(hold_tick))
        assert held.wrote_command

        generation = coordinator.policy_worker.generation
        release_tick = synchronized(
            hold_tick.now_ns + 10_000_000, sequence + 2, EmgIntent.RELEASE
        )
        release = coordinator.step_30hz(release_tick)
        assert release.executive.directive is MotionDirective.CONTROLLED_OPEN
        assert coordinator.policy_worker.generation > generation
        opened = asyncio.run(coordinator.step_servo(release_tick))
        assert opened.wrote_command

        generation = coordinator.policy_worker.generation
        completion.status = CompletionStatus.RELEASED
        complete_tick = synchronized(
            release_tick.now_ns + 10_000_000, sequence + 3, EmgIntent.REST
        )
        complete = coordinator.step_30hz(complete_tick)
        assert complete.executive.output is ExecutiveOutput.COMPLETE
        assert complete.executive.directive is MotionDirective.NONE
        assert coordinator.policy_worker.generation > generation
    finally:
        coordinator.close()


def test_inactive_release_is_wait_none_and_never_writes_idle_open() -> None:
    coordinator, backend = build(SupportedTask.BOTTLE)
    try:
        tick = synchronized(1_000_000_000, 1, EmgIntent.RELEASE)
        result = coordinator.step_30hz(tick)
        assert result.executive.output is ExecutiveOutput.WAIT
        assert result.executive.directive is MotionDirective.NONE
        assert result.executive.reason == "release_without_active_task_ignored"
        servo = asyncio.run(coordinator.step_servo(tick))
        assert not servo.wrote_command
        assert servo.reason == "no_motion"
        assert backend.commands == []
    finally:
        coordinator.close()


def _ask_response():
    return {
        "schema_version": "planner_v1",
        "status": "ASK_CLARIFY",
        "primitive": "POWER_GRASP",
        "target_category": "bottle",
        "target_part": "body",
        "grasp_style": "power",
        "target_present": True,
        "near_ready": True,
        "center_ready": True,
        "compatible": True,
        "ambiguity": {
            "ambiguous": True,
            "reason": "two candidate bottles",
            "candidates": ["bottle"],
            "question": "Use the bottle nearest the center?",
        },
        "target_region": [0.5, 0.5, 0.5, 0.65],
        "ready_frame_count": 3,
        "confidence": 0.8,
        "instruction": "",
        "ask_clarify": True,
        "reason_code": "AMBIGUOUS_TARGET",
    }


def _ready_response():
    return {
        "schema_version": "planner_v1",
        "status": "READY",
        "primitive": "POWER_GRASP",
        "target_category": "bottle",
        "target_part": "body",
        "grasp_style": "power",
        "target_present": True,
        "near_ready": True,
        "center_ready": True,
        "compatible": True,
        "ambiguous": False,
        "target_region": [0.5, 0.5, 0.5, 0.65],
        "ready_frame_count": 3,
        "confidence": 0.95,
        "instruction": "Grasp the centered bottle using a power grasp and hold securely.",
        "ask_clarify": False,
        "reason_code": "READY",
    }


def test_ask_clarify_answer_replans_with_same_identity_then_starts() -> None:
    backend = MockPlannerBackend(
        scripted_responses=[_ask_response(), _ask_response(), _ready_response()]
    )
    worker = AsyncPlannerWorker(AskToClarifyPlanner(backend))
    coordinator, _ = build(SupportedTask.BOTTLE, planner_worker=worker)
    now = 8_000_000_000
    sequence = 0
    try:
        for _ in range(3):
            coordinator.step_30hz(synchronized(now, sequence, EmgIntent.REST))
            now += 33_333_333
            sequence += 1
        result = coordinator.step_30hz(
            synchronized(now, sequence, EmgIntent.POWER_GRASP)
        )
        deadline = time.monotonic() + 1.0
        while coordinator.clarification_token is None and time.monotonic() < deadline:
            now += 33_333_333
            sequence += 1
            result = coordinator.step_30hz(synchronized(now, sequence, EmgIntent.REST))
            time.sleep(0.001)
        token = coordinator.clarification_token
        assert token is not None
        assert result.planner is not None and result.planner.status is PlannerStatus.ASK_CLARIFY
        assert result.executive.output is ExecutiveOutput.WAIT
        assert result.executive.directive is MotionDirective.NONE

        coordinator.submit_clarification(
            "Yes, use the bottle nearest the center.",
            event_id=token.event_id,
            planner_generation=token.planner_generation,
            task_version=token.task_version,
            received_at_ns=now,
        )
        accepted = None
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            now += 33_333_333
            sequence += 1
            accepted = coordinator.step_30hz(
                synchronized(now, sequence, EmgIntent.REST)
            )
            if accepted.executive.output is ExecutiveOutput.START:
                break
            time.sleep(0.001)
        assert accepted is not None
        assert accepted.executive.output is ExecutiveOutput.START
        assert accepted.planner is not None and accepted.planner.status is PlannerStatus.READY
        assert any(
            turn.get("content") == "Yes, use the bottle nearest the center."
            for call in backend.calls
            for turn in call["conversation"]
        )
    finally:
        coordinator.close()


def test_stale_or_wrong_clarification_answer_is_rejected_without_motion() -> None:
    backend = MockPlannerBackend(scripted_responses=[_ask_response()])
    coordinator, hand = build(
        SupportedTask.BOTTLE,
        planner_worker=AsyncPlannerWorker(AskToClarifyPlanner(backend)),
    )
    now = 9_000_000_000
    try:
        for sequence in range(3):
            coordinator.step_30hz(synchronized(now, sequence, EmgIntent.REST))
            now += 33_333_333
        result = coordinator.step_30hz(
            synchronized(now, 3, EmgIntent.POWER_GRASP)
        )
        sequence = 3
        deadline = time.monotonic() + 1.0
        while coordinator.clarification_token is None and time.monotonic() < deadline:
            now += 33_333_333
            sequence += 1
            result = coordinator.step_30hz(synchronized(now, sequence, EmgIntent.REST))
            time.sleep(0.001)
        token = coordinator.clarification_token
        assert token is not None
        with pytest.raises(RuntimeError, match="stale_or_wrong"):
            coordinator.submit_clarification(
                "Use the center bottle.",
                event_id=token.event_id,
                planner_generation=token.planner_generation + 1,
                task_version=token.task_version,
                received_at_ns=now,
            )
        with pytest.raises(ValueError, match="must not be empty"):
            coordinator.submit_clarification(
                "  ",
                event_id=token.event_id,
                planner_generation=token.planner_generation,
                task_version=token.task_version,
                received_at_ns=now,
            )
        assert result.executive.output is ExecutiveOutput.WAIT
        assert result.executive.directive is MotionDirective.NONE
        assert hand.commands == []
    finally:
        coordinator.close()


def test_ask_clarify_never_auto_resubmits_or_moves_without_an_answer() -> None:
    backend = MockPlannerBackend(scripted_responses=[_ask_response(), _ready_response()])
    coordinator, hand = build(
        SupportedTask.BOTTLE,
        planner_worker=AsyncPlannerWorker(AskToClarifyPlanner(backend)),
    )
    now = 9_500_000_000
    sequence = 0
    try:
        for _ in range(3):
            coordinator.step_30hz(synchronized(now, sequence, EmgIntent.REST))
            now += 33_333_333
            sequence += 1
        result = coordinator.step_30hz(
            synchronized(now, sequence, EmgIntent.POWER_GRASP)
        )
        deadline = time.monotonic() + 1.0
        while coordinator.clarification_token is None and time.monotonic() < deadline:
            now += 33_333_333
            sequence += 1
            result = coordinator.step_30hz(synchronized(now, sequence, EmgIntent.REST))
            time.sleep(0.001)
        assert coordinator.clarification_token is not None
        assert len(backend.calls) == 1
        for _ in range(10):
            now += 1_000_000
            sequence += 1
            result = coordinator.step_30hz(synchronized(now, sequence, EmgIntent.REST))
            assert result.executive.output is ExecutiveOutput.WAIT
            assert result.executive.directive is MotionDirective.NONE
        assert len(backend.calls) == 1
        assert hand.commands == []
    finally:
        coordinator.close()


def test_terminal_acknowledgement_resets_every_task_local_component() -> None:
    completion = ScriptedCompletion()
    emg_bridge = ResetRecordingEMGBridge()
    coordinator, _ = build(
        SupportedTask.BOTTLE,
        completion=completion,
        emg_bridge=emg_bridge,
    )
    try:
        accepted, tick, sequence = drive_to_first_chunk(
            coordinator, SupportedTask.BOTTLE, base_ns=5_500_000_000
        )
        assert accepted.executive.lease is not None
        completion.status = CompletionStatus.GRASP_STABLE
        hold_tick = synchronized(tick.now_ns + 10_000_000, sequence + 1, EmgIntent.REST)
        coordinator.step_30hz(hold_tick)
        release_tick = synchronized(
            hold_tick.now_ns + 10_000_000, sequence + 2, EmgIntent.RELEASE
        )
        coordinator.step_30hz(release_tick)
        completion.status = CompletionStatus.RELEASED
        completed = coordinator.step_30hz(
            synchronized(
                release_tick.now_ns + 10_000_000,
                sequence + 3,
                EmgIntent.REST,
            )
        )
        assert completed.executive.output is ExecutiveOutput.COMPLETE

        generation = coordinator.policy_worker.generation
        coordinator.reset_terminal(safe_state_confirmed=True)
        assert coordinator.executive.phase.value == "IDLE_WAIT"
        assert coordinator.executive.lease is None
        assert coordinator.policy_worker.generation > generation
        assert not coordinator.context.ready
        assert coordinator._last_executive is None
        assert emg_bridge.reset_active == [False]
    finally:
        coordinator.close()


def test_pending_intent_expiry_resets_emg_gate_for_new_start_without_release() -> None:
    planner = NeverResultPlannerWorker()
    emg_bridge = ResetRecordingEMGBridge()
    ttl_ns = 50_000_000
    coordinator, _ = build(
        SupportedTask.BOTTLE,
        planner_worker=planner,
        emg_bridge=emg_bridge,
        executive_config=TaskExecutiveConfig(
            commit_stability_ns=0,
            pending_intent_ttl_ns=ttl_ns,
            planner_sla_ns=ttl_ns,
            planner_source_max_age_ns=ttl_ns,
        ),
    )
    # This test exercises EMG/planner lifecycle only; avoid coupling it to the
    # low-texture synthetic RGB focus metric used by a separate vision suite.
    coordinator.camera_health = CameraHealthMonitor(
        CameraHealthConfig(CALIBRATION, min_focus_measure=0.0)
    )
    now = 1_500_000_000
    try:
        for sequence in range(3):
            coordinator.step_30hz(
                synchronized(now, sequence, EmgIntent.REST)
            )
            now += 33_333_333

        first = coordinator.step_30hz(
            synchronized(now, 3, EmgIntent.POWER_GRASP)
        )
        assert first.executive.phase.value == "CONTEXT_WAIT"
        assert planner.submit_count == 1

        expired = coordinator.step_30hz(
            synchronized(now + ttl_ns + 1, 4, EmgIntent.REST)
        )
        assert expired.executive.reason == "pending_intent_expired"
        assert expired.executive.phase.value == "IDLE_WAIT"
        assert emg_bridge.reset_active == [False]
        assert coordinator._latched_primitive is None

        # No RELEASE edge is required after a pre-START timeout.  A new start
        # edge is independently latched and returns to CONTEXT_WAIT.
        second = coordinator.step_30hz(
            synchronized(now + ttl_ns + 2, 5, EmgIntent.POWER_GRASP)
        )
        assert second.executive.phase.value == "CONTEXT_WAIT"
        assert second.executive.reason != "waiting_for_start_primitive"
    finally:
        coordinator.close()


def test_abort_softstops_without_issuing_an_open_command_and_clears_cache() -> None:
    coordinator, backend = build(SupportedTask.BOTTLE)
    try:
        accepted, tick, sequence = drive_to_first_chunk(
            coordinator, SupportedTask.BOTTLE, base_ns=6_000_000_000
        )
        first = asyncio.run(
            coordinator.step_servo(
                tick, source_chunk_id=accepted.policy.chunk.chunk_id
            )
        )
        assert first.wrote_command
        command_count = len(backend.commands)
        generation = coordinator.policy_worker.generation
        abort_tick = replace(
            synchronized(tick.now_ns + 10_000_000, sequence + 1, EmgIntent.REST),
            safety=SafetySignal(SafetyLevel.ABORT, "fixture_fault"),
        )
        aborted = coordinator.step_30hz(abort_tick)
        assert aborted.executive.output is ExecutiveOutput.ABORT
        assert aborted.executive.directive is MotionDirective.SAFE_STOP
        assert coordinator.policy_worker.generation > generation
        servo = asyncio.run(coordinator.step_servo(abort_tick))
        assert not servo.wrote_command
        assert len(backend.commands) == command_count
        assert backend.soft_stop_reasons
        with pytest.raises(ValueError, match="safe_state_confirmed"):
            coordinator.reset_terminal(safe_state_confirmed=False)
        with pytest.raises(ValueError, match="operator_reset_confirmed"):
            coordinator.reset_terminal(safe_state_confirmed=True)
        coordinator.reset_terminal(
            safe_state_confirmed=True,
            operator_reset_confirmed=True,
        )
        assert coordinator.executive.phase.value == "IDLE_WAIT"
        assert not coordinator.servo.pipeline.hard_fault_latched
        # The same still-open coordinator can accept a fresh task only after
        # the explicit safe/operator acknowledgement above.
        restarted, _, _ = drive_to_first_chunk(
            coordinator, SupportedTask.BOTTLE, base_ns=7_000_000_000
        )
        assert restarted.executive.output in {
            ExecutiveOutput.START,
            ExecutiveOutput.CONTINUE,
        }
    finally:
        coordinator.close()


def test_profile_b_full_runtime_is_blocked_until_diff_completion_is_calibrated() -> None:
    coordinator, _ = build(SupportedTask.BOTTLE)
    with pytest.raises(RuntimeError, match="no calibrated DIFF-only CompletionMonitor"):
        OnlineV1Coordinator(
            planner_worker=coordinator.planner_worker,
            camera_health=coordinator.camera_health,
            visual_gate=coordinator.visual_gate,
            executive=coordinator.executive,
            policy_worker=coordinator.policy_worker,
            completion=coordinator.completion,
            servo=coordinator.servo,
            versions=versions(),
            tactile_profile="profile_b_diff_only",
        )
    coordinator.close()


class DelayedFirstChunkBackend(MockTReXBackend):
    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def slow_and_fast(self, observation):
        self.entered.set()
        if not self.release.wait(2.0):
            raise TimeoutError("test did not release delayed policy")
        return super().slow_and_fast(observation)


class DelayedSecondSlowBackend(MockTReXBackend):
    def __init__(self) -> None:
        super().__init__()
        self.slow_calls = 0
        self.second_entered = threading.Event()
        self.release_second = threading.Event()

    def slow_and_fast(self, observation):
        self.slow_calls += 1
        if self.slow_calls == 2:
            self.second_entered.set()
            if not self.release_second.wait(2.0):
                raise TimeoutError("test did not release second slow chunk")
        return super().slow_and_fast(observation)


def test_first_slow_chunk_rebases_execution_epoch_after_more_than_one_chunk_latency() -> None:
    backend = DelayedFirstChunkBackend()
    coordinator, _ = build(SupportedTask.BOTTLE, policy_backend=backend)
    now = 2_000_000_000
    sequence = 0
    try:
        for _ in range(3):
            coordinator.step_30hz(synchronized(now, sequence, EmgIntent.REST))
            now += 33_333_333
            sequence += 1
        coordinator.step_30hz(
            synchronized(now, sequence, EmgIntent.POWER_GRASP)
        )
        now += 33_333_333
        sequence += 1
        deadline = time.monotonic() + 1.0
        while not backend.entered.is_set() and time.monotonic() < deadline:
            result = coordinator.step_30hz(
                synchronized(now, sequence, EmgIntent.REST)
            )
            assert result.executive.output in {
                ExecutiveOutput.WAIT,
                ExecutiveOutput.START,
                ExecutiveOutput.HOLD,
                ExecutiveOutput.CONTINUE,
            }
            now += 33_333_333
            sequence += 1
            time.sleep(0.001)
        assert backend.entered.is_set()

        # The control path continues ticking while the GPU is blocked.  The
        # measured remote slow-and-fast path is about 1.22 s; exercise at
        # least 1.3 s while renewing only the current task lease/modalities.
        delay_started_ns = now
        while now - delay_started_ns < 1_300_000_000:
            coordinator.step_30hz(
                synchronized(now, sequence, EmgIntent.REST)
            )
            now += 33_333_333
            sequence += 1
        backend.release.set()
        accepted = None
        latest_tick = None
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            latest_tick = synchronized(now, sequence, EmgIntent.REST)
            accepted = coordinator.step_30hz(latest_tick)
            if accepted.policy and accepted.policy.state is AsyncPolicyState.READY:
                break
            now += 1_000_000
            sequence += 1
            time.sleep(0.001)
        assert accepted is not None and accepted.policy is not None
        assert accepted.policy.state is AsyncPolicyState.READY
        assert latest_tick is not None
        servo = asyncio.run(
            coordinator.step_servo(
                latest_tick, source_chunk_id=accepted.policy.chunk.chunk_id
            )
        )
        expected_first = accepted.policy.chunk.q_target_rad[0]
        np.testing.assert_allclose(servo.nominal_q_rad, expected_first, atol=1e-6)
    finally:
        backend.release.set()
        coordinator.close()


def test_slow_response_over_latency_budget_is_dropped_to_hold_not_abort() -> None:
    backend = DelayedFirstChunkBackend()
    coordinator, _ = build(SupportedTask.BOTTLE, policy_backend=backend)
    now = 3_500_000_000
    sequence = 0
    try:
        for _ in range(3):
            coordinator.step_30hz(synchronized(now, sequence, EmgIntent.REST))
            now += 33_333_333
            sequence += 1
        coordinator.step_30hz(
            synchronized(now, sequence, EmgIntent.POWER_GRASP)
        )
        now += 33_333_333
        sequence += 1
        deadline = time.monotonic() + 1.0
        while not backend.entered.is_set() and time.monotonic() < deadline:
            coordinator.step_30hz(synchronized(now, sequence, EmgIntent.REST))
            now += 33_333_333
            sequence += 1
            time.sleep(0.001)
        assert backend.entered.is_set()

        delay_started_ns = now
        while now - delay_started_ns < 1_600_000_000:
            interim = coordinator.step_30hz(
                synchronized(now, sequence, EmgIntent.REST)
            )
            assert interim.executive.output is not ExecutiveOutput.ABORT
            now += 33_333_333
            sequence += 1
        backend.release.set()

        rejected = None
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            rejected = coordinator.step_30hz(
                synchronized(now, sequence, EmgIntent.REST)
            )
            if rejected.policy and rejected.policy.state is AsyncPolicyState.READY:
                break
            now += 1_000_000
            sequence += 1
            time.sleep(0.001)
        assert rejected is not None and rejected.policy is not None
        assert rejected.policy.state is AsyncPolicyState.READY
        assert rejected.executive.output is ExecutiveOutput.HOLD
        assert "slow_response_latency_budget_exceeded" in rejected.executive.reason
        assert coordinator._action_epoch_ns is None
    finally:
        backend.release.set()
        coordinator.close()


def test_every_new_slow_chunk_rebases_after_more_than_one_chunk_latency() -> None:
    backend = DelayedSecondSlowBackend()
    coordinator, _ = build(SupportedTask.BOTTLE, policy_backend=backend)
    period_ns = coordinator.policy_worker.schedule.action_period_ns
    try:
        first, tick, sequence = drive_to_first_chunk(
            coordinator, SupportedTask.BOTTLE, base_ns=5_000_000_000
        )
        assert first.policy is not None and first.policy.chunk.start_step == 0
        now = tick.now_ns + period_ns
        sequence += 1

        deadline = time.monotonic() + 2.0
        while not backend.second_entered.is_set() and time.monotonic() < deadline:
            result = coordinator.step_30hz(
                synchronized(now, sequence, EmgIntent.REST)
            )
            now += period_ns
            sequence += 1
            time.sleep(0.001)
        assert backend.second_entered.is_set()
        assert backend.slow_calls == 2

        # The second slow request was submitted for start_step=16.  Make its
        # GPU latency exceed the entire 16/30 s chunk before accepting it.
        delay_started_ns = now
        while now - delay_started_ns < 1_300_000_000:
            coordinator.step_30hz(
                synchronized(now, sequence, EmgIntent.REST)
            )
            now += period_ns
            sequence += 1
        backend.release_second.set()
        accepted = None
        accepted_tick = None
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            accepted_tick = synchronized(now, sequence, EmgIntent.REST)
            accepted = coordinator.step_30hz(accepted_tick)
            if accepted.policy and accepted.policy.state is AsyncPolicyState.READY:
                break
            now += 1_000_000
            sequence += 1
            time.sleep(0.001)
        assert accepted is not None and accepted.policy is not None
        assert accepted.policy.state is AsyncPolicyState.READY
        assert accepted.policy.chunk.start_step == 16
        expected_epoch = accepted_tick.now_ns - 16 * period_ns
        assert coordinator._action_epoch_ns == expected_epoch

        servo = asyncio.run(
            coordinator.step_servo(
                accepted_tick,
                source_chunk_id=accepted.policy.chunk.chunk_id,
            )
        )
        np.testing.assert_allclose(
            servo.nominal_q_rad,
            accepted.policy.chunk.q_target_rad[0],
            atol=1e-6,
        )

        # READY consumption suppresses an offset-zero submit on the acceptance
        # tick; the next real 30 Hz tick advances to step 17, not another 16.
        coordinator.step_30hz(
            synchronized(accepted_tick.now_ns + period_ns, sequence + 1, EmgIntent.REST)
        )
        time.sleep(0.01)
        assert backend.slow_calls == 2
    finally:
        backend.release_second.set()
        coordinator.close()


def test_runtime_input_rejects_noncausal_or_malformed_diff_before_start() -> None:
    base = synchronized(3_000_000_000, 1, EmgIntent.REST)
    current_only = replace(
        base,
        tactile_deform_delayed=None,
        tactile_deform_delayed_timestamps_ns=None,
    )
    assert current_only.tactile_deform.shape == (5, 240, 240)
    with pytest.raises(ValueError, match="current DIFF must be uint8"):
        replace(base, tactile_deform=base.tactile_deform.astype(np.float32))
    with pytest.raises(ValueError, match="current DIFF must be uint8"):
        replace(base, tactile_deform=base.tactile_deform[:, :-1])
    bad_current = base.tactile_deform_timestamp_ns.copy()
    bad_current[0] = base.now_ns + 1
    with pytest.raises(ValueError, match="causal"):
        replace(base, tactile_deform_timestamp_ns=bad_current)


def test_profile_a_start_waits_when_any_current_diff_finger_is_stale() -> None:
    coordinator, _ = build(SupportedTask.BOTTLE)
    now = 4_000_000_000
    sequence = 0
    try:
        for _ in range(3):
            coordinator.step_30hz(synchronized(now, sequence, EmgIntent.REST))
            now += 33_333_333
            sequence += 1
        start_tick = synchronized(now, sequence, EmgIntent.POWER_GRASP)
        stale_ts = start_tick.tactile_deform_timestamp_ns.copy()
        stale_ts[0] = now - coordinator.executive.config.touch_ttl_ns - 1
        stale_delayed_ts = start_tick.tactile_deform_delayed_timestamps_ns.copy()
        stale_delayed_ts[:, 0] = stale_ts[0] - np.arange(3, -1, -1) * 1_000_000
        result = coordinator.step_30hz(
            replace(
                start_tick,
                tactile_deform_timestamp_ns=stale_ts,
                tactile_deform_delayed_timestamps_ns=stale_delayed_ts,
            )
        )
        deadline = time.monotonic() + 1.0
        while result.planner is None and time.monotonic() < deadline:
            now += 1_000_000
            sequence += 1
            tick = synchronized(now, sequence, EmgIntent.REST)
            stale_ts = tick.tactile_deform_timestamp_ns.copy()
            stale_ts[0] = now - coordinator.executive.config.touch_ttl_ns - 1
            stale_delayed_ts = tick.tactile_deform_delayed_timestamps_ns.copy()
            stale_delayed_ts[:, 0] = stale_ts[0] - np.arange(3, -1, -1) * 1_000_000
            result = coordinator.step_30hz(
                replace(
                    tick,
                    tactile_deform_timestamp_ns=stale_ts,
                    tactile_deform_delayed_timestamps_ns=stale_delayed_ts,
                )
            )
            time.sleep(0.001)
        assert result.executive.output is ExecutiveOutput.WAIT
        assert "diff_five_fingers_not_ready" in result.executive.reason
    finally:
        coordinator.close()
