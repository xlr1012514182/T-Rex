from dataclasses import replace

from revo3_v1.executive import (
    CompletionState,
    EmgEvent,
    EmgIntent,
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
    TaskExecutiveConfig,
)
from revo3_v1.planner import (
    Ambiguity,
    NormalizedBBox,
    PlannerDecision,
    PlannerStatus,
    SupportedTask,
    VisualGateResult,
)


def versions(policy="policy-a"):
    return RuntimeVersions(
        schema_version="v1",
        planner_revision="planner-a",
        policy_revision=policy,
        hardware_manifest_hash="hardware-a",
        joint_order_hash="joints-a",
        tactile_profile_hash="touch-a",
    )


def planner(t, instruction="Grasp the bottle and hold it."):
    bbox = NormalizedBBox(0.2, 0.2, 0.8, 0.8)
    return PlannerDecision(
        status=PlannerStatus.READY,
        task=SupportedTask.BOTTLE,
        bbox=bbox,
        area=bbox.area,
        confidence=0.95,
        target_present=True,
        near_ready=True,
        compatible=True,
        ambiguity=Ambiguity(False),
        instruction=instruction,
        timestamp_ns=t,
    )


def visual(t, ready=True):
    return VisualGateResult(
        ready=ready,
        reason="ready" if ready else "target_too_small",
        consecutive_ready=2 if ready else 0,
        task=SupportedTask.BOTTLE,
        area=0.36,
        center_distance=0.0,
        timestamp_ns=t,
    )


def tick(
    t,
    intent=EmgIntent.REST,
    *,
    visual_ready=True,
    planner_value=None,
    completion=CompletionState.IN_PROGRESS,
    safety=SafetySignal(),
    runtime_versions=None,
    policy_timestamp=True,
):
    return ExecutiveTick(
        now_ns=t,
        emg=EmgEvent(intent, t, confidence=0.95, event_id=f"event-{t}"),
        visual=visual(t, visual_ready),
        planner=planner_value if planner_value is not None else planner(t),
        timestamps=ModalityTimestamps(
            camera_ns=t,
            state_ns=t,
            touch_ns=t,
            policy_ns=t if policy_timestamp else None,
        ),
        versions=runtime_versions or versions(),
        safety=safety,
        completion=completion,
    )


def executive():
    ids = iter((f"id-{index}" for index in range(100)))
    return TaskExecutive(id_factory=lambda: next(ids))


def test_start_requires_both_close_and_visual_ready_then_latches_instruction():
    ex = executive()
    waiting = ex.step(tick(100, EmgIntent.CLOSE, visual_ready=False))
    assert waiting.output == ExecutiveOutput.WAIT
    assert ex.phase == ExecutivePhase.CONTEXT_WAIT

    # User can relax after CLOSE is latched; a ready visual/planner context starts.
    started = ex.step(tick(110, EmgIntent.REST))
    assert started.output == ExecutiveOutput.START
    assert started.directive == MotionDirective.POLICY
    assert started.clear_policy_cache
    assert started.latched_instruction == "Grasp the bottle and hold it."

    no_planner_tick = tick(115, EmgIntent.REST)
    no_planner_tick = replace(no_planner_tick, planner=None, visual=None)
    no_planner = ex.step(no_planner_tick)
    assert no_planner.output == ExecutiveOutput.CONTINUE
    assert no_planner.latched_instruction == "Grasp the bottle and hold it."

    # A new planner object is ignored during ACTIVE; the instruction is not replanned each tick.
    continued = ex.step(
        tick(
            120,
            EmgIntent.REST,
            planner_value=planner(120, "A different uncommitted instruction."),
        )
    )
    assert continued.output == ExecutiveOutput.CONTINUE
    assert continued.latched_instruction == "Grasp the bottle and hold it."


def test_rest_does_not_start_or_cancel_task():
    ex = executive()
    assert ex.step(tick(10, EmgIntent.REST)).output == ExecutiveOutput.WAIT
    assert ex.step(tick(20, EmgIntent.CLOSE)).output == ExecutiveOutput.START
    assert ex.step(tick(30, EmgIntent.REST)).output == ExecutiveOutput.CONTINUE


def test_explicit_release_uses_controlled_open_until_release_confirmation():
    ex = executive()
    ex.step(tick(10, EmgIntent.CLOSE))
    releasing = ex.step(tick(20, EmgIntent.RELEASE))
    assert releasing.output == ExecutiveOutput.CONTINUE
    assert releasing.directive == MotionDirective.CONTROLLED_OPEN
    assert releasing.clear_policy_cache

    still_releasing = ex.step(tick(30, EmgIntent.RELEASE))
    assert still_releasing.directive == MotionDirective.CONTROLLED_OPEN

    complete = ex.step(
        tick(40, EmgIntent.REST, completion=CompletionState.RELEASED)
    )
    assert complete.output == ExecutiveOutput.COMPLETE
    assert complete.clear_policy_cache


def test_open_before_start_cancels_pending_close_latch():
    ex = executive()
    waiting = ex.step(tick(10, EmgIntent.CLOSE, visual_ready=False))
    assert waiting.output == ExecutiveOutput.WAIT
    opened = ex.step(tick(20, EmgIntent.OPEN, visual_ready=True))
    assert opened.directive == MotionDirective.CONTROLLED_OPEN
    assert ex.phase == ExecutivePhase.IDLE_WAIT
    # REST cannot resurrect the cancelled close event.
    assert ex.step(tick(30, EmgIntent.REST)).output == ExecutiveOutput.WAIT


def test_stale_state_during_controlled_release_aborts():
    ex = executive()
    ex.step(tick(1_000_000_000, EmgIntent.CLOSE))
    ex.step(tick(1_010_000_000, EmgIntent.RELEASE))
    release_tick = tick(1_200_000_000, EmgIntent.REST)
    release_tick = replace(
        release_tick,
        timestamps=replace(release_tick.timestamps, state_ns=1_000_000_000),
    )
    result = ex.step(release_tick)
    assert result.output == ExecutiveOutput.ABORT
    assert result.reason == "stale_state_during_release"


def test_stable_grasp_holds_until_explicit_release():
    ex = executive()
    ex.step(tick(10, EmgIntent.CLOSE))
    held = ex.step(
        tick(20, EmgIntent.REST, completion=CompletionState.GRASP_STABLE)
    )
    assert held.output == ExecutiveOutput.HOLD
    assert held.directive == MotionDirective.HOLD_POSITION
    assert ex.step(tick(30, EmgIntent.REST)).output == ExecutiveOutput.HOLD


def test_hard_safety_is_final_veto_even_for_close():
    ex = executive()
    result = ex.step(
        tick(
            10,
            EmgIntent.CLOSE,
            safety=SafetySignal(SafetyLevel.EMERGENCY, "overcurrent"),
        )
    )
    assert result.output == ExecutiveOutput.ABORT
    assert result.directive == MotionDirective.SAFE_STOP


def test_future_emg_event_is_rejected():
    ex = executive()
    bad = tick(10, EmgIntent.CLOSE)
    bad = replace(bad, emg=replace(bad.emg, timestamp_ns=20_000_000))
    result = ex.step(bad)
    assert result.output == ExecutiveOutput.ABORT
    assert result.reason == "future_emg_event"


def test_stale_state_aborts_but_stale_camera_holds_active_task():
    ex = executive()
    ex.step(tick(1_000_000_000, EmgIntent.CLOSE))
    active = tick(1_200_000_000, EmgIntent.REST)
    active = replace(
        active,
        timestamps=replace(active.timestamps, state_ns=1_000_000_000),
    )
    assert ex.step(active).reason == "stale_state"

    ex2 = executive()
    ex2.step(tick(1_000_000_000, EmgIntent.CLOSE))
    stale_camera = tick(1_200_000_000, EmgIntent.REST)
    stale_camera = replace(
        stale_camera,
        timestamps=replace(stale_camera.timestamps, camera_ns=1_000_000_000),
    )
    result = ex2.step(stale_camera)
    assert result.output == ExecutiveOutput.HOLD
    assert result.stale_modalities == ("camera",)
    recovered = ex2.step(tick(1_210_000_000, EmgIntent.REST))
    assert recovered.output == ExecutiveOutput.CONTINUE


def test_runtime_version_change_aborts_and_clears_cache():
    ex = executive()
    ex.step(tick(10, EmgIntent.CLOSE))
    result = ex.step(
        tick(20, EmgIntent.REST, runtime_versions=versions(policy="policy-b"))
    )
    assert result.output == ExecutiveOutput.ABORT
    assert result.reason == "runtime_version_changed"
    assert result.clear_policy_cache


def test_policy_response_must_match_task_version_lease_instruction_and_version():
    ex = executive()
    started = ex.step(tick(100, EmgIntent.CLOSE))
    lease = started.lease
    assert lease is not None
    valid = PolicyResponseEnvelope(
        task_id=lease.task_id,
        task_version=lease.task_version,
        lease_id=lease.lease_id,
        instruction_hash=lease.instruction_hash,
        version_fingerprint=lease.version_fingerprint,
        observation_timestamp_ns=100,
        produced_at_ns=100,
    )
    assert ex.validate_policy_response(valid, now_ns=110) == (True, "accepted")
    invalid = replace(valid, task_version=99)
    assert ex.validate_policy_response(invalid, now_ns=110) == (
        False,
        "task_version_mismatch",
    )


def test_no_progress_replans_once_and_requires_a_new_planner_result():
    ex = executive()
    ex.step(tick(100, EmgIntent.CLOSE))
    result = ex.step(
        tick(200, EmgIntent.REST, completion=CompletionState.NO_PROGRESS)
    )
    assert result.output == ExecutiveOutput.REPLAN
    assert result.clear_policy_cache

    old_tick = tick(200, EmgIntent.REST, planner_value=planner(199))
    old = ex.step(replace(old_tick, visual=visual(199)))
    assert old.output == ExecutiveOutput.HOLD
    assert "old_planner_result" in old.reason

    committed = ex.step(tick(210, EmgIntent.REST, planner_value=planner(210)))
    assert committed.output == ExecutiveOutput.CONTINUE
    assert committed.reason == "replan_committed"
    assert committed.clear_policy_cache
