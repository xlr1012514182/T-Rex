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
    TactileProfileReadiness,
    completion_state_from_status,
)
from revo3_v1.revo import CompletionStatus
from revo3_v1.planner import (
    Ambiguity,
    NormalizedBBox,
    PlannerDecision,
    PlannerStatus,
    SupportedTask,
    VisualGateResult,
)
from revo3_v1.emg import EMGPrimitive


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
        primitive=EMGPrimitive.POWER_GRASP,
        target_part="body",
        grasp_style="power",
        center_ready=True,
        ready_frame_count=3,
        reason_code="READY",
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
        tactile_readiness=TactileProfileReadiness.evaluate(
            profile_kind="A",
            profile_hash=(runtime_versions or versions()).tactile_profile_hash,
            force6d_history_frames=16,
            diff_valid_fingers=5,
        ),
    )


def executive():
    ids = iter((f"id-{index}" for index in range(100)))
    return TaskExecutive(
        TaskExecutiveConfig(commit_stability_ns=0),
        id_factory=lambda: next(ids),
    )


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


def test_expired_active_lease_cannot_be_renewed_or_revived_after_control_gap():
    ex = executive()
    started = ex.step(tick(10, EmgIntent.CLOSE))
    assert started.output is ExecutiveOutput.START
    assert started.lease is not None
    expired_at = started.lease.expires_at_ns
    result = ex.step(tick(expired_at, EmgIntent.REST))
    assert result.output is ExecutiveOutput.ABORT
    assert result.directive is MotionDirective.SAFE_STOP
    assert result.reason == "active_task_lease_expired"
    assert ex.phase is ExecutivePhase.FAULT_LATCHED


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
    assert opened.directive == MotionDirective.NONE
    assert opened.reason == "release_without_active_task_ignored"
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


def test_policy_response_latency_budgets_separate_slow_from_fast() -> None:
    ex = executive()
    started = ex.step(tick(100, EmgIntent.CLOSE))
    lease = started.lease
    assert lease is not None
    base = PolicyResponseEnvelope(
        task_id=lease.task_id,
        task_version=lease.task_version,
        lease_id=lease.lease_id,
        instruction_hash=lease.instruction_hash,
        version_fingerprint=lease.version_fingerprint,
        observation_timestamp_ns=100,
        produced_at_ns=1_300_000_100,
        inference_mode="slow_and_fast",
    )
    # Refreshing an active tick renews the lease independently of the model's
    # observation timestamp; identity and current modality checks remain live.
    ex.step(tick(1_300_000_100, EmgIntent.REST))
    assert ex.validate_policy_response(base, now_ns=1_300_000_100) == (
        True,
        "accepted",
    )
    too_slow = replace(base, produced_at_ns=1_500_000_101)
    assert ex.validate_policy_response(too_slow, now_ns=1_500_000_101) == (
        False,
        "slow_response_latency_budget_exceeded",
    )
    late_fast = replace(
        base,
        produced_at_ns=500_000_101,
        inference_mode="fast",
    )
    assert ex.validate_policy_response(late_fast, now_ns=500_000_101) == (
        False,
        "fast_response_latency_budget_exceeded",
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


def test_no_progress_does_not_replan_after_contact_is_established():
    ex = executive()
    ex.step(tick(100, EmgIntent.POWER_GRASP))
    contact = ex.step(
        tick(110, EmgIntent.REST, completion=CompletionState.CONTACT_ESTABLISHED)
    )
    assert contact.output == ExecutiveOutput.CONTINUE
    assert ex.phase == ExecutivePhase.CONTACT_BUILD
    after_contact = ex.step(
        tick(120, EmgIntent.REST, completion=CompletionState.NO_PROGRESS)
    )
    assert after_contact.output == ExecutiveOutput.ABORT
    assert after_contact.reason == "no_progress_after_contact"
    assert after_contact.directive == MotionDirective.SAFE_STOP


def test_single_revo_completion_monitor_status_maps_into_executive_contract():
    assert completion_state_from_status(CompletionStatus.CONTACT_ESTABLISHED) == CompletionState.CONTACT_ESTABLISHED
    assert completion_state_from_status(CompletionStatus.GRASP_STABLE) == CompletionState.GRASP_STABLE
    assert completion_state_from_status(CompletionStatus.RELEASED) == CompletionState.RELEASED


def test_each_grasp_primitive_can_start_and_is_written_to_lease():
    for intent, task, style in (
        (EmgIntent.POWER_GRASP, SupportedTask.BOTTLE, "power"),
        (EmgIntent.PRECISION_GRASP, SupportedTask.PHONE, "precision"),
        (EmgIntent.LATERAL_GRASP, SupportedTask.REFRIGERATOR_DOOR, "lateral"),
    ):
        ex = executive()
        decision = replace(
            planner(10),
            task=task,
            primitive=EMGPrimitive(intent.value),
            instruction=f"Grasp the centered {task.value} using a {style} grasp.",
            grasp_style=style,
        )
        current = tick(10, intent, planner_value=decision)
        current = replace(current, visual=replace(current.visual, task=task))
        result = ex.step(current)
        assert result.output == ExecutiveOutput.START
        assert result.lease.primitive == intent.value


def test_atomic_commit_requires_150ms_stability_and_rejects_toctou_change():
    ids = iter((f"atomic-{index}" for index in range(20)))
    ex = TaskExecutive(id_factory=lambda: next(ids))
    first = ex.step(tick(1_000_000_000, EmgIntent.POWER_GRASP))
    assert first.output == ExecutiveOutput.WAIT
    assert first.reason == "atomic_commit_pending"
    changed = tick(
        1_100_000_000,
        EmgIntent.REST,
        planner_value=planner(1_000_000_000, "Grasp the other bottle using a power grasp."),
    )
    changed = replace(changed, visual=visual(1_000_000_000))
    restarted = ex.step(changed)
    assert restarted.reason == "atomic_commit_candidate_changed"
    stable = tick(
        1_250_000_000,
        EmgIntent.REST,
        planner_value=changed.planner,
    )
    stable = replace(stable, visual=changed.visual)
    started = ex.step(stable)
    assert started.output == ExecutiveOutput.START


def test_default_atomic_commit_accepts_fresher_equivalent_context():
    ex = TaskExecutive()
    assert ex.step(tick(1_000_000_000, EmgIntent.POWER_GRASP)).reason == "atomic_commit_pending"
    for current in (1_050_000_000, 1_100_000_000):
        pending = ex.step(tick(current, EmgIntent.REST))
        assert pending.output == ExecutiveOutput.WAIT
        assert pending.reason == "atomic_commit_pending"
    started = ex.step(tick(1_150_000_000, EmgIntent.REST))
    assert started.output == ExecutiveOutput.START


def test_atomic_commit_fails_closed_on_profile_readiness_or_hash_change():
    ids = iter((f"readiness-{index}" for index in range(20)))
    ex = TaskExecutive(id_factory=lambda: next(ids))
    candidate = tick(1_000_000_000, EmgIntent.POWER_GRASP)
    assert ex.step(candidate).reason == "atomic_commit_pending"
    not_ready = replace(
        tick(1_100_000_000, EmgIntent.REST, planner_value=candidate.planner),
        visual=candidate.visual,
        tactile_readiness=TactileProfileReadiness.evaluate(
            profile_kind="A", profile_hash="touch-a", force6d_history_frames=15
        ),
    )
    result = ex.step(not_ready)
    assert result.output == ExecutiveOutput.WAIT
    assert "tactile_profile_not_ready" in result.reason
    mismatched = replace(
        tick(1_200_000_000, EmgIntent.REST, planner_value=candidate.planner),
        visual=candidate.visual,
        tactile_readiness=TactileProfileReadiness.evaluate(
            profile_kind="B", profile_hash="wrong", diff_valid_fingers=5
        ),
    )
    assert ex.step(mismatched).reason == "tactile_profile_readiness_hash_mismatch"


def test_profile_specific_readiness_rules():
    assert TactileProfileReadiness.evaluate(
        profile_kind="A", profile_hash="a", force6d_history_frames=16,
        diff_valid_fingers=5,
    ).ready
    assert not TactileProfileReadiness.evaluate(
        profile_kind="A", profile_hash="a", force6d_history_frames=16,
        diff_valid_fingers=4,
    ).ready
    assert TactileProfileReadiness.evaluate(
        profile_kind="B", profile_hash="b", diff_valid_fingers=5
    ).ready
    assert not TactileProfileReadiness.evaluate(
        profile_kind="C",
        profile_hash="c",
        pressure_present=True,
        pressure_valid_mask_present=True,
    ).ready
    assert TactileProfileReadiness.evaluate(
        profile_kind="C",
        profile_hash="c",
        pressure_present=True,
        pressure_valid_mask_present=True,
        policy_adapter_ready=True,
    ).ready


def test_cached_planner_and_visual_from_before_start_event_cannot_start():
    ex = executive()
    start = tick(200, EmgIntent.POWER_GRASP)
    old = replace(start, planner=planner(199), visual=visual(199))
    waiting = ex.step(old)
    assert waiting.output == ExecutiveOutput.WAIT
    assert waiting.reason == "planner_precedes_latched_start_event"
    fresh = ex.step(tick(210, EmgIntent.REST))
    assert fresh.output == ExecutiveOutput.START


def test_slow_planner_source_remains_causal_while_produced_result_is_fresh():
    ex = executive()
    source_ns = 1_000_000_000
    assert ex.step(tick(source_ns, EmgIntent.POWER_GRASP, visual_ready=False)).output is ExecutiveOutput.WAIT
    produced_ns = source_ns + 15_000_000_000
    slow_planner = replace(planner(source_ns), produced_at_ns=produced_ns)
    current = tick(produced_ns, EmgIntent.REST, planner_value=slow_planner)
    current = replace(
        current,
        visual=replace(visual(source_ns), produced_at_ns=produced_ns),
    )
    started = ex.step(current)
    assert started.output is ExecutiveOutput.START


def test_fresh_planner_result_still_requires_current_state_camera_and_touch():
    ex = executive()
    source_ns = 1_000_000_000
    ex.step(tick(source_ns, EmgIntent.POWER_GRASP, visual_ready=False))
    produced_ns = source_ns + 15_000_000_000
    slow_planner = replace(planner(source_ns), produced_at_ns=produced_ns)
    current = tick(produced_ns, EmgIntent.REST, planner_value=slow_planner)
    current = replace(
        current,
        visual=replace(visual(source_ns), produced_at_ns=produced_ns),
        timestamps=replace(current.timestamps, state_ns=source_ns),
    )
    waiting = ex.step(current)
    assert waiting.output is ExecutiveOutput.WAIT
    assert waiting.reason == "stale_start_context"
    assert "state" in waiting.stale_modalities


def test_produced_fresh_result_with_expired_source_never_starts():
    ex = executive()
    source_ns = 1_000_000_000
    ex.step(tick(source_ns, EmgIntent.POWER_GRASP, visual_ready=False))
    now_ns = source_ns + ex.config.planner_source_max_age_ns + 1
    old = replace(planner(source_ns), produced_at_ns=now_ns)
    current = tick(now_ns, EmgIntent.REST, planner_value=old)
    current = replace(
        current,
        visual=replace(visual(source_ns), produced_at_ns=now_ns),
    )
    waiting = ex.step(current)
    assert waiting.output is ExecutiveOutput.WAIT
    assert waiting.directive is MotionDirective.NONE
    assert waiting.reason == "planner_source_expired"


def test_recoverable_stale_holds_then_aborts_at_bounded_timeout():
    ex = executive()
    ex.step(tick(1_000_000_000, EmgIntent.POWER_GRASP))
    first = tick(1_200_000_000, EmgIntent.REST)
    first = replace(first, timestamps=replace(first.timestamps, touch_ns=1_000_000_000))
    held = ex.step(first)
    assert held.output == ExecutiveOutput.HOLD
    assert held.stale_modalities == ("touch",)
    later = tick(1_700_000_000, EmgIntent.REST)
    later = replace(later, timestamps=replace(later.timestamps, touch_ns=1_000_000_000))
    aborted = ex.step(later)
    assert aborted.output == ExecutiveOutput.ABORT
    assert aborted.reason == "touch_stale_timeout"


def test_policy_stale_holds_then_aborts_at_bounded_timeout():
    ex = executive()
    ex.step(tick(1_000_000_000, EmgIntent.POWER_GRASP))
    first = tick(1_800_000_000, EmgIntent.REST)
    first = replace(first, timestamps=replace(first.timestamps, policy_ns=1_000_000_000))
    assert ex.step(first).output == ExecutiveOutput.HOLD
    later = tick(2_800_000_000, EmgIntent.REST)
    later = replace(later, timestamps=replace(later.timestamps, policy_ns=1_000_000_000))
    aborted = ex.step(later)
    assert aborted.output == ExecutiveOutput.ABORT
    assert aborted.reason == "policy_stale_timeout"


def test_first_policy_chunk_has_separate_three_second_startup_budget():
    ex = executive()
    ex.step(tick(1_000_000_000, EmgIntent.POWER_GRASP))
    at_recorded_trex_latency = tick(
        2_300_000_000, EmgIntent.REST, policy_timestamp=False
    )
    held = ex.step(at_recorded_trex_latency)
    assert held.output == ExecutiveOutput.HOLD
    assert held.stale_modalities == ("policy",)
    beyond_startup_budget = tick(
        4_100_000_000, EmgIntent.REST, policy_timestamp=False
    )
    aborted = ex.step(beyond_startup_budget)
    assert aborted.output == ExecutiveOutput.ABORT
    assert aborted.reason == "policy_stale_timeout"


def test_atomic_commit_safety_change_restarts_candidate():
    ex = TaskExecutive()
    initial = tick(1_000_000_000, EmgIntent.POWER_GRASP)
    assert ex.step(initial).reason == "atomic_commit_pending"
    changed = replace(
        tick(1_100_000_000, EmgIntent.REST, planner_value=initial.planner),
        visual=initial.visual,
        safety=SafetySignal(SafetyLevel.SAFE, "safety-profile-updated"),
    )
    assert ex.step(changed).reason == "atomic_commit_candidate_changed"
