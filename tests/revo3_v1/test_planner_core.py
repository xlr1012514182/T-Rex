import json

import pytest

from revo3_v1.planner import (
    AskToClarifyPlanner,
    MockPlannerBackend,
    PlannerRequest,
    PlannerStatus,
    Qwen3VLBackend,
    SupportedTask,
    VisualGate,
    VisualGateConfig,
)


def response(task, bbox=(0.2, 0.2, 0.8, 0.8), **overrides):
    x1, y1, x2, y2 = bbox
    value = {
        "schema_version": "revo3_planner_v1",
        "status": "READY",
        "task": task.value,
        "bbox": list(bbox),
        "area": (x2 - x1) * (y2 - y1),
        "confidence": 0.95,
        "target_present": True,
        "near_ready": True,
        "compatible": True,
        "ambiguity": {"ambiguous": False},
        "instruction": f"Execute the {task.value} task.",
    }
    value.update(overrides)
    return value


@pytest.mark.parametrize("task", list(SupportedTask))
def test_all_four_tasks_are_supported(task):
    backend = MockPlannerBackend(scripted_responses=[response(task)])
    planner = AskToClarifyPlanner(backend)
    decision = planner.plan(
        PlannerRequest(emg_action="CLOSE", images=(object(),), timestamp_ns=10)
    )
    assert decision.status == PlannerStatus.READY
    assert decision.task == task
    assert decision.area == pytest.approx(0.36)
    assert decision.instruction == f"Execute the {task.value} task."


def test_qwen_backend_is_lazy_and_does_not_load_on_construction():
    backend = Qwen3VLBackend(local_files_only=True)
    assert not backend.is_loaded


def test_non_close_action_is_rejected_before_backend_call():
    with pytest.raises(ValueError, match="only a latched CLOSE"):
        PlannerRequest(emg_action="OPEN", images=(object(),), timestamp_ns=10)


def test_ask_to_clarify_runs_second_turn_and_returns_resolved_instruction():
    ambiguous = response(
        SupportedTask.BOTTLE,
        status="ASK_CLARIFY",
        ambiguity={
            "ambiguous": True,
            "reason": "two plausible objects",
            "candidates": ["bottle", "phone"],
            "question": "Bottle or phone?",
        },
        instruction="",
    )
    resolved = response(SupportedTask.PHONE)
    backend = MockPlannerBackend(scripted_responses=[ambiguous, resolved])
    planner = AskToClarifyPlanner(backend)
    request = PlannerRequest(
        emg_action="CLOSE",
        images=("frame-0", "frame-1"),
        timestamp_ns=100,
        clarification_answer="the phone",
    )
    decision = planner.plan(request)
    assert decision.status == PlannerStatus.READY
    assert decision.task == SupportedTask.PHONE
    assert len(backend.calls) == 2
    assert backend.calls[1]["conversation"][-1]["content"] == "the phone"


def test_backend_json_must_not_have_trailing_commentary():
    payload = json.dumps(response(SupportedTask.BOTTLE)) + " extra words"
    planner = AskToClarifyPlanner(MockPlannerBackend(scripted_responses=[payload]))
    with pytest.raises(ValueError, match="trailing"):
        planner.plan(PlannerRequest(emg_action="CLOSE", images=(1,), timestamp_ns=1))


def test_planner_schema_mismatch_is_rejected():
    bad = response(SupportedTask.BOTTLE, schema_version="old_schema")
    planner = AskToClarifyPlanner(MockPlannerBackend(scripted_responses=[bad]))
    with pytest.raises(ValueError, match="schema mismatch"):
        planner.plan(PlannerRequest("CLOSE", (1,), 1))


def test_visual_gate_requires_large_centered_target_for_consecutive_updates():
    planner = AskToClarifyPlanner(
        MockPlannerBackend(scripted_responses=[response(SupportedTask.BOTTLE)] * 2)
    )
    gate = VisualGate(
        VisualGateConfig(
            min_area=0.15,
            max_center_distance=0.25,
            min_consecutive_ready=2,
        )
    )
    first = planner.plan(PlannerRequest("CLOSE", (1,), 100))
    second = planner.plan(PlannerRequest("CLOSE", (2,), 110))
    first_result = gate.update(first, now_ns=100)
    second_result = gate.update(second, now_ns=110)
    assert not first_result.ready
    assert first_result.reason == "awaiting_consecutive_ready"
    assert second_result.ready
    assert second_result.consecutive_ready == 2


def test_visual_gate_rejects_small_or_future_target():
    small = AskToClarifyPlanner(
        MockPlannerBackend(
            scripted_responses=[response(SupportedTask.BOTTLE, bbox=(0.45, 0.45, 0.55, 0.55))]
        )
    ).plan(PlannerRequest("CLOSE", (1,), 100))
    gate = VisualGate(VisualGateConfig(min_consecutive_ready=1))
    assert gate.update(small, now_ns=100).reason == "target_too_small"

    normal = AskToClarifyPlanner(
        MockPlannerBackend(scripted_responses=[response(SupportedTask.BOTTLE)])
    ).plan(PlannerRequest("CLOSE", (1,), 200))
    assert gate.update(normal, now_ns=199).reason == "future_planner_decision"


def test_visual_gate_does_not_count_the_same_planner_result_twice():
    decision = AskToClarifyPlanner(
        MockPlannerBackend(scripted_responses=[response(SupportedTask.BOTTLE)])
    ).plan(PlannerRequest("CLOSE", (1,), 100))
    gate = VisualGate(VisualGateConfig(min_consecutive_ready=2))
    assert not gate.update(decision, now_ns=100).ready
    duplicate = gate.update(decision, now_ns=101)
    assert not duplicate.ready
    assert duplicate.reason == "non_increasing_planner_timestamp"
