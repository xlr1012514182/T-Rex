import json

import pytest

from revo3_v1.planner import (
    AskToClarifyPlanner,
    MockPlannerBackend,
    PlannerRequest,
    PlannerStatus,
    Qwen3VLBackend,
    SupportedTask,
    TASK_GRASP_PRIMITIVES,
    VisualGate,
    VisualGateConfig,
)


def response(task, region=(0.5, 0.5, 0.6, 0.6), primitive=None, **overrides):
    primitive = primitive or TASK_GRASP_PRIMITIVES[task].value
    style = {"POWER_GRASP": "power", "PRECISION_GRASP": "precision", "LATERAL_GRASP": "lateral"}[primitive]
    value = {
        "schema_version": "planner_v1",
        "status": "READY",
        "primitive": primitive,
        "target_category": task.value,
        "target_part": "body",
        "grasp_style": style,
        "target_region": list(region),
        "confidence": 0.95,
        "target_present": True,
        "near_ready": True,
        "center_ready": True,
        "compatible": True,
        "ambiguous": False,
        "ready_frame_count": 3,
        "reason_code": "READY",
        "instruction": f"Grasp the centered {task.value} body using a {style} grasp and hold it securely.",
    }
    value.update(overrides)
    return value


def aligned_request(primitive="POWER_GRASP", timestamp=300, **kwargs):
    return PlannerRequest.from_aligned_views(
        primitive=primitive,
        full_view_history=("full-0", "full-1", "full-2"),
        center_view="center",
        full_view_timestamps_ns=(100, 200, timestamp),
        center_timestamp_ns=timestamp,
        **kwargs,
    )


@pytest.mark.parametrize("task", list(SupportedTask))
def test_all_four_tasks_are_supported(task):
    primitive = TASK_GRASP_PRIMITIVES[task].value
    planner = AskToClarifyPlanner(MockPlannerBackend(scripted_responses=[response(task)]))
    decision = planner.plan(aligned_request(primitive))
    assert decision.status == PlannerStatus.READY
    assert decision.task == task
    assert decision.area == pytest.approx(0.36)
    assert decision.primitive == TASK_GRASP_PRIMITIVES[task]


def test_aligned_request_requires_three_full_frames_and_same_current_timestamp():
    with pytest.raises(ValueError, match="3 full frames"):
        PlannerRequest.from_aligned_views(
            primitive="POWER_GRASP",
            full_view_history=(1, 2),
            center_view=3,
            full_view_timestamps_ns=(1, 2),
            center_timestamp_ns=2,
        )
    with pytest.raises(ValueError, match="share one timestamp"):
        PlannerRequest.from_aligned_views(
            primitive="POWER_GRASP",
            full_view_history=(1, 2, 3),
            center_view=4,
            full_view_timestamps_ns=(1, 2, 3),
            center_timestamp_ns=4,
        )


def test_qwen_backend_is_lazy_and_uses_bounded_output():
    backend = Qwen3VLBackend(local_files_only=True)
    assert not backend.is_loaded
    assert backend.max_new_tokens == 384


def test_release_is_rejected_before_backend_call():
    with pytest.raises(ValueError, match="grasp primitive"):
        PlannerRequest(emg_action="RELEASE", images=(object(),), timestamp_ns=10)


def test_primitive_mismatch_is_repaired_once_then_invalid():
    bad = response(SupportedTask.BOTTLE, primitive="PRECISION_GRASP")
    backend = MockPlannerBackend(scripted_responses=[bad, bad])
    decision = AskToClarifyPlanner(backend).plan(aligned_request("POWER_GRASP"))
    assert decision.status == PlannerStatus.INVALID
    assert len(backend.calls) == 2
    assert "primitive" in decision.ambiguity.reason


def test_invalid_json_retries_once_and_accepts_repair():
    backend = MockPlannerBackend(
        scripted_responses=["not json", response(SupportedTask.BOTTLE)]
    )
    decision = AskToClarifyPlanner(backend).plan(aligned_request())
    assert decision.status == PlannerStatus.READY
    assert len(backend.calls) == 2
    assert "REPAIR" in backend.calls[1]["prompt"]


def test_ask_to_clarify_runs_second_turn_and_preserves_primitive():
    ambiguous = response(
        SupportedTask.BOTTLE,
        primitive="PRECISION_GRASP",
        status="ASK_CLARIFY",
        ambiguous=True,
        ask_clarify=True,
        instruction="",
        reason_code="MULTIPLE_TARGETS",
    )
    # Flat ambiguity does not carry a question; use the supported nested form.
    ambiguous["ambiguity"] = {
        "ambiguous": True,
        "reason": "two plausible objects",
        "candidates": ["bottle", "phone"],
        "question": "Bottle or phone?",
    }
    resolved = response(SupportedTask.PHONE)
    backend = MockPlannerBackend(scripted_responses=[ambiguous, resolved])
    request = aligned_request("PRECISION_GRASP", clarification_answer="the phone")
    decision = AskToClarifyPlanner(backend).plan(request)
    assert decision.status == PlannerStatus.READY
    assert decision.task == SupportedTask.PHONE
    assert decision.primitive.value == "PRECISION_GRASP"
    assert len(backend.calls) == 2


def test_ready_decision_rejects_task_primitive_pair_outside_frozen_mapping():
    invalid = response(SupportedTask.PHONE, primitive="POWER_GRASP")
    decision = AskToClarifyPlanner(
        MockPlannerBackend(scripted_responses=[invalid, invalid])
    ).plan(aligned_request("POWER_GRASP"))
    assert decision.status == PlannerStatus.INVALID
    assert "task/primitive" in decision.ambiguity.reason


def test_trailing_commentary_twice_fails_closed():
    payload = json.dumps(response(SupportedTask.BOTTLE)) + " extra words"
    backend = MockPlannerBackend(scripted_responses=[payload, payload])
    decision = AskToClarifyPlanner(backend).plan(aligned_request())
    assert decision.status == PlannerStatus.INVALID
    assert "trailing" in decision.ambiguity.reason


def test_not_ready_zero_area_sentinel_is_a_valid_non_ready_decision():
    value = response(SupportedTask.BOTTLE)
    value.update({
        "status": "NOT_READY",
        "target_category": None,
        "target_region": [0.5, 0.5, 0.0, 0.0],
        "target_present": False,
        "near_ready": False,
        "center_ready": False,
        "compatible": False,
        "ready_frame_count": 0,
        "instruction": "",
        "reason_code": "TOO_FAR",
    })
    decision = AskToClarifyPlanner(
        MockPlannerBackend(scripted_responses=[value])
    ).plan(aligned_request())
    assert decision.status == PlannerStatus.NOT_READY
    assert decision.bbox is None


def test_ask_clarify_schema_parses_without_ready_target_region():
    value = response(SupportedTask.BOTTLE)
    value.update({
        "status": "ASK_CLARIFY",
        "target_region": None,
        "target_present": False,
        "near_ready": False,
        "center_ready": False,
        "compatible": False,
        "ready_frame_count": 0,
        "instruction": "",
        "ambiguity": {
            "ambiguous": True,
            "reason": "multiple objects",
            "candidates": ["bottle", "phone"],
            "question": "Bottle or phone?",
        },
    })
    decision = AskToClarifyPlanner(
        MockPlannerBackend(scripted_responses=[value])
    ).plan(aligned_request())
    assert decision.status == PlannerStatus.ASK_CLARIFY


def test_visual_gate_uses_planner_three_frame_evidence_without_tracker():
    decision = AskToClarifyPlanner(
        MockPlannerBackend(scripted_responses=[response(SupportedTask.BOTTLE)])
    ).plan(aligned_request())
    gate = VisualGate(VisualGateConfig(min_consecutive_ready=2))
    result = gate.update(decision, now_ns=300)
    assert result.ready
    assert result.consecutive_ready == 1


def test_visual_gate_rejects_small_future_or_insufficient_history():
    small = AskToClarifyPlanner(
        MockPlannerBackend(scripted_responses=[response(SupportedTask.BOTTLE, region=(0.5, 0.5, 0.1, 0.1))])
    ).plan(aligned_request())
    gate = VisualGate(VisualGateConfig(min_consecutive_ready=2))
    assert gate.update(small, now_ns=300).reason == "target_too_small"

    insufficient = AskToClarifyPlanner(
        MockPlannerBackend(scripted_responses=[response(SupportedTask.BOTTLE, ready_frame_count=1)])
    ).plan(aligned_request(timestamp=400))
    assert gate.update(insufficient, now_ns=400).reason == "insufficient_ready_frames"

    normal = AskToClarifyPlanner(
        MockPlannerBackend(scripted_responses=[response(SupportedTask.BOTTLE)])
    ).plan(aligned_request(timestamp=500))
    assert gate.update(normal, now_ns=499).reason == "future_planner_decision"
