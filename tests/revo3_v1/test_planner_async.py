import threading
import time
import queue
from dataclasses import replace

import numpy as np
import pytest

from revo3_v1.emg import EMGPrimitive
from revo3_v1.planner import (
    AskToClarifyPlanner,
    AsyncPlannerWorker,
    MockPlannerBackend,
    PlannerContextBuffer,
    PlannerContextFrame,
    PlannerWorkItem,
    PlannerWorkResult,
    planner_scene_signature,
)


def context_request(timestamp=300):
    buffer = PlannerContextBuffer()
    for sequence, capture in enumerate((100, 200, timestamp), 1):
        buffer.append(
            PlannerContextFrame(
                sequence,
                capture,
                np.zeros((8, 8, 3), dtype=np.uint8),
                np.zeros((4, 4, 3), dtype=np.uint8),
            ),
            now_ns=timestamp,
        )
    return buffer.build_request(
        primitive=EMGPrimitive.POWER_GRASP,
        now_ns=timestamp,
    )


def test_context_buffer_requires_three_distinct_causal_rectified_captures():
    buffer = PlannerContextBuffer()
    frame = PlannerContextFrame(1, 100, object(), object())
    buffer.append(frame, now_ns=100)
    with pytest.raises(ValueError, match="sequence"):
        buffer.append(frame, now_ns=100)
    with pytest.raises(RuntimeError, match="three"):
        buffer.build_request(primitive="POWER_GRASP", now_ns=100)
    with pytest.raises(ValueError, match="future"):
        buffer.append(PlannerContextFrame(2, 201, object(), object()), now_ns=200)


class BlockingBackend(MockPlannerBackend):
    def __init__(self, entered, release):
        super().__init__()
        self.entered = entered
        self.release = release

    def generate(self, **kwargs):
        self.entered.set()
        if not self.release.wait(timeout=2.0):
            raise TimeoutError("test backend release timeout")
        return super().generate(**kwargs)


class _FailFirstPlannerSentinelQueue:
    def __init__(self, delegate):
        self.delegate = delegate
        self.failed = False

    def get(self):
        return self.delegate.get()

    def put_nowait(self, value):
        if value is None and not self.failed:
            self.failed = True
            raise queue.Full
        return self.delegate.put_nowait(value)

    def empty(self):
        return self.delegate.empty()


def test_worker_never_blocks_caller_and_drops_obsolete_generation():
    entered, release = threading.Event(), threading.Event()
    worker = AsyncPlannerWorker(AskToClarifyPlanner(BlockingBackend(entered, release)))
    request = context_request()
    item = PlannerWorkItem(1, "event-1", EMGPrimitive.POWER_GRASP, 1, request, 300)
    started = time.perf_counter()
    assert worker.submit(item)
    assert time.perf_counter() - started < 0.1
    assert entered.wait(timeout=1.0)
    assert worker.poll(
        expected_generation=1,
        expected_event_id="event-1",
        expected_primitive="POWER_GRASP",
        expected_task_version=1,
        now_ns=300,
        max_age_ns=1000,
    ) is None
    assert not worker.submit(item)
    release.set()
    deadline = time.monotonic() + 1.0
    while worker.busy and time.monotonic() < deadline:
        time.sleep(0.005)
    assert worker.poll(
        expected_generation=2,
        expected_event_id="event-1",
        expected_primitive="POWER_GRASP",
        expected_task_version=1,
        now_ns=300,
        max_age_ns=1000,
    ) is None
    assert worker.last_discard_reason == "generation_changed"
    worker.close(wait=True)


def test_planner_close_reports_actual_idle_and_inflight_thread_exit():
    idle = AsyncPlannerWorker(AskToClarifyPlanner(MockPlannerBackend()))
    assert idle.close(timeout_s=1.0)
    assert idle.close(timeout_s=1.0)

    entered, release = threading.Event(), threading.Event()
    worker = AsyncPlannerWorker(AskToClarifyPlanner(BlockingBackend(entered, release)))
    request = context_request()
    assert worker.submit(
        PlannerWorkItem(1, "event-close", EMGPrimitive.POWER_GRASP, 1, request, 300)
    )
    assert entered.wait(timeout=1.0)
    assert not worker.close(timeout_s=0.01)
    release.set()
    assert worker.close(timeout_s=1.0)
    assert worker.close(timeout_s=1.0)


def test_planner_close_retries_sentinel_after_queue_full_race():
    entered, release = threading.Event(), threading.Event()
    worker = AsyncPlannerWorker(AskToClarifyPlanner(BlockingBackend(entered, release)))
    request = context_request()
    assert worker.submit(
        PlannerWorkItem(1, "event-single-close", EMGPrimitive.POWER_GRASP, 1, request, 300)
    )
    assert entered.wait(timeout=1.0)
    worker._requests = _FailFirstPlannerSentinelQueue(worker._requests)
    timer = threading.Timer(0.02, release.set)
    timer.start()
    try:
        assert worker.close(timeout_s=1.0)
    finally:
        timer.cancel()


def test_worker_returns_only_matching_versioned_result():
    worker = AsyncPlannerWorker(AskToClarifyPlanner(MockPlannerBackend()))
    request = context_request()
    assert worker.submit(
        PlannerWorkItem(7, "event-7", EMGPrimitive.POWER_GRASP, 3, request, 300)
    )
    deadline = time.monotonic() + 1.0
    result = None
    while result is None and time.monotonic() < deadline:
        result = worker.poll(
            expected_generation=7,
            expected_event_id="event-7",
            expected_primitive="POWER_GRASP",
            expected_task_version=3,
            now_ns=350,
            max_age_ns=100,
        )
        if result is None:
            time.sleep(0.005)
    assert result is not None and result.decision is not None
    assert result.decision.timestamp_ns == request.timestamp_ns
    worker.close(wait=True)


def _inject_result(
    worker: AsyncPlannerWorker,
    *,
    submitted_at_ns: int,
    completed_at_ns: int,
) -> None:
    request = context_request(timestamp=submitted_at_ns)
    decision = AskToClarifyPlanner(MockPlannerBackend()).plan(request)
    worker._results.put_nowait(
        PlannerWorkResult(
            generation=1,
            event_id="event-1",
            primitive=EMGPrimitive.POWER_GRASP,
            task_version=0,
            request_timestamp_ns=request.timestamp_ns,
            submitted_at_ns=submitted_at_ns,
            completed_at_ns=completed_at_ns,
            decision=replace(decision, produced_at_ns=completed_at_ns),
            scene_signature=planner_scene_signature(
                request.full_view_history[-1], request.center_view
            ),
        )
    )


def _poll_injected(worker: AsyncPlannerWorker, *, now_ns: int, scene=None):
    return worker.poll(
        expected_generation=1,
        expected_event_id="event-1",
        expected_primitive=EMGPrimitive.POWER_GRASP,
        expected_task_version=0,
        now_ns=now_ns,
        result_ttl_ns=1_000_000_000,
        planner_sla_ns=20_000_000_000,
        current_scene_signature=scene,
    )


def test_fifteen_second_qwen_result_is_accepted_by_produced_freshness():
    worker = AsyncPlannerWorker(AskToClarifyPlanner(MockPlannerBackend()))
    _inject_result(worker, submitted_at_ns=1_000_000_000, completed_at_ns=16_000_000_000)
    same_scene = planner_scene_signature(
        np.zeros((8, 8, 3), np.uint8), np.zeros((4, 4, 3), np.uint8)
    )
    result = _poll_injected(worker, now_ns=16_100_000_000, scene=same_scene)
    assert result is not None and result.decision is not None
    assert result.decision.timestamp_ns == 1_000_000_000
    assert result.decision.produced_at_ns == 16_000_000_000
    worker.close(wait=True)


def test_scene_signature_allows_small_sensor_noise_but_rejects_changed_scene():
    worker = AsyncPlannerWorker(AskToClarifyPlanner(MockPlannerBackend()))
    _inject_result(worker, submitted_at_ns=1_000_000_000, completed_at_ns=15_000_000_000)
    noisy = planner_scene_signature(
        np.full((8, 8, 3), 2, np.uint8), np.full((4, 4, 3), 2, np.uint8)
    )
    assert _poll_injected(worker, now_ns=15_100_000_000, scene=noisy) is not None
    worker.close(wait=True)

    worker = AsyncPlannerWorker(AskToClarifyPlanner(MockPlannerBackend()))
    _inject_result(worker, submitted_at_ns=1_000_000_000, completed_at_ns=15_000_000_000)
    changed = planner_scene_signature(
        np.full((8, 8, 3), 255, np.uint8),
        np.full((4, 4, 3), 255, np.uint8),
    )
    assert _poll_injected(worker, now_ns=15_100_000_000, scene=changed) is None
    assert worker.last_discard_reason == "planner_scene_changed"
    worker.close(wait=True)


def test_qwen_result_beyond_explicit_sla_is_rejected():
    worker = AsyncPlannerWorker(AskToClarifyPlanner(MockPlannerBackend()))
    _inject_result(worker, submitted_at_ns=1_000_000_000, completed_at_ns=21_000_000_001)
    assert _poll_injected(worker, now_ns=21_100_000_001) is None
    assert worker.last_discard_reason == "planner_sla_exceeded"
    worker.close(wait=True)


def test_qwen_result_with_stale_produced_timestamp_is_rejected():
    worker = AsyncPlannerWorker(AskToClarifyPlanner(MockPlannerBackend()))
    _inject_result(worker, submitted_at_ns=1_000_000_000, completed_at_ns=16_000_000_000)
    assert _poll_injected(worker, now_ns=17_000_000_001) is None
    assert worker.last_discard_reason == "planner_result_stale"
    worker.close(wait=True)
