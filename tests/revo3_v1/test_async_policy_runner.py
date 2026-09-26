from __future__ import annotations

import threading
import time
import queue

import numpy as np

from revo3_v1.policy import (
    AsyncPolicyState,
    AsyncTReXPolicyRunner,
    PolicyObservation,
    TaskKey,
    TReXRevoPolicyAdapter,
)


class BlockingBackend:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def _run(self, observation: PolicyObservation) -> np.ndarray:
        self.entered.set()
        if not self.release.wait(2.0):
            raise TimeoutError("fixture did not release")
        return np.repeat(observation.q_rad[None], 16, axis=0)

    def slow_and_fast(self, observation: PolicyObservation) -> np.ndarray:
        return self._run(observation)

    def slow(self, observation: PolicyObservation) -> np.ndarray:
        return self._run(observation)

    def fast(self, observation, cached_chunk, chunk_offset):
        del observation, chunk_offset
        return np.asarray(cached_chunk, dtype=np.float32)


class CloseTrackingBackend(BlockingBackend):
    def __init__(self, *, fail_close: bool = False) -> None:
        super().__init__()
        self.close_calls = 0
        self.fail_close = fail_close

    def close(self):
        self.close_calls += 1
        if self.fail_close:
            raise RuntimeError("socket close failed")


class _FailFirstSentinelQueue:
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


def observation(*, lease: str = "lease-a", timestamp_ns: int = 100) -> PolicyObservation:
    instruction = "Grasp the centered bottle using a power grasp."
    task_key = TaskKey.from_instruction(
        task_id="bottle",
        task_version=1,
        instruction=instruction,
        lease_id=lease,
        version_fingerprint="versions",
    )
    current = np.zeros((5, 6), np.float32)
    return PolicyObservation(
        timestamp_ns=timestamp_ns,
        state_timestamp_ns=timestamp_ns,
        rgb_timestamp_ns=timestamp_ns,
        tactile_timestamp_ns=timestamp_ns,
        q_rad=np.zeros(21, np.float32),
        tactile_f6=current,
        tactile_history_f6=np.zeros((16, 5, 6), np.float32),
        tactile_history_timestamps_ns=np.arange(timestamp_ns - 15, timestamp_ns + 1),
        tactile_history_sequences=np.arange(16),
        instruction=instruction,
        task_key=task_key,
        lease_expires_at_ns=timestamp_ns + 2_000_000_000,
        tactile_profile="ablation_force6d_only",
    )


def wait_for_state(runner, key, expected, *, now_ns=101):
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        result = runner.poll(now_ns=now_ns, expected_task_key=key)
        if result.state is expected:
            return result
        time.sleep(0.001)
    raise AssertionError(f"did not observe {expected}")


def test_gpu_block_does_not_block_control_and_backpressure_is_bounded() -> None:
    backend = BlockingBackend()
    runner = AsyncTReXPolicyRunner(
        TReXRevoPolicyAdapter(backend), request_timeout_ns=1_000_000_000
    )
    obs = observation()
    assert runner.submit_if_due(global_step=0, observation=obs, now_ns=100).state is AsyncPolicyState.PENDING
    assert backend.entered.wait(0.5)
    started = time.perf_counter()
    assert runner.poll(now_ns=101, expected_task_key=obs.task_key).state is AsyncPolicyState.PENDING
    assert time.perf_counter() - started < 0.05
    assert runner.submit_if_due(global_step=0, observation=obs, now_ns=102).reason == "worker_backpressure"
    backend.release.set()
    assert wait_for_state(runner, obs.task_key, AsyncPolicyState.READY).chunk is not None
    assert runner.close()


def test_timeout_and_reset_race_discard_old_generation_before_new_lease() -> None:
    backend = BlockingBackend()
    runner = AsyncTReXPolicyRunner(
        TReXRevoPolicyAdapter(backend), request_timeout_ns=10
    )
    old = observation()
    runner.submit_if_due(global_step=0, observation=old, now_ns=100)
    assert backend.entered.wait(0.5)
    timed_out = runner.poll(now_ns=111, expected_task_key=old.task_key)
    assert timed_out.state is AsyncPolicyState.HOLD
    assert timed_out.reason == "policy_request_timeout"
    runner.reset("new_lease")
    backend.release.set()
    deadline = time.monotonic() + 1.0
    while True:
        discarded = runner.poll(now_ns=112, expected_task_key=old.task_key)
        if "discarded" in discarded.reason:
            break
        if time.monotonic() >= deadline:
            raise AssertionError("old generation did not unwind")
        time.sleep(0.001)

    backend.entered.clear()
    backend.release.clear()
    new = observation(lease="lease-b", timestamp_ns=200)
    assert runner.submit_if_due(global_step=0, observation=new, now_ns=200).state is AsyncPolicyState.PENDING
    assert backend.entered.wait(0.5)
    backend.release.set()
    accepted = wait_for_state(runner, new.task_key, AsyncPolicyState.READY, now_ns=201)
    assert accepted.chunk.task_key == new.task_key
    assert backend.reset_calls >= 2
    assert runner.close()


def test_clean_worker_shutdown_closes_owned_policy_backend_once() -> None:
    backend = CloseTrackingBackend()
    backend.release.set()
    runner = AsyncTReXPolicyRunner(TReXRevoPolicyAdapter(backend))
    obs = observation()
    runner.submit_if_due(global_step=0, observation=obs, now_ns=100)
    wait_for_state(runner, obs.task_key, AsyncPolicyState.READY)
    assert runner.close()
    assert runner.close()
    assert backend.close_calls == 1


def test_policy_backend_close_failure_is_not_reported_clean() -> None:
    backend = CloseTrackingBackend(fail_close=True)
    runner = AsyncTReXPolicyRunner(TReXRevoPolicyAdapter(backend))
    assert not runner.close()
    assert backend.close_calls == 1


def test_blocked_worker_is_never_closed_concurrently_with_backend_call() -> None:
    backend = CloseTrackingBackend()
    runner = AsyncTReXPolicyRunner(TReXRevoPolicyAdapter(backend))
    obs = observation()
    runner.submit_if_due(global_step=0, observation=obs, now_ns=100)
    assert backend.entered.wait(0.5)
    assert not runner.close(timeout_s=0.01)
    assert backend.close_calls == 0
    backend.release.set()
    assert runner.close(timeout_s=1.0)
    assert backend.close_calls == 1


def test_close_retries_sentinel_after_initial_bounded_queue_full_race() -> None:
    backend = CloseTrackingBackend()
    runner = AsyncTReXPolicyRunner(TReXRevoPolicyAdapter(backend))
    obs = observation()
    runner.submit_if_due(global_step=0, observation=obs, now_ns=100)
    assert backend.entered.wait(0.5)
    runner._jobs = _FailFirstSentinelQueue(runner._jobs)
    assert not runner.close(timeout_s=0.01)
    assert backend.close_calls == 0
    backend.release.set()
    time.sleep(0.02)
    assert runner.close(timeout_s=1.0)
    assert backend.close_calls == 1


def test_single_close_retries_sentinel_until_inflight_job_releases() -> None:
    backend = CloseTrackingBackend()
    runner = AsyncTReXPolicyRunner(TReXRevoPolicyAdapter(backend))
    obs = observation()
    runner.submit_if_due(global_step=0, observation=obs, now_ns=100)
    assert backend.entered.wait(0.5)
    runner._jobs = _FailFirstSentinelQueue(runner._jobs)
    timer = threading.Timer(0.02, backend.release.set)
    timer.start()
    try:
        assert runner.close(timeout_s=1.0)
    finally:
        timer.cancel()
    assert backend.close_calls == 1


def test_recommended_close_budget_covers_request_and_transport_deadlines() -> None:
    backend = CloseTrackingBackend()
    backend.timeout_ms = 5_000
    runner = AsyncTReXPolicyRunner(
        TReXRevoPolicyAdapter(backend), request_timeout_ns=2_000_000_000
    )
    assert runner.recommended_close_timeout_s >= 5.0
    backend.release.set()
    assert runner.close(timeout_s=runner.recommended_close_timeout_s)
