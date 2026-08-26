"""Non-blocking single-worker boundary for remote T-Rex inference.

The 100 Hz writer/control thread never calls the GPU backend.  One daemon
worker owns the adapter and its server/cache identity; the control side owns
temporal aggregation and rejects results from old generations/task leases.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
import queue
import threading
import time
from typing import Optional

from .adapter import TReXRevoPolicyAdapter
from .aggregation import ActionTemporalAggregator, TemporalAggregationError
from .contracts import ActionChunk, InferenceMode, PolicyObservation, PolicyRequest, TaskKey
from .schedule import MAIN_ALIGNED_SCHEDULE, PolicySchedule
from .zmq_backend import TReXWireProtocolError


class AsyncPolicyState(str, Enum):
    IDLE = "IDLE"
    PENDING = "PENDING"
    READY = "READY"
    HOLD = "HOLD"
    ABORT = "ABORT"
    CLOSED = "CLOSED"


@dataclass(frozen=True)
class AsyncPolicyPoll:
    state: AsyncPolicyState
    reason: str
    chunk: Optional[ActionChunk] = None


@dataclass(frozen=True)
class _Job:
    generation: int
    global_step: int
    request: PolicyRequest
    submitted_ns: int


@dataclass(frozen=True)
class _Outcome:
    generation: int
    task_key: TaskKey
    submitted_ns: int
    chunk: Optional[ActionChunk]
    error: Optional[BaseException]


class AsyncTReXPolicyRunner:
    """One outstanding request, one bounded queue, generation-safe results."""

    def __init__(
        self,
        adapter: TReXRevoPolicyAdapter,
        *,
        schedule: PolicySchedule = MAIN_ALIGNED_SCHEDULE,
        aggregator: ActionTemporalAggregator | None = None,
        request_timeout_ns: int = 2_000_000_000,
    ) -> None:
        if request_timeout_ns <= 0:
            raise ValueError("request_timeout_ns must be positive")
        self.adapter = adapter
        self.schedule = schedule
        self.aggregator = aggregator or ActionTemporalAggregator(k=0.0)
        self.request_timeout_ns = int(request_timeout_ns)
        self._jobs: queue.Queue[_Job | None] = queue.Queue(maxsize=1)
        self._outcomes: queue.Queue[_Outcome] = queue.Queue(maxsize=1)
        self._generation = 0
        self._worker_generation = -1
        self._busy = False
        self._submitted_ns: int | None = None
        self._timed_out_generation: int | None = None
        self._closed = False
        self._backend_close_attempted = False
        self._backend_close_clean = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._worker_main,
            name="revo3-trex-policy-worker",
            daemon=True,
        )
        self._thread.start()

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    @property
    def recommended_close_timeout_s(self) -> float:
        """Bounded join budget that cannot undercut the configured request.

        The ZMQ transport can legitimately remain inside its own bounded
        receive timeout after shutdown starts.  Joining for a fixed one
        second would therefore misreport an ordinary 1--5 second request as
        an unclean orphan.  Keep a small scheduler margin while still making
        the total shutdown budget explicit and finite.
        """

        transport_timeout_s = max(
            0.0,
            float(getattr(self.adapter.backend, "timeout_ms", 0)) / 1000.0,
        )
        request_timeout_s = self.request_timeout_ns / 1_000_000_000.0
        return max(1.0, request_timeout_s, transport_timeout_s) + 0.25

    def _worker_main(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            try:
                if job.generation != self._worker_generation:
                    self.adapter.reset(f"async_generation_{job.generation}")
                    self._worker_generation = job.generation
                chunk = self.adapter.infer(
                    job.request,
                    global_start_step=job.global_step - job.request.chunk_offset,
                    now_ns=job.submitted_ns,
                )
                outcome = _Outcome(
                    job.generation,
                    job.request.observation.task_key,
                    job.submitted_ns,
                    chunk,
                    None,
                )
            except BaseException as exc:  # delivered to control side, never hidden
                outcome = _Outcome(
                    job.generation,
                    job.request.observation.task_key,
                    job.submitted_ns,
                    None,
                    exc,
                )
            # Exactly one outstanding job means one result slot is sufficient.
            self._outcomes.put(outcome)

    def reset(self, reason: str) -> None:
        del reason
        with self._lock:
            if self._closed:
                return
            self._generation += 1
            if not self._busy:
                self._timed_out_generation = None
            self.aggregator.clear()
        # A blocked backend cannot be interrupted safely.  Its eventual
        # outcome carries the old generation and will be discarded.  The next
        # accepted job performs adapter/server-cache reset in the owner thread.

    def submit_if_due(
        self,
        *,
        global_step: int,
        observation: PolicyObservation,
        now_ns: int,
    ) -> AsyncPolicyPoll:
        if global_step < 0:
            raise ValueError("global_step must be non-negative")
        offset = global_step % self.schedule.chunk_size
        mode = self.schedule.mode_at_chunk_offset(offset)
        if mode is InferenceMode.NONE:
            return AsyncPolicyPoll(AsyncPolicyState.IDLE, "not_due")
        tactile_age_ns = int(now_ns) - int(observation.tactile_timestamp_ns)
        if tactile_age_ns < 0:
            return AsyncPolicyPoll(AsyncPolicyState.ABORT, "future_touch")
        if mode is InferenceMode.FAST and tactile_age_ns > self.adapter.cache.max_tactile_age_ns:
            return AsyncPolicyPoll(AsyncPolicyState.HOLD, "stale_touch_fast_disabled")
        with self._lock:
            if self._closed:
                return AsyncPolicyPoll(AsyncPolicyState.CLOSED, "closed")
            if self._busy:
                return AsyncPolicyPoll(AsyncPolicyState.HOLD, "worker_backpressure")
            generation = self._generation
            job = _Job(
                generation,
                global_step,
                PolicyRequest(mode=mode, chunk_offset=offset, observation=observation),
                int(now_ns),
            )
            try:
                self._jobs.put_nowait(job)
            except queue.Full:
                return AsyncPolicyPoll(AsyncPolicyState.HOLD, "worker_queue_full")
            self._busy = True
            self._submitted_ns = int(now_ns)
            return AsyncPolicyPoll(AsyncPolicyState.PENDING, "submitted")

    def poll(
        self,
        *,
        now_ns: int,
        expected_task_key: TaskKey,
    ) -> AsyncPolicyPoll:
        with self._lock:
            if self._closed:
                return AsyncPolicyPoll(AsyncPolicyState.CLOSED, "closed")
            generation = self._generation
            submitted = self._submitted_ns
            busy = self._busy
            timeout_already_latched = self._timed_out_generation is not None
        try:
            outcome = self._outcomes.get_nowait()
        except queue.Empty:
            if timeout_already_latched:
                return AsyncPolicyPoll(
                    AsyncPolicyState.HOLD, "policy_timeout_waiting_for_worker_unwind"
                )
            if busy and submitted is not None and int(now_ns) - submitted > self.request_timeout_ns:
                with self._lock:
                    self._timed_out_generation = generation
                    self._generation += 1
                    self.aggregator.clear()
                return AsyncPolicyPoll(AsyncPolicyState.HOLD, "policy_request_timeout")
            return AsyncPolicyPoll(
                AsyncPolicyState.PENDING if busy else AsyncPolicyState.IDLE,
                "pending" if busy else "no_result",
            )

        with self._lock:
            self._busy = False
            self._submitted_ns = None
            current_generation = self._generation
            timed_out_generation = self._timed_out_generation
            if timed_out_generation == outcome.generation:
                self._timed_out_generation = None
        if outcome.generation != current_generation:
            return AsyncPolicyPoll(AsyncPolicyState.HOLD, "stale_generation_result_discarded")
        if outcome.task_key != expected_task_key:
            self.reset("task_key_mismatch")
            return AsyncPolicyPoll(AsyncPolicyState.ABORT, "task_key_mismatch")
        if outcome.error is not None:
            self.reset("worker_error")
            if isinstance(outcome.error, TReXWireProtocolError):
                return AsyncPolicyPoll(
                    AsyncPolicyState.ABORT,
                    f"policy_protocol_error:{type(outcome.error).__name__}",
                )
            return AsyncPolicyPoll(
                AsyncPolicyState.HOLD,
                f"policy_backend_error:{type(outcome.error).__name__}",
            )
        assert outcome.chunk is not None
        if int(now_ns) - outcome.chunk.generated_ns > self.request_timeout_ns:
            self.reset("stale_result")
            return AsyncPolicyPoll(AsyncPolicyState.HOLD, "stale_policy_result_discarded")
        # ``adapter.infer`` receives the submission clock so cache validation
        # stays deterministic while the backend is blocked.  At this control
        # boundary the response becomes usable only now; stamp that acceptance
        # time so TaskExecutive does not mistake ordinary GPU latency for an
        # already-expired action chunk.
        accepted_chunk = replace(outcome.chunk, generated_ns=int(now_ns))
        self.aggregator.add(accepted_chunk, now_ns=int(now_ns))
        return AsyncPolicyPoll(AsyncPolicyState.READY, "accepted", accepted_chunk)

    def target_for_step(self, *, global_step: int, task_key: TaskKey, now_ns: int):
        return self.aggregator.target_at(global_step, task_key=task_key, now_ns=now_ns)

    def interpolated_target_for_tick(
        self,
        *,
        executor_timestamp_ns: int,
        action_epoch_ns: int,
        task_key: TaskKey,
        now_ns: int,
    ):
        elapsed_ns = int(executor_timestamp_ns) - int(action_epoch_ns)
        if elapsed_ns < 0:
            raise ValueError("executor tick precedes action epoch")
        position = elapsed_ns / self.schedule.action_period_ns
        lower_step = int(math.floor(position))
        alpha = float(position - lower_step)
        lower = self.target_for_step(
            global_step=lower_step, task_key=task_key, now_ns=now_ns
        )
        if alpha <= 1e-12:
            return lower
        try:
            upper = self.target_for_step(
                global_step=lower_step + 1, task_key=task_key, now_ns=now_ns
            )
        except TemporalAggregationError:
            return lower
        return ((1.0 - alpha) * lower + alpha * upper).astype("float32")

    def close(self, *, timeout_s: float = 1.0) -> bool:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        with self._lock:
            if not self._closed:
                self._closed = True
                self._generation += 1
                self.aggregator.clear()
        deadline = time.monotonic() + float(timeout_s)
        while self._thread.is_alive():
            try:
                self._jobs.put_nowait(None)
            except queue.Full:
                # Retry within this same deadline after the worker consumes
                # the queued job; never require an external second close.
                pass
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            self._thread.join(min(0.01, remaining))
        if self._thread.is_alive():
            # A blocked backend still owns its transport.  Closing it here
            # would race the in-flight call, so report unclean instead.
            return False
        with self._lock:
            if self._backend_close_attempted:
                return self._backend_close_clean
            self._backend_close_attempted = True
        close_backend = getattr(self.adapter.backend, "close", None)
        try:
            clean = True if not callable(close_backend) else close_backend() is not False
        except BaseException:
            clean = False
        with self._lock:
            self._backend_close_clean = bool(clean)
        return bool(clean)


__all__ = ["AsyncPolicyPoll", "AsyncPolicyState", "AsyncTReXPolicyRunner"]
