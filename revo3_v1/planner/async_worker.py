"""Non-blocking Planner execution and causal three-frame context assembly.

The worker is deliberately not a lifecycle authority.  It only executes one
bounded Qwen request off the control thread and tags the result so the
``TaskExecutive`` caller can reject an obsolete generation, event, primitive,
or task version before supplying the decision to the state machine.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import queue
import threading
import time
from typing import Any, Deque, Optional, Protocol

import numpy as np

from revo3_v1.emg.primitives import EMGPrimitive, START_PRIMITIVES, normalize_emg_primitive

from .schema import PlannerDecision, PlannerRequest


@dataclass(frozen=True)
class PlannerSceneSignature:
    """Small deterministic appearance feature; not a detector or tracker."""

    feature: np.ndarray

    def __post_init__(self) -> None:
        value = np.asarray(self.feature, dtype=np.float32)
        if value.shape != (2, 16, 16, 3) or not np.isfinite(value).all():
            raise ValueError("planner scene signature must be finite [2,16,16,3]")
        object.__setattr__(self, "feature", value.copy())

    def distance(self, other: "PlannerSceneSignature") -> float:
        if not isinstance(other, PlannerSceneSignature):
            raise TypeError("other must be PlannerSceneSignature")
        return float(np.mean(np.abs(self.feature - other.feature)))


def planner_scene_signature(full_view: Any, center_view: Any) -> PlannerSceneSignature:
    features = []
    for name, raw in (("full", full_view), ("center", center_view)):
        value = np.asarray(raw)
        if value.ndim != 3 or value.shape[2] != 3 or value.size == 0:
            raise ValueError(f"planner {name} view must be non-empty HxWx3")
        if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
            raise ValueError(f"planner {name} view must contain finite numeric pixels")
        y = np.linspace(0, value.shape[0] - 1, 16).round().astype(np.int64)
        x = np.linspace(0, value.shape[1] - 1, 16).round().astype(np.int64)
        sampled = value[np.ix_(y, x, np.arange(3))].astype(np.float32)
        if np.issubdtype(value.dtype, np.integer):
            sampled /= float(np.iinfo(value.dtype).max)
        elif float(np.max(sampled)) > 1.0:
            sampled /= 255.0
        features.append(np.clip(sampled, 0.0, 1.0))
    return PlannerSceneSignature(np.stack(features, axis=0))


@dataclass(frozen=True)
class PlannerContextFrame:
    sequence: int
    capture_timestamp_ns: int
    full_view: Any
    center_view: Any
    rectified: bool = True

    def __post_init__(self) -> None:
        if self.sequence < 0 or self.capture_timestamp_ns < 0:
            raise ValueError("planner frame sequence/timestamp must be non-negative")
        if self.full_view is None or self.center_view is None:
            raise ValueError("planner frame requires full and center views from one capture")
        if not self.rectified:
            raise ValueError("planner context accepts rectified frames only")


class PlannerContextBuffer:
    """Keep exactly the latest causal full views and their current center crop."""

    def __init__(self, history: int = 3) -> None:
        if history != 3:
            raise ValueError("V1 planner context is frozen to exactly three full frames")
        self._frames: Deque[PlannerContextFrame] = deque(maxlen=history)

    def clear(self) -> None:
        self._frames.clear()

    def append(self, frame: PlannerContextFrame, *, now_ns: int) -> None:
        if frame.capture_timestamp_ns > int(now_ns):
            raise ValueError("planner frame cannot come from the future")
        if self._frames:
            previous = self._frames[-1]
            if frame.sequence <= previous.sequence:
                raise ValueError("planner frame sequence must be strictly increasing")
            if frame.capture_timestamp_ns <= previous.capture_timestamp_ns:
                raise ValueError("planner capture timestamps must be strictly increasing")
        self._frames.append(frame)

    @property
    def ready(self) -> bool:
        return len(self._frames) == 3

    def build_request(
        self,
        *,
        primitive: str | EMGPrimitive,
        now_ns: int,
        metadata: Optional[dict[str, Any]] = None,
        clarification_answer: str = "",
    ) -> PlannerRequest:
        if not self.ready:
            raise RuntimeError("planner context needs three distinct causal frames")
        latest = self._frames[-1]
        if latest.capture_timestamp_ns > int(now_ns):
            raise ValueError("latest planner context is from the future")
        merged_metadata = dict(metadata or {})
        merged_metadata.update(
            {
                "full_view_sequences": [frame.sequence for frame in self._frames],
                "current_center_sequence": latest.sequence,
                "rectified": True,
            }
        )
        return PlannerRequest.from_aligned_views(
            primitive=primitive,
            full_view_history=tuple(frame.full_view for frame in self._frames),
            center_view=latest.center_view,
            full_view_timestamps_ns=tuple(
                frame.capture_timestamp_ns for frame in self._frames
            ),
            center_timestamp_ns=latest.capture_timestamp_ns,
            metadata=merged_metadata,
            clarification_answer=clarification_answer,
        )


@dataclass(frozen=True)
class PlannerWorkItem:
    generation: int
    event_id: str
    primitive: EMGPrimitive
    task_version: int
    request: PlannerRequest
    submitted_at_ns: int
    scene_signature: PlannerSceneSignature | None = None

    def __post_init__(self) -> None:
        primitive = normalize_emg_primitive(self.primitive)
        if primitive not in START_PRIMITIVES:
            raise ValueError("planner work item requires a start primitive")
        object.__setattr__(self, "primitive", primitive)
        if self.generation < 0 or self.task_version < 0 or self.submitted_at_ns < 0:
            raise ValueError("planner generation/version/timestamp must be non-negative")
        if not self.event_id:
            raise ValueError("planner work item requires the latched event_id")
        if self.request.primitive != primitive:
            raise ValueError("planner work primitive/request mismatch")
        signature = self.scene_signature
        if signature is None:
            if not self.request.uses_aligned_views:
                raise ValueError("planner work requires aligned views for scene consistency")
            signature = planner_scene_signature(
                self.request.full_view_history[-1], self.request.center_view
            )
            object.__setattr__(self, "scene_signature", signature)


@dataclass(frozen=True)
class PlannerWorkResult:
    generation: int
    event_id: str
    primitive: EMGPrimitive
    task_version: int
    request_timestamp_ns: int
    submitted_at_ns: int
    completed_at_ns: int
    decision: Optional[PlannerDecision]
    error: str = ""
    scene_signature: PlannerSceneSignature | None = None


class _PlannerLike(Protocol):
    def plan(self, request: PlannerRequest) -> PlannerDecision:
        ...


@dataclass(frozen=True)
class _QueuedPlannerWork:
    item: PlannerWorkItem
    enqueued_clock_ns: int


class AsyncPlannerWorker:
    """One daemon worker with at most one queued/in-flight Planner request."""

    def __init__(self, planner: _PlannerLike, *, clock_ns=time.monotonic_ns) -> None:
        self._planner = planner
        self._clock_ns = clock_ns
        self._requests: queue.Queue[_QueuedPlannerWork | None] = queue.Queue(maxsize=1)
        self._results: queue.Queue[PlannerWorkResult] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._busy = False
        self._closed = False
        self.last_discard_reason = ""
        self._thread = threading.Thread(
            target=self._run,
            name="revo3-planner-worker",
            daemon=True,
        )
        self._thread.start()

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    @property
    def pending(self) -> bool:
        """True while work is running *or* a completed result awaits polling."""

        with self._lock:
            return self._busy or not self._results.empty()

    def submit(self, item: PlannerWorkItem) -> bool:
        """Queue without blocking; return ``False`` while a request is active."""

        with self._lock:
            if self._closed:
                raise RuntimeError("planner worker is closed")
            if self._busy:
                return False
            self._busy = True
        try:
            self._requests.put_nowait(
                _QueuedPlannerWork(item=item, enqueued_clock_ns=int(self._clock_ns()))
            )
        except queue.Full:
            with self._lock:
                self._busy = False
            return False
        return True

    def _publish(self, result: PlannerWorkResult) -> None:
        try:
            self._results.put_nowait(result)
        except queue.Full:
            try:
                self._results.get_nowait()
            except queue.Empty:
                pass
            self._results.put_nowait(result)

    def _run(self) -> None:
        while True:
            queued = self._requests.get()
            if queued is None:
                return
            item = queued.item
            decision: Optional[PlannerDecision] = None
            error = ""
            try:
                decision = self._planner.plan(item.request)
            except Exception as exc:  # fail-closed boundary around model/runtime errors
                error = f"{type(exc).__name__}: {exc}"[:512]
            # Map elapsed worker time into the caller/control clock domain.
            # In production both clocks are monotonic_ns; this mapping also
            # keeps deterministic simulated clocks causally meaningful.
            elapsed_ns = max(0, int(self._clock_ns()) - queued.enqueued_clock_ns)
            completed_at_ns = item.submitted_at_ns + elapsed_ns
            if decision is not None:
                decision = replace(decision, produced_at_ns=completed_at_ns)
            self._publish(
                PlannerWorkResult(
                    generation=item.generation,
                    event_id=item.event_id,
                    primitive=item.primitive,
                    task_version=item.task_version,
                    request_timestamp_ns=item.request.timestamp_ns,
                    submitted_at_ns=item.submitted_at_ns,
                    completed_at_ns=completed_at_ns,
                    decision=decision,
                    error=error,
                    scene_signature=item.scene_signature,
                )
            )
            with self._lock:
                self._busy = False

    def poll(
        self,
        *,
        expected_generation: int,
        expected_event_id: str,
        expected_primitive: str | EMGPrimitive,
        expected_task_version: int,
        now_ns: int,
        result_ttl_ns: int = 1_000_000_000,
        planner_sla_ns: int = 20_000_000_000,
        allowed_future_skew_ns: int = 5_000_000,
        max_age_ns: int | None = None,
        current_scene_signature: PlannerSceneSignature | None = None,
        max_scene_distance: float = 0.08,
    ) -> Optional[PlannerWorkResult]:
        """Return only a current result; obsolete work is consumed and dropped."""

        if max_age_ns is not None:
            # Source-compatible spelling from the initial mock runtime.  Its
            # semantics are now explicitly result freshness, never source age.
            result_ttl_ns = int(max_age_ns)
        if result_ttl_ns <= 0 or planner_sla_ns <= 0 or allowed_future_skew_ns < 0:
            raise ValueError("planner SLA/freshness limits must be positive")
        if not 0.0 < float(max_scene_distance) <= 1.0:
            raise ValueError("max_scene_distance must be in (0,1]")
        try:
            result = self._results.get_nowait()
        except queue.Empty:
            return None
        expected = normalize_emg_primitive(expected_primitive)
        checks = (
            (result.generation == expected_generation, "generation_changed"),
            (result.event_id == expected_event_id, "event_changed"),
            (result.primitive == expected, "primitive_changed"),
            (result.task_version == expected_task_version, "task_version_changed"),
            (result.request_timestamp_ns <= int(now_ns), "request_from_future"),
            (result.submitted_at_ns >= result.request_timestamp_ns, "submitted_before_capture"),
            (result.completed_at_ns >= result.submitted_at_ns, "completed_before_submit"),
            (
                result.completed_at_ns - result.submitted_at_ns <= int(planner_sla_ns),
                "planner_sla_exceeded",
            ),
            (
                result.completed_at_ns <= int(now_ns) + int(allowed_future_skew_ns),
                "planner_result_from_future",
            ),
            (
                int(now_ns) - result.completed_at_ns <= int(result_ttl_ns),
                "planner_result_stale",
            ),
        )
        for accepted, reason in checks:
            if not accepted:
                self.last_discard_reason = reason
                return None
        if result.error or result.decision is None:
            self.last_discard_reason = "planner_error"
            return result
        if result.decision.timestamp_ns != result.request_timestamp_ns:
            self.last_discard_reason = "decision_request_timestamp_mismatch"
            return None
        if current_scene_signature is not None:
            if result.scene_signature is None:
                self.last_discard_reason = "planner_scene_signature_missing"
                return None
            if result.scene_signature.distance(current_scene_signature) > float(max_scene_distance):
                self.last_discard_reason = "planner_scene_changed"
                return None
        self.last_discard_reason = ""
        return result

    def close(
        self,
        *,
        timeout_s: float = 1.0,
        wait: bool | None = None,
    ) -> bool:
        """Request worker shutdown and report whether its thread actually exited.

        A model call cannot be interrupted safely from another thread.  Every
        invocation therefore retries the sentinel when a previous attempt
        raced a full request queue, then joins for the caller's explicit
        bounded budget.  ``False`` means the daemon still owns an in-flight
        Planner call; callers must report an unclean shutdown rather than
        infer cleanliness from ``busy`` or silently abandon the thread.
        """

        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        with self._lock:
            self._closed = True
        deadline = time.monotonic() + float(timeout_s)
        while self._thread.is_alive():
            try:
                self._requests.put_nowait(None)
            except queue.Full:
                # A queued request can initially occupy the only slot.  Keep
                # retrying inside this same caller-owned deadline; shutdown
                # correctness must not depend on a second close invocation.
                pass
            # ``wait`` is retained only for source compatibility with the
            # initial smoke API.  New callers always use the explicit budget.
            if wait is False:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            self._thread.join(min(0.01, remaining))
        return not self._thread.is_alive()
