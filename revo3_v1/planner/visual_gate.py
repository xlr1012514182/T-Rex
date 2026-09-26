"""Stateful visual-readiness gate driven by structured Planner output."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Optional, Tuple

from .schema import PlannerDecision, PlannerStatus, SupportedTask


@dataclass(frozen=True)
class VisualGateConfig:
    min_area: float = 0.15
    max_center_distance: float = 0.25
    min_confidence: float = 0.75
    min_consecutive_ready: int = 2
    max_decision_age_ns: int = 1_000_000_000
    max_center_jump: float = 0.18
    max_area_ratio_change: float = 0.75

    def __post_init__(self) -> None:
        if not 0.0 < self.min_area <= 1.0:
            raise ValueError("min_area must be in (0, 1]")
        if self.min_consecutive_ready < 1:
            raise ValueError("min_consecutive_ready must be positive")


@dataclass(frozen=True)
class VisualGateResult:
    ready: bool
    reason: str
    consecutive_ready: int
    task: Optional[SupportedTask]
    area: float
    center_distance: float
    timestamp_ns: int
    produced_at_ns: Optional[int] = None

    def __post_init__(self) -> None:
        produced = self.timestamp_ns if self.produced_at_ns is None else int(self.produced_at_ns)
        if self.timestamp_ns < 0 or produced < self.timestamp_ns:
            raise ValueError("visual gate timestamps must be causal and non-negative")
        object.__setattr__(self, "produced_at_ns", produced)


class VisualGate:
    """Require a large, centered VLM target for consecutive observations.

    This is not a detector or tracker.  It only validates the current and prior
    structured planner decisions, avoiding an extra segmentation/tracking model.
    """

    def __init__(self, config: VisualGateConfig = VisualGateConfig()) -> None:
        self.config = config
        self._consecutive = 0
        self._last_task: Optional[SupportedTask] = None
        self._last_center: Optional[Tuple[float, float]] = None
        self._last_area: Optional[float] = None
        self._last_timestamp_ns: Optional[int] = None

    def reset(self) -> None:
        self._consecutive = 0
        self._last_task = None
        self._last_center = None
        self._last_area = None
        self._last_timestamp_ns = None

    def _reject(self, decision: PlannerDecision, reason: str) -> VisualGateResult:
        self.reset()
        distance = decision.bbox.center_distance if decision.bbox else float("inf")
        return VisualGateResult(
            ready=False,
            reason=reason,
            consecutive_ready=0,
            task=decision.task,
            area=decision.area,
            center_distance=distance,
            timestamp_ns=decision.timestamp_ns,
            produced_at_ns=int(decision.produced_at_ns),
        )

    def update(self, decision: PlannerDecision, *, now_ns: int) -> VisualGateResult:
        produced_at_ns = int(decision.produced_at_ns)
        if decision.timestamp_ns > now_ns:
            return self._reject(decision, "future_planner_decision")
        if produced_at_ns > now_ns:
            return self._reject(decision, "future_planner_result")
        if self._last_timestamp_ns is not None and decision.timestamp_ns <= self._last_timestamp_ns:
            return self._reject(decision, "non_increasing_planner_timestamp")
        if now_ns - produced_at_ns > self.config.max_decision_age_ns:
            return self._reject(decision, "stale_planner_result")
        if decision.status != PlannerStatus.READY:
            return self._reject(decision, "planner_not_ready")
        if decision.ambiguity.ambiguous:
            return self._reject(decision, "ambiguous_target")
        if (
            not decision.target_present
            or not decision.compatible
            or not decision.near_ready
            or not decision.center_ready
        ):
            return self._reject(decision, "target_not_actionable")
        if decision.ready_frame_count < self.config.min_consecutive_ready:
            return self._reject(decision, "insufficient_ready_frames")
        if decision.bbox is None or decision.task is None:
            return self._reject(decision, "missing_target_geometry")
        if decision.confidence < self.config.min_confidence:
            return self._reject(decision, "low_confidence")
        if decision.area < self.config.min_area:
            return self._reject(decision, "target_too_small")
        if decision.bbox.center_distance > self.config.max_center_distance:
            return self._reject(decision, "target_off_center")

        center = decision.bbox.center
        stable = self._last_task in (None, decision.task)
        if self._last_center is not None:
            stable = stable and hypot(
                center[0] - self._last_center[0], center[1] - self._last_center[1]
            ) <= self.config.max_center_jump
        if self._last_area is not None and self._last_area > 0:
            relative_change = abs(decision.area - self._last_area) / self._last_area
            stable = stable and relative_change <= self.config.max_area_ratio_change

        self._consecutive = self._consecutive + 1 if stable else 1
        self._last_task = decision.task
        self._last_center = center
        self._last_area = decision.area
        self._last_timestamp_ns = decision.timestamp_ns
        # The planner sees three recent full frames in one call.  Its
        # ready_frame_count is the primary temporal evidence; this local count
        # remains an additional consistency check across repeated calls.
        ready = (
            decision.ready_frame_count >= self.config.min_consecutive_ready
            and self._consecutive >= 1
        )
        return VisualGateResult(
            ready=ready,
            reason="ready" if ready else "awaiting_consecutive_ready",
            consecutive_ready=self._consecutive,
            task=decision.task,
            area=decision.area,
            center_distance=decision.bbox.center_distance,
            timestamp_ns=decision.timestamp_ns,
            produced_at_ns=produced_at_ns,
        )
