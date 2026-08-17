"""Small synchronous orchestrator for the main-aligned policy cadence."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .adapter import TReXRevoPolicyAdapter
from .aggregation import ActionTemporalAggregator, TemporalAggregationError
from .contracts import ActionChunk, InferenceMode, PolicyObservation, PolicyRequest, TaskKey
from .schedule import MAIN_ALIGNED_SCHEDULE, PolicySchedule


class TReXPolicyRunner:
    """Issues due slow/fast requests and aggregates their absolute-q chunks."""

    def __init__(
        self,
        adapter: TReXRevoPolicyAdapter,
        *,
        schedule: PolicySchedule = MAIN_ALIGNED_SCHEDULE,
        aggregator: ActionTemporalAggregator | None = None,
    ) -> None:
        self.adapter = adapter
        self.schedule = schedule
        self.aggregator = (
            ActionTemporalAggregator(k=0.0) if aggregator is None else aggregator
        )
        self.last_skip_reason: str | None = None

    def reset(self, reason: str) -> None:
        self.adapter.reset(reason)
        self.aggregator.clear()
        self.last_skip_reason = None

    def infer_if_due(
        self,
        *,
        global_step: int,
        observation: PolicyObservation,
        now_ns: int,
    ) -> Optional[ActionChunk]:
        if global_step < 0:
            raise ValueError("global_step must be non-negative.")
        offset = global_step % self.schedule.chunk_size
        mode = self.schedule.mode_at_chunk_offset(offset)
        if mode is InferenceMode.NONE:
            self.last_skip_reason = "not_due"
            return None
        tactile_age_ns = int(now_ns) - int(observation.tactile_timestamp_ns)
        if tactile_age_ns < 0:
            raise ValueError("tactile observation cannot be from the future.")
        if (
            mode is InferenceMode.FAST
            and tactile_age_ns > self.adapter.cache.max_tactile_age_ns
        ):
            # A stale tactile tick never reaches the fast expert.  Keep the
            # last valid nominal chunk/cache so the execution layer can HOLD;
            # do not synthesize a duplicate tactile sample or clear the task.
            self.last_skip_reason = "stale_touch_fast_disabled"
            return None
        start_step = global_step - offset
        request = PolicyRequest(mode=mode, chunk_offset=offset, observation=observation)
        chunk = self.adapter.infer(
            request,
            global_start_step=start_step,
            now_ns=now_ns,
        )
        self.aggregator.add(chunk, now_ns=now_ns)
        self.last_skip_reason = None
        return chunk

    def target_for_step(
        self,
        *,
        global_step: int,
        task_key: TaskKey,
        now_ns: int,
    ) -> np.ndarray:
        return self.aggregator.target_at(
            global_step, task_key=task_key, now_ns=now_ns
        )

    def interpolated_target_for_tick(
        self,
        *,
        executor_timestamp_ns: int,
        action_epoch_ns: int,
        task_key: TaskKey,
        now_ns: int,
    ) -> np.ndarray:
        """Linearly lift the frozen 30 Hz action grid to a faster servo tick.

        The method never extrapolates a completed chunk.  Between its final
        action sample and the next 30 Hz boundary it holds that final sample;
        at/after the boundary a new covering chunk is required.
        """

        elapsed_ns = int(executor_timestamp_ns) - int(action_epoch_ns)
        if elapsed_ns < 0:
            raise ValueError("executor tick precedes the action-grid epoch.")
        period = self.schedule.action_period_ns
        position = elapsed_ns / period
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
            # Only the last action interval may hold instead of interpolate.
            current = self.adapter.cache.current_chunk
            final_step = (
                None
                if current is None
                else current.start_step + self.schedule.chunk_size - 1
            )
            if lower_step != final_step:
                raise
            return lower
        return ((1.0 - alpha) * lower + alpha * upper).astype(np.float32)
