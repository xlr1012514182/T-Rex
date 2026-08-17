"""Small synchronous orchestrator for the main-aligned policy cadence."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .adapter import TReXRevoPolicyAdapter
from .aggregation import ActionTemporalAggregator
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

    def reset(self, reason: str) -> None:
        self.adapter.reset(reason)
        self.aggregator.clear()

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
            return None
        start_step = global_step - offset
        request = PolicyRequest(mode=mode, chunk_offset=offset, observation=observation)
        chunk = self.adapter.infer(
            request,
            global_start_step=start_step,
            now_ns=now_ns,
        )
        self.aggregator.add(chunk, now_ns=now_ns)
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
