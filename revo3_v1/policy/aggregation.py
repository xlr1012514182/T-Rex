"""ACT-style temporal aggregation used by the official T-Rex main client."""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from .contracts import ActionChunk, InferenceMode, TaskKey


class TemporalAggregationError(RuntimeError):
    pass


class ActionTemporalAggregator:
    """Aggregate slow and successive fast revisions covering one robot step.

    T-Rex ``main`` resets its local chunk buffer at each new 16-step slow
    chunk, then appends the initial prediction and fast refinements sharing the
    same start step.  Newer entries receive weights ``exp(-k * age)``; main's
    default ``k=0`` is an arithmetic mean.
    """

    def __init__(
        self,
        *,
        k: float = 0.0,
        max_chunk_age_ns: int = 1_000_000_000,
        max_revisions: int = 8,
    ) -> None:
        if k < 0:
            raise ValueError("k must be non-negative.")
        if max_chunk_age_ns <= 0 or max_revisions <= 0:
            raise ValueError("age and revision limits must be positive.")
        self.k = float(k)
        self.max_chunk_age_ns = int(max_chunk_age_ns)
        self.max_revisions = int(max_revisions)
        self._chunks: deque[ActionChunk] = deque(maxlen=self.max_revisions)
        self._task_key: Optional[TaskKey] = None
        self._start_step: Optional[int] = None

    def clear(self) -> None:
        self._chunks.clear()
        self._task_key = None
        self._start_step = None

    def add(self, chunk: ActionChunk, *, now_ns: int) -> None:
        age = int(now_ns) - chunk.generated_ns
        if age < 0:
            raise TemporalAggregationError("chunk was generated in the future.")
        if age > self.max_chunk_age_ns:
            raise TemporalAggregationError("stale chunk rejected.")
        if self._task_key is not None and chunk.task_key != self._task_key:
            raise TemporalAggregationError("task/version/instruction/lease mismatch.")

        if self._start_step is not None and chunk.start_step != self._start_step:
            # Mirrors main's ``chunk_buffer = []`` at each slow chunk.
            self._chunks.clear()
            self._start_step = None
        if self._chunks:
            last = self._chunks[-1]
            if chunk.generated_ns <= last.generated_ns:
                raise TemporalAggregationError(
                    "chunk revisions must have strictly increasing generation time."
                )
            if chunk.chunk_id != last.chunk_id:
                if chunk.mode in {InferenceMode.SLOW, InferenceMode.SLOW_AND_FAST}:
                    # A repeated slow boundary is a replacement generation,
                    # never a refinement.  Main clears its whole temporal
                    # buffer whenever slow inference runs; mirror that even
                    # if scheduling jitter repeats the same start step.
                    self._chunks.clear()
                else:
                    raise TemporalAggregationError(
                        "fast refinements for one start step must retain the stable chunk_id."
                    )

        self._task_key = chunk.task_key
        self._start_step = chunk.start_step
        self._chunks.append(chunk)

    def target_at(
        self,
        global_step: int,
        *,
        task_key: TaskKey,
        now_ns: int,
    ) -> np.ndarray:
        if global_step < 0:
            raise ValueError("global_step must be non-negative.")
        if self._task_key is None or not self._chunks:
            raise TemporalAggregationError("no action chunk is available.")
        if task_key != self._task_key:
            raise TemporalAggregationError("task/version/instruction/lease mismatch.")

        predictions = []
        for chunk in self._chunks:
            age = int(now_ns) - chunk.generated_ns
            if age < 0:
                raise TemporalAggregationError("chunk was generated in the future.")
            if age > self.max_chunk_age_ns:
                continue
            relative = global_step - chunk.start_step
            if 0 <= relative < len(chunk.q_target_rad):
                predictions.append(chunk.q_target_rad[relative])
        if not predictions:
            raise TemporalAggregationError("no non-stale chunk covers this global step.")

        stacked = np.stack(predictions, axis=0)
        if len(predictions) == 1:
            return stacked[0].copy()
        # Oldest -> newest in deque; this is identical to main's implementation.
        weights = np.exp(-self.k * np.arange(len(predictions))[::-1])
        weights /= weights.sum()
        return (stacked * weights[:, None]).sum(axis=0).astype(np.float32)
