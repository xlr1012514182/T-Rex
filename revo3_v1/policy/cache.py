"""Task/version/time-safe cache for T-Rex cascaded slow/fast inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .contracts import ActionChunk, InferenceMode, PolicyRequest, TaskKey
from .schedule import MAIN_ALIGNED_SCHEDULE, PolicySchedule


class CacheProtocolError(RuntimeError):
    pass


@dataclass
class _CacheRecord:
    task_key: TaskKey
    chunk: ActionChunk
    last_offset: int
    last_observation_timestamp_ns: int
    clear_generation: int


class SlowFastCache:
    """Prevents stale KV/action reuse across tasks, versions and time."""

    def __init__(
        self,
        schedule: PolicySchedule = MAIN_ALIGNED_SCHEDULE,
        *,
        max_observation_age_ns: int = 250_000_000,
        max_tactile_age_ns: int = 150_000_000,
    ) -> None:
        if max_observation_age_ns <= 0 or max_tactile_age_ns <= 0:
            raise ValueError("cache age limits must be positive.")
        self.schedule = schedule
        self.max_observation_age_ns = int(max_observation_age_ns)
        self.max_tactile_age_ns = int(max_tactile_age_ns)
        self._record: Optional[_CacheRecord] = None
        self._generation = 0
        self.last_clear_reason = "initial"

    @property
    def active_task_key(self) -> Optional[TaskKey]:
        return None if self._record is None else self._record.task_key

    @property
    def current_chunk(self) -> Optional[ActionChunk]:
        return None if self._record is None else self._record.chunk

    @property
    def generation(self) -> int:
        return self._generation

    def clear(self, reason: str) -> None:
        self._record = None
        self._generation += 1
        self.last_clear_reason = reason or "unspecified"

    def begin(self, request: PolicyRequest, chunk: ActionChunk, *, now_ns: int) -> None:
        if request.mode not in (InferenceMode.SLOW, InferenceMode.SLOW_AND_FAST):
            raise CacheProtocolError("cache begin requires a slow request.")
        if request.chunk_offset != 0:
            raise CacheProtocolError("slow cache must begin at chunk offset 0.")
        self._validate_age(request, now_ns=now_ns)
        if chunk.task_key != request.observation.task_key:
            raise CacheProtocolError("chunk task key differs from request task key.")
        self._record = _CacheRecord(
            task_key=chunk.task_key,
            chunk=chunk,
            last_offset=0,
            last_observation_timestamp_ns=request.observation.timestamp_ns,
            clear_generation=self._generation,
        )

    def validate_fast(self, request: PolicyRequest, *, now_ns: int) -> ActionChunk:
        if request.mode is not InferenceMode.FAST:
            raise CacheProtocolError("validate_fast requires mode=fast.")
        record = self._record
        if record is None:
            raise CacheProtocolError("fast request has no slow cache.")
        if request.observation.task_key != record.task_key:
            raise CacheProtocolError("fast request task/version/instruction/lease mismatch.")
        if request.chunk_offset not in self.schedule.refine_offsets:
            raise CacheProtocolError(
                f"fast offset {request.chunk_offset} not in {self.schedule.refine_offsets}."
            )
        if request.chunk_offset <= record.last_offset:
            raise CacheProtocolError("fast offsets must be strictly increasing within a chunk.")
        if request.observation.timestamp_ns <= record.last_observation_timestamp_ns:
            raise CacheProtocolError("fast observation timestamp must be strictly newer.")
        self._validate_age(request, now_ns=now_ns)
        return record.chunk

    def update_fast(self, request: PolicyRequest, chunk: ActionChunk) -> None:
        record = self._record
        if record is None:
            raise CacheProtocolError("cannot update an empty cache.")
        if chunk.task_key != record.task_key:
            raise CacheProtocolError("refined chunk task key mismatch.")
        if chunk.start_step != record.chunk.start_step:
            raise CacheProtocolError("refined chunk changed its global start step.")
        if chunk.chunk_id != record.chunk.chunk_id:
            raise CacheProtocolError("refined chunk changed its stable chunk_id.")
        record.chunk = chunk
        record.last_offset = request.chunk_offset
        record.last_observation_timestamp_ns = request.observation.timestamp_ns

    def _validate_age(self, request: PolicyRequest, *, now_ns: int) -> None:
        observation = request.observation
        observation_age = int(now_ns) - observation.timestamp_ns
        tactile_age = int(now_ns) - observation.tactile_timestamp_ns
        if observation_age < 0 or tactile_age < 0:
            raise CacheProtocolError("request contains a future timestamp.")
        if observation_age > self.max_observation_age_ns:
            raise CacheProtocolError("policy observation is stale.")
        if tactile_age > self.max_tactile_age_ns:
            raise CacheProtocolError("tactile observation is stale.")
