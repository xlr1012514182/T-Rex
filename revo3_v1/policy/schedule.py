"""T-Rex ``main``-aligned slow/fast schedule."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import ACTION_CHUNK, InferenceMode


@dataclass(frozen=True)
class PolicySchedule:
    command_hz: float = 30.0
    chunk_size: int = ACTION_CHUNK
    execute_steps_per_chunk: int = ACTION_CHUNK
    refine_offsets: tuple[int, ...] = (4, 8, 12)

    def __post_init__(self) -> None:
        if self.command_hz <= 0:
            raise ValueError("command_hz must be positive.")
        if self.chunk_size != ACTION_CHUNK:
            raise ValueError(f"Revo V1 freezes chunk_size={ACTION_CHUNK}.")
        if self.execute_steps_per_chunk != self.chunk_size:
            raise ValueError("T-Rex main executes the full 16-step chunk.")
        offsets = tuple(int(value) for value in self.refine_offsets)
        if tuple(sorted(set(offsets))) != offsets:
            raise ValueError("refine_offsets must be sorted and unique.")
        if any(offset <= 0 or offset >= self.chunk_size for offset in offsets):
            raise ValueError("Every refinement must lie strictly inside the chunk.")
        object.__setattr__(self, "refine_offsets", offsets)

    @property
    def action_period_ns(self) -> int:
        return round(1e9 / self.command_hz)

    @property
    def chunk_period_s(self) -> float:
        return self.chunk_size / self.command_hz

    def mode_at_chunk_offset(self, offset: int) -> InferenceMode:
        if offset < 0 or offset >= self.chunk_size:
            raise ValueError(f"offset must be in [0,{self.chunk_size - 1}].")
        if offset == 0:
            return InferenceMode.SLOW_AND_FAST
        if offset in self.refine_offsets:
            return InferenceMode.FAST
        return InferenceMode.NONE

    def mode_at_global_step(self, step: int) -> InferenceMode:
        if step < 0:
            raise ValueError("step must be non-negative.")
        return self.mode_at_chunk_offset(step % self.chunk_size)


MAIN_ALIGNED_SCHEDULE = PolicySchedule()
