"""Causal multi-rate timestamp alignment for the Revo 3 V1 pipeline.

The aligner has two intentionally distinct grid modes:

``lowest``
    Use the slowest configured stream rate.  This is the recommended demo mode
    when all modules should rendezvous at the slowest producer.

``gcd``
    Use the rational greatest common divisor of all configured rates.  This is
    useful when rates share a smaller exact clock (for example 100, 30, 20 Hz
    produce a 10 Hz grid).

At every anchor only the newest sample with ``sample.timestamp <= anchor`` is
selected.  Interpolation from, or nearest-neighbour selection of, a future
sample is deliberately forbidden.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import deque
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from functools import reduce
from math import gcd
from typing import Any, Deque, Dict, Iterable, Mapping, Optional, Sequence, Tuple


NANOSECONDS_PER_SECOND = 1_000_000_000


class AlignmentMode(str, Enum):
    LOWEST = "lowest"
    GCD = "gcd"


def _as_rate(value: Any) -> Fraction:
    if isinstance(value, Fraction):
        rate = value
    else:
        rate = Fraction(str(value)).limit_denominator(100_000)
    if rate <= 0:
        raise ValueError("stream rates must be positive")
    return rate


def rational_gcd(values: Iterable[Fraction]) -> Fraction:
    """Greatest common divisor for positive rational numbers."""

    fractions = tuple(_as_rate(value) for value in values)
    if not fractions:
        raise ValueError("at least one rate is required")

    def lcm(left: int, right: int) -> int:
        return abs(left * right) // gcd(left, right)

    denominator = reduce(lcm, (value.denominator for value in fractions))
    scaled = [value.numerator * (denominator // value.denominator) for value in fractions]
    numerator = reduce(gcd, scaled)
    return Fraction(numerator, denominator)


@dataclass(frozen=True)
class StreamConfig:
    name: str
    rate_hz: Fraction
    max_age_ns: int
    required: bool = True

    def __init__(
        self,
        name: str,
        rate_hz: Any,
        *,
        max_age_ns: Optional[int] = None,
        required: bool = True,
    ) -> None:
        rate = _as_rate(rate_hz)
        if not name:
            raise ValueError("stream name must not be empty")
        if max_age_ns is None:
            # Three nominal periods is a conservative initial stale limit.
            max_age_ns = (3 * NANOSECONDS_PER_SECOND * rate.denominator) // rate.numerator
        if max_age_ns < 0:
            raise ValueError("max_age_ns must be non-negative")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "rate_hz", rate)
        object.__setattr__(self, "max_age_ns", int(max_age_ns))
        object.__setattr__(self, "required", bool(required))


@dataclass(frozen=True)
class TimestampedSample:
    timestamp_ns: int
    value: Any
    sequence: int = 0

    def __post_init__(self) -> None:
        if self.timestamp_ns < 0:
            raise ValueError("sample timestamp must be non-negative")


@dataclass(frozen=True)
class AlignedSample:
    stream: str
    sample: TimestampedSample
    anchor_ns: int

    @property
    def age_ns(self) -> int:
        return self.anchor_ns - self.sample.timestamp_ns


@dataclass(frozen=True)
class AlignedFrame:
    anchor_ns: int
    grid_hz: Fraction
    samples: Mapping[str, AlignedSample]
    missing: Tuple[str, ...]
    stale: Tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.missing and not self.stale

    def value(self, stream: str) -> Any:
        return self.samples[stream].sample.value


class CausalTimestampAligner:
    """Bounded, causal online alignment over heterogeneous streams."""

    def __init__(
        self,
        streams: Sequence[StreamConfig],
        *,
        mode: AlignmentMode = AlignmentMode.LOWEST,
        buffer_size: int = 4096,
        epoch_ns: Optional[int] = None,
    ) -> None:
        if not streams:
            raise ValueError("at least one stream is required")
        if len({stream.name for stream in streams}) != len(streams):
            raise ValueError("stream names must be unique")
        if buffer_size < 2:
            raise ValueError("buffer_size must be at least two")
        if epoch_ns is not None and epoch_ns < 0:
            raise ValueError("epoch_ns must be non-negative")
        self.streams = {stream.name: stream for stream in streams}
        self.mode = AlignmentMode(mode)
        rates = tuple(stream.rate_hz for stream in streams)
        self.grid_hz = min(rates) if self.mode == AlignmentMode.LOWEST else rational_gcd(rates)
        self._period_ns = Fraction(NANOSECONDS_PER_SECOND, 1) / self.grid_hz
        self._buffers: Dict[str, Deque[TimestampedSample]] = {
            stream.name: deque(maxlen=buffer_size) for stream in streams
        }
        self.epoch_ns = int(epoch_ns or 0)
        self._epoch_initialized = epoch_ns is not None
        self._next_index = 0

    @property
    def nominal_period_ns(self) -> Fraction:
        return self._period_ns

    def anchor_at_index(self, index: int) -> int:
        if index < 0:
            raise ValueError("anchor index must be non-negative")
        # Integer nanoseconds cannot represent every rational-rate period.  The
        # absolute-index calculation avoids accumulating rounding drift.
        offset = self._period_ns * index
        return self.epoch_ns + int(offset.numerator // offset.denominator)

    def offer(self, stream: str, sample: TimestampedSample) -> None:
        if stream not in self._buffers:
            raise KeyError(f"unknown stream: {stream}")
        if not self._epoch_initialized:
            self.epoch_ns = sample.timestamp_ns
            self._epoch_initialized = True
        buffer = self._buffers[stream]
        if buffer and sample.timestamp_ns < buffer[-1].timestamp_ns:
            # Small out-of-order deliveries are inserted in timestamp order.
            items = list(buffer)
            timestamps = [item.timestamp_ns for item in items]
            index = bisect_right(timestamps, sample.timestamp_ns)
            items.insert(index, sample)
            buffer.clear()
            buffer.extend(items[-buffer.maxlen :])
        else:
            buffer.append(sample)

    def _latest_not_after(
        self, stream: str, anchor_ns: int
    ) -> Optional[TimestampedSample]:
        buffer = self._buffers[stream]
        if not buffer:
            return None
        timestamps = [sample.timestamp_ns for sample in buffer]
        index = bisect_right(timestamps, anchor_ns) - 1
        if index < 0:
            return None
        return buffer[index]

    def align_at(self, anchor_ns: int) -> AlignedFrame:
        if anchor_ns < 0:
            raise ValueError("anchor timestamp must be non-negative")
        selected: Dict[str, AlignedSample] = {}
        missing = []
        stale = []
        for name, config in self.streams.items():
            sample = self._latest_not_after(name, anchor_ns)
            if sample is None:
                if config.required:
                    missing.append(name)
                continue
            aligned = AlignedSample(name, sample, anchor_ns)
            selected[name] = aligned
            if config.required and aligned.age_ns > config.max_age_ns:
                stale.append(name)
        return AlignedFrame(
            anchor_ns=anchor_ns,
            grid_hz=self.grid_hz,
            samples=selected,
            missing=tuple(sorted(missing)),
            stale=tuple(sorted(stale)),
        )

    def causal_window(
        self, stream: str, *, end_ns: int, count: int
    ) -> Tuple[TimestampedSample, ...]:
        """Return up to ``count`` historical samples, never future samples."""

        if count < 1:
            raise ValueError("count must be positive")
        if stream not in self._buffers:
            raise KeyError(f"unknown stream: {stream}")
        values = [sample for sample in self._buffers[stream] if sample.timestamp_ns <= end_ns]
        return tuple(values[-count:])

    def drain_until(self, end_ns: int, *, include_unready: bool = False) -> Tuple[AlignedFrame, ...]:
        """Emit grid anchors through ``end_ns`` exactly once."""

        if not self._epoch_initialized:
            return ()
        frames = []
        while self.anchor_at_index(self._next_index) <= end_ns:
            frame = self.align_at(self.anchor_at_index(self._next_index))
            self._next_index += 1
            if include_unready or frame.ready:
                frames.append(frame)
        return tuple(frames)

    def reset_grid(self, *, epoch_ns: int) -> None:
        if epoch_ns < 0:
            raise ValueError("epoch_ns must be non-negative")
        self.epoch_ns = epoch_ns
        self._epoch_initialized = True
        self._next_index = 0

    def clear(self) -> None:
        for buffer in self._buffers.values():
            buffer.clear()
        self._next_index = 0
