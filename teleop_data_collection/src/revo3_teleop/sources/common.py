"""Shared, dependency-light helpers for callback-driven sensor sources.

The public BrainCo EDU callbacks and the official MANUS ROS ``ManusGlove``
message do not expose a trustworthy absolute acquisition timestamp.  Sources
therefore keep host callback time as evidence and never label it as device
time.  A caller may later replace this clock with a verified hardware clock,
but that is an explicit adapter change rather than an inference here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable, Sequence

import numpy as np


Clock = Callable[[], int]


def monotonic_ns() -> int:
    return time.monotonic_ns()


def strict_nonnegative_int(value: object, *, name: str, maximum: int | None = None) -> int:
    """Parse an integer-valued scalar without silently truncating floats."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not bool")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{name} must be a finite integer")
    parsed = int(numeric)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return parsed


def finite_row(row: Sequence[object], *, name: str, exact_size: int | None = None) -> np.ndarray:
    try:
        values = np.asarray(row, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a flat numeric row") from exc
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if exact_size is not None and values.size != exact_size:
        raise ValueError(f"{name} must contain exactly {exact_size} values, got {values.size}")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return values


def reconstructed_sample_times(
    callback_timestamp_ns: int,
    *,
    sample_count: int,
    sample_rate_hz: float,
) -> np.ndarray:
    """End-anchor a block on host callback time and reconstruct sample times.

    This is deliberately named *reconstructed*: callback arrival is not a
    device timestamp and includes unknown transport/scheduling latency.
    """

    callback_timestamp_ns = strict_nonnegative_int(
        callback_timestamp_ns, name="callback_timestamp_ns"
    )
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be finite and positive")
    period_ns = int(round(1_000_000_000.0 / sample_rate_hz))
    first = callback_timestamp_ns - (sample_count - 1) * period_ns
    if first < 0:
        raise ValueError("callback timestamp is too small to reconstruct the sample block")
    return first + np.arange(sample_count, dtype=np.int64) * period_ns


@dataclass
class SequenceGapTracker:
    """Strict monotonic packet tracker.

    No wrap modulus is guessed because neither public EDU example documents
    one for EMG/Flex packets.  A regression or duplicate is rejected so the
    collector cannot hide ordering ambiguity.
    """

    last_sequence: int | None = None

    def observe(self, sequence: int) -> int:
        sequence = strict_nonnegative_int(sequence, name="sequence")
        if self.last_sequence is None:
            self.last_sequence = sequence
            return 0
        if sequence <= self.last_sequence:
            raise ValueError(
                f"sequence must increase strictly: previous={self.last_sequence}, current={sequence}"
            )
        dropped = sequence - self.last_sequence - 1
        self.last_sequence = sequence
        return dropped
