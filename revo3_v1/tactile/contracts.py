"""Canonical single-hand Force6D contracts."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


FINGER_ORDER = ("thumb", "index", "middle", "ring", "little")
FINGER_COUNT = len(FINGER_ORDER)
F6_DIM = 6
HISTORY_LENGTH = 16


def _as_f6(value: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    expected = (FINGER_COUNT, F6_DIM)
    if arr.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {arr.shape}.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinity.")
    return arr.copy()


@dataclass(frozen=True)
class TactileFrame:
    timestamp_ns: int
    f6: np.ndarray
    sequence: int
    valid_fingers: np.ndarray = field(
        default_factory=lambda: np.ones(FINGER_COUNT, dtype=bool)
    )

    def __post_init__(self) -> None:
        if self.timestamp_ns < 0 or self.sequence < 0:
            raise ValueError("timestamp_ns and sequence must be non-negative.")
        object.__setattr__(self, "f6", _as_f6(self.f6, name="f6"))
        valid = np.asarray(self.valid_fingers, dtype=bool)
        if valid.shape != (FINGER_COUNT,):
            raise ValueError(
                f"valid_fingers must have shape ({FINGER_COUNT},), got {valid.shape}."
            )
        object.__setattr__(self, "valid_fingers", valid.copy())


@dataclass(frozen=True)
class TactileWindow:
    f6: np.ndarray
    timestamps_ns: np.ndarray
    sequences: np.ndarray
    valid_fingers: np.ndarray

    def __post_init__(self) -> None:
        expected = (HISTORY_LENGTH, FINGER_COUNT, F6_DIM)
        f6 = np.asarray(self.f6, dtype=np.float32)
        if f6.shape != expected:
            raise ValueError(f"f6 window must have shape {expected}, got {f6.shape}.")
        ts = np.asarray(self.timestamps_ns, dtype=np.int64)
        seq = np.asarray(self.sequences, dtype=np.int64)
        valid = np.asarray(self.valid_fingers, dtype=bool)
        if ts.shape != (HISTORY_LENGTH,) or seq.shape != (HISTORY_LENGTH,):
            raise ValueError("timestamps_ns and sequences must have shape (16,).")
        if valid.shape != (HISTORY_LENGTH, FINGER_COUNT):
            raise ValueError("valid_fingers must have shape (16,5).")
        if np.any(np.diff(ts) <= 0) or np.any(np.diff(seq) <= 0):
            raise ValueError("window timestamps and sequences must be strictly increasing.")
        object.__setattr__(self, "f6", f6.copy())
        object.__setattr__(self, "timestamps_ns", ts.copy())
        object.__setattr__(self, "sequences", seq.copy())
        object.__setattr__(self, "valid_fingers", valid.copy())

    @property
    def current(self) -> np.ndarray:
        return self.f6[-1].copy()

    @property
    def newest_timestamp_ns(self) -> int:
        return int(self.timestamps_ns[-1])

    @property
    def span_ns(self) -> int:
        return int(self.timestamps_ns[-1] - self.timestamps_ns[0])
