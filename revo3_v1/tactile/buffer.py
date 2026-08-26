"""A dense history of real sensor samples, independent of policy requests."""

from __future__ import annotations

from collections import deque
import threading
from typing import Optional

import numpy as np

from .contracts import HISTORY_LENGTH, TactileFrame, TactileWindow


class TactileBufferError(RuntimeError):
    pass


class TactileNotReady(TactileBufferError):
    pass


class DenseTactileBuffer:
    """Maintains the last 16 *new* frames; requests never append samples."""

    def __init__(self, capacity: int = HISTORY_LENGTH) -> None:
        if capacity < HISTORY_LENGTH:
            raise ValueError("capacity must be at least 16.")
        self.capacity = int(capacity)
        self._frames: deque[TactileFrame] = deque(maxlen=self.capacity)
        self._lock = threading.Lock()

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)

    def append(self, frame: TactileFrame) -> bool:
        """Append a truly new frame; identical duplicates return ``False``."""

        with self._lock:
            if self._frames:
                last = self._frames[-1]
                if frame.timestamp_ns < last.timestamp_ns or frame.sequence < last.sequence:
                    raise TactileBufferError("out-of-order tactile frame rejected.")
                same_clock = (
                    frame.timestamp_ns == last.timestamp_ns or frame.sequence == last.sequence
                )
                if same_clock:
                    identical = (
                        frame.timestamp_ns == last.timestamp_ns
                        and frame.sequence == last.sequence
                        and np.array_equal(frame.f6, last.f6)
                        and np.array_equal(frame.valid_fingers, last.valid_fingers)
                    )
                    if identical:
                        return False
                    raise TactileBufferError(
                        "duplicate timestamp/sequence carries different tactile data."
                    )
            self._frames.append(frame)
            return True

    def snapshot(
        self,
        *,
        now_ns: int,
        max_age_ns: int,
        max_gap_ns: Optional[int] = None,
    ) -> TactileWindow:
        if max_age_ns <= 0:
            raise ValueError("max_age_ns must be positive.")
        with self._lock:
            if len(self._frames) < HISTORY_LENGTH:
                raise TactileNotReady(
                    f"need {HISTORY_LENGTH} real frames, have {len(self._frames)}."
                )
            frames = list(self._frames)[-HISTORY_LENGTH:]
        newest_age = int(now_ns) - frames[-1].timestamp_ns
        if newest_age < 0:
            raise TactileBufferError("newest tactile frame is from the future.")
        if newest_age > max_age_ns:
            raise TactileNotReady(
                f"tactile frame stale by {newest_age}ns (limit {max_age_ns}ns)."
            )
        timestamps = np.asarray([frame.timestamp_ns for frame in frames], dtype=np.int64)
        if max_gap_ns is not None and np.any(np.diff(timestamps) > int(max_gap_ns)):
            raise TactileNotReady("dense tactile history contains an excessive sample gap.")
        return TactileWindow(
            f6=np.stack([frame.f6 for frame in frames]),
            timestamps_ns=timestamps,
            sequences=np.asarray([frame.sequence for frame in frames], dtype=np.int64),
            valid_fingers=np.stack([frame.valid_fingers for frame in frames]),
        )
