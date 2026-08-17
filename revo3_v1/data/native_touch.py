"""Reconstruct native Force6D samples from exported overlapping rings.

The policy grid is 30 Hz, but the tactile sensor need not be.  A sequence of
per-policy-frame values therefore cannot stand in for a 16-sample tactile
history.  Exporters attach the latest 16 native samples to every policy
anchor; this module deduplicates those rings by device sequence number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .episode import RevoEpisode


@dataclass(frozen=True)
class NativeForceStream:
    sequence: np.ndarray
    timestamp_ns: np.ndarray
    values: np.ndarray

    def window_ending_at(
        self,
        latest_sequence: int,
        *,
        relative_end: int = 0,
        length: int = 16,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return an unpadded native window ending at/behind current.

        ``relative_end`` is an offset in the native stream, not in 30 Hz
        policy frames.  Positive offsets are rejected to prevent leakage.
        """

        if relative_end > 0:
            raise ValueError("native tactile jitter cannot select a future sample")
        positions = np.flatnonzero(self.sequence == int(latest_sequence))
        if positions.size != 1:
            raise ValueError(
                f"native Force6D sequence {latest_sequence} is missing or ambiguous"
            )
        end = int(positions[0]) + int(relative_end)
        start = end - int(length) + 1
        if start < 0 or end >= self.sequence.size:
            raise ValueError("native Force6D history cannot form an unpadded window")
        sequence = self.sequence[start : end + 1]
        timestamp = self.timestamp_ns[start : end + 1]
        values = self.values[start : end + 1]
        if sequence.shape != (length,) or np.any(np.diff(timestamp) <= 0):
            raise ValueError("native Force6D window must contain distinct real samples")
        return values.copy(), timestamp.copy(), sequence.copy()


def native_force_stream(episode: "RevoEpisode") -> NativeForceStream:
    """Deduplicate exported ``[16,5,6]`` rings into one native stream."""

    if (
        episode.tactile_history_f6 is None
        or episode.tactile_history_timestamp_ns is None
        or episode.tactile_history_sequence is None
    ):
        raise ValueError("native Force6D histories/timestamps/sequences are required")
    observed: dict[int, tuple[int, np.ndarray]] = {}
    for row_values, row_ts, row_sequence in zip(
        episode.tactile_history_f6,
        episode.tactile_history_timestamp_ns,
        episode.tactile_history_sequence,
    ):
        for value, timestamp, sequence in zip(row_values, row_ts, row_sequence):
            key = int(sequence)
            candidate = (int(timestamp), np.asarray(value, dtype=np.float32))
            previous = observed.get(key)
            if previous is not None and (
                previous[0] != candidate[0]
                or not np.array_equal(previous[1], candidate[1])
            ):
                raise ValueError(
                    f"native Force6D sequence {key} has inconsistent duplicate values"
                )
            observed[key] = candidate
    ordered = sorted(observed)
    if len(ordered) < 16:
        raise ValueError("episode contains fewer than 16 distinct native Force6D samples")
    timestamps = np.asarray([observed[key][0] for key in ordered], dtype=np.int64)
    values = np.stack([observed[key][1] for key in ordered]).astype(np.float32, copy=False)
    if values.shape[1:] != (5, 6):
        raise ValueError("native Revo Force6D stream must be [N,5,6]")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("native Force6D sequence timestamps are not strictly increasing")
    return NativeForceStream(
        sequence=np.asarray(ordered, dtype=np.int64),
        timestamp_ns=timestamps,
        values=values,
    )
