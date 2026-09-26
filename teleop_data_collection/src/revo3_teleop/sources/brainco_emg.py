"""BrainCo EDU armband EMG parsing and an opt-in injected source adapter.

The official EDU example defines one row as ``[seq, lead_off, 8 * 20]``.
This module preserves that exact evidence, returns EMG as ``[8, 20]`` at
250 Hz, and reconstructs per-sample timestamps backwards from the host
callback.  It does not claim that reconstructed time is a device clock.
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Iterable, Protocol, Sequence

import numpy as np

from revo3_teleop.contracts import NativeSample, SampleHeader

from .common import (
    Clock,
    SequenceGapTracker,
    finite_row,
    monotonic_ns,
    reconstructed_sample_times,
    strict_nonnegative_int,
)


EMG_CHANNELS = 8
EMG_SAMPLES_PER_PACKET = 20
EMG_SAMPLE_RATE_HZ = 250.0
EMG_PACKET_VALUES = 2 + EMG_CHANNELS * EMG_SAMPLES_PER_PACKET
EMG_PACKET_DURATION_NS = int(round(1_000_000_000 * EMG_SAMPLES_PER_PACKET / EMG_SAMPLE_RATE_HZ))
EMG_CLOCK_METHOD = "host_callback_end_anchored_reconstruction"


class BrainCoEduEMGClient(Protocol):
    """Small injection boundary; a real bc-edu wrapper may implement it."""

    def register_emg_callback(self, callback: Callable[[Sequence[Sequence[object]]], None]) -> None: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...


class BrainCoEduEMGParser:
    """Stateful strict parser with packet-drop evidence."""

    def __init__(self, *, source_id: str = "brainco_edu_emg") -> None:
        self.source_id = source_id
        self._sequences = SequenceGapTracker()

    def parse_row(
        self,
        row: Sequence[object],
        *,
        callback_timestamp_ns: int,
        reconstructed_packet_end_timestamp_ns: int | None = None,
    ) -> NativeSample:
        values = finite_row(row, name="BrainCo EDU EMG row", exact_size=EMG_PACKET_VALUES)
        sequence = strict_nonnegative_int(values[0], name="EMG sequence")
        lead_off = strict_nonnegative_int(values[1], name="EMG lead_off_bits", maximum=0xFF)
        callback_ns = strict_nonnegative_int(
            callback_timestamp_ns, name="callback_timestamp_ns"
        )
        packet_end_ns = (
            callback_ns
            if reconstructed_packet_end_timestamp_ns is None
            else strict_nonnegative_int(
                reconstructed_packet_end_timestamp_ns,
                name="reconstructed_packet_end_timestamp_ns",
            )
        )
        if packet_end_ns > callback_ns:
            raise ValueError("reconstructed packet end cannot follow callback arrival")
        sample_times = reconstructed_sample_times(
            packet_end_ns,
            sample_count=EMG_SAMPLES_PER_PACKET,
            sample_rate_hz=EMG_SAMPLE_RATE_HZ,
        )
        emg = values[2:].astype(np.float32, copy=True).reshape(
            EMG_CHANNELS, EMG_SAMPLES_PER_PACKET
        )
        lead_mask = np.asarray([(lead_off >> index) & 1 for index in range(EMG_CHANNELS)], dtype=np.uint8)
        # Mutate packet-order state only after every stateless validation has
        # succeeded, so a malformed callback cannot consume a sequence id.
        dropped = self._sequences.observe(sequence)
        return NativeSample(
            SampleHeader(
                source_id=self.source_id,
                sequence=sequence,
                # The last reconstructed sample is conservatively anchored at
                # or before callback arrival; no device time is fabricated.
                capture_timestamp_ns=packet_end_ns,
                receive_timestamp_ns=callback_ns,
                clock_domain=EMG_CLOCK_METHOD,
                device_timestamp_ns=None,
                valid=lead_off == 0,
                dropped_since_previous=dropped,
            ),
            {
                "signal": emg,
                "raw_packet": values.astype(np.float32, copy=True),
                "lead_off_bits": np.asarray([lead_off], dtype=np.uint16),
                "lead_off_mask": lead_mask,
                "sample_rate_hz": np.asarray([EMG_SAMPLE_RATE_HZ], dtype=np.float32),
                "samples_per_channel": np.asarray([EMG_SAMPLES_PER_PACKET], dtype=np.int16),
                "sample_timestamp_ns": sample_times,
                "host_callback_timestamp_ns": np.asarray([callback_ns], dtype=np.int64),
                "reconstructed_packet_end_timestamp_ns": np.asarray(
                    [packet_end_ns], dtype=np.int64
                ),
                "callback_to_packet_end_latency_ns": np.asarray(
                    [callback_ns - packet_end_ns], dtype=np.int64
                ),
                "clock_is_host_reconstruction": np.asarray([1], dtype=np.uint8),
                "device_timestamp_available": np.asarray([0], dtype=np.uint8),
                "sequence_gap_packets": np.asarray([dropped], dtype=np.int64),
            },
        )


class BrainCoEduEMGSource:
    """Queueing callback adapter that never starts hardware implicitly.

    ``client_factory`` is lazy and is not called by construction.  Starting a
    real or fake client requires ``allow_hardware_start=True`` explicitly.
    Raw callbacks can always be injected through :meth:`ingest_rows` for
    offline tests and replay.
    """

    hardware_autostart = False

    def __init__(
        self,
        *,
        client_factory: Callable[[], BrainCoEduEMGClient] | None = None,
        allow_hardware_start: bool = False,
        clock: Clock = monotonic_ns,
        parser: BrainCoEduEMGParser | None = None,
    ) -> None:
        self._client_factory = client_factory
        self._allow_hardware_start = bool(allow_hardware_start)
        self._clock = clock
        self._parser = parser or BrainCoEduEMGParser()
        self._client: BrainCoEduEMGClient | None = None
        self._queue: deque[NativeSample] = deque()

    @property
    def started(self) -> bool:
        return self._client is not None

    def start(self) -> None:
        if not self._allow_hardware_start:
            raise PermissionError("BrainCo EDU hardware start is disabled; opt in explicitly")
        if self._client_factory is None:
            raise RuntimeError("no injected BrainCo EDU client factory")
        if self._client is not None:
            raise RuntimeError("BrainCo EDU EMG source is already started")
        client = self._client_factory()
        # A failed SDK start may still own a worker thread or module-global
        # callback.  Retain the client before registration/start so cleanup can
        # be retried and a second start cannot steal authority.
        self._client = client
        client.register_emg_callback(self._on_callback)
        client.start()

    def stop(self) -> None:
        if self._client is None:
            return
        client = self._client
        client.stop()
        self._client = None

    def _on_callback(self, rows: Sequence[Sequence[object]]) -> None:
        self.ingest_rows(rows)

    def ingest_rows(
        self,
        rows: Iterable[Sequence[object]],
        *,
        callback_timestamp_ns: int | None = None,
    ) -> tuple[NativeSample, ...]:
        materialized = list(rows)
        callback_ns = self._clock() if callback_timestamp_ns is None else int(callback_timestamp_ns)
        parsed: list[NativeSample] = []
        for index, row in enumerate(materialized):
            # A batched callback contains older packets before the final one.
            packet_end_ns = callback_ns - (len(materialized) - 1 - index) * EMG_PACKET_DURATION_NS
            sample = self._parser.parse_row(
                row,
                callback_timestamp_ns=callback_ns,
                reconstructed_packet_end_timestamp_ns=packet_end_ns,
            )
            self._queue.append(sample)
            parsed.append(sample)
        return tuple(parsed)

    def drain(self) -> tuple[NativeSample, ...]:
        samples = tuple(self._queue)
        self._queue.clear()
        return samples
