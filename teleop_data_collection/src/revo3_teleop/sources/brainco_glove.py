"""BrainCo EDU motion-glove raw parsers and opt-in injected source.

The public glove example provides six flex channels plus IMU and magnetometer
callbacks.  Those signals are preserved as raw telemetry.  They are not a
verified wrist pose, so this source always declares ``provides_wrist_pose``
as ``False``.
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Protocol, Sequence

import numpy as np

from revo3_teleop.contracts import NativeSample, SampleHeader

from .common import Clock, SequenceGapTracker, finite_row, monotonic_ns, strict_nonnegative_int


FLEX_CHANNELS = 6
FLEX_SAMPLE_RATE_HZ = 50.0
IMU_SAMPLE_RATE_HZ = 100.0
MAG_SAMPLE_RATE_HZ = 20.0
GLOVE_CLOCK_DOMAIN = "host_callback_monotonic"


def _period_ns(sample_rate_hz: float) -> int:
    return int(round(1_000_000_000.0 / sample_rate_hz))


class BrainCoGloveClient(Protocol):
    def register_flex_callback(self, callback: Callable[[Sequence[Sequence[object]]], None]) -> None: ...

    def register_imu_callback(self, callback: Callable[[Sequence[Sequence[object]]], None]) -> None: ...

    def register_mag_callback(self, callback: Callable[[Sequence[Sequence[object]]], None]) -> None: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...


class BrainCoGloveParser:
    """Preserve official EDU rows without inventing kinematic semantics."""

    provides_wrist_pose = False

    def __init__(self, *, source_prefix: str = "brainco_glove") -> None:
        self.source_prefix = source_prefix
        self._trackers = {
            "flex": SequenceGapTracker(),
            "imu": SequenceGapTracker(),
            "mag": SequenceGapTracker(),
        }

    def _sample(
        self,
        *,
        stream: str,
        sequence: int,
        callback_timestamp_ns: int,
        reconstructed_capture_timestamp_ns: int | None,
        sample_rate_hz: float,
        dropped: int,
        payload: dict[str, np.ndarray],
    ) -> NativeSample:
        callback_ns = strict_nonnegative_int(callback_timestamp_ns, name="callback_timestamp_ns")
        capture_ns = (
            callback_ns
            if reconstructed_capture_timestamp_ns is None
            else strict_nonnegative_int(
                reconstructed_capture_timestamp_ns,
                name="reconstructed_capture_timestamp_ns",
            )
        )
        if capture_ns > callback_ns:
            raise ValueError("reconstructed capture time cannot follow callback arrival")
        evidence = {
            **payload,
            "sample_rate_hz": np.asarray([sample_rate_hz], dtype=np.float32),
            "host_callback_timestamp_ns": np.asarray([callback_ns], dtype=np.int64),
            "reconstructed_capture_timestamp_ns": np.asarray([capture_ns], dtype=np.int64),
            "callback_to_capture_latency_ns": np.asarray(
                [callback_ns - capture_ns], dtype=np.int64
            ),
            "device_timestamp_available": np.asarray([0], dtype=np.uint8),
            "provides_wrist_pose": np.asarray([0], dtype=np.uint8),
            "sequence_gap_packets": np.asarray([dropped], dtype=np.int64),
        }
        return NativeSample(
            SampleHeader(
                source_id=f"{self.source_prefix}_{stream}",
                sequence=sequence,
                capture_timestamp_ns=capture_ns,
                receive_timestamp_ns=callback_ns,
                clock_domain=GLOVE_CLOCK_DOMAIN,
                device_timestamp_ns=None,
                dropped_since_previous=dropped,
            ),
            evidence,
        )

    def parse_flex(
        self,
        row: Sequence[object],
        *,
        callback_timestamp_ns: int,
        reconstructed_capture_timestamp_ns: int | None = None,
    ) -> NativeSample:
        values = finite_row(row, name="BrainCo glove flex row", exact_size=1 + FLEX_CHANNELS)
        sequence = strict_nonnegative_int(values[0], name="flex sequence")
        dropped = self._trackers["flex"].observe(sequence)
        return self._sample(
            stream="flex",
            sequence=sequence,
            callback_timestamp_ns=callback_timestamp_ns,
            reconstructed_capture_timestamp_ns=reconstructed_capture_timestamp_ns,
            sample_rate_hz=FLEX_SAMPLE_RATE_HZ,
            dropped=dropped,
            payload={
                "flex_raw": values[1:].astype(np.float32, copy=True),
                "raw_packet": values.astype(np.float32, copy=True),
            },
        )

    def parse_imu(
        self,
        row: Sequence[object],
        *,
        callback_timestamp_ns: int,
        reconstructed_capture_timestamp_ns: int | None = None,
    ) -> NativeSample:
        values = finite_row(row, name="BrainCo glove IMU row")
        # The official helper has two layouts: compact seven-value rows, or
        # extended rows (>=13) where gyro occupies columns 7:10.  Reject the
        # ambiguous intermediate lengths instead of guessing.
        if values.size != 7 and values.size < 13:
            raise ValueError("BrainCo glove IMU row must contain 7 or at least 13 values")
        sequence = strict_nonnegative_int(values[0], name="IMU sequence")
        dropped = self._trackers["imu"].observe(sequence)
        gyro = values[7:10] if values.size >= 13 else values[4:7]
        return self._sample(
            stream="imu",
            sequence=sequence,
            callback_timestamp_ns=callback_timestamp_ns,
            reconstructed_capture_timestamp_ns=reconstructed_capture_timestamp_ns,
            sample_rate_hz=IMU_SAMPLE_RATE_HZ,
            dropped=dropped,
            payload={
                "imu_raw": values.astype(np.float32, copy=True),
                "acc_raw": values[1:4].astype(np.float32, copy=True),
                "gyro_raw": gyro.astype(np.float32, copy=True),
            },
        )

    def parse_mag(
        self,
        row: Sequence[object],
        *,
        callback_timestamp_ns: int,
        reconstructed_capture_timestamp_ns: int | None = None,
    ) -> NativeSample:
        values = finite_row(row, name="BrainCo glove magnetometer row", exact_size=4)
        sequence = strict_nonnegative_int(values[0], name="magnetometer sequence")
        dropped = self._trackers["mag"].observe(sequence)
        return self._sample(
            stream="mag",
            sequence=sequence,
            callback_timestamp_ns=callback_timestamp_ns,
            reconstructed_capture_timestamp_ns=reconstructed_capture_timestamp_ns,
            sample_rate_hz=MAG_SAMPLE_RATE_HZ,
            dropped=dropped,
            payload={
                "mag_raw": values[1:].astype(np.float32, copy=True),
                "raw_packet": values.astype(np.float32, copy=True),
            },
        )


class BrainCoGloveSource:
    """Injected callback adapter; construction never opens a serial device."""

    hardware_autostart = False
    provides_wrist_pose = False

    def __init__(
        self,
        *,
        client_factory: Callable[[], BrainCoGloveClient] | None = None,
        allow_hardware_start: bool = False,
        clock: Clock = monotonic_ns,
        parser: BrainCoGloveParser | None = None,
    ) -> None:
        self._client_factory = client_factory
        self._allow_hardware_start = bool(allow_hardware_start)
        self._clock = clock
        self._parser = parser or BrainCoGloveParser()
        self._client: BrainCoGloveClient | None = None
        self._queue: deque[NativeSample] = deque()

    def start(self) -> None:
        if not self._allow_hardware_start:
            raise PermissionError("BrainCo glove hardware start is disabled; opt in explicitly")
        if self._client_factory is None:
            raise RuntimeError("no injected BrainCo glove client factory")
        if self._client is not None:
            raise RuntimeError("BrainCo glove source is already started")
        client = self._client_factory()
        # Retain the concrete client before any callback/start operation.  A
        # failed or timed-out SDK start may still own module-global callbacks
        # or a live worker thread; dropping the only reference would make a
        # deliberate stop/recovery attempt impossible.
        self._client = client
        client.register_flex_callback(self._on_flex)
        client.register_imu_callback(self._on_imu)
        client.register_mag_callback(self._on_mag)
        client.start()

    def stop(self) -> None:
        if self._client is None:
            return
        client = self._client
        # Clear the reference only after a confirmed stop.  If stop raises or
        # times out, the source remains faulted-but-owned and a second start is
        # blocked until the same client is successfully stopped.
        client.stop()
        self._client = None

    def _ingest(
        self,
        rows: Sequence[Sequence[object]],
        parser: Callable[..., NativeSample],
        *,
        sample_rate_hz: float,
        callback_timestamp_ns: int | None = None,
    ) -> tuple[NativeSample, ...]:
        callback_ns = self._clock() if callback_timestamp_ns is None else int(callback_timestamp_ns)
        # EDU callbacks may contain more than one native-rate row.  Preserve
        # their order and back-date earlier rows at the documented nominal
        # period so a batch cannot create duplicate capture timestamps.  This
        # is still host-arrival reconstruction, never a device timestamp.
        count = len(rows)
        period_ns = _period_ns(sample_rate_hz)
        samples = tuple(
            parser(
                row,
                callback_timestamp_ns=callback_ns,
                reconstructed_capture_timestamp_ns=(
                    callback_ns - (count - 1 - index) * period_ns
                ),
            )
            for index, row in enumerate(rows)
        )
        self._queue.extend(samples)
        return samples

    def _on_flex(self, rows: Sequence[Sequence[object]]) -> None:
        self.ingest_flex(rows)

    def _on_imu(self, rows: Sequence[Sequence[object]]) -> None:
        self.ingest_imu(rows)

    def _on_mag(self, rows: Sequence[Sequence[object]]) -> None:
        self.ingest_mag(rows)

    def ingest_flex(self, rows: Sequence[Sequence[object]], *, callback_timestamp_ns: int | None = None) -> tuple[NativeSample, ...]:
        return self._ingest(
            rows,
            self._parser.parse_flex,
            sample_rate_hz=FLEX_SAMPLE_RATE_HZ,
            callback_timestamp_ns=callback_timestamp_ns,
        )

    def ingest_imu(self, rows: Sequence[Sequence[object]], *, callback_timestamp_ns: int | None = None) -> tuple[NativeSample, ...]:
        return self._ingest(
            rows,
            self._parser.parse_imu,
            sample_rate_hz=IMU_SAMPLE_RATE_HZ,
            callback_timestamp_ns=callback_timestamp_ns,
        )

    def ingest_mag(self, rows: Sequence[Sequence[object]], *, callback_timestamp_ns: int | None = None) -> tuple[NativeSample, ...]:
        return self._ingest(
            rows,
            self._parser.parse_mag,
            sample_rate_hz=MAG_SAMPLE_RATE_HZ,
            callback_timestamp_ns=callback_timestamp_ns,
        )

    def drain(self) -> tuple[NativeSample, ...]:
        samples = tuple(self._queue)
        self._queue.clear()
        return samples
