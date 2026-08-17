"""Native-rate real-sensor pump with no actuator command authority.

This runner is intentionally a source scheduler, not a safety authority.  It
can feed an existing ``CollectionSession.accept_sample`` method, while the
session and controller pipeline retain episode and actuator ownership.  EMG
and optional BrainCo glove callbacks write directly to the thread-safe sink so
no callback packets are lost through an unsynchronised polling queue.  Because
the EDU SDK callback namespace is module-global, one runner process accepts
EMG or glove, never both.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import threading
from typing import Callable, Mapping, Protocol, Sequence

from revo3_teleop.contracts import NativeSample
from revo3_teleop.hardware.brainco_edu import BrainCoEduSdkEMGClient
from revo3_teleop.hardware.brainco_glove import BrainCoEduSdkGloveClient
from revo3_teleop.hardware.revo3_sdk import Revo3TelemetrySource
from revo3_teleop.hardware.visiontouch import VisionTouchForce6DSource
from revo3_teleop.sources.brainco_emg import (
    EMG_PACKET_DURATION_NS,
    BrainCoEduEMGParser,
)
from revo3_teleop.sources.brainco_glove import BrainCoGloveSource
from revo3_teleop.sources.camera import RgbCameraSource
from revo3_teleop.sources.common import Clock, monotonic_ns


class NativeSampleSink(Protocol):
    def accept_sample(self, stream: str, sample: NativeSample) -> object: ...


class CameraSampleRectifier(Protocol):
    @property
    def episode_metadata(self) -> Mapping[str, object]: ...

    def rectify(self, raw: NativeSample) -> NativeSample: ...


@dataclass(frozen=True)
class RealSensorRunnerConfig:
    revo_state_hz: float = 60.0
    # Native 42-zone pressure telemetry; never exported as T-Rex Force6D.
    tactile_hz: float = 20.0
    visiontouch_hz: float = 20.0
    camera_hz: float = 30.0
    pressure_tactile_stream: str = "tactile_pressure"
    visiontouch_stream: str = "tactile"
    camera_raw_stream: str = "camera_raw"
    camera_rectified_stream: str = "camera_rectified"
    glove_flex_stream: str = "glove_flex"
    glove_imu_stream: str = "glove_imu"
    glove_mag_stream: str = "glove_mag"

    def __post_init__(self) -> None:
        for name, value in (
            ("revo_state_hz", self.revo_state_hz),
            ("tactile_hz", self.tactile_hz),
            ("visiontouch_hz", self.visiontouch_hz),
            ("camera_hz", self.camera_hz),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        for name, value in (
            ("camera_raw_stream", self.camera_raw_stream),
            ("camera_rectified_stream", self.camera_rectified_stream),
            ("pressure_tactile_stream", self.pressure_tactile_stream),
            ("visiontouch_stream", self.visiontouch_stream),
            ("glove_flex_stream", self.glove_flex_stream),
            ("glove_imu_stream", self.glove_imu_stream),
            ("glove_mag_stream", self.glove_mag_stream),
        ):
            if not str(value).strip():
                raise ValueError(f"{name} must be non-empty")
        if self.camera_raw_stream == self.camera_rectified_stream:
            raise ValueError("raw and rectified camera streams must be physically distinct")
        if self.pressure_tactile_stream == self.visiontouch_stream:
            raise ValueError("pressure-zone and VisionTouch Force6D streams must be distinct")


class RealSensorRunner:
    """Pump camera/Revo/U21VT and one EDU source at independent source rates.

    ``run`` never generates a Revo target and never changes the VLA/control
    frequency.  It stops all sources on the first callback/poll failure and
    propagates that failure to the caller, which must fault/quarantine the
    surrounding ``CollectionSession``.
    """

    def __init__(
        self,
        *,
        sink: NativeSampleSink,
        revo: Revo3TelemetrySource,
        camera: RgbCameraSource,
        emg_client: BrainCoEduSdkEMGClient | None,
        glove_client: BrainCoEduSdkGloveClient | None = None,
        visiontouch: VisionTouchForce6DSource | None = None,
        camera_rectifier: CameraSampleRectifier | None = None,
        config: RealSensorRunnerConfig = RealSensorRunnerConfig(),
        clock: Clock = monotonic_ns,
    ) -> None:
        if not callable(getattr(sink, "accept_sample", None)):
            raise TypeError("sink must expose accept_sample(stream, sample)")
        if emg_client is not None and glove_client is not None:
            raise ValueError(
                "BrainCo EDU EMG and glove must use separate acquisition processes; "
                "their SDK callbacks are module-global and device-ID routing is unverified"
            )
        self.sink = sink
        self.revo = revo
        self.camera = camera
        self.emg_client = emg_client
        self.glove_client = glove_client
        self.visiontouch = visiontouch
        self.camera_rectifier = camera_rectifier
        self.config = config
        self._clock = clock
        self._stop = threading.Event()
        self._failure_lock = threading.Lock()
        self._failure: BaseException | None = None
        self._emg_parser = BrainCoEduEMGParser(source_id="brainco_edu_emg_hardware")
        self._glove_source = BrainCoGloveSource(
            clock=clock,
            parser=None,
        )
        self._glove_callback_lock = threading.Lock()

    @property
    def camera_episode_metadata(self) -> dict[str, object]:
        """Manifest fragment that keeps physical and derived frames separate."""

        result: dict[str, object] = {
            self.config.camera_raw_stream: {
                "stream_role": "physical_raw",
                **self.camera.episode_metadata,
            }
        }
        if self.camera_rectifier is not None:
            result[self.config.camera_rectified_stream] = {
                "stream_role": "derived_rectified",
                "source_stream": self.config.camera_raw_stream,
                **dict(self.camera_rectifier.episode_metadata),
            }
        return result

    @property
    def tactile_episode_metadata(self) -> dict[str, object]:
        """Keep pressure zones separate from model-backed six-axis force."""

        result: dict[str, object] = {
            self.config.pressure_tactile_stream: {
                "stream_role": "raw_pressure_zones",
                "exporter_features_eligible": False,
                **self.revo.episode_metadata,
            }
        }
        if self.visiontouch is not None:
            result[self.config.visiontouch_stream] = {
                "stream_role": "visiontouch_force6d",
                "exporter_features_eligible": True,
                **self.visiontouch.episode_metadata,
            }
        return result

    def _set_failure(self, exc: BaseException) -> None:
        with self._failure_lock:
            if self._failure is None:
                self._failure = exc
        self._stop.set()

    def _emg_callback(self, rows: Sequence[Sequence[object]]) -> None:
        try:
            materialized = list(rows)
            callback_ns = int(self._clock())
            for index, row in enumerate(materialized):
                packet_end_ns = callback_ns - (
                    len(materialized) - 1 - index
                ) * EMG_PACKET_DURATION_NS
                sample = self._emg_parser.parse_row(
                    row,
                    callback_timestamp_ns=callback_ns,
                    reconstructed_packet_end_timestamp_ns=packet_end_ns,
                )
                self.sink.accept_sample("emg", sample)
        except Exception as exc:
            self._set_failure(exc)

    def _glove_callback(
        self,
        stream: str,
        ingest: Callable[..., tuple[NativeSample, ...]],
        rows: Sequence[Sequence[object]],
    ) -> None:
        try:
            # The SDK exposes process-global callbacks and does not promise
            # callback serialization.  Protect parser sequence trackers and
            # its temporary queue as one atomic batch operation.
            with self._glove_callback_lock:
                samples = ingest(rows)
                self._glove_source.drain()
            for sample in samples:
                self.sink.accept_sample(stream, sample)
        except Exception as exc:
            self._set_failure(exc)

    def _glove_flex_callback(self, rows: Sequence[Sequence[object]]) -> None:
        self._glove_callback(
            self.config.glove_flex_stream,
            self._glove_source.ingest_flex,
            rows,
        )

    def _glove_imu_callback(self, rows: Sequence[Sequence[object]]) -> None:
        self._glove_callback(
            self.config.glove_imu_stream,
            self._glove_source.ingest_imu,
            rows,
        )

    def _glove_mag_callback(self, rows: Sequence[Sequence[object]]) -> None:
        self._glove_callback(
            self.config.glove_mag_stream,
            self._glove_source.ingest_mag,
            rows,
        )

    async def _periodic(
        self,
        hz: float,
        acquire: Callable[[], object],
        stream: str | None,
        *,
        threaded: bool = False,
    ) -> None:
        period_ns = int(round(1_000_000_000.0 / hz))
        deadline_ns = int(self._clock())
        while not self._stop.is_set():
            try:
                value = (
                    await asyncio.to_thread(acquire)
                    if threaded
                    else acquire()
                )
                if asyncio.iscoroutine(value):
                    value = await value
                if value is not None and stream is not None:
                    self.sink.accept_sample(stream, value)  # type: ignore[arg-type]
                elif value is not None:
                    raise RuntimeError("periodic acquire returned a sample without a stream")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._set_failure(exc)
                return
            deadline_ns += period_ns
            delay_s = (deadline_ns - int(self._clock())) / 1_000_000_000.0
            if delay_s > 0:
                await asyncio.sleep(delay_s)
            else:
                # Do not issue a burst of stale catch-up reads after a stall.
                deadline_ns = int(self._clock())

    async def _poll_camera(self) -> None:
        raw = await asyncio.to_thread(self.camera.read)
        if raw is None:
            return
        self.sink.accept_sample(self.config.camera_raw_stream, raw)
        if self.camera_rectifier is not None:
            rectified = await asyncio.to_thread(self.camera_rectifier.rectify, raw)
            self.sink.accept_sample(self.config.camera_rectified_stream, rectified)

    async def run(self, *, duration_s: float | None = None) -> None:
        if duration_s is not None and duration_s <= 0:
            raise ValueError("duration_s must be positive when supplied")
        self._stop.clear()
        self._failure = None
        if self.emg_client is not None:
            self.emg_client.register_emg_callback(self._emg_callback)
        if self.glove_client is not None:
            self.glove_client.register_flex_callback(self._glove_flex_callback)
            self.glove_client.register_imu_callback(self._glove_imu_callback)
            self.glove_client.register_mag_callback(self._glove_mag_callback)
        # These flags mean ownership/start was attempted, not merely that the
        # call returned successfully.  A concrete adapter may partially open a
        # device and deliberately retain its handle before raising; cleanup
        # must still call stop() so that handle gets a bounded retry.
        camera_attempted = False
        emg_attempted = False
        glove_attempted = False
        visiontouch_attempted = False
        tasks: list[asyncio.Task[None]] = []
        cleanup_failures: list[tuple[str, BaseException]] = []
        run_failure: BaseException | None = None
        try:
            if self.visiontouch is not None:
                visiontouch_attempted = True
                await asyncio.to_thread(self.visiontouch.start)
            camera_attempted = True
            await asyncio.to_thread(self.camera.start)
            if self.glove_client is not None:
                glove_attempted = True
                await asyncio.to_thread(self.glove_client.start)
            if self.emg_client is not None:
                emg_attempted = True
                await asyncio.to_thread(self.emg_client.start)
            tasks = [
                asyncio.create_task(
                    self._periodic(
                        self.config.revo_state_hz,
                        self.revo.poll_state,
                        "revo_state",
                    )
                ),
                asyncio.create_task(
                    self._periodic(
                        self.config.tactile_hz,
                        self.revo.poll_touch,
                        self.config.pressure_tactile_stream,
                    )
                ),
                asyncio.create_task(
                    self._periodic(
                        self.config.camera_hz,
                        self._poll_camera,
                        None,
                    )
                ),
            ]
            if self.visiontouch is not None:
                tasks.append(
                    asyncio.create_task(
                        self._periodic(
                            self.config.visiontouch_hz,
                            self.visiontouch.poll_force6d,
                            self.config.visiontouch_stream,
                            threaded=True,
                        )
                    )
                )
            if duration_s is None:
                while not self._stop.is_set():
                    await asyncio.sleep(0.02)
            else:
                try:
                    await asyncio.wait_for(self._wait_until_stopped(), timeout=duration_s)
                except asyncio.TimeoutError:
                    self._stop.set()
        except BaseException as exc:
            run_failure = exc
        finally:
            self._stop.set()
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            if emg_attempted and self.emg_client is not None:
                try:
                    await asyncio.to_thread(self.emg_client.stop)
                except BaseException as exc:
                    cleanup_failures.append(("emg", exc))
            if glove_attempted and self.glove_client is not None:
                try:
                    await asyncio.to_thread(self.glove_client.stop)
                except BaseException as exc:
                    cleanup_failures.append(("glove", exc))
            if camera_attempted:
                try:
                    await asyncio.to_thread(self.camera.stop)
                except BaseException as exc:
                    cleanup_failures.append(("camera", exc))
            if visiontouch_attempted and self.visiontouch is not None:
                try:
                    await asyncio.to_thread(self.visiontouch.stop)
                except BaseException as exc:
                    cleanup_failures.append(("visiontouch", exc))
        terminal_failure = self._failure if self._failure is not None else run_failure
        if terminal_failure is not None:
            suffix = ""
            if cleanup_failures:
                suffix = "; cleanup also failed for " + ",".join(
                    name for name, _ in cleanup_failures
                )
            raise RuntimeError(
                "real sensor runner failed; quarantine the episode" + suffix
            ) from terminal_failure
        if cleanup_failures:
            names = ",".join(name for name, _ in cleanup_failures)
            raise RuntimeError(
                f"real sensor runner cleanup failed for {names}; quarantine the episode"
            ) from cleanup_failures[0][1]

    async def _wait_until_stopped(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(0.01)

    def request_stop(self) -> None:
        self._stop.set()


__all__ = [
    "CameraSampleRectifier",
    "NativeSampleSink",
    "RealSensorRunner",
    "RealSensorRunnerConfig",
]
