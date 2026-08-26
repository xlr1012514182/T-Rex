from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

from revo3_teleop.contracts import NativeSample, SampleHeader
from revo3_teleop.hardware.runner import RealSensorRunner, RealSensorRunnerConfig


def _sample(source: str, sequence: int, payload: dict[str, np.ndarray]) -> NativeSample:
    timestamp = time.monotonic_ns()
    return NativeSample(
        SampleHeader(
            source_id=source,
            sequence=sequence,
            capture_timestamp_ns=timestamp,
            receive_timestamp_ns=timestamp,
        ),
        payload,
    )


class _Sink:
    def __init__(self) -> None:
        self.rows: list[tuple[str, NativeSample]] = []

    def accept_sample(self, stream: str, sample: NativeSample) -> None:
        self.rows.append((stream, sample))


class _Revo:
    def __init__(self) -> None:
        self.state_sequence = 0
        self.touch_sequence = 0

    async def poll_state(self):
        result = _sample(
            "revo",
            self.state_sequence,
            {"q_rad": np.zeros(21, dtype=np.float32)},
        )
        self.state_sequence += 1
        return result

    async def poll_touch(self):
        result = _sample(
            "touch",
            self.touch_sequence,
            {"summary_mn": np.zeros(42, dtype=np.float32)},
        )
        self.touch_sequence += 1
        return result

    @property
    def episode_metadata(self):
        return {"u21vt_pressure_zone_projection": None}


class _FailingRevo(_Revo):
    async def poll_state(self):
        raise ValueError("injected acquisition failure")


class _VisionTouch:
    def __init__(self) -> None:
        self.sequence = 0
        self.started = 0
        self.stopped = 0

    def start(self) -> None:
        self.started += 1

    def poll_force6d(self):
        result = _sample(
            "visiontouch",
            self.sequence,
            {"features": np.ones((5, 6), dtype=np.float32)},
        )
        self.sequence += 1
        return result

    def stop(self) -> None:
        self.stopped += 1

    @property
    def episode_metadata(self):
        return {"visiontouch_force6d": {"output_shape": [5, 6]}}


class _Camera:
    def __init__(self) -> None:
        self.sequence = 0
        self.started = 0
        self.stopped = 0

    def start(self) -> None:
        self.started += 1

    def read(self):
        result = _sample(
            "camera",
            self.sequence,
            {"rgb": np.zeros((2, 2, 3), dtype=np.uint8)},
        )
        self.sequence += 1
        return result

    def stop(self) -> None:
        self.stopped += 1

    @property
    def episode_metadata(self):
        return {"camera_id": "fake-raw"}


class _Rectifier:
    @property
    def episode_metadata(self):
        return {"transform_hash": "fake-hash", "learned_or_dino_calibration": False}

    def rectify(self, raw: NativeSample) -> NativeSample:
        return NativeSample(
            SampleHeader(
                source_id="camera_rectified_fake",
                sequence=raw.header.sequence,
                capture_timestamp_ns=raw.header.capture_timestamp_ns,
                receive_timestamp_ns=raw.header.receive_timestamp_ns,
                device_timestamp_ns=raw.header.device_timestamp_ns,
            ),
            {"rgb": np.ones((2, 2, 3), dtype=np.uint8)},
        )


class _EMG:
    def __init__(self) -> None:
        self.callback = None
        self.started = 0
        self.stopped = 0

    def register_emg_callback(self, callback) -> None:
        self.callback = callback

    def start(self) -> None:
        self.started += 1
        self.callback([[0, 0, *np.zeros(160, dtype=np.float32).tolist()]])

    def stop(self) -> None:
        self.stopped += 1


class _FailingStopEMG(_EMG):
    def stop(self) -> None:
        super().stop()
        raise RuntimeError("injected EMG stop failure")


class _FailingStartEMG(_EMG):
    def start(self) -> None:
        super().start()
        raise RuntimeError("injected EMG partial-start failure")


def test_real_sensor_runner_preserves_separate_streams_and_stops_sources() -> None:
    async def scenario():
        sink, camera, emg = _Sink(), _Camera(), _EMG()
        runner = RealSensorRunner(
            sink=sink,
            revo=_Revo(),
            camera=camera,
            emg_client=emg,
            camera_rectifier=_Rectifier(),
            config=RealSensorRunnerConfig(
                revo_state_hz=100.0, tactile_hz=80.0, camera_hz=60.0
            ),
        )
        await runner.run(duration_s=0.04)
        streams = {stream for stream, _ in sink.rows}
        assert streams == {
            "camera_raw",
            "camera_rectified",
            "emg",
            "revo_state",
            "tactile_pressure",
        }
        camera_metadata = runner.camera_episode_metadata
        assert camera_metadata["camera_raw"]["stream_role"] == "physical_raw"
        assert camera_metadata["camera_rectified"]["stream_role"] == "derived_rectified"
        emg_sample = next(sample for stream, sample in sink.rows if stream == "emg")
        assert emg_sample.header.device_timestamp_ns is None
        assert emg_sample.payload["clock_is_host_reconstruction"].item() == 1
        assert camera.started == camera.stopped == 1
        assert emg.started == emg.stopped == 1

    asyncio.run(scenario())


def test_real_sensor_runner_keeps_visiontouch_force6d_separate_from_pressure() -> None:
    async def scenario():
        sink, camera, emg, visiontouch = _Sink(), _Camera(), _EMG(), _VisionTouch()
        runner = RealSensorRunner(
            sink=sink,
            revo=_Revo(),
            camera=camera,
            emg_client=emg,
            visiontouch=visiontouch,
            config=RealSensorRunnerConfig(
                revo_state_hz=100.0,
                tactile_hz=80.0,
                visiontouch_hz=80.0,
                camera_hz=60.0,
            ),
        )
        await runner.run(duration_s=0.04)
        by_stream = {}
        for stream, sample in sink.rows:
            by_stream.setdefault(stream, []).append(sample)
        assert "features" not in by_stream["tactile_pressure"][0].payload
        assert by_stream["tactile"][0].payload["features"].shape == (5, 6)
        metadata = runner.tactile_episode_metadata
        assert not metadata["tactile_pressure"]["exporter_features_eligible"]
        assert metadata["tactile"]["exporter_features_eligible"]
        assert visiontouch.started == visiontouch.stopped == 1

    asyncio.run(scenario())


def test_real_sensor_runner_best_effort_stops_every_source_after_cleanup_failure() -> None:
    async def scenario():
        sink, camera, emg, visiontouch = (
            _Sink(),
            _Camera(),
            _FailingStopEMG(),
            _VisionTouch(),
        )
        runner = RealSensorRunner(
            sink=sink,
            revo=_Revo(),
            camera=camera,
            emg_client=emg,
            visiontouch=visiontouch,
            config=RealSensorRunnerConfig(
                revo_state_hz=100.0,
                tactile_hz=80.0,
                visiontouch_hz=80.0,
                camera_hz=60.0,
            ),
        )
        with pytest.raises(RuntimeError, match="cleanup failed for emg") as exc:
            await runner.run(duration_s=0.02)
        assert isinstance(exc.value.__cause__, RuntimeError)
        assert camera.stopped == 1
        assert visiontouch.stopped == 1
        assert emg.stopped == 1

    asyncio.run(scenario())


def test_real_sensor_runner_preserves_acquisition_failure_over_cleanup_failure() -> None:
    async def scenario():
        sink, camera, emg, visiontouch = (
            _Sink(),
            _Camera(),
            _FailingStopEMG(),
            _VisionTouch(),
        )
        runner = RealSensorRunner(
            sink=sink,
            revo=_FailingRevo(),
            camera=camera,
            emg_client=emg,
            visiontouch=visiontouch,
            config=RealSensorRunnerConfig(
                revo_state_hz=100.0,
                tactile_hz=80.0,
                visiontouch_hz=80.0,
                camera_hz=60.0,
            ),
        )
        with pytest.raises(RuntimeError, match="cleanup also failed for emg") as exc:
            await runner.run(duration_s=0.2)
        assert isinstance(exc.value.__cause__, ValueError)
        assert str(exc.value.__cause__) == "injected acquisition failure"
        assert camera.stopped == visiontouch.stopped == emg.stopped == 1

    asyncio.run(scenario())


def test_real_sensor_runner_cleans_every_attempted_partial_start() -> None:
    async def scenario():
        sink, camera, emg, visiontouch = (
            _Sink(),
            _Camera(),
            _FailingStartEMG(),
            _VisionTouch(),
        )
        runner = RealSensorRunner(
            sink=sink,
            revo=_Revo(),
            camera=camera,
            emg_client=emg,
            visiontouch=visiontouch,
        )
        with pytest.raises(RuntimeError, match="real sensor runner failed") as exc:
            await runner.run(duration_s=0.02)
        assert isinstance(exc.value.__cause__, RuntimeError)
        assert "partial-start" in str(exc.value.__cause__)
        assert visiontouch.started == visiontouch.stopped == 1
        assert camera.started == camera.stopped == 1
        assert emg.started == emg.stopped == 1

    asyncio.run(scenario())
