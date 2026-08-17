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
        self.sequence = 0

    async def poll_state(self):
        self.sequence += 1
        return _sample("revo", self.sequence, {"q_rad": np.zeros(21, np.float32)})

    async def poll_touch(self):
        self.sequence += 1
        return _sample("touch", self.sequence, {"summary_mn": np.zeros(42, np.float32)})

    @property
    def episode_metadata(self):
        return {"u21vt_pressure_zone_projection": None}


class _Camera:
    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0
        self.sequence = 0

    def start(self) -> None:
        self.started += 1

    def read(self):
        self.sequence += 1
        return _sample("camera", self.sequence, {"rgb": np.zeros((2, 2, 3), np.uint8)})

    def stop(self) -> None:
        self.stopped += 1

    @property
    def episode_metadata(self):
        return {"camera_id": "fake"}


class _Glove:
    def __init__(self) -> None:
        self.callbacks = {}
        self.started = 0
        self.stopped = 0

    def register_flex_callback(self, callback) -> None:
        self.callbacks["flex"] = callback

    def register_imu_callback(self, callback) -> None:
        self.callbacks["imu"] = callback

    def register_mag_callback(self, callback) -> None:
        self.callbacks["mag"] = callback

    def start(self) -> None:
        self.started += 1
        self.callbacks["flex"]([[1, 1, 2, 3, 4, 5, 6]])
        self.callbacks["imu"]([[1, 0, 0, 1, 2, 3, 4]])
        self.callbacks["mag"]([[1, 4, 5, 6]])

    def stop(self) -> None:
        self.stopped += 1


class _EMG:
    def register_emg_callback(self, callback) -> None:
        self.callback = callback

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None


def test_runner_forwards_glove_batches_through_existing_source_parser() -> None:
    async def scenario() -> None:
        sink, camera, glove = _Sink(), _Camera(), _Glove()
        runner = RealSensorRunner(
            sink=sink,
            revo=_Revo(),
            camera=camera,
            emg_client=None,
            glove_client=glove,
            config=RealSensorRunnerConfig(
                revo_state_hz=100.0,
                tactile_hz=80.0,
                camera_hz=60.0,
            ),
        )
        await runner.run(duration_s=0.03)
        by_stream = {stream: sample for stream, sample in sink.rows}
        assert by_stream["glove_flex"].payload["flex_raw"].shape == (6,)
        assert by_stream["glove_imu"].payload["provides_wrist_pose"].item() == 0
        assert by_stream["glove_mag"].header.device_timestamp_ns is None
        assert glove.started == glove.stopped == 1
        assert camera.started == camera.stopped == 1

    asyncio.run(scenario())


def test_runner_blocks_same_process_brainco_emg_and_glove() -> None:
    with pytest.raises(ValueError, match="separate acquisition processes"):
        RealSensorRunner(
            sink=_Sink(),
            revo=_Revo(),
            camera=_Camera(),
            emg_client=_EMG(),
            glove_client=_Glove(),
        )
