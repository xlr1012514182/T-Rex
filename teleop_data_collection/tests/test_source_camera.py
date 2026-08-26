from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from revo3_teleop.sources.camera import (
    CameraCalibration,
    CameraRead,
    CameraSourceConfig,
    RgbCameraSource,
    RgbFrameParser,
)


def _config(*, freeze_after_ns: int = 100) -> CameraSourceConfig:
    return CameraSourceConfig(
        camera_id="wrist-fisheye-01",
        native_colour_order="BGR",
        freeze_after_ns=freeze_after_ns,
        calibration=CameraCalibration(
            image_size_wh=(2, 2),
            intrinsics=np.asarray(
                [[100.0, 0.0, 1.0], [0.0, 101.0, 1.0], [0.0, 0.0, 1.0]]
            ),
            distortion=np.asarray([0.1, -0.2, 0.0, 0.0]),
            distortion_model="fisheye_equidistant",
            revision="cal-2026-08-17-r1",
        ),
        extra_episode_metadata={"lens_serial": "lens-42"},
    )


def _bgr(value: int = 0) -> np.ndarray:
    frame = np.full((2, 2, 3), value, dtype=np.uint8)
    frame[0, 0] = np.asarray([1, 2, 3], dtype=np.uint8)
    return frame


def test_camera_metadata_is_episode_level_and_frame_is_uint8_rgb() -> None:
    config = _config()
    parser = RgbFrameParser(config)
    sample = parser.parse(
        _bgr(),
        sequence=7,
        capture_timestamp_ns=1_000,
        receive_timestamp_ns=1_010,
    )

    assert sample.header.source_id == "camera_wrist-fisheye-01"
    assert sample.header.sequence == 7
    assert sample.header.capture_timestamp_ns == 1_000
    assert sample.header.receive_timestamp_ns == 1_010
    assert sample.payload["rgb"].dtype == np.uint8
    assert sample.payload["rgb"].shape == (2, 2, 3)
    assert sample.payload["rgb"][0, 0].tolist() == [3, 2, 1]
    assert "intrinsics" not in sample.payload
    assert "distortion" not in sample.payload
    assert "camera_id" not in sample.payload

    metadata = config.episode_metadata()
    assert metadata["camera_id"] == "wrist-fisheye-01"
    assert metadata["colour_conversion"] == "bgr_to_rgb_channel_reverse"
    assert metadata["stored_colour_order"] == "RGB"
    assert metadata["calibration"]["calibration_revision"] == "cal-2026-08-17-r1"
    assert metadata["calibration"]["intrinsics"][0] == [100.0, 0.0, 1.0]


class _FakeCamera:
    def __init__(self, frames: list[CameraRead]) -> None:
        self.frames = deque(frames)
        self.start_calls = 0
        self.stop_calls = 0
        self.fail_start = False
        self.fail_stop_once = False

    def start(self) -> None:
        self.start_calls += 1
        if self.fail_start:
            raise RuntimeError("injected camera start failure")

    def read(self) -> CameraRead | None:
        return self.frames.popleft() if self.frames else None

    def stop(self) -> None:
        self.stop_calls += 1
        if self.fail_stop_once:
            self.fail_stop_once = False
            raise RuntimeError("injected camera stop failure")


def test_construction_does_not_open_device_and_start_requires_opt_in() -> None:
    fake = _FakeCamera([])
    source = RgbCameraSource(_config(), client_factory=lambda: fake)
    assert fake.start_calls == 0
    with pytest.raises(PermissionError, match="opt in explicitly"):
        source.start()
    assert fake.start_calls == 0

    source = RgbCameraSource(
        _config(), client_factory=lambda: fake, allow_hardware_start=True
    )
    assert fake.start_calls == 0
    source.start()
    assert fake.start_calls == 1
    source.stop()
    assert fake.stop_calls == 1


def test_camera_source_retains_failed_start_and_stop_client_for_cleanup() -> None:
    failed_start = _FakeCamera([])
    failed_start.fail_start = True
    source = RgbCameraSource(
        _config(), client_factory=lambda: failed_start, allow_hardware_start=True
    )
    with pytest.raises(RuntimeError, match="start failure"):
        source.start()
    failed_start.fail_start = False
    source.stop()

    failed_stop = _FakeCamera([])
    failed_stop.fail_stop_once = True
    source = RgbCameraSource(
        _config(), client_factory=lambda: failed_stop, allow_hardware_start=True
    )
    source.start()
    with pytest.raises(RuntimeError, match="stop failure"):
        source.stop()
    with pytest.raises(RuntimeError, match="already started"):
        source.start()
    source.stop()
    assert failed_stop.stop_calls == 2


def test_fake_read_and_callback_paths_share_validation_and_queue() -> None:
    fake = _FakeCamera(
        [
            CameraRead(
                image=_bgr(4),
                sequence=3,
                capture_timestamp_ns=1_000,
                device_timestamp_ns=55,
            )
        ]
    )
    source = RgbCameraSource(
        _config(),
        client_factory=lambda: fake,
        allow_hardware_start=True,
        clock=lambda: 1_010,
    )
    source.start()
    first = source.read()
    assert first is not None
    assert first.header.device_timestamp_ns == 55

    second = source.ingest(
        CameraRead(image=_bgr(5), sequence=5, capture_timestamp_ns=1_020),
        receive_timestamp_ns=1_030,
    )
    assert second.header.dropped_since_previous == 1
    assert [sample.header.sequence for sample in source.drain()] == [3, 5]
    assert source.drain() == ()
    source.stop()


def test_duplicate_pixels_are_flagged_and_long_duplicate_run_is_frozen() -> None:
    parser = RgbFrameParser(_config(freeze_after_ns=100))
    first = parser.parse(
        _bgr(), sequence=0, capture_timestamp_ns=1_000, receive_timestamp_ns=1_001
    )
    repeated = parser.parse(
        _bgr(), sequence=1, capture_timestamp_ns=1_050, receive_timestamp_ns=1_051
    )
    frozen = parser.parse(
        _bgr(), sequence=2, capture_timestamp_ns=1_100, receive_timestamp_ns=1_101
    )

    assert first.payload["frame_repeated"].item() == 0
    assert repeated.payload["frame_repeated"].item() == 1
    assert repeated.payload["freeze_detected"].item() == 0
    assert repeated.header.valid
    assert frozen.payload["repeat_run_length"].item() == 2
    assert frozen.payload["freeze_detected"].item() == 1
    assert not frozen.header.valid


def test_duplicate_sequence_timestamp_shape_and_future_capture_are_rejected() -> None:
    parser = RgbFrameParser(_config())
    parser.parse(
        _bgr(), sequence=0, capture_timestamp_ns=1_000, receive_timestamp_ns=1_001
    )

    with pytest.raises(ValueError, match="increase strictly"):
        parser.parse(
            _bgr(1), sequence=1, capture_timestamp_ns=1_000, receive_timestamp_ns=1_002
        )
    with pytest.raises(ValueError, match="future"):
        parser.parse(
            _bgr(1), sequence=1, capture_timestamp_ns=1_010, receive_timestamp_ns=1_009
        )
    with pytest.raises(ValueError, match="shape changed"):
        parser.parse(
            np.zeros((3, 2, 3), dtype=np.uint8),
            sequence=1,
            capture_timestamp_ns=1_010,
            receive_timestamp_ns=1_011,
        )
    with pytest.raises(TypeError, match="uint8"):
        parser.parse(
            np.zeros((2, 2, 3), dtype=np.float32),
            sequence=1,
            capture_timestamp_ns=1_010,
            receive_timestamp_ns=1_011,
        )
    with pytest.raises(ValueError, match="sequence must increase strictly"):
        parser.parse(
            _bgr(1), sequence=0, capture_timestamp_ns=1_010, receive_timestamp_ns=1_011
        )
