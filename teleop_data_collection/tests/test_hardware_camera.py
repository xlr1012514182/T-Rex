from __future__ import annotations

import numpy as np
import pytest

from revo3_teleop.hardware.camera import (
    CameraProbeConfig,
    FisheyeRectificationConfig,
    OpenCvFisheyeRectifier,
    ProbedOpenCvCameraClient,
    probe_opencv_camera,
)
from revo3_teleop.sources.camera import (
    CameraCalibration,
    CameraSourceConfig,
    RgbCameraSource,
    RgbFrameParser,
)


class _FakeCapture:
    def __init__(self, width: int = 2, height: int = 2, fps: float = 30.0) -> None:
        self.props = {3: float(width), 4: float(height), 5: fps}
        self.released = 0
        self.release_failures_remaining = 0

    def isOpened(self) -> bool:
        return True

    def set(self, prop: int, value: float) -> bool:
        self.props[prop] = value
        return True

    def get(self, prop: int) -> float:
        return self.props[prop]

    def read(self):
        width, height = int(self.props[3]), int(self.props[4])
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[0, 0] = [1, 2, 3]
        return True, frame

    def getBackendName(self) -> str:
        return "FAKE-CAP"

    def release(self) -> None:
        self.released += 1
        if self.release_failures_remaining:
            self.release_failures_remaining -= 1
            raise RuntimeError("injected capture release failure")


class _FakeCv2:
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5

    def __init__(self) -> None:
        self.captures: list[_FakeCapture] = []

    def VideoCapture(self, *args):
        capture = _FakeCapture()
        self.captures.append(capture)
        return capture


def test_camera_probe_and_client_are_two_explicit_hardware_gates() -> None:
    cv2 = _FakeCv2()
    config = CameraProbeConfig(
        device=0,
        requested_width=2,
        requested_height=2,
        requested_fps=30.0,
    )
    with pytest.raises(PermissionError, match="probe is disabled"):
        probe_opencv_camera(config, cv2_module=cv2)
    capability = probe_opencv_camera(
        config, allow_hardware_probe=True, cv2_module=cv2
    )
    assert capability.first_frame_shape == (2, 2, 3)
    assert capability.native_colour_order == "BGR"
    assert not capability.device_timestamp_available
    assert cv2.captures[0].released == 1

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        ProbedOpenCvCameraClient(
            config, capability, confirmed_fingerprint="wrong", cv2_module=cv2
        )
    client = ProbedOpenCvCameraClient(
        config,
        capability,
        confirmed_fingerprint=capability.fingerprint(),
        allow_hardware_start=True,
        cv2_module=cv2,
        clock=lambda: 1_000,
    )
    source = RgbCameraSource(
        CameraSourceConfig(
            camera_id="fisheye-01",
            native_colour_order="BGR",
            capture_clock_provenance=capability.capture_clock_provenance,
            calibration=CameraCalibration(
                image_size_wh=(2, 2),
                intrinsics=np.eye(3),
                distortion=np.zeros(4),
                distortion_model="fisheye_equidistant",
                revision="bench-cal-r1",
            ),
        ),
        client_factory=lambda: client,
        allow_hardware_start=True,
        clock=lambda: 1_001,
    )
    source.start()
    sample = source.read()
    assert sample is not None
    assert sample.header.capture_timestamp_ns == 1_000
    assert sample.header.receive_timestamp_ns == 1_001
    assert sample.header.device_timestamp_ns is None
    assert sample.payload["rgb"][0, 0].tolist() == [3, 2, 1]
    source.stop()


def test_camera_probe_rejects_negotiation_or_frame_schema_mismatch() -> None:
    class IgnoringCv2(_FakeCv2):
        def VideoCapture(self, *args):
            capture = _FakeCapture(width=3, height=2)

            def ignore_set(prop, value):
                return False

            capture.set = ignore_set
            self.captures.append(capture)
            return capture

    with pytest.raises(RuntimeError, match="width negotiation failed"):
        probe_opencv_camera(
            CameraProbeConfig(device=0, requested_width=2, requested_height=2),
            allow_hardware_probe=True,
            cv2_module=IgnoringCv2(),
        )

    class UnknownFpsCv2(_FakeCv2):
        def VideoCapture(self, *args):
            capture = _FakeCapture(fps=0.0)
            capture.set = lambda prop, value: prop != self.CAP_PROP_FPS
            self.captures.append(capture)
            return capture

    with pytest.raises(RuntimeError, match="fps negotiation failed"):
        probe_opencv_camera(
            CameraProbeConfig(device=0, requested_fps=30.0),
            allow_hardware_probe=True,
            cv2_module=UnknownFpsCv2(),
        )


def test_probed_camera_release_failure_keeps_capture_retryable() -> None:
    cv2 = _FakeCv2()
    config = CameraProbeConfig(device=0, requested_width=2, requested_height=2)
    capability = probe_opencv_camera(
        config, allow_hardware_probe=True, cv2_module=cv2
    )
    client = ProbedOpenCvCameraClient(
        config,
        capability,
        confirmed_fingerprint=capability.fingerprint(),
        allow_hardware_start=True,
        cv2_module=cv2,
    )
    source = RgbCameraSource(
        CameraSourceConfig(
            camera_id="fisheye-retry",
            native_colour_order="BGR",
            calibration=CameraCalibration(
                image_size_wh=(2, 2),
                intrinsics=np.eye(3),
                distortion=np.zeros(4),
                distortion_model="fisheye_equidistant",
                revision="bench-cal-retry",
            ),
        ),
        client_factory=lambda: client,
        allow_hardware_start=True,
    )
    source.start()
    capture = cv2.captures[-1]
    capture.release_failures_remaining = 1
    with pytest.raises(RuntimeError, match="capture release failure"):
        source.stop()
    source.stop()
    assert capture.released == 2


def test_probed_camera_partial_start_release_failure_keeps_capture_retryable() -> None:
    probe_cv2 = _FakeCv2()
    config = CameraProbeConfig(device=0, requested_width=2, requested_height=2)
    capability = probe_opencv_camera(
        config, allow_hardware_probe=True, cv2_module=probe_cv2
    )

    class MismatchedCv2(_FakeCv2):
        def VideoCapture(self, *args):
            capture = _FakeCapture(width=3, height=2)
            capture.set = lambda prop, value: False
            capture.release_failures_remaining = 1
            self.captures.append(capture)
            return capture

    runtime_cv2 = MismatchedCv2()
    client = ProbedOpenCvCameraClient(
        config,
        capability,
        confirmed_fingerprint=capability.fingerprint(),
        allow_hardware_start=True,
        cv2_module=runtime_cv2,
    )
    with pytest.raises(RuntimeError, match="handle retained") as exc:
        client.start()
    assert isinstance(exc.value.__cause__, RuntimeError)
    assert "width negotiation failed" in str(exc.value.__cause__)
    capture = runtime_cv2.captures[0]
    client.stop()
    assert capture.released == 2


def test_fisheye_rectification_is_versioned_derived_evidence() -> None:
    intrinsics = np.asarray(
        [[100.0, 0.0, 1.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]]
    )
    distortion = np.asarray([0.1, -0.01, 0.001, 0.0])
    source_config = CameraSourceConfig(
        camera_id="fisheye-01",
        native_colour_order="RGB",
        calibration=CameraCalibration(
            image_size_wh=(2, 2),
            intrinsics=intrinsics,
            distortion=distortion,
            distortion_model="fisheye_equidistant",
            revision="physical-calibration-r3",
        ),
    )
    config = FisheyeRectificationConfig(
        input_size_wh=(2, 2),
        output_size_wh=(3, 1),
        intrinsics=intrinsics,
        distortion=distortion,
        new_intrinsics=np.asarray(
            [[90.0, 0.0, 1.5], [0.0, 90.0, 0.5], [0.0, 0.0, 1.0]]
        ),
        source_calibration_revision="physical-calibration-r3",
        rectification_revision="rectify-r1",
    )

    class FakeFisheye:
        def __init__(self) -> None:
            self.calls = []

        def initUndistortRectifyMap(self, K, D, R, new_K, size, map_type):
            self.calls.append((K.copy(), D.copy(), R.copy(), new_K.copy(), size, map_type))
            width, height = size
            return (
                np.zeros((height, width), dtype=np.float32),
                np.zeros((height, width), dtype=np.float32),
            )

    class FakeRectifyCv2:
        CV_32FC1 = 5
        INTER_LINEAR = 1
        BORDER_CONSTANT = 0

        def __init__(self) -> None:
            self.fisheye = FakeFisheye()
            self.remap_calls = 0

        def remap(self, image, map1, map2, *, interpolation, borderMode):
            assert interpolation == self.INTER_LINEAR
            assert borderMode == self.BORDER_CONSTANT
            self.remap_calls += 1
            return np.full((*map1.shape, 3), 17, dtype=np.uint8)

    cv2 = FakeRectifyCv2()
    rectifier = OpenCvFisheyeRectifier(
        config, source_config, cv2_module=cv2, clock=lambda: 1_100
    )
    raw_rgb = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    raw = RgbFrameParser(source_config).parse(
        raw_rgb,
        sequence=4,
        capture_timestamp_ns=1_000,
        receive_timestamp_ns=1_001,
    )
    raw_copy = raw.payload["rgb"].copy()
    derived = rectifier.rectify(raw)

    np.testing.assert_array_equal(raw.payload["rgb"], raw_copy)
    assert derived.payload["rgb"].shape == (1, 3, 3)
    assert derived.payload["derived_from_raw_sequence"].item() == 4
    assert derived.header.capture_timestamp_ns == raw.header.capture_timestamp_ns
    assert derived.header.receive_timestamp_ns == 1_100
    assert derived.payload["derived_from_raw_receive_timestamp_ns"].item() == 1_001
    assert derived.header.source_id != raw.header.source_id
    assert config.transform_hash()[:12] in derived.header.source_id
    assert rectifier.episode_metadata["transform_hash"] == config.transform_hash()
    assert rectifier.episode_metadata["learned_or_dino_calibration"] is False
    assert len(cv2.fisheye.calls) == 1

    pinhole_source = CameraSourceConfig(
        camera_id="not-fisheye",
        native_colour_order="RGB",
        calibration=CameraCalibration(
            image_size_wh=(2, 2),
            intrinsics=intrinsics,
            distortion=distortion,
            distortion_model="pinhole",
            revision="physical-calibration-r3",
        ),
    )
    with pytest.raises(ValueError, match="fisheye calibration"):
        OpenCvFisheyeRectifier(config, pinhole_source, cv2_module=cv2)
