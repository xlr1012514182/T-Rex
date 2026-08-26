"""Capability-locked OpenCV client for RGB and fisheye cameras.

This module opens a camera only inside an explicitly authorised probe or
``start`` call.  It does not rectify, crop, detect, segment, or track.  A
fisheye camera is supported by storing its native frame together with the
``CameraCalibration(distortion_model='fisheye_equidistant')`` evidence in the
existing source.  Rectification, if desired, must be a separately versioned
preprocessing transform so raw evidence is not silently destroyed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from importlib import import_module
import json
from typing import Any

import numpy as np

from revo3_teleop.contracts import NativeSample, SampleHeader
from revo3_teleop.sources.camera import CameraRead, CameraSourceConfig
from revo3_teleop.sources.common import Clock, monotonic_ns


def _fingerprint(value: dict[str, object]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _positive_optional(value: int | float | None, *, name: str) -> int | float | None:
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be positive when supplied")
    return value


@dataclass(frozen=True)
class CameraProbeConfig:
    device: int | str
    requested_width: int | None = None
    requested_height: int | None = None
    requested_fps: float | None = None
    backend: int | None = None
    require_exact_resolution: bool = True
    fps_tolerance: float = 1.0

    def __post_init__(self) -> None:
        _positive_optional(self.requested_width, name="requested_width")
        _positive_optional(self.requested_height, name="requested_height")
        _positive_optional(self.requested_fps, name="requested_fps")
        if self.fps_tolerance < 0:
            raise ValueError("fps_tolerance must be non-negative")


@dataclass(frozen=True)
class OpenCvCameraCapability:
    device: int | str
    backend_id: int | None
    backend_name: str
    negotiated_width: int
    negotiated_height: int
    negotiated_fps: float
    native_colour_order: str
    frame_dtype: str
    first_frame_shape: tuple[int, int, int]
    capture_clock_domain: str
    capture_clock_provenance: str
    device_timestamp_available: bool

    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))

    def to_json(self) -> dict[str, object]:
        result = asdict(self)
        result["fingerprint"] = self.fingerprint()
        return result


def _matrix(value: object, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array.copy()


def _size_wh(value: tuple[int, int], *, name: str) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain (width, height)")
    width, height = int(value[0]), int(value[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"{name} dimensions must be positive")
    return width, height


@dataclass(frozen=True)
class FisheyeRectificationConfig:
    """Versioned OpenCV fisheye transform; never a learned calibration.

    The hash binds the source K/D, selected new-K, input/output dimensions and
    both calibration revisions.  Raw frames remain a separate physical stream;
    this config describes only a reproducible derived product.
    """

    input_size_wh: tuple[int, int]
    output_size_wh: tuple[int, int]
    intrinsics: np.ndarray
    distortion: np.ndarray
    new_intrinsics: np.ndarray
    source_calibration_revision: str
    rectification_revision: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "input_size_wh", _size_wh(self.input_size_wh, name="input_size_wh")
        )
        object.__setattr__(
            self,
            "output_size_wh",
            _size_wh(self.output_size_wh, name="output_size_wh"),
        )
        object.__setattr__(
            self,
            "intrinsics",
            _matrix(self.intrinsics, name="intrinsics", shape=(3, 3)),
        )
        distortion = np.asarray(self.distortion, dtype=np.float64).reshape(-1)
        if distortion.shape != (4,) or not np.isfinite(distortion).all():
            raise ValueError("OpenCV fisheye distortion must be finite with shape [4]")
        object.__setattr__(self, "distortion", distortion.copy())
        object.__setattr__(
            self,
            "new_intrinsics",
            _matrix(self.new_intrinsics, name="new_intrinsics", shape=(3, 3)),
        )
        for field_name in ("source_calibration_revision", "rectification_revision"):
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise ValueError(f"{field_name} must be non-empty")
            object.__setattr__(self, field_name, value)

    def transform_hash(self) -> str:
        evidence = {
            "schema": "opencv-fisheye-rectification-v1",
            "input_size_wh": list(self.input_size_wh),
            "output_size_wh": list(self.output_size_wh),
            "K": self.intrinsics.tolist(),
            "D": self.distortion.tolist(),
            "new_K": self.new_intrinsics.tolist(),
            "source_calibration_revision": self.source_calibration_revision,
            "rectification_revision": self.rectification_revision,
            "interpolation": "INTER_LINEAR",
            "border_mode": "BORDER_CONSTANT",
        }
        return _fingerprint(evidence)

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema": "opencv-fisheye-rectification-v1",
            "input_size_wh": list(self.input_size_wh),
            "output_size_wh": list(self.output_size_wh),
            "K": self.intrinsics.tolist(),
            "D": self.distortion.tolist(),
            "new_K": self.new_intrinsics.tolist(),
            "source_calibration_revision": self.source_calibration_revision,
            "rectification_revision": self.rectification_revision,
            "interpolation": "INTER_LINEAR",
            "border_mode": "BORDER_CONSTANT",
            "transform_hash": self.transform_hash(),
            "learned_or_dino_calibration": False,
        }

    def validate_source(self, source: CameraSourceConfig) -> None:
        calibration = source.calibration
        if calibration.distortion_model not in {
            "fisheye",
            "fisheye_equidistant",
            "opencv_fisheye",
        }:
            raise ValueError("rectification requires an explicit fisheye calibration model")
        if calibration.image_size_wh != self.input_size_wh:
            raise ValueError("rectification input size differs from camera calibration")
        if calibration.revision != self.source_calibration_revision:
            raise ValueError("rectification source calibration revision mismatch")
        if not np.array_equal(calibration.intrinsics, self.intrinsics):
            raise ValueError("rectification K differs from camera calibration")
        if not np.array_equal(calibration.distortion, self.distortion):
            raise ValueError("rectification D differs from camera calibration")


class OpenCvFisheyeRectifier:
    """Create a derived rectified sample while leaving the raw sample intact."""

    def __init__(
        self,
        config: FisheyeRectificationConfig,
        source: CameraSourceConfig,
        *,
        cv2_module: Any | None = None,
        clock: Clock = monotonic_ns,
    ) -> None:
        config.validate_source(source)
        self.config = config
        self.source = source
        self._cv2_module = cv2_module
        self._clock = clock
        self._maps: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def episode_metadata(self) -> dict[str, object]:
        return {
            "source_stream_role": "derived_from_camera_raw",
            "source_camera_id": self.source.camera_id,
            **self.config.to_metadata(),
        }

    def _rectification_maps(self) -> tuple[np.ndarray, np.ndarray]:
        if self._maps is None:
            cv2 = _cv2(self._cv2_module)
            fisheye = getattr(cv2, "fisheye", None)
            build = getattr(fisheye, "initUndistortRectifyMap", None)
            if not callable(build):
                raise RuntimeError("OpenCV build exposes no fisheye rectification API")
            maps = build(
                self.config.intrinsics,
                self.config.distortion.reshape(4, 1),
                np.eye(3, dtype=np.float64),
                self.config.new_intrinsics,
                self.config.output_size_wh,
                cv2.CV_32FC1,
            )
            if not isinstance(maps, tuple) or len(maps) != 2:
                raise RuntimeError("OpenCV fisheye map builder returned an invalid result")
            self._maps = (np.asarray(maps[0]), np.asarray(maps[1]))
        return self._maps

    def rectify(self, raw: NativeSample) -> NativeSample:
        if "rgb" not in raw.payload:
            raise ValueError("camera raw sample has no rgb payload")
        rgb = np.asarray(raw.payload["rgb"])
        input_width, input_height = self.config.input_size_wh
        if rgb.dtype != np.uint8 or rgb.shape != (input_height, input_width, 3):
            raise ValueError("camera raw sample does not match rectification input schema")
        cv2 = _cv2(self._cv2_module)
        map1, map2 = self._rectification_maps()
        rectified = np.asarray(
            cv2.remap(
                rgb,
                map1,
                map2,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
        )
        output_width, output_height = self.config.output_size_wh
        if rectified.dtype != np.uint8 or rectified.shape != (
            output_height,
            output_width,
            3,
        ):
            raise RuntimeError("OpenCV rectified frame does not match configured output schema")
        payload = {key: value for key, value in raw.payload.items() if key != "rgb"}
        rectification_complete_ns = max(
            int(self._clock()), raw.header.receive_timestamp_ns
        )
        payload.update(
            {
                "rgb": rectified,
                "derived_from_raw_sequence": np.asarray(
                    [raw.header.sequence], dtype=np.int64
                ),
                "derived_from_raw_receive_timestamp_ns": np.asarray(
                    [raw.header.receive_timestamp_ns], dtype=np.int64
                ),
                "rectification_complete_timestamp_ns": np.asarray(
                    [rectification_complete_ns], dtype=np.int64
                ),
                "rectification_applied": np.asarray([1], dtype=np.uint8),
            }
        )
        return NativeSample(
            SampleHeader(
                source_id=(
                    f"{raw.header.source_id}_rectified_"
                    f"{self.config.transform_hash()[:12]}"
                ),
                sequence=raw.header.sequence,
                capture_timestamp_ns=raw.header.capture_timestamp_ns,
                receive_timestamp_ns=rectification_complete_ns,
                clock_domain=raw.header.clock_domain,
                device_timestamp_ns=raw.header.device_timestamp_ns,
                valid=raw.header.valid,
                dropped_since_previous=raw.header.dropped_since_previous,
            ),
            payload,
        )


def _cv2(module: Any | None) -> Any:
    return import_module("cv2") if module is None else module


def _open_capture(cv2: Any, config: CameraProbeConfig) -> Any:
    capture = (
        cv2.VideoCapture(config.device)
        if config.backend is None
        else cv2.VideoCapture(config.device, config.backend)
    )
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"failed to open camera device {config.device!r}")
    if config.requested_width is not None:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(config.requested_width))
    if config.requested_height is not None:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(config.requested_height))
    if config.requested_fps is not None:
        capture.set(cv2.CAP_PROP_FPS, float(config.requested_fps))
    return capture


def _backend_name(capture: Any) -> str:
    method = getattr(capture, "getBackendName", None)
    if not callable(method):
        return "unavailable"
    try:
        value = str(method()).strip()
    except Exception:
        return "unavailable"
    return value or "unavailable"


def _negotiated(capture: Any, cv2: Any) -> tuple[int, int, float]:
    width = int(round(float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))))
    height = int(round(float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if width <= 0 or height <= 0 or not np.isfinite(fps) or fps < 0:
        raise RuntimeError("camera returned invalid negotiated properties")
    return width, height, fps


def _check_negotiated(
    config: CameraProbeConfig, width: int, height: int, fps: float
) -> None:
    if config.require_exact_resolution:
        if config.requested_width is not None and width != config.requested_width:
            raise RuntimeError(
                f"camera width negotiation failed: requested={config.requested_width}, got={width}"
            )
        if config.requested_height is not None and height != config.requested_height:
            raise RuntimeError(
                f"camera height negotiation failed: requested={config.requested_height}, got={height}"
            )
    if config.requested_fps is not None:
        if fps <= 0 or abs(fps - config.requested_fps) > config.fps_tolerance:
            raise RuntimeError(
                f"camera fps negotiation failed: requested={config.requested_fps}, got={fps}"
            )


def probe_opencv_camera(
    config: CameraProbeConfig,
    *,
    allow_hardware_probe: bool = False,
    cv2_module: Any | None = None,
) -> OpenCvCameraCapability:
    """Open/read/release one camera frame and return immutable evidence."""

    if not allow_hardware_probe:
        raise PermissionError("camera hardware probe is disabled")
    cv2 = _cv2(cv2_module)
    capture = _open_capture(cv2, config)
    try:
        width, height, fps = _negotiated(capture, cv2)
        _check_negotiated(config, width, height, fps)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError("camera opened but did not return a probe frame")
        array = np.asarray(frame)
        expected = (height, width, 3)
        if array.shape != expected or array.dtype != np.uint8:
            raise RuntimeError(
                f"camera probe frame contract mismatch: expected uint8{expected}, "
                f"got {array.dtype}{array.shape}"
            )
        return OpenCvCameraCapability(
            device=config.device,
            backend_id=config.backend,
            backend_name=_backend_name(capture),
            negotiated_width=width,
            negotiated_height=height,
            negotiated_fps=fps,
            native_colour_order="BGR",
            frame_dtype="uint8",
            first_frame_shape=expected,
            capture_clock_domain="workstation_monotonic",
            capture_clock_provenance="host_read_completion_monotonic",
            device_timestamp_available=False,
        )
    finally:
        capture.release()


class ProbedOpenCvCameraClient:
    """OpenCV client that can start only against a confirmed probe fingerprint."""

    def __init__(
        self,
        config: CameraProbeConfig,
        capability: OpenCvCameraCapability,
        *,
        confirmed_fingerprint: str,
        allow_hardware_start: bool = False,
        cv2_module: Any | None = None,
        clock: Clock = monotonic_ns,
    ) -> None:
        if capability.device != config.device or capability.backend_id != config.backend:
            raise ValueError("camera capability belongs to another configured device/backend")
        if str(confirmed_fingerprint) != capability.fingerprint():
            raise ValueError("camera capability fingerprint mismatch")
        self.config = config
        self.capability = capability
        self._allow_hardware_start = bool(allow_hardware_start)
        self._cv2_module = cv2_module
        self._clock = clock
        self._capture: Any | None = None

    def start(self) -> None:
        if not self._allow_hardware_start:
            raise PermissionError("camera hardware start is disabled")
        if self._capture is not None:
            raise RuntimeError("camera is already started")
        cv2 = _cv2(self._cv2_module)
        capture = _open_capture(cv2, self.config)
        # Retain the capture before validating negotiated properties.  Some
        # backends open the device successfully and only then reveal a schema
        # mismatch.  If release also fails, stop() must remain able to retry
        # instead of losing the only handle to a partially opened camera.
        self._capture = capture
        try:
            width, height, fps = _negotiated(capture, cv2)
            _check_negotiated(self.config, width, height, fps)
            if (width, height) != (
                self.capability.negotiated_width,
                self.capability.negotiated_height,
            ):
                raise RuntimeError("camera resolution changed since confirmed probe")
            if _backend_name(capture) != self.capability.backend_name:
                raise RuntimeError("camera backend changed since confirmed probe")
        except Exception as start_failure:
            try:
                self.stop()
            except BaseException as release_failure:
                raise RuntimeError(
                    "camera startup validation failed and capture release failed; "
                    "handle retained for retry or process/device intervention"
                ) from start_failure
            raise

    def read(self) -> CameraRead | None:
        if self._capture is None:
            raise RuntimeError("camera is not started")
        ok, frame = self._capture.read()
        read_complete_ns = int(self._clock())
        if not ok or frame is None:
            return None
        array = np.asarray(frame)
        if array.dtype != np.uint8 or array.shape != self.capability.first_frame_shape:
            raise RuntimeError("camera frame schema changed after start")
        # OpenCV does not expose a trustworthy exposure timestamp through this
        # interface.  The host read-completion time is explicitly retained as
        # reconstruction evidence; no device timestamp is fabricated.
        return CameraRead(
            image=array,
            capture_timestamp_ns=read_complete_ns,
            device_timestamp_ns=None,
        )

    def stop(self) -> None:
        if self._capture is None:
            return
        capture = self._capture
        capture.release()
        self._capture = None


__all__ = [
    "CameraProbeConfig",
    "FisheyeRectificationConfig",
    "OpenCvCameraCapability",
    "OpenCvFisheyeRectifier",
    "ProbedOpenCvCameraClient",
    "probe_opencv_camera",
]
