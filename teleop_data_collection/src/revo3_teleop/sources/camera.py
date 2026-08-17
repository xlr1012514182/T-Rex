"""Minimal, opt-in RGB camera source for teleoperation collection.

This module deliberately stops at acquisition.  It does not perform object
detection, segmentation, tracking, or fisheye rectification.  Camera identity,
calibration, and colour-conversion provenance are episode-level metadata;
individual samples contain only numeric pixels and numeric health evidence.

Construction is side-effect free.  A real device can only be opened by
calling :meth:`RgbCameraSource.start` after setting ``allow_hardware_start``.
OpenCV is an optional runtime dependency and is imported only when an
``OpenCvCameraClient`` is started.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Mapping, Protocol

import numpy as np

from revo3_teleop.contracts import NativeSample, SampleHeader

from .common import Clock, SequenceGapTracker, monotonic_ns, strict_nonnegative_int


SUPPORTED_NATIVE_COLOUR_ORDERS = frozenset({"RGB", "BGR"})
OUTPUT_COLOUR_ORDER = "RGB"
CAMERA_CLOCK_DOMAIN = "workstation_monotonic"


def _non_empty(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


def _positive_image_size(value: tuple[int, int]) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError("image_size_wh must contain (width, height)")
    width = strict_nonnegative_int(value[0], name="image width")
    height = strict_nonnegative_int(value[1], name="image height")
    if width < 1 or height < 1:
        raise ValueError("image dimensions must be positive")
    return width, height


def _finite_array(value: object, *, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array.copy()


@dataclass(frozen=True)
class CameraCalibration:
    """Episode-level intrinsic calibration evidence.

    ``image_size_wh`` is the resolution at which ``intrinsics`` and
    ``distortion`` were calibrated.  Frames with another shape are rejected;
    silently reusing calibration after a resolution change is unsafe.
    """

    image_size_wh: tuple[int, int]
    intrinsics: np.ndarray
    distortion: np.ndarray
    revision: str
    distortion_model: str = "unspecified"

    def __post_init__(self) -> None:
        object.__setattr__(self, "image_size_wh", _positive_image_size(self.image_size_wh))
        object.__setattr__(
            self,
            "intrinsics",
            _finite_array(self.intrinsics, name="intrinsics", shape=(3, 3)),
        )
        distortion = _finite_array(self.distortion, name="distortion")
        if distortion.ndim != 1:
            raise ValueError("distortion must be a one-dimensional coefficient vector")
        object.__setattr__(self, "distortion", distortion)
        object.__setattr__(self, "revision", _non_empty(self.revision, name="revision"))
        object.__setattr__(
            self,
            "distortion_model",
            _non_empty(self.distortion_model, name="distortion_model"),
        )

    def episode_metadata(self) -> dict[str, object]:
        """Return a JSON-serialisable copy for the episode manifest."""

        return {
            "image_size_wh": list(self.image_size_wh),
            "intrinsics": self.intrinsics.tolist(),
            "distortion": self.distortion.tolist(),
            "distortion_model": self.distortion_model,
            "calibration_revision": self.revision,
        }


@dataclass(frozen=True)
class CameraSourceConfig:
    """Stable episode configuration shared by every frame from one camera."""

    camera_id: str
    calibration: CameraCalibration
    native_colour_order: str = "BGR"
    capture_clock_provenance: str = "host_read_completion_monotonic"
    freeze_after_ns: int = 1_000_000_000
    extra_episode_metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "camera_id", _non_empty(self.camera_id, name="camera_id"))
        colour = str(self.native_colour_order).strip().upper()
        if colour not in SUPPORTED_NATIVE_COLOUR_ORDERS:
            raise ValueError(
                f"native_colour_order must be one of {sorted(SUPPORTED_NATIVE_COLOUR_ORDERS)}"
            )
        object.__setattr__(self, "native_colour_order", colour)
        object.__setattr__(
            self,
            "capture_clock_provenance",
            _non_empty(self.capture_clock_provenance, name="capture_clock_provenance"),
        )
        freeze_after_ns = strict_nonnegative_int(self.freeze_after_ns, name="freeze_after_ns")
        if freeze_after_ns < 1:
            raise ValueError("freeze_after_ns must be positive")
        object.__setattr__(self, "freeze_after_ns", freeze_after_ns)
        object.__setattr__(
            self,
            "extra_episode_metadata",
            MappingProxyType(dict(self.extra_episode_metadata)),
        )

    @property
    def expected_rgb_shape(self) -> tuple[int, int, int]:
        width, height = self.calibration.image_size_wh
        return height, width, 3

    def episode_metadata(self) -> dict[str, object]:
        """Return camera/calibration/conversion provenance for one episode.

        Colour information is intentionally recorded here rather than copied
        as text into every frame.
        """

        conversion = (
            "identity"
            if self.native_colour_order == OUTPUT_COLOUR_ORDER
            else "bgr_to_rgb_channel_reverse"
        )
        return {
            "camera_id": self.camera_id,
            "native_colour_order": self.native_colour_order,
            "stored_colour_order": OUTPUT_COLOUR_ORDER,
            "colour_conversion": conversion,
            "capture_clock_domain": CAMERA_CLOCK_DOMAIN,
            "capture_clock_provenance": self.capture_clock_provenance,
            "calibration": self.calibration.episode_metadata(),
            **dict(self.extra_episode_metadata),
        }


@dataclass(frozen=True)
class CameraRead:
    """One frame returned by an injected camera client.

    A client may omit ``sequence`` and mapped ``capture_timestamp_ns``.  The
    source then assigns a local sequence and uses host read-completion time as
    mapped capture time.  A raw device timestamp is retained separately and
    is never assumed to share the workstation clock domain.
    """

    image: np.ndarray
    sequence: int | None = None
    capture_timestamp_ns: int | None = None
    device_timestamp_ns: int | None = None


class CameraClient(Protocol):
    """Dependency-injected read interface used by :class:`RgbCameraSource`."""

    def start(self) -> None: ...

    def read(self) -> CameraRead | None: ...

    def stop(self) -> None: ...


class OpenCvCameraClient:
    """Optional OpenCV client whose constructor never opens a device."""

    def __init__(
        self,
        device: int | str,
        *,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
        backend: int | None = None,
    ) -> None:
        self._device = device
        self._width = width
        self._height = height
        self._fps = fps
        self._backend = backend
        self._capture: object | None = None

    def start(self) -> None:
        if self._capture is not None:
            raise RuntimeError("OpenCV camera is already started")
        try:
            import cv2  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on optional runtime
            raise RuntimeError(
                "OpenCV camera support requires the optional 'opencv-python' package"
            ) from exc
        capture = (
            cv2.VideoCapture(self._device)
            if self._backend is None
            else cv2.VideoCapture(self._device, self._backend)
        )
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"failed to open camera device {self._device!r}")
        if self._width is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._width))
        if self._height is not None:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._height))
        if self._fps is not None:
            capture.set(cv2.CAP_PROP_FPS, float(self._fps))
        self._capture = capture

    def read(self) -> CameraRead | None:
        if self._capture is None:
            raise RuntimeError("OpenCV camera is not started")
        ok, frame = self._capture.read()  # type: ignore[union-attr]
        if not ok or frame is None:
            return None
        # OpenCV exposes BGR frames and no trustworthy workstation-mapped
        # exposure timestamp through this interface.
        return CameraRead(image=np.asarray(frame))

    def stop(self) -> None:
        if self._capture is None:
            return
        capture, self._capture = self._capture, None
        capture.release()  # type: ignore[union-attr]


class RgbFrameParser:
    """Validate, convert, and attach health evidence to native camera frames."""

    def __init__(self, config: CameraSourceConfig) -> None:
        self.config = config
        self._sequence = SequenceGapTracker()
        self._last_capture_timestamp_ns: int | None = None
        self._last_rgb: np.ndarray | None = None
        self._repeat_run_length = 0
        self._repeat_started_ns: int | None = None

    def parse(
        self,
        image: np.ndarray,
        *,
        sequence: int,
        capture_timestamp_ns: int,
        receive_timestamp_ns: int,
        device_timestamp_ns: int | None = None,
    ) -> NativeSample:
        sequence = strict_nonnegative_int(sequence, name="camera sequence")
        capture_ns = strict_nonnegative_int(
            capture_timestamp_ns, name="capture_timestamp_ns"
        )
        receive_ns = strict_nonnegative_int(
            receive_timestamp_ns, name="receive_timestamp_ns"
        )
        if capture_ns > receive_ns:
            raise ValueError("camera capture timestamp cannot be in the future")
        if self._last_capture_timestamp_ns is not None and capture_ns <= self._last_capture_timestamp_ns:
            raise ValueError("camera capture timestamps must increase strictly")

        native = np.asarray(image)
        if native.dtype != np.uint8:
            raise TypeError(f"camera frame must be uint8, got {native.dtype}")
        if native.shape != self.config.expected_rgb_shape:
            raise ValueError(
                "camera frame shape changed or disagrees with calibration: "
                f"expected {self.config.expected_rgb_shape}, got {native.shape}"
            )
        rgb = (
            native.copy()
            if self.config.native_colour_order == OUTPUT_COLOUR_ORDER
            else native[..., ::-1].copy()
        )

        dropped = self._sequence.observe(sequence)
        repeated = self._last_rgb is not None and np.array_equal(rgb, self._last_rgb)
        if repeated:
            self._repeat_run_length += 1
            if self._repeat_started_ns is None:
                # The exact duplicate run begins at the preceding frame.
                assert self._last_capture_timestamp_ns is not None
                self._repeat_started_ns = self._last_capture_timestamp_ns
        else:
            self._repeat_run_length = 0
            self._repeat_started_ns = None
        frozen = bool(
            repeated
            and self._repeat_started_ns is not None
            and capture_ns - self._repeat_started_ns >= self.config.freeze_after_ns
        )

        device_ns = (
            None
            if device_timestamp_ns is None
            else strict_nonnegative_int(device_timestamp_ns, name="device_timestamp_ns")
        )
        sample = NativeSample(
            SampleHeader(
                source_id=f"camera_{self.config.camera_id}",
                sequence=sequence,
                capture_timestamp_ns=capture_ns,
                receive_timestamp_ns=receive_ns,
                clock_domain=CAMERA_CLOCK_DOMAIN,
                device_timestamp_ns=device_ns,
                valid=not frozen,
                dropped_since_previous=dropped,
            ),
            {
                "rgb": rgb,
                "frame_repeated": np.asarray([int(repeated)], dtype=np.uint8),
                "repeat_run_length": np.asarray([self._repeat_run_length], dtype=np.int64),
                "freeze_detected": np.asarray([int(frozen)], dtype=np.uint8),
            },
        )
        self._last_capture_timestamp_ns = capture_ns
        self._last_rgb = rgb.copy()
        return sample


class RgbCameraSource:
    """Side-effect-free camera source with both callback and read ingestion.

    ``ingest`` is the callback path and does not require a device to be open.
    ``read`` polls an injected client after an explicitly authorised start.
    Both paths pass through the same parser and health checks.
    """

    hardware_autostart = False

    def __init__(
        self,
        config: CameraSourceConfig,
        *,
        client_factory: Callable[[], CameraClient] | None = None,
        allow_hardware_start: bool = False,
        clock: Clock = monotonic_ns,
        parser: RgbFrameParser | None = None,
    ) -> None:
        self.config = config
        self._client_factory = client_factory
        self._allow_hardware_start = bool(allow_hardware_start)
        self._clock = clock
        self._parser = parser or RgbFrameParser(config)
        self._client: CameraClient | None = None
        self._queue: deque[NativeSample] = deque()
        self._next_sequence = 0

    @property
    def episode_metadata(self) -> dict[str, object]:
        return self.config.episode_metadata()

    def start(self) -> None:
        if not self._allow_hardware_start:
            raise PermissionError("camera hardware start is disabled; opt in explicitly")
        if self._client_factory is None:
            raise RuntimeError("no injected camera client factory")
        if self._client is not None:
            raise RuntimeError("camera source is already started")
        client = self._client_factory()
        client.start()
        self._client = client

    def stop(self) -> None:
        if self._client is None:
            return
        client, self._client = self._client, None
        client.stop()

    def _assign_sequence(self, supplied: int | None) -> int:
        if supplied is None:
            sequence = self._next_sequence
        else:
            sequence = strict_nonnegative_int(supplied, name="camera sequence")
        self._next_sequence = max(self._next_sequence, sequence + 1)
        return sequence

    def ingest(
        self,
        frame: CameraRead | np.ndarray,
        *,
        receive_timestamp_ns: int | None = None,
    ) -> NativeSample:
        """Callback-compatible ingestion path for an external camera loop."""

        read = frame if isinstance(frame, CameraRead) else CameraRead(image=np.asarray(frame))
        receive_ns = (
            self._clock()
            if receive_timestamp_ns is None
            else strict_nonnegative_int(receive_timestamp_ns, name="receive_timestamp_ns")
        )
        capture_ns = receive_ns if read.capture_timestamp_ns is None else read.capture_timestamp_ns
        sample = self._parser.parse(
            read.image,
            sequence=self._assign_sequence(read.sequence),
            capture_timestamp_ns=capture_ns,
            receive_timestamp_ns=receive_ns,
            device_timestamp_ns=read.device_timestamp_ns,
        )
        self._queue.append(sample)
        return sample

    def read(self) -> NativeSample | None:
        """Poll one frame from the authorised injected client."""

        if self._client is None:
            raise RuntimeError("camera source is not started")
        result = self._client.read()
        if result is None:
            return None
        return self.ingest(result)

    def drain(self) -> tuple[NativeSample, ...]:
        samples = tuple(self._queue)
        self._queue.clear()
        return samples


__all__ = [
    "CAMERA_CLOCK_DOMAIN",
    "OUTPUT_COLOUR_ORDER",
    "CameraCalibration",
    "CameraClient",
    "CameraRead",
    "CameraSourceConfig",
    "OpenCvCameraClient",
    "RgbCameraSource",
    "RgbFrameParser",
]
