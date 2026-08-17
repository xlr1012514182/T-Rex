"""Single-camera visual boundary for the Revo3 V1 runtime.

This module deliberately contains no detector, segmenter, SAM model, or
tracker.  It accepts one *already rectified* RGB capture, deterministically
derives the full and fixed-centre policy views, and checks image health.  A
coarse target region may be supplied by the VLM planner, but it must belong to
the exact same capture; without such evidence the gate fails closed to FAR.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
from math import hypot
from typing import Mapping

import numpy as np
from PIL import Image


MODEL_IMAGE_WIDTH = 384
MODEL_IMAGE_HEIGHT = 288
_MODEL_SHAPE = (MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH, 3)


def _rgb_u8(value: np.ndarray, *, name: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"{name} must have shape HxWx3, got {image.shape}.")
    if image.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8 RGB, got {image.dtype}.")
    if image.shape[0] < 2 or image.shape[1] < 2:
        raise ValueError(f"{name} is too small to derive policy views.")
    return np.ascontiguousarray(image)


def _resize_rgb(image: np.ndarray) -> np.ndarray:
    # LANCZOS matches the repository inference server's resize convention.
    resized = Image.fromarray(image, mode="RGB").resize(
        (MODEL_IMAGE_WIDTH, MODEL_IMAGE_HEIGHT), Image.Resampling.LANCZOS
    )
    return np.asarray(resized, dtype=np.uint8).copy()


@dataclass(frozen=True)
class SingleCameraViewConfig:
    """Frozen crop/calibration contract; coordinates are normalized xyxy."""

    calibration_hash: str
    center_crop_xyxy: tuple[float, float, float, float] = (0.20, 0.20, 0.80, 0.80)

    def __post_init__(self) -> None:
        if not self.calibration_hash.strip():
            raise ValueError("calibration_hash must identify a verified rectifier profile.")
        x0, y0, x1, y1 = (float(value) for value in self.center_crop_xyxy)
        if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
            raise ValueError("center_crop_xyxy must be normalized xyxy within [0,1].")
        object.__setattr__(self, "center_crop_xyxy", (x0, y0, x1, y1))


@dataclass(frozen=True)
class DerivedCameraViews:
    """Two policy views derived atomically from one physical capture."""

    capture_timestamp_ns: int
    sequence: int
    calibration_hash: str
    full: np.ndarray
    fixed_center: np.ndarray
    source_shape: tuple[int, int, int]
    center_crop_px: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        if self.capture_timestamp_ns < 0 or self.sequence < 0:
            raise ValueError("capture timestamp and sequence must be non-negative.")
        if not self.calibration_hash.strip():
            raise ValueError("calibration_hash must be non-empty.")
        full = _rgb_u8(self.full, name="full")
        center = _rgb_u8(self.fixed_center, name="fixed_center")
        if full.shape != _MODEL_SHAPE or center.shape != _MODEL_SHAPE:
            raise ValueError(
                "full and fixed_center must both be exactly 384x288 RGB "
                f"(array shape {_MODEL_SHAPE})."
            )
        object.__setattr__(self, "full", full.copy())
        object.__setattr__(self, "fixed_center", center.copy())

    @property
    def policy_images(self) -> Mapping[str, np.ndarray]:
        return {"full": self.full.copy(), "fixed_center": self.fixed_center.copy()}


class SingleCameraViewDeriver:
    """Derive full/centre views without any semantic image processing."""

    def __init__(self, config: SingleCameraViewConfig) -> None:
        self.config = config

    def derive(
        self,
        rectified_rgb: np.ndarray,
        *,
        capture_timestamp_ns: int,
        sequence: int,
        calibration_hash: str,
    ) -> DerivedCameraViews:
        if calibration_hash != self.config.calibration_hash:
            raise ValueError("rectifier calibration hash does not match the frozen profile.")
        source = _rgb_u8(rectified_rgb, name="rectified_rgb")
        height, width = source.shape[:2]
        x0n, y0n, x1n, y1n = self.config.center_crop_xyxy
        x0 = max(0, min(width - 1, int(round(x0n * width))))
        y0 = max(0, min(height - 1, int(round(y0n * height))))
        x1 = max(x0 + 1, min(width, int(round(x1n * width))))
        y1 = max(y0 + 1, min(height, int(round(y1n * height))))
        center = source[y0:y1, x0:x1]
        return DerivedCameraViews(
            capture_timestamp_ns=int(capture_timestamp_ns),
            sequence=int(sequence),
            calibration_hash=calibration_hash,
            full=_resize_rgb(source),
            fixed_center=_resize_rgb(center),
            source_shape=tuple(int(value) for value in source.shape),
            center_crop_px=(x0, y0, x1, y1),
        )


class VisualReadiness(str, Enum):
    FAR = "FAR"
    NEAR_CONTACT = "NEAR_CONTACT"
    INVALID = "INVALID"


class CameraHealthStatus(str, Enum):
    HEALTHY = "HEALTHY"
    INVALID = "INVALID"


@dataclass(frozen=True)
class CameraHealthConfig:
    """Pure camera-service checks; contains no task/target semantics."""

    calibration_hash: str
    max_frame_age_ns: int = 100_000_000
    min_mean_luma: float = 8.0
    max_mean_luma: float = 247.0
    min_focus_measure: float = 1.0
    frozen_after_identical_frames: int = 3

    def __post_init__(self) -> None:
        if not self.calibration_hash.strip():
            raise ValueError("calibration_hash must be non-empty.")
        if self.max_frame_age_ns <= 0 or self.frozen_after_identical_frames < 2:
            raise ValueError("frame age must be positive and frozen count at least two.")
        if not 0 <= self.min_mean_luma < self.max_mean_luma <= 255:
            raise ValueError("invalid luma bounds.")
        if self.min_focus_measure < 0:
            raise ValueError("min_focus_measure must be non-negative.")


@dataclass(frozen=True)
class CameraHealthResult:
    status: CameraHealthStatus
    reason: str
    capture_timestamp_ns: int
    frame_age_ns: int
    mean_luma: float
    focus_measure: float

    @property
    def healthy(self) -> bool:
        return self.status is CameraHealthStatus.HEALTHY


class CameraHealthMonitor:
    """The sole vision-front-end gate before Planner/VisualGate.

    This monitor checks only calibration identity, causal freshness, sequence,
    frozen frames, exposure and a simple focus statistic.  It never decides
    whether an object is near or centred; those semantics belong exclusively
    to ``planner.VisualGate``.
    """

    def __init__(self, config: CameraHealthConfig) -> None:
        self.config = config
        self._last_sequence: int | None = None
        self._last_timestamp_ns: int | None = None
        self._last_fingerprint: bytes | None = None
        self._identical_frames = 0

    def reset(self) -> None:
        self._last_sequence = None
        self._last_timestamp_ns = None
        self._last_fingerprint = None
        self._identical_frames = 0

    @staticmethod
    def _health(image: np.ndarray) -> tuple[float, float]:
        rgb = image.astype(np.float32)
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        dx = np.diff(gray, axis=1)
        dy = np.diff(gray, axis=0)
        focus = float(0.5 * (np.mean(dx * dx) + np.mean(dy * dy)))
        return float(gray.mean()), focus

    def evaluate(self, views: DerivedCameraViews, *, now_ns: int) -> CameraHealthResult:
        age = int(now_ns) - int(views.capture_timestamp_ns)
        mean_luma, focus = self._health(views.full)

        def result(status: CameraHealthStatus, reason: str) -> CameraHealthResult:
            return CameraHealthResult(
                status, reason, views.capture_timestamp_ns, age, mean_luma, focus
            )

        if views.calibration_hash != self.config.calibration_hash:
            return result(CameraHealthStatus.INVALID, "calibration_hash_mismatch")
        if age < 0:
            return result(CameraHealthStatus.INVALID, "future_capture")
        if age > self.config.max_frame_age_ns:
            return result(CameraHealthStatus.INVALID, "stale_capture")
        if self._last_sequence is not None:
            if views.sequence <= self._last_sequence:
                return result(CameraHealthStatus.INVALID, "non_increasing_sequence")
            if views.capture_timestamp_ns <= int(self._last_timestamp_ns):
                return result(CameraHealthStatus.INVALID, "non_increasing_capture_timestamp")
        fingerprint = hashlib.blake2b(views.full, digest_size=16).digest()
        self._identical_frames = (
            self._identical_frames + 1 if fingerprint == self._last_fingerprint else 1
        )
        self._last_fingerprint = fingerprint
        self._last_sequence = views.sequence
        self._last_timestamp_ns = views.capture_timestamp_ns
        if self._identical_frames >= self.config.frozen_after_identical_frames:
            return result(CameraHealthStatus.INVALID, "frozen_camera_frame")
        if mean_luma < self.config.min_mean_luma:
            return result(CameraHealthStatus.INVALID, "underexposed")
        if mean_luma > self.config.max_mean_luma:
            return result(CameraHealthStatus.INVALID, "overexposed")
        if focus < self.config.min_focus_measure:
            return result(CameraHealthStatus.INVALID, "blurred_or_textureless")
        return result(CameraHealthStatus.HEALTHY, "healthy")


@dataclass(frozen=True)
class TargetRegionEvidence:
    """VLM-produced coarse current-frame region, never a detector/tracker mask."""

    capture_timestamp_ns: int
    cx: float
    cy: float
    width: float
    height: float
    confidence: float
    near_ready: bool = True
    center_ready: bool = True

    def __post_init__(self) -> None:
        values = (self.cx, self.cy, self.width, self.height, self.confidence)
        if not all(np.isfinite(float(value)) for value in values):
            raise ValueError("target-region fields must be finite.")
        if not (0 <= self.cx <= 1 and 0 <= self.cy <= 1):
            raise ValueError("target centre must lie within [0,1].")
        if not (0 < self.width <= 1 and 0 < self.height <= 1):
            raise ValueError("target width/height must lie within (0,1].")
        if not 0 <= self.confidence <= 1:
            raise ValueError("target confidence must lie within [0,1].")

    @property
    def area(self) -> float:
        return float(self.width * self.height)

    @property
    def center_distance(self) -> float:
        return hypot(float(self.cx) - 0.5, float(self.cy) - 0.5)


@dataclass(frozen=True)
class VisualReadinessConfig:
    calibration_hash: str
    max_frame_age_ns: int = 100_000_000
    min_mean_luma: float = 8.0
    max_mean_luma: float = 247.0
    min_focus_measure: float = 1.0
    frozen_after_identical_frames: int = 3
    min_target_area: float = 0.15
    max_center_distance: float = 0.25
    min_target_confidence: float = 0.75

    def __post_init__(self) -> None:
        if not self.calibration_hash.strip():
            raise ValueError("calibration_hash must be non-empty.")
        if self.max_frame_age_ns <= 0 or self.frozen_after_identical_frames < 2:
            raise ValueError("frame age must be positive and frozen count at least two.")
        if not 0 <= self.min_mean_luma < self.max_mean_luma <= 255:
            raise ValueError("invalid luma bounds.")
        if self.min_focus_measure < 0:
            raise ValueError("min_focus_measure must be non-negative.")
        if not 0 < self.min_target_area <= 1:
            raise ValueError("min_target_area must lie within (0,1].")
        if self.max_center_distance < 0 or not 0 <= self.min_target_confidence <= 1:
            raise ValueError("invalid target readiness thresholds.")


@dataclass(frozen=True)
class VisualReadinessResult:
    status: VisualReadiness
    reason: str
    capture_timestamp_ns: int
    frame_age_ns: int
    mean_luma: float
    focus_measure: float
    target_area: float = 0.0
    target_center_distance: float = float("inf")


class VisualReadinessGate:
    """Image-health plus same-capture coarse-region gate.

    The gate cannot infer object occupancy without semantic evidence.  Missing,
    stale, or low-confidence target evidence therefore yields FAR rather than
    inventing a target from image texture.
    """

    def __init__(self, config: VisualReadinessConfig) -> None:
        self.config = config
        self._health_monitor = CameraHealthMonitor(
            CameraHealthConfig(
                calibration_hash=config.calibration_hash,
                max_frame_age_ns=config.max_frame_age_ns,
                min_mean_luma=config.min_mean_luma,
                max_mean_luma=config.max_mean_luma,
                min_focus_measure=config.min_focus_measure,
                frozen_after_identical_frames=config.frozen_after_identical_frames,
            )
        )

    def reset(self) -> None:
        self._health_monitor.reset()

    def evaluate(
        self,
        views: DerivedCameraViews,
        *,
        now_ns: int,
        target: TargetRegionEvidence | None,
    ) -> VisualReadinessResult:
        health = self._health_monitor.evaluate(views, now_ns=now_ns)
        age = health.frame_age_ns
        mean_luma, focus = health.mean_luma, health.focus_measure

        def result(status: VisualReadiness, reason: str) -> VisualReadinessResult:
            return VisualReadinessResult(
                status=status,
                reason=reason,
                capture_timestamp_ns=views.capture_timestamp_ns,
                frame_age_ns=age,
                mean_luma=mean_luma,
                focus_measure=focus,
                target_area=0.0 if target is None else target.area,
                target_center_distance=(
                    float("inf") if target is None else target.center_distance
                ),
            )

        if not health.healthy:
            return result(VisualReadiness.INVALID, health.reason)

        if target is None:
            return result(VisualReadiness.FAR, "no_same_capture_target_evidence")
        if target.capture_timestamp_ns != views.capture_timestamp_ns:
            return result(VisualReadiness.FAR, "target_evidence_not_from_capture")
        if target.confidence < self.config.min_target_confidence:
            return result(VisualReadiness.FAR, "low_target_confidence")
        if not target.near_ready or not target.center_ready:
            return result(VisualReadiness.FAR, "planner_reports_far_or_off_center")
        if target.area < self.config.min_target_area:
            return result(VisualReadiness.FAR, "target_area_too_small")
        if target.center_distance > self.config.max_center_distance:
            return result(VisualReadiness.FAR, "target_off_center")
        return result(VisualReadiness.NEAR_CONTACT, "near_contact")
