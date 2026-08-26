import numpy as np

from revo3_v1.vision import (
    CameraHealthConfig,
    CameraHealthMonitor,
    SingleCameraViewConfig,
    SingleCameraViewDeriver,
    TargetRegionEvidence,
    VisualReadiness,
    VisualReadinessConfig,
    VisualReadinessGate,
)


CALIBRATION = "sha256:verified-rectifier-unit-test"


def _frame(shift: int = 0) -> np.ndarray:
    yy, xx = np.indices((480, 640))
    return np.stack(
        (
            (xx + shift) % 256,
            (2 * yy + shift) % 256,
            (xx + yy + shift) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)


def _views(timestamp: int, sequence: int, shift: int = 0):
    return SingleCameraViewDeriver(
        SingleCameraViewConfig(calibration_hash=CALIBRATION)
    ).derive(
        _frame(shift),
        capture_timestamp_ns=timestamp,
        sequence=sequence,
        calibration_hash=CALIBRATION,
    )


def test_full_and_fixed_center_share_one_capture_and_are_exact_384x288():
    views = _views(1_000, 7)
    assert views.capture_timestamp_ns == 1_000
    assert views.full.shape == (288, 384, 3)
    assert views.fixed_center.shape == (288, 384, 3)
    assert views.full.dtype == views.fixed_center.dtype == np.uint8
    assert set(views.policy_images) == {"full", "fixed_center"}
    # The crop is a real deterministic derived view, not an undeclared copy.
    assert not np.array_equal(views.full, views.fixed_center)


def test_view_derivation_rejects_unreviewed_rectifier_profile():
    deriver = SingleCameraViewDeriver(
        SingleCameraViewConfig(calibration_hash=CALIBRATION)
    )
    try:
        deriver.derive(
            _frame(),
            capture_timestamp_ns=1,
            sequence=1,
            calibration_hash="sha256:other",
        )
    except ValueError as exc:
        assert "calibration hash" in str(exc)
    else:  # pragma: no cover - makes failure message clearer than bare pytest
        raise AssertionError("unreviewed rectifier profile was accepted")


def test_readiness_has_only_far_near_contact_invalid_without_detector():
    gate = VisualReadinessGate(
        VisualReadinessConfig(calibration_hash=CALIBRATION)
    )
    first = _views(1_000, 1)
    assert gate.evaluate(first, now_ns=1_001, target=None).status is VisualReadiness.FAR

    second = _views(2_000, 2, shift=1)
    far = gate.evaluate(
        second,
        now_ns=2_001,
        target=TargetRegionEvidence(2_000, 0.5, 0.5, 0.2, 0.2, 0.95),
    )
    assert far.status is VisualReadiness.FAR
    assert far.reason == "target_area_too_small"

    third = _views(3_000, 3, shift=2)
    near = gate.evaluate(
        third,
        now_ns=3_001,
        target=TargetRegionEvidence(3_000, 0.51, 0.49, 0.5, 0.5, 0.95),
    )
    assert near.status is VisualReadiness.NEAR_CONTACT

    fourth = _views(4_000, 4, shift=3)
    mismatch = gate.evaluate(
        fourth,
        now_ns=4_001,
        target=TargetRegionEvidence(3_000, 0.5, 0.5, 0.5, 0.5, 0.95),
    )
    assert mismatch.status is VisualReadiness.FAR
    assert mismatch.reason == "target_evidence_not_from_capture"


def test_mainline_camera_health_has_no_target_area_or_center_semantics():
    monitor = CameraHealthMonitor(CameraHealthConfig(calibration_hash=CALIBRATION))
    healthy = monitor.evaluate(_views(1_000, 1), now_ns=1_001)
    assert healthy.healthy
    assert healthy.reason == "healthy"
    assert not hasattr(healthy, "target_area")


def test_stale_or_frozen_camera_is_invalid_not_near():
    config = VisualReadinessConfig(
        calibration_hash=CALIBRATION,
        max_frame_age_ns=10,
        frozen_after_identical_frames=2,
    )
    stale_gate = VisualReadinessGate(config)
    stale = stale_gate.evaluate(_views(1, 1), now_ns=100, target=None)
    assert stale.status is VisualReadiness.INVALID
    assert stale.reason == "stale_capture"

    frozen_gate = VisualReadinessGate(config)
    frozen_gate.evaluate(_views(100, 1), now_ns=100, target=None)
    frozen = frozen_gate.evaluate(_views(101, 2), now_ns=101, target=None)
    assert frozen.status is VisualReadiness.INVALID
    assert frozen.reason == "frozen_camera_frame"
