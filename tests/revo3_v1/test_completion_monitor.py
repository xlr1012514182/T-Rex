import numpy as np

from revo3_v1.revo import (
    CompletionConfig,
    CompletionMonitor,
    CompletionPhase,
    CompletionStatus,
    RevoState,
)
from revo3_v1.tactile import TactileFrame


def _touch(timestamp_ns: int, force: float) -> TactileFrame:
    f6 = np.zeros((5, 6), dtype=np.float32)
    f6[:, 2] = force
    return TactileFrame(timestamp_ns, f6, sequence=timestamp_ns)


def _state(timestamp_ns: int, q: float = 0.2) -> RevoState:
    return RevoState(timestamp_ns, np.full(21, q, dtype=np.float32))


def test_stable_grasp_reports_hold_signal_not_terminal_completion():
    monitor = CompletionMonitor(CompletionConfig.demo())
    nominal = np.full(21, 0.2, dtype=np.float32)
    result = None
    for now in (0, 100_000_000, 300_000_000, 500_000_000):
        result = monitor.update(
            phase=CompletionPhase.CONTACT_BUILD,
            state=_state(now),
            tactile=_touch(now, 0.2),
            nominal_q_rad=nominal,
            now_ns=now,
        )
    assert result is not None
    assert result.status is CompletionStatus.GRASP_STABLE
    assert result.reason == "stable_grasp_hold"


def test_contact_established_is_a_single_edge_before_stable_hold():
    monitor = CompletionMonitor(CompletionConfig.demo())
    nominal = np.full(21, 0.2, dtype=np.float32)
    first = monitor.update(
        phase=CompletionPhase.CONTACT_BUILD,
        state=_state(0), tactile=_touch(0, 0.2),
        nominal_q_rad=nominal, now_ns=0,
    )
    assert first.status is CompletionStatus.IN_PROGRESS
    edge = monitor.update(
        phase=CompletionPhase.CONTACT_BUILD,
        state=_state(100_000_000), tactile=_touch(100_000_000, 0.2),
        nominal_q_rad=nominal, now_ns=100_000_000,
    )
    assert edge.status is CompletionStatus.CONTACT_ESTABLISHED
    next_tick = monitor.update(
        phase=CompletionPhase.CONTACT_BUILD,
        state=_state(200_000_000), tactile=_touch(200_000_000, 0.2),
        nominal_q_rad=nominal, now_ns=200_000_000,
    )
    assert next_tick.status is CompletionStatus.IN_PROGRESS


def test_release_is_only_completed_inside_controlled_release_after_dwell():
    monitor = CompletionMonitor(CompletionConfig.demo())
    nominal = np.zeros(21, dtype=np.float32)
    first = monitor.update(
        phase=CompletionPhase.CONTROLLED_RELEASE,
        state=_state(1_000_000_000, q=0.0),
        tactile=_touch(1_000_000_000, 0.0),
        nominal_q_rad=nominal,
        now_ns=1_000_000_000,
    )
    assert first.status is CompletionStatus.IN_PROGRESS
    done = monitor.update(
        phase=CompletionPhase.CONTROLLED_RELEASE,
        state=_state(1_200_000_000, q=0.0),
        tactile=_touch(1_200_000_000, 0.0),
        nominal_q_rad=nominal,
        now_ns=1_200_000_000,
    )
    assert done.status is CompletionStatus.RELEASED


def test_no_progress_and_stale_touch_are_fail_closed():
    config = CompletionConfig.demo()
    monitor = CompletionMonitor(config)
    nominal = np.full(21, 0.5, dtype=np.float32)
    monitor.update(
        phase=CompletionPhase.PRECONTACT_RUN,
        state=_state(0, q=0.0),
        tactile=_touch(0, 0.0),
        nominal_q_rad=nominal,
        now_ns=0,
    )
    no_progress = monitor.update(
        phase=CompletionPhase.PRECONTACT_RUN,
        state=_state(config.no_progress_timeout_ns, q=0.0),
        tactile=_touch(config.no_progress_timeout_ns, 0.0),
        nominal_q_rad=nominal,
        now_ns=config.no_progress_timeout_ns,
    )
    assert no_progress.status is CompletionStatus.NO_PROGRESS

    stale = monitor.update(
        phase=CompletionPhase.STABLE_HOLD,
        state=_state(3_000_000_000),
        tactile=_touch(3_000_000_000 - config.max_touch_age_ns - 1, 0.2),
        nominal_q_rad=np.full(21, 0.2),
        now_ns=3_000_000_000,
    )
    assert stale.status is CompletionStatus.IN_PROGRESS
    assert stale.reason == "touch_stale_hold"


def test_real_completion_thresholds_require_calibration_identity():
    try:
        CompletionConfig(
            contact_force_threshold=np.full(5, 0.1),
            release_force_threshold=np.full(5, 0.03),
            safe_open_q_rad=np.zeros(21),
        )
    except ValueError as exc:
        assert "calibration_id" in str(exc)
    else:
        raise AssertionError("unidentified physical thresholds were accepted")
