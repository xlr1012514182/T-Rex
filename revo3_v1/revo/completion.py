"""Small, deterministic completion monitor for short Revo3 grasp tasks.

The monitor is intentionally not a second Task Executive.  It only converts
fresh Revo/tactile evidence into one completion signal.  Stable grasp means
HOLD, not terminal completion; release completion is evaluated only after an
upstream explicit release has selected ``CONTROLLED_RELEASE``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from revo3_v1.tactile import FINGER_COUNT, TactileFrame

from .contracts import JOINT_COUNT, RevoState, assert_joint_vector


class CompletionPhase(str, Enum):
    PRECONTACT_RUN = "precontact_run"
    CONTACT_BUILD = "contact_build"
    STABLE_HOLD = "stable_hold"
    CONTROLLED_RELEASE = "controlled_release"
    ABORTED = "aborted"


class CompletionStatus(str, Enum):
    IN_PROGRESS = "IN_PROGRESS"
    CONTACT_ESTABLISHED = "CONTACT_ESTABLISHED"
    GRASP_STABLE = "GRASP_STABLE"
    TASK_SUCCESS = "TASK_SUCCESS"
    NO_PROGRESS = "NO_PROGRESS"
    FAILED = "FAILED"
    RELEASED = "RELEASED"


def _finger_threshold(value: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (FINGER_COUNT,) or not np.isfinite(arr).all() or np.any(arr <= 0):
        raise ValueError(f"{name} must contain five finite positive calibrated values.")
    return arr.copy()


@dataclass(frozen=True)
class CompletionConfig:
    contact_force_threshold: np.ndarray
    release_force_threshold: np.ndarray
    safe_open_q_rad: np.ndarray
    normal_force_axis: int = 2
    min_contact_fingers: int = 2
    contact_dwell_ns: int = 100_000_000
    stable_dwell_ns: int = 400_000_000
    release_dwell_ns: int = 200_000_000
    no_progress_timeout_ns: int = 2_000_000_000
    max_state_age_ns: int = 50_000_000
    max_touch_age_ns: int = 150_000_000
    max_stable_velocity_rad_s: float = 0.08
    max_stable_tracking_error_rad: float = 0.08
    max_force_change_per_sample: float = 0.10
    min_progress_rad: float = 0.005
    safe_open_tolerance_rad: float = 0.08
    simulation_only: bool = False
    calibration_id: str = ""

    def __post_init__(self) -> None:
        contact = _finger_threshold(
            self.contact_force_threshold, name="contact_force_threshold"
        )
        release = _finger_threshold(
            self.release_force_threshold, name="release_force_threshold"
        )
        if np.any(release >= contact):
            raise ValueError("release thresholds must be below contact thresholds.")
        safe_open = assert_joint_vector(self.safe_open_q_rad, name="safe_open_q_rad")
        if not 0 <= self.normal_force_axis < 6:
            raise ValueError("normal_force_axis must be in [0,5].")
        if not 1 <= self.min_contact_fingers <= FINGER_COUNT:
            raise ValueError("min_contact_fingers must be in [1,5].")
        for name in (
            "contact_dwell_ns",
            "stable_dwell_ns",
            "release_dwell_ns",
            "no_progress_timeout_ns",
            "max_state_age_ns",
            "max_touch_age_ns",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive.")
        for name in (
            "max_stable_velocity_rad_s",
            "max_stable_tracking_error_rad",
            "max_force_change_per_sample",
            "min_progress_rad",
            "safe_open_tolerance_rad",
        ):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive.")
        if not self.simulation_only and not self.calibration_id.strip():
            raise ValueError("real completion config requires a calibration_id.")
        object.__setattr__(self, "contact_force_threshold", contact)
        object.__setattr__(self, "release_force_threshold", release)
        object.__setattr__(self, "safe_open_q_rad", safe_open)

    @classmethod
    def demo(cls) -> "CompletionConfig":
        return cls(
            contact_force_threshold=np.full(FINGER_COUNT, 0.10, dtype=np.float32),
            release_force_threshold=np.full(FINGER_COUNT, 0.03, dtype=np.float32),
            safe_open_q_rad=np.zeros(JOINT_COUNT, dtype=np.float32),
            simulation_only=True,
            calibration_id="simulation-only",
        )


@dataclass(frozen=True)
class CompletionResult:
    status: CompletionStatus
    reason: str
    contact_mask: np.ndarray
    contact_count: int
    timestamp_ns: int


class CompletionMonitor:
    """Minimal contact/stability/progress/release monitor."""

    def __init__(self, config: CompletionConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._contact_since_ns: int | None = None
        self._stable_since_ns: int | None = None
        self._release_since_ns: int | None = None
        self._last_progress_ns: int | None = None
        self._last_q: np.ndarray | None = None
        self._last_normal_force: np.ndarray | None = None
        self._last_contact_count = 0
        self._grasp_stable = False
        self._contact_event_emitted = False

    def _result(
        self,
        status: CompletionStatus,
        reason: str,
        contact: np.ndarray,
        now_ns: int,
    ) -> CompletionResult:
        return CompletionResult(status, reason, contact.copy(), int(contact.sum()), int(now_ns))

    def update(
        self,
        *,
        phase: CompletionPhase,
        state: RevoState,
        tactile: TactileFrame,
        nominal_q_rad: np.ndarray,
        now_ns: int,
        safety_failed: bool = False,
        task_postcondition_met: bool = False,
    ) -> CompletionResult:
        now = int(now_ns)
        nominal = assert_joint_vector(nominal_q_rad, name="nominal_q_rad")
        normal = np.abs(tactile.f6[:, self.config.normal_force_axis])
        contact = tactile.valid_fingers & (normal >= self.config.contact_force_threshold)

        if phase is CompletionPhase.ABORTED or safety_failed:
            return self._result(CompletionStatus.FAILED, "safety_or_abort", contact, now)
        state_age = now - int(state.timestamp_ns)
        if state_age < 0 or state_age > self.config.max_state_age_ns:
            return self._result(CompletionStatus.FAILED, "state_not_fresh", contact, now)
        touch_age = now - int(tactile.timestamp_ns)
        if touch_age < 0:
            return self._result(CompletionStatus.FAILED, "touch_from_future", contact, now)
        if touch_age > self.config.max_touch_age_ns:
            # Never declare stable/released from old touch.  The surrounding
            # execution layer holds the object and disables fast/CAIR.
            return self._result(CompletionStatus.IN_PROGRESS, "touch_stale_hold", contact, now)

        contact_count = int(contact.sum())
        if self._last_progress_ns is None:
            self._last_progress_ns = now
        progressed = self._last_q is None or float(
            np.max(np.abs(state.q_rad - self._last_q))
        ) >= self.config.min_progress_rad
        progressed = progressed or contact_count > self._last_contact_count
        if progressed:
            self._last_progress_ns = now

        force_stable = self._last_normal_force is not None and float(
            np.max(np.abs(normal - self._last_normal_force))
        ) <= self.config.max_force_change_per_sample
        self._last_q = state.q_rad.copy()
        self._last_normal_force = normal.copy()
        self._last_contact_count = contact_count

        if phase is CompletionPhase.CONTROLLED_RELEASE:
            released_force = bool(
                np.all((~tactile.valid_fingers) | (normal <= self.config.release_force_threshold))
            )
            safe_open = bool(
                np.max(np.abs(state.q_rad - self.config.safe_open_q_rad))
                <= self.config.safe_open_tolerance_rad
            )
            if released_force and safe_open:
                if self._release_since_ns is None:
                    self._release_since_ns = now
                if now - self._release_since_ns >= self.config.release_dwell_ns:
                    return self._result(CompletionStatus.RELEASED, "released_and_safe_open", contact, now)
            else:
                self._release_since_ns = None
            return self._result(CompletionStatus.IN_PROGRESS, "controlled_release", contact, now)

        self._release_since_ns = None
        enough_contact = contact_count >= self.config.min_contact_fingers
        if enough_contact:
            if self._contact_since_ns is None:
                self._contact_since_ns = now
        else:
            self._contact_since_ns = None
            self._stable_since_ns = None
            self._contact_event_emitted = False

        contact_dwelled = (
            self._contact_since_ns is not None
            and now - self._contact_since_ns >= self.config.contact_dwell_ns
        )
        mechanically_stable = bool(
            contact_dwelled
            and force_stable
            and np.max(np.abs(state.dq_rad_s)) <= self.config.max_stable_velocity_rad_s
            and np.max(np.abs(nominal - state.q_rad))
            <= self.config.max_stable_tracking_error_rad
        )
        if mechanically_stable:
            if self._stable_since_ns is None:
                self._stable_since_ns = now
            if now - self._stable_since_ns >= self.config.stable_dwell_ns:
                self._grasp_stable = True
        else:
            self._stable_since_ns = None

        if contact_dwelled and not self._contact_event_emitted:
            self._contact_event_emitted = True
            return self._result(
                CompletionStatus.CONTACT_ESTABLISHED,
                "contact_dwell_established",
                contact,
                now,
            )

        if self._grasp_stable:
            if task_postcondition_met:
                return self._result(CompletionStatus.TASK_SUCCESS, "task_postcondition_met", contact, now)
            return self._result(CompletionStatus.GRASP_STABLE, "stable_grasp_hold", contact, now)

        if (
            phase in (CompletionPhase.PRECONTACT_RUN, CompletionPhase.CONTACT_BUILD)
            and self._last_progress_ns is not None
            and now - self._last_progress_ns >= self.config.no_progress_timeout_ns
        ):
            return self._result(CompletionStatus.NO_PROGRESS, "no_joint_or_contact_progress", contact, now)
        return self._result(CompletionStatus.IN_PROGRESS, "not_yet_stable", contact, now)
