"""TactileReflex-inspired, bounded synergy residual for Revo 3.

This is a project adapter, not a reproduction of the gripper-specific IROS
controller.  It never replaces T-Rex and never writes hardware.  Its output is
an additive residual that must still pass ``SafetySupervisor``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from revo3_v1.revo.contracts import JOINT_COUNT, assert_joint_vector

from .contracts import FINGER_COUNT, TactileFrame


class ReflexPhase(str, Enum):
    PRECONTACT = "precontact"
    CONTACT_BUILD = "contact_build"
    HOLD = "hold"
    RELEASE = "release"


def demo_closing_synergy() -> np.ndarray:
    """Simulation-only positive-flexion convention; hardware must calibrate it."""

    return np.asarray(
        [
            0.15, 0.80, 0.95, 0.75,
            0.10, 0.85, 1.00, 0.80,
            0.05, 0.85, 1.00, 0.80,
            0.05, 0.85, 1.00, 0.80,
            0.65, 0.80, 0.70, 0.40, 0.20,
        ],
        dtype=np.float32,
    )


def _finger_vector(value: float | np.ndarray, *, name: str) -> np.ndarray:
    if np.isscalar(value):
        return np.full(FINGER_COUNT, float(value), dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (FINGER_COUNT,) or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be a finite ({FINGER_COUNT},) vector.")
    return arr.copy()


@dataclass(frozen=True)
class ReflexConfig:
    enabled: bool = False
    closing_synergy: np.ndarray = field(
        default_factory=lambda: np.zeros(JOINT_COUNT, dtype=np.float32)
    )
    normal_force_axis: int = 2
    baseline_median: np.ndarray | float = 0.0
    baseline_mad: np.ndarray | float = 0.01
    contact_on_mad: float = 6.0
    contact_off_mad: float = 3.0
    target_force: np.ndarray | float = 0.5
    protect_force: np.ndarray | float = 1.0
    hard_overload_force: np.ndarray | float = 2.0
    deadband: float = 0.05
    tighten_step_rad: float = 0.002
    loosen_step_rad: float = 0.003
    max_abs_step_rad: float = 0.003
    max_abs_cumulative_rad: float = 0.035
    slip_enabled: bool = False
    slip_threshold: float = 0.5

    def __post_init__(self) -> None:
        synergy = assert_joint_vector(self.closing_synergy, name="closing_synergy")
        max_abs = float(np.max(np.abs(synergy)))
        if self.enabled and max_abs <= 0:
            raise ValueError("enabled reflex requires a calibrated non-zero closing_synergy.")
        if max_abs > 1.0 + 1e-6:
            raise ValueError("closing_synergy must be normalized to max absolute value <= 1.")
        if self.normal_force_axis < 0 or self.normal_force_axis >= 6:
            raise ValueError("normal_force_axis must be in [0,5].")
        baseline = _finger_vector(self.baseline_median, name="baseline_median")
        mad = _finger_vector(self.baseline_mad, name="baseline_mad")
        target = _finger_vector(self.target_force, name="target_force")
        protect = _finger_vector(self.protect_force, name="protect_force")
        hard = _finger_vector(self.hard_overload_force, name="hard_overload_force")
        if np.any(mad <= 0) or np.any(target <= 0):
            raise ValueError("baseline_mad and target_force must be positive.")
        if np.any(protect <= target) or np.any(hard <= protect):
            raise ValueError("Require target_force < protect_force < hard_overload_force.")
        if not (0 <= self.contact_off_mad < self.contact_on_mad):
            raise ValueError("contact hysteresis MAD multipliers are invalid.")
        for name in (
            "tighten_step_rad",
            "loosen_step_rad",
            "max_abs_step_rad",
            "max_abs_cumulative_rad",
        ):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive.")
        object.__setattr__(self, "closing_synergy", synergy)
        object.__setattr__(self, "baseline_median", baseline)
        object.__setattr__(self, "baseline_mad", mad)
        object.__setattr__(self, "target_force", target)
        object.__setattr__(self, "protect_force", protect)
        object.__setattr__(self, "hard_overload_force", hard)

    @classmethod
    def demo(cls) -> "ReflexConfig":
        return cls(enabled=True, closing_synergy=demo_closing_synergy())


@dataclass(frozen=True)
class ReflexResult:
    residual_q_rad: np.ndarray
    delta_q_rad: np.ndarray
    contact_mask: np.ndarray
    overload: bool
    hard_overload: bool
    updated: bool
    reason: str
    tactile_timestamp_ns: int


class TactileReflexPlugin:
    """Stateful residual controller that integrates only new tactile samples."""

    def __init__(self, config: ReflexConfig) -> None:
        self.config = config
        self._residual = np.zeros(JOINT_COUNT, dtype=np.float32)
        self._contact = np.zeros(FINGER_COUNT, dtype=bool)
        self._last_timestamp_ns: int | None = None
        self._last_overload = False
        self._last_hard_overload = False

    @property
    def residual_q_rad(self) -> np.ndarray:
        return self._residual.copy()

    def reset(self) -> None:
        self._residual.fill(0)
        self._contact.fill(False)
        self._last_timestamp_ns = None
        self._last_overload = False
        self._last_hard_overload = False

    def update(
        self,
        frame: TactileFrame,
        *,
        phase: ReflexPhase,
        slip_score: float = 0.0,
    ) -> ReflexResult:
        if self._last_timestamp_ns is not None and frame.timestamp_ns < self._last_timestamp_ns:
            raise ValueError("out-of-order tactile frame cannot drive reflex.")

        if not self.config.enabled or phase in (ReflexPhase.PRECONTACT, ReflexPhase.RELEASE):
            self._last_timestamp_ns = frame.timestamp_ns
            self._residual.fill(0)
            self._contact.fill(False)
            self._last_overload = False
            self._last_hard_overload = False
            return self._result(
                delta=np.zeros(JOINT_COUNT, dtype=np.float32),
                overload=False,
                hard_overload=False,
                updated=True,
                reason="disabled_or_phase_zero",
                timestamp_ns=frame.timestamp_ns,
            )

        if self._last_timestamp_ns is not None and frame.timestamp_ns == self._last_timestamp_ns:
            return self._result(
                delta=np.zeros(JOINT_COUNT, dtype=np.float32),
                overload=self._last_overload,
                hard_overload=self._last_hard_overload,
                updated=False,
                reason="no_new_touch_sample",
                timestamp_ns=frame.timestamp_ns,
            )
        self._last_timestamp_ns = frame.timestamp_ns

        cfg = self.config
        normal = np.abs(frame.f6[:, cfg.normal_force_axis])
        valid = frame.valid_fingers
        on = cfg.baseline_median + cfg.contact_on_mad * cfg.baseline_mad
        off = cfg.baseline_median + cfg.contact_off_mad * cfg.baseline_mad
        self._contact = valid & np.where(self._contact, normal >= off, normal >= on)

        overload_mask = valid & (normal >= cfg.protect_force)
        hard_mask = valid & (normal >= cfg.hard_overload_force)
        overload = bool(overload_mask.any())
        hard_overload = bool(hard_mask.any())
        self._last_overload = overload
        self._last_hard_overload = hard_overload

        scalar_step = 0.0
        reason = "no_contact_or_deadband"
        if hard_overload or overload:
            scalar_step = -cfg.loosen_step_rad
            reason = "hard_force_protect" if hard_overload else "force_protect"
        elif phase is ReflexPhase.HOLD and self._contact.any():
            error = cfg.target_force[self._contact] - normal[self._contact]
            mean_error = float(np.mean(error))
            if mean_error > cfg.deadband:
                scalar_step = cfg.tighten_step_rad
                reason = "hold_tighten"
            elif mean_error < -cfg.deadband:
                scalar_step = -cfg.loosen_step_rad
                reason = "hold_loosen"
            if cfg.slip_enabled and float(slip_score) >= cfg.slip_threshold:
                scalar_step = max(scalar_step, cfg.tighten_step_rad)
                reason = "slip_tighten"

        scalar_step = float(np.clip(scalar_step, -cfg.max_abs_step_rad, cfg.max_abs_step_rad))
        delta = (scalar_step * cfg.closing_synergy).astype(np.float32)
        before = self._residual.copy()
        self._residual = np.clip(
            self._residual + delta,
            -cfg.max_abs_cumulative_rad,
            cfg.max_abs_cumulative_rad,
        ).astype(np.float32)
        actual_delta = self._residual - before
        return self._result(
            delta=actual_delta,
            overload=overload,
            hard_overload=hard_overload,
            updated=True,
            reason=reason,
            timestamp_ns=frame.timestamp_ns,
        )

    def _result(
        self,
        *,
        delta: np.ndarray,
        overload: bool,
        hard_overload: bool,
        updated: bool,
        reason: str,
        timestamp_ns: int,
    ) -> ReflexResult:
        return ReflexResult(
            residual_q_rad=self._residual.copy(),
            delta_q_rad=delta.copy(),
            contact_mask=self._contact.copy(),
            overload=overload,
            hard_overload=hard_overload,
            updated=updated,
            reason=reason,
            tactile_timestamp_ns=timestamp_ns,
        )
