"""Final, independent safety authority for Revo commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .contracts import JOINT_COUNT, RevoState, assert_joint_vector


def _array21(value: float | Iterable[float], *, name: str) -> np.ndarray:
    if np.isscalar(value):
        return np.full(JOINT_COUNT, float(value), dtype=np.float32)
    return assert_joint_vector(value, name=name)


@dataclass(frozen=True)
class SafetyEnvelope:
    """Bench values are required; deliberately wide defaults are demo-only."""

    q_min_rad: np.ndarray
    q_max_rad: np.ndarray
    max_step_rad: np.ndarray
    max_state_age_ns: int = 50_000_000
    max_abs_current_a: np.ndarray | float = 1000.0

    def __post_init__(self) -> None:
        q_min = assert_joint_vector(self.q_min_rad, name="q_min_rad")
        q_max = assert_joint_vector(self.q_max_rad, name="q_max_rad")
        max_step = assert_joint_vector(self.max_step_rad, name="max_step_rad")
        max_current = _array21(self.max_abs_current_a, name="max_abs_current_a")
        if np.any(q_min >= q_max):
            raise ValueError("Every q_min_rad must be strictly below q_max_rad.")
        if np.any(max_step <= 0):
            raise ValueError("Every max_step_rad must be positive.")
        if np.any(max_current <= 0):
            raise ValueError("Every max_abs_current_a must be positive.")
        if self.max_state_age_ns <= 0:
            raise ValueError("max_state_age_ns must be positive.")
        object.__setattr__(self, "q_min_rad", q_min)
        object.__setattr__(self, "q_max_rad", q_max)
        object.__setattr__(self, "max_step_rad", max_step)
        object.__setattr__(self, "max_abs_current_a", max_current)

    @classmethod
    def demo(cls, *, max_step_rad: float = 0.05) -> "SafetyEnvelope":
        """Simulation-only envelope; never a patient/hardware calibration."""

        return cls(
            q_min_rad=np.full(JOINT_COUNT, -np.pi, dtype=np.float32),
            q_max_rad=np.full(JOINT_COUNT, np.pi, dtype=np.float32),
            max_step_rad=np.full(JOINT_COUNT, max_step_rad, dtype=np.float32),
            max_state_age_ns=1_000_000_000,
            max_abs_current_a=np.full(JOINT_COUNT, 1000.0, dtype=np.float32),
        )


@dataclass(frozen=True)
class SafetyContext:
    collision_active: bool = False
    tactile_overload: bool = False
    emergency_stop: bool = False
    fault_latched: bool = False
    command_lease_valid: bool = True
    reason: str = ""


@dataclass(frozen=True)
class SafetyResult:
    q_authorized_rad: np.ndarray
    vetoed: bool
    clipped: bool
    reason: str


class SafetySupervisor:
    """Owns final authorization; an EMG close request cannot bypass it."""

    def __init__(self, envelope: SafetyEnvelope) -> None:
        self.envelope = envelope

    def authorize(
        self,
        desired_q_rad: np.ndarray,
        state: RevoState,
        *,
        now_ns: int,
        context: SafetyContext,
        emg_requests_close: bool = False,
    ) -> SafetyResult:
        del emg_requests_close  # explicit: intent has no authority inside safety
        try:
            desired = assert_joint_vector(desired_q_rad, name="desired_q_rad")
        except ValueError as exc:
            return SafetyResult(state.q_rad.copy(), True, False, f"invalid_command:{exc}")

        state_age = int(now_ns) - int(state.timestamp_ns)
        hard_reasons = []
        if state_age < 0:
            hard_reasons.append("state_from_future")
        elif state_age > self.envelope.max_state_age_ns:
            hard_reasons.append("state_stale")
        if context.emergency_stop:
            hard_reasons.append("emergency_stop")
        if context.fault_latched:
            hard_reasons.append("fault_latched")
        if context.collision_active:
            hard_reasons.append("collision_active")
        if context.tactile_overload:
            hard_reasons.append("tactile_overload")
        if not context.command_lease_valid:
            hard_reasons.append("invalid_command_lease")
        if np.any(np.abs(state.current_a) > self.envelope.max_abs_current_a):
            hard_reasons.append("over_current")

        if hard_reasons:
            suffix = f":{context.reason}" if context.reason else ""
            return SafetyResult(
                state.q_rad.copy(), True, False, ",".join(hard_reasons) + suffix
            )

        joint_limited = np.clip(desired, self.envelope.q_min_rad, self.envelope.q_max_rad)
        delta = joint_limited - state.q_rad
        step_limited = state.q_rad + np.clip(
            delta, -self.envelope.max_step_rad, self.envelope.max_step_rad
        )
        step_limited = np.clip(
            step_limited, self.envelope.q_min_rad, self.envelope.q_max_rad
        ).astype(np.float32)
        clipped = not np.array_equal(step_limited, desired)
        return SafetyResult(step_limited, False, clipped, "clipped" if clipped else "ok")
