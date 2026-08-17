"""Final, independent safety authority for Revo commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .contracts import JOINT_COUNT, RevoState, assert_joint_vector


# Official Revo3 u16 motor status error bits: over-current, over/under-voltage,
# over-temperature, current spike, and stalled.  Bit 11 means Running and is
# intentionally not an error.  See BrainCo REVO3_MOTOR_API.md.
REVO3_FAULT_STATUS_MASK = sum(1 << bit for bit in (0, 1, 2, 3, 4, 8))


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
    fault_status_mask: int = REVO3_FAULT_STATUS_MASK
    max_abs_velocity_rad_s: np.ndarray | float | None = None
    max_abs_acceleration_rad_s2: np.ndarray | float | None = None
    max_temperature_c: np.ndarray | float | None = None
    require_temperature_telemetry: bool = False
    hardware_profile_id: str = ""
    simulation_only: bool = False
    command_period_ns: int = 10_000_000

    def __post_init__(self) -> None:
        q_min = assert_joint_vector(self.q_min_rad, name="q_min_rad")
        q_max = assert_joint_vector(self.q_max_rad, name="q_max_rad")
        max_step = assert_joint_vector(self.max_step_rad, name="max_step_rad")
        max_current = _array21(self.max_abs_current_a, name="max_abs_current_a")
        max_velocity = (
            None
            if self.max_abs_velocity_rad_s is None
            else _array21(self.max_abs_velocity_rad_s, name="max_abs_velocity_rad_s")
        )
        max_acceleration = (
            None
            if self.max_abs_acceleration_rad_s2 is None
            else _array21(
                self.max_abs_acceleration_rad_s2,
                name="max_abs_acceleration_rad_s2",
            )
        )
        max_temperature = (
            None
            if self.max_temperature_c is None
            else _array21(self.max_temperature_c, name="max_temperature_c")
        )
        if np.any(q_min >= q_max):
            raise ValueError("Every q_min_rad must be strictly below q_max_rad.")
        if np.any(max_step <= 0):
            raise ValueError("Every max_step_rad must be positive.")
        if np.any(max_current <= 0):
            raise ValueError("Every max_abs_current_a must be positive.")
        if max_velocity is not None and (
            np.any(max_velocity <= 0) or not np.isfinite(max_velocity).all()
        ):
            raise ValueError("Every configured max_abs_velocity_rad_s must be finite and positive.")
        if max_acceleration is not None and (
            np.any(max_acceleration <= 0) or not np.isfinite(max_acceleration).all()
        ):
            raise ValueError(
                "Every configured max_abs_acceleration_rad_s2 must be finite and positive."
            )
        if max_temperature is not None and (
            np.any(max_temperature <= 0) or not np.isfinite(max_temperature).all()
        ):
            raise ValueError("Every configured max_temperature_c must be finite and positive.")
        if self.max_state_age_ns <= 0:
            raise ValueError("max_state_age_ns must be positive.")
        if self.command_period_ns <= 0:
            raise ValueError("command_period_ns must be positive.")
        if not 0 <= int(self.fault_status_mask) <= 0xFFFF:
            raise ValueError("fault_status_mask must be a u16 bitmask.")
        object.__setattr__(self, "q_min_rad", q_min)
        object.__setattr__(self, "q_max_rad", q_max)
        object.__setattr__(self, "max_step_rad", max_step)
        object.__setattr__(self, "max_abs_current_a", max_current)
        object.__setattr__(self, "max_abs_velocity_rad_s", max_velocity)
        object.__setattr__(self, "max_abs_acceleration_rad_s2", max_acceleration)
        object.__setattr__(self, "max_temperature_c", max_temperature)

    @property
    def hardware_ready(self) -> bool:
        """Whether limits are explicitly calibrated for a real writer."""

        return bool(
            not self.simulation_only
            and self.hardware_profile_id.strip()
            and self.max_abs_velocity_rad_s is not None
            and self.max_abs_acceleration_rad_s2 is not None
            and self.max_temperature_c is not None
            and self.require_temperature_telemetry
            and np.isfinite(self.max_abs_current_a).all()
            and self.command_period_ns == 10_000_000
        )

    @classmethod
    def demo(cls, *, max_step_rad: float = 0.05) -> "SafetyEnvelope":
        """Simulation-only envelope; never a patient/hardware calibration."""

        return cls(
            q_min_rad=np.full(JOINT_COUNT, -np.pi, dtype=np.float32),
            q_max_rad=np.full(JOINT_COUNT, np.pi, dtype=np.float32),
            max_step_rad=np.full(JOINT_COUNT, max_step_rad, dtype=np.float32),
            max_state_age_ns=1_000_000_000,
            max_abs_current_a=np.full(JOINT_COUNT, 1000.0, dtype=np.float32),
            max_abs_velocity_rad_s=np.full(JOINT_COUNT, 1000.0, dtype=np.float32),
            max_abs_acceleration_rad_s2=np.full(JOINT_COUNT, 1000.0, dtype=np.float32),
            max_temperature_c=np.full(JOINT_COUNT, 1000.0, dtype=np.float32),
            require_temperature_telemetry=False,
            hardware_profile_id="simulation-only-demo",
            simulation_only=True,
        )


@dataclass(frozen=True)
class SafetyContext:
    collision_active: bool = False
    tactile_overload: bool = False
    emergency_stop: bool = False
    fault_latched: bool = False
    command_lease_valid: bool = True
    policy_aborted: bool = False
    holding_object: bool = False
    reason: str = ""


@dataclass(frozen=True)
class SafetyResult:
    q_authorized_rad: np.ndarray
    vetoed: bool
    clipped: bool
    reason: str
    hard_fault_latched: bool = False
    clear_policy_cache: bool = False
    soft_stop_requested: bool = False
    soft_stop_succeeded: bool = False


class SafetySupervisor:
    """Owns final authorization; an EMG close request cannot bypass it."""

    def __init__(self, envelope: SafetyEnvelope) -> None:
        self.envelope = envelope
        self._previous_velocity_rad_s: np.ndarray | None = None
        self._previous_velocity_timestamp_ns: int | None = None
        self._previous_velocity_sequence: int | None = None
        self._previous_command_q_rad: np.ndarray | None = None
        self._previous_command_velocity_rad_s: np.ndarray | None = None
        self._previous_command_timestamp_ns: int | None = None

    def reset_telemetry_history(self) -> None:
        """Reset acceleration estimation only at an explicit lifecycle boundary."""

        self._previous_velocity_rad_s = None
        self._previous_velocity_timestamp_ns = None
        self._previous_velocity_sequence = None
        self._previous_command_q_rad = None
        self._previous_command_velocity_rad_s = None
        self._previous_command_timestamp_ns = None

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
        if context.policy_aborted:
            hard_reasons.append("policy_aborted_hold" if context.holding_object else "policy_aborted")
        if self.envelope.max_abs_velocity_rad_s is not None and np.any(
            np.abs(state.dq_rad_s) > self.envelope.max_abs_velocity_rad_s
        ):
            hard_reasons.append("over_velocity")
        # Acceleration is derived from two consecutive, strictly newer
        # telemetry samples.  The first sample establishes history and makes
        # no acceleration claim; repeated/out-of-order samples are rejected by
        # the hardware writer freshness gate and are never used here.
        acceleration = None
        if (
            self._previous_velocity_timestamp_ns is not None
            and state.timestamp_ns > self._previous_velocity_timestamp_ns
            and (
                self._previous_velocity_sequence is None
                or state.sequence > self._previous_velocity_sequence
            )
        ):
            dt_s = (state.timestamp_ns - self._previous_velocity_timestamp_ns) / 1e9
            acceleration = (
                state.dq_rad_s - self._previous_velocity_rad_s
            ) / dt_s
        if (
            acceleration is not None
            and self.envelope.max_abs_acceleration_rad_s2 is not None
            and np.any(
                np.abs(acceleration) > self.envelope.max_abs_acceleration_rad_s2
            )
        ):
            hard_reasons.append("over_acceleration")
        if (
            self._previous_velocity_timestamp_ns is None
            or state.timestamp_ns > self._previous_velocity_timestamp_ns
        ):
            self._previous_velocity_rad_s = state.dq_rad_s.copy()
            self._previous_velocity_timestamp_ns = state.timestamp_ns
            self._previous_velocity_sequence = state.sequence
        if np.any(np.abs(state.current_a) > self.envelope.max_abs_current_a):
            hard_reasons.append("over_current")
        if self.envelope.require_temperature_telemetry:
            if state.temperature_c is None:
                hard_reasons.append("temperature_telemetry_missing")
            elif self.envelope.max_temperature_c is None:
                hard_reasons.append("temperature_limit_unconfigured")
            elif np.any(state.temperature_c > self.envelope.max_temperature_c):
                hard_reasons.append("over_temperature")
        if np.any(np.bitwise_and(state.status, self.envelope.fault_status_mask) != 0):
            hard_reasons.append("motor_fault_or_stall")

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
        if self.envelope.max_abs_velocity_rad_s is not None:
            if self._previous_command_q_rad is None:
                command_origin = state.q_rad
                dt_s = self.envelope.command_period_ns / 1e9
            else:
                command_origin = self._previous_command_q_rad
                elapsed_ns = int(now_ns) - int(self._previous_command_timestamp_ns)
                if elapsed_ns <= 0:
                    return SafetyResult(
                        state.q_rad.copy(), True, False, "command_timestamp_not_fresh"
                    )
                dt_s = elapsed_ns / 1e9
            command_velocity = (step_limited - command_origin) / dt_s
            if np.any(
                np.abs(command_velocity) > self.envelope.max_abs_velocity_rad_s
            ):
                return SafetyResult(
                    state.q_rad.copy(), True, False, "command_over_velocity"
                )
            if (
                self._previous_command_velocity_rad_s is not None
                and self.envelope.max_abs_acceleration_rad_s2 is not None
            ):
                command_acceleration = (
                    command_velocity - self._previous_command_velocity_rad_s
                ) / dt_s
                if np.any(
                    np.abs(command_acceleration)
                    > self.envelope.max_abs_acceleration_rad_s2
                ):
                    return SafetyResult(
                        state.q_rad.copy(), True, False, "command_over_acceleration"
                    )
            self._previous_command_q_rad = step_limited.copy()
            self._previous_command_velocity_rad_s = command_velocity.copy()
            self._previous_command_timestamp_ns = int(now_ns)
        clipped = not np.array_equal(step_limited, desired)
        return SafetyResult(step_limited, False, clipped, "clipped" if clipped else "ok")
