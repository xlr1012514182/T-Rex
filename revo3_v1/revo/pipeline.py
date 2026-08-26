"""Safety-owned composition of VLA nominal action and tactile residual."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .backend import RevoBackend
from .contracts import RevoCommand, RevoState, assert_joint_vector
from dataclasses import replace

from .safety import SafetyContext, SafetyResult, SafetySupervisor


class RevoCommandPipeline:
    """The only path that should call ``backend.write_command``."""

    def __init__(self, backend: RevoBackend, safety: SafetySupervisor) -> None:
        self.backend = backend
        self.safety = safety
        self._last_hardware_state_timestamp_ns: int | None = None
        self._last_hardware_state_sequence: int | None = None
        self._hard_fault_latched = False
        self._soft_stop_confirmed = False
        self._last_observed_q_rad: np.ndarray | None = None

    @property
    def hard_fault_latched(self) -> bool:
        return self._hard_fault_latched

    @property
    def soft_stop_confirmed(self) -> bool:
        """True only after ``backend.soft_stop`` actually returned success."""

        return self._soft_stop_confirmed

    def clear_fault_latch_after_operator_reset(self) -> None:
        """Out-of-band reset only; never called automatically by task logic."""

        self._hard_fault_latched = False
        self._soft_stop_confirmed = False
        self.safety.reset_telemetry_history()

    async def _latch_and_soft_stop(
        self, result: SafetyResult, *, hold_q_rad: np.ndarray
    ) -> SafetyResult:
        self._hard_fault_latched = True
        try:
            await self.backend.soft_stop(result.reason)
        except BaseException as exc:
            return replace(
                result,
                q_authorized_rad=hold_q_rad.copy(),
                reason=f"{result.reason},soft_stop_failed:{type(exc).__name__}",
                hard_fault_latched=True,
                clear_policy_cache=True,
                soft_stop_requested=True,
                soft_stop_succeeded=False,
            )
        self._soft_stop_confirmed = True
        return replace(
            result,
            q_authorized_rad=hold_q_rad.copy(),
            hard_fault_latched=True,
            clear_policy_cache=True,
            soft_stop_requested=True,
            soft_stop_succeeded=True,
        )

    async def abort(
        self,
        reason: str,
        *,
        state: Optional[RevoState] = None,
    ) -> SafetyResult:
        """Latch a no-open SoftStop through the sole motor-writer boundary."""

        text = str(reason).strip()
        if not text:
            raise ValueError("abort reason must be non-empty")
        is_hardware = bool(getattr(self.backend, "is_hardware", True))
        if is_hardware and state is not None:
            raise RuntimeError("real abort forbids caller-injected telemetry")
        observed = None
        read_failure: BaseException | None = None
        try:
            observed = await self.backend.read_state() if state is None else state
            self._last_observed_q_rad = observed.q_rad.copy()
        except BaseException as exc:
            # Emergency stop authority must not depend on readable telemetry.
            # Retain the read failure in evidence and still attempt SoftStop.
            read_failure = exc
        hold_q_rad = (
            self._last_observed_q_rad.copy()
            if self._last_observed_q_rad is not None
            else np.zeros(21, dtype=np.float32)
        )
        failure_suffix = (
            ""
            if read_failure is None
            else f",telemetry_read_failed:{type(read_failure).__name__}"
        )
        result = SafetyResult(
            hold_q_rad,
            True,
            False,
            f"explicit_abort:{text}{failure_suffix}",
            hard_fault_latched=True,
            clear_policy_cache=True,
            soft_stop_requested=True,
            soft_stop_succeeded=self._soft_stop_confirmed,
        )
        if self._hard_fault_latched and self._soft_stop_confirmed:
            return result
        # A prior failed stop remains latched but is retried at this explicit
        # shutdown boundary.  Only a confirmed backend callback is success.
        return await self._latch_and_soft_stop(result, hold_q_rad=hold_q_rad)

    async def execute(
        self,
        *,
        nominal_q_rad: np.ndarray,
        residual_q_rad: Optional[np.ndarray],
        task_id: str,
        task_version: int,
        source_chunk_id: Optional[str],
        safety_context: SafetyContext,
        emg_requests_close: bool,
        now_ns: Optional[int] = None,
        state: Optional[RevoState] = None,
    ) -> SafetyResult:
        now = time.monotonic_ns() if now_ns is None else int(now_ns)
        is_hardware = bool(getattr(self.backend, "is_hardware", True))
        if is_hardware and not self.safety.envelope.hardware_ready:
            raise RuntimeError(
                "real Revo writer requires an explicit, non-demo safety profile "
                "with velocity/current/temperature limits."
            )
        if is_hardware and state is not None:
            raise RuntimeError(
                "real Revo execution forbids caller-injected state; every writer "
                "tick must acquire fresh backend telemetry."
            )
        observed = await self.backend.read_state() if state is None else state
        self._last_observed_q_rad = observed.q_rad.copy()
        if is_hardware:
            if (
                self._last_hardware_state_timestamp_ns is not None
                and observed.timestamp_ns <= self._last_hardware_state_timestamp_ns
            ) or (
                self._last_hardware_state_sequence is not None
                and observed.sequence <= self._last_hardware_state_sequence
            ):
                result = SafetyResult(
                    observed.q_rad.copy(), True, False, "telemetry_not_fresh"
                )
                return await self._latch_and_soft_stop(
                    result, hold_q_rad=observed.q_rad
                )
            self._last_hardware_state_timestamp_ns = observed.timestamp_ns
            self._last_hardware_state_sequence = observed.sequence
        # Single-command SDK writes do not run a hidden telemetry/collision
        # loop.  The repository-owned writer therefore polls collision state
        # on every command and merges it into the final safety context.
        collision_active = await self.backend.collision_active()
        if collision_active and not safety_context.collision_active:
            safety_context = replace(
                safety_context,
                collision_active=True,
                reason=safety_context.reason or "backend_collision_active",
            )
        if self._hard_fault_latched and not safety_context.fault_latched:
            safety_context = replace(
                safety_context,
                fault_latched=True,
                reason=safety_context.reason or "pipeline_hard_fault_latched",
            )
        nominal = assert_joint_vector(nominal_q_rad, name="nominal_q_rad")
        residual = (
            np.zeros_like(nominal)
            if residual_q_rad is None
            else assert_joint_vector(residual_q_rad, name="residual_q_rad")
        )
        result = self.safety.authorize(
            nominal + residual,
            observed,
            now_ns=now,
            context=safety_context,
            emg_requests_close=emg_requests_close,
        )
        if result.vetoed:
            hard_tokens = (
                "collision_active",
                "motor_fault_or_stall",
                "emergency_stop",
                "over_current",
                "over_temperature",
                "over_velocity",
                "over_acceleration",
                "command_over_velocity",
                "command_over_acceleration",
                "command_timestamp_not_fresh",
                "state_stale",
                "state_from_future",
                "telemetry_not_fresh",
                "tactile_overload",
                "temperature_telemetry_missing",
                "temperature_limit_unconfigured",
            )
            hard_fault = any(token in result.reason for token in hard_tokens)
            if hard_fault and not self._hard_fault_latched:
                return await self._latch_and_soft_stop(
                    result, hold_q_rad=observed.q_rad
                )
            return replace(
                result,
                hard_fault_latched=self._hard_fault_latched,
                clear_policy_cache=self._hard_fault_latched,
            )
        command = RevoCommand(
            timestamp_ns=now,
            q_target_rad=result.q_authorized_rad,
            task_id=task_id,
            task_version=task_version,
            source_chunk_id=source_chunk_id,
        )
        try:
            await self.backend.write_command(command)
        except BaseException as exc:
            failure = SafetyResult(
                observed.q_rad.copy(),
                True,
                result.clipped,
                f"backend_write_failed:{type(exc).__name__}",
            )
            return await self._latch_and_soft_stop(
                failure, hold_q_rad=observed.q_rad
            )
        return result
