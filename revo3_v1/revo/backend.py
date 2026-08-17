"""Revo hardware abstraction with a deterministic simulation backend.

``BrainCoSDKBackend`` intentionally imports no SDK module.  It adapts an
already-created client implementing the official ``revo3_*`` methods.  This
keeps import-only demos runnable while preserving an explicit arming boundary
before any real motor write.
"""

from __future__ import annotations

import inspect
import time
from typing import Any, Awaitable, Protocol, runtime_checkable

import numpy as np

from .contracts import JOINT_COUNT, RevoCommand, RevoState, assert_joint_vector


class HardwareWriteNotArmed(RuntimeError):
    """Raised when a real SDK write is attempted without explicit arming."""


@runtime_checkable
class RevoBackend(Protocol):
    """Only this interface may read or write a Revo hand."""

    async def read_state(self) -> RevoState:
        ...

    async def write_command(self, command: RevoCommand) -> None:
        ...

    async def collision_active(self) -> bool:
        ...


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class MockRevoBackend:
    """In-memory, first-order Revo simulator used by the end-to-end demo."""

    def __init__(self, initial_q_rad: np.ndarray | None = None) -> None:
        initial = np.zeros(JOINT_COUNT, dtype=np.float32) if initial_q_rad is None else initial_q_rad
        self._q = assert_joint_vector(initial, name="initial_q_rad")
        self._dq = np.zeros(JOINT_COUNT, dtype=np.float32)
        self._current = np.zeros(JOINT_COUNT, dtype=np.float32)
        self._status = np.zeros(JOINT_COUNT, dtype=np.int64)
        self._sequence = 0
        self._last_timestamp_ns = time.monotonic_ns()
        self._collision_active = False
        self.commands: list[RevoCommand] = []

    def set_collision(self, active: bool) -> None:
        self._collision_active = bool(active)

    async def read_state(self) -> RevoState:
        return RevoState(
            timestamp_ns=self._last_timestamp_ns,
            q_rad=self._q,
            dq_rad_s=self._dq,
            current_a=self._current,
            status=self._status,
            sequence=self._sequence,
        )

    async def write_command(self, command: RevoCommand) -> None:
        if self._collision_active:
            raise RuntimeError("mock collision_active: regular command rejected")
        now = max(time.monotonic_ns(), self._last_timestamp_ns + 1)
        dt = max((now - self._last_timestamp_ns) / 1e9, 1e-6)
        previous = self._q.copy()
        self._q = command.q_target_rad.copy()
        self._dq = (self._q - previous) / dt
        self._sequence += 1
        self._last_timestamp_ns = now
        self.commands.append(command)

    async def collision_active(self) -> bool:
        return self._collision_active


class BrainCoSDKBackend:
    """Boundary adapter for the official BrainCo Revo3 SDK protocol.

    Official motor position methods use degrees.  The rest of this repository
    uses radians, so conversions happen exactly once here.  The supplied SDK
    client may expose synchronous methods (current SDK) or awaitables (a user
    transport wrapper); both are accepted.
    """

    def __init__(
        self,
        client: Any,
        *,
        slave_id: int,
        allow_hardware_write: bool = False,
        feedback_velocity_unit: str = "rpm",
        feedback_current_unit: str = "mA",
        capability_probe_confirmed: bool = False,
    ) -> None:
        self.client = client
        self.slave_id = int(slave_id)
        self.allow_hardware_write = bool(allow_hardware_write)
        if feedback_velocity_unit not in {"rpm", "deg/s"}:
            raise ValueError("feedback_velocity_unit must be 'rpm' or 'deg/s'.")
        if feedback_current_unit not in {"mA", "A"}:
            raise ValueError("feedback_current_unit must be 'mA' or 'A'.")
        self.feedback_velocity_unit = feedback_velocity_unit
        self.feedback_current_unit = feedback_current_unit
        self.capability_probe_confirmed = bool(capability_probe_confirmed)
        self._sequence = 0

    async def _call_required(self, name: str) -> np.ndarray:
        method = getattr(self.client, name, None)
        if not callable(method):
            raise AttributeError(
                f"SDK client has no required {name} method; refusing to "
                "fabricate Revo telemetry."
            )
        value = await _maybe_await(method(self.slave_id))
        arr = np.asarray(value)
        if arr.shape != (JOINT_COUNT,):
            raise ValueError(f"SDK {name} returned shape {arr.shape}; expected ({JOINT_COUNT},).")
        return arr

    async def read_state(self) -> RevoState:
        # The official SDK exposes a coherent Revo3MotorStatusData call.  Use
        # it when available rather than composing position/velocity/current
        # values acquired at three different instants.  Older wrappers are
        # supported only as an explicit fallback.
        read_all = getattr(self.client, "revo3_get_motor_status_data", None)
        if read_all is not None:
            sample = await _maybe_await(read_all(self.slave_id))
            positions_deg = np.asarray(sample.positions, dtype=np.float32)
            velocities = np.asarray(sample.velocities, dtype=np.float32)
            currents = np.asarray(sample.currents, dtype=np.float32)
            for name, values in (
                ("positions", positions_deg),
                ("velocities", velocities),
                ("currents", currents),
            ):
                if values.shape != (JOINT_COUNT,):
                    raise ValueError(
                        f"SDK motor status {name} has shape {values.shape}; "
                        f"expected ({JOINT_COUNT},)."
                    )
        else:
            positions_deg = await self._call_required("revo3_get_all_motor_positions")
            velocities = await self._call_required("revo3_get_all_motor_velocities")
            currents = await self._call_required("revo3_get_all_motor_currents")
        status = await self._call_required("revo3_get_all_motor_status")
        if self.feedback_velocity_unit == "rpm":
            velocities_rad_s = velocities * (2.0 * np.pi / 60.0)
        else:
            velocities_rad_s = np.deg2rad(velocities)
        if self.feedback_current_unit == "mA":
            currents_a = currents / 1000.0
        else:
            currents_a = currents
        self._sequence += 1
        return RevoState(
            timestamp_ns=time.monotonic_ns(),
            q_rad=np.deg2rad(positions_deg).astype(np.float32),
            dq_rad_s=velocities_rad_s.astype(np.float32),
            current_a=currents_a.astype(np.float32),
            status=status.astype(np.int64),
            sequence=self._sequence,
        )

    async def write_command(self, command: RevoCommand) -> None:
        if not self.allow_hardware_write or not self.capability_probe_confirmed:
            raise HardwareWriteNotArmed(
                "Real Revo write blocked. Set allow_hardware_write=True and "
                "capability_probe_confirmed=True only after verifying the actual "
                "Revo feedback units, joint order, limits, status freshness, and "
                "collision/SoftStop path on the bench."
            )
        method = getattr(self.client, "revo3_set_all_motor_positions", None)
        if method is None:
            raise AttributeError("SDK client has no revo3_set_all_motor_positions method.")
        degrees = np.rad2deg(command.q_target_rad).astype(np.float32).tolist()
        await _maybe_await(method(self.slave_id, degrees))

    async def collision_active(self) -> bool:
        batch = getattr(self.client, "revo3_get_all_collision_active", None)
        if callable(batch):
            values = await _maybe_await(batch(self.slave_id))
            return bool(np.asarray(values, dtype=bool).any())
        per_joint = getattr(self.client, "revo3_is_collision_active", None)
        if not callable(per_joint):
            raise AttributeError(
                "SDK client exposes neither revo3_get_all_collision_active nor "
                "revo3_is_collision_active; refusing to treat unknown collision "
                "state as safe."
            )
        for joint_id in range(JOINT_COUNT):
            if bool(await _maybe_await(per_joint(self.slave_id, joint_id))):
                return True
        return False
