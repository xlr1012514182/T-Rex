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
    ) -> None:
        self.client = client
        self.slave_id = int(slave_id)
        self.allow_hardware_write = bool(allow_hardware_write)
        self._sequence = 0

    async def _call_optional(self, name: str, default: np.ndarray) -> np.ndarray:
        method = getattr(self.client, name, None)
        if method is None:
            return default.copy()
        value = await _maybe_await(method(self.slave_id))
        arr = np.asarray(value)
        if arr.shape != (JOINT_COUNT,):
            raise ValueError(f"SDK {name} returned shape {arr.shape}; expected ({JOINT_COUNT},).")
        return arr

    async def read_state(self) -> RevoState:
        positions_deg = await self._call_optional(
            "revo3_get_all_motor_positions", np.zeros(JOINT_COUNT, dtype=np.float32)
        )
        velocities_deg_s = await self._call_optional(
            "revo3_get_all_motor_velocities", np.zeros(JOINT_COUNT, dtype=np.float32)
        )
        currents_ma = await self._call_optional(
            "revo3_get_all_motor_currents", np.zeros(JOINT_COUNT, dtype=np.float32)
        )
        status = await self._call_optional(
            "revo3_get_all_motor_status", np.zeros(JOINT_COUNT, dtype=np.int64)
        )
        self._sequence += 1
        return RevoState(
            timestamp_ns=time.monotonic_ns(),
            q_rad=np.deg2rad(positions_deg).astype(np.float32),
            dq_rad_s=np.deg2rad(velocities_deg_s).astype(np.float32),
            current_a=(currents_ma.astype(np.float32) / 1000.0),
            status=status.astype(np.int64),
            sequence=self._sequence,
        )

    async def write_command(self, command: RevoCommand) -> None:
        if not self.allow_hardware_write:
            raise HardwareWriteNotArmed(
                "Real Revo write blocked. Construct BrainCoSDKBackend with "
                "allow_hardware_write=True only after the hardware safety gate passes."
            )
        method = getattr(self.client, "revo3_set_all_motor_positions", None)
        if method is None:
            raise AttributeError("SDK client has no revo3_set_all_motor_positions method.")
        degrees = np.rad2deg(command.q_target_rad).astype(np.float32).tolist()
        await _maybe_await(method(self.slave_id, degrees))

    async def collision_active(self) -> bool:
        batch = getattr(self.client, "revo3_get_all_collision_active", None)
        if batch is not None:
            values = await _maybe_await(batch(self.slave_id))
            return bool(np.asarray(values, dtype=bool).any())
        per_joint = getattr(self.client, "revo3_is_collision_active", None)
        if per_joint is None:
            return False
        for joint_id in range(JOINT_COUNT):
            if bool(await _maybe_await(per_joint(self.slave_id, joint_id))):
                return True
        return False
