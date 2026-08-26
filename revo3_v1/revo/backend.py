"""Revo hardware abstraction with a deterministic simulation backend.

``BrainCoSDKBackend`` intentionally imports no SDK module.  It adapts an
already-created client implementing the official ``revo3_*`` methods.  This
keeps import-only demos runnable while preserving an explicit arming boundary
before any real motor write.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
import inspect
import time
from typing import Any, AsyncIterator, Callable, Protocol, runtime_checkable

import numpy as np

from .contracts import JOINT_COUNT, RevoCommand, RevoState, assert_joint_vector


class HardwareWriteNotArmed(RuntimeError):
    """Raised when a real SDK write is attempted without explicit arming."""


class SDKBackendClosed(RuntimeError):
    """Raised when code tries to use a closing or closed SDK adapter."""


class SDKBackendFault(RuntimeError):
    """Raised after an SDK failure has latched the adapter fail-closed."""


class SDKCallTimeout(TimeoutError):
    """Raised when a serialized vendor-SDK operation exceeds its deadline."""


@runtime_checkable
class RevoBackend(Protocol):
    """Only this interface may read or write a Revo hand."""

    async def read_state(self) -> RevoState:
        ...

    async def write_command(self, command: RevoCommand) -> None:
        ...

    async def collision_active(self) -> bool:
        ...

    async def soft_stop(self, reason: str) -> None:
        """Stop motion without opening the hand; must not auto-clear faults."""
        ...

    async def close(self) -> bool:
        """Release backend-owned workers; false requires operator intervention."""
        ...


class MockRevoBackend:
    """In-memory, first-order Revo simulator used by the end-to-end demo."""

    def __init__(self, initial_q_rad: np.ndarray | None = None) -> None:
        self.is_hardware = False
        initial = np.zeros(JOINT_COUNT, dtype=np.float32) if initial_q_rad is None else initial_q_rad
        self._q = assert_joint_vector(initial, name="initial_q_rad")
        self._dq = np.zeros(JOINT_COUNT, dtype=np.float32)
        self._current = np.zeros(JOINT_COUNT, dtype=np.float32)
        self._status = np.zeros(JOINT_COUNT, dtype=np.int64)
        self._temperature = np.full(JOINT_COUNT, 25.0, dtype=np.float32)
        self._sequence = 0
        self._last_timestamp_ns = time.monotonic_ns()
        self._collision_active = False
        self.commands: list[RevoCommand] = []
        self.soft_stop_reasons: list[str] = []
        self.closed = False

    def set_collision(self, active: bool) -> None:
        self._collision_active = bool(active)

    @property
    def production_ready(self) -> bool:
        return False

    async def read_state(self) -> RevoState:
        return RevoState(
            timestamp_ns=self._last_timestamp_ns,
            q_rad=self._q,
            dq_rad_s=self._dq,
            current_a=self._current,
            status=self._status,
            sequence=self._sequence,
            temperature_c=self._temperature,
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

    async def soft_stop(self, reason: str) -> None:
        self._dq.fill(0)
        self.soft_stop_reasons.append(str(reason))

    async def close(self) -> bool:
        self.closed = True
        return True


class BrainCoSDKBackend:
    """Boundary adapter for the official BrainCo Revo3 SDK protocol.

    Official motor position methods use degrees.  The rest of this repository
    uses radians, so conversions happen exactly once here.  The supplied SDK
    client may expose synchronous methods (current SDK) or awaitables (a user
    transport wrapper); both are accepted.  Synchronous SDK calls execute on
    one backend-owned worker thread, while asynchronous SDK methods remain on
    the event loop.  A single asyncio lock serializes both forms, so reads,
    collision polling, writes, and SoftStop cannot overlap or reorder.

    A timed-out synchronous call cannot be killed safely by Python.  The
    adapter therefore retains ownership of its future, latches fail-closed,
    and makes a later SoftStop wait for that call to drain before touching the
    client.  If it cannot drain within the SoftStop deadline, the stop fails
    explicitly and ``intervention_required`` remains true; it never launches
    a concurrent fallback call behind the caller's back.
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
        temperature_reader: Callable[[Any, int], Any] | None = None,
        temperature_telemetry_verified: bool = False,
        soft_stop_callback: Callable[[Any, int, str], Any] | None = None,
        soft_stop_capability_verified: bool = False,
        collision_profile_id: str = "",
        collision_debounce_ms: int = 100,
        collision_status_cache_ms: int = 50,
        collision_auto_clear: bool = False,
        collision_profile_verified: bool = False,
        sdk_call_timeout_s: float = 0.25,
        soft_stop_timeout_s: float = 0.50,
        queue_timeout_s: float = 0.50,
        close_timeout_s: float = 1.00,
    ) -> None:
        self.is_hardware = True
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
        self.temperature_reader = temperature_reader
        self.temperature_telemetry_verified = bool(temperature_telemetry_verified)
        self.soft_stop_callback = soft_stop_callback
        self.soft_stop_capability_verified = bool(soft_stop_capability_verified)
        self.collision_profile_id = str(collision_profile_id).strip()
        self.collision_debounce_ms = int(collision_debounce_ms)
        self.collision_status_cache_ms = int(collision_status_cache_ms)
        self.collision_auto_clear = bool(collision_auto_clear)
        self.collision_profile_verified = bool(collision_profile_verified)
        if self.collision_debounce_ms != 100:
            raise ValueError("V1 collision_debounce_ms is frozen at 100 ms.")
        if self.collision_status_cache_ms != 50:
            raise ValueError("V1 collision_status_cache_ms is frozen at 50 ms.")
        if self.collision_auto_clear:
            raise ValueError("V1 collision_auto_clear must remain disabled.")
        timeout_values = {
            "sdk_call_timeout_s": sdk_call_timeout_s,
            "soft_stop_timeout_s": soft_stop_timeout_s,
            "queue_timeout_s": queue_timeout_s,
            "close_timeout_s": close_timeout_s,
        }
        for name, value in timeout_values.items():
            value = float(value)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
            setattr(self, name, value)
        self._sequence = 0
        self._call_lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"brainco-revo3-{self.slave_id}",
        )
        self._active_sync_future: Future[Any] | None = None
        self._active_async_task: asyncio.Future[Any] | None = None
        self._fault_reason: str | None = None
        self._timed_out = False
        self._intervention_required = False
        self._closing = False
        self._closed = False
        self._clean_close: bool | None = None

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def fault_reason(self) -> str | None:
        return self._fault_reason

    @property
    def intervention_required(self) -> bool:
        return self._intervention_required

    @property
    def timed_out(self) -> bool:
        return self._timed_out

    @property
    def production_ready(self) -> bool:
        """Explicit factory capability proof; never inferred from class name."""

        return bool(
            self.is_hardware
            and self.allow_hardware_write
            and self.capability_probe_confirmed
            and self.temperature_telemetry_verified
            and self.soft_stop_capability_verified
            and callable(self.soft_stop_callback)
            and self.collision_profile_verified
            and self.collision_profile_id
            and not self.collision_auto_clear
            and not self._closing
            and not self._closed
            and self._fault_reason is None
            and not self._timed_out
            and not self._intervention_required
        )

    def _ensure_available(self, *, allow_fault: bool) -> None:
        if self._closing or self._closed:
            raise SDKBackendClosed("BrainCo SDK backend is closing or closed")
        if self._fault_reason is not None and not allow_fault:
            raise SDKBackendFault(
                "BrainCo SDK backend is fail-closed after " + self._fault_reason
            )

    def _latch_failure(self, operation: str, exc: BaseException) -> None:
        if self._fault_reason is None:
            self._fault_reason = f"{operation}:{type(exc).__name__}"
        if isinstance(exc, SDKCallTimeout):
            self._timed_out = True
        self._intervention_required = True

    @staticmethod
    def _remaining(deadline: float, *, operation: str) -> float:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0.0:
            raise SDKCallTimeout(f"BrainCo SDK {operation} timed out")
        return remaining

    @asynccontextmanager
    async def _serialized_operation(
        self,
        operation: str,
        *,
        timeout_s: float,
        allow_fault: bool = False,
    ) -> AsyncIterator[float]:
        """Own one complete public SDK operation and its total deadline."""

        self._ensure_available(allow_fault=allow_fault)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + float(timeout_s)
        try:
            await asyncio.wait_for(
                self._call_lock.acquire(),
                timeout=min(self.queue_timeout_s, float(timeout_s)),
            )
        except asyncio.TimeoutError as exc:
            timeout = SDKCallTimeout(
                f"BrainCo SDK {operation} timed out waiting for serialized ownership"
            )
            self._latch_failure(operation, timeout)
            raise timeout from exc
        try:
            self._ensure_available(allow_fault=allow_fault)
            try:
                yield deadline
            except (SDKBackendClosed, SDKBackendFault):
                raise
            except BaseException as exc:
                self._latch_failure(operation, exc)
                raise
        finally:
            self._call_lock.release()

    async def _drain_owned_call_locked(self, *, deadline: float, operation: str) -> None:
        """Drain a timed-out call without ever starting a concurrent SDK call."""

        future = self._active_sync_future
        if future is not None and not future.done():
            wrapped = asyncio.wrap_future(future)
            done, _ = await asyncio.wait(
                {wrapped},
                timeout=self._remaining(deadline, operation=operation),
            )
            if not done:
                raise SDKCallTimeout(
                    f"BrainCo SDK {operation} timed out waiting for prior call ownership"
                )
        if future is not None and future.done():
            # Retrieve a late exception from the retained concurrent future;
            # the public call already reported the timeout/fail-closed state.
            try:
                future.exception()
            except BaseException:
                pass
            self._active_sync_future = None

        task = self._active_async_task
        if task is not None and not task.done():
            done, _ = await asyncio.wait(
                {task},
                timeout=self._remaining(deadline, operation=operation),
            )
            if not done:
                raise SDKCallTimeout(
                    f"BrainCo SDK {operation} timed out waiting for prior async ownership"
                )
        if task is not None and task.done():
            try:
                task.exception()
            except BaseException:
                pass
            self._active_async_task = None

    async def _await_owned_async_locked(
        self,
        awaitable: Any,
        *,
        deadline: float,
        operation: str,
    ) -> Any:
        """Await without cancellation-transfer; retain ownership after timeout."""

        task = asyncio.ensure_future(awaitable)
        self._active_async_task = task
        try:
            return await asyncio.wait_for(
                asyncio.shield(task),
                timeout=self._remaining(deadline, operation=operation),
            )
        except asyncio.TimeoutError as exc:
            raise SDKCallTimeout(f"BrainCo SDK {operation} timed out") from exc
        finally:
            if task.done():
                self._active_async_task = None

    async def _invoke_locked(
        self,
        method: Callable[..., Any],
        *args: Any,
        deadline: float,
        operation: str,
    ) -> Any:
        """Invoke one callable while the caller owns ``_call_lock``."""

        await self._drain_owned_call_locked(deadline=deadline, operation=operation)
        if inspect.iscoroutinefunction(method):
            return await self._await_owned_async_locked(
                method(*args), deadline=deadline, operation=operation
            )

        try:
            future = self._executor.submit(method, *args)
        except RuntimeError as exc:
            raise SDKBackendClosed("BrainCo SDK executor is unavailable") from exc
        self._active_sync_future = future
        try:
            value = await asyncio.wait_for(
                asyncio.shield(asyncio.wrap_future(future)),
                timeout=self._remaining(deadline, operation=operation),
            )
        except asyncio.TimeoutError as exc:
            # Do not cancel or forget the vendor call.  It remains owned by
            # this single-worker adapter and must drain before SoftStop/close.
            raise SDKCallTimeout(f"BrainCo SDK {operation} timed out") from exc
        finally:
            if future.done():
                self._active_sync_future = None
        if inspect.isawaitable(value):
            value = await self._await_owned_async_locked(
                value, deadline=deadline, operation=operation
            )
        return value

    async def _call_required_locked(
        self, name: str, *, deadline: float, operation: str
    ) -> np.ndarray:
        method = getattr(self.client, name, None)
        if not callable(method):
            raise AttributeError(
                f"SDK client has no required {name} method; refusing to "
                "fabricate Revo telemetry."
            )
        value = await self._invoke_locked(
            method,
            self.slave_id,
            deadline=deadline,
            operation=f"{operation}.{name}",
        )
        arr = np.asarray(value)
        if arr.shape != (JOINT_COUNT,):
            raise ValueError(f"SDK {name} returned shape {arr.shape}; expected ({JOINT_COUNT},).")
        return arr

    async def read_state(self) -> RevoState:
        # The official SDK exposes a coherent Revo3MotorStatusData call.  Use
        # it when available rather than composing position/velocity/current
        # values acquired at three different instants.  Older wrappers are
        # supported only as an explicit fallback.
        async with self._serialized_operation(
            "read_state", timeout_s=self.sdk_call_timeout_s
        ) as deadline:
            read_all = getattr(self.client, "revo3_get_motor_status_data", None)
            if read_all is not None:
                if not callable(read_all):
                    raise AttributeError("SDK revo3_get_motor_status_data is not callable")
                sample = await self._invoke_locked(
                    read_all,
                    self.slave_id,
                    deadline=deadline,
                    operation="read_state.motor_status",
                )
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
                positions_deg = await self._call_required_locked(
                    "revo3_get_all_motor_positions",
                    deadline=deadline,
                    operation="read_state",
                )
                velocities = await self._call_required_locked(
                    "revo3_get_all_motor_velocities",
                    deadline=deadline,
                    operation="read_state",
                )
                currents = await self._call_required_locked(
                    "revo3_get_all_motor_currents",
                    deadline=deadline,
                    operation="read_state",
                )
            status = await self._call_required_locked(
                "revo3_get_all_motor_status",
                deadline=deadline,
                operation="read_state",
            )
            temperature_c = None
            if self.temperature_reader is not None:
                raw_temperature = await self._invoke_locked(
                    self.temperature_reader,
                    self.client,
                    self.slave_id,
                    deadline=deadline,
                    operation="read_state.temperature",
                )
                temperature_c = np.asarray(raw_temperature, dtype=np.float32)
                if temperature_c.shape != (JOINT_COUNT,) or not np.isfinite(
                    temperature_c
                ).all():
                    raise ValueError(
                        "temperature_reader must return 21 finite values in degrees C."
                    )
            elif read_all is not None and hasattr(sample, "temperatures"):
                temperature_c = np.asarray(sample.temperatures, dtype=np.float32)
                if temperature_c.shape != (JOINT_COUNT,) or not np.isfinite(
                    temperature_c
                ).all():
                    raise ValueError(
                        "SDK motor-status temperatures must contain 21 finite values."
                    )
            if self.temperature_telemetry_verified and temperature_c is None:
                raise AttributeError(
                    "temperature telemetry was marked verified but no reader/atomic field is available."
                )
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
                temperature_c=temperature_c,
            )

    async def write_command(self, command: RevoCommand) -> None:
        if (
            not self.allow_hardware_write
            or not self.capability_probe_confirmed
            or not self.temperature_telemetry_verified
            or not self.soft_stop_capability_verified
            or not self.collision_profile_verified
            or not self.collision_profile_id
        ):
            raise HardwareWriteNotArmed(
                "Real Revo write blocked. Set allow_hardware_write=True, "
                "capability_probe_confirmed=True, and "
                "temperature_telemetry_verified=True only after verifying the actual "
                "Revo feedback units, joint order, limits, status freshness, and "
                "collision/SoftStop path and temperature telemetry on the bench; "
                "soft_stop_capability_verified, collision_profile_verified, and a "
                "collision_profile_id are also required."
            )
        degrees = np.rad2deg(command.q_target_rad).astype(np.float32).tolist()
        async with self._serialized_operation(
            "write_command", timeout_s=self.sdk_call_timeout_s
        ) as deadline:
            method = getattr(self.client, "revo3_set_all_motor_positions", None)
            if not callable(method):
                raise AttributeError(
                    "SDK client has no revo3_set_all_motor_positions method."
                )
            await self._invoke_locked(
                method,
                self.slave_id,
                degrees,
                deadline=deadline,
                operation="write_command.motor_positions",
            )

    async def collision_active(self) -> bool:
        async with self._serialized_operation(
            "collision_active", timeout_s=self.sdk_call_timeout_s
        ) as deadline:
            batch = getattr(self.client, "revo3_get_all_collision_active", None)
            if callable(batch):
                values = await self._invoke_locked(
                    batch,
                    self.slave_id,
                    deadline=deadline,
                    operation="collision_active.batch",
                )
                return bool(np.asarray(values, dtype=bool).any())
            per_joint = getattr(self.client, "revo3_is_collision_active", None)
            if not callable(per_joint):
                raise AttributeError(
                    "SDK client exposes neither revo3_get_all_collision_active nor "
                    "revo3_is_collision_active; refusing to treat unknown collision "
                    "state as safe."
                )
            for joint_id in range(JOINT_COUNT):
                active = await self._invoke_locked(
                    per_joint,
                    self.slave_id,
                    joint_id,
                    deadline=deadline,
                    operation="collision_active.per_joint",
                )
                if bool(active):
                    return True
            return False

    async def soft_stop(self, reason: str) -> None:
        if (
            not self.soft_stop_capability_verified
            or self.soft_stop_callback is None
            or not self.collision_profile_verified
            or not self.collision_profile_id
            or self.collision_auto_clear
        ):
            raise HardwareWriteNotArmed(
                "SoftStop blocked until an injected, bench-verified callback and "
                "non-auto-clear collision profile are configured."
            )
        async with self._serialized_operation(
            "soft_stop",
            timeout_s=self.soft_stop_timeout_s,
            allow_fault=True,
        ) as deadline:
            await self._invoke_locked(
                self.soft_stop_callback,
                self.client,
                self.slave_id,
                str(reason),
                deadline=deadline,
                operation="soft_stop.callback",
            )

    async def close(self) -> bool:
        """Bounded shutdown of the adapter-owned serialized SDK worker.

        The SDK connection itself is owned by ``Revo3ProbedConnection``; this
        method only closes the execution boundary.  False means a synchronous
        vendor call failed to drain before the deadline, so process/device
        intervention is still required.  No new calls are accepted once close
        starts, regardless of the result.
        """

        if self._closed:
            return bool(self._clean_close)
        self._closing = True
        clean = True
        acquired = False
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.close_timeout_s
        try:
            try:
                await asyncio.wait_for(
                    self._call_lock.acquire(), timeout=self.close_timeout_s
                )
                acquired = True
            except asyncio.TimeoutError:
                clean = False
            if acquired:
                try:
                    await self._drain_owned_call_locked(
                        deadline=deadline, operation="close"
                    )
                except BaseException:
                    clean = False
        except BaseException:
            clean = False
            raise
        finally:
            if acquired:
                self._call_lock.release()
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._closed = True
            self._closing = False
            self._clean_close = clean
            if not clean:
                self._timed_out = True
                self._intervention_required = True
        return clean
