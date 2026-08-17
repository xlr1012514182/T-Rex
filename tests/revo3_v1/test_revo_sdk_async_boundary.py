import asyncio
import threading
import time

import numpy as np
import pytest

from revo3_v1.revo import (
    BrainCoSDKBackend,
    RevoCommand,
    SDKBackendClosed,
    SDKBackendFault,
    SDKCallTimeout,
)


class _Status:
    positions = [0.0] * 21
    velocities = [0.0] * 21
    currents = [0.0] * 21
    temperatures = [25.0] * 21


def _command(value: float, *, timestamp_ns: int = 1) -> RevoCommand:
    return RevoCommand(
        timestamp_ns,
        np.full(21, value, dtype=np.float32),
        "task",
        1,
    )


def _armed_backend(client, **kwargs) -> BrainCoSDKBackend:
    return BrainCoSDKBackend(
        client,
        slave_id=7,
        allow_hardware_write=True,
        capability_probe_confirmed=True,
        temperature_telemetry_verified=True,
        soft_stop_callback=kwargs.pop(
            "soft_stop_callback", lambda client, slave_id, reason: None
        ),
        soft_stop_capability_verified=True,
        collision_profile_id="bench-profile-sha256",
        collision_profile_verified=True,
        **kwargs,
    )


async def _heartbeat(stop: asyncio.Event, ticks: list[int]) -> None:
    while not stop.is_set():
        ticks[0] += 1
        await asyncio.sleep(0.002)


def test_blocking_sync_read_runs_off_event_loop_and_close_is_explicit():
    class BlockingReadSDK:
        def revo3_get_motor_status_data(self, slave_id):
            assert slave_id == 7
            time.sleep(0.06)
            return _Status()

        def revo3_get_all_motor_status(self, slave_id):
            return [0] * 21

    async def run():
        backend = BrainCoSDKBackend(
            BlockingReadSDK(), slave_id=7, sdk_call_timeout_s=0.2
        )
        stop = asyncio.Event()
        ticks = [0]
        pulse = asyncio.create_task(_heartbeat(stop, ticks))
        state = await backend.read_state()
        stop.set()
        await pulse
        assert state.q_rad.shape == (21,)
        # A direct synchronous SDK call would leave this at zero or one.
        assert ticks[0] >= 5
        assert await backend.close()
        assert backend.closed
        with pytest.raises(SDKBackendClosed):
            await backend.read_state()

    asyncio.run(run())


def test_blocking_sync_writes_are_off_loop_and_strictly_serialized_in_order():
    class OrderedSDK:
        def __init__(self):
            self.events: list[tuple[str, int]] = []
            self.active = 0
            self.max_active = 0
            self.lock = threading.Lock()

        def revo3_set_all_motor_positions(self, slave_id, degrees):
            marker = int(round(degrees[0]))
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                self.events.append(("start", marker))
            time.sleep(0.04)
            with self.lock:
                self.events.append(("end", marker))
                self.active -= 1

    async def run():
        sdk = OrderedSDK()
        backend = _armed_backend(sdk, sdk_call_timeout_s=0.2)
        stop = asyncio.Event()
        ticks = [0]
        pulse = asyncio.create_task(_heartbeat(stop, ticks))
        first = asyncio.create_task(backend.write_command(_command(np.pi / 180)))
        await asyncio.sleep(0)
        second = asyncio.create_task(backend.write_command(_command(2 * np.pi / 180)))
        await asyncio.gather(first, second)
        stop.set()
        await pulse
        assert sdk.events == [
            ("start", 1),
            ("end", 1),
            ("start", 2),
            ("end", 2),
        ]
        assert sdk.max_active == 1
        assert ticks[0] >= 10
        assert await backend.close()

    asyncio.run(run())

def test_timeout_latches_fail_closed_then_serialized_soft_stop_can_drain():
    class SlowWriteSDK:
        def __init__(self):
            self.writes = 0

        def revo3_set_all_motor_positions(self, slave_id, degrees):
            self.writes += 1
            time.sleep(0.06)

        def revo3_get_all_collision_active(self, slave_id):
            return [False] * 21

    async def run():
        sdk = SlowWriteSDK()
        stops: list[str] = []
        backend = _armed_backend(
            sdk,
            sdk_call_timeout_s=0.01,
            soft_stop_timeout_s=0.20,
            soft_stop_callback=lambda client, slave_id, reason: stops.append(reason),
        )
        with pytest.raises(SDKCallTimeout):
            await backend.write_command(_command(0.1))
        assert backend.fault_reason == "write_command:SDKCallTimeout"
        assert backend.intervention_required
        # Normal control is now fail-closed; it cannot slip a new call behind
        # the timed-out vendor write.
        with pytest.raises(SDKBackendFault):
            await backend.collision_active()
        await backend.soft_stop("write_timeout")
        assert stops == ["write_timeout"]
        assert sdk.writes == 1
        assert await backend.close()

    asyncio.run(run())


def test_soft_stop_and_close_timeout_report_unresolved_manual_intervention():
    class HungEnoughSDK:
        def revo3_set_all_motor_positions(self, slave_id, degrees):
            time.sleep(0.15)

    async def run():
        stops: list[str] = []
        backend = _armed_backend(
            HungEnoughSDK(),
            sdk_call_timeout_s=0.01,
            soft_stop_timeout_s=0.02,
            close_timeout_s=0.02,
            soft_stop_callback=lambda client, slave_id, reason: stops.append(reason),
        )
        with pytest.raises(SDKCallTimeout):
            await backend.write_command(_command(0.1))
        with pytest.raises(SDKCallTimeout):
            await backend.soft_stop("write_timeout")
        assert stops == []
        assert not await backend.close()
        assert backend.closed and backend.intervention_required
        # Let the finite fake vendor call leave the executor thread; close did
        # not block the event loop or launch a late unowned SoftStop.
        await asyncio.sleep(0.16)
        assert stops == []

    asyncio.run(run())


def test_async_sdk_methods_remain_awaited_and_share_the_same_serial_boundary():
    class AsyncSDK:
        def __init__(self):
            self.events: list[str] = []

        async def revo3_get_motor_status_data(self, slave_id):
            self.events.append("read:start")
            await asyncio.sleep(0.005)
            self.events.append("read:end")
            return _Status()

        async def revo3_get_all_motor_status(self, slave_id):
            await asyncio.sleep(0)
            return [0] * 21

        async def revo3_get_all_collision_active(self, slave_id):
            await asyncio.sleep(0)
            return [False] * 21

        async def revo3_set_all_motor_positions(self, slave_id, degrees):
            self.events.append("write")
            await asyncio.sleep(0)

    async def run():
        sdk = AsyncSDK()

        async def stop(client, slave_id, reason):
            await asyncio.sleep(0)
            client.events.append(f"stop:{reason}")

        backend = _armed_backend(sdk, soft_stop_callback=stop)
        state = await backend.read_state()
        assert state.temperature_c is not None
        assert not await backend.collision_active()
        await backend.write_command(_command(0.0))
        await backend.soft_stop("done")
        assert sdk.events == ["read:start", "read:end", "write", "stop:done"]
        assert await backend.close()

    asyncio.run(run())
