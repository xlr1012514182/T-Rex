import asyncio

import numpy as np
import pytest

from revo3_v1.revo import (
    BrainCoSDKBackend,
    HardwareWriteNotArmed,
    JOINT_ORDER,
    JOINT_ORDER_HASH,
    RevoCommand,
)


EXPECTED_ORDER = (
    "little_MPR", "little_MCP", "little_PIP", "little_DIP",
    "ring_MPR", "ring_MCP", "ring_PIP", "ring_DIP",
    "middle_MPR", "middle_MCP", "middle_PIP", "middle_DIP",
    "index_MPR", "index_MCP", "index_PIP", "index_DIP",
    "thumb_MCP", "thumb_PIP", "thumb_DIP", "thumb_CMP", "thumb_CMR",
)


class _FakeSDK:
    def __init__(self):
        self.written = None

    def revo3_get_all_motor_positions(self, slave_id):
        assert slave_id == 7
        return [180.0] * 21

    def revo3_get_all_motor_velocities(self, slave_id):
        return [90.0] * 21

    def revo3_get_all_motor_currents(self, slave_id):
        return [500.0] * 21

    def revo3_get_all_motor_status(self, slave_id):
        return [0] * 21

    def revo3_set_all_motor_positions(self, slave_id, degrees):
        self.written = (slave_id, degrees)

    def revo3_get_all_collision_active(self, slave_id):
        return [False] * 21


class _Status:
    positions = [180.0] * 21
    velocities = [90.0] * 21
    currents = [500.0] * 21


class _AtomicFakeSDK(_FakeSDK):
    def revo3_get_motor_status_data(self, slave_id):
        assert slave_id == 7
        return _Status()


def test_joint_order_is_frozen_and_hashed():
    assert JOINT_ORDER == EXPECTED_ORDER
    assert len(JOINT_ORDER_HASH) == 64


def test_sdk_boundary_converts_degrees_rpm_and_milliamps_to_si():
    async def run():
        backend = BrainCoSDKBackend(_AtomicFakeSDK(), slave_id=7)
        state = await backend.read_state()
        np.testing.assert_allclose(state.q_rad, np.pi)
        np.testing.assert_allclose(state.dq_rad_s, 3 * np.pi)
        np.testing.assert_allclose(state.current_a, 0.5)

    asyncio.run(run())


def test_real_write_requires_explicit_arm_and_converts_back_to_degrees():
    async def run():
        sdk = _FakeSDK()
        command = RevoCommand(1, np.full(21, np.pi / 2), "task", 0)
        backend = BrainCoSDKBackend(sdk, slave_id=7)
        with pytest.raises(HardwareWriteNotArmed):
            await backend.write_command(command)
        armed = BrainCoSDKBackend(
            sdk,
            slave_id=7,
            allow_hardware_write=True,
            capability_probe_confirmed=True,
        )
        await armed.write_command(command)
        assert sdk.written[0] == 7
        np.testing.assert_allclose(sdk.written[1], 90.0, atol=1e-5)

    asyncio.run(run())


def test_hardware_write_stays_blocked_until_capability_probe_is_confirmed():
    async def run():
        sdk = _FakeSDK()
        command = RevoCommand(1, np.zeros(21), "task", 0)
        backend = BrainCoSDKBackend(sdk, slave_id=7, allow_hardware_write=True)
        with pytest.raises(HardwareWriteNotArmed, match="capability_probe_confirmed"):
            await backend.write_command(command)

    asyncio.run(run())


def test_missing_telemetry_or_collision_api_fails_closed():
    class MissingTelemetry:
        pass

    class MissingCollision(_AtomicFakeSDK):
        revo3_get_all_collision_active = None

    async def run():
        with pytest.raises(AttributeError, match="refusing to fabricate"):
            await BrainCoSDKBackend(MissingTelemetry(), slave_id=7).read_state()
        with pytest.raises(AttributeError, match="unknown collision state"):
            await BrainCoSDKBackend(MissingCollision(), slave_id=7).collision_active()

    asyncio.run(run())
