from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from revo3_teleop.hardware.revo3_sdk import (
    BrainCoRevo3SdkAssembly,
    Revo3BenchApproval,
    Revo3ProbeConfig,
    Revo3TelemetrySource,
    U21VTPressureZoneProjection,
)
from revo3_v1.revo import JOINT_ORDER_HASH


class _FakeRevoClient:
    def __init__(self, *, collision: bool = True) -> None:
        self.collision = collision

    async def revo3_get_device_info(self, slave_id):
        return SimpleNamespace(
            serial_number="RV3-001",
            firmware_version="fw-1",
            hardware_type=31,
        )

    async def revo3_get_motor_status_data(self, slave_id):
        return SimpleNamespace(
            positions=np.full(21, 180.0),
            velocities=np.full(21, 60.0),
            currents=np.full(21, 1_000.0),
        )

    async def revo3_get_all_motor_status(self, slave_id):
        return np.zeros(21, dtype=np.int64)

    async def revo3_get_all_collision_active(self, slave_id):
        return np.zeros(21, dtype=bool)

    async def revo3_get_touch_vendor(self, slave_id):
        return 1

    async def revo3_get_all_touch_modules_enabled(self, slave_id):
        return 0x7FF

    async def revo3_get_touch_summary(self, slave_id):
        return np.arange(42, dtype=np.float32)

    async def revo3_get_all_touch_data(self, slave_id):
        return SimpleNamespace(
            summary=np.arange(42, dtype=np.float32),
            modules=[np.arange(index + 1, dtype=np.float32) for index in range(11)],
        )

    async def revo3_set_all_motor_positions(self, slave_id, values):
        self.sent = values


class _FakeRevoSdk:
    def __init__(self, client=None) -> None:
        self.client = client or _FakeRevoClient()
        self.closed = 0
        self.close_failures_remaining = 0
        self.device = SimpleNamespace(
            port_name="COM7",
            protocol_type=2,
            slave_id=126,
            serial_number="RV3-001",
            firmware_version="fw-1",
            hardware_type=31,
        )

    async def revo3_auto_detect(self, **kwargs):
        self.detect_kwargs = kwargs
        return [self.device]

    async def init_from_detected(self, device):
        assert device is self.device
        return self.client

    async def modbus_close(self, client):
        assert client is self.client
        self.closed += 1
        if self.close_failures_remaining:
            self.close_failures_remaining -= 1
            raise RuntimeError("injected close failure")


def _approval(report, *, allow_write: bool = False) -> Revo3BenchApproval:
    return Revo3BenchApproval(
        probe_fingerprint=report.fingerprint(),
        joint_order_hash=JOINT_ORDER_HASH,
        feedback_velocity_unit="rpm",
        feedback_current_unit="mA",
        state_units_bench_verified=True,
        u21vt_identity_verified=True,
        tactile_zero_and_saturation_verified=True,
        allow_hardware_write=allow_write,
        physical_estop_verified=allow_write,
        hold_path_verified=allow_write,
        limits_verified=allow_write,
    )


def test_revo_probe_is_read_only_and_units_require_separate_bench_approval() -> None:
    async def scenario():
        sdk = _FakeRevoSdk()
        disabled = BrainCoRevo3SdkAssembly(
            Revo3ProbeConfig(), sdk_module=sdk, sdk_version="1.5.1"
        )
        with pytest.raises(PermissionError, match="probe is disabled"):
            await disabled.probe()

        assembly = BrainCoRevo3SdkAssembly(
            Revo3ProbeConfig(
                expected_serial="RV3-001", allow_hardware_probe=True
            ),
            sdk_module=sdk,
            sdk_version="1.5.1",
        )
        connection = await assembly.probe()
        report = connection.report
        assert report.motor_count == 21
        assert report.touch_summary_size == 42
        assert report.touch_module_lengths == tuple(range(1, 12))
        assert report.collision_status_shape == (21,)
        assert report.velocity_sdk_unit.startswith("UNVERIFIED")
        assert report.current_sdk_unit.startswith("UNVERIFIED")
        assert not report.u21vt_identity_verified

        bad = _approval(report)
        object.__setattr__(bad, "state_units_bench_verified", False)
        with pytest.raises(ValueError, match="state units"):
            connection.make_backend(bad)

        projection = U21VTPressureZoneProjection(
            summary_indices=tuple(
                tuple(range(finger * 6, finger * 6 + 6)) for finger in range(5)
            ),
            revision="physical-map-r1",
            bench_verified=True,
        )
        ticks = iter([1_000, 1_001, 2_000, 2_001])
        source = Revo3TelemetrySource(
            connection, _approval(report), projection=projection, clock=lambda: next(ticks)
        )
        state = await source.poll_state()
        np.testing.assert_allclose(state.payload["q_rad"], np.pi, rtol=1e-6)
        np.testing.assert_allclose(state.payload["dq_rad_s"], 2 * np.pi, rtol=1e-6)
        np.testing.assert_allclose(state.payload["current_a"], 1.0, rtol=1e-6)
        assert state.header.device_timestamp_ns is None
        touch = await source.poll_touch()
        assert touch.payload["summary_mn"].shape == (42,)
        assert "features" not in touch.payload
        assert touch.payload["pressure_zones_n"].shape == (5, 6)
        assert touch.payload["pressure_zones_n"][0, 1] == pytest.approx(0.001)
        assert touch.payload["modules_flat_raw"].size == sum(range(1, 12))
        await connection.close()
        assert sdk.closed == 1

    asyncio.run(scenario())


def test_revo_probe_fails_closed_without_collision_capability() -> None:
    class NoCollision(_FakeRevoClient):
        revo3_get_all_collision_active = None

    async def scenario():
        sdk = _FakeRevoSdk(NoCollision())
        assembly = BrainCoRevo3SdkAssembly(
            Revo3ProbeConfig(allow_hardware_probe=True),
            sdk_module=sdk,
            sdk_version="1.5.1",
        )
        with pytest.raises(RuntimeError, match="collision capability"):
            await assembly.probe()
        assert sdk.closed == 1

    asyncio.run(scenario())


def test_revo_connection_close_failure_keeps_handle_retryable() -> None:
    async def scenario():
        sdk = _FakeRevoSdk()
        sdk.close_failures_remaining = 1
        assembly = BrainCoRevo3SdkAssembly(
            Revo3ProbeConfig(allow_hardware_probe=True),
            sdk_module=sdk,
            sdk_version="1.5.1",
        )
        connection = await assembly.probe()
        with pytest.raises(RuntimeError, match="injected close failure"):
            await connection.close()
        await connection.close()
        assert sdk.closed == 2

    asyncio.run(scenario())


def test_revo_failed_probe_reports_close_intervention() -> None:
    class NoCollision(_FakeRevoClient):
        revo3_get_all_collision_active = None

    async def scenario():
        sdk = _FakeRevoSdk(NoCollision())
        sdk.close_failures_remaining = 1
        assembly = BrainCoRevo3SdkAssembly(
            Revo3ProbeConfig(allow_hardware_probe=True),
            sdk_module=sdk,
            sdk_version="1.5.1",
        )
        with pytest.raises(RuntimeError, match="process/device intervention"):
            await assembly.probe()

    asyncio.run(scenario())
