"""Official ``bc-revo3-sdk`` connection, probe, and native telemetry source.

The pinned BrainCo example installs ``bc-revo3-sdk==1.5.1`` and imports
``bc_revo3_sdk.main_mod``.  This module follows the documented auto-detect,
``init_from_detected``, motor-status, and tactile-summary APIs.  It never
silently assumes feedback units or a U21VT-to-T-Rex feature mapping.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from importlib import import_module, metadata
import inspect
import json
from typing import Any, Iterable

import numpy as np

from revo3_teleop.contracts import NativeSample, SampleHeader
from revo3_teleop.sources.common import Clock, monotonic_ns
from revo3_v1.revo import (
    JOINT_COUNT,
    JOINT_ORDER_HASH,
    BrainCoSDKBackend,
)


REVO3_SDK_IMPORT = "bc_revo3_sdk.main_mod"
REVO3_TOUCH_SUMMARY_SIZE = 42
REVO3_TOUCH_MODULE_COUNT = 11
REVO3_CLOCK_DOMAIN = "workstation_monotonic"


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _sdk_version() -> str:
    try:
        return metadata.version("bc-revo3-sdk")
    except metadata.PackageNotFoundError:
        return "unknown-injected-or-uninstalled"


def _json_scalar(value: Any) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raw = getattr(value, "value", None)
    if raw is not None and isinstance(raw, (str, int, float, bool)):
        return raw
    return str(value)


def _sha(value: dict[str, object]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _shape21(value: object, *, name: str, dtype: Any = np.float64) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.shape != (JOINT_COUNT,) or not np.isfinite(array).all():
        raise RuntimeError(f"Revo3 SDK {name} must be finite [{JOINT_COUNT}]")
    return array


@dataclass(frozen=True)
class Revo3ProbeConfig:
    port: str | None = None
    slave_id: int | None = None
    expected_serial: str | None = None
    expected_sdk_version: str | None = "1.5.1"
    allow_hardware_probe: bool = False

    def __post_init__(self) -> None:
        if self.port is not None and not self.port.strip():
            raise ValueError("port must be non-empty when supplied")
        if self.slave_id is not None and not 0 <= self.slave_id <= 255:
            raise ValueError("slave_id must be in [0,255]")
        if self.expected_serial is not None and not self.expected_serial.strip():
            raise ValueError("expected_serial must be non-empty when supplied")


@dataclass(frozen=True)
class Revo3ProbeReport:
    sdk_import: str
    sdk_version: str
    port: str
    protocol: object
    slave_id: int
    serial_number: str
    firmware_version: str
    hardware_type: object
    motor_count: int
    motor_status_shape: tuple[int, ...]
    motor_status_bitmask_shape: tuple[int, ...]
    collision_api: str
    collision_status_shape: tuple[int, ...]
    touch_vendor: int
    touch_enabled_mask: int
    touch_summary_size: int
    touch_module_lengths: tuple[int, ...]
    position_sdk_unit: str
    velocity_sdk_unit: str
    current_sdk_unit: str
    touch_summary_unit_claim: str
    device_timestamp_available: bool
    capture_clock_provenance: str
    u21vt_identity_verified: bool

    def fingerprint(self) -> str:
        return _sha(asdict(self))

    def to_json(self) -> dict[str, object]:
        result = asdict(self)
        result["fingerprint"] = self.fingerprint()
        return result


@dataclass(frozen=True)
class Revo3BenchApproval:
    """Human-reviewed facts that cannot be established from public API shape."""

    probe_fingerprint: str
    joint_order_hash: str
    feedback_velocity_unit: str
    feedback_current_unit: str
    state_units_bench_verified: bool
    u21vt_identity_verified: bool
    tactile_zero_and_saturation_verified: bool
    allow_hardware_write: bool = False
    physical_estop_verified: bool = False
    hold_path_verified: bool = False
    limits_verified: bool = False

    def validate(self, report: Revo3ProbeReport) -> None:
        if self.probe_fingerprint != report.fingerprint():
            raise ValueError("Revo3 bench approval does not match capability probe")
        if self.joint_order_hash != JOINT_ORDER_HASH:
            raise ValueError("Revo3 bench approval joint order does not match canonical order")
        if self.feedback_velocity_unit not in {"rpm", "deg/s"}:
            raise ValueError("feedback_velocity_unit must be bench-approved as rpm or deg/s")
        if self.feedback_current_unit not in {"mA", "A"}:
            raise ValueError("feedback_current_unit must be bench-approved as mA or A")
        if not self.state_units_bench_verified:
            raise ValueError("state units must be bench verified before SI conversion")
        if not self.u21vt_identity_verified or not self.tactile_zero_and_saturation_verified:
            raise ValueError("U21VT identity, zero, and saturation must be bench verified")
        if self.allow_hardware_write and not all(
            (self.physical_estop_verified, self.hold_path_verified, self.limits_verified)
        ):
            raise ValueError("Revo writes require estop, hold path, and limits verification")


@dataclass(frozen=True)
class U21VTPressureZoneProjection:
    """Bench-defined selection of 30 values from the 42 pressure zones.

    This is *not* a six-axis force/torque observation.  The public Revo3 SDK
    documents the values as aggregate pressure zones; selecting six zones per
    finger cannot turn them into ``[Fx, Fy, Fz, Mx, My, Mz]``.  The result is
    therefore stored only as ``pressure_zones_n`` and is intentionally
    ineligible for the T-Rex ``features`` exporter key.
    """

    summary_indices: tuple[tuple[int, int, int, int, int, int], ...]
    revision: str
    bench_verified: bool
    output_unit: str = "N"

    def __post_init__(self) -> None:
        if len(self.summary_indices) != 5 or any(len(row) != 6 for row in self.summary_indices):
            raise ValueError("summary_indices must have shape [5,6]")
        flat = [int(index) for row in self.summary_indices for index in row]
        if any(index < 0 or index >= REVO3_TOUCH_SUMMARY_SIZE for index in flat):
            raise ValueError("U21VT summary index must be in [0,41]")
        if len(set(flat)) != len(flat):
            raise ValueError("U21VT pressure-zone projection cannot duplicate zones")
        if not self.revision.strip():
            raise ValueError("projection revision must be non-empty")
        if not self.bench_verified:
            raise ValueError("pressure-zone projection must be physically bench verified")
        if self.output_unit != "N":
            raise ValueError("the supported projection converts documented mN to SI newtons")

    def project(self, summary_mn: Iterable[float]) -> np.ndarray:
        summary = np.asarray(summary_mn, dtype=np.float32)
        if summary.shape != (REVO3_TOUCH_SUMMARY_SIZE,) or not np.isfinite(summary).all():
            raise ValueError("touch summary must be finite [42]")
        indices = np.asarray(self.summary_indices, dtype=np.int64)
        return (summary[indices] * 1e-3).astype(np.float32)


# Backwards-compatible import name only.  Its output is pressure-zone data and
# never receives the exporter-reserved ``features`` key.
U21VTSummaryProjection = U21VTPressureZoneProjection


class Revo3ProbedConnection:
    """Open SDK connection carrying immutable probe evidence."""

    def __init__(self, sdk: Any, client: Any, report: Revo3ProbeReport) -> None:
        self.sdk = sdk
        self.client = client
        self.report = report
        self._closed = False

    def make_backend(self, approval: Revo3BenchApproval) -> BrainCoSDKBackend:
        if self._closed:
            raise RuntimeError("Revo3 connection is closed")
        approval.validate(self.report)
        return BrainCoSDKBackend(
            self.client,
            slave_id=self.report.slave_id,
            allow_hardware_write=approval.allow_hardware_write,
            feedback_velocity_unit=approval.feedback_velocity_unit,
            feedback_current_unit=approval.feedback_current_unit,
            capability_probe_confirmed=True,
        )

    async def close(self) -> None:
        if self._closed:
            return
        close = getattr(self.sdk, "modbus_close", None)
        if not callable(close):
            raise RuntimeError("bc-revo3-sdk exposes no reviewed close function")
        await _maybe_await(close(self.client))
        # Mark closed only after the vendor close succeeds.  If it raises, the
        # caller retains this live connection object and can retry or escalate
        # to process/device intervention instead of silently losing authority.
        self._closed = True


class BrainCoRevo3SdkAssembly:
    """Lazy official SDK loader and read-only capability probe."""

    def __init__(
        self,
        config: Revo3ProbeConfig,
        *,
        sdk_module: Any | None = None,
        sdk_version: str | None = None,
    ) -> None:
        self.config = config
        self._sdk = sdk_module
        self._sdk_version = _sdk_version() if sdk_version is None else str(sdk_version)

    def _module(self) -> Any:
        if self._sdk is None:
            self._sdk = import_module(REVO3_SDK_IMPORT)
        return self._sdk

    async def probe(self) -> Revo3ProbedConnection:
        if not self.config.allow_hardware_probe:
            raise PermissionError("Revo3 hardware probe is disabled")
        if (
            self.config.expected_sdk_version is not None
            and self._sdk_version != self.config.expected_sdk_version
        ):
            raise RuntimeError(
                "bc-revo3-sdk version mismatch: "
                f"expected={self.config.expected_sdk_version}, observed={self._sdk_version}"
            )
        sdk = self._module()
        for name in ("revo3_auto_detect", "init_from_detected", "modbus_close"):
            if not callable(getattr(sdk, name, None)):
                raise RuntimeError(f"bc-revo3-sdk API mismatch: missing {name}")
        kwargs: dict[str, object] = {"scan_all": True}
        if self.config.port is not None:
            kwargs["port"] = self.config.port
        if self.config.slave_id is not None:
            kwargs["slave_id"] = self.config.slave_id
        devices = list(await _maybe_await(sdk.revo3_auto_detect(**kwargs)))
        if self.config.expected_serial is not None:
            devices = [
                device
                for device in devices
                if str(getattr(device, "serial_number", "")) == self.config.expected_serial
            ]
        if len(devices) != 1:
            raise RuntimeError(
                f"Revo3 probe requires exactly one matching device, observed {len(devices)}"
            )
        device = devices[0]
        client = await _maybe_await(sdk.init_from_detected(device))
        try:
            slave_id = int(getattr(device, "slave_id"))
            info = await _maybe_await(client.revo3_get_device_info(slave_id))
            serial = str(
                getattr(info, "serial_number", None)
                or getattr(device, "serial_number", "")
            )
            if not serial:
                raise RuntimeError("Revo3 probe could not read a serial number")
            if self.config.expected_serial is not None and serial != self.config.expected_serial:
                raise RuntimeError("connected Revo3 serial differs from expected_serial")
            status = await _maybe_await(client.revo3_get_motor_status_data(slave_id))
            _shape21(status.positions, name="positions")
            _shape21(status.velocities, name="velocities")
            _shape21(status.currents, name="currents")
            status_method = getattr(client, "revo3_get_all_motor_status", None)
            if not callable(status_method):
                raise RuntimeError("Revo3 safety probe requires revo3_get_all_motor_status")
            status_bits = _shape21(
                await _maybe_await(status_method(slave_id)),
                name="motor status bitmask",
                dtype=np.int64,
            )
            collision_api = ""
            if callable(getattr(client, "revo3_get_all_collision_active", None)):
                collision_api = "revo3_get_all_collision_active"
                collision_active = _shape21(
                    await _maybe_await(
                        client.revo3_get_all_collision_active(slave_id)
                    ),
                    name="collision active",
                    dtype=np.bool_,
                )
            elif callable(getattr(client, "revo3_is_collision_active", None)):
                collision_api = "revo3_is_collision_active"
                collision_active = _shape21(
                    [
                        await _maybe_await(
                            client.revo3_is_collision_active(slave_id, joint_id)
                        )
                        for joint_id in range(JOINT_COUNT)
                    ],
                    name="collision active",
                    dtype=np.bool_,
                )
            if not collision_api:
                raise RuntimeError("Revo3 collision capability is unavailable")

            touch_vendor = int(await _maybe_await(client.revo3_get_touch_vendor(slave_id)))
            touch_enabled = int(
                await _maybe_await(client.revo3_get_all_touch_modules_enabled(slave_id))
            )
            summary = np.asarray(
                await _maybe_await(client.revo3_get_touch_summary(slave_id)),
                dtype=np.float64,
            )
            if summary.shape != (REVO3_TOUCH_SUMMARY_SIZE,) or not np.isfinite(summary).all():
                raise RuntimeError("Revo3 tactile summary must be finite [42]")
            module_lengths: tuple[int, ...] = ()
            all_touch = getattr(client, "revo3_get_all_touch_data", None)
            if callable(all_touch):
                touch_data = await _maybe_await(all_touch(slave_id))
                modules = list(touch_data.modules)
                if len(modules) != REVO3_TOUCH_MODULE_COUNT:
                    raise RuntimeError("Revo3 all-touch response must contain 11 modules")
                module_lengths = tuple(len(module) for module in modules)

            report = Revo3ProbeReport(
                sdk_import=REVO3_SDK_IMPORT,
                sdk_version=self._sdk_version,
                port=str(getattr(device, "port_name", self.config.port or "unknown")),
                protocol=_json_scalar(getattr(device, "protocol_type", "unknown")),
                slave_id=slave_id,
                serial_number=serial,
                firmware_version=str(
                    getattr(info, "firmware_version", None)
                    or getattr(device, "firmware_version", "unknown")
                ),
                hardware_type=_json_scalar(
                    getattr(info, "hardware_type", None)
                    or getattr(device, "hardware_type", "unknown")
                ),
                motor_count=JOINT_COUNT,
                motor_status_shape=tuple(np.asarray(status.positions).shape),
                motor_status_bitmask_shape=tuple(status_bits.shape),
                collision_api=collision_api,
                collision_status_shape=tuple(collision_active.shape),
                touch_vendor=touch_vendor,
                touch_enabled_mask=touch_enabled,
                touch_summary_size=int(summary.size),
                touch_module_lengths=module_lengths,
                position_sdk_unit="deg_pinned_sdk_documentation",
                velocity_sdk_unit="UNVERIFIED_REQUIRES_BENCH_APPROVAL",
                current_sdk_unit="UNVERIFIED_REQUIRES_BENCH_APPROVAL",
                touch_summary_unit_claim="mN_pinned_sdk_documentation_not_bench_verified",
                device_timestamp_available=False,
                capture_clock_provenance="host_sdk_read_completion_monotonic",
                u21vt_identity_verified=False,
            )
            return Revo3ProbedConnection(sdk, client, report)
        except Exception as probe_failure:
            try:
                await _maybe_await(sdk.modbus_close(client))
            except BaseException as close_failure:
                raise RuntimeError(
                    "Revo3 capability probe failed and the SDK connection could not "
                    "be closed; process/device intervention is required"
                ) from probe_failure
            raise


class Revo3TelemetrySource:
    """Native-rate SI state and raw U21VT evidence from a probed connection."""

    def __init__(
        self,
        connection: Revo3ProbedConnection,
        approval: Revo3BenchApproval,
        *,
        projection: U21VTPressureZoneProjection | None = None,
        clock: Clock = monotonic_ns,
    ) -> None:
        approval.validate(connection.report)
        self.connection = connection
        self.approval = approval
        self.backend = connection.make_backend(approval)
        self.projection = projection
        self._clock = clock
        self._state_sequence = 0
        self._touch_sequence = 0

    @property
    def episode_metadata(self) -> dict[str, object]:
        return {
            "revo3_probe": self.connection.report.to_json(),
            "joint_order_hash": JOINT_ORDER_HASH,
            "internal_position_unit": "rad",
            "internal_velocity_unit": "rad/s",
            "internal_current_unit": "A",
            "device_timestamp_available": False,
            "capture_clock_provenance": "host_sdk_read_completion_monotonic",
            "u21vt_pressure_zone_projection": None
            if self.projection is None
            else {
                "revision": self.projection.revision,
                "summary_indices": [list(row) for row in self.projection.summary_indices],
                "source_unit": "mN",
                "output_unit": self.projection.output_unit,
            },
        }

    async def poll_state(self) -> NativeSample:
        read_start_ns = int(self._clock())
        state = await self.backend.read_state()
        read_end_ns = max(int(self._clock()), read_start_ns + 1, state.timestamp_ns)
        sequence = self._state_sequence
        self._state_sequence += 1
        return NativeSample(
            SampleHeader(
                source_id="revo3_motor_sdk",
                sequence=sequence,
                capture_timestamp_ns=read_end_ns,
                receive_timestamp_ns=read_end_ns,
                clock_domain=REVO3_CLOCK_DOMAIN,
                device_timestamp_ns=None,
            ),
            {
                "q_rad": state.q_rad,
                "dq_rad_s": state.dq_rad_s,
                "current_a": state.current_a,
                "status": state.status,
                "host_read_start_timestamp_ns": np.asarray([read_start_ns], dtype=np.int64),
                "host_read_end_timestamp_ns": np.asarray([read_end_ns], dtype=np.int64),
                "device_timestamp_available": np.asarray([0], dtype=np.uint8),
                "clock_is_host_reconstruction": np.asarray([1], dtype=np.uint8),
            },
        )

    async def poll_touch(self) -> NativeSample:
        read_start_ns = int(self._clock())
        client = self.connection.client
        slave_id = self.connection.report.slave_id
        all_touch = getattr(client, "revo3_get_all_touch_data", None)
        modules_flat = np.empty(0, dtype=np.float32)
        module_lengths = np.empty(0, dtype=np.int16)
        if callable(all_touch):
            result = await _maybe_await(all_touch(slave_id))
            summary = np.asarray(result.summary, dtype=np.float32)
            modules = [np.asarray(module, dtype=np.float32) for module in result.modules]
            observed_lengths = tuple(int(module.size) for module in modules)
            if observed_lengths != self.connection.report.touch_module_lengths:
                raise RuntimeError("U21VT module layout changed since capability probe")
            module_lengths = np.asarray(observed_lengths, dtype=np.int16)
            modules_flat = np.concatenate(modules) if modules else modules_flat
        else:
            summary = np.asarray(
                await _maybe_await(client.revo3_get_touch_summary(slave_id)),
                dtype=np.float32,
            )
        if summary.shape != (REVO3_TOUCH_SUMMARY_SIZE,) or not np.isfinite(summary).all():
            raise RuntimeError("U21VT summary changed schema or contains non-finite values")
        read_end_ns = max(int(self._clock()), read_start_ns + 1)
        sequence = self._touch_sequence
        self._touch_sequence += 1
        payload: dict[str, np.ndarray] = {
            "summary_mn": summary,
            "modules_flat_raw": modules_flat,
            "module_lengths": module_lengths,
            "host_read_start_timestamp_ns": np.asarray([read_start_ns], dtype=np.int64),
            "host_read_end_timestamp_ns": np.asarray([read_end_ns], dtype=np.int64),
            "device_timestamp_available": np.asarray([0], dtype=np.uint8),
            "clock_is_host_reconstruction": np.asarray([1], dtype=np.uint8),
        }
        if self.projection is not None:
            payload["pressure_zones_n"] = self.projection.project(summary)
        return NativeSample(
            SampleHeader(
                source_id="revo3_u21vt_sdk",
                sequence=sequence,
                capture_timestamp_ns=read_end_ns,
                receive_timestamp_ns=read_end_ns,
                clock_domain=REVO3_CLOCK_DOMAIN,
                device_timestamp_ns=None,
            ),
            payload,
        )


__all__ = [
    "BrainCoRevo3SdkAssembly",
    "REVO3_CLOCK_DOMAIN",
    "REVO3_SDK_IMPORT",
    "REVO3_TOUCH_MODULE_COUNT",
    "REVO3_TOUCH_SUMMARY_SIZE",
    "Revo3BenchApproval",
    "Revo3ProbeConfig",
    "Revo3ProbeReport",
    "Revo3ProbedConnection",
    "Revo3TelemetrySource",
    "U21VTPressureZoneProjection",
    "U21VTSummaryProjection",
]
