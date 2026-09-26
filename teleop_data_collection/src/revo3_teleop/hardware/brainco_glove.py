"""Concrete, fail-closed ``bc-edu-sdk`` motion-glove lifecycle.

This adapter follows BrainCoTech ``brainco-hand-sdk`` commit
``5c399113efd35f5ce664d5d8f3e8ff750ced5d23``
(``python/edu/glove_example.py``): USB VID 21059 with glove PIDs 6/2,
``EduDevice``/``MessageParser``/``SensorProfile``, six flex channels at
50 Hz, calibrated IMU at 100 Hz, and calibrated magnetometer at 20 Hz.

The SDK callback setters are process-global.  Import, USB discovery, operator
confirmation, and streaming are therefore separate gates.  A stuck SDK thread
retains both its thread handle and callback ownership so another client cannot
silently steal the process-global callbacks.

This module exposes raw glove telemetry only.  It does not infer a 6-DoF wrist
pose and does not map six flex values into a 21-DoF Revo target.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import hashlib
from importlib import import_module, metadata
import json
import threading
from typing import Any, Callable, Sequence

from .brainco_edu_callbacks import (
    BRAINCO_EDU_CALLBACK_NAMESPACE,
    claim_brainco_edu_callback_namespace,
    release_brainco_edu_callback_namespace,
)


BRAINCO_EDU_IMPORT = "bc_edu_sdk.main_mod"
BRAINCO_USB_VID = 21059
GLOVE_USB_PIDS = (6, 2)


def _stable_fingerprint(value: dict[str, object]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sdk_version() -> str:
    try:
        return metadata.version("bc-edu-sdk")
    except metadata.PackageNotFoundError:
        return "unknown-injected-or-uninstalled"


def _port_rows(raw: object) -> list[dict[str, object]]:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, list):
        raise ValueError("bc-edu-sdk available_usb_ports must return a JSON list")
    result: list[dict[str, object]] = []
    for item in raw:
        if not isinstance(item, dict) or not str(item.get("port_name", "")).strip():
            raise ValueError("bc-edu-sdk returned a glove port row without port_name")
        result.append({str(key): value for key, value in item.items()})
    return result


@dataclass(frozen=True)
class BrainCoEduGloveConfig:
    """Static glove configuration; construction has no hardware side effect."""

    port_name: str | None = None
    baudrate: int = 115_200
    expected_serial: str | None = None
    expected_sdk_version: str | None = "0.5.0"
    allow_hardware_discovery: bool = False
    allow_hardware_stream: bool = False

    def __post_init__(self) -> None:
        if self.port_name is not None and not self.port_name.strip():
            raise ValueError("port_name must be non-empty when supplied")
        if self.baudrate <= 0:
            raise ValueError("baudrate must be positive")
        if self.expected_serial is not None and not self.expected_serial.strip():
            raise ValueError("expected_serial must be non-empty when supplied")
        if self.expected_sdk_version is not None and not self.expected_sdk_version.strip():
            raise ValueError("expected_sdk_version must be non-empty when supplied")


@dataclass(frozen=True)
class BrainCoEduGloveDiscovery:
    sdk_import: str
    sdk_version: str
    selected_port: str
    discovered_ports: tuple[str, ...]
    matched_pid: int
    selected_usb_descriptor_hash: str
    selected_serial: str | None
    baudrate: int
    flex_channels: int = 6
    flex_sample_rate_hz: int = 50
    imu_sample_rate_hz: int = 100
    imu_upload_mode: str = "calibrated"
    mag_sample_rate_hz: int = 20
    mag_upload_mode: str = "calibrated"
    device_timestamp_available: bool = False
    provides_wrist_pose: bool = False
    produces_revo_target: bool = False
    callback_scope: str = "module_global_per_signal"

    def fingerprint(self) -> str:
        return _stable_fingerprint(asdict(self))

    def to_json(self) -> dict[str, object]:
        result = asdict(self)
        result["fingerprint"] = self.fingerprint()
        return result


class BrainCoEduSdkGloveClient:
    """Thread-hosted official SDK client implementing ``BrainCoGloveClient``."""

    _global_owner_lock = threading.Lock()
    _global_owner: "BrainCoEduSdkGloveClient | None" = None
    callback_namespace = BRAINCO_EDU_CALLBACK_NAMESPACE

    def __init__(
        self,
        config: BrainCoEduGloveConfig,
        *,
        sdk_module: Any | None = None,
        sdk_version: str | None = None,
        startup_timeout_s: float = 10.0,
    ) -> None:
        self.config = config
        self._sdk = sdk_module
        self._sdk_version = _sdk_version() if sdk_version is None else str(sdk_version)
        self._startup_timeout_s = float(startup_timeout_s)
        if self._startup_timeout_s <= 0:
            raise ValueError("startup_timeout_s must be positive")
        self._flex_callback: Callable[[Sequence[Sequence[object]]], None] | None = None
        self._imu_callback: Callable[[Sequence[Sequence[object]]], None] | None = None
        self._mag_callback: Callable[[Sequence[Sequence[object]]], None] | None = None
        self._discovery: BrainCoEduGloveDiscovery | None = None
        self._expected_fingerprint: str | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread_failure: BaseException | None = None

    def _module(self) -> Any:
        if self._sdk is None:
            self._sdk = import_module(BRAINCO_EDU_IMPORT)
        return self._sdk

    @staticmethod
    def _validate_api(sdk: Any) -> None:
        required = (
            "available_usb_ports",
            "EduDevice",
            "MessageParser",
            "SensorProfile",
            "set_msg_resp_callback",
            "set_flex_data_callback",
            "set_imu_data_callback",
            "set_imu_calibration_data_callback",
            "set_mag_data_callback",
            "set_mag_calibration_data_callback",
        )
        missing = [name for name in required if not callable(getattr(sdk, name, None))]
        for enum_name in (
            "MsgType",
            "SamplingRate",
            "ImuSampleRate",
            "MagSampleRate",
            "UploadDataType",
        ):
            if not hasattr(sdk, enum_name):
                missing.append(enum_name)
        if missing:
            raise RuntimeError("bc-edu-sdk glove API mismatch: missing " + ", ".join(missing))

    def discover(self) -> BrainCoEduGloveDiscovery:
        if not self.config.allow_hardware_discovery:
            raise PermissionError("BrainCo EDU glove hardware discovery is disabled")
        if (
            self.config.expected_sdk_version is not None
            and self._sdk_version != self.config.expected_sdk_version
        ):
            raise RuntimeError(
                "bc-edu-sdk version mismatch: "
                f"expected={self.config.expected_sdk_version}, observed={self._sdk_version}"
            )
        sdk = self._module()
        self._validate_api(sdk)

        matches: list[tuple[int, dict[str, object]]] = []
        for pid in GLOVE_USB_PIDS:
            for row in _port_rows(sdk.available_usb_ports(BRAINCO_USB_VID, pid)):
                matches.append((pid, row))
        by_port: dict[str, tuple[int, dict[str, object]]] = {}
        for pid, row in matches:
            by_port.setdefault(str(row["port_name"]), (pid, row))
        if self.config.port_name is not None:
            if self.config.port_name not in by_port:
                raise RuntimeError(
                    f"configured glove port {self.config.port_name!r} was not discovered"
                )
            selected = self.config.port_name
        elif len(by_port) == 1:
            selected = next(iter(by_port))
        elif not by_port:
            raise RuntimeError("no BrainCo EDU glove was discovered")
        else:
            raise RuntimeError("multiple gloves discovered; configure port_name explicitly")

        matched_pid, selected_row = by_port[selected]
        serial_value = selected_row.get("serial_number", selected_row.get("serial"))
        selected_serial = None if serial_value is None else str(serial_value)
        if (
            self.config.expected_serial is not None
            and selected_serial != self.config.expected_serial
        ):
            raise RuntimeError(
                "BrainCo EDU glove serial mismatch: "
                f"expected={self.config.expected_serial!r}, observed={selected_serial!r}"
            )
        report = BrainCoEduGloveDiscovery(
            sdk_import=BRAINCO_EDU_IMPORT,
            sdk_version=self._sdk_version,
            selected_port=selected,
            discovered_ports=tuple(sorted(by_port)),
            matched_pid=matched_pid,
            selected_usb_descriptor_hash=_stable_fingerprint(selected_row),
            selected_serial=selected_serial,
            baudrate=self.config.baudrate,
        )
        self._discovery = report
        return report

    def confirm_discovery(self, fingerprint: str) -> None:
        if self._discovery is None:
            raise RuntimeError("discover() must succeed before confirmation")
        if str(fingerprint) != self._discovery.fingerprint():
            raise ValueError("BrainCo EDU glove discovery fingerprint mismatch")
        self._expected_fingerprint = str(fingerprint)

    def _register_callback(
        self,
        name: str,
        callback: Callable[[Sequence[Sequence[object]]], None],
    ) -> None:
        if self._thread is not None:
            raise RuntimeError("register glove callbacks before starting the EDU stream")
        if not callable(callback):
            raise TypeError(f"{name} callback must be callable")
        setattr(self, f"_{name}_callback", callback)

    def register_flex_callback(
        self, callback: Callable[[Sequence[Sequence[object]]], None]
    ) -> None:
        self._register_callback("flex", callback)

    def register_imu_callback(
        self, callback: Callable[[Sequence[Sequence[object]]], None]
    ) -> None:
        self._register_callback("imu", callback)

    def register_mag_callback(
        self, callback: Callable[[Sequence[Sequence[object]]], None]
    ) -> None:
        self._register_callback("mag", callback)

    def _claim_global_callbacks(self) -> None:
        claim_brainco_edu_callback_namespace(self, owner_kind="glove")
        try:
            with self._global_owner_lock:
                if self._global_owner is not None and self._global_owner is not self:
                    raise RuntimeError(
                        "bc-edu-sdk glove callbacks are module-global; "
                        "use a separate process per glove"
                    )
                type(self)._global_owner = self
        except BaseException:
            release_brainco_edu_callback_namespace(self)
            raise

    def _release_global_callbacks(self) -> None:
        with self._global_owner_lock:
            if self._global_owner is self:
                type(self)._global_owner = None
        release_brainco_edu_callback_namespace(self)

    def start(self) -> None:
        if not self.config.allow_hardware_stream:
            raise PermissionError("BrainCo EDU glove hardware streaming is disabled")
        if self._discovery is None or self._expected_fingerprint is None:
            raise RuntimeError("confirmed BrainCo EDU glove discovery is required before streaming")
        if self._expected_fingerprint != self._discovery.fingerprint():
            raise RuntimeError("BrainCo EDU glove discovery changed after confirmation")
        if any(
            callback is None
            for callback in (self._flex_callback, self._imu_callback, self._mag_callback)
        ):
            raise RuntimeError("flex, IMU, and magnetometer callbacks are required before start")
        if self._thread is not None:
            raise RuntimeError("BrainCo EDU glove stream is already started")

        self._claim_global_callbacks()
        self._stop_event.clear()
        self._ready_event.clear()
        self._thread_failure = None
        self._thread = threading.Thread(
            target=self._thread_main,
            name="brainco-edu-glove",
            daemon=True,
        )
        self._thread.start()
        if not self._ready_event.wait(self._startup_timeout_s):
            self._stop_event.set()
            self._thread.join(timeout=self._startup_timeout_s)
            if self._thread.is_alive():
                raise RuntimeError(
                    "bc-edu-sdk glove startup thread is still alive after timeout; "
                    "global callback ownership remains locked and process/device "
                    "intervention is required"
                )
            self._thread = None
            self._release_global_callbacks()
            raise TimeoutError("bc-edu-sdk glove did not start within startup_timeout_s")
        if self._thread_failure is not None:
            failure = self._thread_failure
            self._thread.join(timeout=self._startup_timeout_s)
            if self._thread.is_alive():
                raise RuntimeError(
                    "bc-edu-sdk glove failed during startup but its thread is still alive; "
                    "global callback ownership remains locked and process/device "
                    "intervention is required"
                ) from failure
            self._thread = None
            self._release_global_callbacks()
            raise RuntimeError("bc-edu-sdk glove stream startup failed") from failure

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._run_stream())
        except BaseException as exc:  # surfaced by start()/stop()
            self._thread_failure = exc
            self._ready_event.set()
        finally:
            self._release_global_callbacks()

    @staticmethod
    def _message_response_callback(_device_id: object, _message: object) -> None:
        # The recorder intentionally does not treat command responses as sensor data.
        return None

    def _install_callbacks(self, sdk: Any) -> None:
        assert self._flex_callback is not None
        assert self._imu_callback is not None
        assert self._mag_callback is not None
        sdk.set_msg_resp_callback(self._message_response_callback)
        sdk.set_flex_data_callback(self._flex_callback)
        # The pinned official example registers both raw and calibrated setter
        # names while requesting CALIBRATED_DATA in SensorProfile.
        sdk.set_imu_data_callback(self._imu_callback)
        sdk.set_imu_calibration_data_callback(self._imu_callback)
        sdk.set_mag_data_callback(self._mag_callback)
        sdk.set_mag_calibration_data_callback(self._mag_callback)

    @staticmethod
    def _clear_callbacks(sdk: Any) -> None:
        failures: list[BaseException] = []
        for setter_name in (
            "set_flex_data_callback",
            "set_imu_data_callback",
            "set_imu_calibration_data_callback",
            "set_mag_data_callback",
            "set_mag_calibration_data_callback",
            "set_msg_resp_callback",
        ):
            try:
                getattr(sdk, setter_name)(None)
            except BaseException as exc:
                failures.append(exc)
        if failures:
            raise RuntimeError("failed to clear one or more bc-edu-sdk glove callbacks") from failures[0]

    async def _run_stream(self) -> None:
        assert self._discovery is not None
        sdk = self._module()
        device: Any | None = None
        cleanup_failures: list[BaseException] = []
        try:
            self._install_callbacks(sdk)
            device = sdk.EduDevice(self._discovery.selected_port, self.config.baudrate)
            parser = sdk.MessageParser("Glove-device", sdk.MsgType.Edu)
            profile = sdk.SensorProfile(
                flex_rate=sdk.SamplingRate.SAMPLING_RATE_50,
                imu_rate=sdk.ImuSampleRate.IMU_SR_100,
                imu_data_type=sdk.UploadDataType.CALIBRATED_DATA,
                mag_rate=sdk.MagSampleRate.MAG_SR_20,
                mag_data_type=sdk.UploadDataType.CALIBRATED_DATA,
            )
            await device.start_stream(parser, profile)
            await device.get_dongle_pair_stat()
            await asyncio.sleep(0.1)
            self._ready_event.set()
            while not self._stop_event.is_set():
                await asyncio.sleep(0.02)
        finally:
            if device is not None:
                try:
                    await device.stop_stream()
                except BaseException as exc:
                    cleanup_failures.append(exc)
            try:
                self._clear_callbacks(sdk)
            except BaseException as exc:
                cleanup_failures.append(exc)
            if cleanup_failures:
                raise RuntimeError("bc-edu-sdk glove cleanup failed") from cleanup_failures[0]

    def stop(self) -> None:
        thread = self._thread
        if thread is None:
            return
        self._stop_event.set()
        thread.join(timeout=self._startup_timeout_s)
        if thread.is_alive():
            raise RuntimeError(
                "bc-edu-sdk glove stream thread did not stop; the thread handle and "
                "global callback ownership remain locked, so process/device "
                "intervention is required"
            )
        self._thread = None
        failure, self._thread_failure = self._thread_failure, None
        if failure is not None:
            raise RuntimeError("bc-edu-sdk glove stream failed") from failure


__all__ = [
    "BRAINCO_EDU_IMPORT",
    "BRAINCO_USB_VID",
    "GLOVE_USB_PIDS",
    "BrainCoEduGloveConfig",
    "BrainCoEduGloveDiscovery",
    "BrainCoEduSdkGloveClient",
]
