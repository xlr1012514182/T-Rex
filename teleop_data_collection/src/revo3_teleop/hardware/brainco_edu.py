"""Concrete, fail-closed ``bc-edu-sdk`` EMG armband lifecycle.

The pinned BrainCo example imports ``bc_edu_sdk.main_mod``, discovers USB
VID 21059 with armband PIDs 1/5, configures eight EMG channels at 250 Hz, and
registers a module-global callback.  This adapter follows that public API but
keeps import, discovery, and streaming behind three separate boundaries.

The public callback contains a packet sequence and samples, but no acquisition
timestamp.  Timestamp reconstruction therefore remains in
``BrainCoEduEMGSource`` and is explicitly labelled host reconstruction.
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
ARMBAND_USB_PIDS = (1, 5)


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
            raise ValueError("bc-edu-sdk returned a port row without port_name")
        result.append({str(key): value for key, value in item.items()})
    return result


@dataclass(frozen=True)
class BrainCoEduArmbandConfig:
    """Static stream configuration; construction has no hardware side effect."""

    port_name: str | None = None
    baudrate: int = 115_200
    emg_buffer_length: int = 1_250
    expected_sdk_version: str | None = "0.5.0"
    allow_hardware_discovery: bool = False
    allow_hardware_stream: bool = False

    def __post_init__(self) -> None:
        if self.port_name is not None and not self.port_name.strip():
            raise ValueError("port_name must be non-empty when supplied")
        if self.baudrate <= 0:
            raise ValueError("baudrate must be positive")
        if self.emg_buffer_length < 20:
            raise ValueError("emg_buffer_length must hold at least one 20-sample packet")
        if self.expected_sdk_version is not None and not self.expected_sdk_version.strip():
            raise ValueError("expected_sdk_version must be non-empty when supplied")


@dataclass(frozen=True)
class BrainCoEduArmbandDiscovery:
    sdk_import: str
    sdk_version: str
    selected_port: str
    discovered_ports: tuple[str, ...]
    matched_pid: int
    selected_usb_descriptor_hash: str
    selected_serial: str | None
    baudrate: int
    emg_channels: int = 8
    emg_sample_rate_hz: int = 250
    samples_per_packet: int = 20
    device_timestamp_available: bool = False
    callback_scope: str = "module_global"

    def fingerprint(self) -> str:
        return _stable_fingerprint(asdict(self))

    def to_json(self) -> dict[str, object]:
        result = asdict(self)
        result["fingerprint"] = self.fingerprint()
        return result


class BrainCoEduSdkEMGClient:
    """Thread-hosted official SDK client implementing ``BrainCoEduEMGClient``.

    The source calls the synchronous ``start``/``stop`` methods while the
    official device lifecycle is asynchronous.  A private event-loop thread
    bridges that mismatch.  Only one instance may own the SDK's module-global
    EMG callback in a process.
    """

    _global_owner_lock = threading.Lock()
    _global_owner: "BrainCoEduSdkEMGClient | None" = None
    callback_namespace = BRAINCO_EDU_CALLBACK_NAMESPACE

    def __init__(
        self,
        config: BrainCoEduArmbandConfig,
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
        self._callback: Callable[[Sequence[Sequence[object]]], None] | None = None
        self._discovery: BrainCoEduArmbandDiscovery | None = None
        self._expected_fingerprint: str | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread_failure: BaseException | None = None

    def _module(self) -> Any:
        if self._sdk is None:
            self._sdk = import_module(BRAINCO_EDU_IMPORT)
        return self._sdk

    def discover(self) -> BrainCoEduArmbandDiscovery:
        if not self.config.allow_hardware_discovery:
            raise PermissionError("BrainCo EDU hardware discovery is disabled")
        if (
            self.config.expected_sdk_version is not None
            and self._sdk_version != self.config.expected_sdk_version
        ):
            raise RuntimeError(
                "bc-edu-sdk version mismatch: "
                f"expected={self.config.expected_sdk_version}, observed={self._sdk_version}"
            )
        sdk = self._module()
        required = (
            "available_usb_ports",
            "EduDevice",
            "MessageParser",
            "SensorProfile",
            "set_emg_buffer_cfg",
            "set_emg_data_callback",
        )
        missing = [name for name in required if not callable(getattr(sdk, name, None))]
        for enum_name in (
            "MsgType",
            "ImuSampleRate",
            "AfeSampleRate",
            "MagSampleRate",
            "UploadDataType",
        ):
            if not hasattr(sdk, enum_name):
                missing.append(enum_name)
        if missing:
            raise RuntimeError("bc-edu-sdk API mismatch: missing " + ", ".join(missing))

        matches: list[tuple[int, dict[str, object]]] = []
        for pid in ARMBAND_USB_PIDS:
            for row in _port_rows(sdk.available_usb_ports(BRAINCO_USB_VID, pid)):
                matches.append((pid, row))
        by_port: dict[str, tuple[int, dict[str, object]]] = {}
        for pid, row in matches:
            by_port.setdefault(str(row["port_name"]), (pid, row))
        if self.config.port_name is not None:
            if self.config.port_name not in by_port:
                raise RuntimeError(
                    f"configured armband port {self.config.port_name!r} was not discovered"
                )
            selected = self.config.port_name
        elif len(by_port) == 1:
            selected = next(iter(by_port))
        elif not by_port:
            raise RuntimeError("no BrainCo EDU armband was discovered")
        else:
            raise RuntimeError("multiple armbands discovered; configure port_name explicitly")
        matched_pid = by_port[selected][0]
        selected_row = by_port[selected][1]
        serial_value = selected_row.get("serial_number", selected_row.get("serial"))
        report = BrainCoEduArmbandDiscovery(
            sdk_import=BRAINCO_EDU_IMPORT,
            sdk_version=self._sdk_version,
            selected_port=selected,
            discovered_ports=tuple(sorted(by_port)),
            matched_pid=matched_pid,
            selected_usb_descriptor_hash=_stable_fingerprint(selected_row),
            selected_serial=None if serial_value is None else str(serial_value),
            baudrate=self.config.baudrate,
        )
        self._discovery = report
        return report

    def confirm_discovery(self, fingerprint: str) -> None:
        if self._discovery is None:
            raise RuntimeError("discover() must succeed before confirmation")
        if str(fingerprint) != self._discovery.fingerprint():
            raise ValueError("BrainCo EDU discovery fingerprint mismatch")
        self._expected_fingerprint = str(fingerprint)

    def register_emg_callback(
        self, callback: Callable[[Sequence[Sequence[object]]], None]
    ) -> None:
        if self._thread is not None:
            raise RuntimeError("register callback before starting the EDU stream")
        if not callable(callback):
            raise TypeError("EMG callback must be callable")
        self._callback = callback

    def _claim_global_callback(self) -> None:
        claim_brainco_edu_callback_namespace(self, owner_kind="emg")
        try:
            with self._global_owner_lock:
                if self._global_owner is not None and self._global_owner is not self:
                    raise RuntimeError(
                        "bc-edu-sdk EMG callback is module-global; "
                        "use a separate process per device"
                    )
                type(self)._global_owner = self
        except BaseException:
            release_brainco_edu_callback_namespace(self)
            raise

    def _release_global_callback(self) -> None:
        with self._global_owner_lock:
            if self._global_owner is self:
                type(self)._global_owner = None
        release_brainco_edu_callback_namespace(self)

    def start(self) -> None:
        if not self.config.allow_hardware_stream:
            raise PermissionError("BrainCo EDU hardware streaming is disabled")
        if self._discovery is None or self._expected_fingerprint is None:
            raise RuntimeError("confirmed BrainCo EDU discovery is required before streaming")
        if self._expected_fingerprint != self._discovery.fingerprint():
            raise RuntimeError("BrainCo EDU discovery changed after confirmation")
        if self._callback is None:
            raise RuntimeError("register_emg_callback is required before start")
        if self._thread is not None:
            raise RuntimeError("BrainCo EDU stream is already started")
        self._claim_global_callback()
        self._stop_event.clear()
        self._ready_event.clear()
        self._thread_failure = None
        self._thread = threading.Thread(
            target=self._thread_main,
            name="brainco-edu-emg",
            daemon=True,
        )
        self._thread.start()
        if not self._ready_event.wait(self._startup_timeout_s):
            self._stop_event.set()
            self._thread.join(timeout=self._startup_timeout_s)
            if self._thread.is_alive():
                # The SDK may be blocked inside start_stream().  Keep both the
                # thread handle and the module-global callback ownership so a
                # second client cannot silently steal the callback while the
                # first SDK invocation is still alive.
                raise RuntimeError(
                    "bc-edu-sdk startup thread is still alive after timeout; "
                    "global callback ownership remains locked and process/device "
                    "intervention is required"
                )
            self._thread = None
            self._release_global_callback()
            raise TimeoutError("bc-edu-sdk did not start within startup_timeout_s")
        if self._thread_failure is not None:
            failure = self._thread_failure
            self._thread.join(timeout=self._startup_timeout_s)
            if self._thread.is_alive():
                raise RuntimeError(
                    "bc-edu-sdk failed during startup but its thread is still alive; "
                    "global callback ownership remains locked and process/device "
                    "intervention is required"
                ) from failure
            self._thread = None
            self._release_global_callback()
            raise RuntimeError("bc-edu-sdk stream startup failed") from failure

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._run_stream())
        except BaseException as exc:  # propagated synchronously by start/stop
            self._thread_failure = exc
            self._ready_event.set()
        finally:
            self._release_global_callback()

    async def _run_stream(self) -> None:
        assert self._discovery is not None
        assert self._callback is not None
        sdk = self._module()
        device: Any | None = None
        try:
            sdk.set_emg_buffer_cfg(self.config.emg_buffer_length)
            sdk.set_emg_data_callback(self._callback)
            device = sdk.EduDevice(self._discovery.selected_port, self.config.baudrate)
            parser = sdk.MessageParser("ARMBAND-device", sdk.MsgType.Edu)
            profile = sdk.SensorProfile(
                flex_rate=None,
                imu_rate=sdk.ImuSampleRate.IMU_SR_100,
                imu_data_type=sdk.UploadDataType.CALIBRATED_DATA,
                emg_rate=sdk.AfeSampleRate.AFE_SR_250,
                emg_channel_bits=0xFF,
                mag_rate=sdk.MagSampleRate.MAG_SR_100,
                mag_data_type=sdk.UploadDataType.CALIBRATED_DATA,
            )
            await device.start_stream(parser, profile)
            self._ready_event.set()
            while not self._stop_event.is_set():
                await asyncio.sleep(0.02)
        finally:
            try:
                if device is not None:
                    await device.stop_stream()
            finally:
                sdk.set_emg_data_callback(None)

    def stop(self) -> None:
        thread = self._thread
        if thread is None:
            return
        self._stop_event.set()
        thread.join(timeout=self._startup_timeout_s)
        if thread.is_alive():
            raise RuntimeError(
                "bc-edu-sdk stream thread did not stop; the thread handle and "
                "global callback ownership remain locked, so process/device "
                "intervention is required"
            )
        self._thread = None
        failure, self._thread_failure = self._thread_failure, None
        if failure is not None:
            raise RuntimeError("bc-edu-sdk stream failed") from failure


__all__ = [
    "ARMBAND_USB_PIDS",
    "BRAINCO_EDU_IMPORT",
    "BRAINCO_USB_VID",
    "BrainCoEduArmbandConfig",
    "BrainCoEduArmbandDiscovery",
    "BrainCoEduSdkEMGClient",
]
