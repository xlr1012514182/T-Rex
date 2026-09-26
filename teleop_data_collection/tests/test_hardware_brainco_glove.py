from __future__ import annotations

import asyncio
import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from revo3_teleop.cli import hardware_probe
from revo3_teleop.hardware.brainco_glove import (
    BRAINCO_USB_VID,
    BrainCoEduGloveConfig,
    BrainCoEduSdkGloveClient,
)
from revo3_teleop.hardware.brainco_edu import (
    BrainCoEduArmbandConfig,
    BrainCoEduSdkEMGClient,
)
from revo3_teleop.hardware.brainco_edu_callbacks import (
    brainco_edu_callback_namespace_status,
)
from revo3_teleop.sources.brainco_glove import BrainCoGloveSource


class _FakeGloveDevice:
    instances: list["_FakeGloveDevice"] = []

    def __init__(self, port: str, baudrate: int) -> None:
        self.port = port
        self.baudrate = baudrate
        self.parser = None
        self.profile = None
        self.started = 0
        self.stopped = 0
        self.pair_stat_calls = 0
        self.instances.append(self)

    async def start_stream(self, parser, profile) -> None:
        self.parser = parser
        self.profile = profile
        self.started += 1

    async def get_dongle_pair_stat(self) -> None:
        self.pair_stat_calls += 1

    async def stop_stream(self) -> None:
        self.stopped += 1


class _FakeGloveSdk:
    MsgType = SimpleNamespace(Edu="edu")
    SamplingRate = SimpleNamespace(SAMPLING_RATE_50="flex-50")
    ImuSampleRate = SimpleNamespace(IMU_SR_100="imu-100")
    MagSampleRate = SimpleNamespace(MAG_SR_20="mag-20")
    UploadDataType = SimpleNamespace(CALIBRATED_DATA="calibrated")

    def __init__(self, ports: dict[int, list[dict[str, object]]]) -> None:
        self.ports = ports
        self.available_calls: list[tuple[int, int]] = []
        self.callbacks: dict[str, object] = {}
        self.profile_kwargs = None
        self.devices: list[_FakeGloveDevice] = []

    def available_usb_ports(self, vid: int, pid: int) -> bytes:
        assert vid == BRAINCO_USB_VID
        self.available_calls.append((vid, pid))
        return json.dumps(self.ports.get(pid, [])).encode("utf-8")

    def EduDevice(self, port: str, baudrate: int):
        device = _FakeGloveDevice(port, baudrate)
        self.devices.append(device)
        return device

    def MessageParser(self, name: str, kind: object):
        return (name, kind)

    def SensorProfile(self, **kwargs):
        self.profile_kwargs = kwargs
        return kwargs

    def _set(self, name: str, callback) -> None:
        self.callbacks[name] = callback

    def set_msg_resp_callback(self, callback) -> None:
        self._set("message", callback)

    def set_flex_data_callback(self, callback) -> None:
        self._set("flex", callback)

    def set_imu_data_callback(self, callback) -> None:
        self._set("imu", callback)

    def set_imu_calibration_data_callback(self, callback) -> None:
        self._set("imu_calibrated", callback)

    def set_mag_data_callback(self, callback) -> None:
        self._set("mag", callback)

    def set_mag_calibration_data_callback(self, callback) -> None:
        self._set("mag_calibrated", callback)


class _BlockingStartDevice(_FakeGloveDevice):
    def __init__(self, port: str, baudrate: int, release: threading.Event) -> None:
        super().__init__(port, baudrate)
        self.release = release

    async def start_stream(self, parser, profile) -> None:
        self.parser = parser
        self.profile = profile
        self.started += 1
        await asyncio.to_thread(self.release.wait)


class _BlockingStartSdk(_FakeGloveSdk):
    def __init__(self, release: threading.Event) -> None:
        super().__init__({6: [{"port_name": "COM6", "serial": "GLOVE-A"}], 2: []})
        self.release = release

    def EduDevice(self, port: str, baudrate: int):
        device = _BlockingStartDevice(port, baudrate, self.release)
        self.devices.append(device)
        return device


class _BlockingStopDevice(_FakeGloveDevice):
    def __init__(self, port: str, baudrate: int, release: threading.Event) -> None:
        super().__init__(port, baudrate)
        self.release = release

    async def stop_stream(self) -> None:
        self.stopped += 1
        await asyncio.to_thread(self.release.wait)


class _BlockingStopSdk(_FakeGloveSdk):
    def __init__(self, release: threading.Event) -> None:
        super().__init__({6: [{"port_name": "COM6", "serial": "GLOVE-A"}], 2: []})
        self.release = release

    def EduDevice(self, port: str, baudrate: int):
        device = _BlockingStopDevice(port, baudrate, self.release)
        self.devices.append(device)
        return device


def _prepared(
    sdk: _FakeGloveSdk,
    *,
    startup_timeout_s: float = 0.25,
) -> BrainCoEduSdkGloveClient:
    client = BrainCoEduSdkGloveClient(
        BrainCoEduGloveConfig(
            port_name="COM6",
            expected_serial="GLOVE-A",
            expected_sdk_version="fake-0.5.0",
            allow_hardware_discovery=True,
            allow_hardware_stream=True,
        ),
        sdk_module=sdk,
        sdk_version="fake-0.5.0",
        startup_timeout_s=startup_timeout_s,
    )
    report = client.discover()
    client.confirm_discovery(report.fingerprint())
    client.register_flex_callback(lambda rows: None)
    client.register_imu_callback(lambda rows: None)
    client.register_mag_callback(lambda rows: None)
    return client


def test_concrete_glove_client_is_lazy_and_matches_pinned_official_profile() -> None:
    sdk = _FakeGloveSdk(
        {6: [{"port_name": "COM6", "serial": "GLOVE-A", "location": "usb-1"}], 2: []}
    )
    disabled = BrainCoEduSdkGloveClient(
        BrainCoEduGloveConfig(expected_sdk_version="fake-0.5.0"),
        sdk_module=sdk,
        sdk_version="fake-0.5.0",
    )
    assert sdk.available_calls == []
    with pytest.raises(PermissionError, match="discovery is disabled"):
        disabled.discover()
    assert sdk.available_calls == []

    client = _prepared(sdk)
    report = client._discovery
    assert report is not None
    assert report.selected_port == "COM6"
    assert report.selected_serial == "GLOVE-A"
    assert report.matched_pid == 6
    assert len(report.selected_usb_descriptor_hash) == 64
    assert (report.flex_channels, report.flex_sample_rate_hz) == (6, 50)
    assert (report.imu_sample_rate_hz, report.imu_upload_mode) == (100, "calibrated")
    assert (report.mag_sample_rate_hz, report.mag_upload_mode) == (20, "calibrated")
    assert not report.device_timestamp_available
    assert not report.provides_wrist_pose
    assert not report.produces_revo_target

    source = BrainCoGloveSource(
        client_factory=lambda: client,
        allow_hardware_start=True,
        clock=lambda: 5_000_000_000,
    )
    source.start()
    assert sdk.profile_kwargs == {
        "flex_rate": "flex-50",
        "imu_rate": "imu-100",
        "imu_data_type": "calibrated",
        "mag_rate": "mag-20",
        "mag_data_type": "calibrated",
    }
    assert sdk.devices[-1].parser == ("Glove-device", "edu")
    assert sdk.devices[-1].pair_stat_calls == 1
    assert sdk.callbacks["imu"] is sdk.callbacks["imu_calibrated"]
    assert sdk.callbacks["mag"] is sdk.callbacks["mag_calibrated"]

    sdk.callbacks["flex"]([[3, 1, 2, 3, 4, 5, 6]])
    sdk.callbacks["imu_calibrated"]([[4, 0.1, 0.2, 0.3, 1.1, 1.2, 1.3]])
    sdk.callbacks["mag_calibrated"]([[5, 0.4, 0.5, 0.6]])
    samples = source.drain()
    assert [sample.header.source_id for sample in samples] == [
        "brainco_glove_flex",
        "brainco_glove_imu",
        "brainco_glove_mag",
    ]
    np.testing.assert_array_equal(samples[0].payload["flex_raw"], [1, 2, 3, 4, 5, 6])
    assert all(sample.payload["provides_wrist_pose"].item() == 0 for sample in samples)
    source.stop()
    assert sdk.devices[-1].stopped == 1
    assert set(sdk.callbacks) == {
        "message", "flex", "imu", "imu_calibrated", "mag", "mag_calibrated"
    }
    assert all(callback is None for callback in sdk.callbacks.values())


def test_glove_discovery_rejects_ambiguity_serial_mismatch_and_unconfirmed_start() -> None:
    sdk = _FakeGloveSdk(
        {
            6: [{"port_name": "COM6", "serial": "GLOVE-A"}],
            2: [{"port_name": "COM2", "serial": "GLOVE-B"}],
        }
    )
    ambiguous = BrainCoEduSdkGloveClient(
        BrainCoEduGloveConfig(
            expected_sdk_version="fake", allow_hardware_discovery=True
        ),
        sdk_module=sdk,
        sdk_version="fake",
    )
    with pytest.raises(RuntimeError, match="multiple gloves"):
        ambiguous.discover()

    mismatch = BrainCoEduSdkGloveClient(
        BrainCoEduGloveConfig(
            port_name="COM6",
            expected_serial="WRONG",
            expected_sdk_version="fake",
            allow_hardware_discovery=True,
        ),
        sdk_module=sdk,
        sdk_version="fake",
    )
    with pytest.raises(RuntimeError, match="serial mismatch"):
        mismatch.discover()

    unconfirmed = BrainCoEduSdkGloveClient(
        BrainCoEduGloveConfig(
            port_name="COM6",
            expected_sdk_version="fake",
            allow_hardware_discovery=True,
            allow_hardware_stream=True,
        ),
        sdk_module=sdk,
        sdk_version="fake",
    )
    unconfirmed.discover()
    unconfirmed.register_flex_callback(lambda rows: None)
    unconfirmed.register_imu_callback(lambda rows: None)
    unconfirmed.register_mag_callback(lambda rows: None)
    with pytest.raises(RuntimeError, match="confirmed"):
        unconfirmed.start()


def test_stuck_glove_start_and_stop_retain_process_global_ownership() -> None:
    start_release = threading.Event()
    start_sdk = _BlockingStartSdk(start_release)
    first = _prepared(start_sdk, startup_timeout_s=0.02)
    second = _prepared(start_sdk, startup_timeout_s=0.02)
    try:
        with pytest.raises(RuntimeError, match="still alive after timeout"):
            first.start()
        assert first._thread is not None and first._thread.is_alive()
        assert BrainCoEduSdkGloveClient._global_owner is first
        with pytest.raises(RuntimeError, match="module-global"):
            second.start()
    finally:
        start_release.set()
        assert first._thread is not None
        first._thread.join(timeout=1.0)
        first.stop()
    assert BrainCoEduSdkGloveClient._global_owner is None

    stop_release = threading.Event()
    stop_sdk = _BlockingStopSdk(stop_release)
    first = _prepared(stop_sdk, startup_timeout_s=0.25)
    second = _prepared(stop_sdk, startup_timeout_s=0.25)
    first.start()
    try:
        with pytest.raises(RuntimeError, match="thread did not stop"):
            first.stop()
        assert first._thread is not None and first._thread.is_alive()
        assert BrainCoEduSdkGloveClient._global_owner is first
        with pytest.raises(RuntimeError, match="module-global"):
            second.start()
    finally:
        stop_release.set()
        assert first._thread is not None
        first._thread.join(timeout=1.0)
        first.stop()
    assert BrainCoEduSdkGloveClient._global_owner is None


def test_concrete_emg_and_glove_share_one_process_global_callback_guard() -> None:
    glove_sdk = _FakeGloveSdk(
        {6: [{"port_name": "COM6", "serial": "GLOVE-A"}], 2: []}
    )
    glove = _prepared(glove_sdk)
    # This check is deliberately at the concrete clients' ownership boundary,
    # before either fake SDK opens a stream.
    emg = BrainCoEduSdkEMGClient(
        BrainCoEduArmbandConfig(
            expected_sdk_version="fake",
            allow_hardware_discovery=True,
            allow_hardware_stream=True,
        ),
        sdk_module=object(),
        sdk_version="fake",
    )
    glove._claim_global_callbacks()
    try:
        status = brainco_edu_callback_namespace_status()
        assert status.busy and status.owner_kind == "glove"
        assert glove.callback_namespace == emg.callback_namespace == status.namespace
        with pytest.raises(RuntimeError, match="separate processes"):
            emg._claim_global_callback()
    finally:
        glove._release_global_callbacks()
    assert not brainco_edu_callback_namespace_status().busy


def test_glove_source_retains_failed_start_or_stop_client_until_confirmed_stop() -> None:
    class _FaultingClient:
        def __init__(self, *, fail_start: bool = False, fail_stop_once: bool = False) -> None:
            self.fail_start = fail_start
            self.fail_stop_once = fail_stop_once
            self.stop_calls = 0

        def register_flex_callback(self, callback) -> None:
            return None

        def register_imu_callback(self, callback) -> None:
            return None

        def register_mag_callback(self, callback) -> None:
            return None

        def start(self) -> None:
            if self.fail_start:
                raise RuntimeError("injected partial start failure")

        def stop(self) -> None:
            self.stop_calls += 1
            if self.fail_stop_once:
                self.fail_stop_once = False
                raise RuntimeError("injected stuck stop")

    failed_start = _FaultingClient(fail_start=True)
    source = BrainCoGloveSource(
        client_factory=lambda: failed_start,
        allow_hardware_start=True,
    )
    with pytest.raises(RuntimeError, match="partial start failure"):
        source.start()
    assert source._client is failed_start
    with pytest.raises(RuntimeError, match="already started"):
        source.start()
    source.stop()
    assert source._client is None

    failed_stop = _FaultingClient(fail_stop_once=True)
    source = BrainCoGloveSource(
        client_factory=lambda: failed_stop,
        allow_hardware_start=True,
    )
    source.start()
    with pytest.raises(RuntimeError, match="stuck stop"):
        source.stop()
    assert source._client is failed_stop
    with pytest.raises(RuntimeError, match="already started"):
        source.start()
    source.stop()
    assert failed_stop.stop_calls == 2
    assert source._client is None


def test_hardware_probe_wires_glove_as_read_only_component(monkeypatch) -> None:
    observed = {}

    class _Report:
        def to_json(self):
            return {"kind": "glove", "probe_only": True}

    class _ProbeClient:
        def __init__(self, config) -> None:
            observed["config"] = config

        def discover(self):
            return _Report()

    monkeypatch.setattr(hardware_probe, "BrainCoEduSdkGloveClient", _ProbeClient)
    manifest = asyncio.run(
        hardware_probe.run_probe(
            {
                "glove": {
                    "enabled": True,
                    "port_name": "COM6",
                    "expected_serial": "GLOVE-A",
                    "expected_sdk_version": "0.5.0",
                }
            }
        )
    )
    config = observed["config"]
    assert config.allow_hardware_discovery
    assert not config.allow_hardware_stream
    assert manifest["components"]["brainco_edu_glove"] == {
        "kind": "glove", "probe_only": True
    }
    assert manifest["hardware_write_performed"] is False
