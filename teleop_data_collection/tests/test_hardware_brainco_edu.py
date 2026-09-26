from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from revo3_teleop.hardware.brainco_edu import (
    BRAINCO_USB_VID,
    BrainCoEduArmbandConfig,
    BrainCoEduSdkEMGClient,
)
from revo3_teleop.sources.brainco_emg import BrainCoEduEMGSource


class _FakeEduDevice:
    instances: list["_FakeEduDevice"] = []

    def __init__(self, port: str, baudrate: int) -> None:
        self.port = port
        self.baudrate = baudrate
        self.profile = None
        self.started = 0
        self.stopped = 0
        self.instances.append(self)

    async def start_stream(self, parser, profile) -> None:
        self.parser = parser
        self.profile = profile
        self.started += 1

    async def stop_stream(self) -> None:
        self.stopped += 1


class _FakeEduSdk:
    MsgType = SimpleNamespace(Edu=7)
    ImuSampleRate = SimpleNamespace(IMU_SR_100=100)
    AfeSampleRate = SimpleNamespace(AFE_SR_250=250)
    MagSampleRate = SimpleNamespace(MAG_SR_100=100)
    UploadDataType = SimpleNamespace(CALIBRATED_DATA=1)
    EduDevice = _FakeEduDevice

    def __init__(self, ports: dict[int, list[dict[str, object]]]) -> None:
        self.ports = ports
        self.callback = None
        self.buffer_cfg = None
        self.profile_kwargs = None

    def available_usb_ports(self, vid: int, pid: int) -> bytes:
        assert vid == BRAINCO_USB_VID
        return json.dumps(self.ports.get(pid, [])).encode()

    def MessageParser(self, name: str, kind: int):
        return (name, kind)

    def SensorProfile(self, **kwargs):
        self.profile_kwargs = kwargs
        return kwargs

    def set_emg_buffer_cfg(self, length: int) -> None:
        self.buffer_cfg = length

    def set_emg_data_callback(self, callback) -> None:
        self.callback = callback


class _BlockingEduDevice(_FakeEduDevice):
    def __init__(self, port: str, baudrate: int, release: threading.Event) -> None:
        super().__init__(port, baudrate)
        self.release = release

    async def start_stream(self, parser, profile) -> None:
        self.parser = parser
        self.profile = profile
        self.started += 1
        await asyncio.to_thread(self.release.wait)


class _BlockingEduSdk(_FakeEduSdk):
    def __init__(self, release: threading.Event) -> None:
        super().__init__({1: [{"port_name": "COM9", "serial": "A"}], 5: []})
        self.release = release

    def EduDevice(self, port: str, baudrate: int):
        return _BlockingEduDevice(port, baudrate, self.release)


class _BlockingStopEduDevice(_FakeEduDevice):
    def __init__(self, port: str, baudrate: int, release: threading.Event) -> None:
        super().__init__(port, baudrate)
        self.release = release

    async def stop_stream(self) -> None:
        self.stopped += 1
        await asyncio.to_thread(self.release.wait)


class _BlockingStopEduSdk(_FakeEduSdk):
    def __init__(self, release: threading.Event) -> None:
        super().__init__({1: [{"port_name": "COM9", "serial": "A"}], 5: []})
        self.release = release

    def EduDevice(self, port: str, baudrate: int):
        return _BlockingStopEduDevice(port, baudrate, self.release)


def _row(sequence: int) -> list[float]:
    return [sequence, 0, *np.arange(160, dtype=np.float32).tolist()]


def test_official_edu_client_requires_discovery_confirmation_and_stream_opt_in() -> None:
    sdk = _FakeEduSdk({1: [{"port_name": "COM9", "serial": "A"}], 5: []})
    disabled = BrainCoEduSdkEMGClient(
        BrainCoEduArmbandConfig(expected_sdk_version="fake"),
        sdk_module=sdk,
        sdk_version="fake",
    )
    with pytest.raises(PermissionError, match="discovery is disabled"):
        disabled.discover()

    client = BrainCoEduSdkEMGClient(
        BrainCoEduArmbandConfig(
            port_name="COM9",
            expected_sdk_version="fake-1.0",
            allow_hardware_discovery=True,
            allow_hardware_stream=True,
        ),
        sdk_module=sdk,
        sdk_version="fake-1.0",
    )
    report = client.discover()
    assert report.selected_port == "COM9"
    assert report.selected_serial == "A"
    assert len(report.selected_usb_descriptor_hash) == 64
    assert report.emg_channels == 8
    assert report.emg_sample_rate_hz == 250
    assert not report.device_timestamp_available
    with pytest.raises(RuntimeError, match="confirmed"):
        client.register_emg_callback(lambda rows: None)
        client.start()

    client.confirm_discovery(report.fingerprint())
    source = BrainCoEduEMGSource(
        client_factory=lambda: client,
        allow_hardware_start=True,
        clock=lambda: 5_000_000_000,
    )
    source.start()
    assert sdk.buffer_cfg == 1_250
    assert sdk.profile_kwargs["emg_rate"] == 250
    assert sdk.profile_kwargs["emg_channel_bits"] == 0xFF
    assert sdk.callback is not None
    sdk.callback([_row(3)])
    samples = source.drain()
    assert len(samples) == 1
    assert samples[0].payload["signal"].shape == (8, 20)
    assert samples[0].header.device_timestamp_ns is None
    assert samples[0].payload["clock_is_host_reconstruction"].item() == 1
    source.stop()
    assert sdk.callback is None
    assert _FakeEduDevice.instances[-1].stopped == 1


def test_edu_discovery_rejects_ambiguous_devices_and_bad_fingerprint() -> None:
    sdk = _FakeEduSdk(
        {1: [{"port_name": "COM1"}], 5: [{"port_name": "COM2"}]}
    )
    client = BrainCoEduSdkEMGClient(
        BrainCoEduArmbandConfig(
            expected_sdk_version="fake", allow_hardware_discovery=True
        ),
        sdk_module=sdk,
        sdk_version="fake",
    )
    with pytest.raises(RuntimeError, match="multiple armbands"):
        client.discover()

    selected = BrainCoEduSdkEMGClient(
        BrainCoEduArmbandConfig(
            port_name="COM2",
            expected_sdk_version="fake",
            allow_hardware_discovery=True,
        ),
        sdk_module=sdk,
        sdk_version="fake",
    )
    selected.discover()
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        selected.confirm_discovery("not-the-probe")


def test_stuck_start_keeps_global_callback_ownership_until_thread_exits() -> None:
    release = threading.Event()
    sdk = _BlockingEduSdk(release)

    def prepared_client() -> BrainCoEduSdkEMGClient:
        client = BrainCoEduSdkEMGClient(
            BrainCoEduArmbandConfig(
                port_name="COM9",
                expected_sdk_version="fake",
                allow_hardware_discovery=True,
                allow_hardware_stream=True,
            ),
            sdk_module=sdk,
            sdk_version="fake",
            startup_timeout_s=0.02,
        )
        report = client.discover()
        client.confirm_discovery(report.fingerprint())
        client.register_emg_callback(lambda rows: None)
        return client

    first = prepared_client()
    second = prepared_client()
    try:
        with pytest.raises(RuntimeError, match="still alive after timeout"):
            first.start()
        assert first._thread is not None and first._thread.is_alive()
        assert BrainCoEduSdkEMGClient._global_owner is first

        with pytest.raises(RuntimeError, match="module-global"):
            second.start()
    finally:
        release.set()
        first.stop()

    assert first._thread is None
    assert BrainCoEduSdkEMGClient._global_owner is None


def test_stuck_stop_keeps_global_callback_ownership_until_thread_exits() -> None:
    release = threading.Event()
    sdk = _BlockingStopEduSdk(release)

    def prepared_client() -> BrainCoEduSdkEMGClient:
        client = BrainCoEduSdkEMGClient(
            BrainCoEduArmbandConfig(
                port_name="COM9",
                expected_sdk_version="fake",
                allow_hardware_discovery=True,
                allow_hardware_stream=True,
            ),
            sdk_module=sdk,
            sdk_version="fake",
            startup_timeout_s=0.02,
        )
        report = client.discover()
        client.confirm_discovery(report.fingerprint())
        client.register_emg_callback(lambda rows: None)
        return client

    first = prepared_client()
    second = prepared_client()
    first.start()
    try:
        with pytest.raises(RuntimeError, match="thread did not stop"):
            first.stop()
        assert first._thread is not None and first._thread.is_alive()
        assert BrainCoEduSdkEMGClient._global_owner is first

        with pytest.raises(RuntimeError, match="module-global"):
            second.start()
    finally:
        release.set()
        first.stop()

    assert first._thread is None
    assert BrainCoEduSdkEMGClient._global_owner is None
