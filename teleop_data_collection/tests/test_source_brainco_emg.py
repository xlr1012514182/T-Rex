from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from revo3_teleop.sources.brainco_emg import (
    EMG_PACKET_DURATION_NS,
    BrainCoEduEMGParser,
    BrainCoEduEMGSource,
)


def emg_row(sequence: int, lead_off: int = 0) -> list[float]:
    return [sequence, lead_off, *np.arange(160, dtype=np.float32).tolist()]


class FakeEMGClient:
    def __init__(self) -> None:
        self.callback = None
        self.start_count = 0
        self.stop_count = 0
        self.fail_start = False
        self.fail_stop_once = False

    def register_emg_callback(self, callback) -> None:
        self.callback = callback

    def start(self) -> None:
        self.start_count += 1
        if self.fail_start:
            raise RuntimeError("injected EMG start failure")

    def stop(self) -> None:
        self.stop_count += 1
        if self.fail_stop_once:
            self.fail_stop_once = False
            raise RuntimeError("injected EMG stop failure")

    def emit(self, rows) -> None:
        assert self.callback is not None
        self.callback(rows)


def test_strict_packet_shape_layout_clock_and_lead_off_evidence() -> None:
    parser = BrainCoEduEMGParser()
    callback_ns = 2_000_000_000
    sample = parser.parse_row(emg_row(10, lead_off=0b00000101), callback_timestamp_ns=callback_ns)

    assert sample.payload["signal"].shape == (8, 20)
    # Public EDU model documents channel-major 8 x 20 values.
    np.testing.assert_array_equal(sample.payload["signal"][0], np.arange(20))
    np.testing.assert_array_equal(sample.payload["signal"][1], np.arange(20, 40))
    assert sample.payload["sample_rate_hz"].item() == 250.0
    np.testing.assert_array_equal(
        sample.payload["lead_off_mask"], np.asarray([1, 0, 1, 0, 0, 0, 0, 0])
    )
    assert not sample.header.valid
    assert sample.header.device_timestamp_ns is None
    assert sample.payload["device_timestamp_available"].item() == 0
    assert sample.payload["clock_is_host_reconstruction"].item() == 1

    timestamps = sample.payload["sample_timestamp_ns"]
    assert timestamps.shape == (20,)
    assert timestamps[-1] == callback_ns
    np.testing.assert_array_equal(np.diff(timestamps), np.full(19, 4_000_000))
    assert sample.payload["host_callback_timestamp_ns"].item() == callback_ns


def test_sequence_gap_is_saved_and_ambiguous_packets_are_rejected() -> None:
    parser = BrainCoEduEMGParser()
    parser.parse_row(emg_row(4), callback_timestamp_ns=1_000_000_000)
    jumped = parser.parse_row(emg_row(7), callback_timestamp_ns=1_100_000_000)
    assert jumped.header.dropped_since_previous == 2
    assert jumped.payload["sequence_gap_packets"].item() == 2

    with pytest.raises(ValueError, match="exactly 162"):
        BrainCoEduEMGParser().parse_row(emg_row(1)[:-1], callback_timestamp_ns=1_000_000_000)
    with pytest.raises(ValueError, match="finite integer"):
        bad = emg_row(1)
        bad[0] = 1.5
        BrainCoEduEMGParser().parse_row(bad, callback_timestamp_ns=1_000_000_000)
    with pytest.raises(ValueError, match="<= 255"):
        BrainCoEduEMGParser().parse_row(emg_row(1, 256), callback_timestamp_ns=1_000_000_000)
    with pytest.raises(ValueError, match="increase strictly"):
        parser.parse_row(emg_row(7), callback_timestamp_ns=1_200_000_000)


def test_batch_callback_reconstructs_packet_ends_at_250_hz() -> None:
    source = BrainCoEduEMGSource()
    callback_ns = 3_000_000_000
    samples = source.ingest_rows([emg_row(1), emg_row(2)], callback_timestamp_ns=callback_ns)

    assert samples[1].header.capture_timestamp_ns == callback_ns
    assert samples[0].header.capture_timestamp_ns == callback_ns - EMG_PACKET_DURATION_NS
    assert samples[1].header.capture_timestamp_ns - samples[0].header.capture_timestamp_ns == 80_000_000
    assert samples[0].header.receive_timestamp_ns == callback_ns
    assert samples[0].payload["host_callback_timestamp_ns"].item() == callback_ns
    assert samples[0].payload["callback_to_packet_end_latency_ns"].item() == 80_000_000
    assert source.drain() == samples
    assert source.drain() == ()


def test_client_is_lazy_opt_in_and_fake_callback_is_recorded() -> None:
    fake = FakeEMGClient()
    factory_calls = 0

    def factory() -> FakeEMGClient:
        nonlocal factory_calls
        factory_calls += 1
        return fake

    disabled = BrainCoEduEMGSource(client_factory=factory)
    assert factory_calls == 0
    with pytest.raises(PermissionError, match="disabled"):
        disabled.start()
    assert factory_calls == 0

    enabled = BrainCoEduEMGSource(
        client_factory=factory,
        allow_hardware_start=True,
        clock=lambda: 5_000_000_000,
    )
    assert factory_calls == 0
    enabled.start()
    assert factory_calls == 1
    assert fake.start_count == 1
    fake.emit([emg_row(20)])
    captured = enabled.drain()
    assert len(captured) == 1
    assert captured[0].header.capture_timestamp_ns == 5_000_000_000
    enabled.stop()
    assert fake.stop_count == 1


def test_emg_source_retains_failed_start_and_stop_client_for_cleanup() -> None:
    failed_start = FakeEMGClient()
    failed_start.fail_start = True
    source = BrainCoEduEMGSource(
        client_factory=lambda: failed_start,
        allow_hardware_start=True,
    )
    with pytest.raises(RuntimeError, match="start failure"):
        source.start()
    assert source.started
    failed_start.fail_start = False
    source.stop()

    failed_stop = FakeEMGClient()
    failed_stop.fail_stop_once = True
    source = BrainCoEduEMGSource(
        client_factory=lambda: failed_stop,
        allow_hardware_start=True,
    )
    source.start()
    with pytest.raises(RuntimeError, match="stop failure"):
        source.stop()
    assert source.started
    source.stop()
    assert not source.started
    assert failed_stop.stop_count == 2
