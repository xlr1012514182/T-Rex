from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from revo3_teleop.sources.brainco_glove import BrainCoGloveParser, BrainCoGloveSource


class FakeGloveClient:
    def __init__(self) -> None:
        self.callbacks = {}
        self.start_count = 0
        self.stop_count = 0

    def register_flex_callback(self, callback) -> None:
        self.callbacks["flex"] = callback

    def register_imu_callback(self, callback) -> None:
        self.callbacks["imu"] = callback

    def register_mag_callback(self, callback) -> None:
        self.callbacks["mag"] = callback

    def start(self) -> None:
        self.start_count += 1

    def stop(self) -> None:
        self.stop_count += 1


def test_raw_flex_imu_mag_are_preserved_without_claiming_wrist_pose() -> None:
    parser = BrainCoGloveParser()
    flex = parser.parse_flex([1, 10, 20, 30, 40, 50, 60], callback_timestamp_ns=1_000_000_000)
    compact_imu = parser.parse_imu(
        [1, 0.1, 0.2, 0.3, 1.1, 1.2, 1.3], callback_timestamp_ns=1_010_000_000
    )
    extended_values = [2, 0.4, 0.5, 0.6, 91, 92, 93, 1.4, 1.5, 1.6, 94, 95, 96]
    extended_imu = parser.parse_imu(extended_values, callback_timestamp_ns=1_020_000_000)
    mag = parser.parse_mag([1, 3.1, 3.2, 3.3], callback_timestamp_ns=1_030_000_000)

    np.testing.assert_array_equal(flex.payload["flex_raw"], [10, 20, 30, 40, 50, 60])
    np.testing.assert_array_equal(flex.payload["raw_packet"], [1, 10, 20, 30, 40, 50, 60])
    np.testing.assert_allclose(compact_imu.payload["acc_raw"], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(compact_imu.payload["gyro_raw"], [1.1, 1.2, 1.3])
    np.testing.assert_allclose(extended_imu.payload["gyro_raw"], [1.4, 1.5, 1.6])
    np.testing.assert_allclose(extended_imu.payload["imu_raw"], extended_values)
    np.testing.assert_allclose(mag.payload["mag_raw"], [3.1, 3.2, 3.3])

    for sample, rate in ((flex, 50.0), (compact_imu, 100.0), (mag, 20.0)):
        assert sample.payload["sample_rate_hz"].item() == rate
        assert sample.payload["provides_wrist_pose"].item() == 0
        assert sample.payload["device_timestamp_available"].item() == 0
        assert sample.header.device_timestamp_ns is None
    assert parser.provides_wrist_pose is False


def test_glove_parser_rejects_ambiguous_lengths_and_tracks_drops() -> None:
    parser = BrainCoGloveParser()
    parser.parse_flex([5, 1, 2, 3, 4, 5, 6], callback_timestamp_ns=2_000_000_000)
    jumped = parser.parse_flex([8, 1, 2, 3, 4, 5, 6], callback_timestamp_ns=2_100_000_000)
    assert jumped.header.dropped_since_previous == 2

    with pytest.raises(ValueError, match="exactly 7"):
        BrainCoGloveParser().parse_flex([1, 2, 3], callback_timestamp_ns=2_000_000_000)
    with pytest.raises(ValueError, match="7 or at least 13"):
        BrainCoGloveParser().parse_imu(list(range(10)), callback_timestamp_ns=2_000_000_000)
    with pytest.raises(ValueError, match="exactly 4"):
        BrainCoGloveParser().parse_mag([1, 2, 3], callback_timestamp_ns=2_000_000_000)


def test_glove_batch_times_are_reconstructed_per_native_rate() -> None:
    source = BrainCoGloveSource()
    samples = source.ingest_flex(
        [[1, 1, 2, 3, 4, 5, 6], [2, 7, 8, 9, 10, 11, 12]],
        callback_timestamp_ns=3_000_000_000,
    )
    assert samples[0].header.capture_timestamp_ns == 2_980_000_000
    assert samples[1].header.capture_timestamp_ns == 3_000_000_000
    assert samples[0].header.receive_timestamp_ns == 3_000_000_000
    assert samples[0].payload["host_callback_timestamp_ns"].item() == 3_000_000_000
    assert samples[0].payload["callback_to_capture_latency_ns"].item() == 20_000_000


def test_glove_client_is_lazy_opt_in_and_fake_callbacks_work() -> None:
    fake = FakeGloveClient()
    factory_calls = 0

    def factory() -> FakeGloveClient:
        nonlocal factory_calls
        factory_calls += 1
        return fake

    disabled = BrainCoGloveSource(client_factory=factory)
    with pytest.raises(PermissionError, match="disabled"):
        disabled.start()
    assert factory_calls == 0

    enabled = BrainCoGloveSource(
        client_factory=factory,
        allow_hardware_start=True,
        clock=lambda: 4_000_000_000,
    )
    enabled.start()
    assert factory_calls == 1
    assert fake.start_count == 1
    assert set(fake.callbacks) == {"flex", "imu", "mag"}
    fake.callbacks["flex"]([[10, 1, 2, 3, 4, 5, 6]])
    fake.callbacks["imu"]([[10, 0, 1, 2, 3, 4, 5]])
    fake.callbacks["mag"]([[10, 6, 7, 8]])
    assert [sample.header.source_id for sample in enabled.drain()] == [
        "brainco_glove_flex",
        "brainco_glove_imu",
        "brainco_glove_mag",
    ]
    enabled.stop()
    assert fake.stop_count == 1
