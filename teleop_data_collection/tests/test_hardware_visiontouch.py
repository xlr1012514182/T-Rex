from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
import pytest

from revo3_teleop.hardware.visiontouch import (
    BRAINCO_REVO3_SDK_COMMIT,
    VITAI_SDK_COMMIT,
    VISIONTOUCH_AXIS_ORDER,
    VISIONTOUCH_FINGER_ORDER,
    VISIONTOUCH_MAX_RAW_RANK,
    VisionTouchForce6DConfig,
    VisionTouchForce6DSource,
)


class _DataType:
    FORCE6D_VECTOR = "force6d"


class _FakeFinder:
    def __init__(self, sdk) -> None:
        self.sdk = sdk

    def get_sns(self):
        return list(self.sdk.sns)

    def get_device_by_sn(self, serial):
        return None if serial not in self.sdk.sns else {"serial": serial}


class _FakeSensor:
    def __init__(self, sdk, *, config, force_model_path) -> None:
        self.sdk = sdk
        self.serial = config["serial"]
        self.force_model_path = force_model_path
        self.calibrated = 0
        self.released = 0
        sdk.sensors[self.serial] = self

    def calibrate(self):
        self.calibrated += 1
        if self.sdk.fail_calibrate_serial == self.serial:
            raise RuntimeError("injected calibration failure")

    def collect_sensor_data(self, dtype):
        assert dtype == _DataType.FORCE6D_VECTOR
        value = self.sdk.values[self.serial]
        if value is self.sdk.missing_key:
            return {}
        return {dtype: value}

    def release(self):
        self.released += 1
        remaining = self.sdk.release_failures_remaining.get(self.serial, 0)
        if remaining:
            self.sdk.release_failures_remaining[self.serial] = remaining - 1
            raise RuntimeError("injected release failure")


class _FakeSdk:
    VTSDataType = _DataType

    def __init__(self, sns, values) -> None:
        self.sns = list(sns)
        self.values = dict(values)
        self.sensors = {}
        self.missing_key = object()
        self.fail_calibrate_serial = None
        self.release_failures_remaining = {}

    def VTSDeviceFinder(self):
        return _FakeFinder(self)

    def VTSensor(self, *, config, force_model_path):
        return _FakeSensor(
            self,
            config=config,
            force_model_path=force_model_path,
        )


def _fixture(tmp_path: Path):
    serials = {finger: f"VTS-{index}" for index, finger in enumerate(VISIONTOUCH_FINGER_ORDER)}
    hashes = {}
    for finger, serial in serials.items():
        path = tmp_path / serial / f"{serial}.onnx.enc"
        path.parent.mkdir()
        content = f"encrypted-model-{finger}".encode()
        path.write_bytes(content)
        hashes[finger] = hashlib.sha256(content).hexdigest()
    values = {
        serials[finger]: np.arange(6, dtype=np.float32) + index * 10
        for index, finger in enumerate(VISIONTOUCH_FINGER_ORDER)
    }
    return serials, hashes, values


def _config(tmp_path, serials, hashes, **overrides):
    kwargs = {
        "force_model_dir": tmp_path,
        "finger_serials": serials,
        "expected_model_sha256": hashes,
        "allow_hardware_probe": True,
        "allow_hardware_stream": True,
    }
    kwargs.update(overrides)
    return VisionTouchForce6DConfig(**kwargs)


def test_visiontouch_is_lazy_and_emits_only_verified_force6d_features(tmp_path: Path) -> None:
    serials, hashes, values = _fixture(tmp_path)
    sdk = _FakeSdk(reversed(tuple(serials.values())), values)
    ticks = iter([10_000, 10_100])
    source = VisionTouchForce6DSource(
        _config(tmp_path, serials, hashes),
        sdk_module=sdk,
        sdk_version="1.0.10",
        clock=lambda: next(ticks),
    )
    assert not sdk.sensors
    report = source.probe()
    assert report.output_shape == (5, 6)
    assert report.axis_order == VISIONTOUCH_AXIS_ORDER
    assert report.brainco_commit == BRAINCO_REVO3_SDK_COMMIT
    assert report.vitai_commit == VITAI_SDK_COMMIT
    assert not report.real_hardware_function_verified
    assert not sdk.sensors

    source.start()
    assert all(sensor.calibrated == 1 for sensor in sdk.sensors.values())
    for finger, serial in serials.items():
        assert sdk.sensors[serial].force_model_path.endswith(f"{serial}.onnx.enc")
    sample = source.poll_force6d()
    expected = np.stack([values[serials[finger]] for finger in VISIONTOUCH_FINGER_ORDER])
    np.testing.assert_array_equal(sample.payload["features"], expected)
    np.testing.assert_array_equal(sample.payload["finger_valid"], np.ones(5, np.uint8))
    expected_shapes = np.full((5, VISIONTOUCH_MAX_RAW_RANK + 1), -1, np.int32)
    expected_shapes[:, :2] = [1, 6]
    np.testing.assert_array_equal(sample.payload["raw_return_shape"], expected_shapes)
    assert sample.header.capture_timestamp_ns == 10_100
    assert sample.header.device_timestamp_ns is None
    assert sample.header.valid
    assert source.episode_metadata["visiontouch_force6d"]["force_model_sha256"] == hashes
    source.stop()
    assert all(sensor.released == 1 for sensor in sdk.sensors.values())


def test_visiontouch_requires_explicit_probe_and_stream_permissions(tmp_path: Path) -> None:
    serials, hashes, values = _fixture(tmp_path)
    sdk = _FakeSdk(serials.values(), values)
    source = VisionTouchForce6DSource(
        _config(
            tmp_path,
            serials,
            hashes,
            allow_hardware_probe=False,
            allow_hardware_stream=False,
        ),
        sdk_module=sdk,
        sdk_version="1.0.10",
    )
    with pytest.raises(PermissionError, match="probe is disabled"):
        source.probe()
    assert not sdk.sensors

    probe_only = VisionTouchForce6DSource(
        _config(tmp_path, serials, hashes, allow_hardware_stream=False),
        sdk_module=sdk,
        sdk_version="1.0.10",
    )
    probe_only.probe()
    with pytest.raises(PermissionError, match="stream is disabled"):
        probe_only.start()
    assert not sdk.sensors


def test_visiontouch_fails_on_missing_finger_duplicate_or_model_mismatch(tmp_path: Path) -> None:
    serials, hashes, values = _fixture(tmp_path)
    missing_sdk = _FakeSdk(tuple(serials.values())[:-1], values)
    missing = VisionTouchForce6DSource(
        _config(tmp_path, serials, hashes),
        sdk_module=missing_sdk,
        sdk_version="1.0.10",
    )
    with pytest.raises(RuntimeError, match="sensors are missing"):
        missing.probe()

    duplicate = dict(serials)
    duplicate["pinky"] = duplicate["ring"]
    with pytest.raises(ValueError, match="must be unique"):
        _config(tmp_path, duplicate, hashes)

    wrong_hash = dict(hashes)
    wrong_hash["thumb"] = "0" * 64
    mismatch = VisionTouchForce6DSource(
        _config(tmp_path, serials, wrong_hash),
        sdk_module=_FakeSdk(serials.values(), values),
        sdk_version="1.0.10",
    )
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        mismatch.probe()


@pytest.mark.parametrize(
    "bad_value, message",
    [
        (np.zeros((1, 7), dtype=np.float32), r"non-empty \(\.\.\.,6\)"),
        (np.zeros((0, 6), dtype=np.float32), r"non-empty \(\.\.\.,6\)"),
        (np.asarray([0, 1, 2, 3, 4, np.nan]), "non-finite"),
    ],
)
def test_visiontouch_rejects_bad_force_samples(tmp_path: Path, bad_value, message) -> None:
    serials, hashes, values = _fixture(tmp_path)
    values[serials["middle"]] = bad_value
    source = VisionTouchForce6DSource(
        _config(tmp_path, serials, hashes),
        sdk_module=_FakeSdk(serials.values(), values),
        sdk_version="1.0.10",
    )
    source.probe()
    source.start()
    with pytest.raises(RuntimeError, match=message):
        source.poll_force6d()
    source.stop()


def test_visiontouch_accepts_official_multivector_shape_without_component_truncation(
    tmp_path: Path,
) -> None:
    serials, hashes, values = _fixture(tmp_path)
    raw = np.arange(36, dtype=np.float64).reshape(2, 3, 6)
    values[serials["thumb"]] = raw
    source = VisionTouchForce6DSource(
        _config(tmp_path, serials, hashes),
        sdk_module=_FakeSdk(serials.values(), values),
        sdk_version="1.0.10",
        clock=iter([1, 2]).__next__,
    )
    source.probe()
    source.start()
    sample = source.poll_force6d()
    np.testing.assert_allclose(
        sample.payload["features"][0],
        raw.reshape(-1, 6).mean(axis=0).astype(np.float32),
    )
    encoded = sample.payload["raw_return_shape"][0]
    np.testing.assert_array_equal(encoded[:4], [3, 2, 3, 6])
    assert np.all(encoded[4:] == -1)
    assert "mean over all leading axes" in source.episode_metadata["force6d_aggregation_rule"]
    source.stop()


def test_visiontouch_rechecks_model_hash_before_stream_start(tmp_path: Path) -> None:
    serials, hashes, values = _fixture(tmp_path)
    source = VisionTouchForce6DSource(
        _config(tmp_path, serials, hashes),
        sdk_module=_FakeSdk(serials.values(), values),
        sdk_version="1.0.10",
    )
    source.probe()
    serial = serials["index"]
    (tmp_path / serial / f"{serial}.onnx.enc").write_bytes(b"mutated")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        source.start()


def test_visiontouch_requires_pinned_sdk_version(tmp_path: Path) -> None:
    serials, hashes, values = _fixture(tmp_path)
    source = VisionTouchForce6DSource(
        _config(tmp_path, serials, hashes),
        sdk_module=_FakeSdk(serials.values(), values),
        sdk_version="1.0.15",
    )
    with pytest.raises(RuntimeError, match="version mismatch"):
        source.probe()


def test_visiontouch_partial_start_retains_unreleased_handle_for_retry(tmp_path: Path) -> None:
    serials, hashes, values = _fixture(tmp_path)
    sdk = _FakeSdk(serials.values(), values)
    sdk.fail_calibrate_serial = serials["middle"]
    sdk.release_failures_remaining[serials["index"]] = 1
    source = VisionTouchForce6DSource(
        _config(tmp_path, serials, hashes),
        sdk_module=sdk,
        sdk_version="1.0.10",
    )
    source.probe()
    with pytest.raises(RuntimeError, match="process/device intervention"):
        source.start()
    # The first release failed, so stop() must still own and retry this handle.
    assert sdk.sensors[serials["index"]].released == 1
    source.stop()
    assert sdk.sensors[serials["index"]].released == 2


def test_visiontouch_stop_retains_failed_release_for_retry(tmp_path: Path) -> None:
    serials, hashes, values = _fixture(tmp_path)
    sdk = _FakeSdk(serials.values(), values)
    source = VisionTouchForce6DSource(
        _config(tmp_path, serials, hashes),
        sdk_module=sdk,
        sdk_version="1.0.10",
    )
    source.probe()
    source.start()
    sdk.release_failures_remaining[serials["ring"]] = 1
    with pytest.raises(RuntimeError, match="retained handles"):
        source.stop()
    assert sdk.sensors[serials["ring"]].released == 1
    source.stop()
    assert sdk.sensors[serials["ring"]].released == 2
