"""Fail-closed Revo3 Ultra VisionTouch six-axis force source.

The implementation follows the public BrainCo Revo3 example at commit
``7ba96ebd5bee16b730577e2f7c5df98600da53eb`` and the public ViTai API
examples at commit ``0071c61a1bf13539ab8c12b537a69e11bb78430a``:

``VTSDeviceFinder -> get_sns/get_device_by_sn ->
VTSensor(config=..., force_model_path=...) -> calibrate ->
collect_sensor_data(VTSDataType.FORCE6D_VECTOR)``.

Importing this module never imports ``pyvitaisdk``, enumerates USB devices,
loads a model, or calibrates a sensor.  Those actions require explicit,
separate probe and stream permissions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from importlib import import_module, metadata
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from revo3_teleop.contracts import NativeSample, SampleHeader
from revo3_teleop.sources.common import Clock, monotonic_ns


BRAINCO_REVO3_SDK_REPOSITORY = "https://github.com/BrainCoTech/brainco-revo3-sdk"
BRAINCO_REVO3_SDK_COMMIT = "7ba96ebd5bee16b730577e2f7c5df98600da53eb"
VITAI_SDK_REPOSITORY = "https://github.com/ViTai-Tech/ViTai-SDK-Release"
VITAI_SDK_COMMIT = "0071c61a1bf13539ab8c12b537a69e11bb78430a"
VISIONTOUCH_DISTRIBUTION = "pyvitaisdk4bc"
VISIONTOUCH_IMPORT = "pyvitaisdk"
VISIONTOUCH_PINNED_VERSION = "1.0.10"
VISIONTOUCH_CLOCK_DOMAIN = "workstation_monotonic"
VISIONTOUCH_FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")
VISIONTOUCH_AXIS_ORDER = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
VISIONTOUCH_AXIS_UNITS = ("N", "N", "N", "Nm", "Nm", "Nm")
VISIONTOUCH_MAX_RAW_RANK = 8
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _installed_sdk_version() -> str:
    try:
        return metadata.version(VISIONTOUCH_DISTRIBUTION)
    except metadata.PackageNotFoundError:
        return "unknown-injected-or-uninstalled"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: Mapping[str, object]) -> str:
    encoded = json.dumps(dict(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _serial(value: object, *, finger: str) -> str:
    serial = str(value).strip()
    if (
        not serial
        or serial in {".", ".."}
        or "/" in serial
        or "\\" in serial
        or "\x00" in serial
    ):
        raise ValueError(f"invalid VisionTouch serial for {finger}")
    return serial


@dataclass(frozen=True)
class VisionTouchForce6DConfig:
    """Operator-reviewed identity and model contract for all five fingers."""

    force_model_dir: Path
    finger_serials: Mapping[str, str]
    expected_model_sha256: Mapping[str, str]
    expected_sdk_version: str = VISIONTOUCH_PINNED_VERSION
    allow_hardware_probe: bool = False
    allow_hardware_stream: bool = False

    def __post_init__(self) -> None:
        model_dir = Path(self.force_model_dir).expanduser()
        object.__setattr__(self, "force_model_dir", model_dir)
        finger_serials = {
            str(key): _serial(value, finger=str(key))
            for key, value in dict(self.finger_serials).items()
        }
        expected_hashes = {
            str(key): str(value).strip().lower()
            for key, value in dict(self.expected_model_sha256).items()
        }
        required = set(VISIONTOUCH_FINGER_ORDER)
        if set(finger_serials) != required:
            raise ValueError(
                "finger_serials must contain exactly thumb/index/middle/ring/pinky"
            )
        if len(set(finger_serials.values())) != len(VISIONTOUCH_FINGER_ORDER):
            raise ValueError("VisionTouch finger serial numbers must be unique")
        if set(expected_hashes) != required:
            raise ValueError("expected_model_sha256 keys must exactly match the five fingers")
        for finger, expected in expected_hashes.items():
            if _SHA256_RE.fullmatch(expected) is None:
                raise ValueError(f"expected model SHA-256 for {finger} must be 64 lowercase hex")
        if not str(self.expected_sdk_version).strip():
            raise ValueError("expected_sdk_version must be non-empty")
        object.__setattr__(self, "finger_serials", MappingProxyType(finger_serials))
        object.__setattr__(self, "expected_model_sha256", MappingProxyType(expected_hashes))


@dataclass(frozen=True)
class VisionTouchProbeReport:
    sdk_import: str
    sdk_distribution: str
    sdk_version: str
    brainco_repository: str
    brainco_commit: str
    vitai_repository: str
    vitai_commit: str
    discovered_sns: tuple[str, ...]
    finger_serials: tuple[tuple[str, str], ...]
    force_model_sha256: tuple[tuple[str, str], ...]
    force_model_relative_paths: tuple[tuple[str, str], ...]
    output_shape: tuple[int, int]
    finger_order: tuple[str, ...]
    axis_order: tuple[str, ...]
    axis_units: tuple[str, ...]
    capture_clock_provenance: str
    device_timestamp_available: bool
    force_model_mode: str
    real_hardware_function_verified: bool

    def fingerprint(self) -> str:
        return _stable_hash(asdict(self))

    def to_json(self) -> dict[str, object]:
        value = asdict(self)
        value["finger_serials"] = dict(self.finger_serials)
        value["force_model_sha256"] = dict(self.force_model_sha256)
        value["force_model_relative_paths"] = dict(self.force_model_relative_paths)
        value["fingerprint"] = self.fingerprint()
        return value


class VisionTouchForce6DSource:
    """Five explicitly mapped VisionTouch sensors exported as proven ``[5,6]``.

    A returned sample is all-or-nothing.  Any missing finger, duplicate serial,
    missing/mutated model, SDK error, wrong raw shape, or non-finite value raises
    before a sample can enter the recorder.
    """

    def __init__(
        self,
        config: VisionTouchForce6DConfig,
        *,
        sdk_module: Any | None = None,
        sdk_version: str | None = None,
        clock: Clock = monotonic_ns,
    ) -> None:
        self.config = config
        self._sdk = sdk_module
        self._sdk_version = _installed_sdk_version() if sdk_version is None else str(sdk_version)
        self._clock = clock
        self._report: VisionTouchProbeReport | None = None
        self._sensors: dict[str, Any] = {}
        self._force_dtype: object | None = None
        self._sequence = 0

    def _module(self) -> Any:
        if self._sdk is None:
            self._sdk = import_module(VISIONTOUCH_IMPORT)
        for name in ("VTSDeviceFinder", "VTSensor", "VTSDataType"):
            if getattr(self._sdk, name, None) is None:
                raise RuntimeError(f"pyvitaisdk API mismatch: missing {name}")
        return self._sdk

    def _model_path(self, serial: str) -> Path:
        root = self.config.force_model_dir.resolve(strict=True)
        candidate = (root / serial / f"{serial}.onnx.enc").resolve(strict=True)
        if not candidate.is_relative_to(root):
            raise RuntimeError(f"VisionTouch model path escapes force_model_dir for {serial}")
        if not candidate.is_file():
            raise RuntimeError(f"VisionTouch force model is not a file for {serial}")
        return candidate

    def _verify_models(self) -> dict[str, tuple[Path, str]]:
        verified: dict[str, tuple[Path, str]] = {}
        for finger in VISIONTOUCH_FINGER_ORDER:
            serial = self.config.finger_serials[finger]
            try:
                path = self._model_path(serial)
            except (OSError, RuntimeError) as exc:
                raise RuntimeError(
                    f"required VisionTouch force model is missing for {finger}/{serial}"
                ) from exc
            observed = _sha256_file(path)
            expected = self.config.expected_model_sha256[finger]
            if observed != expected:
                raise RuntimeError(
                    f"VisionTouch force model SHA-256 mismatch for {finger}/{serial}"
                )
            verified[finger] = (path, observed)
        return verified

    @staticmethod
    def _enumerate(finder: Any) -> tuple[str, ...]:
        get_sns = getattr(finder, "get_sns", None)
        get_device = getattr(finder, "get_device_by_sn", None)
        if not callable(get_sns) or not callable(get_device):
            raise RuntimeError("pyvitaisdk finder API mismatch")
        sns = tuple(str(value).strip() for value in get_sns())
        if any(not value for value in sns):
            raise RuntimeError("VisionTouch discovery returned an empty serial")
        if len(set(sns)) != len(sns):
            raise RuntimeError("VisionTouch discovery returned duplicate serial numbers")
        return tuple(sorted(sns))

    def probe(self) -> VisionTouchProbeReport:
        if self._sensors:
            raise RuntimeError("cannot probe while VisionTouch streaming is active")
        if not self.config.allow_hardware_probe:
            raise PermissionError("VisionTouch hardware probe is disabled")
        if self._sdk_version != self.config.expected_sdk_version:
            raise RuntimeError(
                "pyvitaisdk4bc version mismatch: "
                f"expected={self.config.expected_sdk_version}, observed={self._sdk_version}"
            )
        sdk = self._module()
        finder = sdk.VTSDeviceFinder()
        discovered = self._enumerate(finder)
        missing = [
            self.config.finger_serials[finger]
            for finger in VISIONTOUCH_FINGER_ORDER
            if self.config.finger_serials[finger] not in discovered
        ]
        if missing:
            raise RuntimeError(f"configured VisionTouch sensors are missing: {missing}")
        # Resolve each opaque SDK config during the read-only capability phase.
        for finger in VISIONTOUCH_FINGER_ORDER:
            serial = self.config.finger_serials[finger]
            if finder.get_device_by_sn(serial) is None:
                raise RuntimeError(f"VisionTouch finder returned no config for {finger}/{serial}")
        verified = self._verify_models()
        report = VisionTouchProbeReport(
            sdk_import=VISIONTOUCH_IMPORT,
            sdk_distribution=VISIONTOUCH_DISTRIBUTION,
            sdk_version=self._sdk_version,
            brainco_repository=BRAINCO_REVO3_SDK_REPOSITORY,
            brainco_commit=BRAINCO_REVO3_SDK_COMMIT,
            vitai_repository=VITAI_SDK_REPOSITORY,
            vitai_commit=VITAI_SDK_COMMIT,
            discovered_sns=discovered,
            finger_serials=tuple(
                (finger, self.config.finger_serials[finger])
                for finger in VISIONTOUCH_FINGER_ORDER
            ),
            force_model_sha256=tuple(
                (finger, verified[finger][1]) for finger in VISIONTOUCH_FINGER_ORDER
            ),
            force_model_relative_paths=tuple(
                (
                    finger,
                    verified[finger][0]
                    .relative_to(self.config.force_model_dir.resolve(strict=True))
                    .as_posix(),
                )
                for finger in VISIONTOUCH_FINGER_ORDER
            ),
            output_shape=(5, 6),
            finger_order=VISIONTOUCH_FINGER_ORDER,
            axis_order=VISIONTOUCH_AXIS_ORDER,
            axis_units=VISIONTOUCH_AXIS_UNITS,
            capture_clock_provenance="host_sdk_read_completion_monotonic",
            device_timestamp_available=False,
            force_model_mode="required_per_serial_sha256_verified",
            real_hardware_function_verified=False,
        )
        self._report = report
        return report

    @property
    def report(self) -> VisionTouchProbeReport:
        if self._report is None:
            raise RuntimeError("VisionTouch source has not passed capability probe")
        return self._report

    @property
    def episode_metadata(self) -> dict[str, object]:
        return {
            "visiontouch_force6d": self.report.to_json(),
            "sample_payload": {
                "features": {
                    "shape": [5, 6],
                    "finger_order": list(VISIONTOUCH_FINGER_ORDER),
                    "axis_order": list(VISIONTOUCH_AXIS_ORDER),
                    "axis_units": list(VISIONTOUCH_AXIS_UNITS),
                },
                "raw_return_shape": {
                    "shape": [5, VISIONTOUCH_MAX_RAW_RANK + 1],
                    "encoding": "[rank, dim0, ..., dim(rank-1), -1 padding]",
                },
                "finger_valid": [5],
            },
            "force6d_aggregation_rule": (
                "require non-empty raw (...,6); preserve (6,); otherwise mean over all "
                "leading axes; never truncate or zero-pad components"
            ),
        }

    def start(self) -> None:
        if self._sensors:
            raise RuntimeError("VisionTouch source is already streaming")
        report = self.report
        if not self.config.allow_hardware_stream:
            raise PermissionError("VisionTouch hardware stream is disabled")
        sdk = self._module()
        finder = sdk.VTSDeviceFinder()
        discovered = self._enumerate(finder)
        if discovered != report.discovered_sns:
            raise RuntimeError("VisionTouch discovery changed since capability probe")
        verified = self._verify_models()
        dtype = getattr(sdk.VTSDataType, "FORCE6D_VECTOR", None)
        if dtype is None:
            raise RuntimeError("pyvitaisdk exposes no FORCE6D_VECTOR data type")
        opened: dict[str, Any] = {}
        try:
            for finger in VISIONTOUCH_FINGER_ORDER:
                serial = self.config.finger_serials[finger]
                device_config = finder.get_device_by_sn(serial)
                if device_config is None:
                    raise RuntimeError(f"VisionTouch config disappeared for {finger}/{serial}")
                sensor = sdk.VTSensor(
                    config=device_config,
                    force_model_path=str(verified[finger][0]),
                )
                for method in ("calibrate", "collect_sensor_data", "release"):
                    if not callable(getattr(sensor, method, None)):
                        raise RuntimeError(f"VisionTouch sensor API mismatch: missing {method}")
                opened[finger] = sensor
                sensor.calibrate()
        except Exception as start_failure:
            unreleased: dict[str, Any] = {}
            release_failures: list[BaseException] = []
            for finger, sensor in opened.items():
                try:
                    sensor.release()
                except BaseException as exc:
                    unreleased[finger] = sensor
                    release_failures.append(exc)
            if release_failures:
                # Keep every unreleased handle reachable so stop()/the assembly
                # close path can retry.  Dropping these objects would make a
                # partially opened USB sensor invisible to later cleanup.
                self._sensors = unreleased
                self._force_dtype = None
                raise RuntimeError(
                    "VisionTouch startup failed and one or more sensors could not "
                    "be released; process/device intervention is required"
                ) from start_failure
            raise
        self._sensors = opened
        self._force_dtype = dtype
        self._sequence = 0

    def poll_force6d(self) -> NativeSample:
        if tuple(self._sensors) != VISIONTOUCH_FINGER_ORDER or self._force_dtype is None:
            raise RuntimeError("VisionTouch source is not streaming with all five fingers")
        read_start_ns = int(self._clock())
        rows: list[np.ndarray] = []
        raw_shapes: list[tuple[int, ...]] = []
        for finger in VISIONTOUCH_FINGER_ORDER:
            sensor = self._sensors[finger]
            result = sensor.collect_sensor_data(self._force_dtype)
            if not isinstance(result, Mapping) or self._force_dtype not in result:
                raise RuntimeError(f"VisionTouch FORCE6D_VECTOR missing for {finger}")
            raw = np.asarray(result[self._force_dtype])
            raw_shapes.append(tuple(int(value) for value in raw.shape))
            if raw.ndim < 1 or raw.ndim > VISIONTOUCH_MAX_RAW_RANK:
                raise RuntimeError(
                    f"VisionTouch FORCE6D_VECTOR for {finger} has unsupported rank {raw.ndim}"
                )
            if raw.shape[-1] != 6 or raw.size == 0:
                raise RuntimeError(
                    f"VisionTouch FORCE6D_VECTOR for {finger} must be non-empty (...,6), "
                    f"observed {raw.shape}"
                )
            numeric = raw.astype(np.float64, copy=False)
            if not np.isfinite(numeric).all():
                raise RuntimeError(f"VisionTouch FORCE6D_VECTOR contains non-finite data for {finger}")
            row64 = numeric if numeric.ndim == 1 else numeric.reshape(-1, 6).mean(axis=0)
            row = np.asarray(row64, dtype=np.float32)
            if row.shape != (6,) or not np.isfinite(row).all():
                raise RuntimeError(f"VisionTouch FORCE6D aggregation is invalid for {finger}")
            rows.append(row.copy())
        features = np.stack(rows, axis=0)
        if features.shape != (5, 6) or not np.isfinite(features).all():
            raise RuntimeError("VisionTouch aggregate must be finite [5,6]")
        read_end_ns = max(int(self._clock()), read_start_ns + 1)
        sequence = self._sequence
        self._sequence += 1
        encoded_shapes = np.full(
            (len(VISIONTOUCH_FINGER_ORDER), VISIONTOUCH_MAX_RAW_RANK + 1),
            -1,
            dtype=np.int32,
        )
        for row_index, shape in enumerate(raw_shapes):
            encoded_shapes[row_index, 0] = len(shape)
            encoded_shapes[row_index, 1 : 1 + len(shape)] = shape
        return NativeSample(
            SampleHeader(
                source_id=f"revo3_u21vt_visiontouch_force6d_{self.report.fingerprint()[:12]}",
                sequence=sequence,
                capture_timestamp_ns=read_end_ns,
                receive_timestamp_ns=read_end_ns,
                clock_domain=VISIONTOUCH_CLOCK_DOMAIN,
                device_timestamp_ns=None,
                valid=True,
            ),
            {
                "features": features,
                "finger_valid": np.ones(5, dtype=np.uint8),
                "raw_return_shape": encoded_shapes,
                "host_read_start_timestamp_ns": np.asarray([read_start_ns], dtype=np.int64),
                "host_read_end_timestamp_ns": np.asarray([read_end_ns], dtype=np.int64),
                "device_timestamp_available": np.asarray([0], dtype=np.uint8),
                "clock_is_host_reconstruction": np.asarray([1], dtype=np.uint8),
            },
        )

    def stop(self) -> None:
        sensors = self._sensors
        unreleased: dict[str, Any] = {}
        failures: list[BaseException] = []
        for finger, sensor in sensors.items():
            try:
                sensor.release()
            except BaseException as exc:
                unreleased[finger] = sensor
                failures.append(exc)
        self._sensors = unreleased
        if not unreleased:
            self._force_dtype = None
        if failures:
            raise RuntimeError(
                "one or more VisionTouch sensors failed to release; retained handles "
                "require retry or process/device intervention"
            ) from failures[0]


__all__ = [
    "BRAINCO_REVO3_SDK_COMMIT",
    "BRAINCO_REVO3_SDK_REPOSITORY",
    "VITAI_SDK_COMMIT",
    "VITAI_SDK_REPOSITORY",
    "VISIONTOUCH_AXIS_ORDER",
    "VISIONTOUCH_AXIS_UNITS",
    "VISIONTOUCH_CLOCK_DOMAIN",
    "VISIONTOUCH_DISTRIBUTION",
    "VISIONTOUCH_FINGER_ORDER",
    "VISIONTOUCH_IMPORT",
    "VISIONTOUCH_MAX_RAW_RANK",
    "VISIONTOUCH_PINNED_VERSION",
    "VisionTouchForce6DConfig",
    "VisionTouchForce6DSource",
    "VisionTouchProbeReport",
]
