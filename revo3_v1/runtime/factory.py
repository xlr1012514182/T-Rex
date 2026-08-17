"""Fail-closed assembly of the executable Revo3 V1 runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from revo3_v1.emg.calibration import load_emg_calibration
from revo3_v1.emg.model import load_emg_checkpoint
from revo3_v1.emg.preprocessing import EmgPreprocessingProfile
from revo3_v1.emg.streaming import StreamingEMGClassifier
from revo3_v1.executive import RuntimeVersions, TaskExecutive, TaskExecutiveConfig
from revo3_v1.planner import (
    AskToClarifyPlanner,
    AsyncPlannerWorker,
    MockPlannerBackend,
    Qwen3VLBackend,
    VisualGate,
    VisualGateConfig,
    SupportedTask,
    TASK_GRASP_PRIMITIVES,
)
from revo3_v1.policy import (
    AsyncTReXPolicyRunner,
    MockTReXBackend,
    TReXRevoPolicyAdapter,
    TReXServerIdentity,
    TReXZmqError,
    ZmqTReXBackend,
)
from revo3_v1.revo import (
    JOINT_ORDER_HASH,
    CompletionConfig,
    CompletionMonitor,
    MockRevoBackend,
    RevoBackend,
    RevoCommandPipeline,
    SafetyEnvelope,
    SafetySupervisor,
)
from revo3_v1.revo.servo import RevoServoConfig, RevoServoExecutor
from revo3_v1.tactile import (
    ClosingSynergyArtifact,
    ReflexConfig,
    TactileFrame,
    TactileReflexPlugin,
    TactileWindow,
)
from revo3_v1.vision import (
    CameraHealthConfig,
    CameraHealthMonitor,
    SingleCameraViewConfig,
    SingleCameraViewDeriver,
)
from revo3_v1.vision.frontend import DerivedCameraViews

from .emg_bridge import StreamingEMGEventBridge
from .orchestrator import OnlineV1Coordinator, RuntimeSynchronizedInput
from .service import DoubleRateRuntimeService, EMGPacket, RuntimeIO


CONTROL_SCHEMA = "revo3-v1-control-v1"
RUNTIME_SCHEMA = "revo3-v1-runtime-v1"
HARDWARE_CALIBRATION_SCHEMA = "revo3-v1-hardware-calibration-v1"


class RuntimeAssemblyError(RuntimeError):
    pass


def _load_json(path: str | Path, *, label: str) -> tuple[Path, Mapping[str, Any]]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise RuntimeAssemblyError(f"{label} is missing: {resolved}")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RuntimeAssemblyError(f"{label} must be a JSON object")
    return resolved, value


def _resolve_required(root: Path, value: object, *, label: str, directory: bool = False) -> Path:
    text = str(value or "").strip()
    if not text:
        raise RuntimeAssemblyError(f"production requires {label}")
    path = Path(text)
    if not path.is_absolute():
        path = (root / path).resolve()
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        kind = "directory" if directory else "file"
        raise RuntimeAssemblyError(f"production {label} {kind} is missing: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_sha256(value: object, *, label: str) -> str:
    text = str(value or "").strip()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise RuntimeAssemblyError(
            f"{label} must be a lowercase 64-character SHA-256 digest"
        )
    return text


def _validate_policy_tactile_profile_binding(
    hardware: Mapping[str, Any], identity: TReXServerIdentity
) -> str:
    """Prove the calibrated sensor profile is compatible with served policy bytes."""

    compatible = _required_sha256(
        hardware.get("compatible_policy_tactile_profile_manifest_sha256"),
        label="hardware compatible_policy_tactile_profile_manifest_sha256",
    )
    if compatible != identity.tactile_profile_manifest_sha256:
        raise RuntimeAssemblyError(
            "hardware/policy tactile profile manifest SHA-256 mismatch; "
            "a matching profile name or combined runtime fingerprint is not compatibility proof"
        )
    return compatible


def _ms(value: object, *, name: str) -> int:
    number = int(value)
    if number < 0:
        raise RuntimeAssemblyError(f"{name} must be non-negative")
    return number * 1_000_000


def _runtime_service_kwargs(control: Mapping[str, Any]) -> dict[str, int]:
    runtime = control.get("runtime", {})
    if not isinstance(runtime, Mapping):
        raise RuntimeAssemblyError("control config runtime must be an object")
    control_watchdog_ns = _ms(
        runtime.get("control_watchdog_ms", 100), name="control_watchdog_ms"
    )
    servo_io_timeout_ns = _ms(
        runtime.get("servo_io_timeout_ms", 50), name="servo_io_timeout_ms"
    )
    io_close_timeout_ns = _ms(
        runtime.get("io_close_timeout_ms", 1000), name="io_close_timeout_ms"
    )
    if min(control_watchdog_ns, servo_io_timeout_ns, io_close_timeout_ns) <= 0:
        raise RuntimeAssemblyError("runtime service timeouts must be positive")
    if servo_io_timeout_ns > control_watchdog_ns:
        raise RuntimeAssemblyError(
            "runtime.servo_io_timeout_ms must not exceed control_watchdog_ms"
        )
    return {
        "control_watchdog_ns": control_watchdog_ns,
        "servo_io_timeout_ns": servo_io_timeout_ns,
        "io_close_timeout_ns": io_close_timeout_ns,
    }


def load_control_config(
    path: str | Path, *, strict_production: bool = False
) -> tuple[Path, Mapping[str, Any], TaskExecutiveConfig]:
    resolved, value = _load_json(path, label="control config")
    if value.get("schema_version") != CONTROL_SCHEMA:
        raise RuntimeAssemblyError("unsupported control config schema")
    planner = value.get("planner")
    executive = value.get("executive")
    if not isinstance(planner, Mapping) or not isinstance(executive, Mapping):
        raise RuntimeAssemblyError("control config requires planner and executive objects")
    emg = value.get("emg")
    if not isinstance(emg, Mapping):
        raise RuntimeAssemblyError("control config requires emg object")
    if strict_production:
        runtime = value.get("runtime")
        if not isinstance(runtime, Mapping):
            raise RuntimeAssemblyError("production control config requires runtime object")
        required_planner = {
            "base_model", "base_revision", "max_new_tokens", "repair_retries"
        }
        required_executive = {
            "emg_start_ttl_ms", "pending_intent_ttl_ms", "camera_ttl_ms",
            "state_ttl_ms", "touch_ttl_ms", "policy_ttl_ms",
            "slow_response_observation_budget_ms",
            "fast_response_observation_budget_ms", "planner_ttl_ms",
            "planner_sla_ms", "planner_source_max_age_ms",
            "planner_scene_max_distance", "lease_ttl_ms", "release_timeout_ms",
            "allowed_future_skew_ms", "commit_stability_ms",
            "touch_stale_abort_ms", "camera_stale_abort_ms",
            "policy_stale_abort_ms", "policy_startup_abort_ms", "max_replans",
        }
        required_emg = {
            "start_confidence", "start_margin", "min_signal_quality"
        }
        required_runtime = {
            "control_watchdog_ms", "servo_io_timeout_ms", "io_close_timeout_ms"
        }
        missing = (
            [f"planner.{key}" for key in sorted(required_planner - set(planner))]
            + [f"executive.{key}" for key in sorted(required_executive - set(executive))]
            + [f"emg.{key}" for key in sorted(required_emg - set(emg))]
            + [f"runtime.{key}" for key in sorted(required_runtime - set(runtime))]
        )
        if missing:
            raise RuntimeAssemblyError(
                "production control config is missing critical keys: " + ", ".join(missing)
            )
    if int(planner.get("max_new_tokens", 0)) <= 0:
        raise RuntimeAssemblyError("planner.max_new_tokens must be positive")
    if strict_production and int(planner["max_new_tokens"]) != 384:
        raise RuntimeAssemblyError(
            "production requires planner.max_new_tokens=384; 128 is smoke/ablation only"
        )
    _runtime_service_kwargs(value)
    config = TaskExecutiveConfig(
        emg_start_ttl_ns=_ms(executive.get("emg_start_ttl_ms", 3000), name="emg_start_ttl_ms"),
        pending_intent_ttl_ns=_ms(
            executive.get("pending_intent_ttl_ms", 22000), name="pending_intent_ttl_ms"
        ),
        camera_ttl_ns=_ms(executive.get("camera_ttl_ms", 100), name="camera_ttl_ms"),
        state_ttl_ns=_ms(executive.get("state_ttl_ms", 50), name="state_ttl_ms"),
        touch_ttl_ns=_ms(executive.get("touch_ttl_ms", 150), name="touch_ttl_ms"),
        policy_ttl_ns=_ms(executive.get("policy_ttl_ms", 750), name="policy_ttl_ms"),
        slow_response_observation_budget_ns=_ms(
            executive.get("slow_response_observation_budget_ms", 1500),
            name="slow_response_observation_budget_ms",
        ),
        fast_response_observation_budget_ns=_ms(
            executive.get("fast_response_observation_budget_ms", 500),
            name="fast_response_observation_budget_ms",
        ),
        planner_ttl_ns=_ms(executive.get("planner_ttl_ms", 1000), name="planner_ttl_ms"),
        planner_sla_ns=_ms(executive.get("planner_sla_ms", 20000), name="planner_sla_ms"),
        planner_source_max_age_ns=_ms(
            executive.get("planner_source_max_age_ms", 18000),
            name="planner_source_max_age_ms",
        ),
        lease_ttl_ns=_ms(executive.get("lease_ttl_ms", 3000), name="lease_ttl_ms"),
        release_timeout_ns=_ms(
            executive.get("release_timeout_ms", 3000), name="release_timeout_ms"
        ),
        allowed_future_skew_ns=_ms(
            executive.get("allowed_future_skew_ms", 5), name="allowed_future_skew_ms"
        ),
        min_start_confidence=float(emg.get("start_confidence", 0.8)),
        min_start_margin=float(emg.get("start_margin", 0.2)),
        min_signal_quality=float(emg.get("min_signal_quality", 0.8)),
        max_replans=int(executive.get("max_replans", 1)),
        commit_stability_ns=_ms(
            executive.get("commit_stability_ms", 150), name="commit_stability_ms"
        ),
        touch_stale_abort_ns=_ms(
            executive.get("touch_stale_abort_ms", 500), name="touch_stale_abort_ms"
        ),
        camera_stale_abort_ns=_ms(
            executive.get("camera_stale_abort_ms", 1000), name="camera_stale_abort_ms"
        ),
        policy_stale_abort_ns=_ms(
            executive.get("policy_stale_abort_ms", 1000), name="policy_stale_abort_ms"
        ),
        policy_startup_abort_ns=_ms(
            executive.get("policy_startup_abort_ms", 3000), name="policy_startup_abort_ms"
        ),
    )
    return resolved, value, config


def _array21(value: object, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (21,) or not np.isfinite(result).all():
        raise RuntimeAssemblyError(f"{name} must contain 21 finite values")
    return result


def _array5(value: object, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (5,) or not np.isfinite(result).all():
        raise RuntimeAssemblyError(f"{name} must contain five finite values")
    return result


@dataclass(frozen=True)
class RuntimeAssembly:
    mode: str
    coordinator: OnlineV1Coordinator
    service: DoubleRateRuntimeService
    resolved: Mapping[str, Any]


@dataclass(frozen=True)
class ProductionBindings:
    revo_backend: RevoBackend
    io: RuntimeIO

    def __post_init__(self) -> None:
        if not isinstance(self.revo_backend, RevoBackend):
            raise RuntimeAssemblyError(
                "production requires a complete RevoBackend including bounded close()"
            )
        if isinstance(self.revo_backend, MockRevoBackend) or not bool(
            getattr(self.revo_backend, "is_hardware", False)
        ):
            raise RuntimeAssemblyError("production requires an armed, non-Mock hardware Revo backend")
        if not bool(getattr(self.revo_backend, "production_ready", False)):
            raise RuntimeAssemblyError(
                "production Revo backend must explicitly declare production_ready=True "
                "after write arming, capability probe, temperature, SoftStop and "
                "non-auto-clear collision profile verification, with no latched fault/timeout"
            )
        if not isinstance(self.io, RuntimeIO):
            raise RuntimeAssemblyError("production requires a RuntimeIO implementation")
        if isinstance(self.io, SyntheticRuntimeIO) or not bool(
            getattr(self.io, "is_hardware", False)
        ):
            raise RuntimeAssemblyError(
                "production requires a non-synthetic RuntimeIO with is_hardware=True"
            )
        if not bool(getattr(self.io, "raw_emg_only", False)):
            raise RuntimeAssemblyError(
                "production RuntimeIO must declare raw_emg_only=True; direct EmgEvent fixtures are forbidden"
            )


class _NoEventEMGSource:
    def push_many(self, samples, sample_timestamps_ns, signal_quality=1.0):
        del samples, sample_timestamps_ns, signal_quality
        return []

    def reset(self, active=False):
        del active


class _ScriptedFixtureEMGSource:
    """Explicit simulation-only edge script; never a classifier substitute."""

    def __init__(self, primitive: str, *, release_after_packets: int = 25) -> None:
        self.primitive = primitive
        self.release_after_packets = int(release_after_packets)
        self.count = 0
        self.started = False
        self.released = False

    def push_many(self, samples, sample_timestamps_ns, signal_quality=1.0):
        del samples
        self.count += 1
        timestamp_ns = int(np.asarray(sample_timestamps_ns, dtype=np.int64)[-1])
        common = {
            "confidence": 0.99,
            "margin": 0.9,
            "signal_quality": float(signal_quality),
            "timestamp_ns": timestamp_ns,
        }
        if not self.started:
            self.started = True
            return [{
                **common,
                "primitive": self.primitive,
                "event": {
                    **common,
                    "type": "StartIntentEvent",
                    "primitive": self.primitive,
                    "event_id": "simulation-start",
                },
            }]
        if not self.released and self.count >= self.release_after_packets:
            self.released = True
            return [{
                **common,
                "primitive": "RELEASE",
                "event": {
                    **common,
                    "type": "ReleaseEvent",
                    "primitive": "RELEASE",
                    "event_id": "simulation-release",
                },
            }]
        return [{**common, "primitive": "REST", "event": None}]

    def reset(self, active=False):
        # Lifecycle resets do not replay a fixture edge inside one assembly.
        del active


class SyntheticRuntimeIO:
    """Deterministic acquisition fixture for CLI dry-run/simulation only."""

    def __init__(self, backend: MockRevoBackend, *, calibration_hash: str) -> None:
        self.is_hardware = False
        self.raw_emg_only = True
        self.backend = backend
        self.deriver = SingleCameraViewDeriver(SingleCameraViewConfig(calibration_hash))
        self.calibration_hash = calibration_hash
        self.sequence = 0
        self.closed = False
        yy, xx = np.indices((288, 384))
        base = np.stack(
            (xx % 255, (2 * yy) % 255, (xx + yy) % 255), axis=-1
        ).astype(np.uint8)
        self._fixture_frame = base

    async def poll_emg_packet(self, *, now_ns: int):
        timestamps = np.arange(now_ns - 48_000_000, now_ns + 1, 4_000_000, dtype=np.int64)
        return EMGPacket(
            samples=np.zeros((8, timestamps.size), dtype=np.float32),
            sample_timestamps_ns=timestamps,
            signal_quality=0.99,
        )

    async def _input(self, now_ns: int) -> RuntimeSynchronizedInput:
        self.sequence += 1
        rgb = self._fixture_frame.copy()
        # Change one non-signature pixel so the camera-health frozen-frame
        # check sees a causal sequence without simulating semantic scene drift.
        rgb[0, 0, 0] = np.uint8(self.sequence % 255)
        views = DerivedCameraViews(
            capture_timestamp_ns=now_ns,
            sequence=self.sequence,
            calibration_hash=self.calibration_hash,
            full=rgb,
            fixed_center=rgb,
            source_shape=rgb.shape,
            center_crop_px=(0, 0, rgb.shape[1], rgb.shape[0]),
        )
        state = await self.backend.read_state()
        # Simulation clock is authoritative for the fixture's observed state.
        state = type(state)(
            timestamp_ns=now_ns,
            q_rad=state.q_rad,
            # The in-memory backend uses a host clock unrelated to this
            # deterministic fixture clock.  Reconstruct causal, stationary
            # telemetry here rather than leaking that cross-clock derivative.
            dq_rad_s=np.zeros(21, dtype=np.float32),
            current_a=state.current_a,
            status=state.status,
            sequence=self.sequence,
            temperature_c=state.temperature_c,
        )
        history_ts = np.arange(now_ns - 15_000_000, now_ns + 1, 1_000_000, dtype=np.int64)
        history = np.zeros((16, 5, 6), dtype=np.float32)
        tactile = TactileFrame(now_ns, history[-1], self.sequence)
        window = TactileWindow(
            history,
            history_ts,
            np.arange(self.sequence * 16, self.sequence * 16 + 16, dtype=np.int64),
            np.ones((16, 5), dtype=bool),
        )
        return RuntimeSynchronizedInput(
            now_ns=now_ns,
            emg=None,
            views=views,
            state=state,
            tactile=tactile,
            tactile_window=window,
            tactile_deform=np.zeros((5, 240, 240), dtype=np.uint8),
            tactile_deform_timestamp_ns=np.full(5, now_ns, dtype=np.int64),
        )

    async def control_input(self, *, now_ns: int) -> RuntimeSynchronizedInput:
        return await self._input(now_ns)

    async def servo_input(self, *, now_ns: int, latest_control: RuntimeSynchronizedInput):
        del latest_control
        return await self._input(now_ns)

    async def close(self) -> None:
        self.closed = True


def build_simulation_runtime(
    control_config: str | Path,
    *,
    task: SupportedTask = SupportedTask.BOTTLE,
    scripted_emg: bool = True,
) -> RuntimeAssembly:
    _, control, executive_config = load_control_config(control_config)
    calibration_hash = "simulation-camera-calibration"
    hand = MockRevoBackend()
    planner_backend = MockPlannerBackend(default_task=task)
    policy_backend = MockTReXBackend()
    policy_worker = AsyncTReXPolicyRunner(TReXRevoPolicyAdapter(policy_backend))
    servo = RevoServoExecutor(
        RevoCommandPipeline(hand, SafetySupervisor(SafetyEnvelope.demo())),
        policy_worker,
        TactileReflexPlugin(ReflexConfig(enabled=False)),
        RevoServoConfig.demo(),
    )
    coordinator = OnlineV1Coordinator(
        planner_worker=AsyncPlannerWorker(AskToClarifyPlanner(planner_backend)),
        camera_health=CameraHealthMonitor(CameraHealthConfig(calibration_hash)),
        visual_gate=VisualGate(
            VisualGateConfig(max_decision_age_ns=executive_config.planner_ttl_ns)
        ),
        executive=TaskExecutive(executive_config),
        policy_worker=policy_worker,
        completion=CompletionMonitor(CompletionConfig.demo()),
        servo=servo,
        versions=RuntimeVersions(
            CONTROL_SCHEMA,
            "mock-planner-simulation-only",
            "mock-trex-simulation-only",
            "mock-hardware-simulation-only",
            JOINT_ORDER_HASH,
            "profile-a-simulation-only",
        ),
        emg_bridge=StreamingEMGEventBridge(
            _ScriptedFixtureEMGSource(TASK_GRASP_PRIMITIVES[task].value)
            if scripted_emg
            else _NoEventEMGSource()
        ),
        planner_scene_max_distance=float(
            control["executive"].get("planner_scene_max_distance", 0.08)
        ),
    )
    io = SyntheticRuntimeIO(hand, calibration_hash=calibration_hash)
    service = DoubleRateRuntimeService(
        coordinator,
        io,
        **_runtime_service_kwargs(control),
    )
    return RuntimeAssembly(
        "simulation",
        coordinator,
        service,
        {
            "mode": "simulation",
            "planner_backend": type(planner_backend).__name__,
            "policy_backend": type(policy_backend).__name__,
            "max_new_tokens": int(control["planner"]["max_new_tokens"]),
            "claim": "wiring smoke only",
            "task": task.value,
            "scripted_emg": bool(scripted_emg),
        },
    )


def _hardware_components(path: Path, payload: Mapping[str, Any]):
    if payload.get("schema_version") != HARDWARE_CALIBRATION_SCHEMA:
        raise RuntimeAssemblyError("unsupported hardware calibration schema")
    if payload.get("example_only") is True:
        raise RuntimeAssemblyError("example-only hardware calibration cannot arm production")
    if payload.get("joint_order_hash") != JOINT_ORDER_HASH:
        raise RuntimeAssemblyError("hardware calibration joint order mismatch")
    safety = payload.get("safety")
    completion = payload.get("completion")
    if not isinstance(safety, Mapping) or not isinstance(completion, Mapping):
        raise RuntimeAssemblyError("hardware calibration requires safety/completion objects")
    safe_open = _array21(completion.get("safe_open_q_rad"), name="safe_open_q_rad")
    envelope = SafetyEnvelope(
        q_min_rad=_array21(safety.get("q_min_rad"), name="q_min_rad"),
        q_max_rad=_array21(safety.get("q_max_rad"), name="q_max_rad"),
        max_step_rad=_array21(safety.get("max_step_rad"), name="max_step_rad"),
        max_abs_current_a=_array21(safety.get("max_abs_current_a"), name="max_abs_current_a"),
        max_abs_velocity_rad_s=_array21(
            safety.get("max_abs_velocity_rad_s"), name="max_abs_velocity_rad_s"
        ),
        max_abs_acceleration_rad_s2=_array21(
            safety.get("max_abs_acceleration_rad_s2"), name="max_abs_acceleration_rad_s2"
        ),
        max_temperature_c=_array21(safety.get("max_temperature_c"), name="max_temperature_c"),
        require_temperature_telemetry=True,
        hardware_profile_id=str(safety.get("hardware_profile_id", "")),
        simulation_only=False,
    )
    if not envelope.hardware_ready:
        raise RuntimeAssemblyError("hardware safety envelope is not production-ready")
    calibration_id = str(completion.get("calibration_id", "")).strip()
    completion_config = CompletionConfig(
        contact_force_threshold=_array5(
            completion.get("contact_force_threshold"), name="contact_force_threshold"
        ),
        release_force_threshold=_array5(
            completion.get("release_force_threshold"), name="release_force_threshold"
        ),
        safe_open_q_rad=safe_open,
        simulation_only=False,
        calibration_id=calibration_id,
    )
    servo_config = RevoServoConfig(
        safe_open,
        hardware_mode=True,
        calibration_id=calibration_id,
    )
    return envelope, completion_config, servo_config, _sha256(path)


def build_production_runtime(
    control_config: str | Path,
    runtime_config: str | Path,
    *,
    bindings: ProductionBindings,
) -> RuntimeAssembly:
    _, control, executive_config = load_control_config(
        control_config, strict_production=True
    )
    planner_cfg = control["planner"]
    if int(planner_cfg.get("max_new_tokens", -1)) != 384:
        raise RuntimeAssemblyError(
            "production requires planner.max_new_tokens=384; 128 is smoke/ablation only"
        )
    runtime_path, runtime = _load_json(runtime_config, label="runtime config")
    if runtime.get("schema_version") != RUNTIME_SCHEMA or runtime.get("mode") != "production":
        raise RuntimeAssemblyError("runtime config must declare production schema/mode")
    artifacts = runtime.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RuntimeAssemblyError("production runtime config requires artifacts")
    root = runtime_path.parent
    emg_checkpoint = _resolve_required(root, artifacts.get("emg_checkpoint"), label="EMG checkpoint")
    emg_calibration = _resolve_required(root, artifacts.get("emg_calibration"), label="EMG calibration")
    planner_adapter = _resolve_required(
        root, artifacts.get("planner_adapter"), label="Planner LoRA adapter", directory=True
    )
    policy_identity_path = _resolve_required(
        root,
        artifacts.get("policy_server_identity_manifest"),
        label="policy server identity manifest",
    )
    hardware_path = _resolve_required(
        root, artifacts.get("hardware_calibration"), label="hardware calibration"
    )
    _, hardware = _load_json(hardware_path, label="hardware calibration")
    # Reject a missing/malformed policy-facing compatibility proof before
    # loading models or opening the live policy transport.  Exact equality is
    # checked again against the verified handshake identity below.
    _required_sha256(
        hardware.get("compatible_policy_tactile_profile_manifest_sha256"),
        label="hardware compatible_policy_tactile_profile_manifest_sha256",
    )

    emg = runtime.get("emg")
    policy = runtime.get("policy")
    camera = runtime.get("camera")
    if not all(isinstance(value, Mapping) for value in (emg, policy, camera)):
        raise RuntimeAssemblyError("runtime config requires emg/policy/camera objects")
    channel_order = tuple(str(value) for value in emg.get("channel_order", ()))
    if len(channel_order) != 8 or len(set(channel_order)) != 8:
        raise RuntimeAssemblyError("production EMG channel_order must contain eight unique names")
    model, emg_payload = load_emg_checkpoint(emg_checkpoint, map_location=str(emg.get("device", "cpu")))
    if emg_payload.get("schema_version") != "revo3-emg-checkpoint-v3":
        raise RuntimeAssemblyError("production requires a profile-bound EMG v3 checkpoint")
    profile = EmgPreprocessingProfile.from_mapping(emg_payload["preprocessing_profile"])
    calibration = load_emg_calibration(
        emg_calibration,
        base_checkpoint=emg_checkpoint,
        channel_order=channel_order,
        expected_subject_id=str(emg.get("subject_id", "")) or None,
        expected_day_id=str(emg.get("day_id", "")) or None,
    )
    classifier = StreamingEMGClassifier(
        model,
        emg_payload["normalization"],
        profile.sample_rate_hz,
        profile.window_samples,
        profile.stride_samples_pattern[0],
        device=str(emg.get("device", "cpu")),
        class_labels=model.config.resolved_labels(),
        calibration=calibration,
        preprocessing_profile=profile,
        channel_order=channel_order,
        expected_profile_fingerprint=profile.fingerprint,
    )

    planner_backend = Qwen3VLBackend(
        model_id=str(planner_cfg["base_model"]),
        revision=str(planner_cfg["base_revision"]),
        max_new_tokens=int(planner_cfg["max_new_tokens"]),
        adapter_path=planner_adapter,
        local_files_only=bool(runtime.get("local_files_only", False)),
        production=True,
    )
    endpoint = str(policy.get("endpoint", "")).strip()
    if not endpoint.startswith("tcp://"):
        raise RuntimeAssemblyError("production requires a tcp:// policy endpoint")
    _, policy_identity_payload = _load_json(
        policy_identity_path, label="policy server identity manifest"
    )
    try:
        expected_policy_identity = TReXServerIdentity.from_mapping(
            policy_identity_payload
        )
    except ValueError as exc:
        raise RuntimeAssemblyError("invalid policy server identity manifest") from exc
    if (
        expected_policy_identity.camera_profile != "revo3_full_center_v1"
        or expected_policy_identity.tactile_profile != "profile_a_force6d_diff"
        or expected_policy_identity.joint_order_hash != JOINT_ORDER_HASH
    ):
        raise RuntimeAssemblyError(
            "policy server identity is not the frozen Revo3 Profile-A mainline"
        )
    trex_backend = ZmqTReXBackend(
        endpoint=endpoint,
        timeout_ms=int(policy.get("timeout_ms", 5000)),
        image_profile="revo3_full_center_v1",
        tactile_profile="profile_a_force6d_diff",
        expected_server_identity=expected_policy_identity,
        slow_response_observation_budget_ns=executive_config.slow_response_observation_budget_ns,
        fast_response_observation_budget_ns=executive_config.fast_response_observation_budget_ns,
    )
    try:
        verified_policy_identity = trex_backend.probe_server_identity()
    except TReXZmqError as exc:
        trex_backend.close()
        raise RuntimeAssemblyError(
            "live T-Rex server identity handshake failed"
        ) from exc
    try:
        compatible_policy_tactile_profile_manifest_sha256 = (
            _validate_policy_tactile_profile_binding(
                hardware, verified_policy_identity
            )
        )
    except RuntimeAssemblyError:
        try:
            trex_backend.close()
        except BaseException:
            pass
        raise
    # RuntimeVersions is derived from the actual, manifest-verified server
    # response.  No client-authored policy_revision is trusted.
    policy_revision = verified_policy_identity.identity_sha256
    policy_worker = AsyncTReXPolicyRunner(
        TReXRevoPolicyAdapter(trex_backend),
        request_timeout_ns=_ms(policy.get("request_timeout_ms", 2000), name="request_timeout_ms"),
    )
    envelope, completion_config, servo_config, hardware_hash = _hardware_components(
        hardware_path, hardware
    )
    cair_cfg = hardware.get("cair", {})
    if not isinstance(cair_cfg, Mapping):
        raise RuntimeAssemblyError("hardware cair config must be an object")
    if bool(cair_cfg.get("enabled", False)):
        synergy_path = _resolve_required(
            root, artifacts.get("cair_synergy"), label="CAIR synergy artifact"
        )
        reflex_config = ReflexConfig.from_artifact(
            ClosingSynergyArtifact.load(synergy_path),
            baseline_median=_array5(cair_cfg.get("baseline_median"), name="baseline_median"),
            baseline_mad=_array5(cair_cfg.get("baseline_mad"), name="baseline_mad"),
            target_force=_array5(cair_cfg.get("target_force"), name="target_force"),
            protect_force=_array5(cair_cfg.get("protect_force"), name="protect_force"),
            hard_overload_force=_array5(
                cair_cfg.get("hard_overload_force"), name="hard_overload_force"
            ),
        )
    else:
        reflex_config = ReflexConfig(enabled=False, hardware_mode=True)
    servo = RevoServoExecutor(
        RevoCommandPipeline(bindings.revo_backend, SafetySupervisor(envelope)),
        policy_worker,
        TactileReflexPlugin(reflex_config),
        servo_config,
    )
    calibration_hash = str(camera.get("calibration_hash", "")).strip()
    tactile_profile_hash = str(hardware.get("tactile_profile_hash", "")).strip()
    if not calibration_hash or not tactile_profile_hash:
        raise RuntimeAssemblyError("camera/tactile calibration hashes are required")
    coordinator = OnlineV1Coordinator(
        planner_worker=AsyncPlannerWorker(AskToClarifyPlanner(planner_backend)),
        camera_health=CameraHealthMonitor(CameraHealthConfig(calibration_hash)),
        visual_gate=VisualGate(
            VisualGateConfig(max_decision_age_ns=executive_config.planner_ttl_ns)
        ),
        executive=TaskExecutive(executive_config),
        policy_worker=policy_worker,
        completion=CompletionMonitor(completion_config),
        servo=servo,
        versions=RuntimeVersions(
            CONTROL_SCHEMA,
            planner_backend.planner_revision,
            policy_revision,
            hardware_hash,
            JOINT_ORDER_HASH,
            tactile_profile_hash,
        ),
        emg_bridge=StreamingEMGEventBridge(classifier),
        planner_result_ttl_ns=executive_config.planner_ttl_ns,
        planner_sla_ns=executive_config.planner_sla_ns,
        planner_scene_max_distance=float(
            control["executive"].get("planner_scene_max_distance", 0.08)
        ),
    )
    if isinstance(planner_backend, MockPlannerBackend) or isinstance(trex_backend, MockTReXBackend):
        raise RuntimeAssemblyError("production assembly forbids Mock model backends")
    service = DoubleRateRuntimeService(
        coordinator,
        bindings.io,
        **_runtime_service_kwargs(control),
    )
    return RuntimeAssembly(
        "production",
        coordinator,
        service,
        {
            "mode": "production",
            "planner_revision": planner_backend.planner_revision,
            "policy_revision": policy_revision,
            "policy_checkpoint_sha256": verified_policy_identity.checkpoint_sha256,
            "policy_checkpoint_lineage_sha256": (
                verified_policy_identity.checkpoint_lineage_sha256
            ),
            "hardware_compatible_policy_tactile_profile_manifest_sha256": (
                compatible_policy_tactile_profile_manifest_sha256
            ),
            "verified_policy_tactile_profile_manifest_sha256": (
                verified_policy_identity.tactile_profile_manifest_sha256
            ),
            "policy_normalization_statistics_sha256": (
                verified_policy_identity.normalization_statistics_sha256
            ),
            "policy_normalization_artifact_sha256": (
                verified_policy_identity.normalization_artifact_sha256
            ),
            "hardware_manifest_sha256": hardware_hash,
            "max_new_tokens": planner_backend.max_new_tokens,
            "planner_sla_ms": executive_config.planner_sla_ns // 1_000_000,
            "planner_source_max_age_ms": (
                executive_config.planner_source_max_age_ns // 1_000_000
            ),
            "planner_scene_max_distance": float(
                control["executive"].get("planner_scene_max_distance", 0.08)
            ),
            "pending_intent_ttl_ms": executive_config.pending_intent_ttl_ns // 1_000_000,
            "task_executive": asdict(executive_config),
        },
    )


__all__ = [
    "CONTROL_SCHEMA",
    "HARDWARE_CALIBRATION_SCHEMA",
    "ProductionBindings",
    "RUNTIME_SCHEMA",
    "RuntimeAssembly",
    "RuntimeAssemblyError",
    "SyntheticRuntimeIO",
    "build_production_runtime",
    "build_simulation_runtime",
    "load_control_config",
]
