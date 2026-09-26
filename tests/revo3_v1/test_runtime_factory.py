from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from pathlib import Path
import queue
import threading

import numpy as np
import pytest

from scripts import revo3_v1_runtime as runtime_cli

from revo3_v1.emg import EMGPrimitive
from revo3_v1.planner import (
    PlannerContextBuffer,
    PlannerContextFrame,
    PlannerWorkItem,
    Qwen3VLBackend,
    SupportedTask,
)
from revo3_v1.executive import EmgEvent, EmgIntent
from revo3_v1.policy import TReXServerIdentity
from revo3_v1.revo import BrainCoSDKBackend, MockRevoBackend
from revo3_v1.runtime import (
    DoubleRateRuntimeService,
    ProductionBindings,
    RuntimeAssemblyError,
    SyntheticRuntimeIO,
    build_production_runtime,
    build_simulation_runtime,
    load_control_config,
    StreamingEMGEventBridge,
)
from revo3_v1.runtime.factory import (
    _hardware_components,
    _validate_policy_tactile_profile_binding,
)


ROOT = Path(__file__).resolve().parents[2]
CONTROL = ROOT / "config" / "revo3_v1_control.json"


def _server_identity(*, tactile_hash: str = "a" * 64) -> TReXServerIdentity:
    return TReXServerIdentity(
        checkpoint_sha256="1" * 64,
        model_config_sha256="2" * 64,
        training_args_sha256="3" * 64,
        checkpoint_lineage_sha256="4" * 64,
        normalization_statistics_sha256="5" * 64,
        normalization_artifact_sha256="6" * 64,
        checkpoint_family_id="checkpoint-family",
        normalization_family_id="normalization-family",
        tactile_profile_manifest_sha256=tactile_hash,
        capability_manifest_sha256="7" * 64,
        split_manifest_sha256="8" * 64,
        tactile_profile="profile_a_force6d_diff",
        camera_profile="revo3_full_center_v1",
        joint_order_hash="9" * 64,
        training_stage="sft",
    )


def _ready_brainco_backend(**overrides):
    values = {
        "allow_hardware_write": True,
        "capability_probe_confirmed": True,
        "temperature_telemetry_verified": True,
        "soft_stop_callback": lambda client, slave_id, reason: None,
        "soft_stop_capability_verified": True,
        "collision_profile_id": "bench-collision-v1",
        "collision_profile_verified": True,
    }
    values.update(overrides)
    return BrainCoSDKBackend(object(), slave_id=1, **values)


class _HardwareBackend:
    is_hardware = True
    production_ready = True

    async def read_state(self):  # pragma: no cover - assembly boundary only
        raise AssertionError("not used")

    async def write_command(self, command):  # pragma: no cover
        del command

    async def collision_active(self):  # pragma: no cover
        return False

    async def soft_stop(self, reason):  # pragma: no cover
        del reason

    async def close(self):
        return True


class _HardwareIO:
    is_hardware = True
    raw_emg_only = True

    async def poll_emg_packet(self, *, now_ns):  # pragma: no cover
        del now_ns
        return None

    async def control_input(self, *, now_ns):  # pragma: no cover
        del now_ns
        raise AssertionError("not used")

    async def servo_input(self, *, now_ns, latest_control):  # pragma: no cover
        del now_ns, latest_control
        raise AssertionError("not used")

    async def close(self):
        return None


class _SlowSyntheticIO(SyntheticRuntimeIO):
    async def control_input(self, *, now_ns):
        await asyncio.sleep(0.020)
        return await super().control_input(now_ns=now_ns)


class _HungAfterStartIO(SyntheticRuntimeIO):
    def __init__(self, *args, hang_after=18, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_calls = 0
        self.hang_after = hang_after
        self.never = asyncio.Event()

    async def control_input(self, *, now_ns):
        self.control_calls += 1
        if self.control_calls >= self.hang_after:
            await self.never.wait()
        return await super().control_input(now_ns=now_ns)


class _DirectEmgBypassIO(SyntheticRuntimeIO):
    async def control_input(self, *, now_ns):
        value = await super().control_input(now_ns=now_ns)
        return replace(
            value,
            emg=EmgEvent(
                EmgIntent.POWER_GRASP,
                now_ns,
                confidence=1.0,
                margin=1.0,
                signal_quality=1.0,
                event_id="forged-direct-event",
            ),
        )


class _FailingSoftStopBackend(MockRevoBackend):
    async def soft_stop(self, reason):
        del reason
        raise RuntimeError("simulated stop transport failure")


class _TimedOutCloseBackend(MockRevoBackend):
    timed_out = True
    intervention_required = True

    async def close(self):
        self.closed = True
        return False


class _UncleanBackendClose(MockRevoBackend):
    async def close(self):
        return False


class _SlowWriteBackend(MockRevoBackend):
    def __init__(self):
        super().__init__()
        self.in_write = asyncio.Event()
        self.write_active = False
        self.stop_overlapped_write = False

    async def write_command(self, command):
        self.write_active = True
        self.in_write.set()
        try:
            await asyncio.sleep(0.050)
            await super().write_command(command)
        finally:
            self.write_active = False

    async def soft_stop(self, reason):
        self.stop_overlapped_write = self.stop_overlapped_write or self.write_active
        await super().soft_stop(reason)


class _HungServoInputIO(SyntheticRuntimeIO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entered = asyncio.Event()
        self.cancelled_and_unwound = asyncio.Event()
        self.never = asyncio.Event()

    async def servo_input(self, *, now_ns, latest_control):
        del now_ns, latest_control
        self.entered.set()
        try:
            await self.never.wait()
        finally:
            self.cancelled_and_unwound.set()


class _ErrorServoInputIO(SyntheticRuntimeIO):
    async def servo_input(self, *, now_ns, latest_control):
        del now_ns, latest_control
        raise OSError("fixture servo acquisition failure")


class _DelayedServoInputIO(SyntheticRuntimeIO):
    async def servo_input(self, *, now_ns, latest_control):
        await asyncio.sleep(0.001)
        return await super().servo_input(
            now_ns=now_ns, latest_control=latest_control
        )


class _HungCloseIO(SyntheticRuntimeIO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.close_cancelled_and_unwound = asyncio.Event()
        self.never = asyncio.Event()

    async def close(self):
        try:
            await self.never.wait()
        finally:
            self.close_cancelled_and_unwound.set()


class _CollisionAfterFirstWriteBackend(MockRevoBackend):
    async def collision_active(self):
        return bool(self.commands)


class _CollisionWithUnconfirmedStopBackend(_CollisionAfterFirstWriteBackend):
    async def soft_stop(self, reason):
        del reason
        raise RuntimeError("fixture SoftStop transport failure")


class _RepeatingTaskEMGSource:
    """Two-cycle raw-packet classifier fixture for terminal-ack integration."""

    def __init__(self):
        self.packet = 0
        self.started = False
        self.released = False
        self.start_events = 0
        self.reset_events = 0

    def push_many(self, samples, sample_timestamps_ns, signal_quality=1.0):
        del samples
        self.packet += 1
        timestamp_ns = int(np.asarray(sample_timestamps_ns, dtype=np.int64)[-1])
        common = {
            "confidence": 0.99,
            "margin": 0.9,
            "signal_quality": float(signal_quality),
            "timestamp_ns": timestamp_ns,
        }
        if not self.started:
            self.started = True
            self.start_events += 1
            return [{
                **common,
                "primitive": "POWER_GRASP",
                "event": {
                    **common,
                    "type": "StartIntentEvent",
                    "primitive": "POWER_GRASP",
                    "event_id": f"repeat-start-{self.start_events}",
                },
            }]
        if not self.released and self.packet >= 25:
            self.released = True
            return [{
                **common,
                "primitive": "RELEASE",
                "event": {
                    **common,
                    "type": "ReleaseEvent",
                    "primitive": "RELEASE",
                    "event_id": f"repeat-release-{self.start_events}",
                },
            }]
        return [{**common, "primitive": "REST", "event": None}]

    def reset(self, active=False):
        assert not active
        self.reset_events += 1
        self.packet = 0
        self.started = False
        self.released = False


class _FailFirstCloseQueue:
    def __init__(self, delegate):
        self.delegate = delegate
        self.failed = False

    def get(self):
        return self.delegate.get()

    def put_nowait(self, value):
        if value is None and not self.failed:
            self.failed = True
            raise queue.Full
        return self.delegate.put_nowait(value)

    def empty(self):
        return self.delegate.empty()


class _BlockingPlanner:
    def __init__(self, delegate, entered, release):
        self.delegate = delegate
        self.entered = entered
        self.release = release

    def plan(self, request):
        self.entered.set()
        if not self.release.wait(timeout=2.0):
            raise TimeoutError("fixture Planner release timeout")
        return self.delegate.plan(request)


def test_strict_control_config_resolves_all_frozen_runtime_values():
    _, payload, config = load_control_config(CONTROL, strict_production=True)
    assert payload["planner"]["max_new_tokens"] == 384
    assert config.planner_sla_ns == 20_000_000_000
    assert config.planner_source_max_age_ns == 18_000_000_000
    assert config.pending_intent_ttl_ns == 22_000_000_000
    assert payload["executive"]["planner_scene_max_distance"] == pytest.approx(0.08)
    assert config.policy_ttl_ns == 750_000_000
    assert config.lease_ttl_ns == 3_000_000_000
    assert payload["runtime"] == {
        "control_watchdog_ms": 100,
        "servo_io_timeout_ms": 50,
        "io_close_timeout_ms": 1000,
    }


def test_cli_defaults_to_continuous_production_and_bounded_simulation():
    assert runtime_cli._effective_servo_ticks("production", None) is None
    assert runtime_cli._effective_servo_ticks("simulation", None) == 120
    assert runtime_cli._effective_servo_ticks("production", 7) == 7


def test_runtime_rejects_servo_io_deadline_longer_than_control_watchdog(tmp_path):
    payload = json.loads(CONTROL.read_text(encoding="utf-8"))
    payload["runtime"]["servo_io_timeout_ms"] = 101
    path = tmp_path / "control.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeAssemblyError, match="must not exceed"):
        load_control_config(path, strict_production=True)


def test_strict_control_config_rejects_missing_critical_key(tmp_path):
    payload = json.loads(CONTROL.read_text(encoding="utf-8"))
    del payload["executive"]["policy_ttl_ms"]
    path = tmp_path / "control.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeAssemblyError, match="policy_ttl_ms"):
        load_control_config(path, strict_production=True)


def test_production_rejects_128_token_truncation_before_loading_weights(tmp_path):
    payload = json.loads(CONTROL.read_text(encoding="utf-8"))
    payload["planner"]["max_new_tokens"] = 128
    path = tmp_path / "control.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeAssemblyError, match="max_new_tokens=384"):
        load_control_config(path, strict_production=True)
    with pytest.raises(ValueError, match="max_new_tokens=384"):
        Qwen3VLBackend(
            max_new_tokens=128,
            production=True,
            adapter_path=tmp_path,
        )


def test_production_bindings_reject_mock_synthetic_and_direct_emg_capability():
    with pytest.raises(RuntimeAssemblyError, match="non-Mock"):
        ProductionBindings(MockRevoBackend(), _HardwareIO())
    sim = MockRevoBackend()
    with pytest.raises(RuntimeAssemblyError, match="non-synthetic"):
        ProductionBindings(
            _HardwareBackend(),
            SyntheticRuntimeIO(sim, calibration_hash="fixture"),
        )
    direct = _HardwareIO()
    direct.raw_emg_only = False
    with pytest.raises(RuntimeAssemblyError, match="raw_emg_only"):
        ProductionBindings(_HardwareBackend(), direct)


@pytest.mark.parametrize(
    "hardware,error",
    [
        ({}, "must be a lowercase 64-character"),
        (
            {"compatible_policy_tactile_profile_manifest_sha256": "A" * 64},
            "must be a lowercase 64-character",
        ),
        (
            {"compatible_policy_tactile_profile_manifest_sha256": "b" * 64},
            "manifest SHA-256 mismatch",
        ),
    ],
)
def test_policy_tactile_profile_binding_rejects_missing_invalid_and_same_name_wrong_hash(
    hardware, error
):
    # The identity deliberately retains the exact same human-readable profile
    # name in every case; only byte identity is compatibility proof.
    with pytest.raises(RuntimeAssemblyError, match=error):
        _validate_policy_tactile_profile_binding(hardware, _server_identity())


def test_policy_tactile_profile_binding_accepts_exact_verified_manifest_hash():
    digest = "a" * 64
    assert (
        _validate_policy_tactile_profile_binding(
            {"compatible_policy_tactile_profile_manifest_sha256": digest},
            _server_identity(tactile_hash=digest),
        )
        == digest
    )


def test_hardware_schema_and_example_expose_binding_but_example_cannot_arm():
    schema = json.loads(
        (ROOT / "config" / "revo3_v1_hardware_calibration.schema.json").read_text(
            encoding="utf-8"
        )
    )
    required = set(schema["required"])
    assert "compatible_policy_tactile_profile_manifest_sha256" in required
    example_path = ROOT / "config" / "revo3_v1_hardware_calibration.example.json"
    example = json.loads(example_path.read_text(encoding="utf-8"))
    assert len(example["compatible_policy_tactile_profile_manifest_sha256"]) == 64
    with pytest.raises(RuntimeAssemblyError, match="example-only"):
        _hardware_components(example_path, example)


@pytest.mark.parametrize(
    "missing",
    [
        "allow_hardware_write",
        "capability_probe_confirmed",
        "temperature_telemetry_verified",
        "soft_stop_capability_verified",
        "collision_profile_verified",
        "collision_profile_id",
    ],
)
def test_production_binding_rejects_brainco_when_any_arming_proof_is_missing(missing):
    backend = _ready_brainco_backend()
    setattr(backend, missing, "" if missing == "collision_profile_id" else False)
    try:
        assert not backend.production_ready
        with pytest.raises(RuntimeAssemblyError, match="production_ready=True"):
            ProductionBindings(backend, _HardwareIO())
    finally:
        asyncio.run(backend.close())


def test_production_binding_accepts_fully_verified_brainco_boundary():
    backend = _ready_brainco_backend()
    try:
        assert backend.production_ready
        bindings = ProductionBindings(backend, _HardwareIO())
        assert bindings.revo_backend is backend
    finally:
        asyncio.run(backend.close())


def test_production_missing_artifact_rejects_before_models_or_hardware_start(tmp_path):
    runtime = {
        "schema_version": "revo3-v1-runtime-v1",
        "mode": "production",
        "artifacts": {},
        "emg": {},
        "policy": {},
        "camera": {},
    }
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    bindings = ProductionBindings(_HardwareBackend(), _HardwareIO())
    with pytest.raises(RuntimeAssemblyError, match="EMG checkpoint"):
        build_production_runtime(CONTROL, runtime_path, bindings=bindings)


@pytest.mark.parametrize("task", tuple(SupportedTask))
def test_same_coordinator_scripted_simulation_crosses_start_ready_writer_and_release(task):
    async def run():
        assembly = build_simulation_runtime(CONTROL, task=task)
        backend = assembly.coordinator.servo.pipeline.backend
        shutdown = await assembly.service.run(max_servo_ticks=120)
        evidence = assembly.service.evidence
        assert "START" in evidence.executive_outputs
        assert "CONTINUE" in evidence.executive_outputs
        assert "explicit_release" in evidence.executive_reasons
        assert "CONTROLLED_OPEN" in evidence.motion_directives
        assert "READY" in evidence.policy_states
        assert evidence.servo_write_count > 0
        assert len(backend.commands) == evidence.servo_write_count
        assert shutdown.clean and shutdown.stop_succeeded
        assert shutdown.backend_clean
        assert not shutdown.backend_timed_out
        assert not shutdown.backend_intervention_required
        assert backend.closed

    asyncio.run(run())


def test_slow_control_acquisition_does_not_serialize_the_100hz_writer():
    async def run():
        assembly = build_simulation_runtime(CONTROL, task=SupportedTask.BOTTLE)
        backend = assembly.coordinator.servo.pipeline.backend
        slow_io = _SlowSyntheticIO(
            backend, calibration_hash="simulation-camera-calibration"
        )
        service = DoubleRateRuntimeService(assembly.coordinator, slow_io)
        shutdown = await service.run(max_servo_ticks=120)
        assert "START" in service.evidence.executive_outputs
        assert "READY" in service.evidence.policy_states
        assert service.evidence.servo_write_count > 0
        command_timestamps = np.asarray(
            [command.timestamp_ns for command in backend.commands], dtype=np.int64
        )
        assert np.all(np.diff(command_timestamps) >= service.servo_period_ns)
        assert shutdown.clean

    asyncio.run(run())


def test_finite_servo_endpoint_releases_imminent_deadline_waiter():
    async def run():
        assembly = build_simulation_runtime(CONTROL, task=SupportedTask.BOTTLE)
        service = assembly.service
        stop = asyncio.Event()
        service._run_stop = stop
        now_ns = service._clock_ns()
        service._next_servo_deadline_ns = now_ns
        service._completed_servo_deadline_ns = now_ns - 1

        waiter = asyncio.create_task(service._yield_to_imminent_servo_deadline())
        await asyncio.sleep(0)
        assert not waiter.done()

        # Mirrors _servo_loop reaching max_servo_ticks: the advertised next
        # phase is intentionally never executed, but run-stop must still join
        # the control-side priority wait without requiring service.close().
        stop.set()
        await asyncio.wait_for(waiter, timeout=0.1)
        service._run_stop = None
        shutdown = await service.close()
        assert shutdown.clean

    asyncio.run(run())


def test_control_heartbeat_timeout_stops_instead_of_reusing_old_lease():
    async def run():
        assembly = build_simulation_runtime(CONTROL, task=SupportedTask.BOTTLE)
        backend = assembly.coordinator.servo.pipeline.backend
        hung = _HungAfterStartIO(
            backend,
            calibration_hash="simulation-camera-calibration",
            hang_after=18,
        )
        service = DoubleRateRuntimeService(
            assembly.coordinator, hung, control_watchdog_ns=80_000_000
        )
        with pytest.raises(RuntimeError, match="control_heartbeat_timeout"):
            await service.run(max_servo_ticks=300)
        count_after_stop = len(backend.commands)
        await asyncio.sleep(0.030)
        assert len(backend.commands) == count_after_stop
        assert assembly.coordinator.servo.pipeline.soft_stop_confirmed

    asyncio.run(run())


def test_runtime_rejects_direct_emg_event_bypass_and_stops_without_writing():
    async def run():
        assembly = build_simulation_runtime(CONTROL, task=SupportedTask.BOTTLE)
        backend = assembly.coordinator.servo.pipeline.backend
        io = _DirectEmgBypassIO(
            backend, calibration_hash="simulation-camera-calibration"
        )
        service = DoubleRateRuntimeService(assembly.coordinator, io)
        with pytest.raises(RuntimeError, match="may not inject EMG events"):
            await service.run(max_servo_ticks=10)
        assert not backend.commands
        assert assembly.coordinator.servo.pipeline.soft_stop_confirmed

    asyncio.run(run())


def test_unconfirmed_soft_stop_makes_shutdown_unclean():
    async def run():
        assembly = build_simulation_runtime(CONTROL, task=SupportedTask.BOTTLE)
        failing = _FailingSoftStopBackend()
        assembly.coordinator.servo.pipeline.backend = failing
        shutdown = await assembly.service.close()
        assert not shutdown.stop_succeeded
        assert not shutdown.clean
        assert assembly.coordinator.servo.pipeline.hard_fault_latched
        assert not assembly.coordinator.servo.pipeline.soft_stop_confirmed

    asyncio.run(run())


def test_backend_timeout_or_unresolved_executor_makes_shutdown_unclean():
    async def run():
        assembly = build_simulation_runtime(CONTROL, task=SupportedTask.BOTTLE)
        timed_out = _TimedOutCloseBackend()
        assembly.coordinator.servo.pipeline.backend = timed_out
        shutdown = await assembly.service.close()
        assert shutdown.stop_succeeded
        assert not shutdown.backend_clean
        assert shutdown.backend_timed_out
        assert shutdown.backend_intervention_required
        assert not shutdown.clean
        assert timed_out.closed

    asyncio.run(run())


def test_backend_worker_close_failure_is_part_of_shutdown_cleanliness():
    async def run():
        assembly = build_simulation_runtime(CONTROL, task=SupportedTask.BOTTLE)
        backend = _UncleanBackendClose()
        assembly.coordinator.servo.pipeline.backend = backend
        shutdown = await assembly.service.close()
        assert shutdown.stop_succeeded
        assert not shutdown.backend_clean
        assert not shutdown.clean

    asyncio.run(run())


def test_external_runtime_cancellation_joins_writer_before_final_soft_stop():
    async def run():
        assembly = build_simulation_runtime(CONTROL, task=SupportedTask.BOTTLE)
        backend = _SlowWriteBackend()
        assembly.coordinator.servo.pipeline.backend = backend
        io = SyntheticRuntimeIO(
            backend, calibration_hash="simulation-camera-calibration"
        )
        service = DoubleRateRuntimeService(assembly.coordinator, io)
        running = asyncio.create_task(service.run(max_servo_ticks=300))
        await asyncio.wait_for(backend.in_write.wait(), timeout=2.0)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert not backend.stop_overlapped_write
        assert assembly.coordinator.servo.pipeline.soft_stop_confirmed

    asyncio.run(run())


def test_hung_servo_input_times_out_unwinds_and_never_writes_after_fault():
    async def run():
        assembly = build_simulation_runtime(CONTROL, task=SupportedTask.BOTTLE)
        backend = assembly.coordinator.servo.pipeline.backend
        io = _HungServoInputIO(
            backend, calibration_hash="simulation-camera-calibration"
        )
        service = DoubleRateRuntimeService(
            assembly.coordinator,
            io,
            control_watchdog_ns=80_000_000,
            servo_io_timeout_ns=20_000_000,
            io_close_timeout_ns=50_000_000,
        )
        with pytest.raises(RuntimeError, match="servo_input_timeout"):
            await asyncio.wait_for(service.run(max_servo_ticks=300), timeout=2.0)
        assert io.entered.is_set()
        assert io.cancelled_and_unwound.is_set()
        count_after_fault = len(backend.commands)
        await asyncio.sleep(0.030)
        assert len(backend.commands) == count_after_fault == 0
        assert assembly.coordinator.servo.pipeline.soft_stop_confirmed
        shutdown = await service.close()
        assert not shutdown.clean
        assert not shutdown.io_clean
        assert shutdown.io_timed_out
        assert shutdown.io_intervention_required
        assert shutdown.stop_succeeded

    asyncio.run(run())


def test_servo_input_error_softstops_and_marks_io_intervention():
    async def run():
        assembly = build_simulation_runtime(CONTROL, task=SupportedTask.BOTTLE)
        backend = assembly.coordinator.servo.pipeline.backend
        io = _ErrorServoInputIO(
            backend, calibration_hash="simulation-camera-calibration"
        )
        service = DoubleRateRuntimeService(assembly.coordinator, io)
        with pytest.raises(OSError, match="fixture servo acquisition failure"):
            await asyncio.wait_for(service.run(max_servo_ticks=20), timeout=2.0)
        assert not backend.commands
        assert assembly.coordinator.servo.pipeline.soft_stop_confirmed
        shutdown = await service.close()
        assert not shutdown.clean
        assert not shutdown.io_clean
        assert not shutdown.io_timed_out
        assert shutdown.io_intervention_required

    asyncio.run(run())


def test_delayed_servo_input_inside_explicit_deadline_completes_cleanly():
    async def run():
        assembly = build_simulation_runtime(CONTROL, task=SupportedTask.BOTTLE)
        backend = assembly.coordinator.servo.pipeline.backend
        io = _DelayedServoInputIO(
            backend, calibration_hash="simulation-camera-calibration"
        )
        service = DoubleRateRuntimeService(
            assembly.coordinator,
            io,
            servo_io_timeout_ns=50_000_000,
        )
        now_ns = 1_000_000_000
        await service.step_control(now_ns=now_ns)
        result = await service.step_servo(now_ns=now_ns)
        assert result.reason == "no_motion"
        shutdown = await service.close()
        assert shutdown.clean
        assert service.evidence.servo_write_count == 0

    asyncio.run(run())


def test_hung_runtime_io_close_is_bounded_and_truthfully_unclean():
    async def run():
        assembly = build_simulation_runtime(CONTROL, task=SupportedTask.BOTTLE)
        backend = assembly.coordinator.servo.pipeline.backend
        io = _HungCloseIO(
            backend, calibration_hash="simulation-camera-calibration"
        )
        service = DoubleRateRuntimeService(
            assembly.coordinator,
            io,
            io_close_timeout_ns=20_000_000,
        )
        shutdown = await asyncio.wait_for(service.close(), timeout=2.0)
        assert io.close_cancelled_and_unwound.is_set()
        assert shutdown.stop_succeeded
        assert not shutdown.clean
        assert not shutdown.io_clean
        assert shutdown.io_timed_out
        assert shutdown.io_intervention_required

    asyncio.run(run())


@pytest.mark.parametrize(
    "backend_type,expected_error,stop_confirmed,shutdown_clean",
    (
        (
            _CollisionAfterFirstWriteBackend,
            "servo_hard_fault_latched",
            True,
            True,
        ),
        (
            _CollisionWithUnconfirmedStopBackend,
            "servo_soft_stop_unconfirmed",
            False,
            False,
        ),
    ),
)
def test_servo_hard_fault_is_exposed_as_abort_and_exits_without_late_write(
    backend_type, expected_error, stop_confirmed, shutdown_clean
):
    async def run():
        assembly = build_simulation_runtime(CONTROL, task=SupportedTask.BOTTLE)
        backend = backend_type()
        assembly.coordinator.servo.pipeline.backend = backend
        io = SyntheticRuntimeIO(
            backend, calibration_hash="simulation-camera-calibration"
        )
        service = DoubleRateRuntimeService(assembly.coordinator, io)
        with pytest.raises(RuntimeError, match=expected_error):
            await asyncio.wait_for(service.run(max_servo_ticks=300), timeout=4.0)
        assert "ABORT" in service.evidence.executive_outputs
        count_after_fault = len(backend.commands)
        assert count_after_fault >= 1
        await asyncio.sleep(0.030)
        assert len(backend.commands) == count_after_fault
        assert assembly.coordinator.executive.phase.value == "FAULT_LATCHED"
        assert assembly.coordinator.servo.pipeline.soft_stop_confirmed is stop_confirmed
        shutdown = await service.close()
        assert shutdown.clean is shutdown_clean
        assert shutdown.backend_intervention_required is (not stop_confirmed)

    asyncio.run(run())


def test_complete_remains_observable_until_safe_ack_then_same_service_starts_again():
    async def wait_until(predicate, timeout=4.0):
        deadline = asyncio.get_running_loop().time() + timeout
        while not predicate():
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("timed out waiting for runtime state")
            await asyncio.sleep(0.005)

    async def run():
        assembly = build_simulation_runtime(CONTROL, task=SupportedTask.BOTTLE)
        source = _RepeatingTaskEMGSource()
        assembly.coordinator.emg_bridge = StreamingEMGEventBridge(source)
        service = assembly.service
        backend = assembly.coordinator.servo.pipeline.backend
        running = asyncio.create_task(service.run(max_servo_ticks=None))

        await wait_until(lambda: "COMPLETE" in service.evidence.executive_outputs)
        first_count = len(backend.commands)
        await asyncio.sleep(0.030)
        # Terminal COMPLETE persists and cannot silently restart on another
        # start edge before the supervisor's safe acknowledgement.
        assert assembly.coordinator.executive.phase.value == "COMPLETE"
        assert source.start_events == 1
        with pytest.raises(ValueError, match="safe_state_confirmed"):
            await service.acknowledge_terminal(safe_state_confirmed=False)

        await service.acknowledge_terminal(safe_state_confirmed=True)
        await wait_until(lambda: not service.terminal_ack_pending)
        await wait_until(lambda: source.start_events >= 2)
        await wait_until(lambda: len(backend.commands) > first_count)
        assert source.reset_events >= 1

        service.request_shutdown()
        shutdown = await asyncio.wait_for(running, timeout=4.0)
        assert shutdown.clean
        assert shutdown.stop_succeeded

    asyncio.run(run())


def test_continuous_service_request_shutdown_uses_confirmed_softstop_path():
    async def run():
        assembly = build_simulation_runtime(CONTROL, task=SupportedTask.BOTTLE)
        running = asyncio.create_task(assembly.service.run(max_servo_ticks=None))
        await asyncio.wait_for(assembly.service._control_ready.wait(), timeout=2.0)
        assembly.service.request_shutdown()
        shutdown = await asyncio.wait_for(running, timeout=4.0)
        assert shutdown.clean
        assert shutdown.stop_succeeded
        assert assembly.coordinator.servo.pipeline.soft_stop_confirmed

    asyncio.run(run())


def test_service_single_close_retries_worker_sentinels_until_release():
    assembly = build_simulation_runtime(CONTROL, task=SupportedTask.BOTTLE)
    planner_worker = assembly.coordinator.planner_worker
    entered, release = threading.Event(), threading.Event()
    planner_worker._planner = _BlockingPlanner(
        planner_worker._planner, entered, release
    )
    context = PlannerContextBuffer()
    for sequence, timestamp_ns in enumerate((100, 200, 300), 1):
        context.append(
            PlannerContextFrame(
                sequence,
                timestamp_ns,
                np.zeros((8, 8, 3), dtype=np.uint8),
                np.zeros((4, 4, 3), dtype=np.uint8),
            ),
            now_ns=300,
        )
    request = context.build_request(
        primitive=EMGPrimitive.POWER_GRASP,
        now_ns=300,
    )
    assert planner_worker.submit(
        PlannerWorkItem(
            1,
            "service-close-event",
            EMGPrimitive.POWER_GRASP,
            0,
            request,
            300,
        )
    )
    assert entered.wait(timeout=1.0)
    planner_worker._requests = _FailFirstCloseQueue(planner_worker._requests)
    policy_worker = assembly.coordinator.policy_worker
    policy_worker._jobs = _FailFirstCloseQueue(policy_worker._jobs)
    timer = threading.Timer(0.02, release.set)
    timer.start()
    try:
        shutdown = asyncio.run(assembly.service.close())
    finally:
        timer.cancel()
    assert shutdown.clean
    assert shutdown.planner_clean
    assert shutdown.policy_clean
