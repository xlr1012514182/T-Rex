from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import threading
import time

import numpy as np
import pytest

from revo3_v1.revo import JOINT_ORDER_HASH
from revo3_teleop.cli.collect_hardware import main as collect_hardware_main
from revo3_teleop.contracts import CommandReceipt, NativeSample, SampleHeader
from revo3_teleop.hardware.collection import (
    AuxiliarySourceBinding,
    CollectionControlCycle,
    HardwareCollectionConfig,
    HardwareCollectionDependencies,
    HardwareCollectionOrchestrator,
    assess_hardware_collection_config,
    load_hardware_collection_dependencies,
)
from revo3_teleop.recording import CollectionSessionFault


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ready_config(tmp_path: Path, *, export_vla: bool = True) -> HardwareCollectionConfig:
    calibration = tmp_path / "fisheye.json"
    calibration.write_text("{}\n", encoding="utf-8")
    model_root = tmp_path / "force_models"
    fingers = ("thumb", "index", "middle", "ring", "pinky")
    serials = {finger: f"SN_{finger}" for finger in fingers}
    model_hashes: dict[str, str] = {}
    for finger, serial in serials.items():
        path = model_root / serial / f"{serial}.onnx.enc"
        path.parent.mkdir(parents=True)
        path.write_bytes((finger + "-fixture").encode("ascii"))
        model_hashes[finger] = _sha(path)
    config_path = tmp_path / "collection.json"
    value = {
        "schema_version": "revo3-hardware-collection-v1",
        "mode": "hardware_collection",
        "allow_hardware_connect": True,
        "allow_hardware_write": True,
        "episode": {
            "episode_id": "injected_episode",
            "task_id": "fixture-bottle-v1",
            "task_version": 1,
            "task": "bottle",
            "object_id": "bottle",
            "object_instance": "bottle-fixture-01",
            "operator": "fixture-operator",
            "collection_day": "synthetic-fixture",
            "grasp_primitive": "POWER_GRASP",
            "instruction_source": "manual_canonical",
            "instruction": "Grasp the centered bottle and hold it securely.",
            "duration_s": 0.16,
            "anchor_start_delay_s": 0.05,
        },
        "outputs": {
            "master_root": str(tmp_path / "master"),
            "vla_root": str(tmp_path / "vla"),
            "emg_review_root": str(tmp_path / "emg_review"),
            "export_vla_after_commit": export_vla,
        },
        "runtime": {
            "control_step_timeout_ms": 20,
            "max_command_latency_ms": 20,
            "shutdown_timeout_ms": 50,
            "safety_watchdog_budget_ms": 50,
            "camera_stream": "camera_rectified",
            "state_stream": "revo_state",
            "tactile_stream": "tactile",
            "anchor_streams": ["camera_rectified", "revo_state", "tactile"],
            "required_source_timeouts_ms": {
                "camera_rectified": 100,
                "revo_state": 100,
                "tactile": 100,
                "emg": 100,
            },
            "vla_camera_stream": "camera_rectified",
            "vla_state_stream": "revo_state",
            "vla_tactile_stream": "tactile",
        },
        "revo": {
            "state_stream": "revo_state",
            "expected_serial": "REVO_FIXTURE",
            "expected_probe_fingerprint": "1" * 64,
            "joint_order_hash": JOINT_ORDER_HASH,
            "q_min_rad": [-1.0] * 21,
            "q_max_rad": [1.0] * 21,
            "max_delta_rad": [0.1] * 21,
            "state_units_bench_verified": True,
            "u21vt_identity_verified": True,
            "tactile_zero_and_saturation_verified": True,
            "physical_estop_verified": True,
            "hold_path_verified": True,
            "limits_verified": True,
        },
        "camera": {
            "rectified_stream": "camera_rectified",
            "camera_profile_id": "revo3_full_center_v1",
            "expected_probe_fingerprint": "2" * 64,
            "calibration_file": str(calibration),
            "calibration_sha256": _sha(calibration),
        },
        "emg": {
            "stream": "emg",
            "expected_serial": "EMG_FIXTURE",
            "expected_discovery_fingerprint": "3" * 64,
        },
        "tactile": {
            "mode": "visiontouch_force6d",
            "stream": "tactile",
            "capture_profile": "force6d",
            "vla_tactile_profile": "ablation_force6d_only",
            "max_inter_finger_skew_ns": 50_000_000,
            "checkpoint_family_id": "fixture-revo-profile",
            "normalization_family_id": "fixture-revo-normalization",
            "capability_manifest_sha256": "5" * 64,
            "force_model_dir": str(model_root),
            "finger_serials": serials,
            "expected_model_sha256": model_hashes,
        },
        "glove": {"enabled": False},
        "tianji": {"enabled": False, "config_path": None},
        "assembly_factory": {
            "factory": "unused_fixture:factory",
            "expected_module_sha256": "4" * 64,
        },
    }
    config_path.write_text(json.dumps(value), encoding="utf-8")
    return HardwareCollectionConfig.from_json(config_path)


def _sample(stream: str, sequence: int, timestamp_ns: int) -> NativeSample:
    if stream == "camera_rectified":
        payload = {"rgb": np.full((8, 8, 3), sequence % 255, dtype=np.uint8)}
    elif stream == "revo_state":
        payload = {"q_rad": np.zeros(21, dtype=np.float32)}
    elif stream == "tactile":
        payload = {
            "features": np.zeros((5, 6), dtype=np.float32),
            "force6d_finger_timestamp_ns": np.full(5, timestamp_ns, dtype=np.int64),
        }
    elif stream == "emg":
        payload = {"signal": np.zeros((8, 20), dtype=np.float32)}
    else:
        payload = {"value": np.asarray([sequence], dtype=np.float32)}
    return NativeSample(
        SampleHeader(
            source_id=f"fixture_{stream}",
            sequence=sequence,
            capture_timestamp_ns=timestamp_ns,
            receive_timestamp_ns=timestamp_ns,
        ),
        payload,
    )


def test_visiontouch_profile_and_skew_are_static_readiness_gates(tmp_path: Path) -> None:
    config = _ready_config(tmp_path)
    raw = json.loads(config.path.read_text(encoding="utf-8"))
    raw["tactile"].pop("capture_profile")
    raw["tactile"].pop("max_inter_finger_skew_ns")
    config.path.write_text(json.dumps(raw), encoding="utf-8")
    report = assess_hardware_collection_config(
        HardwareCollectionConfig.from_json(config.path)
    )
    assert "explicit_visiontouch_capture_profile_required" in report.blockers
    assert "approved_visiontouch_max_inter_finger_skew_ns_required" in report.blockers


def test_profile_b_readiness_requires_diff_only_and_no_force_model_claims(
    tmp_path: Path,
) -> None:
    config = _ready_config(tmp_path)
    raw = json.loads(config.path.read_text(encoding="utf-8"))
    raw["tactile"].update(
        capture_profile="diff_only",
        vla_tactile_profile="profile_b_diff_only",
        force_model_dir=None,
        expected_model_sha256={},
    )
    config.path.write_text(json.dumps(raw), encoding="utf-8")
    report = assess_hardware_collection_config(
        HardwareCollectionConfig.from_json(config.path)
    )
    assert report.execute_ready, report.blockers

    raw["tactile"]["vla_tactile_profile"] = "profile_a_force6d_diff"
    config.path.write_text(json.dumps(raw), encoding="utf-8")
    report = assess_hardware_collection_config(
        HardwareCollectionConfig.from_json(config.path)
    )
    assert "visiontouch_capture_and_vla_tactile_profile_mismatch" in report.blockers


class _VirtualClock:
    def __init__(self, now_ns: int = 1_000_000_000) -> None:
        self.now_ns = now_ns

    def __call__(self) -> int:
        return self.now_ns

    async def sleep(self, delay_s: float) -> None:
        self.now_ns += max(1, int(round(delay_s * 1_000_000_000.0)))
        # Yield exactly once without depending on wall-clock load.
        await asyncio.sleep(0)

    async def wait_tick(self, delay_s: float) -> None:
        observed = self.now_ns
        while self.now_ns <= observed:
            await asyncio.sleep(0)

    def advance(self, nanoseconds: int = 1) -> None:
        self.now_ns += max(1, int(nanoseconds))


class _FakeSensorRunner:
    def __init__(
        self,
        session,
        events: list[str],
        *,
        fail: bool = False,
        clock=time.monotonic_ns,
        sleep=asyncio.sleep,
    ) -> None:
        self.session = session
        self.events = events
        self.fail = fail
        self.clock = clock
        self.sleep = sleep
        self.stop_requested = False
        self.sequences = {name: 0 for name in ("camera_rectified", "revo_state", "tactile", "emg")}
        self.last_ns = 0

    async def run(self, *, duration_s=None) -> None:
        self.events.extend(["camera_start", "emg_start", "revo_start", "tactile_start"])
        start_ns = int(self.clock())
        duration_ns = int(round(float(duration_s) * 1_000_000_000.0))
        try:
            while not self.stop_requested and int(self.clock()) - start_ns < duration_ns:
                timestamp = max(int(self.clock()), self.last_ns + 1)
                self.last_ns = timestamp
                for stream in self.sequences:
                    sequence = self.sequences[stream]
                    self.sequences[stream] += 1
                    self.session.accept_sample(stream, _sample(stream, sequence, timestamp))
                if self.fail and self.sequences["camera_rectified"] >= 4:
                    raise RuntimeError("injected sensor failure")
                await self.sleep(0.004)
        finally:
            self.events.extend(["camera_stop", "emg_stop", "revo_stop", "tactile_stop"])

    def request_stop(self) -> None:
        self.stop_requested = True
        advance = getattr(self.clock, "advance", None)
        if callable(advance):
            advance()


class _FakeControlDriver:
    def __init__(self, *, fail_after: int | None = None, clock=time.monotonic_ns) -> None:
        self.sequence = 0
        self.fail_after = fail_after
        self.clock = clock

    async def step(self, anchor_timestamp_ns: int) -> CollectionControlCycle:
        if self.fail_after is not None and self.sequence >= self.fail_after:
            raise RuntimeError("injected control failure")
        decision = max(int(self.clock()), anchor_timestamp_ns)
        sequence = self.sequence
        self.sequence += 1
        target = np.full(21, 0.01 * sequence, dtype=np.float32)
        receipt = CommandReceipt(
            request_id=f"hand-{sequence}",
            component="revo_hand",
            accepted=True,
            requested_target=target,
            authorized_target=target,
            exact_sent_target=target,
            decision_timestamp_ns=decision,
            write_timestamp_ns=decision,
            controller_sequence=sequence,
            reason="fixture_exact_sent",
            unit="rad",
            joint_order_hash=JOINT_ORDER_HASH,
        )
        return CollectionControlCycle(
            receipt,
            phase="precontact",
            policy_loss_eligible=True,
        )


class _HangingSensorRunner:
    def __init__(self, session, events: list[str], clock) -> None:
        self.session = session
        self.events = events
        self.clock = clock
        self.stop_requested = False

    async def run(self, *, duration_s=None) -> None:
        timestamp = int(self.clock())
        self.events.append("hanging_sensor_start")
        for stream in ("camera_rectified", "revo_state", "tactile", "emg"):
            self.session.accept_sample(stream, _sample(stream, 0, timestamp))
        await asyncio.Event().wait()

    def request_stop(self) -> None:
        self.stop_requested = True
        self.events.append("hanging_sensor_stop_requested")


class _AuxSource:
    def __init__(
        self,
        name: str,
        events: list[str],
        *,
        fail_start: bool = False,
        fail_stop: bool = False,
    ) -> None:
        self.name = name
        self.events = events
        self.fail_start = fail_start
        self.fail_stop = fail_stop

    def start(self) -> None:
        self.events.append(f"{self.name}_start")
        if self.fail_start:
            raise RuntimeError(f"{self.name} partial start failed")

    def drain(self):
        return ()

    def stop(self) -> None:
        self.events.append(f"{self.name}_stop")
        if self.fail_stop:
            raise RuntimeError(f"{self.name} stop failed")


def _dependencies(
    events: list[str],
    *,
    sensor_fail: bool = False,
    control_fail_after: int | None = None,
    auxiliary: tuple[AuxiliarySourceBinding, ...] = (),
    clock=time.monotonic_ns,
    sleep=asyncio.sleep,
) -> HardwareCollectionDependencies:
    return HardwareCollectionDependencies(
        sensor_runner_factory=lambda session: _FakeSensorRunner(
            session, events, fail=sensor_fail, clock=clock, sleep=sleep
        ),
        control_driver=_FakeControlDriver(fail_after=control_fail_after, clock=clock),
        stop_targets=lambda: events.append("stop_targets"),
        revo_hold=lambda reason: events.append(f"revo_hold:{reason}"),
        tianji_soft_stop=lambda reason: events.append(f"tianji_soft_stop:{reason}"),
        flush=lambda: events.append("flush"),
        close=lambda: events.append("close"),
        abort_construction=lambda: events.append("abort_construction"),
        assert_disconnected=lambda: None,
        auxiliary_sources=auxiliary,
        metadata={"injected_fixture": True},
    )


def test_injected_hardware_collection_exports_only_exact_sent_vla_and_is_causal(
    tmp_path: Path,
) -> None:
    config = _ready_config(tmp_path)
    readiness = assess_hardware_collection_config(config)
    assert readiness.execute_ready, readiness.blockers
    events: list[str] = []
    scheduler = _VirtualClock()
    result = asyncio.run(
        HardwareCollectionOrchestrator(
            config,
            _dependencies(events, clock=scheduler, sleep=scheduler.wait_tick),
            readiness=readiness,
            clock=scheduler,
            sleep=scheduler.sleep,
            source_sleep=scheduler.wait_tick,
        ).run()
    )
    assert result.vla_episode is not None
    assert (result.master_episode / "streams/emg").is_dir()
    assert not (result.vla_episode / "streams").exists()
    vla_meta = json.loads((result.vla_episode / "meta.json").read_text(encoding="utf-8"))
    # The existing Revo episode schema names this semantic
    # ``controller_target``; the master receipt is the evidence that the
    # serialized value came from exact_sent_target rather than a request.
    assert vla_meta["action_label_source"] == "controller_target"
    assert vla_meta["contains_emg"] is False
    receipts = [
        json.loads(line)
        for line in (result.master_episode / "command_receipts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert receipts and all(
        row["accepted"] and row["exact_sent_target"] is not None for row in receipts
    )
    anchors = [
        json.loads(line)
        for line in (result.master_episode / "anchors_30hz.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(anchors) == result.anchors >= 2
    for anchor in anchors:
        assert all(
            reference["capture_timestamp_ns"] <= anchor["timestamp_ns"]
            for reference in anchor["streams"].values()
        )
    assert events.index("stop_targets") < events.index("close")
    assert events.index("revo_hold:normal_stop") < events.index("close")
    assert events.index("tianji_soft_stop:normal_stop") < events.index("close")


def test_control_failure_holds_soft_stops_aborts_and_stops_every_source(
    tmp_path: Path,
) -> None:
    config = _ready_config(tmp_path, export_vla=False)
    readiness = assess_hardware_collection_config(config)
    events: list[str] = []
    scheduler = _VirtualClock()
    first = _AuxSource("glove", events, fail_stop=True)
    second = _AuxSource("wrist", events)
    bindings = (
        AuxiliarySourceBinding("glove", first, lambda sample: "glove"),
        AuxiliarySourceBinding("wrist", second, lambda sample: "wrist"),
    )
    with pytest.raises(CollectionSessionFault, match="injected control failure"):
        asyncio.run(
            HardwareCollectionOrchestrator(
                config,
                _dependencies(
                    events,
                    control_fail_after=1,
                    auxiliary=bindings,
                    clock=scheduler,
                    sleep=scheduler.wait_tick,
                ),
                readiness=readiness,
                clock=scheduler,
                sleep=scheduler.sleep,
                source_sleep=scheduler.wait_tick,
            ).run()
        )
    assert "glove_stop" in events and "wrist_stop" in events
    assert all(
        name in events
        for name in ("camera_stop", "emg_stop", "revo_stop", "tactile_stop")
    )
    assert "stop_targets" in events
    assert any(value.startswith("revo_hold:backend_fault") for value in events)
    assert any(value.startswith("tianji_soft_stop:backend_fault") for value in events)
    quarantine = tmp_path / "master/quarantine/injected_episode.aborted"
    assert quarantine.is_dir()
    assert not (tmp_path / "master/committed/injected_episode").exists()


def test_auxiliary_partial_start_is_still_stopped_and_quarantined(tmp_path: Path) -> None:
    config = _ready_config(tmp_path, export_vla=False)
    readiness = assess_hardware_collection_config(config)
    events: list[str] = []
    scheduler = _VirtualClock()
    partial = _AuxSource("glove_ipc", events, fail_start=True)
    binding = AuxiliarySourceBinding("glove_ipc", partial, lambda sample: "glove_flex")
    with pytest.raises(CollectionSessionFault, match="partial start failed"):
        asyncio.run(
            HardwareCollectionOrchestrator(
                config,
                _dependencies(
                    events,
                    auxiliary=(binding,),
                    clock=scheduler,
                    sleep=scheduler.wait_tick,
                ),
                readiness=readiness,
                clock=scheduler,
                sleep=scheduler.sleep,
                source_sleep=scheduler.wait_tick,
            ).run()
        )
    assert "glove_ipc_start" in events
    assert "glove_ipc_stop" in events
    assert (tmp_path / "master/quarantine/injected_episode.aborted").is_dir()


def test_checked_in_default_config_audits_without_loading_or_connecting(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import revo3_teleop.cli.collect_hardware as cli

    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("default audit must not load an assembly")

    monkeypatch.setattr(cli, "load_hardware_collection_dependencies", forbidden)
    config = Path(__file__).parents[1] / "configs/hardware_collection.example.json"
    assert collect_hardware_main(["--config", str(config)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["execute_ready"] is False
    assert report["default_action"] == "audit_only_no_import_no_connect_no_write"
    assert not called


def test_orchestrator_refuses_blocked_config_before_factory_or_source_use(
    tmp_path: Path,
) -> None:
    source = Path(__file__).parents[1] / "configs/hardware_collection.example.json"
    config_path = tmp_path / "blocked.json"
    config_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    config = HardwareCollectionConfig.from_json(config_path)
    readiness = assess_hardware_collection_config(config)
    events: list[str] = []
    with pytest.raises(PermissionError, match="readiness has blockers"):
        HardwareCollectionOrchestrator(
            config,
            _dependencies(events),
            readiness=readiness,
        )
    assert events == []


def test_brainco_glove_and_emg_same_process_is_always_blocked(tmp_path: Path) -> None:
    config = _ready_config(tmp_path)
    raw = json.loads(config.path.read_text(encoding="utf-8"))
    raw["glove"].update(
        {
            "enabled": True,
            "acquisition_mode": "same_process_libedu",
            "ipc_clock_domain": "workstation_monotonic",
            "source_factory": "fixture:source",
            "expected_source_module_sha256": "5" * 64,
            "retarget_factory": "fixture:retarget",
            "expected_retarget_module_sha256": "6" * 64,
            "calibration_revision": "fixture-v1",
            "calibration_file": raw["camera"]["calibration_file"],
            "calibration_sha256": raw["camera"]["calibration_sha256"],
        }
    )
    config.path.write_text(json.dumps(raw), encoding="utf-8")
    report = assess_hardware_collection_config(
        HardwareCollectionConfig.from_json(config.path)
    )
    assert "brainco_glove_and_emg_cannot_share_libedu_process" in report.blockers


def test_nested_output_roots_are_not_physically_separate(tmp_path: Path) -> None:
    config = _ready_config(tmp_path)
    raw = json.loads(config.path.read_text(encoding="utf-8"))
    raw["outputs"]["vla_root"] = str(Path(raw["outputs"]["master_root"]) / "derived")
    config.path.write_text(json.dumps(raw), encoding="utf-8")
    report = assess_hardware_collection_config(
        HardwareCollectionConfig.from_json(config.path)
    )
    assert "master_vla_emg_output_roots_must_not_be_nested" in report.blockers
    assert "vla_and_emg_derived_outputs_are_physically_separate" not in report.evidence


def test_pressure_only_mode_blocks_vla_export_before_episode_start(tmp_path: Path) -> None:
    config = _ready_config(tmp_path)
    raw = json.loads(config.path.read_text(encoding="utf-8"))
    raw["tactile"] = {
        "mode": "u21vt_pressure",
        "pressure_projection_revision": "bench-pressure-v1",
        "pressure_projection_bench_verified": True,
    }
    raw["outputs"]["export_vla_after_commit"] = True
    config.path.write_text(json.dumps(raw), encoding="utf-8")
    report = assess_hardware_collection_config(
        HardwareCollectionConfig.from_json(config.path)
    )
    assert "u21vt_pressure_requires_export_vla_after_commit_false" in report.blockers


def test_runtime_stream_contract_blocks_duplicates_missing_emg_and_vla_mismatch(
    tmp_path: Path,
) -> None:
    config = _ready_config(tmp_path)
    raw = json.loads(config.path.read_text(encoding="utf-8"))
    raw["runtime"]["anchor_streams"] = [
        "camera_rectified",
        "revo_state",
        "revo_state",
    ]
    del raw["runtime"]["required_source_timeouts_ms"]["emg"]
    raw["runtime"]["vla_tactile_stream"] = "unrecorded_touch"
    raw["runtime"]["control_step_timeout_ms"] = 200
    raw["runtime"]["safety_watchdog_budget_ms"] = 100
    config.path.write_text(json.dumps(raw), encoding="utf-8")
    report = assess_hardware_collection_config(
        HardwareCollectionConfig.from_json(config.path)
    )
    assert "anchor_streams_must_be_unique" in report.blockers
    assert "emg_required_source_timeout_required" in report.blockers
    assert "vla_streams_must_match_runtime_anchor_roles" in report.blockers
    assert "all_vla_streams_must_be_anchor_streams" in report.blockers
    assert "all_vla_streams_require_timeouts" in report.blockers
    assert "control_step_timeout_exceeds_safety_watchdog_budget" in report.blockers
    assert "control_step_timeout_exceeds_30hz_anchor_period" in report.blockers


def test_required_external_glove_streams_need_explicit_timeouts(tmp_path: Path) -> None:
    config = _ready_config(tmp_path)
    raw = json.loads(config.path.read_text(encoding="utf-8"))
    raw["glove"].update(
        {
            "enabled": True,
            "acquisition_mode": "external_timestamped_ipc",
            "ipc_clock_domain": "shared_monotonic_fixture",
            "ipc_streams": ["glove_flex", "glove_imu", "glove_mag"],
            "required_for_episode": True,
            "source_factory": "fixture:source",
            "expected_source_module_sha256": "5" * 64,
            "retarget_factory": "fixture:retarget",
            "expected_retarget_module_sha256": "6" * 64,
            "calibration_revision": "fixture-v1",
            "calibration_file": raw["camera"]["calibration_file"],
            "calibration_sha256": raw["camera"]["calibration_sha256"],
        }
    )
    config.path.write_text(json.dumps(raw), encoding="utf-8")
    report = assess_hardware_collection_config(
        HardwareCollectionConfig.from_json(config.path)
    )
    assert "required_glove_ipc_streams_need_timeouts" in report.blockers


def test_hashed_assembly_factory_receives_validated_kwargs_and_must_be_disconnected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import revo3_teleop.hardware.collection as collection

    config = _ready_config(tmp_path)
    raw = json.loads(config.path.read_text(encoding="utf-8"))
    raw["assembly_factory"]["kwargs"] = {"station": "bench-a", "revision": 7}
    config.path.write_text(json.dumps(raw), encoding="utf-8")
    config = HardwareCollectionConfig.from_json(config.path)
    readiness = assess_hardware_collection_config(config)
    events: list[str] = []

    def factory(observed_config, **kwargs):
        assert observed_config is config
        assert kwargs == {"station": "bench-a", "revision": 7}
        return _dependencies(events)

    class Provenance:
        hash_verified = True

    monkeypatch.setattr(
        collection,
        "resolve_hashed_callable",
        lambda *args, **kwargs: (factory, Provenance()),
    )
    dependencies = load_hardware_collection_dependencies(config, readiness)
    assert isinstance(dependencies, HardwareCollectionDependencies)
    assert events == []


def test_connected_factory_is_synchronously_unwound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import revo3_teleop.hardware.collection as collection

    config = _ready_config(tmp_path)
    readiness = assess_hardware_collection_config(config)
    events: list[str] = []
    dependencies = _dependencies(events)
    object.__setattr__(
        dependencies,
        "assert_disconnected",
        lambda: (_ for _ in ()).throw(RuntimeError("connected")),
    )

    class Provenance:
        hash_verified = True

    monkeypatch.setattr(
        collection,
        "resolve_hashed_callable",
        lambda *args, **kwargs: (lambda config, **kwargs: dependencies, Provenance()),
    )
    with pytest.raises(RuntimeError, match="construction was synchronously aborted"):
        load_hardware_collection_dependencies(config, readiness)
    assert events == ["abort_construction"]


def test_dependencies_require_explicit_close_and_disconnected_assertion() -> None:
    events: list[str] = []
    with pytest.raises(TypeError):
        HardwareCollectionDependencies(  # type: ignore[call-arg]
            sensor_runner_factory=lambda session: None,
            control_driver=_FakeControlDriver(),
            stop_targets=lambda: None,
            revo_hold=lambda reason: None,
            tianji_soft_stop=lambda reason: None,
            flush=lambda: None,
        )


def test_control_step_timeout_is_deterministic_and_quarantines(tmp_path: Path) -> None:
    config = _ready_config(tmp_path, export_vla=False)
    readiness = assess_hardware_collection_config(config)
    scheduler = _VirtualClock()
    events: list[str] = []
    dependencies = _dependencies(
        events,
        clock=scheduler,
        sleep=scheduler.wait_tick,
    )

    async def deterministic_timeout(awaitable, timeout_s):
        assert timeout_s == pytest.approx(0.02)
        close = getattr(awaitable, "close", None)
        if callable(close):
            close()
        raise asyncio.TimeoutError

    with pytest.raises(CollectionSessionFault, match="control_step_timeout"):
        asyncio.run(
            HardwareCollectionOrchestrator(
                config,
                dependencies,
                readiness=readiness,
                clock=scheduler,
                sleep=scheduler.sleep,
                source_sleep=scheduler.wait_tick,
                control_wait_for=deterministic_timeout,
            ).run()
        )
    assert any(value.startswith("revo_hold:backend_fault") for value in events)
    assert any(value.startswith("tianji_soft_stop:backend_fault") for value in events)
    assert (tmp_path / "master/quarantine/injected_episode.aborted").is_dir()


def test_sensor_runner_shutdown_timeout_requires_intervention_and_quarantine(
    tmp_path: Path,
) -> None:
    config = _ready_config(tmp_path, export_vla=False)
    raw = json.loads(config.path.read_text(encoding="utf-8"))
    raw["runtime"]["shutdown_timeout_ms"] = 1
    raw["runtime"]["required_source_timeouts_ms"] = {
        name: 1000 for name in raw["runtime"]["required_source_timeouts_ms"]
    }
    config.path.write_text(json.dumps(raw), encoding="utf-8")
    config = HardwareCollectionConfig.from_json(config.path)
    readiness = assess_hardware_collection_config(config)
    scheduler = _VirtualClock()
    events: list[str] = []
    dependencies = _dependencies(
        events,
        clock=scheduler,
        sleep=scheduler.wait_tick,
    )
    object.__setattr__(
        dependencies,
        "sensor_runner_factory",
        lambda session: _HangingSensorRunner(session, events, scheduler),
    )
    with pytest.raises(
        CollectionSessionFault,
        match="sensor_runner_shutdown_timeout_process_or_device_intervention_required",
    ):
        asyncio.run(
            HardwareCollectionOrchestrator(
                config,
                dependencies,
                readiness=readiness,
                clock=scheduler,
                sleep=scheduler.sleep,
                source_sleep=scheduler.wait_tick,
            ).run()
        )
    assert "hanging_sensor_stop_requested" in events
    assert any(value.startswith("revo_hold:backend_fault") for value in events)
    assert (tmp_path / "master/quarantine/injected_episode.aborted").is_dir()


def test_dependency_close_timeout_prevents_commit_and_records_intervention(
    tmp_path: Path,
) -> None:
    config = _ready_config(tmp_path, export_vla=False)
    raw = json.loads(config.path.read_text(encoding="utf-8"))
    raw["runtime"]["shutdown_timeout_ms"] = 1
    config.path.write_text(json.dumps(raw), encoding="utf-8")
    config = HardwareCollectionConfig.from_json(config.path)
    readiness = assess_hardware_collection_config(config)
    scheduler = _VirtualClock()
    events: list[str] = []
    dependencies = _dependencies(
        events,
        clock=scheduler,
        sleep=scheduler.wait_tick,
    )
    never = threading.Event()
    object.__setattr__(dependencies, "close", lambda: never.wait())
    with pytest.raises(CollectionSessionFault, match="flush_or_commit_failed"):
        asyncio.run(
            HardwareCollectionOrchestrator(
                config,
                dependencies,
                readiness=readiness,
                clock=scheduler,
                sleep=scheduler.sleep,
                source_sleep=scheduler.wait_tick,
            ).run()
        )
    quarantine = tmp_path / "master/quarantine/injected_episode.aborted"
    evidence = json.loads((quarantine / "cleanup_failure.json").read_text(encoding="utf-8"))
    assert evidence["clean_close"] is False
    assert evidence["intervention_required"] is True
    assert "dependencies_close_timeout" in evidence["detail"]
    assert not (tmp_path / "master/committed/injected_episode").exists()
