from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from revo3_teleop.backends import (  # noqa: E402
    CtypesMarvinClient,
    HistoricalMarvinAbiNotAcknowledged,
    TianjiMarvinBackend,
    TianjiSdkLoadError,
    TianjiSdkPluginSpec,
    load_tianji_sdk,
    resolve_hashed_callable,
)
import pytest
from revo3_teleop.hardware import (  # noqa: E402
    HardwareConfig,
    assemble_tianji_hardware,
    assess_hardware_config,
)
from revo3_teleop.mock import SyntheticTianjiNativeClient  # noqa: E402
from revo3_teleop.retargeting import IKSolution, WristPose6D  # noqa: E402


JOINTS = tuple(f"arm_joint_{index}" for index in range(7))


def test_hashed_callable_binds_reexport_to_defining_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    origin = tmp_path / "tianji_test_origin.py"
    shim = tmp_path / "tianji_test_shim.py"
    origin.write_text("def factory():\n    return object()\n", encoding="utf-8")
    shim.write_text("from tianji_test_origin import factory\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    origin_hash = hashlib.sha256(origin.read_bytes()).hexdigest()
    shim_hash = hashlib.sha256(shim.read_bytes()).hexdigest()
    _, provenance = resolve_hashed_callable(
        "tianji_test_shim:factory",
        expected_module_sha256=origin_hash,
        name="test re-export",
    )
    assert provenance.hash_verified
    assert provenance.module_name == "tianji_test_origin"
    assert provenance.module_path == str(origin.resolve())

    with pytest.raises(TianjiSdkLoadError, match="SHA-256 mismatch"):
        resolve_hashed_callable(
            "tianji_test_shim:factory",
            expected_module_sha256=shim_hash,
            name="test re-export",
        )


def make_test_tianji_client(library_path=None):
    client = SyntheticTianjiNativeClient()
    if library_path is not None:
        client.library_path = str(Path(library_path).resolve())
    return client


class TestPoseProvider:
    __test__ = False
    provides_wrist_pose = True

    def read_pose(self):
        return WristPose6D(
            source_id="test_tracker",
            source_frame="tracker_world",
            capture_timestamp_ns=1,
            receive_timestamp_ns=1,
            clock_domain="workstation_monotonic",
            position_m=np.zeros(3),
            quaternion_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0]),
            calibration_revision="tracker-cal-v1",
        )


def make_test_pose_provider():
    return TestPoseProvider()


class TestIK:
    __test__ = False

    def solve(self, target, seed_q_rad):
        return IKSolution(
            q_rad=np.asarray(seed_q_rad),
            joint_order=JOINTS,
            converged=True,
            position_residual_m=0.0,
            orientation_residual_rad=0.0,
            iterations=1,
            solver_revision="test-ik",
        )


def make_test_ik():
    return TestIK()


def test_checked_in_hardware_config_allows_hand_only_and_hard_blocks_tianji() -> None:
    root = Path(__file__).resolve().parents[1]
    config = HardwareConfig.from_json(root / "configs" / "hardware.example.json")

    report = assess_hardware_config(config)

    assert report.hand_collection_allowed
    assert not report.arm_planning_ready
    assert not report.arm_write_ready
    assert any("brainco_edu" in item for item in report.arm_planning_blockers)
    assert "allow_hardware_write_is_false" in report.arm_write_blockers
    assembly = assemble_tianji_hardware(config)
    assert assembly.backend is None
    assert assembly.runtime is None


def test_tianji_source_audit_never_claims_public_vendor_authority() -> None:
    root = Path(__file__).resolve().parents[1]
    value = json.loads((root / "tianji_sources.lock.json").read_text(encoding="utf-8"))

    assert value["discovery_cutoff"] == "2026-08-17"
    assert value["authoritative_vendor_sdk_found_publicly"] is False
    commits = {item["commit"] for item in value["sources"]}
    assert "82d836122c3cc4fac1a651fd84acc3591cccc677" in commits
    assert "747f5d0279a91d85e32d06008665d96886eff438" in commits


def test_sdk_loader_is_lazy_hash_audited_and_does_not_connect() -> None:
    import revo3_teleop.mock as mock_module

    module_path = Path(mock_module.__file__).resolve()
    digest = hashlib.sha256(module_path.read_bytes()).hexdigest()
    loaded = load_tianji_sdk(
        TianjiSdkPluginSpec(
            client_factory="revo3_teleop.mock:SyntheticTianjiNativeClient",
            feedback_mode="normalized_mapping",
            expected_module_sha256=digest,
            sdk_identity="synthetic-test-fixture",
            sdk_version="test-only",
        )
    )

    assert loaded.provenance.hash_verified
    assert loaded.client.calls == []
    backend = TianjiMarvinBackend(loaded.client, side="A")
    assert backend.probe_capabilities().complete
    assert not backend.connected
    assert loaded.client.calls == []


def test_false_brainco_6dof_declaration_is_rejected(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "configs" / "hardware.example.json"
    value = json.loads(source.read_text(encoding="utf-8"))
    value["wrist_pose"]["provides_6dof"] = True
    path = tmp_path / "false_claim.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    report = assess_hardware_config(HardwareConfig.from_json(path))

    assert "brainco_edu_provides_6dof_claim_rejected" in report.arm_planning_blockers


def test_complete_fixture_constructs_but_never_connects_or_writes(
    tmp_path: Path, monkeypatch
) -> None:
    root = Path(__file__).resolve().parents[1]
    value = json.loads(
        (root / "configs" / "hardware.example.json").read_text(encoding="utf-8")
    )
    this_file = Path(__file__).resolve()
    value["allow_hardware_write"] = True
    value["capability_probe_confirmed"] = True
    value["tianji"].update(
        {
            "robot_ip": "192.168.1.190",
            "joint_order": list(JOINTS),
        }
    )
    value["tianji"]["sdk"].update(
        {
            "client_factory": f"{__name__}:make_test_tianji_client",
            "feedback_mode": "normalized_mapping",
            "feedback_buffer_factory": None,
            "feedback_decoder": None,
            "expected_module_sha256": hashlib.sha256(this_file.read_bytes()).hexdigest(),
            "native_library_path": str(this_file),
            "expected_native_library_sha256": hashlib.sha256(
                this_file.read_bytes()
            ).hexdigest(),
            "sdk_identity": "synthetic-test-fixture",
            "sdk_version": "test-v1",
            "client_kwargs": {"library_path": str(this_file)},
        }
    )
    value["tianji"]["safety_limits"].update(
        {
            "q_min_rad": [-1.0] * 7,
            "q_max_rad": [1.0] * 7,
            "max_delta_rad": [0.05] * 7,
        }
    )
    value["wrist_pose"].update(
        {
            "source_kind": "verified_tracker",
            "provides_6dof": True,
            "provider_factory": f"{__name__}:make_test_pose_provider",
            "expected_provider_module_sha256": hashlib.sha256(
                this_file.read_bytes()
            ).hexdigest(),
            "source_frame": "tracker_world",
            "calibration_revision": "tracker-cal-v1",
            "mapping_verified": True,
        }
    )
    value["retargeting"].update(
        {
            "ik_solver_factory": f"{__name__}:make_test_ik",
            "expected_ik_solver_module_sha256": hashlib.sha256(
                this_file.read_bytes()
            ).hexdigest(),
            "ik_model_revision": "synthetic-urdf-hash",
            "calibration_verified": True,
            "workspace_verified": True,
        }
    )
    value["retargeting"]["calibration"] = {
        "revision": "wrist-arm-v1",
        "source_frame": "tracker_world",
        "robot_base_frame": "tianji_base",
        "tool_frame": "revo_tool",
        "source_reference": {
            "frame": "tracker_world",
            "child_frame": "wrist",
            "position_m": [0.0, 0.0, 0.0],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "robot_reference": {
            "frame": "tianji_base",
            "child_frame": "revo_tool",
            "position_m": [0.4, 0.0, 0.5],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "source_axes_to_robot": np.eye(3).tolist(),
        "translation_gain": [1.0, 1.0, 1.0],
    }
    value["retargeting"]["cartesian_bounds"] = {
        "min_position_m": [0.0, -1.0, 0.0],
        "max_position_m": [1.0, 1.0, 1.0],
        "max_translation_from_reference_m": 0.2,
        "max_orientation_from_reference_rad": 0.5,
    }
    for key in value["physical_safety"]:
        value["physical_safety"][key] = True
    monkeypatch.setenv(value["arm_token_env"], "fixture-token")
    path = tmp_path / "ready_fixture.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    config = HardwareConfig.from_json(path)
    assert assess_hardware_config(config).arm_write_ready
    assembly = assemble_tianji_hardware(
        config,
        load_sdk_plugin=True,
        load_retargeting_plugins=True,
    )

    assert assembly.report.arm_write_ready
    assert assembly.backend is not None and not assembly.backend.connected
    assert assembly.runtime is not None
    assert assembly.sdk is not None and assembly.sdk.client.calls == []
    assert assembly.sdk.provenance.client_native_library_path_verified


def test_blocked_readiness_cannot_create_writable_runtime_or_call_onset(
    tmp_path: Path, monkeypatch
) -> None:
    root = Path(__file__).resolve().parents[1]
    value = json.loads(
        (root / "configs" / "hardware.example.json").read_text(encoding="utf-8")
    )
    this_file = Path(__file__).resolve()
    digest = hashlib.sha256(this_file.read_bytes()).hexdigest()
    value["allow_hardware_write"] = True
    value["capability_probe_confirmed"] = True
    value["tianji"].update(
        {"robot_ip": "192.168.1.190", "joint_order": list(JOINTS)}
    )
    value["tianji"]["sdk"].update(
        {
            "client_factory": f"{__name__}:make_test_tianji_client",
            "feedback_mode": "normalized_mapping",
            "feedback_buffer_factory": None,
            "feedback_decoder": None,
            "expected_module_sha256": digest,
            "native_library_path": str(this_file),
            "expected_native_library_sha256": digest,
            "sdk_identity": "synthetic-test-fixture",
            "sdk_version": "test-v1",
            "client_kwargs": {"library_path": str(this_file)},
        }
    )
    value["tianji"]["safety_limits"].update(
        {
            "q_min_rad": [-1.0] * 7,
            "q_max_rad": [1.0] * 7,
            "max_delta_rad": [0.05] * 7,
        }
    )
    value["wrist_pose"].update(
        {
            "source_kind": "verified_tracker",
            "provides_6dof": True,
            "provider_factory": f"{__name__}:make_test_pose_provider",
            "expected_provider_module_sha256": digest,
            "source_frame": "tracker_world",
            "calibration_revision": "tracker-cal-v1",
            "mapping_verified": True,
        }
    )
    value["retargeting"].update(
        {
            "ik_solver_factory": f"{__name__}:make_test_ik",
            "expected_ik_solver_module_sha256": digest,
            "ik_model_revision": "synthetic-urdf-hash",
            "calibration_verified": True,
            "workspace_verified": True,
            "calibration": {
                "revision": "wrist-arm-v1",
                "source_frame": "tracker_world",
                "robot_base_frame": "tianji_base",
                "tool_frame": "revo_tool",
                "source_reference": {
                    "frame": "tracker_world",
                    "child_frame": "wrist",
                    "position_m": [0.0, 0.0, 0.0],
                    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "robot_reference": {
                    "frame": "tianji_base",
                    "child_frame": "revo_tool",
                    "position_m": [0.4, 0.0, 0.5],
                    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "source_axes_to_robot": np.eye(3).tolist(),
                "translation_gain": [1.0, 1.0, 1.0],
            },
            "cartesian_bounds": {
                "min_position_m": [0.0, -1.0, 0.0],
                "max_position_m": [1.0, 1.0, 1.0],
                "max_translation_from_reference_m": 0.2,
                "max_orientation_from_reference_rad": 0.5,
            },
        }
    )
    # Deliberately leave every physical_safety item false: this must remain a
    # hard runtime write veto even though all executable plugins are hash-bound.
    monkeypatch.setenv(value["arm_token_env"], "fixture-token")
    path = tmp_path / "physically_blocked.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    assembly = assemble_tianji_hardware(
        HardwareConfig.from_json(path),
        load_sdk_plugin=True,
        load_retargeting_plugins=True,
    )

    assert not assembly.report.arm_write_ready
    assert assembly.backend is not None
    assert not assembly.backend.allow_hardware_write
    assert assembly.runtime is None
    client = assembly.sdk.client
    assembly.backend.connect("192.168.1.190")
    receipt = assembly.backend.submit_target(
        request_id="must-block",
        q_target_rad=np.zeros(7),
        target_timestamp_ns=0,
        arm_token="fixture-token",
        wrist_pose_valid=True,
        decision_timestamp_ns=0,
    )
    assert not receipt.accepted
    assert "hardware_write_disabled" in receipt.reason
    assert not any(name.startswith("OnSet") for name, _ in client.calls)


def test_historical_ctypes_boundary_requires_ack_and_only_sets_prototypes() -> None:
    class Function:
        def __init__(self):
            self.calls = []

        def __call__(self, *args):
            self.calls.append(args)
            return True

    class Library:
        pass

    library = Library()
    names = (
        "OnLinkTo",
        "OnRelease",
        "OnGetBuf",
        "OnClearSet",
        "OnSetSend",
        "OnSetJointCmdPos_A",
        "OnSetJointCmdPos_B",
        "OnSetTargetState_A",
        "OnSetTargetState_B",
        "OnEMG_A",
        "OnEMG_B",
    )
    for name in names:
        setattr(library, name, Function())
    with pytest.raises(HistoricalMarvinAbiNotAcknowledged):
        CtypesMarvinClient(
            "not-loaded.so",
            library_loader=lambda _: library,
        )

    client = CtypesMarvinClient(
        "not-loaded.so",
        acknowledge_historical_abi=True,
        library_loader=lambda _: library,
    )

    assert client.OnEMG_A.restype is None
    assert all(getattr(library, name).calls == [] for name in names)
