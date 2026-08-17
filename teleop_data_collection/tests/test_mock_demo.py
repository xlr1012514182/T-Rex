from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import revo3_teleop.mock as mock_module
from revo3_teleop.backends import TIANJI_JOINT_ORDER_HASH
from revo3_teleop.cli.mock_demo import main as mock_cli_main
from revo3_teleop.mock import MockCollectionConfig, run_mock_collection
from revo3_v1.data import RevoEpisode


def test_mock_collection_runs_four_tasks_and_physically_separates_vla(tmp_path: Path) -> None:
    results = run_mock_collection(MockCollectionConfig(output_root=tmp_path, duration_s=0.1))
    assert len(results) == 4
    for result in results:
        master = Path(result["master"])
        derived = Path(result["revo3_vla"])
        assert (master / "streams/emg").is_dir()
        assert (master / "streams/tianji_state").is_dir()
        episode = RevoEpisode.load(derived)
        assert episode.num_frames >= 2
        assert not (derived / "streams").exists()
        meta = json.loads((derived / "meta.json").read_text(encoding="utf-8"))
        assert meta["contains_emg"] is False
        assert set(meta).isdisjoint({"arm", "tianji", "glove", "emg_signal"})


def test_four_task_mock_uses_gated_tianji_backend_and_exact_native_receipts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    clients: list[mock_module.SyntheticTianjiNativeClient] = []
    base_client = mock_module.SyntheticTianjiNativeClient

    class CapturingSyntheticTianjiClient(base_client):
        def __init__(self) -> None:
            super().__init__()
            clients.append(self)

    monkeypatch.setattr(
        mock_module,
        "SyntheticTianjiNativeClient",
        CapturingSyntheticTianjiClient,
    )
    results = run_mock_collection(
        MockCollectionConfig(output_root=tmp_path, duration_s=0.1)
    )

    assert len(clients) == len(results) == 4
    for result, client in zip(results, clients, strict=True):
        master = Path(result["master"])
        manifest = json.loads((master / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["metadata"]["synthetic_fixture"] is True

        receipts = [
            json.loads(line)
            for line in (master / "command_receipts.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        arm_receipts = [row for row in receipts if row["component"] == "tianji_arm"]
        assert len(arm_receipts) >= 2
        assert all(row["accepted"] for row in arm_receipts)
        assert all(row["reason"] == "side_A:sent" for row in arm_receipts)
        assert all(row["unit"] == "rad" for row in arm_receipts)
        assert all(
            row["joint_order_hash"] == TIANJI_JOINT_ORDER_HASH
            for row in arm_receipts
        )

        names = [name for name, _ in client.calls]
        assert client.calls[0] == ("OnLinkTo", (192, 168, 1, 190))
        assert names[-1] == "OnRelease"
        assert "OnSetJointCmdPos_B" not in names
        position_indices = [
            index for index, name in enumerate(names) if name == "OnSetJointCmdPos_A"
        ]
        assert len(position_indices) == len(arm_receipts)
        for call_index, receipt in zip(position_indices, arm_receipts, strict=True):
            assert names[call_index - 1 : call_index + 2] == [
                "OnClearSet",
                "OnSetJointCmdPos_A",
                "OnSetSend",
            ]
            native_deg = np.asarray(client.calls[call_index][1], dtype=np.float64)
            np.testing.assert_allclose(
                native_deg,
                np.rad2deg(np.asarray(receipt["exact_sent_target"])),
                atol=1e-6,
            )

        # Accepted writes require the production backend to observe an
        # advancing feedback serial.  The extra frames cover baseline and the
        # verified state-0 shutdown before TCP release.
        assert client.frame_serial >= len(arm_receipts) + 3
        assert "OnEMG_A" in names
        assert "OnSetTargetState_A" in names


def test_synthetic_tianji_client_preserves_side_b_native_transaction() -> None:
    epoch_ns = 1_000_000_000
    backend, client, clock = mock_module._build_synthetic_tianji_backend(
        epoch_ns=epoch_ns,
        side="B",
    )
    try:
        clock.advance_to(epoch_ns + 1_000_000)
        target = np.linspace(-0.05, 0.05, 7, dtype=np.float32)
        receipt = backend.submit_target(
            request_id="synthetic-side-b",
            q_target_rad=target,
            target_timestamp_ns=epoch_ns + 500_000,
            arm_token=mock_module._SYNTHETIC_TIANJI_ARM_TOKEN,
            wrist_pose_valid=True,
            decision_timestamp_ns=epoch_ns + 750_000,
        )
        assert receipt.accepted
        np.testing.assert_allclose(receipt.exact_sent_target, target, atol=1e-7)
    finally:
        backend.close(timeout_ns=20_000_000, poll_interval_s=0.001)

    names = [name for name, _ in client.calls]
    position_index = names.index("OnSetJointCmdPos_B")
    assert names[position_index - 1 : position_index + 2] == [
        "OnClearSet",
        "OnSetJointCmdPos_B",
        "OnSetSend",
    ]
    assert "OnSetJointCmdPos_A" not in names


def test_mock_rates_drive_native_stream_counts(tmp_path: Path) -> None:
    config = MockCollectionConfig(
        output_root=tmp_path,
        duration_s=0.1,
        tasks=("bottle",),
        camera_hz=30,
        revo_state_hz=80,
        tactile_hz=60,
        glove_hz=40,
        tianji_state_hz=50,
    )
    result = run_mock_collection(config)[0]
    manifest = json.loads(
        (Path(result["master"]) / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["stream_counts"] == {
        "camera": 4,
        "emg": 2,
        "glove": 5,
        "revo_state": 9,
        "tactile": 7,
        "tianji_state": 6,
    }


def test_checked_json_config_is_executed_by_cli(
    tmp_path: Path, capsys
) -> None:
    output = tmp_path / "configured_output"
    config_path = tmp_path / "mock.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "revo3-teleop-config-v1",
                "mode": "mock",
                "output_root": str(output),
                "duration_s": 0.1,
                "anchor_hz": 30,
                "rates_hz": {
                    "camera": 30,
                    "revo_state": 100,
                    "tactile": 120,
                    "glove": 120,
                    "emg_packets": 12.5,
                    "tianji_state": 100,
                    "revo_command": 30,
                },
                "tasks": ["phone"],
                "allow_hardware_write": False,
            }
        ),
        encoding="utf-8",
    )

    assert mock_cli_main(["--config", str(config_path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["synthetic_fixture"] is True
    assert [row["task"] for row in report["episodes"]] == ["phone"]
    assert (output / "master/committed/mock_phone_00/manifest.json").is_file()
