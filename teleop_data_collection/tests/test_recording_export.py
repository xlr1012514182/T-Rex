from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from revo3_teleop import CommandReceipt, NativeSample, SampleHeader
from revo3_teleop.recording import (
    EpisodeRecorder,
    Revo3ExportConfig,
    export_revo3_episode,
)
from revo3_teleop.recording.export_revo3 import FRAME_ALLOWLIST, META_ALLOWLIST
from revo3_v1.data import RevoEpisode
from revo3_v1.revo.contracts import JOINT_ORDER_HASH


def sample(source: str, sequence: int, timestamp_ns: int, **payload) -> NativeSample:
    return NativeSample(
        SampleHeader(
            source_id=source,
            sequence=sequence,
            capture_timestamp_ns=timestamp_ns,
            receive_timestamp_ns=timestamp_ns + 1_000,
        ),
        payload,
    )


def add_cycle(recorder: EpisodeRecorder, index: int, exact_value: float) -> None:
    anchor_ns = recorder.anchor_timestamp_ns(index)
    capture_ns = anchor_ns - 1_000
    recorder.append(
        "camera",
        sample(
            "wrist_camera",
            index,
            capture_ns,
            rgb=np.full((6, 8, 3), 20 + index, dtype=np.uint8),
        ),
    )
    recorder.append(
        "revo_state",
        sample("revo", index, capture_ns, q_rad=np.full(21, index * 0.01, np.float32)),
    )
    recorder.append(
        "tactile",
        sample("u21vt", index, capture_ns, features=np.full((5, 6), index, np.float32)),
    )
    # These native-rate streams deliberately coexist only in the master episode.
    recorder.append(
        "emg",
        sample("emg_band", index, capture_ns, signal=np.full((16, 12), index, np.float32)),
    )
    recorder.append(
        "glove",
        sample("teleop_glove", index, capture_ns, joints=np.full(24, index, np.float32)),
    )
    recorder.append(
        "tianji_state",
        sample("tianji", index, capture_ns, q_rad=np.full(7, index, np.float32)),
    )
    command = CommandReceipt(
        request_id=f"hand-{index}",
        component="revo_hand",
        accepted=True,
        requested_target=np.full(21, 0.9, np.float32),
        authorized_target=np.full(21, 0.5, np.float32),
        exact_sent_target=np.full(21, exact_value, np.float32),
        decision_timestamp_ns=anchor_ns + 10_000,
        write_timestamp_ns=anchor_ns + 20_000,
        controller_sequence=index,
        clipped=True,
        unit="rad",
        joint_order_hash=JOINT_ORDER_HASH,
    )
    recorder.record_command(command)
    recorder.record_anchor(
        anchor_index=index,
        streams=("camera", "revo_state", "tactile"),
        hand_command_request_id=command.request_id,
        max_age_ns={"camera": 100_000, "revo_state": 100_000, "tactile": 100_000},
    )


def test_allowlist_projection_uses_exact_sent_target_and_loads(tmp_path: Path) -> None:
    recorder = EpisodeRecorder(
        tmp_path / "master",
        episode_id="bottle_session_001",
        epoch_ns=3_000_000_000,
        metadata={
            "task": "bottle",
            "instruction": "Grasp the centered bottle and hold it securely.",
            "subject_id": "mock-subject",
        },
    )
    recorder.start()
    add_cycle(recorder, 0, 0.11)
    add_cycle(recorder, 1, 0.22)
    committed = recorder.commit()

    derived = export_revo3_episode(
        committed,
        tmp_path / "derived",
        Revo3ExportConfig(synthetic_fixture=True),
    )
    episode = RevoEpisode.load(derived)
    assert episode.state_rad.shape == (2, 21)
    assert episode.tactile_features.shape == (2, 5, 6)
    np.testing.assert_allclose(episode.action_target_rad[0], 0.11)
    np.testing.assert_allclose(episode.action_target_rad[1], 0.22)
    # Neither the requested target (0.9) nor pre-write authorized target (0.5)
    # is allowed to become action supervision.
    assert not np.any(np.isclose(episode.action_target_rad, 0.9))
    assert not np.any(np.isclose(episode.action_target_rad, 0.5))

    meta = json.loads((derived / "meta.json").read_text(encoding="utf-8"))
    assert set(meta) == META_ALLOWLIST
    assert meta["contains_emg"] is False
    assert meta["action_label_source"] == "controller_target"
    assert "subject_id" not in meta
    with np.load(derived / "frames.npz", allow_pickle=False) as archive:
        assert set(archive.files) == FRAME_ALLOWLIST
    assert not (derived / "streams").exists()
    assert not any(term in path.name.lower() for path in derived.rglob("*") for term in ("arm", "glove"))
    # Native streams remain preserved in the master at their own sample counts.
    for stream in ("emg", "glove", "tianji_state"):
        shards = list((committed / "streams" / stream / "shards").glob("*.h5"))
        assert len(shards) == 1


def test_export_rejects_sample_received_after_controller_decision(tmp_path: Path) -> None:
    recorder = EpisodeRecorder(
        tmp_path / "master",
        episode_id="late_receive_tamper",
        epoch_ns=4_000_000_000,
        metadata={"task": "bottle", "instruction": "Grasp the bottle."},
    )
    recorder.start()
    add_cycle(recorder, 0, 0.1)
    add_cycle(recorder, 1, 0.2)
    committed = recorder.commit()

    receipt_rows = [
        json.loads(line)
        for line in (committed / "command_receipts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    index_path = committed / "streams/camera/index.jsonl"
    index_rows = [
        json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()
    ]
    index_rows[0]["header"]["receive_timestamp_ns"] = (
        int(receipt_rows[0]["decision_timestamp_ns"]) + 1
    )
    index_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in index_rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="received after controller decision"):
        export_revo3_episode(
            committed,
            tmp_path / "derived",
            Revo3ExportConfig(synthetic_fixture=True),
        )
