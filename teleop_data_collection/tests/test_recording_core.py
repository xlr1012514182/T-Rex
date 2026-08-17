from __future__ import annotations

import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from revo3_teleop import CommandReceipt, NativeSample, SampleHeader
from revo3_teleop.recording import EpisodeRecorder, RecorderState, load_native_payload
from revo3_v1.revo.contracts import JOINT_ORDER_HASH


def sample(source: str, sequence: int, timestamp_ns: int, **payload) -> NativeSample:
    return NativeSample(
        SampleHeader(
            source_id=source,
            sequence=sequence,
            capture_timestamp_ns=timestamp_ns,
            receive_timestamp_ns=timestamp_ns + 100,
        ),
        payload,
    )


def receipt(request_id: str, anchor_ns: int, value: float = 0.2) -> CommandReceipt:
    requested = np.full(21, 0.8, dtype=np.float32)
    authorized = np.full(21, 0.4, dtype=np.float32)
    sent = np.full(21, value, dtype=np.float32)
    return CommandReceipt(
        request_id=request_id,
        component="revo_hand",
        accepted=True,
        requested_target=requested,
        authorized_target=authorized,
        exact_sent_target=sent,
        decision_timestamp_ns=anchor_ns + 1_000,
        write_timestamp_ns=anchor_ns + 2_000,
        controller_sequence=int(request_id.rsplit("-", 1)[-1]),
        clipped=True,
        unit="rad",
        joint_order_hash=JOINT_ORDER_HASH,
    )


def test_native_rate_files_strict_order_and_causal_latest_not_after(tmp_path: Path) -> None:
    epoch = 1_000_000_000
    recorder = EpisodeRecorder(
        tmp_path,
        episode_id="episode_001",
        epoch_ns=epoch,
        metadata={"task": "bottle", "instruction": "Grasp the bottle."},
    )
    inprogress = recorder.start()
    assert inprogress.parent.name == ".inprogress"

    recorder.append("camera", sample("cam", 10, epoch - 100, rgb=np.zeros((4, 5, 3), np.uint8)))
    # A future sample may already have arrived, but it must not be selected.
    recorder.append("camera", sample("cam", 11, epoch + 100, rgb=np.ones((4, 5, 3), np.uint8)))
    with pytest.raises(ValueError, match="strictly increasing"):
        recorder.append("camera", sample("cam", 11, epoch + 200, rgb=np.zeros((4, 5, 3), np.uint8)))

    command = receipt("hand-0", epoch)
    recorder.record_command(command)
    anchor = recorder.record_anchor(
        anchor_index=0,
        streams=("camera",),
        hand_command_request_id=command.request_id,
        max_age_ns={"camera": 1_000},
    )
    assert anchor.timestamp_ns == epoch
    assert anchor.streams["camera"].sequence == 10
    assert anchor.streams["camera"].capture_timestamp_ns <= anchor.timestamp_ns

    committed = recorder.commit()
    assert recorder.state == RecorderState.COMMITTED
    assert committed == tmp_path / "committed" / "episode_001"
    assert not inprogress.exists()
    rows = [json.loads(line) for line in (committed / "streams/camera/index.jsonl").read_text().splitlines()]
    assert [row["header"]["sequence"] for row in rows] == [10, 11]
    shards = list((committed / "streams/camera/shards").glob("*.h5"))
    assert len(shards) == 1
    with h5py.File(shards[0], "r") as handle:
        assert int(handle.attrs["committed_rows"]) == 2
        assert bool(handle.attrs["closed_cleanly"])
        assert handle["payload/rgb"].compression == "lzf"
        assert handle["payload/rgb"].chunks is not None
    with pytest.raises(RuntimeError, match="not recording"):
        recorder.append("camera", sample("cam", 12, epoch + 300, rgb=np.zeros((4, 5, 3), np.uint8)))


def test_rejected_or_wrong_component_command_cannot_supervise_anchor(tmp_path: Path) -> None:
    epoch = 2_000_000_000
    recorder = EpisodeRecorder(tmp_path, episode_id="episode_002", epoch_ns=epoch)
    recorder.start()
    recorder.append("camera", sample("cam", 0, epoch, rgb=np.zeros((2, 2, 3), np.uint8)))
    rejected = CommandReceipt(
        request_id="rejected",
        component="revo_hand",
        accepted=False,
        requested_target=np.zeros(21),
        decision_timestamp_ns=epoch,
        reason="safety_veto",
        joint_order_hash=JOINT_ORDER_HASH,
    )
    recorder.record_command(rejected)
    with pytest.raises(ValueError, match="accepted exact_sent_target"):
        recorder.record_anchor(
            anchor_index=0,
            streams=("camera",),
            hand_command_request_id="rejected",
        )
    recorder.abort("expected rejected-command fixture")


def test_abort_quarantines_incomplete_episode(tmp_path: Path) -> None:
    recorder = EpisodeRecorder(tmp_path, episode_id="episode_abort", epoch_ns=10)
    original = recorder.start()
    quarantined = recorder.abort("camera dropout")
    assert not original.exists()
    assert quarantined.parent.name == "quarantine"
    assert recorder.state == RecorderState.ABORTED
    manifest = json.loads((quarantined / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["lifecycle"] == "aborted"
    assert manifest["abort_reason"] == "camera dropout"


def test_one_hundred_native_samples_use_bounded_hdf5_shards(tmp_path: Path) -> None:
    recorder = EpisodeRecorder(
        tmp_path,
        episode_id="long_stream_fixture",
        epoch_ns=5_000_000_000,
        stream_shard_rows=32,
    )
    recorder.start()
    for index in range(100):
        recorder.append(
            "emg",
            sample(
                "emg_band",
                index,
                5_000_000_000 + index * 1_000_000,
                signal=np.full((16, 20), index, dtype=np.float32),
            ),
        )
    committed = recorder.commit()
    shards = sorted((committed / "streams/emg/shards").glob("*.h5"))
    assert len(shards) == 4
    assert len(shards) < 100
    rows = [json.loads(line) for line in (committed / "streams/emg/index.jsonl").read_text().splitlines()]
    assert len(rows) == 100
    assert rows[0]["row_index"] == 0
    assert rows[31]["row_index"] == 31
    assert rows[32]["row_index"] == 0
    loaded = load_native_payload(committed, "emg", rows[57])
    np.testing.assert_array_equal(loaded["signal"], np.full((16, 20), 57, np.float32))
    committed_rows = []
    for shard in shards:
        with h5py.File(shard, "r") as handle:
            committed_rows.append(int(handle.attrs["committed_rows"]))
            assert handle["payload/signal"].compression == "lzf"
            assert handle["payload/signal"].maxshape[0] is None
            assert bool(handle.attrs["closed_cleanly"])
    assert committed_rows == [32, 32, 32, 4]


def test_read_helper_rejects_uncommitted_tail_reference(tmp_path: Path) -> None:
    recorder = EpisodeRecorder(tmp_path, episode_id="crash_boundary", epoch_ns=7_000_000_000)
    recorder.start()
    recorder.append(
        "emg",
        sample("emg_band", 0, 7_000_000_000, signal=np.zeros((16, 20), np.float32)),
    )
    committed = recorder.commit()
    row = json.loads((committed / "streams/emg/index.jsonl").read_text().splitlines()[0])
    shard = committed / row["relative_path"]
    with h5py.File(shard, "r+") as handle:
        handle.attrs.modify("committed_rows", 0)
    with pytest.raises(ValueError, match="uncommitted HDF5 tail"):
        load_native_payload(committed, "emg", row)
