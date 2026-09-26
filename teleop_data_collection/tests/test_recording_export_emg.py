from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from revo3_teleop import NativeSample, SampleHeader
from revo3_teleop.recording import (
    EMGExportConfig,
    EMGLabelInterval,
    EMGSessionSpec,
    EpisodeRecorder,
    export_emg_dataset,
    export_emg_binary_dataset,
)
from revo3_v1.emg.data import verify_split_manifests


def make_session(root: Path, *, name: str, epoch_ns: int, packets: int = 4) -> tuple[Path, int, int]:
    recorder = EpisodeRecorder(root, episode_id=name, epoch_ns=epoch_ns)
    recorder.start()
    period_ns = 4_000_000
    for packet in range(packets):
        timestamps = epoch_ns + np.arange(
            packet * 20, packet * 20 + 20, dtype=np.int64
        ) * period_ns
        signal = np.stack(
            [np.arange(20, dtype=np.float32) + channel + packet for channel in range(8)]
        )
        recorder.append(
            "emg",
            NativeSample(
                SampleHeader(
                    source_id="brainco_edu_emg",
                    sequence=packet,
                    capture_timestamp_ns=int(timestamps[-1]),
                    receive_timestamp_ns=int(timestamps[-1] + 1000),
                    clock_domain="host_callback_reconstructed",
                ),
                {
                    "signal": signal,
                    "sample_timestamp_ns": timestamps.astype(np.int64),
                    "lead_off_bits": np.asarray([0], np.uint16),
                    "sample_rate_hz": np.asarray([250], np.float32),
                    "samples_per_channel": np.asarray([20], np.int16),
                },
            ),
        )
    committed = recorder.commit()
    return committed, epoch_ns, epoch_ns + packets * 20 * period_ns


def make_session_with_explicit_packet_gap(
    root: Path, *, name: str, epoch_ns: int
) -> tuple[Path, int, int]:
    """Create continuous-looking timestamps with a truthful missing-packet marker."""

    recorder = EpisodeRecorder(root, episode_id=name, epoch_ns=epoch_ns)
    recorder.start()
    period_ns = 4_000_000
    sequences = (0, 1, 3, 4)
    for packet, sequence in enumerate(sequences):
        timestamps = epoch_ns + np.arange(
            packet * 20, packet * 20 + 20, dtype=np.int64
        ) * period_ns
        recorder.append(
            "emg",
            NativeSample(
                SampleHeader(
                    source_id="brainco_edu_emg",
                    sequence=sequence,
                    capture_timestamp_ns=int(timestamps[-1]),
                    receive_timestamp_ns=int(timestamps[-1] + 1000),
                    clock_domain="host_callback_reconstructed",
                    dropped_since_previous=1 if packet == 2 else 0,
                ),
                {
                    "signal": np.ones((8, 20), np.float32),
                    "sample_timestamp_ns": timestamps.astype(np.int64),
                    "lead_off_bits": np.asarray([0], np.uint16),
                    "sample_rate_hz": np.asarray([250], np.float32),
                    "samples_per_channel": np.asarray([20], np.int16),
                    "sequence_gap_packets": np.asarray(
                        [1 if packet == 2 else 0], np.int64
                    ),
                },
            ),
        )
    committed = recorder.commit()
    return committed, epoch_ns, epoch_ns + 80 * period_ns


def test_emg_projection_requires_reviewed_labels_and_subject_exclusive_splits(
    tmp_path: Path,
) -> None:
    specs = []
    for index, split in enumerate(("train", "val", "test")):
        root, start, end = make_session(
            tmp_path / "master",
            name=f"session_{index}",
            epoch_ns=1_000_000_000 + index * 1_000_000_000,
        )
        specs.append(
            EMGSessionSpec(
                episode_root=root,
                subject_id=f"subject_{index}",
                session_id=f"session_{index}",
                split=split,
                intervals=(
                    EMGLabelInterval(start, end, index % 2, "protocol_cue_human_reviewed", True),
                ),
            )
        )
    derived = export_emg_binary_dataset(
        specs,
        tmp_path / "derived_emg",
        EMGExportConfig(window_samples=40, stride_samples=20),
    )
    with np.load(derived / "windows.npz", allow_pickle=False) as archive:
        assert archive["signal"].shape == (9, 8, 40)
        assert set(archive.files) == {
            "signal",
            "label",
            "subject_id",
            "session_id",
            "window_start_ns",
            "sample_rate_hz",
        }
    verify_split_manifests(
        {split: derived / "manifests" / f"{split}.jsonl" for split in ("train", "val", "test")}
    )
    meta = json.loads((derived / "dataset_meta.json").read_text(encoding="utf-8"))
    assert meta["contains_rgb"] is False
    assert meta["contains_robot_action"] is False
    assert "glove heuristic" in meta["label_policy"]


def test_emg_projection_rejects_unreviewed_or_cross_split_subject(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="human reviewed"):
        EMGLabelInterval(0, 100, 1, "protocol_cue_human_reviewed", False)

    root, start, end = make_session(tmp_path / "master", name="one", epoch_ns=10_000)
    interval = EMGLabelInterval(start, end, 1, "manual_human_reviewed", True)
    specs = [
        EMGSessionSpec(root, "same_subject", "a", "train", (interval,)),
        EMGSessionSpec(root, "same_subject", "b", "test", (interval,)),
    ]
    with pytest.raises(ValueError, match="subject leakage"):
        export_emg_binary_dataset(
            specs,
            tmp_path / "bad",
            EMGExportConfig(window_samples=40, stride_samples=20),
        )


def test_emg_projection_never_spans_an_explicit_packet_gap(tmp_path: Path) -> None:
    specs = []
    for index, split in enumerate(("train", "val", "test")):
        root, start, end = make_session_with_explicit_packet_gap(
            tmp_path / "master_gap",
            name=f"gap_{index}",
            epoch_ns=2_000_000_000 + index * 1_000_000_000,
        )
        specs.append(
            EMGSessionSpec(
                root,
                f"gap_subject_{index}",
                f"gap_session_{index}",
                split,
                (
                    EMGLabelInterval(
                        start,
                        end,
                        index % 2,
                        "protocol_cue_human_reviewed",
                        True,
                    ),
                ),
            )
        )

    # Each side of the missing packet has only 40 samples.  A 60-sample
    # window would exist only if the exporter incorrectly bridged the gap.
    with pytest.raises(ValueError, match="every split needs at least one reviewed window"):
        export_emg_binary_dataset(
            specs,
            tmp_path / "gap_derived",
            EMGExportConfig(window_samples=60, stride_samples=20),
        )


def test_mainline_projection_exports_five_class_profile_bound_continuous_filter(tmp_path: Path) -> None:
    specs = []
    for index, split in enumerate(("train", "val", "test")):
        root, start, end = make_session(
            tmp_path / "master_mainline",
            name=f"mainline_{index}",
            epoch_ns=5_000_000_000 + index * 5_000_000_000,
            packets=30,
        )
        specs.append(EMGSessionSpec(
            root,
            f"mainline_subject_{index}",
            f"mainline_session_{index}",
            split,
            (EMGLabelInterval(start, end, index, "manual_human_reviewed", True),),
        ))
    derived = export_emg_dataset(specs, tmp_path / "mainline_emg")
    with np.load(derived / "windows.npz", allow_pickle=False) as archive:
        assert archive["signal"].shape[1:] == (8, 500)
        assert bool(archive["preprocessed"])
        assert str(archive["filter_state_provenance"]) == "session_continuous_causal_sos_before_windowing"
        assert tuple(archive["channel_order"].tolist()) == tuple(f"emg_{i}" for i in range(8))
    meta = json.loads((derived / "dataset_meta.json").read_text(encoding="utf-8"))
    assert list(meta["labels"].values()) == [
        "POWER_GRASP", "PRECISION_GRASP", "LATERAL_GRASP", "RELEASE", "REST"
    ]
