import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from revo3_v1.emg.calibration import calibrate_emg_checkpoint, load_emg_calibration
from revo3_v1.emg.model import GNIClassifier, GNIModelConfig, save_emg_checkpoint
from revo3_v1.emg.preprocessing import BRAINCO_EDU_8CH_250HZ


def make_fixture(root):
    profile = BRAINCO_EDU_8CH_250HZ
    checkpoint = root / "base.pt"
    normalization = {"mean": np.zeros(8, np.float32), "std": np.ones(8, np.float32)}
    save_emg_checkpoint(
        checkpoint,
        GNIClassifier(GNIModelConfig.smoke(8, mainline=True)),
        normalization,
        {"train_session_ids": ["train-session"]},
        preprocessing_profile=profile,
    )
    rng = np.random.RandomState(3)
    signals = rng.randn(10, 8, 500).astype(np.float32) * 0.05
    labels = np.repeat(np.arange(5, dtype=np.int64), 2)
    windows = root / "windows.npz"
    np.savez(
        windows,
        signal=signals,
        label=labels,
        sample_rate_hz=np.asarray(250),
        channel_order=np.asarray(profile.channel_order),
        preprocessing_profile_id=np.asarray(profile.profile_id),
        preprocessed=np.asarray(False),
        filter_state_provenance=np.asarray(""),
    )
    rows = []
    for index, label in enumerate(labels.tolist()):
        start = int(index * (300_000_000_000 / 9))
        rows.append({
            "index": index,
            "subject_id": "subject-a",
            "day_id": "day-2",
            "session_id": "calibration-session",
            "label": label,
            "label_name": GNIModelConfig.smoke(8, mainline=True).resolved_labels()[label],
            "window_start_ns": start,
            "window_end_ns": start + 2_000_000_000,
        })
    manifest = root / "calibration.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return profile, checkpoint, windows, manifest


def test_temperature_calibration_artifact_has_strict_lineage(tmp_path):
    profile, checkpoint, windows, manifest = make_fixture(tmp_path)
    artifact_path = tmp_path / "calibration.pt"
    artifact = calibrate_emg_checkpoint(
        base_checkpoint=checkpoint,
        windows_npz=windows,
        manifest=manifest,
        output=artifact_path,
        channel_order=profile.channel_order,
        subject_id="subject-a",
        day_id="day-2",
        method="temperature",
    )
    assert artifact["session_span_s"] >= 300
    assert artifact["filter_lineage"] == "independent_window_zero_state_calibration_fallback"
    loaded = load_emg_calibration(
        artifact_path,
        base_checkpoint=checkpoint,
        channel_order=profile.channel_order,
        expected_subject_id="subject-a",
        expected_day_id="day-2",
    )
    assert loaded.payload["method"] == "temperature"
    with pytest.raises(ValueError, match="channel"):
        load_emg_calibration(
            artifact_path,
            base_checkpoint=checkpoint,
            channel_order=tuple(reversed(profile.channel_order)),
        )


def test_calibration_rejects_base_training_session_leakage(tmp_path):
    profile, checkpoint, windows, manifest = make_fixture(tmp_path)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["session_id"] = "train-session"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        calibrate_emg_checkpoint(
            base_checkpoint=checkpoint,
            windows_npz=windows,
            manifest=manifest,
            output=tmp_path / "bad.pt",
            channel_order=profile.channel_order,
            subject_id="subject-a",
            day_id="day-2",
        )
