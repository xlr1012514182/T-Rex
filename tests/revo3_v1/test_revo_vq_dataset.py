import json
from pathlib import Path
import tempfile

import numpy as np
import pytest

from revo3_v1.data import SyntheticRevoConfig, generate_synthetic_revo_episodes
from tactile_vqvae.data import build_revo_train_val_datasets


def test_revo_vq_dataset_is_single_hand_and_split_before_stats():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        episodes = root / "episodes"
        summary = generate_synthetic_revo_episodes(
            episodes,
            SyntheticRevoConfig(episodes_per_task=2, frames_per_episode=64),
        )
        ids = summary["episode_ids"]
        split_names = (
            "midtrain_train",
            "sft_train",
            "development",
            "locked_test",
            "midtrain_train",
            "midtrain_train",
            "development",
            "locked_test",
        )
        manifest = {
            "schema_version": "revo3-corpus-split-v1",
            "episodes": [
                {
                    "episode_id": episode_id,
                    "split": split,
                    "day": f"day-{index}",
                    "object_instance": f"object-{index}",
                    "operator": f"operator-{index}",
                    "duration_seconds": 64 / 30,
                }
                for index, (episode_id, split) in enumerate(zip(ids, split_names))
            ],
            "evaluation_protocol": {
                "locked_test_isolation": ["day", "object_instance"]
            },
            "duration_targets_hours": {
                "midtrain_train": 12.0,
                "sft_train": 4.0,
                "development": 2.0,
                "locked_test": 2.0,
            },
            "duration_targets_enforced": False,
        }
        split_path = root / "split.json"
        split_path.write_text(json.dumps(manifest), encoding="utf-8")

        # A locked-test outlier must never influence train normalization.
        locked = episodes / ids[3] / "frames.npz"
        with np.load(locked) as archive:
            payload = {key: archive[key] for key in archive.files}
        payload["tactile_features"] = np.full_like(payload["tactile_features"], 1e6)
        np.savez_compressed(locked, **payload)

        # SFT is also downstream of the frozen tokenizer/statistics artifact.
        sft = episodes / ids[1] / "frames.npz"
        with np.load(sft) as archive:
            payload = {key: archive[key] for key in archive.files}
        payload["tactile_features"] = np.full_like(payload["tactile_features"], 1e6)
        np.savez_compressed(sft, **payload)

        train, dev, stats = build_revo_train_val_datasets(
            episodes, split_path, window=16, stride=4
        )
        assert train.num_episodes == 3
        assert dev.num_episodes == 2
        assert stats.tacf6_min.shape == (30,)
        assert float(stats.tacf6_max.max()) < 100.0
        sample = train[0]
        assert sample["f6"].shape == (16, 5, 6)
        assert sample["hand"] == 0


def test_split_manifest_rejects_group_leakage():
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "split.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": "revo3-corpus-split-v1",
                    "episodes": [
                        {"episode_id": "a", "split": "midtrain_train", "day": "d", "object_instance": "o", "operator": "p", "duration_seconds": 1.0},
                        {"episode_id": "b", "split": "development", "day": "d", "object_instance": "o", "operator": "p", "duration_seconds": 1.0},
                        {"episode_id": "c", "split": "sft_train", "day": "s", "object_instance": "s", "operator": "s", "duration_seconds": 1.0},
                        {"episode_id": "d", "split": "locked_test", "day": "x", "object_instance": "y", "operator": "z", "duration_seconds": 1.0},
                    ],
                    "evaluation_protocol": {"locked_test_isolation": ["day"]},
                    "duration_targets_hours": {
                        "midtrain_train": 12.0,
                        "sft_train": 4.0,
                        "development": 2.0,
                        "locked_test": 2.0,
                    },
                    "duration_targets_enforced": False,
                }
            ),
            encoding="utf-8",
        )
        from revo3_v1.data import RevoCorpusSplitManifest
        try:
            RevoCorpusSplitManifest.load(path)
        except ValueError as error:
            assert "leaks across splits" in str(error)
        else:
            raise AssertionError("leaking split manifest was accepted")


def test_split_manifest_rejects_day_only_locked_isolation(tmp_path):
    payload = {
        "schema_version": "revo3-corpus-split-v1",
        "episodes": [
            {"episode_id": "a", "split": "midtrain_train", "day": "d1", "object_instance": "o1", "operator": "p", "duration_seconds": 1.0},
            {"episode_id": "b", "split": "sft_train", "day": "d2", "object_instance": "o2", "operator": "p", "duration_seconds": 1.0},
            {"episode_id": "c", "split": "development", "day": "d3", "object_instance": "o3", "operator": "p", "duration_seconds": 1.0},
            {"episode_id": "d", "split": "locked_test", "day": "d4", "object_instance": "o4", "operator": "p", "duration_seconds": 1.0},
        ],
        "evaluation_protocol": {"locked_test_isolation": ["day"]},
        "duration_targets_hours": {
            "midtrain_train": 12.0, "sft_train": 4.0,
            "development": 2.0, "locked_test": 2.0,
        },
        "duration_targets_enforced": False,
    }
    path = tmp_path / "split.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    from revo3_v1.data import RevoCorpusSplitManifest
    with pytest.raises(ValueError, match="both day and object_instance"):
        RevoCorpusSplitManifest.load(path)
