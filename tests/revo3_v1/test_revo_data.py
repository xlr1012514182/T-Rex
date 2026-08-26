import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from revo3_v1.data import (
    ConversionConfig,
    RevoEpisode,
    SyntheticRevoConfig,
    convert_revo_episodes_to_trex_json,
    generate_synthetic_revo_episodes,
)


class RevoDataTest(unittest.TestCase):
    def test_synthetic_episode_and_trex_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episodes = root / "episodes"
            summary = generate_synthetic_revo_episodes(
                episodes,
                SyntheticRevoConfig(episodes_per_task=1, frames_per_episode=64),
            )
            self.assertEqual(len(summary["episode_ids"]), 4)
            loaded = RevoEpisode.load(episodes / summary["episode_ids"][0])
            self.assertEqual(loaded.action_target_rad.shape, (64, 21))
            self.assertEqual(loaded.tactile_features.shape, (64, 5, 6))

            target = root / "revo3_trex_train.json"
            manifest = convert_revo_episodes_to_trex_json(
                episodes, target, ConversionConfig()
            )
            self.assertFalse(manifest["contains_emg"])
            self.assertEqual(manifest["action_shape"], [16, 21])
            with target.open("r", encoding="utf-8") as handle:
                rows = json.load(handle)
            # 10 Hz anchors over a 30 Hz source; no left/terminal padding.
            self.assertEqual(len(rows), 24)
            self.assertTrue(all(not row["contains_emg"] for row in rows))
            self.assertTrue(all(len(row["action"]) == 16 * 21 for row in rows))
            self.assertTrue(all(len(row["tactile_f6"]) == 5 * 6 for row in rows))
            self.assertTrue(all(len(row["input_image_slow"]) == 1 for row in rows))
            self.assertTrue(all(len(row["input_image_fast"]) == 1 for row in rows))
            self.assertTrue(all(len(row["flare_image_full"]) == 8 for row in rows))
            self.assertTrue(all("_full.png" in row["input_image_slow"][0] for row in rows))
            self.assertTrue(all("_center.png" in row["input_image_fast"][0] for row in rows))
            self.assertTrue(
                all(
                    row["rgb_full_timestamp_ns"] == row["rgb_center_timestamp_ns"]
                    for row in rows
                )
            )
            self.assertTrue(
                all(np.asarray(row["tactile_f6_history_delayed"]).shape == (4, 16, 5, 6)
                    for row in rows)
            )
            self.assertTrue(
                all(
                    np.asarray(row["tactile_f6_history_delayed_jitter"]).shape
                    == (4, 3, 16, 5, 6)
                    for row in rows
                )
            )
            self.assertEqual(rows[0]["frame_index"], 15)
            self.assertTrue(
                all(
                    row["action_semantics"]
                    == "accepted_exact_sent_teleop_target"
                    and row["contains_cair_residual"] is False
                    for row in rows
                )
            )
            expected_primitive = {
                "bottle": "POWER_GRASP",
                "phone": "PRECISION_GRASP",
                "plastic_bag": "PRECISION_GRASP",
                "refrigerator_door": "LATERAL_GRASP",
            }
            for episode_id in summary["episode_ids"]:
                episode = RevoEpisode.load(episodes / episode_id)
                self.assertEqual(
                    episode.meta.grasp_primitive,
                    expected_primitive[episode.meta.task],
                )
            with (root / "revo3_trex_train_statistics.json").open(
                "r", encoding="utf-8"
            ) as handle:
                stats = json.load(handle)["revo3_single_hand"]
            self.assertEqual(np.asarray(stats["action"]["q01"]).shape, (16, 21))
            self.assertEqual(np.asarray(stats["state"]["q01"]).shape, (21,))
            self.assertEqual(np.asarray(stats["tactile_f6"]["q01"]).shape, (30,))

    def test_real_splits_reuse_midtrain_only_frozen_statistics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episodes = root / "episodes"
            summary = generate_synthetic_revo_episodes(
                episodes,
                SyntheticRevoConfig(episodes_per_task=2, frames_per_episode=64),
            )
            ids = summary["episode_ids"]
            splits = [
                "midtrain_train", "midtrain_train", "midtrain_train",
                "sft_train", "sft_train", "development",
                "locked_test", "locked_test",
            ]
            entries = []
            for index, (episode_id, split) in enumerate(zip(ids, splits)):
                meta_path = episodes / episode_id / "meta.json"
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                meta.update(
                    synthetic_fixture=False,
                    collection_day=f"day-{index}",
                    object_instance=f"object-{index}",
                    operator=f"operator-{index}",
                )
                if split == "sft_train" and index == 4:
                    meta["instruction_source"] = "frozen_planner"
                    meta["planner_revision"] = "planner-frozen-test"
                    meta["planner_output_sha256"] = "d" * 64
                meta_path.write_text(json.dumps(meta), encoding="utf-8")
                if split == "sft_train":
                    frame_path = episodes / episode_id / "frames.npz"
                    with np.load(frame_path) as archive:
                        payload = {key: archive[key] for key in archive.files}
                    payload["action_target_rad"] = np.full_like(
                        payload["action_target_rad"], 1e6
                    )
                    np.savez_compressed(frame_path, **payload)
                entries.append(
                    {
                        "episode_id": episode_id,
                        "split": split,
                        "day": f"day-{index}",
                        "object_instance": f"object-{index}",
                        "operator": f"operator-{index}",
                        "duration_seconds": 64 / 30,
                    }
                )
            split_path = root / "split.json"
            split_path.write_text(
                json.dumps(
                    {
                        "schema_version": "revo3-corpus-split-v1",
                        "episodes": entries,
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
                ),
                encoding="utf-8",
            )
            common = dict(
                tactile_profile="ablation_force6d_only",
                checkpoint_family_id="synthetic-ablation_force6d_only-v1",
                normalization_family_id="synthetic-ablation_force6d_only-norm-v1",
            )
            mid_json = root / "mid.json"
            mid_manifest = convert_revo_episodes_to_trex_json(
                episodes,
                mid_json,
                ConversionConfig(dataset_split="midtrain_train", **common),
                split_manifest=split_path,
            )
            stats_path = Path(mid_manifest["stats_path"])
            stats_artifact_path = Path(mid_manifest["stats_artifact_path"])
            artifact = json.loads(stats_artifact_path.read_text(encoding="utf-8"))
            self.assertEqual(artifact["source_split"], "midtrain_train")
            self.assertEqual(
                artifact["stats_episode_ids"],
                [episode_id for episode_id, split in zip(ids, splits) if split == "midtrain_train"],
            )
            with stats_path.open("r", encoding="utf-8") as handle:
                frozen = json.load(handle)["revo3_single_hand"]
            self.assertLess(float(np.asarray(frozen["action"]["q99"]).max()), 1e5)

            dev_json = root / "dev.json"
            with self.assertRaisesRegex(ValueError, "must load the frozen"):
                convert_revo_episodes_to_trex_json(
                    episodes,
                    dev_json,
                    ConversionConfig(dataset_split="development", **common),
                    split_manifest=split_path,
                )
            dev_manifest = convert_revo_episodes_to_trex_json(
                episodes,
                dev_json,
                ConversionConfig(dataset_split="development", **common),
                split_manifest=split_path,
                frozen_statistics_path=stats_path,
                frozen_statistics_artifact=stats_artifact_path,
            )
            self.assertEqual(dev_manifest["stats_path"], str(stats_path.resolve()))
            self.assertEqual(
                dev_manifest["statistics_sha256"],
                mid_manifest["statistics_sha256"],
            )
            self.assertFalse((root / "dev_statistics.json").exists())
            sft_manifest = convert_revo_episodes_to_trex_json(
                episodes,
                root / "sft.json",
                ConversionConfig(dataset_split="sft_train", **common),
                split_manifest=split_path,
                frozen_statistics_path=stats_path,
                frozen_statistics_artifact=stats_artifact_path,
            )
            self.assertEqual(
                sft_manifest["statistics_sha256"],
                mid_manifest["statistics_sha256"],
            )
            locked_manifest = convert_revo_episodes_to_trex_json(
                episodes,
                root / "locked.json",
                ConversionConfig(dataset_split="locked_test", **common),
                split_manifest=split_path,
                frozen_statistics_path=stats_path,
                frozen_statistics_artifact=stats_artifact_path,
            )
            for key in ("statistics_sha256", "statistics_artifact_sha256"):
                self.assertEqual(locked_manifest[key], mid_manifest[key])
            self.assertEqual(locked_manifest["statistics_source_split"], "midtrain_train")


if __name__ == "__main__":
    unittest.main()
