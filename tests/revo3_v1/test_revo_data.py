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
                SyntheticRevoConfig(episodes_per_task=1, frames_per_episode=20),
            )
            self.assertEqual(len(summary["episode_ids"]), 4)
            loaded = RevoEpisode.load(episodes / summary["episode_ids"][0])
            self.assertEqual(loaded.action_target_rad.shape, (20, 21))
            self.assertEqual(loaded.tactile_features.shape, (20, 5, 6))

            target = root / "revo3_trex_train.json"
            manifest = convert_revo_episodes_to_trex_json(
                episodes, target, ConversionConfig()
            )
            self.assertFalse(manifest["contains_emg"])
            self.assertEqual(manifest["action_shape"], [16, 21])
            with target.open("r", encoding="utf-8") as handle:
                rows = json.load(handle)
            self.assertEqual(len(rows), 80)
            self.assertTrue(all(not row["contains_emg"] for row in rows))
            self.assertTrue(all(len(row["action"]) == 16 * 21 for row in rows))
            self.assertTrue(all(len(row["tactile_f6"]) == 5 * 6 for row in rows))
            with (root / "revo3_trex_train_statistics.json").open(
                "r", encoding="utf-8"
            ) as handle:
                stats = json.load(handle)["revo3_single_hand"]
            self.assertEqual(np.asarray(stats["action"]["q01"]).shape, (16, 21))
            self.assertEqual(np.asarray(stats["state"]["q01"]).shape, (21,))
            self.assertEqual(np.asarray(stats["tactile_f6"]["q01"]).shape, (30,))


if __name__ == "__main__":
    unittest.main()
