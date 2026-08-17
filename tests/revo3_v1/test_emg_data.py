import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from revo3_v1.emg.data import fit_train_normalization, load_manifest, verify_split_manifests
from revo3_v1.emg.synthetic import SyntheticEMGConfig, generate_synthetic_dataset


class SyntheticEMGDataTest(unittest.TestCase):
    def test_subject_exclusive_split_and_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = SyntheticEMGConfig(
                n_subjects=5,
                sessions_per_subject=2,
                windows_per_label_per_session=2,
                n_channels=4,
                sample_rate_hz=200,
                window_seconds=0.5,
                window_stride_seconds=0.6,
                seed=17,
            )
            metadata = generate_synthetic_dataset(root, config)
            manifests = {name: root / "manifests" / f"{name}.jsonl" for name in ("train", "val", "test")}
            verify_split_manifests(manifests)
            subject_sets = {
                name: {str(row["subject_id"]) for row in load_manifest(path)}
                for name, path in manifests.items()
            }
            self.assertTrue(subject_sets["train"].isdisjoint(subject_sets["val"]))
            self.assertTrue(subject_sets["train"].isdisjoint(subject_sets["test"]))
            self.assertTrue(subject_sets["val"].isdisjoint(subject_sets["test"]))
            with np.load(root / "windows.npz", allow_pickle=False) as archive:
                self.assertEqual(archive["signal"].shape[1:], (4, 100))
                self.assertEqual(set(archive["label"].tolist()), {0, 1})
            self.assertEqual(metadata["split_unit"], "subject_id")
            for name, rows_path in manifests.items():
                rows = load_manifest(rows_path)
                self.assertTrue(all(int(row["window_end_ns"]) > int(row["window_start_ns"]) for row in rows))

    def test_normalization_uses_only_selected_indices(self):
        signals = np.zeros((3, 2, 4), dtype=np.float32)
        signals[0] = 1.0
        signals[1] = 3.0
        signals[2] = 1000.0
        stats = fit_train_normalization(signals, [0, 1])
        np.testing.assert_allclose(stats["mean"], [2.0, 2.0])
        self.assertTrue(np.all(stats["std"] < 2.0))


if __name__ == "__main__":
    unittest.main()
