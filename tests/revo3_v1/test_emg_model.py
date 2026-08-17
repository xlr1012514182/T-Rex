from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch not installed")
class GNIModelTest(unittest.TestCase):
    def test_forward_and_strict_checkpoint_round_trip(self):
        from revo3_v1.emg.model import (
            GNIBinaryClassifier,
            GNIModelConfig,
            load_emg_checkpoint,
            save_emg_checkpoint,
        )

        config = GNIModelConfig.smoke(input_channels=4)
        model = GNIBinaryClassifier(config).eval()
        inputs = torch.randn(2, 4, 64)
        logits = model(inputs)
        sequence = model.forward_sequence(inputs)
        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertEqual(sequence.shape[0], 2)
        self.assertEqual(sequence.shape[-1], 2)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            save_emg_checkpoint(
                path,
                model,
                {"mean": np.zeros(4, dtype=np.float32), "std": np.ones(4, dtype=np.float32)},
                {"test": True},
            )
            restored, payload = load_emg_checkpoint(path)
            with torch.no_grad():
                torch.testing.assert_close(model(inputs), restored(inputs))
            self.assertEqual(payload["schema_version"], "revo3-emg-checkpoint-v2")

    def test_mainline_head_has_five_learned_classes(self):
        from revo3_v1.emg.model import GNIClassifier, GNIModelConfig

        config = GNIModelConfig.smoke(input_channels=4, mainline=True)
        model = GNIClassifier(config).eval()
        self.assertEqual(config.resolved_labels(), (
            "POWER_GRASP", "PRECISION_GRASP", "LATERAL_GRASP", "RELEASE", "REST"
        ))
        self.assertEqual(tuple(model(torch.randn(2, 4, 64)).shape), (2, 5))


if __name__ == "__main__":
    unittest.main()
