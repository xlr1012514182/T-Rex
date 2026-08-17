import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from revo3_v1.emg.streaming import BinaryIntentGate, IntentGateConfig, StreamingEMGClassifier


class BinaryIntentGateTest(unittest.TestCase):
    def test_start_is_latched_and_open_releases(self):
        gate = BinaryIntentGate(
            IntentGateConfig(
                close_threshold=0.8,
                open_threshold=0.9,
                close_dwell_ms=100,
                open_dwell_ms=200,
                min_signal_quality=0.5,
            )
        )
        self.assertIsNone(gate.update(0.9, 0, 1.0))
        start = gate.update(0.9, 100_000_000, 1.0)
        self.assertEqual(start.event_type, "StartIntentEvent")
        self.assertTrue(gate.active)
        self.assertIsNone(gate.update(0.95, 150_000_000, 1.0))
        self.assertIsNone(gate.update(0.05, 200_000_000, 1.0))
        release = gate.update(0.05, 400_000_000, 1.0)
        self.assertEqual(release.event_type, "ReleaseEvent")
        self.assertFalse(gate.active)

    def test_bad_quality_and_ambiguous_windows_do_not_release(self):
        gate = BinaryIntentGate(
            IntentGateConfig(close_dwell_ms=0, open_dwell_ms=0, min_signal_quality=0.8)
        )
        self.assertEqual(gate.update(0.99, 1, 1.0).event_type, "StartIntentEvent")
        self.assertIsNone(gate.update(0.0, 2, 0.2))
        self.assertTrue(gate.active)
        self.assertIsNone(gate.update(0.5, 3, 1.0))
        self.assertTrue(gate.active)


@unittest.skipIf(torch is None, "PyTorch not installed")
class StreamingEMGClassifierTest(unittest.TestCase):
    def test_probability_output_contract(self):
        from revo3_v1.emg.model import GNIBinaryClassifier, GNIModelConfig

        model = GNIBinaryClassifier(GNIModelConfig.smoke(input_channels=4))
        streaming = StreamingEMGClassifier(
            model=model,
            normalization={
                "mean": np.zeros(4, dtype=np.float32),
                "std": np.ones(4, dtype=np.float32),
            },
            sample_rate_hz=200,
            window_samples=64,
            stride_samples=16,
        )
        self.assertIsNone(streaming.push(np.zeros((4, 32), dtype=np.float32), 1))
        result = streaming.push(np.zeros((4, 32), dtype=np.float32), 2)
        self.assertEqual(
            set(result),
            {
                "timestamp_ns",
                "probability_open",
                "probability_close",
                "signal_quality",
                "event",
            },
        )
        self.assertAlmostEqual(result["probability_open"] + result["probability_close"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
