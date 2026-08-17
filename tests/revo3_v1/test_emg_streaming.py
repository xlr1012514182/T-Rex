import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from revo3_v1.emg.streaming import (
    BinaryIntentGate,
    IntentGateConfig,
    MulticlassIntentGate,
    StreamingEMGClassifier,
)


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
        self.assertIsNone(gate.update(0.05, 300_000_000, 1.0))
        release = gate.update(0.05, 400_000_000, 1.0)
        self.assertEqual(release.event_type, "ReleaseEvent")
        self.assertFalse(gate.active)

    def test_multiclass_margin_rest_and_release_semantics(self):
        gate = MulticlassIntentGate(
            IntentGateConfig(close_dwell_ms=100, open_dwell_ms=200)
        )
        high_power = {
            "POWER_GRASP": 0.82,
            "PRECISION_GRASP": 0.05,
            "LATERAL_GRASP": 0.03,
            "RELEASE": 0.02,
            "REST": 0.08,
        }
        self.assertIsNone(gate.update(high_power, 0).event)
        start = gate.update(high_power, 100_000_000)
        self.assertEqual(start.event.primitive, "POWER_GRASP")
        self.assertGreaterEqual(start.event.margin, 0.2)
        rest = gate.update({**high_power, "POWER_GRASP": 0.05, "REST": 0.85}, 150_000_000)
        self.assertEqual(rest.primitive.value, "REST")
        self.assertTrue(gate.active)
        release_probs = {**high_power, "POWER_GRASP": 0.02, "REST": 0.01, "RELEASE": 0.94}
        self.assertIsNone(gate.update(release_probs, 200_000_000).event)
        self.assertIsNone(gate.update(release_probs, 300_000_000).event)
        released = gate.update(release_probs, 400_000_000)
        self.assertEqual(released.event.event_type, "ReleaseEvent")

    def test_bad_signal_and_low_margin_are_not_commands(self):
        gate = MulticlassIntentGate(IntentGateConfig(close_dwell_ms=0))
        probs = {"POWER_GRASP": 0.55, "PRECISION_GRASP": 0.40, "REST": 0.05}
        self.assertEqual(gate.update(probs, 1).primitive.value, "UNKNOWN")
        self.assertEqual(gate.update({"POWER_GRASP": 0.9, "REST": 0.1}, 2, 0.2).primitive.value, "BAD_SIGNAL")

    def test_bad_quality_and_ambiguous_windows_do_not_release(self):
        gate = BinaryIntentGate(
            IntentGateConfig(close_dwell_ms=0, open_dwell_ms=0, min_signal_quality=0.8)
        )
        self.assertEqual(gate.update(0.99, 1, 1.0).event_type, "StartIntentEvent")
        self.assertIsNone(gate.update(0.0, 2, 0.2))
        self.assertTrue(gate.active)
        self.assertIsNone(gate.update(0.5, 3, 1.0))
        self.assertTrue(gate.active)

    def test_dropped_updates_do_not_satisfy_dwell(self):
        gate = MulticlassIntentGate(
            IntentGateConfig(close_dwell_ms=300, max_update_gap_ms=100)
        )
        probabilities = {"POWER_GRASP": 0.95, "REST": 0.05}
        assert gate.update(probabilities, 0).event is None
        # A 1 s packet gap resets the candidate instead of satisfying dwell.
        assert gate.update(probabilities, 1_000_000_000).event is None
        assert gate.update(probabilities, 1_100_000_000).event is None
        assert gate.update(probabilities, 1_200_000_000).event is None
        assert gate.update(probabilities, 1_300_000_000).event.event_type == "StartIntentEvent"


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
            preprocessing_profile=None,
            allow_unprofiled_fixture=True,
        )
        self.assertIsNone(streaming.push(np.zeros((4, 32), dtype=np.float32), 155_000_000))
        result = streaming.push(np.zeros((4, 32), dtype=np.float32), 315_000_000)
        self.assertTrue({"timestamp_ns", "probabilities", "primitive", "confidence", "margin", "signal_quality", "event"}.issubset(result))
        self.assertAlmostEqual(result["probability_open"] + result["probability_close"], 1.0, places=6)

    def test_brainco_packets_preserve_twenty_hz_sample_clock_cadence(self):
        from revo3_v1.emg.model import GNIClassifier, GNIModelConfig
        from revo3_v1.emg.preprocessing import BRAINCO_EDU_8CH_250HZ

        streaming = StreamingEMGClassifier(
            model=GNIClassifier(GNIModelConfig.smoke(8, mainline=True)),
            normalization={
                "mean": np.zeros(8, dtype=np.float32),
                "std": np.ones(8, dtype=np.float32),
            },
            sample_rate_hz=250,
            window_samples=500,
            stride_samples=12,
            preprocessing_profile=BRAINCO_EDU_8CH_250HZ,
            channel_order=BRAINCO_EDU_8CH_250HZ.channel_order,
        )
        outputs = []
        epoch = 10_000_000_000
        for packet in range(50):
            indices = np.arange(packet * 20, packet * 20 + 20, dtype=np.int64)
            outputs.extend(streaming.push_many(
                np.zeros((8, 20), dtype=np.float32),
                epoch + indices * 4_000_000,
            ))
        timestamps = np.asarray([row["timestamp_ns"] for row in outputs], dtype=np.int64)
        self.assertEqual(len(outputs), 41)
        self.assertTrue(np.all(np.diff(timestamps) > 0))
        self.assertEqual(set(np.diff(timestamps).tolist()), {48_000_000, 52_000_000})
        self.assertAlmostEqual(float(np.mean(np.diff(timestamps))), 50_000_000.0)


if __name__ == "__main__":
    unittest.main()
