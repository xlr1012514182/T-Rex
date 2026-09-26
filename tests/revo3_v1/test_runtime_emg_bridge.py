from __future__ import annotations

import numpy as np
import pytest

from revo3_v1.executive import EmgIntent
from revo3_v1.runtime import StreamingEMGEventBridge, streaming_result_to_emg_event


def output(
    primitive: str,
    timestamp_ns: int,
    *,
    event_type: str | None = None,
    event_id: str = "",
) -> dict[str, object]:
    event = None
    if event_type is not None:
        event = {
            "type": event_type,
            "primitive": primitive,
            "confidence": 0.95,
            "margin": 0.70,
            "signal_quality": 0.98,
            "timestamp_ns": timestamp_ns,
            "event_id": event_id,
        }
    return {
        "primitive": primitive,
        "confidence": 0.95,
        "margin": 0.70,
        "signal_quality": 0.98,
        "timestamp_ns": timestamp_ns,
        "event": event,
    }


class FakeStreamingSource:
    def __init__(self, batches: list[list[dict[str, object]]]) -> None:
        self.batches = list(batches)
        self.reset_active: list[bool] = []

    def push_many(self, samples, sample_timestamps_ns, signal_quality=1.0):
        del samples, sample_timestamps_ns, signal_quality
        return self.batches.pop(0)

    def reset(self, active=False):
        self.reset_active.append(bool(active))


def test_probability_during_dwell_cannot_bypass_streaming_event_gate() -> None:
    source = FakeStreamingSource(
        [
            [output("POWER_GRASP", 100)],
            [output("POWER_GRASP", 200, event_type="StartIntentEvent", event_id="s1")],
            [output("POWER_GRASP", 300)],
            [output("RELEASE", 800, event_type="ReleaseEvent", event_id="r1")],
        ]
    )
    bridge = StreamingEMGEventBridge(source)
    packet = np.zeros((8, 20), np.float32)
    timestamps = np.arange(20, dtype=np.int64)

    bridge.push_many(packet, timestamps)
    assert bridge.event_for_tick(now_ns=100).intent is EmgIntent.UNKNOWN

    bridge.push_many(packet, timestamps)
    start = bridge.event_for_tick(now_ns=200)
    assert start.intent is EmgIntent.POWER_GRASP
    assert start.event_id == "s1"

    bridge.push_many(packet, timestamps)
    assert bridge.event_for_tick(now_ns=300).intent is EmgIntent.UNKNOWN

    bridge.push_many(packet, timestamps)
    release = bridge.event_for_tick(now_ns=800)
    assert release.intent is EmgIntent.RELEASE
    assert release.event_id == "r1"
    assert bridge.pending_event_count == 0


def test_streaming_bridge_rejects_actionable_class_without_correct_edge_type() -> None:
    malformed = output(
        "POWER_GRASP", 100, event_type="ReleaseEvent", event_id="wrong"
    )
    with pytest.raises(ValueError, match="StartIntentEvent"):
        streaming_result_to_emg_event(malformed)


def test_streaming_bridge_consumes_causal_edges_one_at_a_time() -> None:
    source = FakeStreamingSource(
        [[
            output("PRECISION_GRASP", 100, event_type="StartIntentEvent", event_id="s"),
            output("RELEASE", 700, event_type="ReleaseEvent", event_id="r"),
        ]]
    )
    bridge = StreamingEMGEventBridge(source)
    bridge.push_many(np.zeros((8, 1), np.float32), np.asarray([1], np.int64))
    assert bridge.event_for_tick(now_ns=500).event_id == "s"
    assert bridge.event_for_tick(now_ns=500).intent is EmgIntent.UNKNOWN
    assert bridge.event_for_tick(now_ns=700).event_id == "r"
