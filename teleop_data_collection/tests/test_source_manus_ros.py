from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from revo3_teleop.sources.manus_ros import (
    MANUS_OFFICIAL_MESSAGE_HAS_DEVICE_TIMESTAMP,
    ManusRosMessageParser,
    ManusRosSource,
)


@dataclass
class Vec3:
    x: float
    y: float
    z: float


@dataclass
class Quat:
    x: float
    y: float
    z: float
    w: float


def pose(x: float = 0.0):
    return SimpleNamespace(position=Vec3(x, x + 1, x + 2), orientation=Quat(0, 0, 0, 1))


def manus_message(*, nodes=True, side="Left"):
    raw_nodes = (
        [SimpleNamespace(node_id=5, parent_node_id=-1, joint_type="Wrist", chain_type="Arm", pose=pose(1))]
        if nodes
        else []
    )
    ergonomics = [SimpleNamespace(type="ThumbMCPSpread", value=12.5)]
    sensors = [pose(3)]
    return SimpleNamespace(
        glove_id=42,
        side=side,
        raw_node_count=len(raw_nodes),
        raw_nodes=raw_nodes,
        ergonomics_count=len(ergonomics),
        ergonomics=ergonomics,
        raw_sensor_orientation=Quat(0.1, 0.2, 0.3, 0.9),
        raw_sensor_count=len(sensors),
        raw_sensor=sensors,
    )


class FakeManusAdapter:
    def __init__(self) -> None:
        self.callbacks = {}
        self.start_count = 0
        self.stop_count = 0

    def register_callback(self, topic, callback) -> None:
        self.callbacks[topic] = callback

    def start(self) -> None:
        self.start_count += 1

    def stop(self) -> None:
        self.stop_count += 1


def test_official_message_uses_arrival_time_and_does_not_invent_device_time() -> None:
    assert MANUS_OFFICIAL_MESSAGE_HAS_DEVICE_TIMESTAMP is False
    parser = ManusRosMessageParser(wrist_node_ids=(5,))
    frame = parser.parse("/manus_glove_0", manus_message(), arrival_timestamp_ns=6_000_000_000)

    assert frame.topic == "/manus_glove_0"
    assert frame.side == "left"
    assert frame.sample.header.capture_timestamp_ns == 6_000_000_000
    assert frame.sample.header.receive_timestamp_ns == 6_000_000_000
    assert frame.sample.header.device_timestamp_ns is None
    assert frame.sample.payload["host_arrival_timestamp_ns"].item() == 6_000_000_000
    assert frame.sample.payload["device_timestamp_available"].item() == 0
    assert frame.sample.payload["sequence_is_host_assigned"].item() == 1
    assert frame.ergonomics_types == ("ThumbMCPSpread",)
    assert frame.raw_joint_types == ("Wrist",)
    assert frame.raw_chain_types == ("Arm",)
    np.testing.assert_allclose(frame.sample.payload["raw_node_positions"], [[1, 2, 3]])
    np.testing.assert_allclose(frame.sample.payload["raw_sensor_positions"], [[3, 4, 5]])


def test_wrist_pose_capability_requires_verified_raw_node_id() -> None:
    message = manus_message(nodes=True)
    no_mapping = ManusRosMessageParser().parse(
        "/manus_glove_0", message, arrival_timestamp_ns=7_000_000_000
    )
    wrong_mapping = ManusRosMessageParser(wrist_node_ids=(99,)).parse(
        "/manus_glove_0", message, arrival_timestamp_ns=7_000_000_000
    )
    verified_mapping = ManusRosMessageParser(wrist_node_ids=(5,)).parse(
        "/manus_glove_0", message, arrival_timestamp_ns=7_000_000_000
    )

    assert no_mapping.provides_wrist_pose is False
    assert wrong_mapping.provides_wrist_pose is False
    assert verified_mapping.provides_wrist_pose is True
    assert verified_mapping.sample.payload["provides_wrist_pose"].item() == 1


def test_manus_counts_and_topics_are_strict() -> None:
    parser = ManusRosMessageParser()
    bad = manus_message()
    bad.raw_node_count = 2
    with pytest.raises(ValueError, match="raw_node_count"):
        parser.parse("/manus_glove_0", bad, arrival_timestamp_ns=8_000_000_000)

    source = ManusRosSource(topics=("/configured",))
    with pytest.raises(ValueError, match="unconfigured"):
        source.ingest("/other", manus_message(), arrival_timestamp_ns=8_000_000_000)


def test_manus_adapter_is_lazy_opt_in_and_fake_callback_is_recorded() -> None:
    fake = FakeManusAdapter()
    factory_calls = 0

    def factory() -> FakeManusAdapter:
        nonlocal factory_calls
        factory_calls += 1
        return fake

    disabled = ManusRosSource(adapter_factory=factory, topics=("/left",))
    with pytest.raises(PermissionError, match="disabled"):
        disabled.start()
    assert factory_calls == 0

    enabled = ManusRosSource(
        adapter_factory=factory,
        allow_hardware_start=True,
        topics=("/left", "/right"),
        clock=lambda: 9_000_000_000,
    )
    enabled.start()
    assert factory_calls == 1
    assert fake.start_count == 1
    assert set(fake.callbacks) == {"/left", "/right"}
    fake.callbacks["/left"](manus_message(side="Left"))
    fake.callbacks["/right"](manus_message(side="Right"))
    frames = enabled.drain()
    assert [frame.topic for frame in frames] == ["/left", "/right"]
    # Host-assigned counters are per topic because the official message has
    # neither a sequence field nor a device timestamp.
    assert [frame.sample.header.sequence for frame in frames] == [0, 0]
    enabled.stop()
    assert fake.stop_count == 1
