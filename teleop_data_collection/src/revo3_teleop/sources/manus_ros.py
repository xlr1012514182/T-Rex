"""Lazy, injected MANUS ROS message adapter.

No MANUS or ROS package is imported by this module.  The official
``manus_ros2_msgs/msg/ManusGlove`` schema has no ``Header`` and the upstream
bridge discards MANUS ``publishTime``.  Consequently arrival time is recorded
as host evidence and ``device_timestamp_ns`` remains ``None``.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol, Sequence

import numpy as np

from revo3_teleop.contracts import NativeSample, SampleHeader

from .common import Clock, monotonic_ns, strict_nonnegative_int


DEFAULT_MANUS_TOPICS = ("/manus_glove_0", "/manus_glove_1")
MANUS_OFFICIAL_MESSAGE_HAS_DEVICE_TIMESTAMP = False
MANUS_CLOCK_DOMAIN = "host_ros_callback_arrival_monotonic"


class ManusRosAdapter(Protocol):
    def register_callback(self, topic: str, callback: Callable[[object], None]) -> None: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...


@dataclass(frozen=True)
class ManusRosFrame:
    """Numeric recorder sample plus text schema that NativeSample cannot hold."""

    sample: NativeSample
    topic: str
    side: str
    ergonomics_types: tuple[str, ...]
    raw_joint_types: tuple[str, ...]
    raw_chain_types: tuple[str, ...]
    provides_wrist_pose: bool
    official_device_timestamp_available: bool = MANUS_OFFICIAL_MESSAGE_HAS_DEVICE_TIMESTAMP


def _finite_vector(values: Sequence[object], *, name: str, size: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite vector of shape ({size},)")
    return array


def _pose_arrays(pose: object, *, name: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        position = _finite_vector(
            (pose.position.x, pose.position.y, pose.position.z), name=f"{name}.position", size=3
        )
        orientation = _finite_vector(
            (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w),
            name=f"{name}.orientation",
            size=4,
        )
    except AttributeError as exc:
        raise ValueError(f"{name} does not provide a ROS-like pose") from exc
    return position, orientation


class ManusRosMessageParser:
    """Duck-typed parser for the official ``ManusGlove`` field contract.

    A wrist pose is reported only when the caller supplies a verified MANUS
    wrist node id and that id occurs in ``raw_nodes`` with a finite pose.  The
    parser never treats glove IMU orientation or an arbitrary skeleton node as
    a wrist pose.
    """

    def __init__(
        self,
        *,
        source_prefix: str = "manus_ros",
        wrist_node_ids: Iterable[int] = (),
    ) -> None:
        self.source_prefix = source_prefix
        self.wrist_node_ids = frozenset(
            strict_nonnegative_int(value, name="wrist_node_id") for value in wrist_node_ids
        )
        self._host_sequences: defaultdict[str, int] = defaultdict(int)

    def parse(
        self,
        topic: str,
        msg: object,
        *,
        arrival_timestamp_ns: int,
    ) -> ManusRosFrame:
        topic = str(topic).strip()
        if not topic:
            raise ValueError("MANUS topic must be non-empty")
        arrival_ns = strict_nonnegative_int(arrival_timestamp_ns, name="arrival_timestamp_ns")
        try:
            glove_id = strict_nonnegative_int(msg.glove_id, name="MANUS glove_id")
            side = str(msg.side).strip().lower()
            raw_nodes = list(msg.raw_nodes)
            ergonomics = list(msg.ergonomics)
            raw_sensors = list(msg.raw_sensor)
            declared_nodes = strict_nonnegative_int(msg.raw_node_count, name="raw_node_count")
            declared_ergonomics = strict_nonnegative_int(msg.ergonomics_count, name="ergonomics_count")
            declared_sensors = strict_nonnegative_int(msg.raw_sensor_count, name="raw_sensor_count")
        except AttributeError as exc:
            raise ValueError("message does not match official ManusGlove fields") from exc
        if side not in {"left", "right", "l", "r"}:
            raise ValueError(f"unsupported MANUS side: {side!r}")
        side = "left" if side in {"left", "l"} else "right"
        if declared_nodes != len(raw_nodes):
            raise ValueError("raw_node_count disagrees with raw_nodes length")
        if declared_ergonomics != len(ergonomics):
            raise ValueError("ergonomics_count disagrees with ergonomics length")
        if declared_sensors != len(raw_sensors):
            raise ValueError("raw_sensor_count disagrees with raw_sensor length")

        node_ids: list[int] = []
        parent_ids: list[int] = []
        node_positions: list[np.ndarray] = []
        node_orientations: list[np.ndarray] = []
        joint_types: list[str] = []
        chain_types: list[str] = []
        for index, node in enumerate(raw_nodes):
            node_id = strict_nonnegative_int(node.node_id, name=f"raw_nodes[{index}].node_id")
            parent_id = int(node.parent_node_id)
            position, orientation = _pose_arrays(node.pose, name=f"raw_nodes[{index}].pose")
            node_ids.append(node_id)
            parent_ids.append(parent_id)
            node_positions.append(position)
            node_orientations.append(orientation)
            joint_types.append(str(node.joint_type))
            chain_types.append(str(node.chain_type))

        ergo_types: list[str] = []
        ergo_values: list[float] = []
        for index, item in enumerate(ergonomics):
            kind = str(item.type).strip()
            if not kind:
                raise ValueError(f"ergonomics[{index}].type must be non-empty")
            value = float(item.value)
            if not np.isfinite(value):
                raise ValueError(f"ergonomics[{index}].value is not finite")
            ergo_types.append(kind)
            ergo_values.append(value)

        sensor_positions: list[np.ndarray] = []
        sensor_orientations: list[np.ndarray] = []
        for index, pose in enumerate(raw_sensors):
            position, orientation = _pose_arrays(pose, name=f"raw_sensor[{index}]")
            sensor_positions.append(position)
            sensor_orientations.append(orientation)
        try:
            sensor_orientation = _finite_vector(
                (
                    msg.raw_sensor_orientation.x,
                    msg.raw_sensor_orientation.y,
                    msg.raw_sensor_orientation.z,
                    msg.raw_sensor_orientation.w,
                ),
                name="raw_sensor_orientation",
                size=4,
            )
        except AttributeError as exc:
            raise ValueError("raw_sensor_orientation is missing quaternion fields") from exc

        provides_wrist_pose = bool(self.wrist_node_ids.intersection(node_ids))
        sequence = self._host_sequences[topic]
        self._host_sequences[topic] += 1
        side_code = 0 if side == "left" else 1
        payload = {
            "glove_id": np.asarray([glove_id], dtype=np.int64),
            "side_code": np.asarray([side_code], dtype=np.uint8),
            "raw_node_ids": np.asarray(node_ids, dtype=np.int32),
            "raw_parent_node_ids": np.asarray(parent_ids, dtype=np.int32),
            "raw_node_positions": np.asarray(node_positions, dtype=np.float32).reshape((-1, 3)),
            "raw_node_orientations": np.asarray(node_orientations, dtype=np.float32).reshape((-1, 4)),
            "ergonomics_values": np.asarray(ergo_values, dtype=np.float32),
            "raw_sensor_orientation": sensor_orientation.astype(np.float32, copy=True),
            "raw_sensor_positions": np.asarray(sensor_positions, dtype=np.float32).reshape((-1, 3)),
            "raw_sensor_orientations": np.asarray(sensor_orientations, dtype=np.float32).reshape((-1, 4)),
            "host_arrival_timestamp_ns": np.asarray([arrival_ns], dtype=np.int64),
            "device_timestamp_available": np.asarray([0], dtype=np.uint8),
            "sequence_is_host_assigned": np.asarray([1], dtype=np.uint8),
            "provides_wrist_pose": np.asarray([int(provides_wrist_pose)], dtype=np.uint8),
        }
        sample = NativeSample(
            SampleHeader(
                source_id=f"{self.source_prefix}:{topic}",
                sequence=sequence,
                capture_timestamp_ns=arrival_ns,
                receive_timestamp_ns=arrival_ns,
                clock_domain=MANUS_CLOCK_DOMAIN,
                device_timestamp_ns=None,
            ),
            payload,
        )
        return ManusRosFrame(
            sample=sample,
            topic=topic,
            side=side,
            ergonomics_types=tuple(ergo_types),
            raw_joint_types=tuple(joint_types),
            raw_chain_types=tuple(chain_types),
            provides_wrist_pose=provides_wrist_pose,
        )


class ManusRosSource:
    """Lazy ROS boundary; no rclpy node is created until explicit ``start``."""

    hardware_autostart = False

    def __init__(
        self,
        *,
        adapter_factory: Callable[[], ManusRosAdapter] | None = None,
        allow_hardware_start: bool = False,
        topics: Sequence[str] = DEFAULT_MANUS_TOPICS,
        clock: Clock = monotonic_ns,
        parser: ManusRosMessageParser | None = None,
    ) -> None:
        normalized = tuple(str(topic).strip() for topic in topics)
        if not normalized or any(not topic for topic in normalized) or len(set(normalized)) != len(normalized):
            raise ValueError("MANUS topics must be unique, non-empty names")
        self.topics = normalized
        self._adapter_factory = adapter_factory
        self._allow_hardware_start = bool(allow_hardware_start)
        self._clock = clock
        self._parser = parser or ManusRosMessageParser()
        self._adapter: ManusRosAdapter | None = None
        self._queue: deque[ManusRosFrame] = deque()

    def start(self) -> None:
        if not self._allow_hardware_start:
            raise PermissionError("MANUS ROS start is disabled; opt in explicitly")
        if self._adapter_factory is None:
            raise RuntimeError("no injected MANUS ROS adapter factory")
        if self._adapter is not None:
            raise RuntimeError("MANUS ROS source is already started")
        adapter = self._adapter_factory()
        for topic in self.topics:
            adapter.register_callback(topic, lambda msg, topic=topic: self.ingest(topic, msg))
        adapter.start()
        self._adapter = adapter

    def stop(self) -> None:
        if self._adapter is None:
            return
        adapter, self._adapter = self._adapter, None
        adapter.stop()

    def ingest(
        self,
        topic: str,
        msg: object,
        *,
        arrival_timestamp_ns: int | None = None,
    ) -> ManusRosFrame:
        if topic not in self.topics:
            raise ValueError(f"unconfigured MANUS topic: {topic}")
        arrival_ns = self._clock() if arrival_timestamp_ns is None else int(arrival_timestamp_ns)
        frame = self._parser.parse(topic, msg, arrival_timestamp_ns=arrival_ns)
        self._queue.append(frame)
        return frame

    def drain(self) -> tuple[ManusRosFrame, ...]:
        frames = tuple(self._queue)
        self._queue.clear()
        return frames
