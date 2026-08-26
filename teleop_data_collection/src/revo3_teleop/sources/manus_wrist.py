"""Explicit MANUS raw-node to verified wrist-pose bridge.

The public MANUS ROS message does not identify which raw node is the wrist for
this project.  The integrator must provide and sign off the node id, side,
position scale and frame name.  No arbitrary skeleton node or glove IMU is
accepted as a wrist pose.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from revo3_teleop.retargeting import WristPose6D, WristPoseError

from .manus_ros import ManusRosFrame


@dataclass(frozen=True)
class VerifiedManusWristConfig:
    wrist_node_id: int
    side: str
    source_frame: str
    calibration_revision: str
    position_scale_to_m: float
    wrist_node_mapping_verified: bool = False

    def __post_init__(self) -> None:
        if self.wrist_node_id < 0:
            raise ValueError("wrist_node_id must be non-negative")
        side = str(self.side).strip().lower()
        if side not in {"left", "right"}:
            raise ValueError("side must be left or right")
        object.__setattr__(self, "side", side)
        if not str(self.source_frame).strip() or not str(self.calibration_revision).strip():
            raise ValueError("source_frame/calibration_revision must be non-empty")
        scale = float(self.position_scale_to_m)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("position_scale_to_m must be finite and positive")
        object.__setattr__(self, "position_scale_to_m", scale)


class ManusWristPoseExtractor:
    provides_wrist_pose = True

    def __init__(self, config: VerifiedManusWristConfig) -> None:
        if not config.wrist_node_mapping_verified:
            raise WristPoseError("MANUS wrist node mapping has not been verified")
        self.config = config

    def extract(self, frame: ManusRosFrame) -> WristPose6D:
        config = self.config
        if frame.side != config.side:
            raise WristPoseError("MANUS side does not match verified wrist mapping")
        if not frame.provides_wrist_pose:
            raise WristPoseError("MANUS parser did not verify a wrist node in this frame")
        payload = frame.sample.payload
        node_ids = np.asarray(payload["raw_node_ids"], dtype=np.int64)
        matches = np.flatnonzero(node_ids == config.wrist_node_id)
        if matches.size != 1:
            raise WristPoseError("verified MANUS wrist node is missing or duplicated")
        index = int(matches[0])
        position = (
            np.asarray(payload["raw_node_positions"], dtype=np.float64)[index]
            * config.position_scale_to_m
        )
        orientation = np.asarray(
            payload["raw_node_orientations"], dtype=np.float64
        )[index]
        header = frame.sample.header
        if not header.valid:
            raise WristPoseError("MANUS source marked the frame invalid")
        return WristPose6D(
            source_id=header.source_id,
            source_frame=config.source_frame,
            capture_timestamp_ns=header.capture_timestamp_ns,
            receive_timestamp_ns=header.receive_timestamp_ns,
            clock_domain=header.clock_domain,
            position_m=position,
            quaternion_xyzw=orientation,
            calibration_revision=config.calibration_revision,
            confidence=1.0,
            valid=True,
        )


class LatestManusWristPoseProvider:
    """Small stateful provider for an injected ROS callback/collector loop."""

    provides_wrist_pose = True

    def __init__(self, extractor: ManusWristPoseExtractor) -> None:
        self.extractor = extractor
        self._latest: WristPose6D | None = None

    def ingest(self, frame: ManusRosFrame) -> WristPose6D:
        pose = self.extractor.extract(frame)
        if self._latest is not None and (
            pose.capture_timestamp_ns <= self._latest.capture_timestamp_ns
        ):
            raise WristPoseError("MANUS wrist-pose timestamps must be strictly increasing")
        self._latest = pose
        return pose

    def read_pose(self) -> WristPose6D:
        if self._latest is None:
            raise WristPoseError("no verified MANUS wrist pose has been received")
        return self._latest


__all__ = [
    "LatestManusWristPoseProvider",
    "ManusWristPoseExtractor",
    "VerifiedManusWristConfig",
]
