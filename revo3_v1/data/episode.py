"""On-disk Revo3 episode contract.

The action field is the absolute joint target passed to the controller.  It is
not an expert latent, prediction head output, or state-derived proxy.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Tuple

import numpy as np

from revo3_v1.revo.contracts import JOINT_COUNT, JOINT_ORDER_HASH


SUPPORTED_TASKS = (
    "bottle",
    "phone",
    "plastic_bag",
    "refrigerator_door",
)


@dataclass(frozen=True)
class RevoEpisodeMeta:
    episode_id: str
    task: str
    instruction: str
    fps: int
    joint_order_hash: str
    tactile_num_fingers: int
    action_label_source: str
    image_paths: Tuple[str, ...]

    @classmethod
    def load(cls, path: Path) -> "RevoEpisodeMeta":
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return cls(
            episode_id=str(value["episode_id"]),
            task=str(value["task"]),
            instruction=str(value["instruction"]),
            fps=int(value["fps"]),
            joint_order_hash=str(value["joint_order_hash"]),
            tactile_num_fingers=int(value["tactile_num_fingers"]),
            action_label_source=str(value["action_label_source"]),
            image_paths=tuple(str(item) for item in value["image_paths"]),
        )


@dataclass(frozen=True)
class RevoEpisode:
    root: Path
    meta: RevoEpisodeMeta
    timestamp_ns: np.ndarray
    state_rad: np.ndarray
    action_target_rad: np.ndarray
    tactile_features: np.ndarray

    @classmethod
    def load(cls, root: str | Path) -> "RevoEpisode":
        episode_root = Path(root)
        meta = RevoEpisodeMeta.load(episode_root / "meta.json")
        with np.load(episode_root / "frames.npz", allow_pickle=False) as archive:
            result = cls(
                root=episode_root,
                meta=meta,
                timestamp_ns=np.asarray(archive["timestamp_ns"], dtype=np.int64),
                state_rad=np.asarray(archive["state_rad"], dtype=np.float32),
                action_target_rad=np.asarray(archive["action_target_rad"], dtype=np.float32),
                tactile_features=np.asarray(archive["tactile_features"], dtype=np.float32),
            )
        result.validate()
        return result

    @property
    def num_frames(self) -> int:
        return int(self.timestamp_ns.shape[0])

    def validate(self) -> None:
        n = self.num_frames
        if self.meta.task not in SUPPORTED_TASKS:
            raise ValueError(f"unsupported Revo task: {self.meta.task}")
        if self.meta.fps <= 0 or not self.meta.episode_id or not self.meta.instruction:
            raise ValueError("episode metadata is incomplete")
        if self.meta.joint_order_hash != JOINT_ORDER_HASH:
            raise ValueError("episode joint order does not match the Revo3 canonical order")
        if self.meta.action_label_source not in {"controller_target", "executed_joint_target"}:
            raise ValueError("action labels must come from a controller/execution boundary")
        if self.timestamp_ns.shape != (n,) or n < 2:
            raise ValueError("episode timestamps must contain at least two frames")
        if np.any(np.diff(self.timestamp_ns) <= 0):
            raise ValueError("episode timestamps must be strictly monotonic")
        if self.state_rad.shape != (n, JOINT_COUNT):
            raise ValueError(f"state_rad must have shape [N,{JOINT_COUNT}]")
        if self.action_target_rad.shape != (n, JOINT_COUNT):
            raise ValueError(f"action_target_rad must have shape [N,{JOINT_COUNT}]")
        expected_tactile = (n, self.meta.tactile_num_fingers, 6)
        if self.tactile_features.shape != expected_tactile:
            raise ValueError(f"tactile_features must have shape {expected_tactile}")
        for name, values in (
            ("state_rad", self.state_rad),
            ("action_target_rad", self.action_target_rad),
            ("tactile_features", self.tactile_features),
        ):
            if not np.isfinite(values).all():
                raise ValueError(f"{name} contains NaN or infinity")
        if len(self.meta.image_paths) != n:
            raise ValueError("image path count does not match timestamps")
        missing = [path for path in self.meta.image_paths if not (self.root / path).is_file()]
        if missing:
            raise FileNotFoundError(f"episode has missing RGB frames, first={missing[0]}")
