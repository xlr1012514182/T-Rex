"""Wire contracts for Revo-specialized T-Rex inference."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
from typing import Mapping, Optional

import numpy as np

from revo3_v1.revo.contracts import JOINT_COUNT, assert_joint_vector


ACTION_DIM = JOINT_COUNT
ACTION_CHUNK = 16
TACTILE_FINGERS = 5
TACTILE_DIMS = 6
TACTILE_HISTORY = 16


class InferenceMode(str, Enum):
    NONE = "none"
    SLOW = "slow"
    FAST = "fast"
    SLOW_AND_FAST = "slow_and_fast"


@dataclass(frozen=True)
class TaskKey:
    task_id: str
    task_version: int
    instruction_hash: str
    lease_id: str

    def __post_init__(self) -> None:
        if not self.task_id or not self.lease_id or not self.instruction_hash:
            raise ValueError("task_id, lease_id and instruction_hash must be non-empty.")
        if self.task_version < 0:
            raise ValueError("task_version must be non-negative.")

    @classmethod
    def from_instruction(
        cls, *, task_id: str, task_version: int, instruction: str, lease_id: str
    ) -> "TaskKey":
        normalized = " ".join(instruction.strip().split())
        if not normalized:
            raise ValueError("instruction must be non-empty.")
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return cls(task_id, task_version, digest, lease_id)


@dataclass(frozen=True)
class PolicyObservation:
    """One causally aligned observation; no EMG tensor is part of this API."""

    timestamp_ns: int
    state_timestamp_ns: int
    rgb_timestamp_ns: int
    tactile_timestamp_ns: int
    q_rad: np.ndarray
    tactile_f6: np.ndarray
    tactile_history_f6: np.ndarray
    instruction: str
    task_key: TaskKey
    images: Optional[Mapping[str, np.ndarray]] = None

    def __post_init__(self) -> None:
        for name in (
            "timestamp_ns",
            "state_timestamp_ns",
            "rgb_timestamp_ns",
            "tactile_timestamp_ns",
        ):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative.")
            if value > self.timestamp_ns:
                raise ValueError(f"{name} cannot be later than observation timestamp_ns.")
        if not self.instruction.strip():
            raise ValueError("instruction must be non-empty.")
        expected_hash = TaskKey.from_instruction(
            task_id=self.task_key.task_id,
            task_version=self.task_key.task_version,
            instruction=self.instruction,
            lease_id=self.task_key.lease_id,
        ).instruction_hash
        if expected_hash != self.task_key.instruction_hash:
            raise ValueError("instruction does not match task_key.instruction_hash.")
        object.__setattr__(self, "q_rad", assert_joint_vector(self.q_rad, name="q_rad"))

        current = np.asarray(self.tactile_f6, dtype=np.float32)
        if current.shape != (TACTILE_FINGERS, TACTILE_DIMS):
            raise ValueError(
                f"tactile_f6 must have shape ({TACTILE_FINGERS},{TACTILE_DIMS}), "
                f"got {current.shape}."
            )
        history = np.asarray(self.tactile_history_f6, dtype=np.float32)
        expected = (TACTILE_HISTORY, TACTILE_FINGERS, TACTILE_DIMS)
        if history.shape != expected:
            raise ValueError(f"tactile_history_f6 must have shape {expected}, got {history.shape}.")
        if not np.isfinite(current).all() or not np.isfinite(history).all():
            raise ValueError("tactile input contains NaN or infinity.")
        object.__setattr__(self, "tactile_f6", current.copy())
        object.__setattr__(self, "tactile_history_f6", history.copy())

        if self.images is not None:
            copied = {}
            for name, image in self.images.items():
                arr = np.asarray(image)
                if arr.ndim != 3 or arr.shape[-1] != 3:
                    raise ValueError(f"image {name!r} must be HxWx3, got {arr.shape}.")
                copied[str(name)] = arr.copy()
            object.__setattr__(self, "images", copied)


@dataclass(frozen=True)
class PolicyRequest:
    mode: InferenceMode
    chunk_offset: int
    observation: PolicyObservation

    def __post_init__(self) -> None:
        if self.chunk_offset < 0 or self.chunk_offset >= ACTION_CHUNK:
            raise ValueError(f"chunk_offset must be in [0,{ACTION_CHUNK - 1}].")


@dataclass(frozen=True)
class ActionChunk:
    """A chunk of absolute, future Revo joint targets in radians."""

    q_target_rad: np.ndarray
    start_step: int
    generated_ns: int
    observation_timestamp_ns: int
    task_key: TaskKey
    mode: InferenceMode
    chunk_id: str

    def __post_init__(self) -> None:
        arr = np.asarray(self.q_target_rad, dtype=np.float32)
        expected = (ACTION_CHUNK, ACTION_DIM)
        if arr.shape != expected:
            raise ValueError(f"q_target_rad must have shape {expected}, got {arr.shape}.")
        if not np.isfinite(arr).all():
            raise ValueError("q_target_rad contains NaN or infinity.")
        if self.start_step < 0:
            raise ValueError("start_step must be non-negative.")
        if self.generated_ns < self.observation_timestamp_ns:
            raise ValueError("generated_ns cannot precede the observation timestamp.")
        if not self.chunk_id:
            raise ValueError("chunk_id must be non-empty.")
        object.__setattr__(self, "q_target_rad", arr.copy())
