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
    version_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.task_id or not self.lease_id or not self.instruction_hash:
            raise ValueError("task_id, lease_id and instruction_hash must be non-empty.")
        if self.task_version < 0:
            raise ValueError("task_version must be non-negative.")

    @classmethod
    def from_instruction(
        cls,
        *,
        task_id: str,
        task_version: int,
        instruction: str,
        lease_id: str,
        version_fingerprint: str = "",
    ) -> "TaskKey":
        normalized = " ".join(instruction.strip().split())
        if not normalized:
            raise ValueError("instruction must be non-empty.")
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return cls(task_id, task_version, digest, lease_id, version_fingerprint)


@dataclass(frozen=True)
class PolicyObservation:
    """One causally aligned observation; no EMG tensor is part of this API."""

    timestamp_ns: int
    state_timestamp_ns: int
    rgb_timestamp_ns: int
    tactile_timestamp_ns: int
    q_rad: np.ndarray
    tactile_f6: Optional[np.ndarray]
    tactile_history_f6: Optional[np.ndarray]
    instruction: str
    task_key: TaskKey
    images: Optional[Mapping[str, np.ndarray]] = None
    tactile_history_timestamps_ns: Optional[np.ndarray] = None
    tactile_history_sequences: Optional[np.ndarray] = None
    tactile_deform: Optional[np.ndarray] = None
    tactile_deform_timestamp_ns: Optional[np.ndarray] = None
    tactile_deform_delayed: Optional[np.ndarray] = None
    tactile_deform_delayed_timestamps_ns: Optional[np.ndarray] = None
    lease_expires_at_ns: Optional[int] = None
    tactile_profile: str = "legacy_force6d"

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

        profile = str(self.tactile_profile)
        if profile == "profile_c_pressure_matrix":
            raise ValueError("Profile C is not launchable by the current T-Rex runtime.")
        if profile not in {
            "legacy_force6d",
            "profile_a_force6d_diff",
            "profile_b_diff_only",
            "ablation_force6d_only",
        }:
            raise ValueError("unknown tactile_profile")
        requires_force = profile in {
            "legacy_force6d",
            "profile_a_force6d_diff",
            "ablation_force6d_only",
        }
        if requires_force:
            if self.tactile_f6 is None or self.tactile_history_f6 is None:
                raise ValueError(f"{profile} requires Force6D current/history.")
            current = np.asarray(self.tactile_f6, dtype=np.float32)
            history = np.asarray(self.tactile_history_f6, dtype=np.float32)
            expected = (TACTILE_HISTORY, TACTILE_FINGERS, TACTILE_DIMS)
            if current.shape != (TACTILE_FINGERS, TACTILE_DIMS):
                raise ValueError(
                    f"tactile_f6 must have shape ({TACTILE_FINGERS},{TACTILE_DIMS}), "
                    f"got {current.shape}."
                )
            if history.shape != expected:
                raise ValueError(
                    f"tactile_history_f6 must have shape {expected}, got {history.shape}."
                )
            if not np.isfinite(current).all() or not np.isfinite(history).all():
                raise ValueError("tactile input contains NaN or infinity.")
            object.__setattr__(self, "tactile_f6", current.copy())
            object.__setattr__(self, "tactile_history_f6", history.copy())
        else:
            if self.tactile_f6 is not None or self.tactile_history_f6 is not None:
                raise ValueError("profile_b_diff_only forbids fake/unused Force6D tensors.")
            current = history = None

        history_timestamps = self.tactile_history_timestamps_ns
        history_sequences = self.tactile_history_sequences
        if (history_timestamps is None) != (history_sequences is None):
            raise ValueError(
                "tactile history timestamps and sequences must be supplied together."
            )
        if requires_force and history_timestamps is None:
            # Legacy constructors remain valid for mock/official paths; the
            # mainline ZMQ backend requires timestamps before network use.
            pass
        elif not requires_force and history_timestamps is not None:
            raise ValueError("profile_b_diff_only forbids Force6D history metadata.")
        elif history_timestamps is not None:
            timestamps = np.asarray(history_timestamps, dtype=np.int64)
            sequences = np.asarray(history_sequences, dtype=np.int64)
            if timestamps.shape != (TACTILE_HISTORY,) or sequences.shape != (TACTILE_HISTORY,):
                raise ValueError("tactile history timestamps/sequences must have shape (16,).")
            if np.any(np.diff(timestamps) <= 0) or np.any(np.diff(sequences) <= 0):
                raise ValueError("tactile history timestamps/sequences must be strictly increasing.")
            if int(timestamps[-1]) != int(self.tactile_timestamp_ns):
                raise ValueError("latest tactile history timestamp must equal tactile_timestamp_ns.")
            if not np.array_equal(history[-1], current):
                raise ValueError("latest tactile history value must equal tactile_f6.")
            object.__setattr__(self, "tactile_history_timestamps_ns", timestamps.copy())
            object.__setattr__(self, "tactile_history_sequences", sequences.copy())

        deform = self.tactile_deform
        deform_ts = self.tactile_deform_timestamp_ns
        if (deform is None) != (deform_ts is None):
            raise ValueError("tactile_deform and its per-finger timestamps are inseparable.")
        if deform is not None:
            image = np.asarray(deform)
            timestamps = np.asarray(deform_ts, dtype=np.int64)
            if image.shape != (TACTILE_FINGERS, 240, 240):
                raise ValueError("tactile_deform must have shape (5,240,240).")
            if image.dtype != np.uint8:
                raise ValueError("tactile_deform must be uint8 DIFF images.")
            if timestamps.shape != (TACTILE_FINGERS,):
                raise ValueError("tactile_deform_timestamp_ns must have shape (5,).")
            if np.any(timestamps < 0) or np.any(timestamps > self.timestamp_ns):
                raise ValueError("tactile DIFF timestamps must be causal.")
            object.__setattr__(self, "tactile_deform", image.copy())
            object.__setattr__(self, "tactile_deform_timestamp_ns", timestamps.copy())
        if profile in {"profile_a_force6d_diff", "profile_b_diff_only"} and deform is None:
            raise ValueError(f"{profile} requires five current DIFF images.")
        if profile in {"legacy_force6d", "ablation_force6d_only"} and deform is not None:
            raise ValueError(f"{profile} forbids unused DIFF images.")
        delayed = self.tactile_deform_delayed
        delayed_ts = self.tactile_deform_delayed_timestamps_ns
        if (delayed is None) != (delayed_ts is None):
            raise ValueError("delayed tactile DIFF images and timestamps are inseparable.")
        if delayed is not None:
            images = np.asarray(delayed)
            timestamps = np.asarray(delayed_ts, dtype=np.int64)
            if images.shape != (4, TACTILE_FINGERS, 240, 240) or images.dtype != np.uint8:
                raise ValueError("tactile_deform_delayed must be uint8 (4,5,240,240).")
            if timestamps.shape != (4, TACTILE_FINGERS):
                raise ValueError("delayed tactile DIFF timestamps must be (4,5).")
            if np.any(timestamps < 0) or np.any(timestamps > self.timestamp_ns):
                raise ValueError("delayed tactile DIFF timestamps must be causal.")
            if np.any(np.diff(timestamps, axis=0) < 0):
                raise ValueError("delayed tactile DIFF groups must be chronological.")
            object.__setattr__(self, "tactile_deform_delayed", images.copy())
            object.__setattr__(self, "tactile_deform_delayed_timestamps_ns", timestamps.copy())
        # Delayed DIFF groups remain a training-side supervision artifact.
        # Online inference uses only the newest five-finger DIFF frame.  The
        # paired legacy fields above stay accepted for backward compatibility,
        # but are intentionally not a runtime requirement.
        if self.lease_expires_at_ns is not None:
            expiry = int(self.lease_expires_at_ns)
            if expiry <= self.timestamp_ns:
                raise ValueError("lease must expire after the observation timestamp.")
            object.__setattr__(self, "lease_expires_at_ns", expiry)

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
