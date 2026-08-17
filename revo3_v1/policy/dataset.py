"""Training/data contract for Revo-specific T-Rex midtrain and SFT.

The stock T-Rex LeRobot helpers are intentionally *not* reused: they hard-code
62-D bimanual actions and ``[10,6]`` F6.  This adapter emits the analogous
single-hand keys with 21-D state/action and ``[5,6]`` F6, without padding a
fake second hand.  A Revo trainer/loader should consume these items directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Mapping, Optional

import numpy as np

from revo3_v1.revo.contracts import JOINT_COUNT

from .contracts import ACTION_CHUNK, TACTILE_DIMS, TACTILE_FINGERS, TACTILE_HISTORY


KEY_STATE = "observation.state"
KEY_ACTION = "action"
KEY_ACTION_ABS = "action_abs"
KEY_TACTILE_F6 = "observation.tactile_f6"
KEY_IMAGE_FULL = "observation.images.full"
KEY_IMAGE_CENTER = "observation.images.center"


def build_revo_feature_schema(
    *, image_shape_chw: tuple[int, int, int] = (3, 288, 384)
) -> dict[str, dict[str, object]]:
    """Return the intended Revo LeRobot-style feature schema."""

    return {
        KEY_STATE: {"dtype": "float32", "shape": (JOINT_COUNT,)},
        KEY_ACTION: {"dtype": "float32", "shape": (ACTION_CHUNK, JOINT_COUNT)},
        KEY_ACTION_ABS: {"dtype": "float32", "shape": (JOINT_COUNT,)},
        KEY_TACTILE_F6: {
            "dtype": "float32",
            "shape": (TACTILE_FINGERS, TACTILE_DIMS),
        },
        KEY_IMAGE_FULL: {"dtype": "video", "shape": image_shape_chw},
        KEY_IMAGE_CENTER: {"dtype": "video", "shape": image_shape_chw},
    }


def _time_vector(value: np.ndarray, *, length: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.int64)
    if arr.shape != (length,):
        raise ValueError(f"timestamp_ns must have shape ({length},), got {arr.shape}.")
    if np.any(np.diff(arr) <= 0):
        raise ValueError("episode timestamps must be strictly increasing.")
    return arr.copy()


@dataclass(frozen=True)
class RevoEpisode:
    """One cleaned, already time-aligned Revo episode."""

    episode_id: str
    timestamp_ns: np.ndarray
    q_state_rad: np.ndarray
    q_action_abs_rad: np.ndarray
    tactile_f6: np.ndarray
    instruction: str
    full_rgb: Optional[np.ndarray] = None
    center_rgb: Optional[np.ndarray] = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.episode_id or not self.instruction.strip():
            raise ValueError("episode_id and instruction must be non-empty.")
        q_state = np.asarray(self.q_state_rad, dtype=np.float32)
        q_action = np.asarray(self.q_action_abs_rad, dtype=np.float32)
        touch = np.asarray(self.tactile_f6, dtype=np.float32)
        if q_state.ndim != 2 or q_state.shape[1] != JOINT_COUNT:
            raise ValueError(f"q_state_rad must be [T,{JOINT_COUNT}], got {q_state.shape}.")
        length = q_state.shape[0]
        if q_action.shape != (length, JOINT_COUNT):
            raise ValueError(f"q_action_abs_rad must be [T,{JOINT_COUNT}], got {q_action.shape}.")
        if touch.shape != (length, TACTILE_FINGERS, TACTILE_DIMS):
            raise ValueError(
                f"tactile_f6 must be [T,{TACTILE_FINGERS},{TACTILE_DIMS}], "
                f"got {touch.shape}."
            )
        if not np.isfinite(q_state).all() or not np.isfinite(q_action).all():
            raise ValueError("state/action contains NaN or infinity.")
        if not np.isfinite(touch).all():
            raise ValueError("tactile_f6 contains NaN or infinity.")
        timestamps = _time_vector(self.timestamp_ns, length=length)
        metadata_keys = {str(key).lower() for key in self.metadata}
        if any("emg" in key for key in metadata_keys):
            raise ValueError("EMG is not a T-Rex episode feature; keep it in a separate log.")
        object.__setattr__(self, "timestamp_ns", timestamps)
        object.__setattr__(self, "q_state_rad", q_state.copy())
        object.__setattr__(self, "q_action_abs_rad", q_action.copy())
        object.__setattr__(self, "tactile_f6", touch.copy())
        for name in ("full_rgb", "center_rgb"):
            value = getattr(self, name)
            if value is None:
                continue
            images = np.asarray(value)
            if images.shape[0] != length or images.ndim != 4 or images.shape[-1] != 3:
                raise ValueError(f"{name} must be [T,H,W,3], got {images.shape}.")
            object.__setattr__(self, name, images.copy())

    @property
    def length(self) -> int:
        return int(self.q_state_rad.shape[0])


@dataclass(frozen=True)
class RevoTrainingSample:
    episode_id: str
    anchor_index: int
    timestamp_ns: int
    state_q_rad: np.ndarray
    action_chunk_abs_rad: np.ndarray
    tactile_f6: np.ndarray
    tactile_history_f6: np.ndarray
    instruction: str
    full_rgb: Optional[np.ndarray] = None
    center_rgb: Optional[np.ndarray] = None

    def as_trex_item(self) -> dict[str, object]:
        """Return source tensors expected by the Revo-specific collator."""

        item: dict[str, object] = {
            KEY_STATE: self.state_q_rad.copy(),
            KEY_ACTION: self.action_chunk_abs_rad.copy(),
            KEY_ACTION_ABS: self.action_chunk_abs_rad[0].copy(),
            KEY_TACTILE_F6: self.tactile_history_f6.copy(),
            "task": self.instruction,
            "timestamp_ns": self.timestamp_ns,
            "episode_id": self.episode_id,
        }
        if self.full_rgb is not None:
            item[KEY_IMAGE_FULL] = self.full_rgb.copy()
        if self.center_rgb is not None:
            item[KEY_IMAGE_CENTER] = self.center_rgb.copy()
        return item


class RevoEpisodeAdapter:
    """Forms causal history plus a future absolute-q action chunk."""

    def __init__(self, episode: RevoEpisode) -> None:
        self.episode = episode

    @property
    def first_valid_anchor(self) -> int:
        return TACTILE_HISTORY - 1

    @property
    def last_valid_anchor(self) -> int:
        return self.episode.length - ACTION_CHUNK

    def sample(self, anchor_index: int) -> RevoTrainingSample:
        i = int(anchor_index)
        if i < self.first_valid_anchor:
            raise IndexError("anchor has fewer than 16 real tactile history samples.")
        if i > self.last_valid_anchor:
            raise IndexError("anchor has fewer than 16 future action targets.")
        history_start = i - TACTILE_HISTORY + 1
        action_end = i + ACTION_CHUNK
        ep = self.episode
        return RevoTrainingSample(
            episode_id=ep.episode_id,
            anchor_index=i,
            timestamp_ns=int(ep.timestamp_ns[i]),
            state_q_rad=ep.q_state_rad[i].copy(),
            action_chunk_abs_rad=ep.q_action_abs_rad[i:action_end].copy(),
            tactile_f6=ep.tactile_f6[i].copy(),
            tactile_history_f6=ep.tactile_f6[history_start : i + 1].copy(),
            instruction=ep.instruction,
            full_rgb=None if ep.full_rgb is None else ep.full_rgb[i].copy(),
            center_rgb=None if ep.center_rgb is None else ep.center_rgb[i].copy(),
        )

    def iter_samples(self, *, stride: int = 1) -> Iterator[RevoTrainingSample]:
        if stride <= 0:
            raise ValueError("stride must be positive.")
        if self.last_valid_anchor < self.first_valid_anchor:
            return
        for index in range(self.first_valid_anchor, self.last_valid_anchor + 1, stride):
            yield self.sample(index)


@dataclass(frozen=True)
class RevoNormStats:
    """Train-only stats matching T-Rex broadcasting semantics.

    Action percentiles remain per future step, hence ``[16,21]`` rather than
    flattening the horizon into a single 21-D distribution.
    """

    state_q01: np.ndarray
    state_q99: np.ndarray
    action_q01: np.ndarray
    action_q99: np.ndarray
    tactile_q01: np.ndarray
    tactile_q99: np.ndarray
    tracking_error_mean: np.ndarray
    tracking_error_std: np.ndarray

    @classmethod
    def fit_train_episodes(cls, episodes: list[RevoEpisode]) -> "RevoNormStats":
        if not episodes:
            raise ValueError("at least one training episode is required.")
        states = np.concatenate([ep.q_state_rad for ep in episodes], axis=0)
        action_chunks = [
            sample.action_chunk_abs_rad
            for episode in episodes
            for sample in RevoEpisodeAdapter(episode).iter_samples()
        ]
        if not action_chunks:
            raise ValueError("training episodes are too short to form a [16,21] chunk.")
        actions = np.stack(action_chunks, axis=0)  # [N,16,21]
        touch = np.concatenate([ep.tactile_f6.reshape(ep.length, -1) for ep in episodes], axis=0)
        tracking = np.concatenate(
            [
                ep.q_state_rad[1:] - ep.q_action_abs_rad[:-1]
                for ep in episodes
                if ep.length > 1
            ],
            axis=0,
        )
        return cls(
            state_q01=np.quantile(states, 0.01, axis=0).astype(np.float32),
            state_q99=np.quantile(states, 0.99, axis=0).astype(np.float32),
            action_q01=np.quantile(actions, 0.01, axis=0).astype(np.float32),
            action_q99=np.quantile(actions, 0.99, axis=0).astype(np.float32),
            tactile_q01=np.quantile(touch, 0.01, axis=0).astype(np.float32),
            tactile_q99=np.quantile(touch, 0.99, axis=0).astype(np.float32),
            tracking_error_mean=np.mean(tracking, axis=0).astype(np.float32),
            tracking_error_std=np.std(tracking, axis=0).astype(np.float32),
        )
