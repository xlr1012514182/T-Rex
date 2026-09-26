"""Single-hand Revo3 Force6D windows with split-before-statistics semantics."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from revo3_v1.data import (
    CorpusSplit,
    NativeForceStream,
    RevoCorpusSplitManifest,
    RevoEpisode,
    native_force_stream,
)

from .stats import TacF6Stats


def _episodes(root: Path, episode_ids: Sequence[str]) -> list[RevoEpisode]:
    result = [RevoEpisode.load(root / episode_id) for episode_id in episode_ids]
    if not result:
        raise ValueError("selected Revo split contains no episodes")
    if any(episode.meta.tactile_num_fingers != 5 for episode in result):
        raise ValueError("Revo Force6D tokenizer requires exactly five fingers")
    if any(
        episode.meta.tactile_profile
        not in {"profile_a_force6d_diff", "ablation_force6d_only"}
        for episode in result
    ):
        raise ValueError("Revo Force6D tokenizer cannot consume DIFF-only/Profile C data")
    return result


def fit_revo_f6_stats(episodes: Sequence[RevoEpisode]) -> TacF6Stats:
    if not episodes:
        raise ValueError("cannot fit Revo tactile stats without train episodes")
    values = np.concatenate(
        [native_force_stream(episode).values.reshape(-1, 30) for episode in episodes],
        axis=0,
    )
    minimum = np.quantile(values, 0.01, axis=0).astype(np.float32)
    maximum = np.quantile(values, 0.99, axis=0).astype(np.float32)
    return TacF6Stats(
        tacf6_min=minimum,
        tacf6_max=maximum,
        tacf6_mask=np.abs(maximum - minimum) > 1e-6,
    )


class RevoF6WindowDataset(Dataset):
    """No-padding [16,5,6] windows from an explicit episode split."""

    def __init__(
        self,
        episodes: Sequence[RevoEpisode],
        *,
        stats: TacF6Stats,
        window: int = 16,
        stride: int = 4,
    ) -> None:
        self.episodes = tuple(episodes)
        self.stats = stats
        self.window = int(window)
        self.stride = int(stride)
        if self.window != 16 or self.stride <= 0:
            raise ValueError("Revo tokenizer requires window=16 and a positive stride")
        self._streams = tuple(native_force_stream(episode) for episode in self.episodes)
        self._index: list[tuple[int, int]] = []
        for episode_index, stream in enumerate(self._streams):
            for start in range(0, len(stream.sequence) - self.window + 1, self.stride):
                sequence = stream.sequence[start : start + self.window]
                if np.all(np.diff(sequence) == 1):
                    self._index.append((episode_index, start))
        if not self._index:
            raise ValueError("selected Revo episodes contain no complete 16-frame windows")

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> dict[str, object]:
        episode_index, start = self._index[index]
        episode = self.episodes[episode_index]
        stream = self._streams[episode_index]
        raw = stream.values[start : start + self.window]
        if raw.shape != (16, 5, 6):
            raise RuntimeError("internal error: incomplete Revo tactile window")
        normalized = self.stats.normalize(raw).astype(np.float32, copy=False)
        return {
            "f6": torch.from_numpy(normalized),
            "magnitude": torch.tensor(float(np.linalg.norm(raw)), dtype=torch.float32),
            "episode_id": episode.meta.episode_id,
            "ep_idx": episode_index,
            "frame": start,
            "hand": 0,
        }

    @staticmethod
    def collate_fn(batch: list[dict[str, object]]) -> dict[str, object]:
        return {
            "f6": torch.stack([item["f6"] for item in batch]),
            "magnitude": torch.stack([item["magnitude"] for item in batch]),
            "episode_id": [str(item["episode_id"]) for item in batch],
            "ep_idx": torch.tensor([int(item["ep_idx"]) for item in batch]),
            "frame": torch.tensor([int(item["frame"]) for item in batch]),
            "hand": torch.zeros(len(batch), dtype=torch.long),
        }


def build_revo_train_val_datasets(
    data_root: str | Path,
    split_manifest: str | Path,
    *,
    window: int = 16,
    stride: int = 4,
) -> tuple[RevoF6WindowDataset, RevoF6WindowDataset, TacF6Stats]:
    """Use train-authorized episodes for stats/tokenizer and development for val.

    Locked-test episodes are required by the manifest but are never opened by
    this function.  Therefore neither model selection nor normalization can
    accidentally consume them.
    """

    root = Path(data_root)
    manifest = RevoCorpusSplitManifest.load(split_manifest)
    manifest.assert_matches_episode_root(root)
    # The tokenizer and its normalization are frozen before SFT.  SFT data is
    # therefore neither a VQ training source nor a statistics source.
    train_ids = manifest.episode_ids((CorpusSplit.MIDTRAIN_TRAIN,))
    development_ids = manifest.episode_ids((CorpusSplit.DEVELOPMENT,))
    train_episodes = _episodes(root, train_ids)
    development_episodes = _episodes(root, development_ids)
    stats = fit_revo_f6_stats(train_episodes)
    return (
        RevoF6WindowDataset(train_episodes, stats=stats, window=window, stride=stride),
        RevoF6WindowDataset(development_episodes, stats=stats, window=window, stride=stride),
        stats,
    )
