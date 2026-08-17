"""Manifest-backed EMG data loading with train-only normalization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

from .preprocessing import (
    EmgPreprocessingProfile,
    preprocess_emg_windows,
    validate_npz_profile,
)


def load_manifest(path: str | Path) -> List[Dict[str, object]]:
    manifest_path = Path(path)
    rows: List[Dict[str, object]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "index" not in row or "subject_id" not in row or "label" not in row:
                raise ValueError(f"Malformed row {line_number} in {manifest_path}")
            rows.append(row)
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return rows


def verify_split_manifests(manifest_paths: Mapping[str, str | Path]) -> None:
    """Reject subject/session leakage and duplicate window indices."""

    subject_owner: Dict[str, str] = {}
    session_owner: Dict[str, str] = {}
    seen_indices: Dict[int, str] = {}
    for split_name, path in manifest_paths.items():
        for row in load_manifest(path):
            subject = str(row["subject_id"])
            session = str(row["session_id"])
            index = int(row["index"])
            if subject in subject_owner and subject_owner[subject] != split_name:
                raise ValueError(f"Subject leakage: {subject} appears in two splits")
            if session in session_owner and session_owner[session] != split_name:
                raise ValueError(f"Session leakage: {session} appears in two splits")
            if index in seen_indices:
                raise ValueError(f"Duplicate window index {index} in {seen_indices[index]} and {split_name}")
            subject_owner[subject] = split_name
            session_owner[session] = split_name
            seen_indices[index] = split_name


def fit_train_normalization(signals: np.ndarray, indices: Sequence[int], epsilon: float = 1e-6) -> Dict[str, np.ndarray]:
    """Fit per-channel mean/std from train indices only."""

    train = np.asarray(signals)[np.asarray(indices, dtype=np.int64)]
    mean = train.mean(axis=(0, 2), dtype=np.float64).astype(np.float32)
    std = train.std(axis=(0, 2), dtype=np.float64).astype(np.float32)
    std = np.maximum(std, np.float32(epsilon))
    return {"mean": mean, "std": std}


try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # Synthetic generation must still import without torch.
    torch = None
    Dataset = object  # type: ignore[assignment,misc]


class EMGWindowDataset(Dataset):
    def __init__(
        self,
        npz_path: str | Path,
        manifest_path: str | Path,
        normalization: Mapping[str, np.ndarray] | None = None,
        channel_rotation: int = 0,
        preprocessing_profile: EmgPreprocessingProfile | None = None,
        allow_unprofiled_fixture: bool = False,
        preprocessed_signals: np.ndarray | None = None,
        allow_window_reset_fallback: bool = False,
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch is required for EMGWindowDataset")
        with np.load(Path(npz_path), allow_pickle=False) as archive:
            self.signals = np.asarray(archive["signal"], dtype=np.float32)
            self.labels = np.asarray(archive["label"], dtype=np.int64)
            preprocessing_state = None
            if preprocessing_profile is not None:
                preprocessing_state = validate_npz_profile(archive, preprocessing_profile)
            elif not allow_unprofiled_fixture:
                raise ValueError(
                    "An explicit EMG preprocessing_profile is required outside synthetic fixtures"
                )
        self.preprocessing_profile = preprocessing_profile
        if preprocessed_signals is not None:
            supplied = np.asarray(preprocessed_signals, dtype=np.float32)
            if supplied.shape != self.signals.shape:
                raise ValueError("preprocessed_signals shape does not match windows.npz")
            self.signals = supplied
        elif preprocessing_profile is not None and preprocessing_state is not None:
            if preprocessing_state.preprocessed:
                pass  # already filtered continuously before session windowing
            elif not allow_window_reset_fallback:
                raise ValueError(
                    "Raw EMG window-reset filtering is fixture/calibration fallback only"
                )
            else:
                self.signals = preprocess_emg_windows(self.signals, preprocessing_profile)
        self.rows = load_manifest(manifest_path)
        self.indices = np.asarray([int(row["index"]) for row in self.rows], dtype=np.int64)
        if np.any(self.indices < 0) or np.any(self.indices >= self.signals.shape[0]):
            raise IndexError("Manifest contains an out-of-range NPZ index")
        self.mean = None if normalization is None else np.asarray(normalization["mean"], dtype=np.float32)
        self.std = None if normalization is None else np.asarray(normalization["std"], dtype=np.float32)
        self.channel_rotation = int(channel_rotation)
        if self.channel_rotation < 0:
            raise ValueError("channel_rotation cannot be negative")
        if self.mean is not None:
            expected = (self.signals.shape[1],)
            if self.mean.shape != expected or self.std is None or self.std.shape != expected:
                raise ValueError("EMG normalization shape does not match signal channels")
            if np.any(self.std <= 0):
                raise ValueError("EMG normalization std must be positive")

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        signal = self.signals[index]
        if self.channel_rotation:
            shift = int(np.random.randint(-self.channel_rotation, self.channel_rotation + 1))
            signal = np.roll(signal, shift=shift, axis=0)
        if self.mean is not None and self.std is not None:
            signal = (signal - self.mean[:, None]) / self.std[:, None]
        return torch.from_numpy(np.asarray(signal, dtype=np.float32)), torch.tensor(
            int(self.labels[index]), dtype=torch.long
        )
