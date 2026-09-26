"""Synthetic OPEN/CLOSE surface-EMG windows and leakage-safe manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .primitives import MAINLINE_CLASS_LABELS
from .preprocessing import BRAINCO_EDU_8CH_250HZ


LABEL_NAMES: Mapping[int, str] = {0: "OPEN", 1: "CLOSE"}
BINARY_LABELS: Tuple[str, ...] = ("OPEN", "CLOSE")


@dataclass(frozen=True)
class SyntheticEMGConfig:
    """Configuration for a compact, deterministic synthetic EMG corpus.

    The defaults are intended for a runnable demo, not as a substitute for
    participant data.  ``sample_rate_hz=2000`` and ``window_seconds=8`` match
    the released GNI discrete-gesture setup but result in much larger files.
    """

    n_subjects: int = 8
    sessions_per_subject: int = 3
    windows_per_label_per_session: int = 8
    n_channels: int = 16
    sample_rate_hz: int = 1_000
    window_seconds: float = 1.0
    window_stride_seconds: float = 1.25
    seed: int = 20260817
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    label_names: Tuple[str, ...] = BINARY_LABELS

    @property
    def n_samples(self) -> int:
        return int(round(self.sample_rate_hz * self.window_seconds))

    def validate(self) -> None:
        if self.n_subjects < 3:
            raise ValueError("n_subjects must be >= 3 for non-empty train/val/test splits")
        if self.sessions_per_subject < 1 or self.windows_per_label_per_session < 1:
            raise ValueError("sessions and windows per label must be positive")
        if self.n_channels < 2:
            raise ValueError("n_channels must be >= 2")
        if self.sample_rate_hz < 50 or self.n_samples < 32:
            raise ValueError("sample rate/window are too small for EMG synthesis")
        if self.window_stride_seconds < self.window_seconds:
            raise ValueError("window_stride_seconds must avoid overlapping labelled windows")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be in (0, 1)")
        if not 0.0 < self.val_fraction < 1.0 - self.train_fraction:
            raise ValueError("val_fraction must leave a non-empty test fraction")
        if tuple(self.label_names) not in {BINARY_LABELS, tuple(MAINLINE_CLASS_LABELS)}:
            raise ValueError("synthetic fixture labels must be binary or the frozen five-class set")
        if tuple(self.label_names) == tuple(MAINLINE_CLASS_LABELS) and (
            self.n_channels != BRAINCO_EDU_8CH_250HZ.channel_count
            or self.sample_rate_hz != BRAINCO_EDU_8CH_250HZ.sample_rate_hz
            or self.n_samples != BRAINCO_EDU_8CH_250HZ.window_samples
        ):
            raise ValueError(
                "five-class synthetic fixture must match the frozen BrainCo 8ch@250Hz/2s profile"
            )


def _subject_split(config: SyntheticEMGConfig, rng: np.random.RandomState) -> Dict[str, List[str]]:
    subjects = np.asarray([f"subject_{i:03d}" for i in range(config.n_subjects)])
    rng.shuffle(subjects)
    n_train = max(1, int(round(config.n_subjects * config.train_fraction)))
    n_val = max(1, int(round(config.n_subjects * config.val_fraction)))
    if n_train + n_val >= config.n_subjects:
        n_train = config.n_subjects - 2
        n_val = 1
    return {
        "train": sorted(subjects[:n_train].tolist()),
        "val": sorted(subjects[n_train : n_train + n_val].tolist()),
        "test": sorted(subjects[n_train + n_val :].tolist()),
    }


def _burst_envelope(n_samples: int, rng: np.random.RandomState) -> np.ndarray:
    x = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)
    onset = rng.uniform(0.06, 0.18)
    offset = rng.uniform(0.82, 0.96)
    sharpness = rng.uniform(25.0, 45.0)
    rise = 1.0 / (1.0 + np.exp(-sharpness * (x - onset)))
    fall = 1.0 / (1.0 + np.exp(sharpness * (x - offset)))
    return (rise * fall).astype(np.float32)


def _make_window(
    label: int,
    config: SyntheticEMGConfig,
    rng: np.random.RandomState,
    subject_gain: np.ndarray,
    session_gain: np.ndarray,
    channel_permutation: np.ndarray,
) -> np.ndarray:
    """Synthesize an EMG-like window with subject/session domain shifts.

    OPEN and CLOSE excite complementary channel groups.  Broadband motor-unit
    components, line interference, baseline drift, cross-channel mixing, and
    sporadic spikes prevent the task from reducing to a constant DC feature.
    """

    channels, n = config.n_channels, config.n_samples
    t = np.arange(n, dtype=np.float32) / float(config.sample_rate_hz)
    envelope = _burst_envelope(n, rng)
    half = max(1, channels // 2)
    preferred = np.zeros(channels, dtype=np.float32)
    if tuple(config.label_names) == BINARY_LABELS:
        if label == 1:  # CLOSE / flexor-like group
            preferred[:half] = 1.0
            preferred[half:] = 0.30
        else:  # OPEN / extensor-like group
            preferred[:half] = 0.30
            preferred[half:] = 1.0
    else:
        # Deliberately simple, separable CI patterns.  They only verify the
        # five-class plumbing and are not a physiological EMG simulator.
        preferred[:] = 0.20
        if label == 0:  # POWER_GRASP
            preferred[:half] = 1.0
        elif label == 1:  # PRECISION_GRASP
            preferred[::2] = 0.90
        elif label == 2:  # LATERAL_GRASP
            preferred[: max(1, channels // 4)] = 0.85
            preferred[-max(1, channels // 4) :] = 1.0
        elif label == 3:  # RELEASE
            preferred[half:] = 1.0
        elif label == 4:  # REST
            preferred[:] = 0.03
    preferred = preferred[channel_permutation]

    signal = rng.normal(0.0, 0.035, size=(channels, n)).astype(np.float32)
    for ch in range(channels):
        carrier = np.zeros(n, dtype=np.float32)
        for _ in range(4):
            frequency = rng.uniform(55.0, min(220.0, config.sample_rate_hz * 0.42))
            phase = rng.uniform(0.0, 2.0 * np.pi)
            carrier += np.sin(2.0 * np.pi * frequency * t + phase).astype(np.float32)
        carrier /= 4.0
        amplitude = rng.uniform(0.55, 0.95) * preferred[ch]
        signal[ch] += amplitude * envelope * carrier

    # A small common-mode component and line interference mimic acquisition.
    common = rng.normal(0.0, 0.02, size=n).astype(np.float32)
    line_hz = 50.0
    line = np.sin(2.0 * np.pi * line_hz * t + rng.uniform(0.0, 2.0 * np.pi)).astype(np.float32)
    drift = np.sin(2.0 * np.pi * rng.uniform(0.2, 1.0) * t).astype(np.float32)
    signal += 0.25 * common[None, :] + 0.01 * line[None, :] + 0.008 * drift[None, :]

    # Weak nearest-neighbour cross-talk.
    signal = 0.88 * signal + 0.06 * np.roll(signal, 1, axis=0) + 0.06 * np.roll(signal, -1, axis=0)
    signal *= subject_gain[:, None] * session_gain[:, None]

    if rng.rand() < 0.20:
        ch = int(rng.randint(0, channels))
        pos = int(rng.randint(0, n))
        signal[ch, pos : min(n, pos + 2)] += rng.uniform(-0.8, 0.8)
    return signal.astype(np.float32)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def generate_synthetic_dataset(output_dir: str | Path, config: SyntheticEMGConfig) -> Dict[str, object]:
    """Generate one NPZ plus subject-exclusive JSONL split manifests.

    Returns the same metadata written to ``dataset_meta.json``.  The function
    imports only NumPy and therefore remains usable without PyTorch.
    """

    config.validate()
    output = Path(output_dir)
    manifest_dir = output / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(config.seed)
    split_subjects = _subject_split(config, rng)
    subject_to_split = {
        subject: split_name
        for split_name, subjects in split_subjects.items()
        for subject in subjects
    }

    signals: List[np.ndarray] = []
    labels: List[int] = []
    subject_ids: List[str] = []
    session_ids: List[str] = []
    window_start_ns: List[int] = []
    rows_by_split: Dict[str, List[Dict[str, object]]] = {"train": [], "val": [], "test": []}
    stride_ns = int(round(config.window_stride_seconds * 1e9))
    duration_ns = int(round(config.window_seconds * 1e9))

    for subject_index in range(config.n_subjects):
        subject_id = f"subject_{subject_index:03d}"
        subject_rng = np.random.RandomState(config.seed + 10_000 + subject_index)
        subject_gain = subject_rng.lognormal(mean=0.0, sigma=0.18, size=config.n_channels).astype(np.float32)
        # Preserve electrode topology across subjects.  Domain shift is
        # represented by gain/noise/cross-talk; circular electrode-placement
        # changes are introduced by the GNI-style train-time augmentation.
        channel_permutation = np.arange(config.n_channels)
        for session_index in range(config.sessions_per_subject):
            session_id = f"{subject_id}_session_{session_index:02d}"
            session_rng = np.random.RandomState(config.seed + subject_index * 1_000 + session_index)
            session_gain = session_rng.lognormal(mean=0.0, sigma=0.08, size=config.n_channels).astype(np.float32)
            session_base_ns = 1_700_000_000_000_000_000 + subject_index * 10**14 + session_index * 10**12
            ordered_labels = np.tile(
                np.arange(len(config.label_names), dtype=np.int64),
                config.windows_per_label_per_session,
            )
            session_rng.shuffle(ordered_labels)
            for local_index, label_value in enumerate(ordered_labels.tolist()):
                global_index = len(signals)
                start_ns = session_base_ns + local_index * stride_ns
                signal = _make_window(
                    int(label_value), config, session_rng, subject_gain, session_gain, channel_permutation
                )
                signals.append(signal)
                labels.append(int(label_value))
                subject_ids.append(subject_id)
                session_ids.append(session_id)
                window_start_ns.append(start_ns)
                split_name = subject_to_split[subject_id]
                rows_by_split[split_name].append(
                    {
                        "index": global_index,
                        "data_file": "../windows.npz",
                        "subject_id": subject_id,
                        "session_id": session_id,
                        "window_start_ns": start_ns,
                        "window_end_ns": start_ns + duration_ns,
                        "label": int(label_value),
                        "label_name": config.label_names[int(label_value)],
                    }
                )

    archive = {
        "signal": np.stack(signals).astype(np.float32),
        "label": np.asarray(labels, dtype=np.int64),
        "subject_id": np.asarray(subject_ids),
        "session_id": np.asarray(session_ids),
        "window_start_ns": np.asarray(window_start_ns, dtype=np.int64),
        "sample_rate_hz": np.asarray(config.sample_rate_hz, dtype=np.int64),
    }
    is_mainline_fixture = tuple(config.label_names) == tuple(MAINLINE_CLASS_LABELS)
    if is_mainline_fixture:
        archive.update(
            {
                "channel_order": np.asarray(BRAINCO_EDU_8CH_250HZ.channel_order),
                "preprocessing_profile_id": np.asarray(BRAINCO_EDU_8CH_250HZ.profile_id),
                "preprocessed": np.asarray(False),
            }
        )
    np.savez_compressed(output / "windows.npz", **archive)
    for split_name, rows in rows_by_split.items():
        _write_jsonl(manifest_dir / f"{split_name}.jsonl", rows)

    metadata: Dict[str, object] = {
        "schema_version": "revo3-emg-synthetic-v1",
        "generator": "synthetic-demo-not-human-data",
        "config": asdict(config),
        "labels": {str(key): value for key, value in enumerate(config.label_names)},
        "verification_scope": "synthetic CI fixture only; no physiological or clinical claim",
        "split_unit": "subject_id",
        "split_subjects": split_subjects,
        "counts": {name: len(rows) for name, rows in rows_by_split.items()},
        "total_windows": len(signals),
        "signal_shape": [len(signals), config.n_channels, config.n_samples],
        "preprocessing_profile_id": (
            BRAINCO_EDU_8CH_250HZ.profile_id if is_mainline_fixture else None
        ),
        "preprocessed": False,
        "filter_state_provenance": "",
        "future_leakage_policy": (
            "All windows from a subject are assigned to exactly one split; "
            "manifests are produced before any model normalization is computed."
        ),
    }
    with (output / "dataset_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return metadata
