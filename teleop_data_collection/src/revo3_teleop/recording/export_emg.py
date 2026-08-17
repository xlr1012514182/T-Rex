"""Explicitly annotated EMG projection for the binary intent classifier.

Labels never come from glove motion or robot commands.  Every exported window
must be fully contained in a reviewed OPEN/CLOSE protocol annotation and in a
contiguous, lead-off-free 250 Hz signal segment.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np

from revo3_teleop.recording.recorder import load_native_payload


SplitName = Literal["train", "val", "test"]
LABEL_NAMES = {0: "OPEN", 1: "CLOSE"}
REVIEWED_SOURCES = frozenset({"protocol_cue_human_reviewed", "manual_human_reviewed"})


@dataclass(frozen=True)
class EMGLabelInterval:
    start_timestamp_ns: int
    end_timestamp_ns: int
    label: int
    source: str
    human_reviewed: bool

    def __post_init__(self) -> None:
        if self.start_timestamp_ns < 0 or self.end_timestamp_ns <= self.start_timestamp_ns:
            raise ValueError("EMG label interval must have positive duration")
        if self.label not in LABEL_NAMES:
            raise ValueError("EMG label must be 0=OPEN or 1=CLOSE")
        if self.source not in REVIEWED_SOURCES or not self.human_reviewed:
            raise ValueError("EMG labels must be explicit and human reviewed")


@dataclass(frozen=True)
class EMGSessionSpec:
    episode_root: Path
    subject_id: str
    session_id: str
    split: SplitName
    intervals: tuple[EMGLabelInterval, ...]

    def __post_init__(self) -> None:
        if self.split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        if not self.subject_id.strip() or not self.session_id.strip():
            raise ValueError("subject_id and session_id must be non-empty")
        ordered = sorted(self.intervals, key=lambda item: item.start_timestamp_ns)
        if not ordered:
            raise ValueError("an EMG session needs reviewed label intervals")
        if tuple(ordered) != self.intervals:
            raise ValueError("EMG label intervals must be sorted")
        for previous, current in zip(ordered, ordered[1:]):
            if current.start_timestamp_ns < previous.end_timestamp_ns:
                raise ValueError("EMG label intervals cannot overlap")


@dataclass(frozen=True)
class EMGExportConfig:
    stream: str = "emg"
    window_samples: int = 250
    stride_samples: int = 125
    channels: int = 8
    sample_rate_hz: int = 250
    maximum_gap_samples: float = 1.5

    def __post_init__(self) -> None:
        if not self.stream.strip():
            raise ValueError("EMG stream must be non-empty")
        if self.window_samples < 20 or not 1 <= self.stride_samples <= self.window_samples:
            raise ValueError("invalid EMG window/stride")
        if self.channels != 8 or self.sample_rate_hz != 250:
            raise ValueError("BrainCo EDU V1 contract is fixed at 8 channels and 250 Hz")
        if self.maximum_gap_samples < 1.0:
            raise ValueError("maximum_gap_samples must be at least one")


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"row {line_number} in {path} is not an object")
            rows.append(value)
    return rows


def _load_contiguous_signal(
    session: EMGSessionSpec,
    config: EMGExportConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = Path(session.episode_root).resolve()
    manifest = _read_json(root / "manifest.json")
    if manifest.get("schema_version") != "revo3-teleop-master-v1":
        raise ValueError("unsupported master episode schema")
    if manifest.get("lifecycle") != "committed":
        raise ValueError("EMG export accepts only committed master episodes")
    rows = _read_jsonl(root / "streams" / config.stream / "index.jsonl")
    if not rows:
        raise ValueError("EMG native stream is empty")
    signal_parts: list[np.ndarray] = []
    timestamp_parts: list[np.ndarray] = []
    segment_parts: list[np.ndarray] = []
    previous_sequence: int | None = None
    current_segment = 0
    have_valid_packet = False
    break_before_next_valid_packet = False
    for row in rows:
        header = row.get("header")
        if not isinstance(header, Mapping):
            raise ValueError("EMG native index header is malformed")
        sequence = int(header.get("sequence", -1))
        if sequence < 0:
            raise ValueError("EMG native index sequence is invalid")
        if previous_sequence is not None and sequence != previous_sequence + 1:
            break_before_next_valid_packet = True
        dropped = int(header.get("dropped_since_previous", 0))
        if dropped < 0:
            raise ValueError("EMG dropped_since_previous is invalid")
        if dropped > 0:
            break_before_next_valid_packet = True
        previous_sequence = sequence
        if not bool(header.get("valid", False)):
            break_before_next_valid_packet = True
            continue
        payload = load_native_payload(root, config.stream, row)
        required = {
            "signal",
            "sample_timestamp_ns",
            "lead_off_bits",
            "sample_rate_hz",
            "samples_per_channel",
        }
        if not required.issubset(payload):
            raise ValueError(f"EMG payload lacks {sorted(required - set(payload))}")
        signal = np.asarray(payload["signal"], dtype=np.float32)
        timestamps = np.asarray(payload["sample_timestamp_ns"], dtype=np.int64)
        if signal.shape != (config.channels, 20) or timestamps.shape != (20,):
            raise ValueError("EMG packet must be signal[8,20] with 20 sample timestamps")
        if int(np.asarray(payload["samples_per_channel"]).reshape(-1)[0]) != 20:
            raise ValueError("EMG samples_per_channel changed")
        observed_rate = float(np.asarray(payload["sample_rate_hz"]).reshape(-1)[0])
        if not np.isclose(observed_rate, config.sample_rate_hz, atol=0.01):
            raise ValueError("EMG sample rate changed")
        if "sequence_gap_packets" in payload:
            payload_gap = int(np.asarray(payload["sequence_gap_packets"]).reshape(-1)[0])
            if payload_gap != dropped:
                raise ValueError("EMG header/payload sequence-gap evidence disagrees")
            if payload_gap > 0:
                break_before_next_valid_packet = True
        if int(np.asarray(payload["lead_off_bits"]).reshape(-1)[0]) != 0:
            break_before_next_valid_packet = True
            continue
        if not np.isfinite(signal).all() or np.any(np.diff(timestamps) <= 0):
            raise ValueError("EMG packet contains invalid signal/timestamps")
        if have_valid_packet and break_before_next_valid_packet:
            current_segment += 1
        signal_parts.append(signal)
        timestamp_parts.append(timestamps)
        segment_parts.append(np.full(timestamps.shape, current_segment, dtype=np.int64))
        have_valid_packet = True
        break_before_next_valid_packet = False
    if not signal_parts:
        raise ValueError("no valid lead-off-free EMG packet remains")
    signal = np.concatenate(signal_parts, axis=1)
    timestamps = np.concatenate(timestamp_parts)
    segments = np.concatenate(segment_parts)
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("EMG sample timestamps overlap or regress across packets")
    return signal, timestamps, segments


def _windows_for_session(
    session: EMGSessionSpec,
    config: EMGExportConfig,
) -> list[tuple[np.ndarray, int, int, int, str]]:
    signal, timestamps, segments = _load_contiguous_signal(session, config)
    expected_period_ns = 1_000_000_000 / float(config.sample_rate_hz)
    maximum_gap_ns = int(np.ceil(expected_period_ns * config.maximum_gap_samples))
    windows: list[tuple[np.ndarray, int, int, int, str]] = []
    for interval in session.intervals:
        indices = np.flatnonzero(
            (timestamps >= interval.start_timestamp_ns)
            & (timestamps < interval.end_timestamp_ns)
        )
        if indices.size < config.window_samples:
            continue
        # Split the label interval wherever acquisition has a packet/sample gap.
        breaks = np.flatnonzero(
            (np.diff(timestamps[indices]) > maximum_gap_ns)
            | (np.diff(segments[indices]) != 0)
        ) + 1
        for segment in np.split(indices, breaks):
            if segment.size < config.window_samples:
                continue
            for offset in range(
                0,
                int(segment.size) - config.window_samples + 1,
                config.stride_samples,
            ):
                selection = segment[offset : offset + config.window_samples]
                start_ns = int(timestamps[selection[0]])
                end_ns = int(timestamps[selection[-1]] + round(expected_period_ns))
                if start_ns < interval.start_timestamp_ns or end_ns > interval.end_timestamp_ns:
                    continue
                windows.append(
                    (
                        signal[:, selection].astype(np.float32, copy=True),
                        interval.label,
                        start_ns,
                        end_ns,
                        interval.source,
                    )
                )
    return windows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def export_emg_binary_dataset(
    sessions: Sequence[EMGSessionSpec],
    output_root: str | Path,
    config: EMGExportConfig = EMGExportConfig(),
) -> Path:
    """Export the classifier-only view with subject-exclusive manifests."""

    if not sessions:
        raise ValueError("at least one EMG session is required")
    subject_owner: dict[str, str] = {}
    session_ids: set[str] = set()
    for session in sessions:
        owner = subject_owner.setdefault(session.subject_id, session.split)
        if owner != session.split:
            raise ValueError(f"subject leakage across splits: {session.subject_id}")
        if session.session_id in session_ids:
            raise ValueError(f"duplicate session_id: {session.session_id}")
        session_ids.add(session.session_id)

    destination = Path(output_root).resolve()
    temporary = destination.with_name(f".{destination.name}.inprogress")
    if destination.exists() or temporary.exists():
        raise FileExistsError(f"EMG derived dataset already exists: {destination}")
    temporary.mkdir(parents=True)
    (temporary / "manifests").mkdir()
    signals: list[np.ndarray] = []
    labels: list[int] = []
    subjects: list[str] = []
    session_values: list[str] = []
    starts: list[int] = []
    rows_by_split: dict[str, list[dict[str, object]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    try:
        for session in sessions:
            for signal, label, start_ns, end_ns, label_source in _windows_for_session(
                session, config
            ):
                index = len(signals)
                signals.append(signal)
                labels.append(label)
                subjects.append(session.subject_id)
                session_values.append(session.session_id)
                starts.append(start_ns)
                rows_by_split[session.split].append(
                    {
                        "index": index,
                        "data_file": "../windows.npz",
                        "subject_id": session.subject_id,
                        "session_id": session.session_id,
                        "window_start_ns": start_ns,
                        "window_end_ns": end_ns,
                        "label": label,
                        "label_name": LABEL_NAMES[label],
                        "label_source": label_source,
                        "source_episode_id": Path(session.episode_root).name,
                    }
                )
        empty = [name for name, rows in rows_by_split.items() if not rows]
        if empty:
            raise ValueError(f"every split needs at least one reviewed window; empty={empty}")
        np.savez_compressed(
            temporary / "windows.npz",
            signal=np.stack(signals).astype(np.float32),
            label=np.asarray(labels, dtype=np.int64),
            subject_id=np.asarray(subjects),
            session_id=np.asarray(session_values),
            window_start_ns=np.asarray(starts, dtype=np.int64),
            sample_rate_hz=np.asarray(config.sample_rate_hz, dtype=np.int64),
        )
        for split, rows in rows_by_split.items():
            _write_jsonl(temporary / "manifests" / f"{split}.jsonl", rows)
        metadata = {
            "schema_version": "revo3-emg-reviewed-v1",
            "source_view": "committed_native_emg_only",
            "contains_rgb": False,
            "contains_robot_action": False,
            "label_policy": "explicit reviewed OPEN/CLOSE intervals only; no glove heuristic",
            "labels": {str(key): value for key, value in LABEL_NAMES.items()},
            "sample_rate_hz": config.sample_rate_hz,
            "window_samples": config.window_samples,
            "stride_samples": config.stride_samples,
            "split_unit": "subject_id",
            "counts": {name: len(rows) for name, rows in rows_by_split.items()},
            "source_episode_ids": sorted(
                {Path(session.episode_root).name for session in sessions}
            ),
        }
        with (temporary / "dataset_meta.json").open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(metadata, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
        os.replace(temporary, destination)
    except Exception:
        # Preserve the in-progress directory for audit rather than deleting data.
        raise
    return destination
