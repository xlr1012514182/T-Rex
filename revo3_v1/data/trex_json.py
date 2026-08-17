"""Convert validated 30 Hz Revo3 episodes to T-Rex's JSON SFT contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from revo3_v1.revo.contracts import JOINT_COUNT, JOINT_ORDER, JOINT_ORDER_HASH

from .episode import RevoEpisode


@dataclass(frozen=True)
class ConversionConfig:
    action_chunk: int = 16
    tactile_num_fingers: int = 5
    dataset_name: str = "revo3_single_hand"
    terminal_padding: str = "repeat_last_controller_target"

    def validate(self) -> None:
        if self.action_chunk != 16:
            raise ValueError("T-Rex Revo3 V1 fixes action_chunk=16")
        if self.tactile_num_fingers != 5:
            raise ValueError("Revo3 single-hand V1 requires five tactile streams")


def _robust_block(values: np.ndarray) -> Dict[str, object]:
    q01 = np.quantile(values, 0.01, axis=0).astype(np.float32)
    q99 = np.quantile(values, 0.99, axis=0).astype(np.float32)
    mask = np.abs(q99 - q01) > 1e-6
    return {"q01": q01.tolist(), "q99": q99.tolist(), "mask": mask.tolist()}


def _chunk(values: np.ndarray, start: int, length: int) -> np.ndarray:
    selected = values[start : start + length]
    if selected.shape[0] < length:
        selected = np.concatenate(
            [selected, np.repeat(selected[-1:], length - selected.shape[0], axis=0)], axis=0
        )
    return selected.astype(np.float32, copy=False)


def convert_revo_episodes_to_trex_json(
    episodes_root: str | Path,
    output_json: str | Path,
    config: ConversionConfig = ConversionConfig(),
) -> Dict[str, object]:
    """Write training records and q01/q99 statistics.

    The VLA output contains only 21 Revo joint targets.  EMG is intentionally
    absent: it belongs to the upstream instruction planner/runtime executive,
    not to robot-only teleoperation mid/post-training.
    """

    config.validate()
    roots = sorted(Path(episodes_root).glob("*/meta.json"))
    if not roots:
        raise FileNotFoundError(f"no Revo episodes found under {episodes_root}")
    episodes = [RevoEpisode.load(path.parent) for path in roots]
    if any(ep.meta.tactile_num_fingers != config.tactile_num_fingers for ep in episodes):
        raise ValueError("mixed tactile finger counts are not supported")
    if any(ep.meta.fps != 30 for ep in episodes):
        raise ValueError("V1 training records must be causally resampled to 30 Hz first")

    output = Path(output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    action_values, state_values, tactile_values = [], [], []
    for episode in episodes:
        for frame in range(episode.num_frames):
            action_chunk = _chunk(episode.action_target_rad, frame, config.action_chunk)
            image_absolute = episode.root / episode.meta.image_paths[frame]
            image_relative = os.path.relpath(image_absolute, output.parent).replace("\\", "/")
            rows.append(
                {
                    "schema_version": "revo3-trex-json-v1",
                    "episode_id": episode.meta.episode_id,
                    "frame_index": frame,
                    "timestamp_ns": int(episode.timestamp_ns[frame]),
                    "input_prompt": episode.meta.instruction,
                    "input_image_slow": [image_relative],
                    "input_image_fast": [],
                    "state_fast": episode.state_rad[frame].tolist(),
                    "action": action_chunk.reshape(-1).tolist(),
                    "tactile_f6": episode.tactile_features[frame].reshape(-1).tolist(),
                    "tactile_image_deform": [],
                    "action_label_source": episode.meta.action_label_source,
                    "contains_emg": False,
                }
            )
            action_values.append(action_chunk)
            state_values.append(episode.state_rad[frame])
            tactile_values.append(episode.tactile_features[frame].reshape(-1))

    with output.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False)
    statistics = {
        config.dataset_name: {
            "action": _robust_block(np.stack(action_values)),
            "state": _robust_block(np.stack(state_values)),
            "tactile_f6": _robust_block(np.stack(tactile_values)),
            "tracking_error": {},
        }
    }
    stats_path = Path(str(output).replace(".json", "_statistics.json"))
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(statistics, handle, ensure_ascii=False, indent=2)
    manifest = {
        "schema_version": "revo3-trex-conversion-v1",
        "config": asdict(config),
        "source_root": str(Path(episodes_root).resolve()),
        "records": len(rows),
        "episode_ids": [episode.meta.episode_id for episode in episodes],
        "joint_order": list(JOINT_ORDER),
        "joint_order_hash": JOINT_ORDER_HASH,
        "action_shape": [config.action_chunk, JOINT_COUNT],
        "state_shape": [JOINT_COUNT],
        "tactile_shape": [config.tactile_num_fingers, 6],
        "action_label_source": "controller_target",
        "contains_emg": False,
        "stats_path": str(stats_path.resolve()),
        "limitations": [
            "JSON adapter is intended for smoke and bounded post/mid-train runs; use LeRobot for the 20-hour corpus.",
            "Synthetic episodes validate plumbing only and provide no robot-performance evidence.",
        ],
    }
    manifest_path = output.with_name(output.stem + "_manifest.json")
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest
