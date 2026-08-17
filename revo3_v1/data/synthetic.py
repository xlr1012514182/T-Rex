"""Deterministic synthetic Revo3 episodes for plumbing tests only."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image, ImageDraw

from revo3_v1.planner.schema import DEFAULT_INSTRUCTIONS, SupportedTask
from revo3_v1.revo.contracts import JOINT_COUNT, JOINT_ORDER_HASH


@dataclass(frozen=True)
class SyntheticRevoConfig:
    episodes_per_task: int = 2
    frames_per_episode: int = 48
    fps: int = 30
    tactile_num_fingers: int = 5
    image_width: int = 160
    image_height: int = 120
    seed: int = 20260817

    def validate(self) -> None:
        if self.episodes_per_task < 1 or self.frames_per_episode < 17:
            raise ValueError("need at least one episode and 17 frames")
        if self.fps <= 0 or self.tactile_num_fingers != 5:
            raise ValueError("Revo3 V1 synthetic contract requires 30-ish Hz and five fingers")


def _close_synergy(progress: float) -> np.ndarray:
    q = np.zeros(JOINT_COUNT, dtype=np.float32)
    # MPR/spread joints move less than flexion joints.  Values are synthetic
    # radians and must never be reused as real Revo joint limits.
    for index in range(16):
        q[index] = progress * (0.18 if index % 4 == 0 else 0.72)
    q[16:19] = progress * np.asarray([0.52, 0.66, 0.58], dtype=np.float32)
    q[19:21] = progress * np.asarray([0.38, 0.28], dtype=np.float32)
    return q


def _draw_frame(path: Path, task: SupportedTask, frame: int, total: int) -> None:
    image = Image.new("RGB", (160, 120), (24, 27, 33))
    draw = ImageDraw.Draw(image)
    growth = 0.22 + 0.30 * frame / max(1, total - 1)
    half_w, half_h = int(160 * growth / 2), int(120 * growth / 2)
    colors = {
        SupportedTask.BOTTLE: (65, 150, 230),
        SupportedTask.PHONE: (150, 150, 160),
        SupportedTask.PLASTIC_BAG: (235, 210, 110),
        SupportedTask.REFRIGERATOR_DOOR: (210, 210, 220),
    }
    draw.rectangle((80 - half_w, 60 - half_h, 80 + half_w, 60 + half_h), fill=colors[task])
    image.save(path, format="PNG")


def generate_synthetic_revo_episodes(
    output_dir: str | Path, config: SyntheticRevoConfig = SyntheticRevoConfig()
) -> Dict[str, object]:
    """Create fake RGB/state/action/tactile episodes, never EMG."""

    config.validate()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(config.seed)
    episode_ids: List[str] = []
    tasks = tuple(SupportedTask)
    period_ns = int(round(1e9 / config.fps))
    for task in tasks:
        for repetition in range(config.episodes_per_task):
            episode_id = f"{task.value}_{repetition:03d}"
            episode_ids.append(episode_id)
            episode_root = root / episode_id
            frames_root = episode_root / "rgb"
            frames_root.mkdir(parents=True, exist_ok=True)
            n = config.frames_per_episode
            timestamp = np.arange(n, dtype=np.int64) * period_ns + len(episode_ids) * 10**12
            action = np.zeros((n, JOINT_COUNT), dtype=np.float32)
            contact_frame = n // 2
            for index in range(n):
                progress = np.clip((index - n * 0.22) / (n * 0.45), 0.0, 1.0)
                action[index] = _close_synergy(float(progress))
            state = np.vstack([action[0], action[:-1]])
            state += rng.normal(0.0, 0.003, state.shape).astype(np.float32)
            tactile = rng.normal(
                0.0, 0.004, (n, config.tactile_num_fingers, 6)
            ).astype(np.float32)
            for index in range(contact_frame, n):
                level = min(1.0, (index - contact_frame + 1) / 8.0)
                tactile[index, :, 0] += 0.5 * level
                tactile[index, :, 1] += (0.08 if task in {
                    SupportedTask.PLASTIC_BAG, SupportedTask.REFRIGERATOR_DOOR
                } else 0.02) * level
            image_paths = []
            for index in range(n):
                relative = Path("rgb") / f"frame_{index:06d}.png"
                _draw_frame(episode_root / relative, task, index, n)
                image_paths.append(relative.as_posix())
            meta = {
                "schema_version": "revo3-episode-v1",
                "episode_id": episode_id,
                "task": task.value,
                "instruction": DEFAULT_INSTRUCTIONS[task],
                "fps": config.fps,
                "joint_order_hash": JOINT_ORDER_HASH,
                "tactile_num_fingers": config.tactile_num_fingers,
                "action_label_source": "controller_target",
                "image_paths": image_paths,
                "contains_emg": False,
                "synthetic_fixture": True,
            }
            with (episode_root / "meta.json").open("w", encoding="utf-8") as handle:
                json.dump(meta, handle, indent=2, ensure_ascii=False)
            np.savez_compressed(
                episode_root / "frames.npz",
                timestamp_ns=timestamp,
                state_rad=state,
                action_target_rad=action,
                tactile_features=tactile,
            )
    summary = {
        "schema_version": "revo3-synthetic-corpus-v1",
        "synthetic_fixture": True,
        "contains_emg": False,
        "config": asdict(config),
        "episode_ids": episode_ids,
    }
    with (root / "corpus_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary
