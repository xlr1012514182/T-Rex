"""Deterministic synthetic Revo3 episodes for plumbing tests only."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image, ImageDraw

from revo3_v1.planner.schema import (
    DEFAULT_INSTRUCTIONS,
    TASK_GRASP_PRIMITIVES,
    SupportedTask,
)
from revo3_v1.revo.contracts import JOINT_COUNT, JOINT_ORDER_HASH

from .episode import ACTION_SEMANTICS, SUPPORTED_TACTILE_PROFILES


@dataclass(frozen=True)
class SyntheticRevoConfig:
    episodes_per_task: int = 2
    frames_per_episode: int = 64
    fps: int = 30
    tactile_num_fingers: int = 5
    image_width: int = 160
    image_height: int = 120
    seed: int = 20260817
    tactile_profile: str = "ablation_force6d_only"

    def validate(self) -> None:
        if self.episodes_per_task < 1 or self.frames_per_episode < 49:
            raise ValueError("need at least 49 frames for history plus unpadded FLARE futures")
        if self.fps <= 0 or self.tactile_num_fingers != 5:
            raise ValueError("Revo3 V1 synthetic contract requires 30-ish Hz and five fingers")
        if self.fps != 30:
            raise ValueError("the policy smoke fixture is explicitly sampled on the 30 Hz grid")
        if self.tactile_profile not in SUPPORTED_TACTILE_PROFILES:
            raise ValueError(f"unknown tactile profile: {self.tactile_profile}")
        if self.tactile_profile == "profile_c_pressure_matrix":
            raise ValueError("pressure/matrix is not implemented by this policy fixture")


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
            tactile_diff = None
            if config.tactile_profile in {
                "profile_a_force6d_diff", "profile_b_diff_only"
            }:
                # Synthetic gradients are plumbing fixtures, not a learned
                # stand-in for real VisionTouch DIFF observations.
                tactile_diff = np.zeros(
                    (n, config.tactile_num_fingers, 240, 240), dtype=np.uint8
                )
                base = np.linspace(0, 255, 240, dtype=np.uint8)[None, :]
                for index in range(n):
                    tactile_diff[index] = np.clip(
                        base.astype(np.int16)
                        + index
                        + np.arange(config.tactile_num_fingers)[:, None, None],
                        0,
                        255,
                    ).astype(np.uint8)
            image_paths = []
            for index in range(n):
                relative = Path("rgb") / f"frame_{index:06d}.png"
                _draw_frame(episode_root / relative, task, index, n)
                image_paths.append(relative.as_posix())
            instruction = DEFAULT_INSTRUCTIONS[task]
            meta = {
                "schema_version": "revo3-episode-v1",
                "episode_id": episode_id,
                "task_id": f"synthetic-task-{episode_id}",
                "task_version": 1,
                "task": task.value,
                "object_id": task.value,
                "object_instance": f"synthetic-{task.value}-{repetition}",
                "operator": "synthetic-operator",
                "collection_day": f"synthetic-day-{repetition}",
                "grasp_primitive": TASK_GRASP_PRIMITIVES[task].value,
                "instruction": instruction,
                "instruction_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
                "fps": config.fps,
                "joint_order_hash": JOINT_ORDER_HASH,
                "tactile_num_fingers": config.tactile_num_fingers,
                "tactile_profile": config.tactile_profile,
                "checkpoint_family_id": f"synthetic-{config.tactile_profile}-v1",
                "normalization_family_id": f"synthetic-{config.tactile_profile}-norm-v1",
                "capability_manifest_sha256": "a" * 64,
                "camera_profile_id": "revo3_full_center_v1",
                "camera_calibration_sha256": "b" * 64,
                "action_label_source": "controller_target",
                "action_semantics": ACTION_SEMANTICS,
                "contains_cair_residual": False,
                "image_paths": image_paths,
                "instruction_source": "manual_canonical",
                "planner_revision": "",
                "planner_output_sha256": "",
                "time_grid_hz": 30,
                "timestamp_tolerance_ns": 2_000_000,
                "max_command_latency_ns": 1_000_000,
                "max_tactile_inter_finger_skew_ns": 1_000_000,
                "resampled_30hz": True,
                "timestamp_alignment_verified": True,
                "contains_emg": False,
                "synthetic_fixture": True,
            }
            with (episode_root / "meta.json").open("w", encoding="utf-8") as handle:
                json.dump(meta, handle, indent=2, ensure_ascii=False)
            frame_payload = {
                "timestamp_ns": timestamp,
                "camera_capture_timestamp_ns": timestamp - 3_000_000,
                "camera_receive_timestamp_ns": timestamp - 1_000_000,
                "state_timestamp_ns": timestamp - 200_000,
                "action_decision_timestamp_ns": timestamp,
                "action_write_timestamp_ns": timestamp + 100_000,
                "controller_sequence": np.arange(n, dtype=np.int64),
                "request_id_hash": np.asarray(
                    [
                        hashlib.sha256(
                            f"synthetic-{episode_id}-{index:06d}".encode("utf-8")
                        ).hexdigest()
                        for index in range(n)
                    ],
                    dtype="U64",
                ),
                "policy_loss_eligible": np.arange(n) >= 15,
                "phase": np.asarray(
                    [
                        "context" if index < 15
                        else "precontact" if index < contact_frame
                        else "contact" if index < min(n, contact_frame + 12)
                        else "hold"
                        for index in range(n)
                    ],
                    dtype="U16",
                ),
                "state_rad": state,
                "action_target_rad": action,
            }
            if config.tactile_profile in {
                "profile_a_force6d_diff", "ablation_force6d_only"
            }:
                touch_timestamp = timestamp - 500_000
                history_timestamp = np.stack(
                    [
                        touch_timestamp[index]
                        - np.arange(15, -1, -1, dtype=np.int64) * period_ns
                        for index in range(n)
                    ]
                )
                history_sequence = np.stack(
                    [np.arange(index - 15, index + 1, dtype=np.int64) for index in range(n)]
                )
                history_values = np.stack(
                    [
                        np.stack(
                            [tactile[max(0, index - 15 + offset)] for offset in range(16)]
                        )
                        for index in range(n)
                    ]
                )
                frame_payload["tactile_features"] = tactile
                frame_payload["touch_timestamp_ns"] = touch_timestamp
                frame_payload["force6d_finger_timestamp_ns"] = np.repeat(
                    touch_timestamp[:, None], config.tactile_num_fingers, axis=1
                )
                frame_payload["tactile_history_f6"] = history_values
                frame_payload["tactile_history_timestamp_ns"] = history_timestamp
                frame_payload["tactile_history_sequence"] = history_sequence
            if tactile_diff is not None:
                frame_payload["tactile_diff"] = tactile_diff
                frame_payload["tactile_diff_timestamp_ns"] = np.repeat(
                    (timestamp - 600_000)[:, None], config.tactile_num_fingers, axis=1
                )
            np.savez_compressed(episode_root / "frames.npz", **frame_payload)
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
