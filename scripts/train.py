"""
Qwen3-VL MoT training with flare visual prediction for the latent expert.

Extends train_qwen3vl.py with:
  --use_flare 1                   : enable flare query tokens + cosine similarity loss
  --n_flare_tokens_per_frame T    : number of tokens per future frame (default=4)
  --n_flare_steps S               : number of future steps to predict (default=8)
  --flare_loss_weight w           : weight for flare prediction loss (default=0.5)
  --flare_frame_stride s          : temporal stride for flare frame targets (default=frame_stride)
  --flare_layer_index L           : which layer to extract flare hidden states from
                                    (default=-1, i.e. last layer; use e.g. -7 for ~3/4 depth)

  Total flare tokens = n_flare_tokens_per_frame * n_flare_steps.

When --use_flare 0, this script behaves identically to train_qwen3vl.py.
"""

import os, sys

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

import json
import torch
import logging
import argparse
import shutil
import math
import re
import hashlib
import copy
import wandb
import PIL.Image
import PIL.ImageEnhance
import numpy as np

from typing import List, Dict
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from accelerate import Accelerator
from transformers import AutoProcessor, set_seed
from datasets import Dataset as HFDataset

from qwen_vla import Qwen3VLVLAModel, extend_position_ids_for_flare, split_slow_fast_embeds
from revo3_v1.policy.training import (
    build_revo_optimizer_groups,
    configure_revo_trainable_parameters,
    stage_spec as revo_stage_spec,
)
from revo3_v1.policy.checkpoint import load_revo_compatible_state_dict
from revo3_v1.policy.artifacts import (
    sha256_file as artifact_sha256_file,
    validate_revo_deform_artifact,
    validate_revo_vqvae_artifact,
)
from revo3_v1.revo.contracts import JOINT_ORDER_HASH
import cv2

logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")


def _rot6d_to_mat(rot6d):
    """6D rotation (first two columns) → 3×3 rotation matrix."""
    col1 = rot6d[:3]
    col2 = rot6d[3:6]
    col3 = np.cross(col1, col2)
    return np.column_stack([col1, col2, col3])


def _arm9d_to_axis_angle(arm_9d):
    """[trans(3), rot6d(6)] → [trans(3), axis_angle(3)]."""
    R = _rot6d_to_mat(arm_9d[3:9])
    aa, _ = cv2.Rodrigues(R)
    return np.concatenate([arm_9d[:3], aa.flatten()])


def _axis_angle_to_arm9d(arm_aa):
    """[trans(3), axis_angle(3)] → [trans(3), rot6d(6)]."""
    R, _ = cv2.Rodrigues(arm_aa[3:6])
    return np.concatenate([arm_aa[:3], R[:, 0], R[:, 1]])


def add_tracking_error_noise(state, te_mean, te_std, action_dim, clip_rad=None):
    """Add tracking-error noise to robot state.

    Supports single-arm (action_dim=31, te=28D) and bimanual (action_dim=62, te=56D).
    Per arm (28D tracking error = 3 xyz + 3 axis-angle + 22 hand):
      1. Convert arm 9D [trans, rot6d] → [trans, axis_angle]
      2. Sample noise ~ N(te_mean, te_std)
      3. Add noise
      4. Convert back to [trans, rot6d]
    """
    noisy = state.copy()
    if action_dim == 21:
        if np.asarray(state).shape != (21,):
            raise ValueError(f"Revo state must be [21], got {np.asarray(state).shape}")
        if np.asarray(te_mean).shape != (21,) or np.asarray(te_std).shape != (21,):
            raise ValueError(
                "Revo tracking-error statistics must be 21-D; the upstream "
                "action_dim//31 helper would otherwise inject no noise"
            )
        noise = np.random.normal(te_mean, te_std).astype(np.float32)
        if clip_rad is not None and float(clip_rad) > 0:
            noise = np.clip(noise, -float(clip_rad), float(clip_rad))
        return noisy + noise
    n_arms = action_dim // 31
    for arm_idx in range(n_arms):
        offset = arm_idx * 31
        arm_9d = state[offset:offset + 9]
        hand_22d = state[offset + 9:offset + 31]

        arm_aa = _arm9d_to_axis_angle(arm_9d)  # (6,)

        te_off = arm_idx * 28
        noise_arm = np.random.normal(te_mean[te_off:te_off + 6],
                                     te_std[te_off:te_off + 6]).astype(np.float32)
        noise_hand = np.random.normal(te_mean[te_off + 6:te_off + 28],
                                      te_std[te_off + 6:te_off + 28]).astype(np.float32)

        noisy_arm_9d = _axis_angle_to_arm9d(arm_aa + noise_arm)
        noisy[offset:offset + 9] = noisy_arm_9d
        noisy[offset + 9:offset + 31] = hand_22d + noise_hand
    return noisy


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps,
                                     min_lr_ratio=0.0, num_cycles=0.5):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * 2 * num_cycles * progress))
        return (1 - min_lr_ratio) * cosine + min_lr_ratio
    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)


def sample_revo_image_augmentation(seed: int, episode_id: str, frame_index: int):
    """Sample the frozen Revo augmentation once for an atomic two-view item."""
    digest = hashlib.sha256(
        f"{int(seed)}:{episode_id}:{int(frame_index)}".encode("utf-8")
    ).digest()
    rng = np.random.RandomState(int.from_bytes(digest[:4], "little"))
    return {
        "brightness": float(rng.uniform(0.80, 1.20)),
        "contrast": float(rng.uniform(0.85, 1.15)),
        "saturation": float(rng.uniform(0.90, 1.10)),
        "hue": float(rng.uniform(-0.03, 0.03)),
        "rotation_deg": float(rng.uniform(-3.0, 3.0)),
        "translate_x_fraction": float(rng.uniform(-0.03, 0.03)),
        "translate_y_fraction": float(rng.uniform(-0.03, 0.03)),
    }


def apply_revo_image_augmentation(image, params):
    """Apply shared mild geometry/color jitter without changing crop geometry."""
    image = image.convert("RGB")
    width, height = image.size
    # Rotate and translate the already-derived full/center views with the same
    # normalized parameters. No flip and no additional random crop are used.
    image = image.rotate(
        params["rotation_deg"],
        resample=PIL.Image.Resampling.BICUBIC,
        translate=(
            int(round(params["translate_x_fraction"] * width)),
            int(round(params["translate_y_fraction"] * height)),
        ),
        fillcolor=(0, 0, 0),
    )
    image = PIL.ImageEnhance.Brightness(image).enhance(params["brightness"])
    image = PIL.ImageEnhance.Contrast(image).enhance(params["contrast"])
    image = PIL.ImageEnhance.Color(image).enhance(params["saturation"])
    hsv = np.asarray(image.convert("HSV"), dtype=np.uint8).copy()
    hue_shift = int(round(params["hue"] * 255.0))
    hsv[..., 0] = (hsv[..., 0].astype(np.int16) + hue_shift) % 256
    return PIL.Image.fromarray(hsv, mode="HSV").convert("RGB")


class SftDataset(Dataset):
    def __init__(self, config, processor, accelerator, *, training=True):
        self.config = config
        self.processor = processor
        self.accelerator = accelerator
        self.training = bool(training)

        self.hf_dataset = HFDataset.from_json(
            config.data_path, keep_in_memory=False,
        )

        stats_path = getattr(config, "stats_path", "") or config.data_path.replace(
            ".json", "_statistics.json"
        )
        with open(stats_path, "r") as f:
            self.stats_data = json.load(f)
        self.dataset_name = next(iter(self.stats_data))
        first_record = self.hf_dataset[0] if len(self.hf_dataset) else {}
        self.is_revo_dataset = first_record.get("schema_version") == "revo3-trex-json-v1"
        if self.is_revo_dataset:
            if not getattr(config, "stats_path", ""):
                raise ValueError(
                    "Revo datasets require an explicit frozen MIDTRAIN_TRAIN --stats_path"
                )
            expected_profile = str(getattr(config, "tactile_profile", ""))
            if first_record.get("tactile_profile") != expected_profile:
                raise ValueError(
                    "Revo dataset tactile profile does not match the requested checkpoint family"
                )

        def _arr(key, sub):
            return np.array(self.stats_data[self.dataset_name][key][sub])

        self.action_mask = _arr("action", "mask")
        self.action_min  = _arr("action", "q01")
        self.action_max  = _arr("action", "q99")
        self.state_mask  = _arr("state",  "mask")
        self.state_min   = _arr("state",  "q01")
        self.state_max   = _arr("state",  "q99")
        self.tactile_num_fingers = int(getattr(config, "tactile_num_fingers", 10))
        requires_force = bool(
            getattr(config, "use_tactile_vec", 0)
            or getattr(config, "use_tactile_vqvae", 0)
        )
        if requires_force:
            self.tacf6_mask = _arr("tactile_f6", "mask")
            self.tacf6_min = _arr("tactile_f6", "q01")
            self.tacf6_max = _arr("tactile_f6", "q99")
            expected_tactile_dim = self.tactile_num_fingers * 6
            for name, values in (
                ("mask", self.tacf6_mask),
                ("q01", self.tacf6_min),
                ("q99", self.tacf6_max),
            ):
                if np.asarray(values).size != expected_tactile_dim:
                    raise ValueError(
                        f"tactile_f6 {name} has {np.asarray(values).size} values; "
                        f"expected {expected_tactile_dim}; fake hand padding is forbidden"
                    )
            noise = self.stats_data[self.dataset_name].get("tactile_no_contact_noise")
            if not isinstance(noise, dict):
                raise ValueError("Revo Force6D stats require no-contact noise provenance")
            self.tactile_noise_center = np.asarray(
                noise.get("robust_center"), dtype=np.float32
            )
            self.tactile_noise_scale = np.asarray(
                noise.get("robust_scale"), dtype=np.float32
            )
            self.tactile_noise_covariance = np.asarray(
                noise.get("covariance"), dtype=np.float32
            )
            if (
                self.tactile_noise_center.shape != (self.tactile_num_fingers, 6)
                or self.tactile_noise_scale.shape != (self.tactile_num_fingers, 6)
                or self.tactile_noise_covariance.shape
                != (expected_tactile_dim, expected_tactile_dim)
                or noise.get("normalization_family_id")
                != getattr(config, "normalization_family_id", "")
                or noise.get("checkpoint_family_id")
                != getattr(config, "checkpoint_family_id", "")
            ):
                raise ValueError("Revo no-contact noise stats/profile family mismatch")
            if not (
                np.isfinite(self.tactile_noise_center).all()
                and np.isfinite(self.tactile_noise_scale).all()
                and np.isfinite(self.tactile_noise_covariance).all()
            ):
                raise ValueError("Revo no-contact noise statistics must be finite")
            covariance = 0.5 * (
                self.tactile_noise_covariance
                + self.tactile_noise_covariance.T
            )
            eigenvalues, eigenvectors = np.linalg.eigh(covariance.astype(np.float64))
            # Quantizing the train-only covariance to JSON/float32 can create
            # tiny negative eigenvalues.  Project only that numerical residue
            # to the PSD cone; a materially indefinite artifact is rejected.
            if float(eigenvalues.min(initial=0.0)) < -1e-5:
                raise ValueError("Revo no-contact covariance is not positive semidefinite")
            eigenvalues = np.maximum(eigenvalues, 1e-10)
            self.tactile_noise_covariance = (
                (eigenvectors * eigenvalues) @ eigenvectors.T
            ).astype(np.float32)
        else:
            self.tacf6_mask = self.tacf6_min = self.tacf6_max = np.empty((0,))
            self.tactile_noise_center = self.tactile_noise_scale = np.empty((0,))
            self.tactile_noise_covariance = np.empty((0, 0))

        # Tracking error stats for state noise injection (used when use_robot_state=1)
        # Dimension: 28D per arm (6 arm + 22 hand), scales with n_arms
        te_dim = 21 if config.action_dim == 21 else (config.action_dim // 31) * 28
        te_stats = self.stats_data[self.dataset_name].get("tracking_error", {})
        self.te_mean = np.array(te_stats.get("mean", np.zeros(te_dim)), dtype=np.float32)
        self.te_std  = np.array(te_stats.get("std",  np.zeros(te_dim)), dtype=np.float32)

        self.img_dir = os.path.dirname(config.data_path)

        if config.image_size:
            self.image_size = tuple(config.image_size)
        else:
            self.image_size = None

        # For flare frame loading: build per-episode frame lists
        # so we can look up flare frames by index offset.
        self._build_episode_index()

        # Embedded VQ-VAE: build a per-episode F6 index so collate_fn can
        # reconstruct the raw 16-frame history window on-the-fly (same
        # alignment as utils/encode_vqvae_codes_to_json.py).
        self.use_tactile_vqvae = bool(getattr(config, "use_tactile_vqvae", 0))
        self.vqvae_window = int(getattr(config, "vqvae_window", 16))
        self._ep_f6 = {}        # episode -> np.ndarray [N_ep, F, 6]
        self._f6_loc = {}       # (ep_dir, frame) -> position in that episode
        if self.use_tactile_vqvae:
            self._build_f6_history_index()

        accelerator.print(f"Dataset size: {len(self.hf_dataset)}")

    def _build_f6_history_index(self):
        """Group per-sample tactile_f6 by episode (sorted by frame) so a sample
        at frame f can fetch the previous `window` F6 frames (left-padded)."""
        frame_re = re.compile(r"(.+/episode_\d+)/image(\d+)_")
        by_ep = {}  # episode -> list of (frame, f6[F,6])
        n = len(self.hf_dataset)
        for i in range(n):
            s = self.hf_dataset[i]
            deforms = s.get("tactile_image_deform", []) or []
            if s.get("episode_id") is not None and s.get("frame_index") is not None:
                ep_dir, frame = str(s["episode_id"]), int(s["frame_index"])
            else:
                if not deforms:
                    continue
                m = frame_re.search(deforms[0])
                if not m:
                    continue
                ep_dir, frame = m.group(1), int(m.group(2))
            f6 = np.asarray(s["tactile_f6"], dtype=np.float32).reshape(
                self.tactile_num_fingers, 6
            )
            by_ep.setdefault(ep_dir, []).append((frame, f6))
        for ep_dir, items in by_ep.items():
            items.sort(key=lambda t: t[0])
            self._ep_f6[ep_dir] = np.stack([f6 for _, f6 in items], axis=0)
            for pos, (frame, _) in enumerate(items):
                self._f6_loc[(ep_dir, frame)] = pos
        self.accelerator.print(
            f"VQ-VAE F6 history index: {len(self._ep_f6)} episodes, "
            f"window={self.vqvae_window}")

    def _f6_history_for_sample(self, sample) -> np.ndarray:
        """Return the raw [window, F, 6] F6 history ending at this sample.

        Mirrors utils/encode_vqvae_codes_to_json._build_windows: the previous
        `window` samples in episode order, left-edge-padded with the episode's
        first frame.  Falls back to repeating the current frame when the
        episode/frame can't be resolved.
        """
        W = self.vqvae_window
        frame_re = re.compile(r"(.+/episode_\d+)/image(\d+)_")
        deforms = sample.get("tactile_image_deform", []) or []
        cur = np.asarray(sample["tactile_f6"], dtype=np.float32).reshape(
            1, self.tactile_num_fingers, 6
        )
        if sample.get("episode_id") is not None and sample.get("frame_index") is not None:
            ep_dir, frame = str(sample["episode_id"]), int(sample["frame_index"])
        else:
            m = frame_re.search(deforms[0]) if deforms else None
            if m is None:
                return np.repeat(cur, W, axis=0)
            ep_dir, frame = m.group(1), int(m.group(2))
        pos = self._f6_loc.get((ep_dir, frame))
        arr = self._ep_f6.get(ep_dir)
        if pos is None or arr is None:
            return np.repeat(cur, W, axis=0)
        start = pos - (W - 1)
        if start >= 0:
            return arr[start: pos + 1]
        pad = np.repeat(arr[:1], -start, axis=0)
        return np.concatenate([pad, arr[: pos + 1]], axis=0)

    def _build_episode_index(self):
        """Group samples by episode to enable flare frame lookup."""
        # Each sample has image paths like '.../taskname/episodename/imageXX_view.png'
        # We group by episode dir and sort by frame index within each episode.
        # For simplicity, we just store the list of slow image paths per sample index.
        # Flare frames are loaded by offsetting into the video/image sequence.
        pass  # Flare frames are loaded from the JSON data directly

    def __len__(self):
        return len(self.hf_dataset)

    def create_val_split(self, val_ratio=0.05, seed=42):
        """Split train/val, grouping by episode when ``episode_id`` exists."""
        import copy
        n = len(self.hf_dataset)
        rng = np.random.RandomState(seed)
        episode_ids = [self.hf_dataset[i].get("episode_id") for i in range(n)]
        if n and all(value is not None for value in episode_ids):
            unique_episodes = sorted({str(value) for value in episode_ids})
            if len(unique_episodes) < 2:
                raise ValueError("Episode-grouped validation requires at least two episodes")
            permuted = rng.permutation(unique_episodes)
            n_val_episodes = min(
                len(unique_episodes) - 1,
                max(1, int(round(len(unique_episodes) * val_ratio))),
            )
            val_episode_set = set(permuted[:n_val_episodes].tolist())
            val_indices = [
                i for i, value in enumerate(episode_ids) if str(value) in val_episode_set
            ]
            train_indices = [
                i for i, value in enumerate(episode_ids) if str(value) not in val_episode_set
            ]
        else:
            n_val = max(1, int(n * val_ratio))
            perm = rng.permutation(n)
            val_indices = sorted(perm[:n_val].tolist())
            train_indices = sorted(perm[n_val:].tolist())

        val_ds = copy.copy(self)
        val_ds.hf_dataset = self.hf_dataset.select(val_indices)
        val_ds.training = False

        self.hf_dataset = self.hf_dataset.select(train_indices)

        self.accelerator.print(
            f"Train/Val split: {len(self.hf_dataset)} train, "
            f"{len(val_ds.hf_dataset)} val samples")
        return val_ds

    def __getitem__(self, idx):
        return self.hf_dataset[idx]

    @staticmethod
    def _normalize(values, mask, vmin, vmax):
        return np.where(
            mask,
            np.clip(2 * (values - vmin) / (vmax - vmin + 1e-8) - 1, -1, 1),
            values,
        )

    def _open(self, rel_path):
        img = PIL.Image.open(os.path.join(self.img_dir, rel_path)).convert("RGB")
        if self.image_size is not None:
            img = img.resize(self.image_size, PIL.Image.LANCZOS)
        return img

    def _revo_augment_params(self, sample):
        if not (
            self.training
            and self.is_revo_dataset
            and bool(getattr(self.config, "revo_image_augmentation", 0))
        ):
            return None
        return sample_revo_image_augmentation(
            getattr(self.config, "augmentation_seed", getattr(self.config, "seed", 42)),
            str(sample.get("episode_id", "")),
            int(sample.get("frame_index", -1)),
        )

    def _open_with_params(self, rel_path, params):
        image = self._open(rel_path)
        return image if params is None else apply_revo_image_augmentation(image, params)

    def _open_gray(self, path):
        full = path if os.path.isabs(path) else os.path.join(self.img_dir, path)
        img = PIL.Image.open(full).convert("L")
        return np.array(img, dtype=np.float32) / 255.0

    def _beta_sample(self, n, device):
        d = torch.distributions.Beta(
            torch.tensor(1.5, dtype=torch.float32, device=device),
            torch.tensor(1.0, dtype=torch.float32, device=device),
        )
        return (d.sample((n,)) * 0.999 + 0.001).to(torch.bfloat16)

    def _add_revo_tactile_noise(self, values):
        if not (
            self.training
            and self.is_revo_dataset
            and bool(getattr(self.config, "revo_tactile_noise_augmentation", 0))
        ):
            return values
        arr = np.asarray(values, dtype=np.float32).copy()
        flat_count = int(np.prod(arr.shape[:-2]))
        noise = np.random.multivariate_normal(
            np.zeros(self.tactile_num_fingers * 6, dtype=np.float32),
            self.tactile_noise_covariance,
            size=flat_count,
            check_valid="raise",
        ).astype(np.float32)
        bound = np.maximum(3.0 * self.tactile_noise_scale.reshape(-1), 1e-8)
        noise = np.clip(noise, -bound, bound).reshape(arr.shape)
        return arr + noise

    def _load_flare_frame(self, sample, k, K, frame_stride):
        slow_path = sample.get("input_image_slow", [""])[0]
        if not slow_path:
            return None

        # Extract frame index from path: image{idx}_{view}.png
        match = re.search(r'image(\d+)_', os.path.basename(slow_path))
        if not match:
            return None

        current_idx = int(match.group(1))
        flare_idx = current_idx + (k + 1) * frame_stride
        flare_path = re.sub(r'image\d+_', f'image{flare_idx}_', slow_path)

        full_path = os.path.join(self.img_dir, flare_path) if not os.path.isabs(flare_path) else flare_path
        if os.path.exists(full_path):
            try:
                return self._open(flare_path)
            except (PIL.UnidentifiedImageError, OSError):
                return None
        else:
            # Out of episode range → use last available or return None
            return None

    def collate_fn(self, batch: List[Dict]) -> Dict:
        cfg = self.config
        B = len(batch)
        is_revo = [x.get("schema_version") == "revo3-trex-json-v1" for x in batch]
        if any(is_revo) and not all(is_revo):
            raise ValueError("one batch cannot mix Revo and upstream T-Rex schemas")
        if all(is_revo):
            for sample in batch:
                def _contains_emg(value):
                    if isinstance(value, dict):
                        return any(
                            "emg" in str(key).lower() or _contains_emg(item)
                            for key, item in value.items()
                            if str(key).lower() != "contains_emg"
                        )
                    if isinstance(value, (list, tuple)):
                        return any(_contains_emg(item) for item in value)
                    return False
                if _contains_emg(sample) or sample.get("contains_emg") is not False:
                    raise ValueError("EMG is recursively forbidden from every Revo policy batch")
                if sample.get("tactile_profile") != cfg.tactile_profile:
                    raise ValueError("one batch cannot cross tactile profile/checkpoint families")
                if sample.get("checkpoint_family_id") != cfg.checkpoint_family_id:
                    raise ValueError("dataset/checkpoint tactile family mismatch")
                if sample.get("normalization_family_id") != cfg.normalization_family_id:
                    raise ValueError("dataset/normalization tactile family mismatch")
                if sample.get("action_label_source") != "controller_target" or sample.get(
                    "action_semantics"
                ) != "accepted_exact_sent_teleop_target":
                    raise ValueError("Revo labels must be accepted exact-sent controller targets")
                if sample.get("contains_cair_residual") is not False:
                    raise ValueError("CAIR-adjusted commands cannot supervise the nominal policy")
                if sample.get("policy_loss_eligible") is not True:
                    raise ValueError("ineligible pre-roll frame reached the policy loss")
                if len(sample.get("action_write_timestamp_ns", [])) != 16:
                    raise ValueError("Revo action chunk lacks 16 write-boundary receipts")
                if len(set(sample.get("action_request_id_hash", []))) != 16:
                    raise ValueError("Revo action chunk request receipts must be unique")
                slow_views = sample.get("input_image_slow")
                fast_views = sample.get("input_image_fast")
                if not (
                    isinstance(slow_views, list) and len(slow_views) == 1
                    and isinstance(fast_views, list) and len(fast_views) == 1
                ):
                    raise ValueError(
                        "Revo samples require full->slow and fixed_center->fast slots"
                    )
                if sample.get("rgb_full_timestamp_ns") != sample.get("rgb_center_timestamp_ns"):
                    raise ValueError("Revo full and fixed-center views must share a timestamp")
                if sample.get("rgb_receive_timestamp_ns", 2**63 - 1) > sample.get(
                    "action_decision_timestamp_ns", -1
                ):
                    raise ValueError("Revo camera observation arrived after policy decision")

        actions = np.array([x["action"] for x in batch], dtype=np.float32)
        actions = actions.reshape(B, -1, cfg.action_dim)
        norm_actions = self._normalize(actions, self.action_mask, self.action_min, self.action_max)
        norm_actions = torch.tensor(norm_actions, dtype=torch.bfloat16)

        device_cpu = norm_actions.device
        time = self._beta_sample(B, device_cpu)
        t_ = time[:, None, None]
        noise = torch.randn_like(norm_actions)
        x_t = t_ * noise + (1 - t_) * norm_actions
        u_t = noise - norm_actions

        # For the reviewed Revo schema, independently sample one of the
        # physical delay offsets 0/4/8/12 for each item.  The matching dense
        # history is selected below for the embedded VQ-VAE.  Legacy upstream
        # records retain their original current-frame behavior.
        selected_delay_indexes = np.zeros(B, dtype=np.int64)
        # Index 2 is the causal latest sample. Training may jitter to the two
        # preceding candidates; validation/test always use latest.
        selected_jitter_indexes = np.full(B, 2, dtype=np.int64)
        selected_tacf6 = None
        if all(is_revo):
            delayed_rows = []
            force_required = cfg.tactile_profile in {
                "profile_a_force6d_diff", "ablation_force6d_only"
            }
            for item_index, sample in enumerate(batch):
                if sample.get("tactile_delay_offsets") != [0, 4, 8, 12]:
                    raise ValueError("Revo tactile_delay_offsets must be [0,4,8,12]")
                selected_delay_indexes[item_index] = np.random.randint(0, 4)
                if self.training:
                    selected_jitter_indexes[item_index] = np.random.randint(0, 3)
                if sample.get("tactile_temporal_jitter_samples") != [-1, 0, 1]:
                    raise ValueError("Revo tactile jitter contract must be [-1,0,1]")
                if sample.get("tactile_temporal_jitter_native_offsets") != [-2, -1, 0]:
                    raise ValueError(
                        "Revo tactile jitter must use causal native offsets [-2,-1,0]"
                    )
                if force_required:
                    delayed = np.asarray(
                        sample.get("tactile_f6_delayed_jitter"), dtype=np.float32
                    )
                    expected = (4, 3, self.tactile_num_fingers, 6)
                    if delayed.shape != expected:
                        raise ValueError(
                            f"Revo tactile_f6_delayed_jitter must be {expected}"
                        )
                    di = selected_delay_indexes[item_index]
                    ji = selected_jitter_indexes[item_index]
                    delayed_rows.append(delayed[di, ji])
                    touch_ts = np.asarray(
                        sample.get("touch_timestamp_ns_delayed_jitter"), dtype=np.int64
                    )
                    decisions = np.asarray(
                        sample.get("tactile_decision_timestamp_ns_delayed"), dtype=np.int64
                    )
                    if touch_ts.shape != (4, 3) or decisions.shape != (4,):
                        raise ValueError("Revo Force6D delay timestamps have invalid shape")
                    if touch_ts[di, ji] > decisions[di]:
                        raise ValueError("selected Force6D sample is from the future")
                else:
                    if sample.get("tactile_f6") not in (None, []):
                        raise ValueError("DIFF-only Profile B cannot carry fake Force6D")
            if force_required:
                selected_tacf6 = np.stack(delayed_rows, axis=0)
                selected_tacf6 = self._add_revo_tactile_noise(selected_tacf6)

        norm_tacf6 = None
        if cfg.use_tactile_vec:
            tacf6 = (
                selected_tacf6
                if selected_tacf6 is not None
                else np.array([x["tactile_f6"] for x in batch], dtype=np.float32)
            )
            tacf6 = tacf6.reshape(B, -1)
            norm_tacf6 = self._normalize(tacf6, self.tacf6_mask,
                                         self.tacf6_min, self.tacf6_max)
            norm_tacf6 = torch.tensor(norm_tacf6.reshape(B, -1, 6), dtype=torch.bfloat16)

        deforms_tensor = None
        if cfg.use_tactile_deform:
            deforms = []
            for item_index, x in enumerate(batch):
                if all(is_revo):
                    delayed_paths = x.get("tactile_image_deform_delayed_jitter")
                    delayed_ts = np.asarray(
                        x.get("tactile_deform_timestamp_ns_delayed_jitter"),
                        dtype=np.int64,
                    )
                    expected_ts = (4, 3, self.tactile_num_fingers)
                    if not (
                        isinstance(delayed_paths, list)
                        and len(delayed_paths) == 4
                        and all(
                            isinstance(options, list)
                            and len(options) == 3
                            and all(isinstance(paths, list) and len(paths) == 5 for paths in options)
                            for options in delayed_paths
                        )
                    ) or delayed_ts.shape != expected_ts:
                        raise ValueError(
                            "Revo Profile A/B requires delayed DIFF [4,3,5] paths/timestamps"
                        )
                    di = selected_delay_indexes[item_index]
                    ji = selected_jitter_indexes[item_index]
                    decisions = np.asarray(
                        x.get("tactile_decision_timestamp_ns_delayed"), dtype=np.int64
                    )
                    if np.any(delayed_ts[di, ji] > decisions[di]):
                        raise ValueError("selected DIFF image is from the future")
                    paths = delayed_paths[di][ji]
                else:
                    paths = x.get("tactile_image_deform", [])
                imgs = [self._open_gray(p) for p in paths]
                if any(image.shape != (240, 240) for image in imgs):
                    raise ValueError("each tactile DIFF input must be one [240,240] grayscale image")
                deforms.append(imgs)
            observed = np.asarray(deforms, dtype=np.float32)
            if observed.shape != (B, self.tactile_num_fingers, 240, 240):
                raise ValueError("tactile DIFF batch must be [B,5,240,240]")
            deforms_tensor = torch.tensor(observed).unsqueeze(2)

        # Tactile codes: either embedded-VQ-VAE (raw F6 history, encoded in the
        # model) or pre-baked codes from the JSON (legacy).  The two are
        # mutually exclusive — when the model owns the VQ-VAE we ship raw F6.
        tactile_codes_tensor = None
        tactile_f6_history_tensor = None
        if getattr(cfg, "use_tactile_vqvae", 0):
            # Embedded VQ-VAE: prefer prepared codes if the JSON already carries
            # them; otherwise ship the raw F6 history and let the model encode
            # on the fly.
            have_codes = all(
                (x.get("tactile_codes") is not None) for x in batch)
            if have_codes:
                tactile_codes_tensor = torch.from_numpy(
                    np.array([x["tactile_codes"] for x in batch], dtype=np.int64))
            else:
                if all(is_revo):
                    histories = []
                    for item_index, sample in enumerate(batch):
                        delayed_histories = np.asarray(
                            sample.get("tactile_f6_history_delayed_jitter"), dtype=np.float32
                        )
                        expected = (
                            4, 3, self.vqvae_window, self.tactile_num_fingers, 6
                        )
                        if delayed_histories.shape != expected:
                            raise ValueError(
                                "Revo tactile_f6_history_delayed_jitter must have shape "
                                f"{expected}, got {delayed_histories.shape}"
                            )
                        history_ts = np.asarray(
                            sample.get("tactile_f6_history_timestamp_ns_delayed_jitter"),
                            dtype=np.int64,
                        )
                        history_seq = np.asarray(
                            sample.get("tactile_f6_history_sequence_delayed_jitter"),
                            dtype=np.int64,
                        )
                        if history_ts.shape != (4, 3, self.vqvae_window) or history_seq.shape != (
                            4, 3, self.vqvae_window
                        ):
                            raise ValueError("Revo VQ history timestamp/sequence shape mismatch")
                        di = selected_delay_indexes[item_index]
                        ji = selected_jitter_indexes[item_index]
                        if np.any(np.diff(history_ts[di, ji]) <= 0) or np.any(
                            np.diff(history_seq[di, ji]) <= 0
                        ):
                            raise ValueError("Revo VQ history must contain 16 distinct native samples")
                        histories.append(
                            delayed_histories[
                                di,
                                ji,
                            ]
                        )
                    hist = np.stack(histories, axis=0)
                    hist = self._add_revo_tactile_noise(hist)
                else:
                    hist = np.stack(
                        [self._f6_history_for_sample(x) for x in batch], axis=0
                    )
                tactile_f6_history_tensor = torch.from_numpy(hist.astype(np.float32))
        elif getattr(cfg, "use_tactile_code", 0):
            codes = np.array(
                [x["tactile_codes"] for x in batch], dtype=np.int64)   # [B, 2]
            tactile_codes_tensor = torch.from_numpy(codes)

        state_raw_list = []
        if cfg.use_robot_state:
            for x in batch:
                state_raw = np.array(x["state_fast"], dtype=np.float32)
                state_raw = add_tracking_error_noise(state_raw,
                                                     self.te_mean, self.te_std,
                                                     cfg.action_dim,
                                                     getattr(cfg, "tracking_error_clip_rad", None))
                norm_state = self._normalize(state_raw, self.state_mask,
                                     self.state_min, self.state_max)
                if (getattr(cfg, "state_dropout", 0.0) > 0
                        and np.random.random() < float(cfg.state_dropout)):
                    norm_state = np.zeros_like(norm_state)
                state_raw_list.append(torch.tensor(norm_state, dtype=torch.bfloat16))

        all_input_ids = []
        all_pixel_values = []
        all_grid_thw = []
        n_slow_images = 0

        for x in batch:
            slow_imgs = x.get("input_image_slow", [])
            fast_imgs = x.get("input_image_fast", [])
            if n_slow_images and n_slow_images != len(slow_imgs):
                raise ValueError("all samples in a batch must have the same image-slot count")
            n_slow_images = len(slow_imgs)

            aug_params = self._revo_augment_params(x)
            pil_slow = [self._open_with_params(p, aug_params) for p in slow_imgs]
            pil_fast = [self._open_with_params(p, aug_params) for p in fast_imgs]
            all_pil = pil_slow + pil_fast # a list, including all images

            content = []
            for _ in pil_slow:
                content.append({"type": "image"})
            content.append({"type": "text", "text": x.get("input_prompt", "")})
            for _ in pil_fast:
                content.append({"type": "image"})

            messages = [{"role": "user", "content": content}]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inp = self.processor(
                text=text,
                images=all_pil if all_pil else None,
                return_tensors="pt", padding=False,
            )
            all_input_ids.append(inp.input_ids[0])
            if "pixel_values" in inp and inp.pixel_values is not None:
                all_pixel_values.append(inp.pixel_values)
                all_grid_thw.append(inp.image_grid_thw)

        n_flare_steps = cfg.n_flare_steps if cfg.use_flare else 0
        flare_stride = cfg.flare_frame_stride
        flare_pixel_values = None
        flare_grid_thw = None

        if n_flare_steps > 0: # use flare
            flare_pil_imgs = []
            for x in batch:
                aug_params = self._revo_augment_params(x)
                if x.get("schema_version") == "revo3-trex-json-v1":
                    flare_paths = x.get("flare_image_full")
                    flare_timestamps = x.get("flare_timestamp_ns")
                    if not (
                        isinstance(flare_paths, list)
                        and len(flare_paths) == n_flare_steps
                        and isinstance(flare_timestamps, list)
                        and len(flare_timestamps) == n_flare_steps
                    ):
                        raise ValueError(
                            "Revo FLARE requires every explicit unpadded future full frame"
                        )
                    for path in flare_paths:
                        flare_pil_imgs.append(self._open_with_params(path, aug_params))
                else:
                    for k in range(n_flare_steps):
                        flare_image = self._load_flare_frame(
                            x, k, n_flare_steps, flare_stride
                        )
                        if flare_image is None:
                            slow_paths = x.get("input_image_slow", [])
                            flare_image = (
                                self._open(slow_paths[0])
                                if slow_paths
                                else PIL.Image.new(
                                    "RGB", self.image_size or (384, 288)
                                )
                            )
                        flare_pil_imgs.append(flare_image)

            flare_inp = self.processor.image_processor(flare_pil_imgs, return_tensors="pt")
            flare_pixel_values = flare_inp.pixel_values.to(torch.bfloat16)
            flare_grid_thw = flare_inp.image_grid_thw

        pad_id = self.processor.tokenizer.pad_token_id or 0
        max_len = max(ids.shape[0] for ids in all_input_ids)

        padded_ids = []
        attention_ms = []
        for ids in all_input_ids:
            pad_len = max_len - ids.shape[0]
            padded_ids.append(F.pad(ids, (pad_len, 0), value=pad_id))
            attn = torch.ones(max_len, dtype=torch.long)
            if pad_len > 0:
                attn[:pad_len] = 0
            attention_ms.append(attn)

        input_ids = torch.stack(padded_ids)
        attention_mask = torch.stack(attention_ms)
        pixel_values = (torch.cat(all_pixel_values, dim=0) if all_pixel_values else None)
        image_grid_thw = (torch.cat(all_grid_thw, dim=0) if all_grid_thw else None)

        state_raw = torch.stack(state_raw_list) if state_raw_list else None

        # Revo records above supply an explicitly aligned delay/history.  The
        # current-frame fallback remains only for legacy upstream records.
        time_r = self._beta_sample(B, device_cpu)
        eps_r  = torch.randn_like(norm_actions)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "n_slow_images": n_slow_images,
            "noisy_actions": x_t,
            "target": u_t,
            "timesteps": time,
            "norm_actions": norm_actions,            # for L_refine residual target
            "tactile_f6s": norm_tacf6,
            "tactile_deforms": deforms_tensor,
            "tactile_f6s_delayed": norm_tacf6,
            "tactile_deforms_delayed": deforms_tensor,
            "tactile_codes": tactile_codes_tensor,   # [B, 2] int64 or None
            "tactile_f6_history": tactile_f6_history_tensor,
            "tactile_delay_index": torch.from_numpy(selected_delay_indexes),
            "time_r": time_r,
            "eps_r": eps_r,
            "state_raw": state_raw,
            "flare_pixel_values": flare_pixel_values, # [B*S patches, C] or None (S = n_flare_steps)
            "flare_grid_thw": flare_grid_thw, # [B*S, 3] or None
        }


def save_checkpoint(model, processor, accelerator, args, epoch, global_step, stats_data):
    save_dir = os.path.join(args.output_dir, f"checkpoint-{epoch}-{global_step}")

    if accelerator.is_main_process:
        ckpts = [f for f in os.listdir(args.output_dir) if f.startswith("checkpoint-")]
        if args.max_ckpts > 0 and len(ckpts) >= args.max_ckpts:
            oldest = min(ckpts, key=lambda f: os.path.getctime(os.path.join(args.output_dir, f)))
            shutil.rmtree(os.path.join(args.output_dir, oldest))

        os.makedirs(save_dir, exist_ok=True)

        sd = accelerator.get_state_dict(model)
        model_file = os.path.join(save_dir, "model.pt")
        torch.save(sd, model_file)
        digest = hashlib.sha256()
        with open(model_file, "rb") as model_handle:
            for block in iter(lambda: model_handle.read(1024 * 1024), b""):
                digest.update(block)
        checkpoint_sha256 = digest.hexdigest()

        processor.save_pretrained(os.path.join(save_dir, "processor"))

        src_config = os.path.join(args.model_path, "config.json")
        if os.path.exists(src_config):
            shutil.copy(src_config, os.path.join(save_dir, "config.json"))

        with open(os.path.join(save_dir, "training_args.json"), "w") as f:
            json.dump({
                "model_path": args.model_path,
                "action_dim": args.action_dim,
                "action_chunk": args.action_chunk,
                "tactile_num_fingers": getattr(args, "tactile_num_fingers", 10),
                "use_robot_state": args.use_robot_state,
                "use_tactile_deform": args.use_tactile_deform,
                "use_tactile_vec": getattr(args, "use_tactile_vec", 0),
                "tactile_intermediate_size": getattr(args, "tactile_intermediate_size", 0),
                "training_stage": args.training_stage,
                "revo_training_stage": getattr(args, "revo_training_stage", "off"),
                "tactile_profile": getattr(args, "tactile_profile", ""),
                "checkpoint_family_id": getattr(args, "checkpoint_family_id", ""),
                "normalization_family_id": getattr(args, "normalization_family_id", ""),
                "resume_kind": getattr(args, "resume_source", ""),
                "source_checkpoint_sha256": getattr(args, "source_checkpoint_sha256", ""),
                "split_manifest_sha256": getattr(args, "split_manifest_sha256", ""),
                "tactile_profile_manifest_sha256": getattr(args, "tactile_profile_manifest_sha256", ""),
                "normalization_statistics_sha256": getattr(
                    args, "normalization_statistics_sha256", ""
                ),
                "normalization_artifact_sha256": getattr(
                    args, "normalization_artifact_sha256", ""
                ),
                "capability_manifest_sha256": getattr(args, "capability_manifest_sha256", ""),
                "vqvae_artifact_sha256": artifact_sha256_file(args.vqvae_artifact)
                if getattr(args, "vqvae_artifact", "") else "",
                "deform_encoder_artifact_sha256": artifact_sha256_file(
                    args.deform_encoder_artifact
                ) if getattr(args, "deform_encoder_artifact", "") else "",
                "camera_profile": "revo3_full_center_v1" if getattr(args, "revo_training_stage", "off") != "off" else "official_single_view",
                "view_slots": {"slow": "full", "fast": "fixed_center"} if getattr(args, "revo_training_stage", "off") != "off" else None,
                "state_dropout": getattr(args, "state_dropout", 0.0),
                "tracking_error_clip_rad": getattr(args, "tracking_error_clip_rad", None),
                "max_steps": getattr(args, "max_steps", 0),
                "use_flare": args.use_flare,
                "n_flare_tokens_per_frame": args.n_flare_tokens_per_frame,
                "n_flare_steps": args.n_flare_steps,
                "flare_layer_index": args.flare_layer_index,
                "use_tactile_code": getattr(args, "use_tactile_code", 0),
                "vqvae_codebook_size": getattr(args, "vqvae_codebook_size", 64),
                "use_tactile_vqvae": getattr(args, "use_tactile_vqvae", 0),
                "vqvae_config": getattr(args, "vqvae_config_dict", None),
                "paradigm": "cascaded",
                "cascaded_total_steps": getattr(args, "cascaded_total_steps", 10),
                "cascaded_split_step":  getattr(args, "cascaded_split_step", 6),
                "flare_frame_stride": getattr(args, "flare_frame_stride", 2),
                "revo_image_augmentation": getattr(args, "revo_image_augmentation", 0),
                "revo_tactile_noise_augmentation": getattr(
                    args, "revo_tactile_noise_augmentation", 0
                ),
                "augmentation_seed": getattr(args, "augmentation_seed", 42),
            }, f, indent=2)

        with open(os.path.join(save_dir, "stats_data.json"), "w") as f:
            json.dump(stats_data, f, indent=2)

        if getattr(args, "revo_training_stage", "off") != "off":
            with open(
                os.path.join(save_dir, "checkpoint_lineage.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    {
                        "schema_version": "revo3-checkpoint-lineage-v1",
                        "checkpoint_sha256": checkpoint_sha256,
                        "stage": args.revo_training_stage,
                        "resume_kind": args.resume_source,
                        "parent_checkpoint_sha256": getattr(
                            args, "source_checkpoint_sha256", ""
                        ),
                        "split_manifest_sha256": getattr(
                            args, "split_manifest_sha256", ""
                        ),
                        "tactile_profile_manifest_sha256": getattr(
                            args, "tactile_profile_manifest_sha256", ""
                        ),
                        "normalization_statistics_sha256": getattr(
                            args, "normalization_statistics_sha256", ""
                        ),
                        "normalization_artifact_sha256": getattr(
                            args, "normalization_artifact_sha256", ""
                        ),
                        "capability_manifest_sha256": getattr(
                            args, "capability_manifest_sha256", ""
                        ),
                        "vqvae_artifact_sha256": artifact_sha256_file(
                            args.vqvae_artifact
                        ) if getattr(args, "vqvae_artifact", "") else "",
                        "deform_encoder_artifact_sha256": artifact_sha256_file(
                            args.deform_encoder_artifact
                        ) if getattr(args, "deform_encoder_artifact", "") else "",
                        "action_dim": args.action_dim,
                        "action_chunk": args.action_chunk,
                        "joint_order_hash": JOINT_ORDER_HASH,
                        "camera_profile": "revo3_full_center_v1",
                        "view_slots": {"slow": "full", "fast": "fixed_center"},
                        "tactile_profile": args.tactile_profile,
                        "checkpoint_family_id": args.checkpoint_family_id,
                        "normalization_family_id": args.normalization_family_id,
                    },
                    handle,
                    indent=2,
                )

    accelerator.wait_for_everyone()
    logger.info(f"Checkpoint {epoch}-{global_step} saved.")


class TrainingMetrics:
    def __init__(self, device):
        self.n_step       = 0
        self.action_loss  = torch.tensor(0.0, device=device)
        self.tactile_loss = torch.tensor(0.0, device=device)
        self.flare_loss  = torch.tensor(0.0, device=device)
        self.total_loss   = torch.tensor(0.0, device=device)
        self.world_size   = dist.get_world_size()

    def update(self, total, action, tactile=0.0, flare=0.0):
        self.n_step += 1
        self.total_loss   += total.item() if torch.is_tensor(total) else total
        self.action_loss  += action.item() if torch.is_tensor(action) else action
        self.tactile_loss += tactile.item() if torch.is_tensor(tactile) else tactile
        self.flare_loss  += flare.item() if torch.is_tensor(flare) else flare

    def get_metric(self, reset=True):
        for t in [self.total_loss, self.action_loss, self.tactile_loss, self.flare_loss]:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        denom = self.world_size * max(self.n_step, 1)
        m = {
            "total_loss":   self.total_loss.item() / denom,
            "action_loss":  self.action_loss.item() / denom,
            "tactile_loss": self.tactile_loss.item() / denom,
            "flare_loss":  self.flare_loss.item() / denom,
        }
        if reset:
            self.n_step = 0
            for t in [self.total_loss, self.action_loss, self.tactile_loss, self.flare_loss]:
                t.fill_(0)
        return m


@torch.no_grad()
def run_validation(model, val_dataloader, accelerator, args,
                   is_stage1, use_flare, K, T_per_frame, flare_layer_idx):
    """Run validation and return averaged metrics (action + tactile + flare)."""
    model.eval()
    device = torch.cuda.current_device()
    val_act = torch.tensor(0.0, device=device)
    val_tac = torch.tensor(0.0, device=device)
    val_flare = torch.tensor(0.0, device=device)
    n_val = 0
    max_batches = getattr(args, "max_val_batches", 50)

    for i, batch in enumerate(val_dataloader):
        if i >= max_batches:
            break
        raw_model = accelerator.unwrap_model(model)

        inputs_embeds = raw_model.prepare_inputs_embeds(
            input_ids=batch["input_ids"],
            pixel_values=batch.get("pixel_values"),
            image_grid_thw=batch.get("image_grid_thw"))

        n_slow_imgs = batch["n_slow_images"]
        grid_thw = batch.get("image_grid_thw")
        B = inputs_embeds.shape[0]
        if grid_thw is not None and grid_thw.shape[0] > n_slow_imgs:
            merge = getattr(raw_model.visual, "spatial_merge_size", 2)
            n_imgs_per_sample = grid_thw.shape[0] // B
            n_slow_img_tokens = sum(
                int(g[0] * (g[1] // merge) * (g[2] // merge))
                for g in grid_thw[:n_imgs_per_sample][:n_slow_imgs])
            slow_embeds, fast_embeds = split_slow_fast_embeds(
                inputs_embeds, batch["input_ids"],
                raw_model.image_token_id, n_slow_img_tokens)
        else:
            slow_embeds = inputs_embeds
            fast_embeds = inputs_embeds[:, :0]

        L_slow = slow_embeds.shape[1]
        pos_ids, _ = raw_model.get_rope_index(
            input_ids=batch["input_ids"],
            image_grid_thw=batch.get("image_grid_thw"),
            attention_mask=batch["attention_mask"])
        pos_ids = pos_ids[:, :, :L_slow]

        if use_flare:
            flare_q = raw_model.flare_queries.expand(B, -1, -1).to(
                device=slow_embeds.device, dtype=slow_embeds.dtype)
            slow_embeds_ext = torch.cat([slow_embeds, flare_q], dim=1)
            pos_ids = extend_position_ids_for_flare(pos_ids, K)
            L_latent = slow_embeds_ext.shape[1]
        else:
            slow_embeds_ext = slow_embeds
            L_latent = L_slow

        if args.use_robot_state and batch["state_raw"] is not None:
            state_vec = batch["state_raw"].to(slow_embeds.device, dtype=slow_embeds.dtype)
            state_embeds = raw_model.state_embedder(state_vec).unsqueeze(1)
        else:
            state_embeds = torch.empty((B, 0, slow_embeds.shape[2]),
                                       device=slow_embeds.device, dtype=slow_embeds.dtype)
        n_state = state_embeds.shape[1]

        noisy_actions = raw_model.x_embedder(batch["noisy_actions"].to(slow_embeds.dtype))
        timesteps = raw_model.t_embedder(batch["timesteps"].to(slow_embeds.dtype)).unsqueeze(1)
        chunk = args.action_chunk
        target = batch["target"].to(slow_embeds.dtype)
        n_fast = fast_embeds.shape[1]
        has_any_tac = bool(args.use_tactile_vec or args.use_tactile_deform)

        if has_any_tac:
            # Cascaded validation — full L_flow on action expert (τ ∈ [0, 1])
            # + L_flow_tactile on tactile expert at τ ∈ [0, τ_split].
            full_embeds = torch.cat([
                slow_embeds_ext, fast_embeds, state_embeds, timesteps, noisy_actions,
            ], dim=1)
            L_total = full_embeds.shape[1]
            outputs = model.model(
                inputs_embeds=full_embeds, position_ids=pos_ids,
                attention_mask=batch["attention_mask"], use_cache=False,
                output_hidden_states=use_flare,
                latent_indexes=torch.arange(0, L_latent, device=full_embeds.device),
                action_indexes=torch.arange(L_latent, L_total, device=full_embeds.device),
                tactile_indexes=torch.arange(0, 0, device=full_embeds.device))
            hidden = outputs.last_hidden_state
            act_start = L_latent + n_fast + n_state + 1
            v_act = raw_model.final_layer(hidden[:, act_start:act_start + chunk, :])
            loss_act = nn.MSELoss()(v_act, target)

            fe = fast_embeds if n_fast > 0 else None
            se = state_embeds if n_state > 0 else None
            ahat_noise = torch.randn_like(batch["noisy_actions"])
            # cached_kv at τ=tau_split (matches inference).
            _, cached_kv, n_action_in_cache, _ = (
                raw_model.forward_flow_action_partial(
                    inputs_embeds=slow_embeds_ext,
                    position_ids=pos_ids,
                    noise=ahat_noise,
                    attention_mask=batch["attention_mask"],
                    state_embeds=se,
                    fast_embeds=fe,
                    num_steps_total=args.cascaded_total_steps,
                    split_step=args.cascaded_split_step,
                    refresh_clean_kv=True,
                ))

            norm_actions_gt = batch["norm_actions"].to(slow_embeds.dtype)
            eps_r = batch["eps_r"].to(slow_embeds.dtype)
            tau_split = 1.0 - args.cascaded_split_step / args.cascaded_total_steps
            tau_t = batch["time_r"].to(slow_embeds.dtype) * tau_split
            tau_t_b = tau_t[:, None, None]
            x_tau = (1 - tau_t_b) * norm_actions_gt + tau_t_b * eps_r
            v_target_r = eps_r - norm_actions_gt
            v_pred_r = raw_model.tactile_flow_train_step(
                cached_kv=cached_kv,
                latent_position_ids=pos_ids,
                n_action_in_cache=n_action_in_cache,
                x_tau=x_tau,
                tau=tau_t,
                tactile_f6=batch.get("tactile_f6s_delayed"),
                tactile_deform=batch.get("tactile_deforms_delayed"),
                tactile_codes=batch.get("tactile_codes"),
                tactile_f6_history=batch.get("tactile_f6_history"),
            )
            loss_tac = nn.MSELoss()(v_pred_r, v_target_r)
        else:
            # Stage-1 / tactile-free validation: action expert only.
            full_embeds = torch.cat([
                slow_embeds_ext, fast_embeds, state_embeds, timesteps, noisy_actions,
            ], dim=1)
            L_total = full_embeds.shape[1]
            outputs = model.model(
                inputs_embeds=full_embeds, position_ids=pos_ids,
                attention_mask=batch["attention_mask"], use_cache=False,
                output_hidden_states=use_flare,
                latent_indexes=torch.arange(0, L_latent, device=full_embeds.device),
                action_indexes=torch.arange(L_latent, L_total, device=full_embeds.device),
                tactile_indexes=torch.arange(0, 0, device=full_embeds.device))
            hidden = outputs.last_hidden_state
            act_start = L_latent + n_fast + n_state + 1
            v_act = raw_model.final_layer(hidden[:, act_start:act_start + chunk, :])
            loss_act = nn.MSELoss()(v_act, target)
            loss_tac = 0.0

        val_act += loss_act.item()
        val_tac += (loss_tac.item() if torch.is_tensor(loss_tac) else loss_tac)
        n_val += 1

    for t in [val_act, val_tac, val_flare]:
        if dist.is_initialized():
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    ws = dist.get_world_size() if dist.is_initialized() else 1
    denom = ws * max(n_val, 1)

    model.train()
    return {
        "val/action_loss": val_act.item() / denom,
        "val/tactile_loss": val_tac.item() / denom,
    }


def train(args):
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    set_seed(args.seed)

    if accelerator.is_main_process:
        wandb.init(project=args.experiment_name, 
                   name=args.run_name,
                   config=args, 
                   dir=args.log_dir
                )

    accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = args.train_bsz_per_gpu
    accelerator.state.deepspeed_plugin.deepspeed_config["train_batch_size"] = (args.train_bsz_per_gpu * dist.get_world_size() * accelerator.gradient_accumulation_steps)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    tac_isize = args.tactile_intermediate_size if args.tactile_intermediate_size > 0 else None

    if args.revo_training_stage != "off" and args.revo_training_stage != "w0":
        artifact_expected = {
            "tactile_profile": args.tactile_profile,
            "checkpoint_family_id": args.checkpoint_family_id,
            "normalization_family_id": args.normalization_family_id,
            "capability_manifest_sha256": args.capability_manifest_sha256,
            "split_manifest_sha256": args.split_manifest_sha256,
        }
        if args.use_tactile_vqvae:
            validate_revo_vqvae_artifact(
                args.vqvae_ckpt, args.vqvae_artifact, **artifact_expected
            )
        if args.use_tactile_deform:
            validate_revo_deform_artifact(
                args.deform_encoder_ckpt,
                args.deform_encoder_artifact,
                **artifact_expected,
            )

    # Embedded VQ-VAE: tactile codes are encoded on-the-fly inside the model.
    # Enabled either explicitly (--use_tactile_vqvae 1 [--vqvae_ckpt ...]) or
    # AUTOMATICALLY when resuming a checkpoint merged with an embedded VQ-VAE
    # (utils/merge_vqvae_into_ckpt.py writes use_tactile_vqvae=1 + vqvae_config
    # into training_args.json; the weights live in that model.pt).  The config
    # may come from --vqvae_ckpt (which also supplies weights) or from the
    # resume checkpoint's training_args.json (weights then come from model.pt).
    vqvae_blob = None
    vqvae_config = None
    if args.vqvae_ckpt and os.path.exists(args.vqvae_ckpt):
        vqvae_blob = torch.load(args.vqvae_ckpt, map_location="cpu", weights_only=False)
        vqvae_config = vqvae_blob["config"]

    resume_ta = {}
    if args.resume_checkpoint and os.path.isdir(args.resume_checkpoint):
        _ta_path = os.path.join(args.resume_checkpoint, "training_args.json")
        if os.path.exists(_ta_path):
            with open(_ta_path) as f:
                resume_ta = json.load(f)
    if resume_ta.get("use_tactile_vqvae") and not bool(args.use_tactile_vqvae):
        args.use_tactile_vqvae = 1
        accelerator.print("[vqvae] embedded VQ-VAE auto-detected from resume checkpoint "
                          "(codes encoded on the fly; weights from model.pt).")
    if bool(args.use_tactile_vqvae) and vqvae_config is None:
        vqvae_config = resume_ta.get("vqvae_config")

    if bool(args.use_tactile_vqvae):
        if vqvae_config is None:
            raise FileNotFoundError(
                "--use_tactile_vqvae 1 needs either --vqvae_ckpt or a resume checkpoint "
                "whose training_args.json carries vqvae_config.")
        args.vqvae_codebook_size = int(vqvae_config["codebook_size"])
        args.vqvae_window = int(vqvae_config["window"])
        args.vqvae_config_dict = vqvae_config   # persisted into training_args.json
        args.use_tactile_code = 1   # the code embedder is still needed

    model = Qwen3VLVLAModel.from_pretrained_qwen3vl(
        pretrained_path=args.model_path,
        action_dim=args.action_dim,
        action_chunk=args.action_chunk,
        tactile_num_fingers=args.tactile_num_fingers,
        use_tactile_deform=bool(args.use_tactile_deform),
        use_robot_state=bool(args.use_robot_state),
        torch_dtype=torch.bfloat16,
        tactile_intermediate_size=tac_isize,
        n_flare_tokens_per_frame=args.n_flare_tokens_per_frame if args.use_flare else 0,
        n_flare_steps=args.n_flare_steps if args.use_flare else 0,
        flare_layer_index=args.flare_layer_index,
        use_tactile_code=bool(args.use_tactile_code),
        vqvae_codebook_size=args.vqvae_codebook_size,
        use_tactile_vqvae=bool(args.use_tactile_vqvae),
        vqvae_config=vqvae_config,
    )
    if args.use_tactile_deform:
        model.load_deform_encoder_weights(args.deform_encoder_ckpt)
    model.initialize_vla_weights()
    if vqvae_blob is not None:
        model.load_tactile_vqvae(vqvae_blob)
    if args.use_flare:
        accelerator.print(
            f"Flare alignment: {args.n_flare_steps} steps × {args.n_flare_tokens_per_frame} tok/frame "
            f"= {model.n_flare_tokens} total tokens, layer_index={args.flare_layer_index}"
        )

    is_stage1 = (args.training_stage == 1)
    has_any_tactile = bool(args.use_tactile_vec or args.use_tactile_deform)
    freeze_tactile = not has_any_tactile

    if not args.resume_checkpoint:
        named_params = dict(model.named_parameters())
        for name, param in model.named_parameters():
            if "_action" in name:
                base = name.replace("_action", "")
                if base in named_params:
                    param.data.copy_(named_params[base].data)
        accelerator.print("Action expert initialized from latent expert.")

    if args.resume_checkpoint:
        ckpt_path = args.resume_checkpoint
        if os.path.isdir(ckpt_path):
            ckpt_path = os.path.join(ckpt_path, "model.pt")
        resume_sd = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in resume_sd:
            resume_sd = resume_sd["state_dict"]
        if args.revo_training_stage != "off":
            report = load_revo_compatible_state_dict(
                model,
                resume_sd,
                migration_kind=args.resume_source,
                raise_on_error=True,
            )
            accelerator.print(
                "Revo checkpoint migration: "
                f"loaded={len(report.loaded)}, "
                f"declared_reinitialized={len(report.reinitialized_dimension_bound) + len(report.declared_missing_reinitialized)}, "
                "unexpected_missing=0, unexpected_mismatch=0, unexpected_source=0"
            )
            if accelerator.is_main_process:
                os.makedirs(args.output_dir, exist_ok=True)
                with open(
                    os.path.join(args.output_dir, "checkpoint_migration_report.json"),
                    "w",
                    encoding="utf-8",
                ) as handle:
                    json.dump(
                        {
                            "schema_version": "revo3-checkpoint-migration-v1",
                            "migration_kind": args.resume_source,
                            "loaded": list(report.loaded),
                            "reinitialized_dimension_bound": list(
                                report.reinitialized_dimension_bound
                            ),
                            "declared_missing_reinitialized": list(
                                report.declared_missing_reinitialized
                            ),
                            "unexpected_shape_mismatch": list(
                                report.unexpected_shape_mismatch
                            ),
                            "unexpected_source": list(report.unexpected_source),
                            "unexpected_missing_target": list(
                                report.unexpected_missing_target
                            ),
                        },
                        handle,
                        indent=2,
                    )
        else:
            missing, unexpected = model.load_state_dict(resume_sd, strict=False)
            accelerator.print(f"Legacy resume: missing={len(missing)}, unexpected={len(unexpected)}")

    resumed_tactile = bool(args.resume_checkpoint) and args.resume_source in {
        "revo_w1", "revo_midtrain", "revo_sft", "official_midtrain_ablation"
    }
    if not freeze_tactile and not resumed_tactile:
        # Tactile expert is a velocity predictor under cascaded flow matching,
        # so the head must start non-trivial (no final-layer zero-init here).
        for name, param in model.named_parameters():
            if "_tactile" in name:
                if param.ndim >= 2:
                    nn.init.xavier_uniform_(param)
                elif param.ndim == 1:
                    nn.init.zeros_(param)
        accelerator.print("Tactile expert re-initialized (resume_source=pretrain or no resume).")
    elif resumed_tactile:
        accelerator.print("Tactile expert weights kept from resumed midtrain checkpoint.")

    revo_named_groups = None
    if args.revo_training_stage != "off":
        spec = revo_stage_spec(args.revo_training_stage)
        reviewed = {
            "max_steps": (args.max_steps, spec.max_steps),
            "n_epochs": (args.n_epochs, spec.max_epochs),
            "cascaded_tactile_dropout": (
                args.cascaded_tactile_dropout,
                spec.tactile_dropout,
            ),
            "state_dropout": (args.state_dropout, spec.state_dropout),
            "flare_loss_weight": (args.flare_loss_weight, spec.flare_loss_weight),
        }
        mismatches = {
            name: (expected, observed)
            for name, (observed, expected) in reviewed.items()
            if not math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=1e-12)
        }
        if bool(args.use_flare) != spec.flare_enabled:
            mismatches["use_flare"] = (spec.flare_enabled, bool(args.use_flare))
        if mismatches:
            raise ValueError(f"Revo stage contract mismatch: {mismatches}")
        actual_effective_batch = (
            int(args.train_bsz_per_gpu)
            * int(dist.get_world_size())
            * int(accelerator.gradient_accumulation_steps)
        )
        if actual_effective_batch != spec.effective_batch:
            raise ValueError(
                f"Revo effective batch={actual_effective_batch}; expected {spec.effective_batch}"
            )
        revo_named_groups = configure_revo_trainable_parameters(
            model, args.revo_training_stage
        )
    else:
        for name, param in model.named_parameters():
            if (name.startswith("visual") or name.startswith("deform_encoder")
                    or name.startswith("tactile_vqvae")):
                param.requires_grad = False   # embedded VQ-VAE stays frozen
            elif "_tactile" in name or "final_layer_tactile" in name:
                param.requires_grad = (not freeze_tactile)
            else:
                param.requires_grad = True

    # Keep the embedded VQ-VAE in eval + fp32 so on-the-fly codes match the
    # standalone tokenizer (no EMA drift, full-precision normalization).
    if getattr(model, "tactile_vqvae", None) is not None:
        model.tactile_vqvae.float().eval()
        for _b in ("tacf6_vqvae_min", "tacf6_vqvae_max"):
            setattr(model, _b, getattr(model, _b).float())

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(f"Model: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable")

    if revo_named_groups is not None:
        param_groups = build_revo_optimizer_groups(
            revo_named_groups,
            args.revo_training_stage,
            weight_decay=args.weight_decay,
        )
    else:
        no_decay = ["bias", "norm.weight", "q_norm.weight", "k_norm.weight"]
        param_groups = [
            {"params": [p for n, p in model.named_parameters()
                        if p.requires_grad and not any(nd in n for nd in no_decay)],
             "weight_decay": args.weight_decay},
            {"params": [p for n, p in model.named_parameters()
                        if p.requires_grad and any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.learning_rate)

    if getattr(args, "data_format", "json") == "lerobot":
        from qwen_vla.lerobot_dataset import TRexLeRobotDataset
        dataset = TRexLeRobotDataset(args, processor, accelerator)
    else:
        dataset = SftDataset(args, processor, accelerator)

    val_dataloader = None
    if getattr(args, "val_data_path", ""):
        val_args = copy.copy(args)
        val_args.data_path = args.val_data_path
        val_args.stats_path = args.val_stats_path or args.stats_path
        val_dataset = SftDataset(val_args, processor, accelerator, training=False)
        val_dataloader = DataLoader(
            val_dataset, batch_size=args.train_bsz_per_gpu, shuffle=False,
            drop_last=False, collate_fn=val_dataset.collate_fn,
            num_workers=2, pin_memory=True)
    elif getattr(args, "val_ratio", 0) > 0:
        if args.revo_training_stage != "off":
            raise ValueError(
                "Revo training forbids post-stat random validation splits; pass an explicit "
                "manifest-derived --val_data_path development split"
            )
        val_dataset = dataset.create_val_split(
            val_ratio=args.val_ratio, seed=args.seed)
        val_dataloader = DataLoader(
            val_dataset, batch_size=args.train_bsz_per_gpu, shuffle=False,
            drop_last=True, collate_fn=val_dataset.collate_fn,
            num_workers=2, pin_memory=True)

    dataloader = DataLoader(
        dataset, batch_size=args.train_bsz_per_gpu, shuffle=True,
        collate_fn=dataset.collate_fn, num_workers=4, pin_memory=True,
    )

    num_training_steps = (
        int(len(dataloader) * args.n_epochs)
        // accelerator.gradient_accumulation_steps
        // dist.get_world_size()
    )
    if args.max_steps > 0:
        num_training_steps = min(num_training_steps, int(args.max_steps))
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_rates * num_training_steps),
        num_training_steps=num_training_steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    if val_dataloader is not None:
        model, optimizer, dataloader, val_dataloader = accelerator.prepare(
            model, optimizer, dataloader, val_dataloader)
    else:
        model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    metric = TrainingMetrics(device=torch.cuda.current_device())
    global_step = 0
    optimizer_step = 0
    T_per_frame = args.n_flare_tokens_per_frame
    S_steps = args.n_flare_steps
    K = T_per_frame * S_steps  # total flare tokens
    use_flare = bool(args.use_flare and K > 0)
    flare_layer_idx = args.flare_layer_index
    model.train()

    for epoch in range(args.n_epochs):
        from tqdm import tqdm
        it = (tqdm(dataloader, total=len(dataloader))
              if accelerator.is_main_process else dataloader)

        for batch in it:
            if args.max_steps > 0 and optimizer_step >= args.max_steps:
                break
            raw_model = accelerator.unwrap_model(model)

            inputs_embeds = raw_model.prepare_inputs_embeds(
                input_ids=batch["input_ids"],
                pixel_values=batch.get("pixel_values"),
                image_grid_thw=batch.get("image_grid_thw"),
            )

            n_slow_imgs = batch["n_slow_images"]
            grid_thw = batch.get("image_grid_thw")
            B = inputs_embeds.shape[0]
            if grid_thw is not None and grid_thw.shape[0] > n_slow_imgs:
                merge = getattr(raw_model.visual, "spatial_merge_size",
                                getattr(processor.image_processor, "merge_size", 2))
                n_imgs_per_sample = grid_thw.shape[0] // B
                n_slow_img_tokens = sum(
                    int(g[0] * (g[1] // merge) * (g[2] // merge))
                    for g in grid_thw[:n_imgs_per_sample][:n_slow_imgs]
                )
                slow_embeds, fast_embeds = split_slow_fast_embeds(
                    inputs_embeds, batch["input_ids"],
                    raw_model.image_token_id, n_slow_img_tokens)
            else:
                slow_embeds = inputs_embeds
                fast_embeds = inputs_embeds[:, :0]

            L_slow = slow_embeds.shape[1]
            
            torch.set_printoptions(profile="full")
            # print(batch["input_ids"])
            # print("---------------------")
            # print(inputs_embeds.shape)
            # print(slow_embeds.shape)
            # print(fast_embeds.shape)
            # input("pause")

            pos_ids, _ = raw_model.get_rope_index(
                input_ids=batch["input_ids"],
                image_grid_thw=batch.get("image_grid_thw"),
                attention_mask=batch["attention_mask"],
            )
            pos_ids = pos_ids[:, :, :L_slow]

            if use_flare:
                flare_q = raw_model.flare_queries.expand(B, -1, -1).to(device=slow_embeds.device, dtype=slow_embeds.dtype)
                slow_embeds_ext = torch.cat([slow_embeds, flare_q], dim=1)
                pos_ids = extend_position_ids_for_flare(pos_ids, K)
                L_latent = slow_embeds_ext.shape[1]  # L_slow + K
            else:
                slow_embeds_ext = slow_embeds
                L_latent = L_slow

            if args.use_robot_state and batch["state_raw"] is not None:
                state_vec = batch["state_raw"].to(slow_embeds.device,
                                                   dtype=slow_embeds.dtype)
                state_embeds = raw_model.state_embedder(state_vec).unsqueeze(1)
            else:
                state_embeds = torch.empty((B, 0, slow_embeds.shape[2]), device=slow_embeds.device, dtype=slow_embeds.dtype)
            n_state = state_embeds.shape[1]

            noisy_actions = raw_model.x_embedder(
                batch["noisy_actions"].to(slow_embeds.dtype))
            timesteps = raw_model.t_embedder(
                batch["timesteps"].to(slow_embeds.dtype)).unsqueeze(1)

            chunk = args.action_chunk
            target = batch["target"].to(slow_embeds.dtype)
            n_fast = fast_embeds.shape[1]
            r_target_norm = None  # set only when tactile expert ran this step

            has_any_tac = bool(args.use_tactile_vec or args.use_tactile_deform)

            # Cascaded: shared flow split between action + tactile experts.
            # 1) L_flow — action expert single-step (tactile-blind), trained on
            #    full τ ∈ [0, 1] so the action expert can run standalone for
            #    graceful degradation when tactile is missing.
            full_embeds = torch.cat([
                slow_embeds_ext,
                fast_embeds, state_embeds, timesteps, noisy_actions,
            ], dim=1)
            L_total = full_embeds.shape[1]
            outputs = model.model(
                inputs_embeds=full_embeds,
                position_ids=pos_ids,
                attention_mask=batch["attention_mask"],
                use_cache=False,
                output_hidden_states=use_flare,
                latent_indexes=torch.arange(0, L_latent, device=full_embeds.device),
                action_indexes=torch.arange(L_latent, L_total, device=full_embeds.device),
                tactile_indexes=torch.arange(0, 0, device=full_embeds.device),
            )
            hidden = outputs.last_hidden_state
            act_pred_start = L_latent + n_fast + n_state + 1
            v_act = raw_model.final_layer(
                hidden[:, act_pred_start: act_pred_start + chunk, :])
            loss_act = nn.MSELoss()(v_act, target)

            if has_any_tac and not is_stage1:
                fe = fast_embeds if n_fast > 0 else None
                se = state_embeds if n_state > 0 else None
                # cached_kv must summarize the action expert's state at
                # τ=τ_split — exactly what the tactile expert attends to at
                # inference.  forward_flow_action_partial caches τ=τ_split
                # keys (refresh_clean_kv=True writes the partially-denoised
                # x_split into the KV).
                ahat_noise = torch.randn_like(batch["noisy_actions"])
                with torch.no_grad():
                    _, cached_kv, n_action_in_cache, _ = (
                        raw_model.forward_flow_action_partial(
                            inputs_embeds=slow_embeds_ext,
                            position_ids=pos_ids,
                            noise=ahat_noise,
                            attention_mask=batch["attention_mask"],
                            state_embeds=se,
                            fast_embeds=fe,
                            num_steps_total=args.cascaded_total_steps,
                            split_step=args.cascaded_split_step,
                            refresh_clean_kv=True,
                        ))

                # 2) L_flow_tactile — tactile expert predicts velocity at
                #    τ ∈ [0, τ_split] for the SAME flow distribution as the
                #    action expert (no residual).  Same target ε − A_demo.
                norm_actions_gt = batch["norm_actions"].to(slow_embeds.dtype)
                eps_r    = batch["eps_r"].to(slow_embeds.dtype)
                tau_split = 1.0 - args.cascaded_split_step / args.cascaded_total_steps
                # Reuse existing time_r ∈ (0, 1] (Beta-sampled) and rescale to (0, τ_split].
                tau_t    = batch["time_r"].to(slow_embeds.dtype) * tau_split
                tau_t_b  = tau_t[:, None, None]
                x_tau    = (1 - tau_t_b) * norm_actions_gt + tau_t_b * eps_r
                v_target_r = eps_r - norm_actions_gt
                # Diagnostic: magnitude of the velocity target the tactile
                # expert is being asked to predict.
                r_target_norm = v_target_r.float().abs().mean().detach()

                # Tactile dropout — zero the tactile signal (not the tensor
                # shape) so the tactile expert learns "all-zero tactile" as
                # a graceful-degradation fallback.  Nulling the tensors
                # would break _embed_tactile_observations, which requires
                # at least one of {codes, f6, deform} to be present.
                drop = (args.cascaded_tactile_dropout > 0
                        and torch.rand((), device=full_embeds.device).item()
                        < args.cascaded_tactile_dropout)
                tac_f6_in    = batch.get("tactile_f6s_delayed")
                tac_def_in   = batch.get("tactile_deforms_delayed")
                tac_codes_in = batch.get("tactile_codes")
                tac_hist_in  = batch.get("tactile_f6_history")
                if drop:
                    if tac_f6_in is not None:
                        tac_f6_in = torch.zeros_like(tac_f6_in)
                    if tac_def_in is not None:
                        tac_def_in = torch.zeros_like(tac_def_in)
                    if tac_codes_in is not None:
                        tac_codes_in = torch.zeros_like(tac_codes_in)
                    if tac_hist_in is not None:
                        # Zero raw F6 → a consistent "no-contact" code on encode.
                        tac_hist_in = torch.zeros_like(tac_hist_in)

                loss_tac = nn.MSELoss()(
                    raw_model.tactile_flow_train_step(
                        cached_kv=cached_kv,
                        latent_position_ids=pos_ids,
                        n_action_in_cache=n_action_in_cache,
                        x_tau=x_tau,
                        tau=tau_t,
                        tactile_f6=tac_f6_in,
                        tactile_deform=tac_def_in,
                        tactile_codes=tac_codes_in,
                        tactile_f6_history=tac_hist_in,
                    ),
                    v_target_r,
                )
            else:
                loss_tac = 0.0

            loss_flare = 0.0
            if use_flare and batch["flare_pixel_values"] is not None:
                # Extract flare hidden states from intermediate layer or last layer
                if flare_layer_idx == -1:
                    flare_source = hidden  # last layer (already normed)
                else:
                    # outputs.hidden_states is a tuple: (embed, layer0, layer1, ..., layerN)
                    # flare_layer_idx can be negative (e.g. -7 for ~3/4 depth)
                    all_hs = outputs.hidden_states
                    # Skip index 0 (embedding), so layer indices are 1..N
                    n_layers = len(all_hs) - 1  # exclude embedding
                    if flare_layer_idx < 0:
                        layer_i = n_layers + flare_layer_idx  # e.g. 28 + (-7) = 21
                    else:
                        layer_i = flare_layer_idx
                    layer_i = max(0, min(layer_i, n_layers - 1))
                    flare_source = all_hs[layer_i + 1]  # +1 to skip embedding

                flare_hidden = flare_source[:, L_slow: L_slow + K, :]  # [B, K, H]
                flare_pred = raw_model.flare_proj(flare_hidden)  # [B, K, H]

                f_pv = batch["flare_pixel_values"].to(device=flare_pred.device, dtype=flare_pred.dtype)
                f_thw = batch["flare_grid_thw"].to(device=flare_pred.device)

                with torch.no_grad():
                    vit_out = raw_model.visual(f_pv, grid_thw=f_thw)
                    features = vit_out[0] if isinstance(vit_out, (tuple, list)) else vit_out

                    # Adaptive pool each frame to T_per_frame tokens
                    merge = getattr(raw_model.visual, "spatial_merge_size", 2)
                    frame_feats = []
                    offset = 0
                    for g in f_thw:
                        n_tok = int(g[0] * (g[1] // merge) * (g[2] // merge))
                        frame_tokens = features[offset: offset + n_tok]  # [n_tok, H]
                        # Adaptive average pool: [n_tok, H] → [T_per_frame, H]
                        pooled = F.adaptive_avg_pool1d(
                            frame_tokens.unsqueeze(0).permute(0, 2, 1),  # [1, H, n_tok]
                            T_per_frame,
                        ).permute(0, 2, 1).squeeze(0)  # [T_per_frame, H]
                        frame_feats.append(pooled)
                        offset += n_tok
                    # frame_feats: list of B*S_steps tensors, each [T_per_frame, H]
                    # Reshape to [B, S_steps * T_per_frame, H] = [B, K, H]
                    flare_targets = torch.stack(frame_feats).view(B, K, -1)

                # Cosine similarity loss
                pred_norm = F.normalize(flare_pred, dim=-1)
                tgt_norm = F.normalize(flare_targets.detach(), dim=-1)
                loss_flare = (1.0 - (pred_norm * tgt_norm).sum(dim=-1)).mean()

            loss = loss_act
            if torch.is_tensor(loss_tac):
                loss = loss + args.cascaded_loss_weight * loss_tac
            if torch.is_tensor(loss_flare):
                loss = loss + args.flare_loss_weight * loss_flare

            metric.update(loss, loss_act, loss_tac, loss_flare)
            accelerator.backward(loss)

            if (global_step + 1) % accelerator.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                optimizer_step += 1

                m = metric.get_metric()
                if accelerator.is_main_process:
                    lr_now = lr_scheduler.get_last_lr()[0]
                    if hasattr(it, "set_postfix"):
                        it.set_postfix(epoch=epoch, step=global_step,
                                       loss=f"{m['total_loss']:.6f}",
                                       act=f"{m['action_loss']:.6f}",
                                       tac=f"{m['tactile_loss']:.6f}",
                                       fut=f"{m['flare_loss']:.6f}",
                                       lr=f"{lr_now:.2e}")
                    log_dict = {
                        "total_loss":   m["total_loss"],
                        "action_loss":  m["action_loss"],
                        "tactile_loss": m["tactile_loss"],
                        "flare_loss":  m["flare_loss"],
                        "lr": lr_now,
                    }
                    if r_target_norm is not None:
                        log_dict["refine/r_target_norm"] = float(r_target_norm)
                    wandb.log(log_dict, step=global_step)

            # ── Validation ──
            if (val_dataloader is not None
                    and getattr(args, "val_freq", 0) > 0
                    and (global_step + 1) % args.val_freq == 0):
                val_m = run_validation(
                    model, val_dataloader, accelerator, args,
                    is_stage1, use_flare, K, T_per_frame, flare_layer_idx)
                if accelerator.is_main_process:
                    accelerator.print(
                        f"  [Val step={global_step}] "
                        f"act={val_m['val/action_loss']:.6f} "
                        f"tac={val_m['val/tactile_loss']:.6f}")
                    wandb.log(val_m, step=global_step)

            global_step += 1

        if (epoch + 1) % args.save_freq == 0 or epoch == args.n_epochs - 1:
            accelerator.wait_for_everyone()
            save_checkpoint(model, processor, accelerator, args,
                            epoch, global_step, dataset.stats_data)
        if args.max_steps > 0 and optimizer_step >= args.max_steps:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_name", type=str, default="qwen3vl_mot_flare")
    parser.add_argument("--run_name", type=str, default="run_1")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--val_data_path", type=str, default="")
    parser.add_argument("--stats_path", type=str, default="")
    parser.add_argument("--val_stats_path", type=str, default="")
    parser.add_argument("--data_root", type=str, default="")
    # Data source: legacy JSON (default) or a LeRobot v3.0 dataset directory.
    parser.add_argument("--data_format", type=str, default="json", choices=["json", "lerobot"])
    parser.add_argument("--lerobot_root", type=str, default="",
                        help="LeRobot dataset dir (required when --data_format lerobot).")
    parser.add_argument("--lerobot_repo_id", type=str, default="",
                        help="Optional repo_id; defaults to basename(lerobot_root).")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--max_ckpts", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=0,
                        help="Maximum optimizer steps; 0 uses the epoch-derived limit")

    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--save_freq", type=int, default=50)
    parser.add_argument("--train_bsz_per_gpu", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.0)
    parser.add_argument("--warmup_rates", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--action_dim", type=int, default=31)
    parser.add_argument("--action_chunk", type=int, default=8)
    parser.add_argument(
        "--tactile_num_fingers", type=int, default=10,
        help="Number of tactile fingertip feature streams. Upstream T-Rex=10; Revo3 single hand=5.",
    )
    parser.add_argument("--use_robot_state", type=int, default=0)
    parser.add_argument("--use_tactile_vec", type=int, default=0)
    parser.add_argument("--use_tactile_deform", type=int, default=1)
    parser.add_argument("--deform_encoder_ckpt", type=str, default="")
    parser.add_argument("--deform_encoder_artifact", type=str, default="")
    parser.add_argument("--tactile_intermediate_size", type=int, default=0)
    parser.add_argument("--training_stage", type=int, default=2, choices=[1, 2])
    parser.add_argument(
        "--revo_training_stage",
        type=str,
        default="off",
        choices=["off", "w0", "w1", "midtrain", "sft"],
        help="Apply the frozen Revo parameter/optimizer-group policy",
    )
    parser.add_argument("--tactile_profile", type=str, default="")
    parser.add_argument("--checkpoint_family_id", type=str, default="")
    parser.add_argument("--normalization_family_id", type=str, default="")
    parser.add_argument("--source_checkpoint_sha256", type=str, default="")
    parser.add_argument("--split_manifest_sha256", type=str, default="")
    parser.add_argument("--tactile_profile_manifest_sha256", type=str, default="")
    parser.add_argument("--normalization_statistics_sha256", type=str, default="")
    parser.add_argument("--normalization_artifact_sha256", type=str, default="")
    parser.add_argument("--capability_manifest_sha256", type=str, default="")
    parser.add_argument("--resume_checkpoint", type=str, default="")
    parser.add_argument(
        "--resume_source",
        type=str,
        default="official_pretrain",
        choices=[
            "official_pretrain",
            "official_midtrain_ablation",
            "revo_w0",
            "revo_w1",
            "revo_midtrain",
            "revo_sft",
        ],
        help="Explicit checkpoint lineage; only declared adjacent-stage rebuilds are allowed.",
    )
    parser.add_argument("--tactile_loss_weight", type=float, default=1.0)
    parser.add_argument("--state_dropout", type=float, default=0.0)
    parser.add_argument("--tracking_error_clip_rad", type=float, default=None)
    parser.add_argument("--image_size", type=int, nargs=2, default=None, metavar=("W", "H"))
    parser.add_argument("--revo_image_augmentation", type=int, default=0)
    parser.add_argument("--revo_tactile_noise_augmentation", type=int, default=0)
    parser.add_argument("--augmentation_seed", type=int, default=42)

    # Cascaded flow matching — action expert handles τ ∈ [τ_split, 1], tactile
    # expert handles τ ∈ [0, τ_split].  Both predict velocity for the same
    # action distribution (NOT a residual).  At inference, the tactile expert
    # continues the flow from the action expert's intermediate state x_split.
    parser.add_argument("--cascaded_total_steps", type=int, default=10,
                        help="Total Euler steps of the cascaded flow at "
                             "inference (= same dt at training).")
    parser.add_argument("--cascaded_split_step", type=int, default=6,
                        help="How many of total_steps the action expert handles "
                             "(τ ∈ [τ_split, 1]); tactile handles the rest.")
    parser.add_argument("--cascaded_tactile_dropout", type=float, default=0.1,
                        help="Per-batch probability of zeroing tactile inputs "
                             "during cascaded training.  Teaches the tactile "
                             "expert to fall back to action-expert-like "
                             "behavior when tactile is missing.")
    parser.add_argument("--cascaded_loss_weight", type=float, default=1.0,
                        help="Weight on the L_flow_tactile loss term.")

    # VQ-VAE tactile code tokens (fast-path only; pre-baked into the JSON via
    # utils/encode_vqvae_codes_to_json.py).  When 0 (default), no tactile_code
    # embedder is created and the model graph is identical to the pre-feature
    # version — flip the flag to revert.
    parser.add_argument("--use_tactile_code", type=int, default=0,
                        help="1: read tactile_codes [B, 2] from JSON and add 2 "
                             "code tokens to the tactile expert's observation "
                             "in the fast/residual path.  Slow path unchanged.")
    parser.add_argument("--vqvae_codebook_size", type=int, default=64,
                        help="Codebook size of the VQ-VAE that produced tactile_codes.")
    # Embedded (on-the-fly) VQ-VAE.  When 1, the model carries the VQ-VAE and
    # encodes a raw F6 history window internally each step — no pre-baked
    # tactile_codes in the JSON.  --vqvae_ckpt supplies the frozen weights +
    # F6 stats (or resume from a checkpoint already merged via
    # scripts/merge_vqvae_into_ckpt.py).
    parser.add_argument("--use_tactile_vqvae", type=int, default=0,
                        help="1: embed the VQ-VAE in the model and tokenize raw "
                             "F6 history on-the-fly (implies use_tactile_code).")
    parser.add_argument("--vqvae_ckpt", type=str, default="",
                        help="Standalone VQ-VAE checkpoint (config/model_state/stats) "
                             "to load into the embedded module.")
    parser.add_argument("--vqvae_artifact", type=str, default="")

    # Flare
    parser.add_argument("--use_flare", type=int, default=1, help="Enable flare visual prediction for latent expert.")
    parser.add_argument("--n_flare_tokens_per_frame", type=int, default=4, help="Number of tokens per future frame.")
    parser.add_argument("--n_flare_steps", type=int, default=8, help="Number of future steps to predict.")
    parser.add_argument("--flare_loss_weight", type=float, default=0.5, help="Weight for flare prediction cosine loss.")
    parser.add_argument("--flare_frame_stride", type=int, default=2, help="Temporal stride for flare frame targets.")
    parser.add_argument("--flare_layer_index", type=int, default=-1, help="Layer to extract flare hidden states from (-1=last, e.g. -7 for ~3/4 depth).")

    # Validation
    parser.add_argument("--val_ratio", type=float, default=0.0, help="Fraction of samples for validation (0=disable)")
    parser.add_argument("--val_freq", type=int, default=0, help="Run validation every N steps (0=disable)")
    parser.add_argument("--max_val_batches", type=int, default=50, help="Max batches per validation run")

    args = parser.parse_args()

    if args.data_format == "lerobot":
        if not args.lerobot_root:
            parser.error("--data_format lerobot requires --lerobot_root")
    elif not args.data_path:
        parser.error("--data_format json requires --data_path")

    args.log_dir = os.path.join(args.log_dir, args.experiment_name)
    args.output_dir = os.path.join(args.output_dir, args.experiment_name, args.run_name)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    train(args)

