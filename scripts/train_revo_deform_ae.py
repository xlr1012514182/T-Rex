#!/usr/bin/env python3
"""Train the T-Rex DeformEncoder/Decoder from scratch on Revo DIFF images.

Sharpa weights are deliberately not accepted. The split manifest is loaded
before any episode, locked-test is never opened, and the emitted artifact is
bound to one tactile capability/checkpoint family.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from qwen_vla.DeformAE import DeformAEInfer
from revo3_v1.data import CorpusSplit, RevoCorpusSplitManifest, RevoEpisode


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RevoDiffDataset(Dataset):
    def __init__(self, episodes: list[RevoEpisode]) -> None:
        self.episodes = episodes
        self.index = [
            (episode_index, frame, finger)
            for episode_index, episode in enumerate(episodes)
            for frame in range(episode.num_frames)
            for finger in range(5)
        ]
        if not self.index:
            raise ValueError("selected split contains no Revo DIFF frames")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> torch.Tensor:
        episode_index, frame, finger = self.index[index]
        value = self.episodes[episode_index].tactile_diff
        if value is None:
            raise RuntimeError("Profile A/B episode lost its required DIFF tensor")
        image = np.asarray(value[frame, finger], dtype=np.float32)
        if float(image.max(initial=0.0)) > 1.0:
            image = image / 255.0
        return torch.from_numpy(np.clip(image, 0.0, 1.0)).unsqueeze(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tactile-profile",
        choices=("profile_a_force6d_diff", "profile_b_diff_only"),
        required=True,
    )
    parser.add_argument("--checkpoint-family-id", required=True)
    parser.add_argument("--normalization-family-id", required=True)
    parser.add_argument("--capability-manifest-sha256", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--global-effective-batch", type=int, default=256)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load_split(
    root: Path,
    manifest: RevoCorpusSplitManifest,
    splits: tuple[CorpusSplit, ...],
    args: argparse.Namespace,
) -> list[RevoEpisode]:
    episodes = [RevoEpisode.load(root / item) for item in manifest.episode_ids(splits)]
    if not episodes:
        raise ValueError(f"no episodes in splits {[item.value for item in splits]}")
    for episode in episodes:
        expected = (
            args.tactile_profile,
            args.checkpoint_family_id,
            args.normalization_family_id,
            args.capability_manifest_sha256,
        )
        observed = (
            episode.meta.tactile_profile,
            episode.meta.checkpoint_family_id,
            episode.meta.normalization_family_id,
            episode.meta.capability_manifest_sha256,
        )
        if observed != expected:
            raise ValueError(f"DIFF corpus crosses capability/checkpoint families: {observed}")
    return episodes


def load_revo_deform_splits(
    args: argparse.Namespace,
) -> tuple[RevoCorpusSplitManifest, list[RevoEpisode], list[RevoEpisode]]:
    """Open MIDTRAIN_TRAIN for fitting and DEVELOPMENT for selection only."""
    manifest = RevoCorpusSplitManifest.load(args.split_manifest)
    manifest.assert_matches_episode_root(args.data_root)
    train_episodes = _load_split(
        args.data_root, manifest, (CorpusSplit.MIDTRAIN_TRAIN,), args
    )
    dev_episodes = _load_split(
        args.data_root, manifest, (CorpusSplit.DEVELOPMENT,), args
    )
    return manifest, train_episodes, dev_episodes


def _episode_ids_sha256(episodes: list[RevoEpisode]) -> str:
    payload = json.dumps(
        [episode.meta.episode_id for episode in episodes],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    args = parse_args()
    if args.epochs != 30:
        raise ValueError("reviewed Revo DIFF training fixes epochs=30")
    if args.batch_size * args.gradient_accumulation_steps != args.global_effective_batch:
        raise ValueError("batch_size * gradient_accumulation_steps must equal 256")
    if len(args.capability_manifest_sha256) != 64:
        raise ValueError("capability manifest SHA-256 must contain 64 characters")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    manifest, train_episodes, dev_episodes = load_revo_deform_splits(args)
    train_episode_ids = [episode.meta.episode_id for episode in train_episodes]
    dev_episode_ids = [episode.meta.episode_id for episode in dev_episodes]
    train_loader = DataLoader(
        RevoDiffDataset(train_episodes),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    dev_loader = DataLoader(
        RevoDiffDataset(dev_episodes),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeformAEInfer().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    best = float("inf")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        model.train()
        for step, image in enumerate(train_loader):
            image = image.to(device)
            prediction = model(image)
            loss = F.mse_loss(prediction, image) / args.gradient_accumulation_steps
            loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for image in dev_loader:
                image = image.to(device)
                total += float(F.mse_loss(model(image), image, reduction="sum").item())
                count += int(image.numel())
        validation_mse = total / max(count, 1)
        if validation_mse < best:
            best = validation_mse
            checkpoint = {
                "state_dict": model.state_dict(),
                "encoder_state": model.encoder.state_dict(),
                "config": {
                    "input_shape": [1, 240, 240],
                    "architecture": "qwen_vla.DeformAE.DeformAEInfer",
                    "loss": "mse",
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "global_effective_batch": args.global_effective_batch,
                },
                "provenance": {
                    "trained_from_scratch": True,
                    "source_sensor_family": "revo3_visiontouch_diff",
                    "tactile_profile": args.tactile_profile,
                    "checkpoint_family_id": args.checkpoint_family_id,
                    "normalization_family_id": args.normalization_family_id,
                    "capability_manifest_sha256": args.capability_manifest_sha256,
                    "split_manifest_sha256": sha256_file(args.split_manifest),
                    "source_splits": [CorpusSplit.MIDTRAIN_TRAIN.value],
                    "train_episode_ids": train_episode_ids,
                    "train_episode_ids_sha256": _episode_ids_sha256(train_episodes),
                    "validation_split": CorpusSplit.DEVELOPMENT.value,
                    "validation_episode_ids": dev_episode_ids,
                    "validation_episode_ids_sha256": _episode_ids_sha256(dev_episodes),
                    "locked_test_opened": False,
                },
                "validation_mse": validation_mse,
            }
            torch.save(checkpoint, args.output_dir / "revo_deform_ae_best.pt")
        print(json.dumps({"epoch": epoch + 1, "validation_mse": validation_mse}))
    artifact = {
        "schema_version": "revo3-deform-encoder-artifact-v1",
        "checkpoint": "revo_deform_ae_best.pt",
        "checkpoint_sha256": sha256_file(args.output_dir / "revo_deform_ae_best.pt"),
        "trained_from_scratch": True,
        "source_sensor_family": "revo3_visiontouch_diff",
        "input_shape": [5, 1, 240, 240],
        "num_fingers": 5,
        "encoder_state_complete": True,
        "tactile_profile": args.tactile_profile,
        "checkpoint_family_id": args.checkpoint_family_id,
        "normalization_family_id": args.normalization_family_id,
        "capability_manifest_sha256": args.capability_manifest_sha256,
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "source_splits": [CorpusSplit.MIDTRAIN_TRAIN.value],
        "train_episode_ids": train_episode_ids,
        "train_episode_ids_sha256": _episode_ids_sha256(train_episodes),
        "validation_split": CorpusSplit.DEVELOPMENT.value,
        "validation_episode_ids": dev_episode_ids,
        "validation_episode_ids_sha256": _episode_ids_sha256(dev_episodes),
        "locked_test_opened": False,
        "best_validation_mse": best,
    }
    (args.output_dir / "revo_deform_ae_artifact.json").write_text(
        json.dumps(artifact, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
