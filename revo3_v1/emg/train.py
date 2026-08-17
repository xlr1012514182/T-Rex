"""Small dependency-light trainer for the binary GNI-style EMG model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any, Dict, Mapping

import numpy as np

from .data import EMGWindowDataset, fit_train_normalization, load_manifest, verify_split_manifests
from .model import GNIBinaryClassifier, GNIModelConfig, save_emg_checkpoint


@dataclass(frozen=True)
class EMGTrainingConfig:
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 0.5
    seed: int = 20260817
    num_workers: int = 0
    device: str = "auto"
    channel_rotation: int = 2


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)


def _evaluate(model, loader, device: str) -> Dict[str, float]:
    import torch

    model.eval()
    correct, count, total_loss = 0, 0, 0.0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for signal, label in loader:
            signal, label = signal.to(device), label.to(device)
            logits = model(signal)
            loss = criterion(logits, label)
            total_loss += float(loss.item()) * int(label.numel())
            correct += int((logits.argmax(dim=-1) == label).sum().item())
            count += int(label.numel())
    return {"loss": total_loss / max(1, count), "accuracy": correct / max(1, count)}


def train_binary_emg(
    dataset_dir: str | Path,
    output_dir: str | Path,
    model_config: GNIModelConfig,
    training_config: EMGTrainingConfig,
) -> Dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    _set_seed(training_config.seed)
    dataset_root = Path(dataset_dir)
    manifests = {
        name: dataset_root / "manifests" / f"{name}.jsonl" for name in ("train", "val", "test")
    }
    verify_split_manifests(manifests)
    with np.load(dataset_root / "windows.npz", allow_pickle=False) as archive:
        signals = np.asarray(archive["signal"], dtype=np.float32)
    if signals.shape[1] != model_config.input_channels:
        raise ValueError(
            f"Dataset has {signals.shape[1]} channels but model expects {model_config.input_channels}"
        )
    train_rows = load_manifest(manifests["train"])
    normalization = fit_train_normalization(signals, [int(row["index"]) for row in train_rows])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "train_normalization.npz", **normalization)

    datasets = {
        name: EMGWindowDataset(
            dataset_root / "windows.npz",
            path,
            normalization,
            channel_rotation=training_config.channel_rotation if name == "train" else 0,
        )
        for name, path in manifests.items()
    }
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=training_config.batch_size,
            shuffle=name == "train",
            num_workers=training_config.num_workers,
            drop_last=False,
        )
        for name, dataset in datasets.items()
    }
    if training_config.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = training_config.device
    model = GNIBinaryClassifier(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training_config.learning_rate, weight_decay=training_config.weight_decay
    )
    criterion = torch.nn.CrossEntropyLoss()
    best_val_accuracy = -1.0
    history = []
    best_path = output / "best.pt"
    for epoch in range(training_config.epochs):
        model.train()
        correct, count, total_loss = 0, 0, 0.0
        for signal, label in loaders["train"]:
            signal, label = signal.to(device), label.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(signal)
            loss = criterion(logits, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.gradient_clip)
            optimizer.step()
            total_loss += float(loss.item()) * int(label.numel())
            correct += int((logits.argmax(dim=-1) == label).sum().item())
            count += int(label.numel())
        train_metrics = {"loss": total_loss / max(1, count), "accuracy": correct / max(1, count)}
        val_metrics = _evaluate(model, loaders["val"], device)
        record = {"epoch": epoch + 1, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            save_emg_checkpoint(
                best_path,
                model,
                normalization,
                {
                    "epoch": epoch + 1,
                    "val_metrics": val_metrics,
                    "training_config": asdict(training_config),
                },
            )

    from .model import load_emg_checkpoint

    best_model, _ = load_emg_checkpoint(best_path, map_location=device)
    best_model.to(device)
    test_metrics = _evaluate(best_model, loaders["test"], device)
    summary: Dict[str, Any] = {
        "schema_version": "revo3-emg-training-summary-v1",
        "best_checkpoint": str(best_path.resolve()),
        "best_val_accuracy": best_val_accuracy,
        "test_metrics": test_metrics,
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "history": history,
        "normalization_source": "train split only",
    }
    with (output / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary
