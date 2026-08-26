"""Small dependency-light trainer for GNI-style EMG primitive models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import random
from typing import Any, Dict, Mapping

import numpy as np

from .data import EMGWindowDataset, fit_train_normalization, load_manifest, verify_split_manifests
from .model import GNIClassifier, GNIModelConfig, save_emg_checkpoint
from .migration import GNI_SOURCE_COMMIT, migrate_gni_encoder
from .preprocessing import (
    BRAINCO_EDU_8CH_250HZ,
    EmgPreprocessingProfile,
    normalization_sha256,
    preprocess_emg_windows,
    validate_npz_profile,
)


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


def train_emg(
    dataset_dir: str | Path,
    output_dir: str | Path,
    model_config: GNIModelConfig,
    training_config: EMGTrainingConfig,
    preprocessing_profile: EmgPreprocessingProfile | None = BRAINCO_EDU_8CH_250HZ,
    *,
    allow_unprofiled_fixture: bool = False,
    gni_checkpoint: str | Path | None = None,
    gni_source_commit: str = GNI_SOURCE_COMMIT,
    from_scratch_ablation: bool = False,
    allow_window_reset_fallback: bool = False,
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
        if preprocessing_profile is not None:
            preprocessing_state = validate_npz_profile(archive, preprocessing_profile)
        elif not allow_unprofiled_fixture:
            raise ValueError("Production EMG training requires a preprocessing profile")
        else:
            preprocessing_state = None
    if signals.shape[1] != model_config.input_channels:
        raise ValueError(
            f"Dataset has {signals.shape[1]} channels but model expects {model_config.input_channels}"
        )
    if preprocessing_profile is not None:
        if signals.shape[2] != preprocessing_profile.window_samples:
            raise ValueError(
                f"Dataset has {signals.shape[2]} samples/window but profile expects "
                f"{preprocessing_profile.window_samples}"
            )
        if model_config.input_channels != preprocessing_profile.channel_count:
            raise ValueError("Model input channels do not match preprocessing profile")
        if preprocessing_state is not None and preprocessing_state.preprocessed:
            filtered_signals = signals
            filter_lineage = preprocessing_state.provenance
        elif allow_window_reset_fallback:
            filtered_signals = preprocess_emg_windows(signals, preprocessing_profile)
            filter_lineage = "independent_window_zero_state_fallback"
        else:
            raise ValueError(
                "Mainline EMG training requires session-continuously preprocessed windows; "
                "window-reset filtering is an explicit ablation/fallback"
            )
    else:
        filtered_signals = signals
        filter_lineage = "unprofiled_synthetic_fixture"
    train_rows = load_manifest(manifests["train"])
    train_subject_ids = sorted({str(row["subject_id"]) for row in train_rows})
    train_session_ids = sorted({str(row["session_id"]) for row in train_rows})
    normalization = fit_train_normalization(
        filtered_signals, [int(row["index"]) for row in train_rows]
    )
    bound_profile = (
        None
        if preprocessing_profile is None
        else preprocessing_profile.bind_normalization(normalization)
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "train_normalization.npz", **normalization)

    datasets = {
        name: EMGWindowDataset(
            dataset_root / "windows.npz",
            path,
            normalization,
            channel_rotation=training_config.channel_rotation if name == "train" else 0,
            preprocessing_profile=preprocessing_profile,
            allow_unprofiled_fixture=allow_unprofiled_fixture,
            preprocessed_signals=filtered_signals,
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
    model = GNIClassifier(model_config)
    if gni_checkpoint is None and not from_scratch_ablation:
        raise ValueError(
            "Mainline EMG training requires --gni-checkpoint; use the explicit "
            "from_scratch_ablation flag only for an ablation"
        )
    migration_report = None
    if gni_checkpoint is not None:
        if from_scratch_ablation:
            raise ValueError("gni_checkpoint and from_scratch_ablation are mutually exclusive")
        migration_report = migrate_gni_encoder(
            model,
            gni_checkpoint,
            source_commit=gni_source_commit,
            map_location="cpu",
        )
    model.to(device)
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
                    "train_subject_ids": train_subject_ids,
                    "train_session_ids": train_session_ids,
                    "fixture_only": preprocessing_profile is None,
                    "preprocessing_profile_fingerprint": (
                        None if bound_profile is None else bound_profile.fingerprint
                    ),
                    "gni_migration": (
                        None if migration_report is None else dict(migration_report.to_mapping())
                    ),
                    "from_scratch_ablation": bool(from_scratch_ablation),
                    "filter_lineage": filter_lineage,
                },
                preprocessing_profile=bound_profile,
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
        "normalization_sha256": normalization_sha256(normalization),
        "preprocessing_profile": (
            None if bound_profile is None else dict(bound_profile.to_mapping())
        ),
        "verification_scope": (
            "synthetic/unprofiled fixture only"
            if preprocessing_profile is None
            else "profile-bound offline model training; no clinical-performance claim"
        ),
        "gni_migration": (
            None if migration_report is None else dict(migration_report.to_mapping())
        ),
        "from_scratch_ablation": bool(from_scratch_ablation),
        "filter_lineage": filter_lineage,
    }
    with (output / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def train_binary_emg(
    dataset_dir: str | Path,
    output_dir: str | Path,
    model_config: GNIModelConfig,
    training_config: EMGTrainingConfig,
) -> Dict[str, Any]:
    """Compatibility entry point for the synthetic OPEN/CLOSE fixture."""

    binary_config = replace(
        model_config,
        output_channels=2,
        label_names=("OPEN", "CLOSE"),
    )
    # This compatibility path is intentionally limited to generated fixtures.
    # It must never be mistaken for the BrainCo 8ch/250 Hz production domain.
    return train_emg(
        dataset_dir,
        output_dir,
        binary_config,
        training_config,
        preprocessing_profile=None,
        allow_unprofiled_fixture=True,
        from_scratch_ablation=True,
        allow_window_reset_fallback=False,
    )


def main(argv=None) -> int:
    """Train the profile-bound five-class BrainCo adaptation."""

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--gni-checkpoint", type=Path)
    source.add_argument("--from-scratch-ablation", action="store_true")
    parser.add_argument("--gni-source-commit", default=GNI_SOURCE_COMMIT)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-window-reset-fallback", action="store_true")
    args = parser.parse_args(argv)
    summary = train_emg(
        args.dataset,
        args.output,
        GNIModelConfig.brainco_edu(),
        EMGTrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
        ),
        preprocessing_profile=BRAINCO_EDU_8CH_250HZ,
        gni_checkpoint=args.gni_checkpoint,
        gni_source_commit=args.gni_source_commit,
        from_scratch_ablation=args.from_scratch_ablation,
        allow_window_reset_fallback=args.allow_window_reset_fallback,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
