#!/usr/bin/env python
"""Train a GNI-style binary EMG classifier on manifest-backed data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from revo3_v1.emg.model import GNIModelConfig
from revo3_v1.emg.train import EMGTrainingConfig, train_binary_emg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preset", choices=("smoke", "gni"), default="smoke")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260817)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import numpy as np

    with np.load(args.dataset / "windows.npz", allow_pickle=False) as archive:
        input_channels = int(archive["signal"].shape[1])
    model_config = (
        GNIModelConfig.smoke(input_channels) if args.preset == "smoke" else GNIModelConfig(input_channels=input_channels)
    )
    training_config = EMGTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
    )
    summary = train_binary_emg(args.dataset, args.output, model_config, training_config)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
