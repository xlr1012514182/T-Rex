#!/usr/bin/env python
"""Train the five-class BrainCo EMG model (binary only as an explicit fixture)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from revo3_v1.emg.model import GNIModelConfig
from revo3_v1.emg.migration import GNI_SOURCE_COMMIT
from revo3_v1.emg.preprocessing import BRAINCO_EDU_8CH_250HZ
from revo3_v1.emg.train import EMGTrainingConfig, train_binary_emg, train_emg


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fixture-binary",
        action="store_true",
        help="run the legacy synthetic OPEN/CLOSE smoke instead of the V1 mainline",
    )
    parser.add_argument("--preset", choices=("smoke", "gni"), default="gni")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--gni-checkpoint", type=Path)
    source.add_argument("--from-scratch-ablation", action="store_true")
    parser.add_argument("--gni-source-commit", default=GNI_SOURCE_COMMIT)
    parser.add_argument("--allow-window-reset-fallback", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args(argv)
    if not args.fixture_binary and args.gni_checkpoint is None and not args.from_scratch_ablation:
        parser.error("mainline training requires --gni-checkpoint or --from-scratch-ablation")
    if args.fixture_binary and (args.gni_checkpoint or args.from_scratch_ablation):
        parser.error("binary fixture cannot load/misrepresent the mainline GNI migration")
    if not args.fixture_binary and args.preset == "smoke" and not args.from_scratch_ablation:
        parser.error("five-class smoke topology is only a --from-scratch-ablation")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    import numpy as np

    with np.load(args.dataset / "windows.npz", allow_pickle=False) as archive:
        input_channels = int(archive["signal"].shape[1])
    training_config = EMGTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
    )
    if args.fixture_binary:
        model_config = (
            GNIModelConfig.smoke(input_channels)
            if args.preset == "smoke"
            else GNIModelConfig(input_channels=input_channels)
        )
        summary = train_binary_emg(args.dataset, args.output, model_config, training_config)
    else:
        if input_channels != BRAINCO_EDU_8CH_250HZ.channel_count:
            raise ValueError("mainline BrainCo profile requires exactly 8 input channels")
        model_config = (
            GNIModelConfig.smoke(input_channels, mainline=True)
            if args.preset == "smoke"
            else GNIModelConfig.brainco_edu()
        )
        summary = train_emg(
            args.dataset,
            args.output,
            model_config,
            training_config,
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
