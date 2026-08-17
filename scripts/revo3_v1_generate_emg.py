#!/usr/bin/env python
"""Generate synthetic OPEN/CLOSE EMG data for the Revo3 integration demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from revo3_v1.emg.synthetic import SyntheticEMGConfig, generate_synthetic_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preset", choices=("smoke", "demo", "gni-shape"), default="demo")
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--subjects", type=int)
    parser.add_argument("--sessions", type=int)
    parser.add_argument("--windows-per-label", type=int)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--sample-rate", type=int)
    parser.add_argument("--window-seconds", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    presets = {
        "smoke": dict(
            n_subjects=3,
            sessions_per_subject=1,
            windows_per_label_per_session=2,
            sample_rate_hz=200,
            window_seconds=0.5,
            window_stride_seconds=0.6,
        ),
        "demo": dict(
            n_subjects=8,
            sessions_per_subject=3,
            windows_per_label_per_session=8,
            sample_rate_hz=1_000,
            window_seconds=1.0,
            window_stride_seconds=1.25,
        ),
        "gni-shape": dict(
            n_subjects=8,
            sessions_per_subject=2,
            windows_per_label_per_session=4,
            sample_rate_hz=2_000,
            window_seconds=8.0,
            window_stride_seconds=8.25,
        ),
    }
    values = presets[args.preset]
    overrides = {
        "n_subjects": args.subjects,
        "sessions_per_subject": args.sessions,
        "windows_per_label_per_session": args.windows_per_label,
        "sample_rate_hz": args.sample_rate,
        "window_seconds": args.window_seconds,
    }
    values.update({key: value for key, value in overrides.items() if value is not None})
    if args.window_seconds is not None and args.window_seconds >= values["window_stride_seconds"]:
        values["window_stride_seconds"] = args.window_seconds * 1.05
    config = SyntheticEMGConfig(n_channels=args.channels, seed=args.seed, **values)
    metadata = generate_synthetic_dataset(args.output, config)
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

