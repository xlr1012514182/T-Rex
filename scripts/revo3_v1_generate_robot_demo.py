#!/usr/bin/env python
"""Generate and convert synthetic Revo3 robot-only episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from revo3_v1.data import (
    ConversionConfig,
    SyntheticRevoConfig,
    convert_revo_episodes_to_trex_json,
    generate_synthetic_revo_episodes,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=2)
    parser.add_argument("--frames", type=int, default=48)
    args = parser.parse_args()
    episode_root = args.output / "episodes"
    generate_synthetic_revo_episodes(
        episode_root,
        SyntheticRevoConfig(
            episodes_per_task=args.episodes_per_task,
            frames_per_episode=args.frames,
        ),
    )
    result = convert_revo_episodes_to_trex_json(
        episode_root,
        args.output / "revo3_trex_train.json",
        ConversionConfig(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
