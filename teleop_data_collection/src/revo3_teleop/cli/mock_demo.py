from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from revo3_teleop.mock import MockCollectionConfig, TASK_INSTRUCTIONS, run_mock_collection


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a causally aligned, multi-rate synthetic collection fixture."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="checked revo3-teleop-config-v1 JSON; CLI values override it",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--duration-s", type=float)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=tuple(TASK_INSTRUCTIONS),
    )
    arguments = parser.parse_args(argv)
    if arguments.config is None:
        if arguments.output is None:
            parser.error("--output is required when --config is not supplied")
        config = MockCollectionConfig(
            output_root=arguments.output,
            duration_s=0.5 if arguments.duration_s is None else arguments.duration_s,
            tasks=(
                tuple(TASK_INSTRUCTIONS)
                if arguments.tasks is None
                else tuple(arguments.tasks)
            ),
        )
    else:
        config = MockCollectionConfig.from_json(arguments.config)
        overrides = {}
        if arguments.output is not None:
            overrides["output_root"] = arguments.output
        if arguments.duration_s is not None:
            overrides["duration_s"] = arguments.duration_s
        if arguments.tasks is not None:
            overrides["tasks"] = tuple(arguments.tasks)
        config = replace(config, **overrides)
    results = run_mock_collection(config)
    print(json.dumps({"synthetic_fixture": True, "episodes": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
