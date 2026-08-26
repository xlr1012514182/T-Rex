#!/usr/bin/env python
"""Run the Revo3 V1 end-to-end simulation smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from revo3_v1.demo import DemoConfig, run
from revo3_v1.planner import SupportedTask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=[task.value for task in SupportedTask], default="bottle")
    parser.add_argument("--trace", default="")
    parser.add_argument("--hold", action="store_true", help="End in stable HOLD instead of mock release")
    args = parser.parse_args()
    result = run(
        DemoConfig(
            task=SupportedTask(args.task),
            emulate_release=not args.hold,
            output_trace=args.trace,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
