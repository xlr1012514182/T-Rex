"""Audit or explicitly execute one fail-closed hardware collection episode."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from revo3_teleop.hardware.collection import (
    HardwareCollectionConfig,
    HardwareCollectionOrchestrator,
    assess_hardware_collection_config,
    load_hardware_collection_dependencies,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a complete Revo3 collection configuration. The default path "
            "imports no assembly plugin, connects no device and writes no actuator."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--execute-hardware",
        action="store_true",
        help="load the hash-verified assembly and run one configured episode",
    )
    parser.add_argument(
        "--allow-hardware-connect",
        action="store_true",
        help="second command-line confirmation required with --execute-hardware",
    )
    parser.add_argument(
        "--allow-hardware-write",
        action="store_true",
        help="third command-line confirmation required with --execute-hardware",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = HardwareCollectionConfig.from_json(args.config)
        readiness = assess_hardware_collection_config(config)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if not args.execute_hardware:
        print(json.dumps(readiness.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if not args.allow_hardware_connect or not args.allow_hardware_write:
        parser.error(
            "--execute-hardware additionally requires --allow-hardware-connect "
            "and --allow-hardware-write"
        )
    if not readiness.execute_ready:
        parser.error(
            "hardware collection readiness failed: " + ", ".join(readiness.blockers)
        )
    dependencies = load_hardware_collection_dependencies(config, readiness)
    result = asyncio.run(
        HardwareCollectionOrchestrator(
            config,
            dependencies,
            readiness=readiness,
        ).run()
    )
    output = {
        "master_episode": str(result.master_episode),
        "vla_episode": None if result.vla_episode is None else str(result.vla_episode),
        "emg_review_root": str(result.emg_review_root),
        "anchors": result.anchors,
        "action_label_contract": "accepted_revo_hand_exact_sent_target_only",
        "emg_export_status": "requires_human_reviewed_intervals",
        "real_task_success_verified": False,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
