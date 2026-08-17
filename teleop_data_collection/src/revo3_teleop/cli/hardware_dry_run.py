"""Side-effect-free Tianji/hand collection hardware readiness CLI."""

from __future__ import annotations

import argparse
import json

from revo3_teleop.hardware import HardwareConfig, assemble_tianji_hardware


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit Tianji/glove configuration. Default mode imports no plugin, "
            "opens no device and sends no command."
        )
    )
    parser.add_argument("--config", required=True, help="hardware-v2 JSON configuration")
    parser.add_argument(
        "--load-sdk-plugin",
        action="store_true",
        help="explicitly import/construct the configured SDK client; still does not connect",
    )
    parser.add_argument(
        "--load-retargeting-plugins",
        action="store_true",
        help="explicitly construct wrist/IK plugins; they must be side-effect-free factories",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = HardwareConfig.from_json(args.config)
    assembly = assemble_tianji_hardware(
        config,
        load_sdk_plugin=args.load_sdk_plugin,
        load_retargeting_plugins=args.load_retargeting_plugins,
    )
    output = assembly.report.to_dict()
    output.update(
        {
            "sdk_plugin_constructed": assembly.sdk is not None,
            "backend_constructed": assembly.backend is not None,
            "backend_connected": bool(
                assembly.backend is not None and assembly.backend.connected
            ),
            "retargeting_runtime_constructed": assembly.runtime is not None,
        }
    )
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    # Dry-run blockers are the intended, safe result for the checked-in
    # template; reserve nonzero exit codes for malformed input/import failure.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
