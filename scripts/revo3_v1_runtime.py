#!/usr/bin/env python3
"""Validate or run the fail-closed Revo3 V1 double-rate runtime."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
from pathlib import Path
import signal
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from revo3_v1.planner import SupportedTask  # noqa: E402
from revo3_v1.runtime import (  # noqa: E402
    ProductionBindings,
    RuntimeAssemblyError,
    build_production_runtime,
    build_simulation_runtime,
)


DEFAULT_CONTROL = _ROOT / "config" / "revo3_v1_control.json"


def _load_bindings(spec: str) -> ProductionBindings:
    if ":" not in spec:
        raise RuntimeAssemblyError("--bindings-factory must be module:function")
    module_name, function_name = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), function_name)
    value = factory()
    if not isinstance(value, ProductionBindings):
        raise RuntimeAssemblyError("bindings factory must return ProductionBindings")
    return value


async def _run_assembly(assembly, ticks: int | None) -> dict[str, object]:
    backend = assembly.coordinator.servo.pipeline.backend
    shutdown = await assembly.service.run(max_servo_ticks=ticks)
    evidence = assembly.service.evidence
    report = {
        **dict(assembly.resolved),
        "shutdown_clean": shutdown.clean,
        "stop_succeeded": shutdown.stop_succeeded,
        "io_clean": shutdown.io_clean,
        "io_timed_out": shutdown.io_timed_out,
        "io_intervention_required": shutdown.io_intervention_required,
        "backend_clean": shutdown.backend_clean,
        "backend_timed_out": shutdown.backend_timed_out,
        "backend_intervention_required": shutdown.backend_intervention_required,
        "observed_executive_outputs": list(evidence.executive_outputs),
        "observed_executive_reasons": list(evidence.executive_reasons),
        "observed_motion_directives": list(evidence.motion_directives),
        "observed_policy_states": list(evidence.policy_states),
        "authorized_servo_write_count": evidence.servo_write_count,
        "mock_motor_write_count": len(getattr(backend, "commands", ())),
        "mock_soft_stop_count": len(getattr(backend, "soft_stop_reasons", ())),
    }
    if not shutdown.clean:
        raise RuntimeAssemblyError(
            "runtime shutdown was not fail-closed: SoftStop or worker/IO close was unconfirmed"
        )
    return report


async def _validate_assembly(assembly) -> dict[str, object]:
    shutdown = await assembly.service.close()
    if not shutdown.clean:
        raise RuntimeAssemblyError(
            "runtime validation shutdown was not fail-closed: SoftStop or worker/IO close was unconfirmed"
        )
    report = dict(assembly.resolved)
    report["shutdown"] = True
    report["stop_succeeded"] = shutdown.stop_succeeded
    report["io_clean"] = shutdown.io_clean
    report["io_timed_out"] = shutdown.io_timed_out
    report["io_intervention_required"] = shutdown.io_intervention_required
    report["backend_clean"] = shutdown.backend_clean
    report["backend_timed_out"] = shutdown.backend_timed_out
    report["backend_intervention_required"] = (
        shutdown.backend_intervention_required
    )
    return report


def _effective_servo_ticks(mode: str, requested: int | None) -> int | None:
    """Production is continuous unless a bounded smoke count is explicit."""

    if requested is not None:
        return int(requested)
    return None if mode == "production" else 120


async def _run_assembly_with_signals(
    assembly, ticks: int | None
) -> dict[str, object]:
    """Translate SIGINT/SIGTERM into the service's owned shutdown event."""

    loop = asyncio.get_running_loop()
    installed: list[signal.Signals] = []
    fallback_previous: dict[signal.Signals, object] = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, assembly.service.request_shutdown)
        except (NotImplementedError, RuntimeError):
            # Windows' Proactor loop has no add_signal_handler.  Install a
            # minimal main-thread bridge instead of relying on process-kill
            # semantics that could bypass SoftStop.
            try:
                fallback_previous[signum] = signal.getsignal(signum)
                signal.signal(
                    signum,
                    lambda _sig, _frame: loop.call_soon_threadsafe(
                        assembly.service.request_shutdown
                    ),
                )
            except (ValueError, OSError):
                # Non-main-thread embedding cannot own process signals; its
                # supervisor must call request_shutdown directly.
                fallback_previous.pop(signum, None)
            continue
        installed.append(signum)
    try:
        return await _run_assembly(assembly, ticks)
    finally:
        for signum in installed:
            loop.remove_signal_handler(signum)
        for signum, previous in fallback_previous.items():
            signal.signal(signum, previous)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("simulation", "production"), required=True)
    parser.add_argument("--control-config", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("--bindings-factory", default="")
    parser.add_argument(
        "--servo-ticks",
        type=int,
        default=None,
        help=(
            "bounded tick count; simulation defaults to 120, while production "
            "runs continuously when this option is omitted"
        ),
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--task",
        choices=("bottle", "phone", "plastic_bag", "refrigerator_door", "all"),
        default="all",
    )
    args = parser.parse_args(argv)
    if args.servo_ticks is not None and args.servo_ticks <= 0:
        parser.error("--servo-ticks must be positive")
    try:
        if args.mode == "production":
            if args.runtime_config is None or not args.bindings_factory:
                raise RuntimeAssemblyError(
                    "production requires --runtime-config and --bindings-factory"
                )
            assembly = build_production_runtime(
                args.control_config,
                args.runtime_config,
                bindings=_load_bindings(args.bindings_factory),
            )
            if args.validate_only:
                report = asyncio.run(_validate_assembly(assembly))
            else:
                # Production is continuous by default.  Supported event loops
                # map SIGINT/SIGTERM to request_shutdown; task cancellation is
                # still routed through join -> SoftStop -> close.
                report = asyncio.run(
                    _run_assembly_with_signals(
                        assembly,
                        _effective_servo_ticks("production", args.servo_ticks),
                    )
                )
        else:
            tasks = (
                tuple(SupportedTask)
                if args.task == "all"
                else (SupportedTask(args.task),)
            )
            reports = []
            for task in tasks:
                assembly = build_simulation_runtime(args.control_config, task=task)
                if args.validate_only:
                    item = asyncio.run(_validate_assembly(assembly))
                else:
                    item = asyncio.run(
                        _run_assembly_with_signals(
                            assembly,
                            _effective_servo_ticks("simulation", args.servo_ticks),
                        )
                    )
                reports.append(item)
            report = {
                "mode": "simulation",
                "scope": "mock wiring smoke; no robot-task or clinical claim",
                "tasks": reports,
            }
    except (RuntimeAssemblyError, ValueError, RuntimeError, ImportError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
