#!/usr/bin/env python3
"""Run one bounded Qwen3-VL planner generation and save auditable evidence.

This is a model-loading and generation smoke test.  A valid planner JSON is
reported separately from raw generation success; neither result is a task-
grounding, robot-control, latency, or clinical-performance claim.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from revo3_v1.planner import (  # noqa: E402
    AskToClarifyPlanner,
    PlannerDecision,
    PlannerRequest,
    Qwen3VLBackend,
)
from revo3_v1.planner.backends import extract_json_object  # noqa: E402


DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"


def _synthetic_bottle_image() -> Image.Image:
    """Return a deterministic fixture image, not an evaluation sample."""

    image = Image.new("RGB", (384, 288), color=(235, 235, 235))
    draw = ImageDraw.Draw(image)
    draw.rectangle((158, 82, 226, 250), fill=(50, 105, 180), outline=(20, 35, 60), width=4)
    draw.rectangle((174, 52, 210, 86), fill=(50, 105, 180), outline=(20, 35, 60), width=4)
    draw.rectangle((170, 42, 214, 56), fill=(220, 220, 220), outline=(20, 35, 60), width=3)
    return image


def _load_image(path: Path | None) -> tuple[Image.Image, str]:
    if path is None:
        return _synthetic_bottle_image(), "generated_synthetic_bottle_fixture"
    with Image.open(path) as source:
        return source.convert("RGB"), str(path.resolve())


def _cuda_snapshot() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False}
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return {
            "available": True,
            "device": torch.cuda.current_device(),
            "name": torch.cuda.get_device_name(torch.cuda.current_device()),
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        }
    except Exception as exc:  # pragma: no cover - diagnostic boundary
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def _decision_payload(decision: PlannerDecision) -> dict[str, Any]:
    return {
        "status": decision.status.value,
        "primitive": None if decision.primitive is None else decision.primitive.value,
        "task": None if decision.task is None else decision.task.value,
        "bbox": None if decision.bbox is None else list(decision.bbox.as_xyxy()),
        "area": decision.area,
        "confidence": decision.confidence,
        "target_present": decision.target_present,
        "near_ready": decision.near_ready,
        "center_ready": decision.center_ready,
        "ready_frame_count": decision.ready_frame_count,
        "compatible": decision.compatible,
        "instruction": decision.instruction,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument(
        "--require-schema",
        action="store_true",
        help="Return non-zero when generation succeeds but strict planner JSON validation fails.",
    )
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")

    image, image_source = _load_image(args.image)
    timestamp_ns = time.monotonic_ns()
    request = PlannerRequest.from_aligned_views(
        primitive="POWER_GRASP",
        full_view_history=(image.copy(), image.copy(), image.copy()),
        center_view=image,
        full_view_timestamps_ns=(
            timestamp_ns - 200_000_000,
            timestamp_ns - 100_000_000,
            timestamp_ns,
        ),
        center_timestamp_ns=timestamp_ns,
        metadata={"camera_schema": "synthetic-smoke-aligned-views-v1"},
    )
    backend = Qwen3VLBackend(
        model_id=args.model_id,
        revision=args.revision,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        max_new_tokens=args.max_new_tokens,
        local_files_only=args.local_files_only,
        adapter_path=args.adapter,
        production=False,
    )

    started_ns = time.monotonic_ns()
    cuda_before = _cuda_snapshot()
    report: dict[str, Any] = {
        "schema_version": "revo3-qwen-gpu-smoke-v1",
        "verification_scope": "single model load and deterministic generation",
        "model_id": args.model_id,
        "revision": args.revision,
        "planner_revision": backend.planner_revision,
        "deployment_mode": backend.deployment_mode,
        "adapter_path": None if args.adapter is None else str(args.adapter.resolve()),
        "image_source": image_source,
        "emg_action": "POWER_GRASP",
        "view_contract": "3full+current-center; latest full and center share timestamp",
        "generation_succeeded": False,
        "planner_schema_valid": False,
        "task_success_defined": False,
        "claims_excluded": [
            "object-grounding accuracy",
            "robot task success",
            "latency certification",
            "clinical performance",
        ],
        "cuda_before": cuda_before,
    }
    exit_code = 0
    try:
        prompt = AskToClarifyPlanner.build_prompt(request)
        raw = backend.generate(prompt=prompt, images=request.images)
        report["generation_succeeded"] = True
        report["raw_response"] = raw
        try:
            payload = extract_json_object(raw)
            decision = PlannerDecision.from_mapping(
                payload,
                timestamp_ns=request.timestamp_ns,
                expected_primitive=request.primitive,
                raw_response=raw,
            )
            report["planner_schema_valid"] = True
            report["decision"] = _decision_payload(decision)
        except Exception as exc:
            report["schema_error"] = f"{type(exc).__name__}: {exc}"
            if args.require_schema:
                exit_code = 2
    except Exception as exc:
        report["generation_error"] = f"{type(exc).__name__}: {exc}"
        exit_code = 1
    finally:
        report["duration_ms"] = (time.monotonic_ns() - started_ns) / 1_000_000.0
        report["cuda_after"] = _cuda_snapshot()
        report["pid"] = os.getpid()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, args.output)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
