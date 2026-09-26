"""Pinned Qwen Planner adapter and processor lineage contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


QWEN3_VL_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
QWEN3_VL_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
PLANNER_ADAPTER_SCHEMA = "revo3-planner-lora-adapter-v1"
PROCESSOR_SCHEMA = "revo3-qwen3vl-processor-v1"
ADAPTER_MANIFEST_NAME = "adapter_manifest.json"


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def processor_identity(processor: Any) -> Mapping[str, Any]:
    """Capture the behavior-relevant processor classes/config/chat template."""

    image_processor = getattr(processor, "image_processor", None)
    tokenizer = getattr(processor, "tokenizer", None)
    image_config = (
        image_processor.to_dict()
        if image_processor is not None and callable(getattr(image_processor, "to_dict", None))
        else {}
    )
    tokenizer_config = dict(getattr(tokenizer, "init_kwargs", {}) or {})
    # Local cache paths are not semantic and would make the fingerprint host-specific.
    for key in ("name_or_path", "tokenizer_file"):
        tokenizer_config.pop(key, None)
    return {
        "schema_version": PROCESSOR_SCHEMA,
        "processor_class": f"{processor.__class__.__module__}.{processor.__class__.__name__}",
        "image_processor_class": (
            "" if image_processor is None else f"{image_processor.__class__.__module__}.{image_processor.__class__.__name__}"
        ),
        "tokenizer_class": (
            "" if tokenizer is None else f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__name__}"
        ),
        "image_processor_config": image_config,
        "tokenizer_init_config": tokenizer_config,
        "chat_template": str(getattr(processor, "chat_template", "")),
    }


def processor_sha256(processor: Any) -> str:
    return _canonical_hash(processor_identity(processor))


def adapter_files_sha256(adapter_dir: str | Path) -> str:
    root = Path(adapter_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Planner adapter directory does not exist: {root}")
    digest = hashlib.sha256()
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.name != ADAPTER_MANIFEST_NAME
    )
    if not files:
        raise ValueError("Planner adapter directory has no artifact files")
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def load_and_validate_adapter_manifest(
    adapter_dir: str | Path,
    *,
    expected_model_id: str,
    expected_revision: str,
    expected_processor_sha256: str | None = None,
) -> Mapping[str, Any]:
    root = Path(adapter_dir).resolve()
    path = root / ADAPTER_MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Planner adapter manifest is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != PLANNER_ADAPTER_SCHEMA:
        raise ValueError("Unsupported Planner adapter manifest schema")
    if payload.get("base_model") != expected_model_id:
        raise ValueError("Planner adapter/base model mismatch")
    if payload.get("base_revision") != expected_revision:
        raise ValueError("Planner adapter/base revision mismatch")
    if payload.get("processor_schema") != PROCESSOR_SCHEMA:
        raise ValueError("Planner adapter processor schema mismatch")
    if payload.get("adapter_files_sha256") != adapter_files_sha256(root):
        raise ValueError("Planner adapter artifact hash mismatch")
    if (
        expected_processor_sha256 is not None
        and payload.get("processor_sha256") != expected_processor_sha256
    ):
        raise ValueError("Planner adapter/processor fingerprint mismatch")
    return payload
