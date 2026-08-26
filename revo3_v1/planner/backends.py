"""Planner backends, including a lazily loaded Qwen3-VL adapter."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Union

from .schema import SupportedTask
from .artifacts import (
    QWEN3_VL_MODEL_ID,
    QWEN3_VL_REVISION,
    load_and_validate_adapter_manifest,
    processor_sha256,
)


BackendResponse = Union[str, Mapping[str, Any]]


class PlannerBackend(Protocol):
    def generate(
        self,
        *,
        prompt: str,
        images: Sequence[Any],
        conversation: Sequence[Mapping[str, str]] = (),
    ) -> BackendResponse:
        ...


def extract_json_object(text: str) -> Dict[str, Any]:
    """Extract one JSON object without accepting trailing model commentary."""

    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    else:
        start = stripped.find("{")
        if start < 0:
            raise ValueError("planner response contains no JSON object")
        decoder = json.JSONDecoder()
        value, end = decoder.raw_decode(stripped[start:])
        trailing = stripped[start + end :].strip()
        if trailing:
            raise ValueError("planner response contains trailing non-JSON text")
        if not isinstance(value, dict):
            raise ValueError("planner response must be a JSON object")
        return value
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise ValueError("planner response must be a JSON object")
    return value


@dataclass
class MockPlannerBackend:
    """Deterministic backend for CI and simulated end-to-end demos.

    Scripted responses are consumed first.  Otherwise the task/bbox/confidence
    supplied in ``default_metadata`` are converted to a valid planner response.
    This class never inspects image pixels and must not be used as an evaluation
    substitute for the real VLM.
    """

    scripted_responses: List[BackendResponse] = field(default_factory=list)
    default_task: SupportedTask = SupportedTask.BOTTLE
    default_bbox: Sequence[float] = (0.25, 0.20, 0.75, 0.85)
    default_confidence: float = 0.95
    calls: List[Dict[str, Any]] = field(default_factory=list)

    def generate(
        self,
        *,
        prompt: str,
        images: Sequence[Any],
        conversation: Sequence[Mapping[str, str]] = (),
    ) -> BackendResponse:
        self.calls.append(
            {
                "prompt": prompt,
                "image_count": len(images),
                "conversation": list(conversation),
            }
        )
        if self.scripted_responses:
            return self.scripted_responses.pop(0)
        x1, y1, x2, y2 = (float(value) for value in self.default_bbox)
        primitive = "POWER_GRASP"
        match = re.search(r"Latched primitive:\s*([A-Z_]+)", prompt)
        if match:
            primitive = match.group(1)
        grasp_style = {
            "POWER_GRASP": "power",
            "PRECISION_GRASP": "precision",
            "LATERAL_GRASP": "lateral",
        }.get(primitive, "power")
        target_text = {
            SupportedTask.BOTTLE: "centered bottle body",
            SupportedTask.PHONE: "centered phone body",
            SupportedTask.PLASTIC_BAG: "centered plastic bag handles",
            SupportedTask.REFRIGERATOR_DOOR: "centered refrigerator door handle",
        }[self.default_task]
        return {
            "schema_version": "planner_v1",
            "status": "READY",
            "primitive": primitive,
            "target_category": self.default_task.value,
            "target_part": "body",
            "grasp_style": grasp_style,
            "target_region": [
                (x1 + x2) / 2,
                (y1 + y2) / 2,
                x2 - x1,
                y2 - y1,
            ],
            "confidence": self.default_confidence,
            "target_present": True,
            "near_ready": True,
            "center_ready": True,
            "compatible": True,
            "ambiguous": False,
            "ready_frame_count": 3,
            "reason_code": "READY",
            "instruction": f"Grasp the {target_text} using a {grasp_style} grasp and hold it securely.",
        }


class Qwen3VLBackend:
    """Lazily loaded Hugging Face Qwen3-VL backend.

    Importing this module does not import torch/transformers or allocate GPU
    memory.  Model loading occurs only on the first call to :meth:`generate`.
    The pinned revision is the source-aligned model used by the V1 design.
    """

    def __init__(
        self,
        model_id: str = QWEN3_VL_MODEL_ID,
        *,
        revision: str = QWEN3_VL_REVISION,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 384,
        local_files_only: bool = False,
        adapter_path: str | Path | None = None,
        production: bool = False,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.local_files_only = local_files_only
        self.adapter_path = None if adapter_path is None else Path(adapter_path).resolve()
        self.production = bool(production)
        if self.production and int(max_new_tokens) != 384:
            raise ValueError(
                "Production Planner requires max_new_tokens=384; 128 is a truncated-output smoke ablation"
            )
        if self.production and self.adapter_path is None:
            raise ValueError("Production Planner requires a validated LoRA adapter")
        if self.production and (
            self.model_id != QWEN3_VL_MODEL_ID or self.revision != QWEN3_VL_REVISION
        ):
            raise ValueError("Production Planner requires the pinned Qwen3-VL revision")
        self._adapter_manifest = (
            None
            if self.adapter_path is None
            else load_and_validate_adapter_manifest(
                self.adapter_path,
                expected_model_id=self.model_id,
                expected_revision=self.revision,
            )
        )
        self._model: Any = None
        self._processor: Any = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def deployment_mode(self) -> str:
        return "planner_lora" if self.adapter_path is not None else "base_model_smoke_ablation"

    @property
    def planner_revision(self) -> str:
        if self._adapter_manifest is None:
            return f"{self.model_id}@{self.revision}:base-smoke"
        return (
            f"{self.model_id}@{self.revision}:"
            f"{self._adapter_manifest['adapter_files_sha256']}"
        )

    def _load(self) -> None:
        if self.is_loaded:
            return
        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "Qwen3VLBackend requires transformers with "
                "Qwen3VLForConditionalGeneration support"
            ) from exc

        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            revision=self.revision,
            local_files_only=self.local_files_only,
        )
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id,
            revision=self.revision,
            device_map=self.device_map,
            torch_dtype=self.torch_dtype,
            local_files_only=self.local_files_only,
        )
        if self.adapter_path is not None:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("Planner LoRA loading requires peft") from exc
            self._adapter_manifest = load_and_validate_adapter_manifest(
                self.adapter_path,
                expected_model_id=self.model_id,
                expected_revision=self.revision,
                expected_processor_sha256=processor_sha256(self._processor),
            )
            self._model = PeftModel.from_pretrained(
                self._model,
                str(self.adapter_path),
                is_trainable=False,
            )
        self._model.eval()

    @staticmethod
    def _image_content(image: Any) -> Mapping[str, Any]:
        return {"type": "image", "image": image}

    def generate(
        self,
        *,
        prompt: str,
        images: Sequence[Any],
        conversation: Sequence[Mapping[str, str]] = (),
    ) -> str:
        self._load()
        messages: List[Mapping[str, Any]] = []
        for turn in conversation:
            messages.append(
                {
                    "role": str(turn.get("role", "user")),
                    "content": [{"type": "text", "text": str(turn.get("content", ""))}],
                }
            )
        content: List[Mapping[str, Any]] = [self._image_content(image) for image in images]
        content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content})

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        if hasattr(inputs, "to"):
            inputs = inputs.to(self._model.device)
        generated = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        input_ids = inputs["input_ids"]
        trimmed = [output[len(source) :] for source, output in zip(input_ids, generated)]
        return self._processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
