"""Ask-to-Clarify style RGB + EMG-action instruction planner."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, Dict, Mapping, Sequence

from .backends import BackendResponse, PlannerBackend, extract_json_object
from .schema import (
    DEFAULT_INSTRUCTIONS,
    PlannerDecision,
    PlannerRequest,
    PlannerStatus,
    SupportedTask,
)


SYSTEM_CONSTRAINTS = """You are the instruction planner for a Revo 3 assistive hand.
The EMG classifier has already latched CLOSE. Use only the rectified RGB images
to identify one of four allow-listed short tasks:
1. bottle: grasp the centered bottle and hold it;
2. phone: grasp the centered phone and hold it;
3. plastic_bag: grasp the bag handles and lift it;
4. refrigerator_door: grasp the refrigerator handle and pull the door open.

Do not invent an object, do not change CLOSE into another user intent, and do
not plan navigation or arm motion. If the object is absent, too far, badly
off-center, incompatible, or multiple targets are genuinely ambiguous, mark it
not ready. If one short clarification can resolve multiple plausible targets,
set ambiguity.ambiguous=true and ask one concise question.

Return exactly one JSON object and no prose. bbox is normalized [x_min,y_min,
x_max,y_max] in [0,1]. area must equal bbox area. The schema is:
{
  "schema_version": "revo3_planner_v1",
  "status": "READY|ASK_CLARIFY|NOT_READY",
  "task": "bottle|phone|plastic_bag|refrigerator_door|null",
  "bbox": [x_min,y_min,x_max,y_max] or null,
  "area": 0.0,
  "confidence": 0.0,
  "target_present": false,
  "near_ready": false,
  "compatible": false,
  "ambiguity": {
    "ambiguous": false,
    "reason": "",
    "candidates": [],
    "question": ""
  },
  "instruction": ""
}
"""


class AskToClarifyPlanner:
    """Validated multi-turn planner with an injectable backend."""

    def __init__(self, backend: PlannerBackend) -> None:
        self.backend = backend

    @staticmethod
    def _payload(response: BackendResponse) -> tuple[Mapping[str, Any], str]:
        if isinstance(response, Mapping):
            return response, json.dumps(response, ensure_ascii=False, sort_keys=True)
        text = str(response)
        return extract_json_object(text), text

    @staticmethod
    def _normalize_instruction(decision: PlannerDecision) -> PlannerDecision:
        if decision.task is None:
            return decision
        instruction = " ".join(decision.instruction.split())
        if not instruction and decision.status == PlannerStatus.READY:
            instruction = DEFAULT_INSTRUCTIONS[decision.task]
        if len(instruction) > 256:
            raise ValueError("planner instruction is longer than the V1 contract")
        return replace(decision, instruction=instruction)

    @staticmethod
    def build_prompt(request: PlannerRequest) -> str:
        prompt = SYSTEM_CONSTRAINTS
        prompt += "\nLatched EMG action: CLOSE. Inspect the most recent image last."
        if request.metadata:
            public_metadata = {
                key: value
                for key, value in request.metadata.items()
                if key in {"camera_view", "task_context", "frame_spacing_ms"}
            }
            if public_metadata:
                prompt += "\nContext metadata: " + json.dumps(public_metadata)
        return prompt

    def plan(self, request: PlannerRequest) -> PlannerDecision:
        response = self.backend.generate(
            prompt=self.build_prompt(request),
            images=request.images,
        )
        payload, raw = self._payload(response)
        decision = PlannerDecision.from_mapping(
            payload,
            timestamp_ns=request.timestamp_ns,
            raw_response=raw,
        )
        decision = self._normalize_instruction(decision)

        if decision.status == PlannerStatus.ASK_CLARIFY and request.clarification_answer:
            return self.clarify(decision, request.clarification_answer, request)
        return decision

    def clarify(
        self,
        previous: PlannerDecision,
        answer: str,
        request: PlannerRequest,
    ) -> PlannerDecision:
        if not previous.ambiguity.ambiguous:
            raise ValueError("clarify requires an ambiguous previous decision")
        answer = " ".join(answer.split())[:256]
        if not answer:
            raise ValueError("clarification answer must not be empty")
        conversation = (
            {"role": "assistant", "content": previous.raw_response},
            {"role": "user", "content": answer},
        )
        prompt = SYSTEM_CONSTRAINTS + (
            "\nResolve the previous ambiguity using the user's short answer. "
            "Re-inspect the RGB images and return the same strict schema."
        )
        response = self.backend.generate(
            prompt=prompt,
            images=request.images,
            conversation=conversation,
        )
        payload, raw = self._payload(response)
        result = PlannerDecision.from_mapping(
            payload,
            timestamp_ns=request.timestamp_ns,
            raw_response=raw,
        )
        return self._normalize_instruction(result)

