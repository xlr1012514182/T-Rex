"""Qwen-backed RGB + EMG-primitive instruction planner."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from .backends import BackendResponse, PlannerBackend, extract_json_object
from .schema import PlannerDecision, PlannerRequest, PlannerStatus


SYSTEM_CONSTRAINTS = """You are the V1 task planner for a Revo 3 assistive hand.
The input grasp primitive is already latched and MUST be copied unchanged. Use
the three recent full-view frames and the current center-view frame to identify
exactly one short, near-contact hand task. Never plan arm motion or navigation.
Do not invent an object. If absent, too far, off-center, incompatible, or
ambiguous, do not mark READY. Return one JSON object and no prose:
{
 "schema_version":"planner_v1",
 "status":"READY|ASK_CLARIFY|NOT_READY",
 "primitive":"POWER_GRASP|PRECISION_GRASP|LATERAL_GRASP",
 "target_category":"bottle|phone|plastic_bag|refrigerator_door|null",
 "target_part":"",
 "grasp_style":"power|precision|lateral",
 "target_present":false,
 "near_ready":false,
 "center_ready":false,
 "compatible":false,
 "ambiguous":false,
 "target_region":[0.5,0.5,0.0,0.0],
 "ready_frame_count":0,
 "confidence":0.0,
 "instruction":"",
 "ask_clarify":false,
 "reason_code":""
}
target_region is normalized [cx,cy,width,height]. ready_frame_count is 0..3.
The instruction must preserve the primitive, describe one short action, refer
only to a visible target, and contain at most 25 English whitespace tokens.
The frozen V1 compatible pairs are: bottle+POWER_GRASP,
phone+PRECISION_GRASP, plastic_bag+PRECISION_GRASP, and
refrigerator_door+LATERAL_GRASP. Never mark another pair READY.
"""


class AskToClarifyPlanner:
    """Validated Qwen-style planner with one bounded repair retry."""

    def __init__(self, backend: PlannerBackend, *, repair_retries: int = 1) -> None:
        if repair_retries != 1:
            raise ValueError("V1 fixes planner repair_retries to exactly one")
        self.backend = backend
        self.repair_retries = repair_retries

    @staticmethod
    def _payload(response: BackendResponse) -> tuple[Mapping[str, Any], str]:
        if isinstance(response, Mapping):
            return response, json.dumps(response, ensure_ascii=False, sort_keys=True)
        raw = str(response)
        return extract_json_object(raw), raw

    @staticmethod
    def build_prompt(request: PlannerRequest) -> str:
        prompt = SYSTEM_CONSTRAINTS
        prompt += f"\nLatched primitive: {request.primitive.value}. Copy it exactly."
        if request.uses_aligned_views:
            ages_ms = [
                round((request.timestamp_ns - value) / 1_000_000, 3)
                for value in request.frame_timestamps_ns[:3]
            ]
            prompt += (
                "\nImage order: full(t-2), full(t-1), full(t), center(t). "
                f"Full-frame ages_ms={ages_ms}."
            )
        else:
            prompt += "\nCompatibility input: inspect images in order, newest last."
        public_metadata = {
            key: value
            for key, value in request.metadata.items()
            if key in {"camera_schema", "camera_revision", "frame_spacing_ms"}
        }
        if public_metadata:
            prompt += "\nVersion metadata: " + json.dumps(public_metadata, sort_keys=True)
        return prompt

    def _validated_generate(
        self,
        *,
        request: PlannerRequest,
        prompt: str,
        conversation: Sequence[Mapping[str, str]] = (),
    ) -> PlannerDecision:
        last_raw = ""
        last_error = "planner returned no response"
        repair_conversation = list(conversation)
        for attempt in range(2):
            current_prompt = prompt
            if attempt:
                current_prompt += (
                    "\nREPAIR: the preceding output violated the JSON/schema/primitive "
                    "contract. Return one corrected JSON object only. Error: " + last_error[:256]
                )
                if last_raw:
                    repair_conversation.append({"role": "assistant", "content": last_raw[:2048]})
            response = self.backend.generate(
                prompt=current_prompt,
                images=request.images,
                conversation=tuple(repair_conversation),
            )
            last_raw = json.dumps(response, ensure_ascii=False) if isinstance(response, Mapping) else str(response)
            try:
                payload, raw = self._payload(response)
                return PlannerDecision.from_mapping(
                    payload,
                    timestamp_ns=request.timestamp_ns,
                    expected_primitive=request.primitive,
                    raw_response=raw,
                )
            except (KeyError, TypeError, ValueError) as exc:
                last_error = str(exc)
        return PlannerDecision.invalid(
            timestamp_ns=request.timestamp_ns,
            primitive=request.primitive,
            raw_response=last_raw,
            reason=last_error,
        )

    def plan(self, request: PlannerRequest) -> PlannerDecision:
        decision = self._validated_generate(
            request=request,
            prompt=self.build_prompt(request),
        )
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
        prompt = self.build_prompt(request) + (
            "\nResolve the ambiguity using the short answer, re-inspect the images, "
            "and preserve the same latched primitive."
        )
        return self._validated_generate(
            request=request,
            prompt=prompt,
            conversation=conversation,
        )
