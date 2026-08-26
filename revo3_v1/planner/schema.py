"""Strict, auditable contracts for the Revo 3 RGB + primitive planner."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import hypot, isfinite
from typing import Any, Mapping, Optional, Sequence, Tuple

from revo3_v1.emg.primitives import EMGPrimitive, START_PRIMITIVES, normalize_emg_primitive


SCHEMA_VERSION = "planner_v1"


class SupportedTask(str, Enum):
    BOTTLE = "bottle"
    PHONE = "phone"
    PLASTIC_BAG = "plastic_bag"
    REFRIGERATOR_DOOR = "refrigerator_door"


class PlannerStatus(str, Enum):
    READY = "READY"
    ASK_CLARIFY = "ASK_CLARIFY"
    NOT_READY = "NOT_READY"
    INVALID = "INVALID"


TASK_GRASP_PRIMITIVES = {
    SupportedTask.BOTTLE: EMGPrimitive.POWER_GRASP,
    SupportedTask.PHONE: EMGPrimitive.PRECISION_GRASP,
    SupportedTask.PLASTIC_BAG: EMGPrimitive.PRECISION_GRASP,
    SupportedTask.REFRIGERATOR_DOOR: EMGPrimitive.LATERAL_GRASP,
}


DEFAULT_INSTRUCTIONS = {
    SupportedTask.BOTTLE: "Grasp the centered bottle using a power grasp and hold it securely.",
    SupportedTask.PHONE: "Grasp the centered phone using a precision grasp and hold it securely.",
    SupportedTask.PLASTIC_BAG: "Grasp the centered plastic bag handles using a precision grasp and hold them securely.",
    SupportedTask.REFRIGERATOR_DOOR: "Grasp the centered refrigerator door handle using a lateral grasp and hold it securely.",
}

_TASK_ALIASES = {
    "bottle": SupportedTask.BOTTLE,
    "water_bottle": SupportedTask.BOTTLE,
    "phone": SupportedTask.PHONE,
    "mobile_phone": SupportedTask.PHONE,
    "cell_phone": SupportedTask.PHONE,
    "plastic_bag": SupportedTask.PLASTIC_BAG,
    "bag": SupportedTask.PLASTIC_BAG,
    "refrigerator_door": SupportedTask.REFRIGERATOR_DOOR,
    "fridge_door": SupportedTask.REFRIGERATOR_DOOR,
}


def normalize_task(value: Any) -> Optional[SupportedTask]:
    if value is None:
        return None
    if isinstance(value, SupportedTask):
        return value
    return _TASK_ALIASES.get(str(value).strip().lower().replace(" ", "_"))


@dataclass(frozen=True)
class NormalizedBBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        values = (self.x_min, self.y_min, self.x_max, self.y_max)
        if not all(isfinite(value) for value in values):
            raise ValueError("bbox values must be finite")
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError("bbox must use normalized coordinates in [0, 1]")
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("bbox must have positive width and height")

    @classmethod
    def from_xyxy(cls, value: Any) -> "NormalizedBBox":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(*(float(value[key]) for key in ("x_min", "y_min", "x_max", "y_max")))
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 4:
            return cls(*(float(item) for item in value))
        raise ValueError("bbox must be a four-value normalized xyxy region")

    @classmethod
    def from_cxcywh(cls, value: Any) -> "NormalizedBBox":
        if isinstance(value, Mapping):
            cx, cy, width, height = (
                float(value[key]) for key in ("cx", "cy", "width", "height")
            )
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 4:
            cx, cy, width, height = (float(item) for item in value)
        else:
            raise ValueError("target_region must be normalized [cx, cy, width, height]")
        return cls(cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)

    @classmethod
    def from_value(cls, value: Any) -> "NormalizedBBox":
        return cls.from_xyxy(value)

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)

    @property
    def center_distance(self) -> float:
        cx, cy = self.center
        return hypot(cx - 0.5, cy - 0.5)

    def as_xyxy(self) -> Tuple[float, float, float, float]:
        return self.x_min, self.y_min, self.x_max, self.y_max

    def as_cxcywh(self) -> Tuple[float, float, float, float]:
        cx, cy = self.center
        return cx, cy, self.width, self.height


@dataclass(frozen=True)
class Ambiguity:
    ambiguous: bool = False
    reason: str = ""
    candidates: Tuple[SupportedTask, ...] = ()
    question: str = ""

    @classmethod
    def from_value(cls, value: Any) -> "Ambiguity":
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            return cls(ambiguous=value)
        if not isinstance(value, Mapping):
            return cls(ambiguous=bool(value))
        candidates = tuple(
            task for task in (normalize_task(item) for item in value.get("candidates", ()))
            if task is not None
        )
        return cls(
            ambiguous=bool(value.get("ambiguous", False)),
            reason=str(value.get("reason", ""))[:256],
            candidates=candidates,
            question=str(value.get("question", ""))[:256],
        )


@dataclass(frozen=True)
class PlannerRequest:
    """One planner call over a latched primitive and aligned camera context.

    Mainline callers use :meth:`from_aligned_views`, which enforces three
    recent full frames plus the current center crop.  ``images`` remains a
    compatibility surface for old smoke fixtures only.
    """

    emg_action: str
    images: Tuple[Any, ...]
    timestamp_ns: int
    conversation_id: str = ""
    clarification_answer: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    full_view_history: Tuple[Any, ...] = ()
    center_view: Any = None
    frame_timestamps_ns: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        primitive = normalize_emg_primitive(self.emg_action)
        if primitive not in START_PRIMITIVES:
            raise ValueError("V1 planner accepts only a latched grasp primitive")
        object.__setattr__(self, "emg_action", primitive.value)
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        if self.full_view_history or self.center_view is not None or self.frame_timestamps_ns:
            if len(self.full_view_history) != 3 or self.center_view is None:
                raise ValueError("mainline planner input requires 3 full frames and 1 center view")
            if len(self.frame_timestamps_ns) != 4:
                raise ValueError("aligned planner input requires four frame timestamps")
            if any(value < 0 for value in self.frame_timestamps_ns):
                raise ValueError("frame timestamps must be non-negative")
            if tuple(sorted(self.frame_timestamps_ns[:3])) != self.frame_timestamps_ns[:3]:
                raise ValueError("full-view timestamps must be monotonic")
            if self.frame_timestamps_ns[-1] != self.timestamp_ns:
                raise ValueError("center view timestamp must equal request timestamp")
            if self.frame_timestamps_ns[2] != self.timestamp_ns:
                raise ValueError("latest full and center views must share one timestamp")
            if any(value > self.timestamp_ns for value in self.frame_timestamps_ns):
                raise ValueError("planner frames cannot come from the future")
            object.__setattr__(self, "images", self.full_view_history + (self.center_view,))
        elif not self.images:
            raise ValueError("planner request requires RGB input")

    @property
    def primitive(self) -> EMGPrimitive:
        value = normalize_emg_primitive(self.emg_action)
        assert value is not None
        return value

    @property
    def uses_aligned_views(self) -> bool:
        return bool(self.full_view_history)

    @classmethod
    def from_aligned_views(
        cls,
        *,
        primitive: str | EMGPrimitive,
        full_view_history: Sequence[Any],
        center_view: Any,
        full_view_timestamps_ns: Sequence[int],
        center_timestamp_ns: int,
        metadata: Mapping[str, Any] | None = None,
        clarification_answer: str = "",
    ) -> "PlannerRequest":
        full = tuple(full_view_history)
        times = tuple(int(value) for value in full_view_timestamps_ns) + (int(center_timestamp_ns),)
        return cls(
            emg_action=str(primitive.value if isinstance(primitive, EMGPrimitive) else primitive),
            images=(),
            timestamp_ns=int(center_timestamp_ns),
            clarification_answer=clarification_answer,
            metadata=dict(metadata or {}),
            full_view_history=full,
            center_view=center_view,
            frame_timestamps_ns=times,
        )


@dataclass(frozen=True)
class PlannerDecision:
    status: PlannerStatus
    task: Optional[SupportedTask]
    bbox: Optional[NormalizedBBox]
    area: float
    confidence: float
    target_present: bool
    near_ready: bool
    compatible: bool
    ambiguity: Ambiguity
    instruction: str
    timestamp_ns: int
    primitive: Optional[EMGPrimitive] = None
    target_part: str = ""
    grasp_style: str = ""
    center_ready: bool = False
    ready_frame_count: int = 0
    reason_code: str = ""
    schema_version: str = SCHEMA_VERSION
    raw_response: str = ""
    # ``timestamp_ns`` is the source/capture timestamp of the current RGB
    # context.  ``produced_at_ns`` is stamped by the asynchronous runtime when
    # Qwen finishes.  Keeping both prevents a valid slow result from looking
    # stale merely because its source image predates model completion.
    produced_at_ns: Optional[int] = None

    def __post_init__(self) -> None:
        produced_at_ns = self.timestamp_ns if self.produced_at_ns is None else int(self.produced_at_ns)
        if self.timestamp_ns < 0 or produced_at_ns < self.timestamp_ns:
            raise ValueError("planner timestamps must be causal and non-negative")
        object.__setattr__(self, "produced_at_ns", produced_at_ns)
        if not 0 <= self.confidence <= 1 or not 0 <= self.area <= 1:
            raise ValueError("planner confidence and area must be in [0, 1]")
        if self.ready_frame_count < 0 or self.ready_frame_count > 3:
            raise ValueError("ready_frame_count must be in [0, 3]")
        if self.bbox is not None and abs(self.area - self.bbox.area) > 0.02:
            raise ValueError("reported area is inconsistent with target_region")
        if len(self.instruction.split()) > 25:
            raise ValueError("planner instruction exceeds the 25-token V1 bound")
        if self.status == PlannerStatus.READY:
            if self.primitive not in START_PRIMITIVES:
                raise ValueError("READY decision requires the input grasp primitive")
            if self.task is None or self.bbox is None or not self.instruction:
                raise ValueError("READY decision requires target, target_region, and instruction")
            if self.ambiguity.ambiguous:
                raise ValueError("READY decision cannot be ambiguous")
            if self.primitive != TASK_GRASP_PRIMITIVES[self.task]:
                raise ValueError(
                    "READY task/primitive pair is outside the frozen V1 task mapping"
                )

    @classmethod
    def invalid(
        cls, *, timestamp_ns: int, primitive: EMGPrimitive, raw_response: str, reason: str
    ) -> "PlannerDecision":
        return cls(
            status=PlannerStatus.INVALID,
            task=None,
            bbox=None,
            area=0,
            confidence=0,
            target_present=False,
            near_ready=False,
            compatible=False,
            ambiguity=Ambiguity(False, reason=reason[:256]),
            instruction="",
            timestamp_ns=timestamp_ns,
            primitive=primitive,
            reason_code="INVALID_RESPONSE",
            raw_response=raw_response,
        )

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        timestamp_ns: int,
        expected_primitive: Optional[EMGPrimitive] = None,
        raw_response: str = "",
    ) -> "PlannerDecision":
        schema_version = str(payload.get("schema_version", ""))
        if schema_version != SCHEMA_VERSION:
            raise ValueError(f"planner schema mismatch: expected {SCHEMA_VERSION}, got {schema_version}")
        primitive = normalize_emg_primitive(payload.get("primitive"))
        if primitive not in START_PRIMITIVES:
            raise ValueError("planner output requires a valid grasp primitive")
        if expected_primitive is not None and primitive != expected_primitive:
            raise ValueError("planner primitive does not match the latched EMG primitive")
        invariant_primitive = primitive if expected_primitive is None else expected_primitive
        task = normalize_task(payload.get("target_category", payload.get("task")))
        region = payload.get("target_region")
        legacy_bbox = payload.get("bbox")
        # The canonical NOT_READY example uses a zero-area sentinel.  It means
        # "no grounded target" and must not be parsed as an invalid rectangle.
        if isinstance(region, Sequence) and not isinstance(region, (str, bytes)):
            region_values = tuple(region)
            if len(region_values) == 4 and (
                float(region_values[2]) <= 0 or float(region_values[3]) <= 0
            ):
                region = None
        bbox = (
            NormalizedBBox.from_cxcywh(region)
            if region not in (None, [], {})
            else None if legacy_bbox in (None, [], {}) else NormalizedBBox.from_xyxy(legacy_bbox)
        )
        area = bbox.area if bbox is not None else float(payload.get("area", 0))
        ambiguity_value = payload.get("ambiguity", payload.get("ambiguous", False))
        ambiguity = Ambiguity.from_value(ambiguity_value)
        if isinstance(ambiguity_value, bool):
            ambiguity = Ambiguity(
                ambiguity_value,
                str(payload.get("ambiguity_reason", ""))[:256],
                (),
                str(payload.get("clarification_question", ""))[:256],
            )
        confidence = float(payload.get("confidence", 0))
        target_present = bool(payload.get("target_present", bbox is not None))
        near_ready = bool(payload.get("near_ready", False))
        center_ready = bool(payload.get("center_ready", False))
        compatible = bool(payload.get("compatible", False))
        instruction = " ".join(str(payload.get("instruction", "")).split())
        grasp_style = str(payload.get("grasp_style", "")).strip().lower()[:64]
        expected_style = {
            EMGPrimitive.POWER_GRASP: "power",
            EMGPrimitive.PRECISION_GRASP: "precision",
            EMGPrimitive.LATERAL_GRASP: "lateral",
        }[invariant_primitive]
        if grasp_style != expected_style:
            raise ValueError("planner grasp_style does not preserve the latched primitive")
        if instruction and f"{expected_style} grasp" not in instruction.lower():
            raise ValueError("planner instruction does not preserve the latched primitive")
        ready_count = int(payload.get("ready_frame_count", 0))
        explicit_status = str(payload.get("status", "")).upper()
        if bool(payload.get("ask_clarify", False)) or ambiguity.ambiguous:
            status = PlannerStatus.ASK_CLARIFY
        elif explicit_status in PlannerStatus.__members__:
            status = PlannerStatus[explicit_status]
        elif all((task, bbox, target_present, near_ready, center_ready, compatible, instruction)):
            status = PlannerStatus.READY
        else:
            status = PlannerStatus.NOT_READY
        return cls(
            status=status,
            task=task,
            bbox=bbox,
            area=area,
            confidence=confidence,
            target_present=target_present,
            near_ready=near_ready,
            compatible=compatible,
            ambiguity=ambiguity,
            instruction=instruction,
            timestamp_ns=timestamp_ns,
            primitive=primitive,
            target_part=str(payload.get("target_part", ""))[:64],
            grasp_style=grasp_style,
            center_ready=center_ready,
            ready_frame_count=ready_count,
            reason_code=str(payload.get("reason_code", ""))[:64],
            schema_version=schema_version,
            raw_response=raw_response,
        )
