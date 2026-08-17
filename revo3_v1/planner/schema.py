"""Strict data contracts for the Revo 3 instruction planner.

The planner deliberately emits a small, auditable schema instead of free-form
chain-of-thought.  Bounding boxes are normalized ``xyxy`` coordinates in the
rectified RGB image.  No detector or tracker is assumed by this module; the VLM
is the only semantic visual component in the V1 path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import hypot, isfinite
from typing import Any, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "revo3_planner_v1"


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


DEFAULT_INSTRUCTIONS = {
    SupportedTask.BOTTLE: (
        "Grasp the centered bottle with a power grasp and hold it securely."
    ),
    SupportedTask.PHONE: (
        "Grasp the centered phone with a precision grasp and hold it securely."
    ),
    SupportedTask.PLASTIC_BAG: (
        "Grasp the handles of the centered plastic bag and lift it."
    ),
    SupportedTask.REFRIGERATOR_DOOR: (
        "Grasp the refrigerator door handle and pull the door open."
    ),
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
    """Normalize an allow-listed task name and reject everything else."""

    if value is None:
        return None
    if isinstance(value, SupportedTask):
        return value
    return _TASK_ALIASES.get(str(value).strip().lower().replace(" ", "_"))


@dataclass(frozen=True)
class NormalizedBBox:
    """Normalized ``xyxy`` box in a rectified RGB image."""

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
    def from_value(cls, value: Any) -> "NormalizedBBox":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            if {"x_min", "y_min", "x_max", "y_max"}.issubset(value):
                return cls(
                    float(value["x_min"]),
                    float(value["y_min"]),
                    float(value["x_max"]),
                    float(value["y_max"]),
                )
            if {"cx", "cy", "width", "height"}.issubset(value):
                cx = float(value["cx"])
                cy = float(value["cy"])
                width = float(value["width"])
                height = float(value["height"])
                return cls(
                    cx - width / 2.0,
                    cy - height / 2.0,
                    cx + width / 2.0,
                    cy + height / 2.0,
                )
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) != 4:
                raise ValueError("bbox sequence must have four values")
            return cls(*(float(item) for item in value))
        raise ValueError("bbox must be xyxy sequence or coordinate mapping")

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
        return (
            (self.x_min + self.x_max) / 2.0,
            (self.y_min + self.y_max) / 2.0,
        )

    @property
    def center_distance(self) -> float:
        cx, cy = self.center
        return hypot(cx - 0.5, cy - 0.5)

    def as_xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x_min, self.y_min, self.x_max, self.y_max)


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
            task
            for task in (normalize_task(item) for item in value.get("candidates", ()))
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
    """One task-planning request.

    ``images`` may contain PIL images, arrays, local paths, or URLs supported by
    the configured backend.  The planner itself never mutates them.
    """

    emg_action: str
    images: Tuple[Any, ...]
    timestamp_ns: int
    conversation_id: str = ""
    clarification_answer: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.images:
            raise ValueError("planner request requires at least one RGB image")
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        if self.emg_action.strip().upper() != "CLOSE":
            raise ValueError("V1 planner accepts only a latched CLOSE action")


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
    schema_version: str = SCHEMA_VERSION
    raw_response: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("planner confidence must be in [0, 1]")
        if not 0.0 <= self.area <= 1.0:
            raise ValueError("planner area must be in [0, 1]")
        if self.bbox is not None and abs(self.area - self.bbox.area) > 0.02:
            raise ValueError("reported area is inconsistent with bbox")
        if self.status == PlannerStatus.READY:
            if self.task is None or self.bbox is None or not self.instruction:
                raise ValueError("READY decision requires task, bbox, and instruction")
            if self.ambiguity.ambiguous:
                raise ValueError("READY decision cannot be ambiguous")

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        timestamp_ns: int,
        raw_response: str = "",
    ) -> "PlannerDecision":
        schema_version = str(payload.get("schema_version", SCHEMA_VERSION))
        if schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"planner schema mismatch: expected {SCHEMA_VERSION}, got {schema_version}"
            )
        task = normalize_task(payload.get("task", payload.get("target_category")))
        bbox_value = payload.get("bbox", payload.get("target_bbox"))
        bbox = None if bbox_value in (None, [], {}) else NormalizedBBox.from_value(bbox_value)
        area = bbox.area if bbox is not None else float(payload.get("area", 0.0))
        ambiguity = Ambiguity.from_value(payload.get("ambiguity", False))
        confidence = float(payload.get("confidence", 0.0))
        target_present = bool(payload.get("target_present", bbox is not None))
        near_ready = bool(payload.get("near_ready", False))
        compatible = bool(payload.get("compatible", False))
        instruction = str(payload.get("instruction", "")).strip()[:512]

        explicit_status = str(payload.get("status", "")).strip().upper()
        if ambiguity.ambiguous:
            status = PlannerStatus.ASK_CLARIFY
        elif explicit_status in PlannerStatus.__members__:
            status = PlannerStatus[explicit_status]
        elif task and bbox and target_present and near_ready and compatible and instruction:
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
            schema_version=schema_version,
            raw_response=raw_response,
        )
