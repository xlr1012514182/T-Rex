"""Causal bridge from streaming EMG outputs to Task Executive events.

The streaming classifier owns confidence/margin/dwell gating.  This bridge is
deliberately small: an actionable grasp/release reaches the Task Executive only
when the classifier emitted an explicit edge event.  A high class probability
during the dwell interval is therefore never mistaken for a command.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from revo3_v1.emg.primitives import EMGPrimitive, normalize_emg_primitive
from revo3_v1.executive import EmgEvent, EmgIntent


class StreamingEMGSource(Protocol):
    """Structural interface implemented by ``StreamingEMGClassifier``."""

    def push_many(
        self,
        samples: np.ndarray,
        sample_timestamps_ns: Sequence[int] | np.ndarray,
        signal_quality: float = 1.0,
    ) -> list[dict[str, object]]:
        ...

    def reset(self, active: bool = False) -> None:
        ...


def _unit_interval(value: object, *, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"streaming EMG {name} must be finite and in [0,1]")
    return number


def _passive_intent(value: object) -> EmgIntent:
    """Never turn a class observation without an edge into a command."""

    primitive = normalize_emg_primitive(value)
    if primitive is EMGPrimitive.REST:
        return EmgIntent.REST
    if primitive is EMGPrimitive.BAD_SIGNAL:
        return EmgIntent.BAD_SIGNAL
    return EmgIntent.UNKNOWN


def streaming_result_to_emg_event(result: Mapping[str, object]) -> EmgEvent:
    """Validate one classifier output and preserve event/passive semantics."""

    timestamp_ns = int(result["timestamp_ns"])
    confidence = _unit_interval(result.get("confidence", 0.0), name="confidence")
    margin = _unit_interval(result.get("margin", 0.0), name="margin")
    quality = _unit_interval(result.get("signal_quality", 0.0), name="signal_quality")
    payload = result.get("event")
    if payload is None:
        return EmgEvent(
            _passive_intent(result.get("primitive")),
            timestamp_ns,
            confidence=confidence,
            margin=margin,
            signal_quality=quality,
        )
    if not isinstance(payload, Mapping):
        raise ValueError("streaming EMG event must be a mapping or null")
    event_timestamp_ns = int(payload.get("timestamp_ns", timestamp_ns))
    if event_timestamp_ns != timestamp_ns:
        raise ValueError("streaming EMG observation/event timestamps must match")
    primitive = normalize_emg_primitive(payload.get("primitive"))
    event_type = str(payload.get("type", ""))
    if primitive is EMGPrimitive.RELEASE:
        if event_type != "ReleaseEvent":
            raise ValueError("RELEASE must be carried by ReleaseEvent")
        intent = EmgIntent.RELEASE
    elif primitive is not None and primitive.starts_task:
        if event_type != "StartIntentEvent":
            raise ValueError("a grasp primitive must be carried by StartIntentEvent")
        intent = EmgIntent(primitive.value)
    else:
        raise ValueError("only grasp StartIntentEvent or RELEASE may be actionable")
    event_id = str(payload.get("event_id", "")).strip()
    if not event_id:
        raise ValueError("actionable streaming EMG event requires event_id")
    return EmgEvent(
        intent,
        event_timestamp_ns,
        confidence=_unit_interval(payload.get("confidence", confidence), name="event confidence"),
        event_id=event_id,
        margin=_unit_interval(payload.get("margin", margin), name="event margin"),
        signal_quality=_unit_interval(
            payload.get("signal_quality", quality), name="event signal_quality"
        ),
    )


class StreamingEMGEventBridge:
    """Queue debounced EMG edges for consumption by the 30 Hz coordinator."""

    def __init__(self, source: StreamingEMGSource) -> None:
        self.source = source
        self._pending: deque[EmgEvent] = deque()
        self._latest_passive = EmgEvent(
            EmgIntent.UNKNOWN,
            0,
            confidence=0.0,
            margin=0.0,
            signal_quality=0.0,
        )

    @property
    def pending_event_count(self) -> int:
        return len(self._pending)

    def reset(self, *, active: bool = False) -> None:
        self.source.reset(active=active)
        self._pending.clear()
        self._latest_passive = EmgEvent(
            EmgIntent.UNKNOWN,
            0,
            confidence=0.0,
            margin=0.0,
            signal_quality=0.0,
        )

    def push_many(
        self,
        samples: np.ndarray,
        sample_timestamps_ns: Sequence[int] | np.ndarray,
        *,
        signal_quality: float = 1.0,
    ) -> tuple[EmgEvent, ...]:
        converted: list[EmgEvent] = []
        for raw in self.source.push_many(
            samples,
            sample_timestamps_ns,
            signal_quality=signal_quality,
        ):
            event = streaming_result_to_emg_event(raw)
            converted.append(event)
            if event.intent.starts_task or event.intent is EmgIntent.RELEASE:
                if self._pending and event.timestamp_ns < self._pending[-1].timestamp_ns:
                    raise ValueError("actionable streaming EMG events must be chronological")
                self._pending.append(event)
            else:
                if event.timestamp_ns >= self._latest_passive.timestamp_ns:
                    self._latest_passive = event
        return tuple(converted)

    def event_for_tick(self, *, now_ns: int) -> EmgEvent:
        """Consume at most one causal edge; otherwise expose passive evidence."""

        now = int(now_ns)
        if now < 0:
            raise ValueError("now_ns must be non-negative")
        if self._pending and self._pending[0].timestamp_ns <= now:
            return self._pending.popleft()
        if self._latest_passive.timestamp_ns <= now:
            return self._latest_passive
        return EmgEvent(
            EmgIntent.UNKNOWN,
            now,
            confidence=0.0,
            margin=0.0,
            signal_quality=0.0,
        )


__all__ = [
    "StreamingEMGEventBridge",
    "StreamingEMGSource",
    "streaming_result_to_emg_event",
]
