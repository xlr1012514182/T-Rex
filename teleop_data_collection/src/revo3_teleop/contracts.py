"""Stable, hardware-independent collection contracts.

All timestamps used for causal alignment are integer nanoseconds in one
workstation monotonic clock domain.  Device and receive timestamps are kept
alongside that mapped capture time so clock quality can be audited later.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional

import numpy as np


def _non_empty(value: str, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


def _target(value: object, *, name: str, required: bool) -> Optional[np.ndarray]:
    if value is None:
        if required:
            raise ValueError(f"{name} is required")
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1 or array.size < 1:
        raise ValueError(f"{name} must be a non-empty 1-D vector")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array.copy()


@dataclass(frozen=True)
class SampleHeader:
    """Identity and clock evidence attached to one native-rate sample."""

    source_id: str
    sequence: int
    capture_timestamp_ns: int
    receive_timestamp_ns: int
    clock_domain: str = "workstation_monotonic"
    device_timestamp_ns: Optional[int] = None
    valid: bool = True
    dropped_since_previous: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _non_empty(self.source_id, name="source_id"))
        object.__setattr__(self, "clock_domain", _non_empty(self.clock_domain, name="clock_domain"))
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        if self.capture_timestamp_ns < 0 or self.receive_timestamp_ns < 0:
            raise ValueError("capture/receive timestamps must be non-negative")
        if self.receive_timestamp_ns < self.capture_timestamp_ns:
            raise ValueError("receive_timestamp_ns cannot precede mapped capture time")
        if self.device_timestamp_ns is not None and self.device_timestamp_ns < 0:
            raise ValueError("device_timestamp_ns must be non-negative when present")
        if self.dropped_since_previous < 0:
            raise ValueError("dropped_since_previous must be non-negative")


@dataclass(frozen=True)
class NativeSample:
    """One native-rate numeric payload.

    Payload values are numeric NumPy-compatible arrays.  Textual schema and
    calibration metadata belong in the episode manifest, not in every sample.
    Keeping the payload numeric allows loading with ``allow_pickle=False``.
    """

    header: SampleHeader
    payload: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        if not self.payload:
            raise ValueError("sample payload must not be empty")
        frozen: dict[str, np.ndarray] = {}
        for key, value in self.payload.items():
            name = _non_empty(str(key), name="payload key")
            if not name.replace("_", "").isalnum():
                raise ValueError(f"payload key is not portable: {name!r}")
            array = np.asarray(value)
            if array.dtype.kind not in "biufc":
                raise TypeError(f"payload {name!r} must be numeric, got {array.dtype}")
            if array.dtype.kind in "fc" and not np.isfinite(array).all():
                raise ValueError(f"payload {name!r} contains NaN or infinity")
            frozen[name] = array.copy()
        object.__setattr__(self, "payload", MappingProxyType(frozen))


@dataclass(frozen=True)
class CommandReceipt:
    """Evidence at the actual controller-write boundary.

    ``exact_sent_target`` is the only field eligible to become a behavior-
    cloning action label.  A requested or authorized-but-not-written target
    is retained for diagnostics but is never supervision.
    """

    request_id: str
    component: str
    accepted: bool
    requested_target: np.ndarray
    decision_timestamp_ns: int
    authorized_target: Optional[np.ndarray] = None
    exact_sent_target: Optional[np.ndarray] = None
    write_timestamp_ns: Optional[int] = None
    controller_sequence: Optional[int] = None
    clipped: bool = False
    reason: str = ""
    unit: str = "rad"
    joint_order_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _non_empty(self.request_id, name="request_id"))
        object.__setattr__(self, "component", _non_empty(self.component, name="component"))
        object.__setattr__(self, "unit", _non_empty(self.unit, name="unit"))
        if self.decision_timestamp_ns < 0:
            raise ValueError("decision_timestamp_ns must be non-negative")
        if self.write_timestamp_ns is not None and self.write_timestamp_ns < self.decision_timestamp_ns:
            raise ValueError("write_timestamp_ns cannot precede decision_timestamp_ns")
        if self.controller_sequence is not None and self.controller_sequence < 0:
            raise ValueError("controller_sequence must be non-negative")
        requested = _target(self.requested_target, name="requested_target", required=True)
        authorized = _target(self.authorized_target, name="authorized_target", required=False)
        sent = _target(self.exact_sent_target, name="exact_sent_target", required=self.accepted)
        assert requested is not None
        for name, value in (("authorized_target", authorized), ("exact_sent_target", sent)):
            if value is not None and value.shape != requested.shape:
                raise ValueError(f"{name} shape must match requested_target")
        if self.accepted and self.write_timestamp_ns is None:
            raise ValueError("accepted command requires write_timestamp_ns")
        if self.accepted and self.controller_sequence is None:
            raise ValueError("accepted command requires controller_sequence")
        if not self.accepted and sent is not None:
            raise ValueError("rejected command cannot have exact_sent_target")
        object.__setattr__(self, "requested_target", requested)
        object.__setattr__(self, "authorized_target", authorized)
        object.__setattr__(self, "exact_sent_target", sent)


@dataclass(frozen=True)
class StreamReference:
    stream: str
    source_id: str
    clock_domain: str
    sequence: int
    capture_timestamp_ns: int
    age_ns: int
    relative_path: str
    row_index: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "stream", _non_empty(self.stream, name="stream"))
        object.__setattr__(self, "source_id", _non_empty(self.source_id, name="source_id"))
        object.__setattr__(self, "clock_domain", _non_empty(self.clock_domain, name="clock_domain"))
        object.__setattr__(self, "relative_path", _non_empty(self.relative_path, name="relative_path"))
        if min(self.sequence, self.capture_timestamp_ns, self.age_ns, self.row_index) < 0:
            raise ValueError("stream reference indices/timestamps must be non-negative")


@dataclass(frozen=True)
class CausalAnchor:
    anchor_index: int
    timestamp_ns: int
    streams: Mapping[str, StreamReference]
    hand_command_request_id: str

    def __post_init__(self) -> None:
        if self.anchor_index < 0 or self.timestamp_ns < 0:
            raise ValueError("anchor index/timestamp must be non-negative")
        if not self.streams:
            raise ValueError("anchor must reference at least one stream")
        object.__setattr__(
            self,
            "hand_command_request_id",
            _non_empty(self.hand_command_request_id, name="hand_command_request_id"),
        )
        copied = dict(self.streams)
        for name, reference in copied.items():
            if name != reference.stream:
                raise ValueError("stream mapping key and reference.stream disagree")
            if reference.capture_timestamp_ns > self.timestamp_ns:
                raise ValueError("causal anchor cannot reference a future sample")
        object.__setattr__(self, "streams", MappingProxyType(copied))
