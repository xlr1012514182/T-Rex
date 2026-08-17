"""Canonical Revo 3 21-DoF representation.

The order is copied from BrainCo's ``revo3_retargeting`` branch.  Every
internal value is expressed in SI units (radians, radians/second, ampere).
SDK-specific degrees and milliampere conversions belong at the backend
boundary and nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Iterable, Optional

import numpy as np


JOINT_ORDER = (
    "little_MPR",
    "little_MCP",
    "little_PIP",
    "little_DIP",
    "ring_MPR",
    "ring_MCP",
    "ring_PIP",
    "ring_DIP",
    "middle_MPR",
    "middle_MCP",
    "middle_PIP",
    "middle_DIP",
    "index_MPR",
    "index_MCP",
    "index_PIP",
    "index_DIP",
    "thumb_MCP",
    "thumb_PIP",
    "thumb_DIP",
    "thumb_CMP",
    "thumb_CMR",
)
JOINT_COUNT = len(JOINT_ORDER)
JOINT_ORDER_HASH = hashlib.sha256("\n".join(JOINT_ORDER).encode("utf-8")).hexdigest()


def assert_joint_vector(
    value: Iterable[float] | np.ndarray,
    *,
    name: str,
    finite: bool = True,
    copy: bool = True,
) -> np.ndarray:
    """Return a canonical float32 ``[21]`` vector or raise ``ValueError``."""

    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (JOINT_COUNT,):
        raise ValueError(f"{name} must have shape ({JOINT_COUNT},), got {arr.shape}.")
    if finite and not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinity.")
    return arr.copy() if copy else arr


def _zeros() -> np.ndarray:
    return np.zeros(JOINT_COUNT, dtype=np.float32)


@dataclass(frozen=True)
class RevoState:
    """A time-stamped Revo observation in the canonical joint order."""

    timestamp_ns: int
    q_rad: np.ndarray
    dq_rad_s: np.ndarray = field(default_factory=_zeros)
    current_a: np.ndarray = field(default_factory=_zeros)
    status: np.ndarray = field(default_factory=lambda: np.zeros(JOINT_COUNT, dtype=np.int64))
    sequence: int = 0

    def __post_init__(self) -> None:
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative.")
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative.")
        object.__setattr__(self, "q_rad", assert_joint_vector(self.q_rad, name="q_rad"))
        object.__setattr__(
            self, "dq_rad_s", assert_joint_vector(self.dq_rad_s, name="dq_rad_s")
        )
        object.__setattr__(
            self, "current_a", assert_joint_vector(self.current_a, name="current_a")
        )
        status = np.asarray(self.status, dtype=np.int64)
        if status.shape != (JOINT_COUNT,):
            raise ValueError(f"status must have shape ({JOINT_COUNT},), got {status.shape}.")
        object.__setattr__(self, "status", status.copy())


@dataclass(frozen=True)
class RevoCommand:
    """One absolute 21-D joint target authorized for the Revo backend."""

    timestamp_ns: int
    q_target_rad: np.ndarray
    task_id: str
    task_version: int
    source_chunk_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative.")
        if not self.task_id:
            raise ValueError("task_id must be non-empty.")
        if self.task_version < 0:
            raise ValueError("task_version must be non-negative.")
        object.__setattr__(
            self,
            "q_target_rad",
            assert_joint_vector(self.q_target_rad, name="q_target_rad"),
        )
