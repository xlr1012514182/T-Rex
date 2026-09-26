"""Shape-safe T-Rex adapter plus a lightweight deterministic mock policy."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Protocol, runtime_checkable
import uuid

import numpy as np

from revo3_v1.revo.contracts import JOINT_COUNT

from .cache import CacheProtocolError, SlowFastCache
from .contracts import (
    ACTION_CHUNK,
    ActionChunk,
    InferenceMode,
    PolicyObservation,
    PolicyRequest,
)


def _validate_raw_chunk(value: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    expected = (ACTION_CHUNK, JOINT_COUNT)
    if arr.shape != expected:
        raise ValueError(f"{name} must return shape {expected}, got {arr.shape}.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} returned NaN or infinity.")
    return arr.copy()


@runtime_checkable
class TReXBackend(Protocol):
    """Backend implemented by either real T-Rex inference or the mock."""

    def slow_and_fast(self, observation: PolicyObservation) -> np.ndarray:
        ...

    def slow(self, observation: PolicyObservation) -> np.ndarray:
        ...

    def fast(
        self,
        observation: PolicyObservation,
        cached_chunk: np.ndarray,
        chunk_offset: int,
    ) -> np.ndarray:
        ...


@dataclass
class CallableTReXBackend:
    """Injects real server/client callables without importing GPU dependencies."""

    slow_and_fast_fn: Callable[[PolicyObservation], np.ndarray]
    slow_fn: Callable[[PolicyObservation], np.ndarray]
    fast_fn: Callable[[PolicyObservation, np.ndarray, int], np.ndarray]

    def slow_and_fast(self, observation: PolicyObservation) -> np.ndarray:
        return _validate_raw_chunk(self.slow_and_fast_fn(observation), name="slow_and_fast_fn")

    def slow(self, observation: PolicyObservation) -> np.ndarray:
        return _validate_raw_chunk(self.slow_fn(observation), name="slow_fn")

    def fast(
        self,
        observation: PolicyObservation,
        cached_chunk: np.ndarray,
        chunk_offset: int,
    ) -> np.ndarray:
        return _validate_raw_chunk(
            self.fast_fn(observation, cached_chunk, chunk_offset), name="fast_fn"
        )


class MockTReXBackend:
    """Deterministic absolute-q policy for plumbing tests, not task success."""

    # Positive flexion is a provisional simulation convention only.  Hardware
    # deployment must load a measured Revo joint convention manifest.
    _CLOSE_SYNERGY = np.asarray(
        [
            0.15, 0.80, 0.95, 0.75,
            0.10, 0.85, 1.00, 0.80,
            0.05, 0.85, 1.00, 0.80,
            0.05, 0.85, 1.00, 0.80,
            0.65, 0.80, 0.70, 0.40, 0.20,
        ],
        dtype=np.float32,
    )

    def __init__(self, close_delta_rad: float = 0.35) -> None:
        self.close_delta_rad = float(close_delta_rad)

    def _target(self, observation: PolicyObservation) -> np.ndarray:
        text = observation.instruction.lower()
        release = any(word in text for word in ("release", "open", "放开", "松开"))
        close = any(
            word in text
            for word in ("grasp", "hold", "lift", "pull", "bottle", "phone", "bag", "door")
        )
        if release:
            return observation.q_rad - self.close_delta_rad * self._CLOSE_SYNERGY
        if close:
            return observation.q_rad + self.close_delta_rad * self._CLOSE_SYNERGY
        return observation.q_rad.copy()

    def slow_and_fast(self, observation: PolicyObservation) -> np.ndarray:
        target = self._target(observation)
        alpha = np.linspace(1.0 / ACTION_CHUNK, 1.0, ACTION_CHUNK, dtype=np.float32)
        return observation.q_rad[None, :] + alpha[:, None] * (
            target - observation.q_rad
        )[None, :]

    def slow(self, observation: PolicyObservation) -> np.ndarray:
        return self.slow_and_fast(observation)

    def fast(
        self,
        observation: PolicyObservation,
        cached_chunk: np.ndarray,
        chunk_offset: int,
    ) -> np.ndarray:
        del chunk_offset
        refined = _validate_raw_chunk(cached_chunk, name="cached_chunk")
        # A high normalized normal force slightly backs off flexion.  This is
        # just a mock of T-Rex tactile continuation, not the CAIR residual.
        if observation.tactile_f6 is None:
            return refined
        normal_force = np.abs(observation.tactile_f6[:, 2])
        overload = max(float(normal_force.max()) - 1.0, 0.0)
        if overload:
            refined -= min(overload * 0.01, 0.02) * self._CLOSE_SYNERGY[None, :]
        return refined


class TReXRevoPolicyAdapter:
    """Enforces the main slow/fast protocol around any T-Rex backend."""

    def __init__(self, backend: TReXBackend, cache: SlowFastCache | None = None) -> None:
        self.backend = backend
        self.cache = SlowFastCache() if cache is None else cache

    def reset(self, reason: str) -> None:
        self.cache.clear(reason)
        backend_reset = getattr(self.backend, "reset", None)
        if callable(backend_reset):
            backend_reset()

    def infer(
        self,
        request: PolicyRequest,
        *,
        global_start_step: int,
        now_ns: int | None = None,
    ) -> ActionChunk:
        now = time.monotonic_ns() if now_ns is None else int(now_ns)
        cached_for_fast = None
        if request.mode is InferenceMode.NONE:
            raise ValueError("mode=none does not perform inference.")

        if request.mode is InferenceMode.SLOW_AND_FAST:
            raw = self.backend.slow_and_fast(request.observation)
        elif request.mode is InferenceMode.SLOW:
            raw = self.backend.slow(request.observation)
        elif request.mode is InferenceMode.FAST:
            cached = self.cache.validate_fast(request, now_ns=now)
            cached_for_fast = cached
            raw = self.backend.fast(
                request.observation, cached.q_target_rad, request.chunk_offset
            )
        else:
            raise ValueError(f"unsupported inference mode: {request.mode}")

        raw = _validate_raw_chunk(raw, name=f"backend.{request.mode.value}")
        chunk = ActionChunk(
            q_target_rad=raw,
            start_step=(
                cached_for_fast.start_step
                if cached_for_fast is not None
                else int(global_start_step)
            ),
            generated_ns=now,
            observation_timestamp_ns=request.observation.timestamp_ns,
            task_key=request.observation.task_key,
            mode=request.mode,
            chunk_id=(
                cached_for_fast.chunk_id
                if cached_for_fast is not None
                else uuid.uuid4().hex
            ),
        )
        if request.mode in (InferenceMode.SLOW, InferenceMode.SLOW_AND_FAST):
            self.cache.begin(request, chunk, now_ns=now)
        else:
            self.cache.update_fast(request, chunk)
        return chunk
