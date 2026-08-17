"""Streaming EMG inference and debounced Start/Release intent events."""

from __future__ import annotations

from dataclasses import dataclass
import uuid
from typing import Dict, List, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class IntentGateConfig:
    close_threshold: float = 0.80
    open_threshold: float = 0.90
    close_dwell_ms: int = 300
    open_dwell_ms: int = 500
    min_signal_quality: float = 0.80

    def validate(self) -> None:
        if not 0.5 <= self.close_threshold <= 1.0:
            raise ValueError("close_threshold must be in [0.5, 1]")
        if not 0.5 <= self.open_threshold <= 1.0:
            raise ValueError("open_threshold must be in [0.5, 1]")
        if self.close_dwell_ms < 0 or self.open_dwell_ms < 0:
            raise ValueError("dwell times cannot be negative")


@dataclass(frozen=True)
class IntentEvent:
    event_type: str
    primitive: str
    confidence: float
    signal_quality: float
    timestamp_ns: int
    event_id: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "type": self.event_type,
            "primitive": self.primitive,
            "confidence": self.confidence,
            "signal_quality": self.signal_quality,
            "timestamp_ns": self.timestamp_ns,
            "event_id": self.event_id,
        }


class BinaryIntentGate:
    """Convert stable OPEN/CLOSE probabilities into edge-triggered events.

    CLOSE latches an active task and is not repeatedly emitted.  OPEN only
    produces ReleaseEvent while a task is active.  Ambiguous or poor-quality
    windows reset pending dwell but never release an already active task.
    """

    def __init__(self, config: IntentGateConfig | None = None) -> None:
        self.config = config or IntentGateConfig()
        self.config.validate()
        self.active = False
        self._close_since_ns: Optional[int] = None
        self._open_since_ns: Optional[int] = None
        self._last_timestamp_ns: Optional[int] = None

    def reset(self, active: bool = False) -> None:
        self.active = bool(active)
        self._close_since_ns = None
        self._open_since_ns = None
        self._last_timestamp_ns = None

    def update(self, probability_close: float, timestamp_ns: int, signal_quality: float = 1.0) -> Optional[IntentEvent]:
        p_close = float(probability_close)
        timestamp_ns = int(timestamp_ns)
        quality = float(signal_quality)
        if not 0.0 <= p_close <= 1.0:
            raise ValueError("probability_close must be in [0, 1]")
        if self._last_timestamp_ns is not None and timestamp_ns < self._last_timestamp_ns:
            raise ValueError("EMG event timestamps must be monotonic")
        self._last_timestamp_ns = timestamp_ns
        if quality < self.config.min_signal_quality:
            self._close_since_ns = None
            self._open_since_ns = None
            return None

        p_open = 1.0 - p_close
        if not self.active and p_close >= self.config.close_threshold:
            self._open_since_ns = None
            if self._close_since_ns is None:
                self._close_since_ns = timestamp_ns
            dwell_ns = self.config.close_dwell_ms * 1_000_000
            if timestamp_ns - self._close_since_ns >= dwell_ns:
                self.active = True
                self._close_since_ns = None
                return IntentEvent(
                    event_type="StartIntentEvent",
                    primitive="CLOSE",
                    confidence=p_close,
                    signal_quality=quality,
                    timestamp_ns=timestamp_ns,
                    event_id=str(uuid.uuid4()),
                )
        elif self.active and p_open >= self.config.open_threshold:
            self._close_since_ns = None
            if self._open_since_ns is None:
                self._open_since_ns = timestamp_ns
            dwell_ns = self.config.open_dwell_ms * 1_000_000
            if timestamp_ns - self._open_since_ns >= dwell_ns:
                self.active = False
                self._open_since_ns = None
                return IntentEvent(
                    event_type="ReleaseEvent",
                    primitive="OPEN",
                    confidence=p_open,
                    signal_quality=quality,
                    timestamp_ns=timestamp_ns,
                    event_id=str(uuid.uuid4()),
                )
        else:
            self._close_since_ns = None
            self._open_since_ns = None
        return None


class StreamingEMGClassifier:
    """Timestamped ring-buffer inference around a trained PyTorch model."""

    def __init__(
        self,
        model,
        normalization: Mapping[str, object],
        sample_rate_hz: int,
        window_samples: int,
        stride_samples: int,
        gate: BinaryIntentGate | None = None,
        device: str = "cpu",
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise ImportError("PyTorch is required for streaming EMG inference") from exc
        self._torch = torch
        self.model = model.to(device).eval()
        self.device = device
        self.sample_rate_hz = int(sample_rate_hz)
        self.window_samples = int(window_samples)
        self.stride_samples = int(stride_samples)
        if self.sample_rate_hz <= 0 or self.window_samples <= 0 or self.stride_samples <= 0:
            raise ValueError("sample rate, window, and stride must be positive")
        self.n_channels = int(model.config.input_channels)
        self.mean = np.asarray(normalization["mean"], dtype=np.float32)
        self.std = np.asarray(normalization["std"], dtype=np.float32)
        if self.mean.shape != (self.n_channels,) or self.std.shape != (self.n_channels,):
            raise ValueError("Normalization shape does not match model channels")
        self.gate = gate or BinaryIntentGate()
        self._buffer = np.empty((self.n_channels, 0), dtype=np.float32)
        self._new_samples = 0
        self._last_timestamp_ns: Optional[int] = None

    def reset(self, active: bool = False) -> None:
        self._buffer = np.empty((self.n_channels, 0), dtype=np.float32)
        self._new_samples = 0
        self._last_timestamp_ns = None
        self.gate.reset(active=active)

    def push(self, samples: np.ndarray, timestamp_ns: int, signal_quality: float = 1.0) -> Optional[Dict[str, object]]:
        chunk = np.asarray(samples, dtype=np.float32)
        if chunk.ndim == 1:
            chunk = chunk[:, None]
        if chunk.ndim != 2 or chunk.shape[0] != self.n_channels:
            raise ValueError(f"Expected chunk [{self.n_channels}, samples]")
        timestamp_ns = int(timestamp_ns)
        if self._last_timestamp_ns is not None and timestamp_ns <= self._last_timestamp_ns:
            raise ValueError("Streaming EMG timestamps must strictly increase")
        self._last_timestamp_ns = timestamp_ns
        self._buffer = np.concatenate([self._buffer, chunk], axis=1)[:, -self.window_samples :]
        self._new_samples += int(chunk.shape[1])
        if self._buffer.shape[1] < self.window_samples or self._new_samples < self.stride_samples:
            return None
        self._new_samples %= self.stride_samples
        normalized = (self._buffer - self.mean[:, None]) / self.std[:, None]
        tensor = self._torch.from_numpy(normalized[None]).to(self.device)
        with self._torch.no_grad():
            logits = self.model(tensor)
            probabilities = self._torch.softmax(logits, dim=-1)[0].cpu().numpy()
        event = self.gate.update(float(probabilities[1]), timestamp_ns, signal_quality)
        return {
            "timestamp_ns": timestamp_ns,
            "probability_open": float(probabilities[0]),
            "probability_close": float(probabilities[1]),
            "signal_quality": float(signal_quality),
            "event": None if event is None else event.to_dict(),
        }

