"""Streaming GNI inference and confidence/margin/dwell intent events."""

from __future__ import annotations

from dataclasses import dataclass
import uuid
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from .primitives import (
    EMGPrimitive,
    MAINLINE_CLASS_LABELS,
    START_PRIMITIVES,
    normalize_emg_primitive,
)
from .preprocessing import (
    BRAINCO_EDU_8CH_250HZ,
    CausalEMGPreprocessor,
    EmgPreprocessingProfile,
)


@dataclass(frozen=True)
class IntentGateConfig:
    """Frozen V1 event thresholds.

    ``close_*`` and ``open_*`` remain as compatibility names for the initial
    binary demo; semantically they are StartIntent and ReleaseEvent settings.
    """

    close_threshold: float = 0.80
    open_threshold: float = 0.90
    start_margin: float = 0.20
    release_margin: float = 0.30
    close_dwell_ms: int = 300
    open_dwell_ms: int = 500
    min_signal_quality: float = 0.80
    max_update_gap_ms: int = 150

    @property
    def start_confidence(self) -> float:
        return self.close_threshold

    @property
    def release_confidence(self) -> float:
        return self.open_threshold

    @property
    def start_dwell_ms(self) -> int:
        return self.close_dwell_ms

    @property
    def release_dwell_ms(self) -> int:
        return self.open_dwell_ms

    def validate(self) -> None:
        for name, value in (
            ("close_threshold", self.close_threshold),
            ("open_threshold", self.open_threshold),
            ("start_margin", self.start_margin),
            ("release_margin", self.release_margin),
            ("min_signal_quality", self.min_signal_quality),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.close_threshold < 0.5 or self.open_threshold < 0.5:
            raise ValueError("intent confidence thresholds must be at least 0.5")
        if self.close_dwell_ms < 0 or self.open_dwell_ms < 0:
            raise ValueError("dwell times cannot be negative")
        if self.max_update_gap_ms <= 0:
            raise ValueError("max_update_gap_ms must be positive")


@dataclass(frozen=True)
class IntentEvent:
    event_type: str
    primitive: str
    confidence: float
    signal_quality: float
    timestamp_ns: int
    event_id: str
    margin: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "type": self.event_type,
            "primitive": self.primitive,
            "confidence": self.confidence,
            "margin": self.margin,
            "signal_quality": self.signal_quality,
            "timestamp_ns": self.timestamp_ns,
            "event_id": self.event_id,
        }


@dataclass(frozen=True)
class IntentObservation:
    primitive: EMGPrimitive
    confidence: float
    margin: float
    signal_quality: float
    timestamp_ns: int
    event: Optional[IntentEvent] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "primitive": self.primitive.value,
            "confidence": self.confidence,
            "margin": self.margin,
            "signal_quality": self.signal_quality,
            "timestamp_ns": self.timestamp_ns,
            "event": None if self.event is None else self.event.to_dict(),
        }


class MulticlassIntentGate:
    """Convert class probabilities into non-repeating task events.

    Only the three grasp primitives can start a task.  Once active,
    REST/UNKNOWN/BAD_SIGNAL and continued contraction cannot cancel or restart
    it.  Only a separately learned, sustained RELEASE class emits ReleaseEvent.
    """

    def __init__(self, config: IntentGateConfig | None = None) -> None:
        self.config = config or IntentGateConfig()
        self.config.validate()
        self.active = False
        self._pending: Optional[EMGPrimitive] = None
        self._pending_since_ns: Optional[int] = None
        self._last_timestamp_ns: Optional[int] = None

    def reset(self, active: bool = False) -> None:
        self.active = bool(active)
        self._pending = None
        self._pending_since_ns = None
        self._last_timestamp_ns = None

    def _clear_pending(self) -> None:
        self._pending = None
        self._pending_since_ns = None

    @staticmethod
    def _rank(probabilities: Mapping[str, float]) -> tuple[EMGPrimitive, float, float]:
        normalized: Dict[EMGPrimitive, float] = {}
        for label, raw_probability in probabilities.items():
            primitive = normalize_emg_primitive(label)
            if primitive is None or primitive in {EMGPrimitive.UNKNOWN, EMGPrimitive.BAD_SIGNAL}:
                continue
            probability = float(raw_probability)
            if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError("EMG class probabilities must be finite values in [0, 1]")
            normalized[primitive] = normalized.get(primitive, 0.0) + probability
        if not normalized:
            raise ValueError("No recognized EMG classes were supplied")
        ranked = sorted(normalized.items(), key=lambda item: item[1], reverse=True)
        top_primitive, top_probability = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        return top_primitive, top_probability, top_probability - runner_up

    def update(
        self,
        probabilities: Mapping[str, float],
        timestamp_ns: int,
        signal_quality: float = 1.0,
    ) -> IntentObservation:
        timestamp_ns = int(timestamp_ns)
        quality = float(signal_quality)
        if self._last_timestamp_ns is not None and timestamp_ns < self._last_timestamp_ns:
            raise ValueError("EMG event timestamps must be monotonic")
        if (
            self._last_timestamp_ns is not None
            and timestamp_ns - self._last_timestamp_ns
            > self.config.max_update_gap_ms * 1_000_000
        ):
            self._clear_pending()
        self._last_timestamp_ns = timestamp_ns
        if not np.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("signal_quality must be in [0, 1]")

        top, confidence, margin = self._rank(probabilities)
        if quality < self.config.min_signal_quality:
            self._clear_pending()
            return IntentObservation(
                EMGPrimitive.BAD_SIGNAL, confidence, margin, quality, timestamp_ns
            )

        if self.active:
            actionable = (
                top == EMGPrimitive.RELEASE
                and confidence >= self.config.release_confidence
                and margin >= self.config.release_margin
            )
            output = top if actionable or top == EMGPrimitive.REST else EMGPrimitive.UNKNOWN
            dwell_ms = self.config.release_dwell_ms
        else:
            actionable = (
                top in START_PRIMITIVES
                and confidence >= self.config.start_confidence
                and margin >= self.config.start_margin
            )
            output = top if actionable or top == EMGPrimitive.REST else EMGPrimitive.UNKNOWN
            dwell_ms = self.config.start_dwell_ms

        if not actionable:
            self._clear_pending()
            return IntentObservation(output, confidence, margin, quality, timestamp_ns)

        if self._pending != top:
            self._pending = top
            self._pending_since_ns = timestamp_ns
        assert self._pending_since_ns is not None
        if timestamp_ns - self._pending_since_ns < dwell_ms * 1_000_000:
            return IntentObservation(top, confidence, margin, quality, timestamp_ns)

        self._clear_pending()
        if top == EMGPrimitive.RELEASE:
            self.active = False
            event_type = "ReleaseEvent"
        else:
            self.active = True
            event_type = "StartIntentEvent"
        event = IntentEvent(
            event_type=event_type,
            primitive=top.value,
            confidence=confidence,
            margin=margin,
            signal_quality=quality,
            timestamp_ns=timestamp_ns,
            event_id=str(uuid.uuid4()),
        )
        return IntentObservation(top, confidence, margin, quality, timestamp_ns, event)


class BinaryIntentGate:
    """Compatibility adapter: CLOSE -> POWER_GRASP and OPEN -> RELEASE."""

    def __init__(self, config: IntentGateConfig | None = None) -> None:
        self._gate = MulticlassIntentGate(config)

    @property
    def config(self) -> IntentGateConfig:
        return self._gate.config

    @property
    def active(self) -> bool:
        return self._gate.active

    def reset(self, active: bool = False) -> None:
        self._gate.reset(active=active)

    def update(
        self,
        probability_close: float,
        timestamp_ns: int,
        signal_quality: float = 1.0,
    ) -> Optional[IntentEvent]:
        p_close = float(probability_close)
        if not 0.0 <= p_close <= 1.0:
            raise ValueError("probability_close must be in [0, 1]")
        observation = self._gate.update(
            {
                EMGPrimitive.POWER_GRASP.value: p_close,
                EMGPrimitive.RELEASE.value: 1.0 - p_close,
            },
            timestamp_ns,
            signal_quality,
        )
        return observation.event


class StreamingEMGClassifier:
    """Timestamped ring-buffer inference around a trained PyTorch model."""

    def __init__(
        self,
        model,
        normalization: Mapping[str, object],
        sample_rate_hz: int,
        window_samples: int,
        stride_samples: int,
        gate: MulticlassIntentGate | BinaryIntentGate | None = None,
        device: str = "cpu",
        class_labels: Optional[Sequence[str]] = None,
        calibration: object | None = None,
        preprocessing_profile: EmgPreprocessingProfile | None = BRAINCO_EDU_8CH_250HZ,
        channel_order: Optional[Sequence[str]] = None,
        expected_profile_fingerprint: Optional[str] = None,
        allow_unprofiled_fixture: bool = False,
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
        def _numpy(value):
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            return np.asarray(value, dtype=np.float32)

        self.mean = _numpy(normalization["mean"])
        self.std = _numpy(normalization["std"])
        if self.mean.shape != (self.n_channels,) or self.std.shape != (self.n_channels,):
            raise ValueError("Normalization shape does not match model channels")
        if np.any(self.std <= 0):
            raise ValueError("Normalization std must be positive")
        self.preprocessing_profile = preprocessing_profile
        self.channel_order = tuple(channel_order or ())
        self._preprocessor: CausalEMGPreprocessor | None = None
        self._stride_pattern = (self.stride_samples,)
        self._stride_phase = 0
        if preprocessing_profile is None:
            if not allow_unprofiled_fixture:
                raise ValueError(
                    "Streaming EMG requires an explicit preprocessing profile; "
                    "unprofiled mode is fixture-only"
                )
            if self.channel_order:
                raise ValueError("channel_order cannot be supplied without a profile")
        else:
            if not self.channel_order:
                raise ValueError("Streaming EMG requires explicit channel_order")
            preprocessing_profile.validate_stream(
                sample_rate_hz=self.sample_rate_hz,
                channel_order=self.channel_order,
            )
            if self.n_channels != preprocessing_profile.channel_count:
                raise ValueError("EMG model/preprocessing channel mismatch")
            if self.window_samples != preprocessing_profile.window_samples:
                raise ValueError("EMG window/preprocessing profile mismatch")
            self._stride_pattern = preprocessing_profile.stride_samples_pattern
            if self.stride_samples not in self._stride_pattern:
                raise ValueError("EMG stride/preprocessing profile mismatch")
            self._stride_phase = self._stride_pattern.index(self.stride_samples)
            bound_profile = preprocessing_profile.bind_normalization(normalization)
            if (
                expected_profile_fingerprint is not None
                and bound_profile.fingerprint != expected_profile_fingerprint
            ):
                raise ValueError("EMG streaming/checkpoint preprocessing profile mismatch")
            self.preprocessing_profile = bound_profile
            self._preprocessor = CausalEMGPreprocessor(preprocessing_profile)
        inferred = ()
        if callable(getattr(model.config, "resolved_labels", None)):
            inferred = tuple(model.config.resolved_labels())
        self.class_labels = tuple(class_labels or inferred)
        if not self.class_labels:
            output_channels = int(model.config.output_channels)
            self.class_labels = (
                ("OPEN", "CLOSE")
                if output_channels == 2
                else MAINLINE_CLASS_LABELS[:output_channels]
            )
        if len(self.class_labels) != int(model.config.output_channels):
            raise ValueError("class_labels length does not match model output channels")
        self.gate = gate or (
            BinaryIntentGate() if self.class_labels == ("OPEN", "CLOSE") else MulticlassIntentGate()
        )
        self.calibration = calibration
        if calibration is not None:
            calibration.apply_head(self.model)
        self._buffer = np.empty((self.n_channels, 0), dtype=np.float32)
        self._new_samples = 0
        self._has_inferred = False
        self._last_timestamp_ns: Optional[int] = None

    def reset(self, active: bool = False) -> None:
        self._buffer = np.empty((self.n_channels, 0), dtype=np.float32)
        self._new_samples = 0
        self._has_inferred = False
        self._stride_phase = self._stride_pattern.index(self.stride_samples)
        self._last_timestamp_ns = None
        if self._preprocessor is not None:
            self._preprocessor.reset()
        self.gate.reset(active=active)

    def push(
        self,
        samples: np.ndarray,
        timestamp_ns: int,
        signal_quality: float = 1.0,
    ) -> Optional[Dict[str, object]]:
        """Compatibility wrapper treating ``timestamp_ns`` as last-sample time.

        Hardware adapters must call :meth:`push_many` with one timestamp per
        sample; this wrapper cannot recover a jittered hardware sample clock.
        All inference steps are still executed and only the final result is
        returned for source compatibility.
        """

        chunk = np.asarray(samples, dtype=np.float32)
        if chunk.ndim == 1:
            chunk = chunk[:, None]
        if chunk.ndim != 2 or chunk.shape[0] != self.n_channels:
            raise ValueError(f"Expected chunk [{self.n_channels}, samples]")
        period_ns = int(round(1_000_000_000 / self.sample_rate_hz))
        end_ns = int(timestamp_ns)
        start_ns = end_ns - period_ns * (chunk.shape[1] - 1)
        timestamps = np.arange(chunk.shape[1], dtype=np.int64) * period_ns + start_ns
        results = self.push_many(chunk, timestamps, signal_quality=signal_quality)
        return None if not results else results[-1]

    def push_many(
        self,
        samples: np.ndarray,
        sample_timestamps_ns: Sequence[int] | np.ndarray,
        signal_quality: float = 1.0,
    ) -> list[Dict[str, object]]:
        """Consume every timestamped sample and emit every 12/13-sample step.

        BrainCo EDU packets commonly contain 20 samples (80 ms).  Returning a
        list prevents packetization from reducing the intended 50 ms (~20 Hz)
        decision cadence and makes dwell timing use the actual sample clock.
        """

        chunk = np.asarray(samples, dtype=np.float32)
        if chunk.ndim == 1:
            chunk = chunk[:, None]
        if chunk.ndim != 2 or chunk.shape[0] != self.n_channels:
            raise ValueError(f"Expected chunk [{self.n_channels}, samples]")
        if not np.isfinite(chunk).all():
            raise ValueError("Streaming EMG chunk contains non-finite values")
        timestamps = np.asarray(sample_timestamps_ns, dtype=np.int64)
        if timestamps.ndim != 1 or timestamps.size != chunk.shape[1]:
            raise ValueError("sample_timestamps_ns must contain one value per EMG sample")
        if timestamps.size == 0:
            return []
        if np.any(timestamps < 0) or np.any(np.diff(timestamps) <= 0):
            raise ValueError("EMG sample timestamps must be non-negative and strictly increasing")
        if self._last_timestamp_ns is not None:
            deltas = np.diff(np.concatenate(([self._last_timestamp_ns], timestamps)))
        else:
            deltas = np.diff(timestamps)
        expected_period = 1_000_000_000 / self.sample_rate_hz
        if deltas.size and np.any(
            (deltas < expected_period * 0.5) | (deltas > expected_period * 1.5)
        ):
            raise ValueError("EMG per-sample clock is inconsistent with preprocessing sample rate")
        if self._preprocessor is not None:
            chunk = self._preprocessor.process_chunk(
                chunk,
                sample_rate_hz=self.sample_rate_hz,
                channel_order=self.channel_order,
            )
        results: list[Dict[str, object]] = []
        for index, timestamp in enumerate(timestamps.tolist()):
            self._last_timestamp_ns = int(timestamp)
            self._buffer = np.concatenate(
                [self._buffer, chunk[:, index : index + 1]], axis=1
            )[:, -self.window_samples :]
            self._new_samples += 1
            if self._buffer.shape[1] < self.window_samples:
                continue
            current_stride = self._stride_pattern[self._stride_phase]
            if not self._has_inferred:
                self._has_inferred = True
                self._new_samples = 0
            elif self._new_samples < current_stride:
                continue
            else:
                self._new_samples -= current_stride
                self._stride_phase = (self._stride_phase + 1) % len(self._stride_pattern)
            results.append(self._infer(int(timestamp), signal_quality))
        return results

    def _infer(self, timestamp_ns: int, signal_quality: float) -> Dict[str, object]:
        normalized = (self._buffer - self.mean[:, None]) / self.std[:, None]
        tensor = self._torch.from_numpy(normalized[None]).to(self.device)
        with self._torch.no_grad():
            logits = (
                self.model(tensor)
                if self.calibration is None
                else self.calibration.logits(self.model, tensor)
            )
            values = self._torch.softmax(logits, dim=-1)[0].cpu().numpy()
        probabilities = {
            label: float(probability)
            for label, probability in zip(self.class_labels, values.tolist())
        }
        if isinstance(self.gate, BinaryIntentGate):
            close_index = self.class_labels.index("CLOSE")
            event = self.gate.update(float(values[close_index]), timestamp_ns, signal_quality)
            primitive = (
                EMGPrimitive.POWER_GRASP.value
                if int(np.argmax(values)) == close_index
                else EMGPrimitive.RELEASE.value
            )
            ranked = sorted(values.tolist(), reverse=True)
            margin = float(ranked[0] - ranked[1]) if len(ranked) > 1 else float(ranked[0])
            confidence = float(max(ranked))
        else:
            observation = self.gate.update(probabilities, timestamp_ns, signal_quality)
            event = observation.event
            primitive = observation.primitive.value
            margin = observation.margin
            confidence = observation.confidence
        result: Dict[str, object] = {
            "timestamp_ns": timestamp_ns,
            "probabilities": probabilities,
            "primitive": primitive,
            "confidence": confidence,
            "margin": margin,
            "signal_quality": float(signal_quality),
            "event": None if event is None else event.to_dict(),
        }
        if self.class_labels == ("OPEN", "CLOSE"):
            result["probability_open"] = probabilities["OPEN"]
            result["probability_close"] = probabilities["CLOSE"]
        return result
