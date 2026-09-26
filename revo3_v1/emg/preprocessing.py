"""Versioned causal EMG preprocessing shared by training and streaming."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from typing import Any, Mapping, Sequence, Tuple

import numpy as np


PREPROCESSING_SCHEMA = "revo3-emg-preprocessing-v1"


def _as_numpy(value: Any) -> np.ndarray:
    """Detach tensors before hashing/converting without importing torch."""

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def normalization_sha256(normalization: Mapping[str, Any]) -> str:
    """Stable hash used by training, streaming and calibration artifacts."""

    digest = hashlib.sha256()
    for name in ("mean", "std"):
        if name not in normalization:
            raise ValueError(f"normalization is missing {name}")
        array = np.asarray(_as_numpy(normalization[name]), dtype="<f4")
        digest.update(name.encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class EmgPreprocessingProfile:
    profile_id: str
    channel_order: Tuple[str, ...]
    sample_rate_hz: int
    highpass_hz: float
    highpass_order: int
    window_context_s: float
    inference_stride_ms: int
    filter_design: str = "butterworth_sos_causal"
    normalization_sha256: str = ""
    schema_version: str = PREPROCESSING_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != PREPROCESSING_SCHEMA:
            raise ValueError("Unsupported EMG preprocessing schema")
        if not self.profile_id or not self.channel_order:
            raise ValueError("profile_id and channel_order are required")
        if len(set(self.channel_order)) != len(self.channel_order):
            raise ValueError("EMG channel_order must be unique")
        if self.sample_rate_hz <= 0 or not 0 < self.highpass_hz < self.sample_rate_hz / 2:
            raise ValueError("Invalid sample rate/high-pass cutoff")
        if self.highpass_order < 1 or self.window_context_s <= 0 or self.inference_stride_ms <= 0:
            raise ValueError("Invalid filter/window/stride parameters")
        if self.filter_design != "butterworth_sos_causal":
            raise ValueError("V1 supports only causal Butterworth SOS high-pass")

    @property
    def channel_count(self) -> int:
        return len(self.channel_order)

    @property
    def window_samples(self) -> int:
        return int(round(self.window_context_s * self.sample_rate_hz))

    @property
    def stride_samples_pattern(self) -> Tuple[int, ...]:
        # 250 Hz * 50 ms = 12.5 samples.  Alternating 12/13 preserves the exact
        # average without resampling or future interpolation.
        exact = self.sample_rate_hz * self.inference_stride_ms / 1000.0
        lower = int(np.floor(exact))
        upper = int(np.ceil(exact))
        return (lower,) if lower == upper else (lower, upper)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def acquisition_fingerprint(self) -> str:
        """Fingerprint before train-only normalization is fitted."""

        return replace(self, normalization_sha256="").fingerprint

    def to_mapping(self) -> Mapping[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EmgPreprocessingProfile":
        data = dict(value)
        if "channel_order" in data:
            data["channel_order"] = tuple(str(item) for item in data["channel_order"])
        return cls(**data)

    def with_normalization(self, normalization_sha256: str) -> "EmgPreprocessingProfile":
        if len(normalization_sha256) != 64:
            raise ValueError("normalization_sha256 must be a SHA-256 hex digest")
        return replace(self, normalization_sha256=normalization_sha256.lower())

    def bind_normalization(self, normalization: Mapping[str, Any]) -> "EmgPreprocessingProfile":
        digest = normalization_sha256(normalization)
        if self.normalization_sha256 and self.normalization_sha256 != digest:
            raise ValueError("EMG normalization/profile mismatch")
        return self if self.normalization_sha256 else self.with_normalization(digest)

    def validate_stream(self, *, sample_rate_hz: int, channel_order: Sequence[str]) -> None:
        if int(sample_rate_hz) != self.sample_rate_hz:
            raise ValueError(
                f"EMG sample rate/profile mismatch: {sample_rate_hz} != {self.sample_rate_hz}"
            )
        if tuple(channel_order) != self.channel_order:
            raise ValueError("EMG channel order/profile mismatch")

    @classmethod
    def brainco_edu_8ch_250hz(cls) -> "EmgPreprocessingProfile":
        return cls(
            profile_id="brainco_edu_8ch_250hz_hp40_v1",
            channel_order=tuple(f"emg_{index}" for index in range(8)),
            sample_rate_hz=250,
            highpass_hz=40.0,
            highpass_order=4,
            window_context_s=2.0,
            inference_stride_ms=50,
        )


BRAINCO_EDU_8CH_250HZ = EmgPreprocessingProfile.brainco_edu_8ch_250hz()


@dataclass(frozen=True)
class EmgDataPreprocessingState:
    preprocessed: bool
    provenance: str


def validate_npz_profile(
    archive: Mapping[str, Any],
    profile: EmgPreprocessingProfile,
) -> EmgDataPreprocessingState:
    """Fail closed when an acquisition file lacks its sampling-domain identity."""

    required = (
        "sample_rate_hz",
        "channel_order",
        "preprocessing_profile_id",
        "preprocessed",
    )
    missing = [name for name in required if name not in archive]
    if missing:
        raise ValueError("EMG NPZ lacks acquisition metadata: " + ",".join(missing))
    sample_rate = int(np.asarray(archive["sample_rate_hz"]).reshape(()).item())
    channel_order = tuple(str(value) for value in np.asarray(archive["channel_order"]).tolist())
    profile.validate_stream(sample_rate_hz=sample_rate, channel_order=channel_order)
    stored_profile_id = str(np.asarray(archive["preprocessing_profile_id"]).reshape(()).item())
    if stored_profile_id != profile.profile_id:
        raise ValueError(
            f"EMG NPZ preprocessing profile mismatch: {stored_profile_id} != {profile.profile_id}"
        )
    preprocessed = bool(np.asarray(archive["preprocessed"]).reshape(()).item())
    provenance = str(
        np.asarray(archive.get("filter_state_provenance", "")).reshape(()).item()
    )
    if preprocessed:
        if provenance != "session_continuous_causal_sos_before_windowing":
            raise ValueError(
                "Preprocessed EMG requires session-continuous causal filter provenance"
            )
        stored_fingerprint = str(
            np.asarray(archive.get("preprocessing_profile_fingerprint", "")).reshape(()).item()
        )
        if stored_fingerprint != profile.acquisition_fingerprint:
            raise ValueError("Preprocessed EMG profile fingerprint mismatch")
    elif provenance:
        raise ValueError("Raw EMG must not claim filter-state provenance")
    return EmgDataPreprocessingState(preprocessed=preprocessed, provenance=provenance)


def preprocess_emg_windows(
    signals: np.ndarray,
    profile: EmgPreprocessingProfile,
) -> np.ndarray:
    """Apply the production causal filter independently to labelled windows.

    Real acquisition should preferably preserve filter state and cut windows
    after streaming preprocessing.  This helper is the deterministic fallback
    for raw, independently labelled calibration/training windows.
    """

    values = np.asarray(signals, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("EMG windows must have shape [windows, channels, samples]")
    if values.shape[1] != profile.channel_count:
        raise ValueError("EMG window channels/profile mismatch")
    if values.shape[2] != profile.window_samples:
        raise ValueError("EMG window length/profile mismatch")
    processor = CausalEMGPreprocessor(profile)
    output = np.empty_like(values)
    for index, window in enumerate(values):
        output[index] = processor.process_window(window)
    return output


class CausalEMGPreprocessor:
    """Stateful causal high-pass filter with serializable SOS state."""

    def __init__(self, profile: EmgPreprocessingProfile) -> None:
        try:
            from scipy.signal import butter
        except ImportError as exc:
            raise RuntimeError("scipy is required for causal EMG preprocessing") from exc
        self.profile = profile
        self.sos = butter(
            profile.highpass_order,
            profile.highpass_hz,
            btype="highpass",
            fs=profile.sample_rate_hz,
            output="sos",
        ).astype(np.float64)
        self.reset()

    def reset(self) -> None:
        self._zi = np.zeros(
            (self.profile.channel_count, self.sos.shape[0], 2), dtype=np.float64
        )

    def process_chunk(
        self,
        samples: np.ndarray,
        *,
        sample_rate_hz: int,
        channel_order: Sequence[str],
    ) -> np.ndarray:
        from scipy.signal import sosfilt

        self.profile.validate_stream(
            sample_rate_hz=sample_rate_hz, channel_order=channel_order
        )
        values = np.asarray(samples, dtype=np.float32)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or values.shape[0] != self.profile.channel_count:
            raise ValueError(
                f"EMG chunk must be [{self.profile.channel_count}, samples]"
            )
        if not np.isfinite(values).all():
            raise ValueError("EMG chunk contains non-finite values")
        output = np.empty_like(values)
        for channel in range(self.profile.channel_count):
            filtered, state = sosfilt(
                self.sos,
                values[channel].astype(np.float64),
                zi=self._zi[channel],
            )
            output[channel] = filtered.astype(np.float32)
            self._zi[channel] = state
        return output

    def process_window(self, samples: np.ndarray) -> np.ndarray:
        self.reset()
        return self.process_chunk(
            samples,
            sample_rate_hz=self.profile.sample_rate_hz,
            channel_order=self.profile.channel_order,
        )

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": "revo3-emg-filter-state-v1",
            "profile_fingerprint": self.profile.fingerprint,
            "zi": self._zi.copy(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != "revo3-emg-filter-state-v1":
            raise ValueError("Unsupported EMG filter state schema")
        if state.get("profile_fingerprint") != self.profile.fingerprint:
            raise ValueError("EMG filter state/profile mismatch")
        zi = np.asarray(state.get("zi"), dtype=np.float64)
        if zi.shape != self._zi.shape or not np.isfinite(zi).all():
            raise ValueError("Invalid EMG causal filter state")
        self._zi = zi.copy()
