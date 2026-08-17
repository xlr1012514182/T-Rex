"""GNI/Nature-style EMG model and checkpoint helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

from .preprocessing import EmgPreprocessingProfile
from .primitives import MAINLINE_CLASS_LABELS

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised on acquisition-only hosts
    torch = None
    nn = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


@dataclass(frozen=True)
class GNIModelConfig:
    input_channels: int = 16
    conv_output_channels: int = 512
    kernel_width: int = 21
    stride: int = 10
    lstm_hidden_size: int = 512
    lstm_num_layers: int = 3
    output_channels: int = 5
    label_names: Tuple[str, ...] = MAINLINE_CLASS_LABELS
    dropout: float = 0.1
    reinhard_range: float = 64.0
    reinhard_midpoint: float = 32.0
    pooling: str = "mean"

    @classmethod
    def smoke(cls, input_channels: int, *, mainline: bool = False) -> "GNIModelConfig":
        return cls(
            input_channels=input_channels,
            conv_output_channels=32,
            kernel_width=9,
            stride=4,
            lstm_hidden_size=32,
            lstm_num_layers=1,
            output_channels=5 if mainline else 2,
            label_names=MAINLINE_CLASS_LABELS if mainline else ("OPEN", "CLOSE"),
            dropout=0.05,
        )

    @classmethod
    def brainco_edu(cls) -> "GNIModelConfig":
        """Explicit 8-channel/250 Hz domain adaptation of the GNI backbone.

        The released GNI convolution uses width 21/stride 10 at 2 kHz.  The
        nearest causal sampling-domain equivalents at 250 Hz are width 3 and
        stride 1.  Keeping the original 21/10 values would silently change the
        temporal receptive field by 8x, so this must be an explicit preset.
        This is GNI-derived initialization topology, not a claim that the
        original 16-channel/2 kHz weights are directly compatible.
        """

        return cls(
            input_channels=8,
            conv_output_channels=512,
            kernel_width=3,
            stride=1,
            lstm_hidden_size=512,
            lstm_num_layers=3,
            output_channels=len(MAINLINE_CLASS_LABELS),
            label_names=MAINLINE_CLASS_LABELS,
            dropout=0.1,
        )

    def resolved_labels(self) -> Tuple[str, ...]:
        labels = tuple(str(value).upper() for value in self.label_names)
        if len(labels) != self.output_channels:
            # Compatibility with v1 checkpoints written before label_names was
            # stored in model_config.
            if self.output_channels == 2:
                return ("OPEN", "CLOSE")
            raise ValueError("label_names length must equal output_channels")
        if len(set(labels)) != len(labels):
            raise ValueError("label_names must be unique")
        return labels


if nn is not None:

    class ReinhardCompression(nn.Module):
        """Exact operator used in the released Generic Neuromotor Interface."""

        def __init__(self, value_range: float = 64.0, midpoint: float = 32.0) -> None:
            super().__init__()
            self.value_range = float(value_range)
            self.midpoint = float(midpoint)

        def forward(self, inputs):
            return self.value_range * inputs / (self.midpoint + torch.abs(inputs))


    class GNIClassifier(nn.Module):
        """Reinhard -> Conv1D -> three-layer LSTM -> gesture projection.

        The full preset reproduces the released GNI layer dimensions.  GNI
        predicts time-local gesture logits; this project pools the sequence to
        one primitive decision per window.
        """

        def __init__(self, config: GNIModelConfig) -> None:
            super().__init__()
            config.resolved_labels()
            if config.pooling not in {"mean", "last"}:
                raise ValueError("pooling must be 'mean' or 'last'")
            self.config = config
            self.compression = ReinhardCompression(config.reinhard_range, config.reinhard_midpoint)
            self.conv_layer = nn.Conv1d(
                config.input_channels,
                config.conv_output_channels,
                kernel_size=config.kernel_width,
                stride=config.stride,
            )
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=config.dropout)
            self.post_conv_layer_norm = nn.LayerNorm(config.conv_output_channels)
            self.lstm = nn.LSTM(
                input_size=config.conv_output_channels,
                hidden_size=config.lstm_hidden_size,
                num_layers=config.lstm_num_layers,
                batch_first=True,
                dropout=config.dropout if config.lstm_num_layers > 1 else 0.0,
            )
            self.post_lstm_layer_norm = nn.LayerNorm(config.lstm_hidden_size)
            self.projection = nn.Linear(config.lstm_hidden_size, config.output_channels)

        def forward_features(self, inputs):
            if inputs.ndim != 3:
                raise ValueError("EMG input must have shape [batch, channels, samples]")
            if inputs.shape[1] != self.config.input_channels:
                raise ValueError(
                    f"Expected {self.config.input_channels} EMG channels, received {inputs.shape[1]}"
                )
            if inputs.shape[2] < self.config.kernel_width:
                raise ValueError("EMG window is shorter than the Conv1D kernel")
            x = self.compression(inputs)
            x = self.dropout(self.relu(self.conv_layer(x)))
            x = self.post_conv_layer_norm(x.transpose(1, 2))
            x, _ = self.lstm(x)
            return self.post_lstm_layer_norm(x)

        def forward_sequence(self, inputs):
            return self.projection(self.forward_features(inputs))

        def forward(self, inputs):
            sequence_logits = self.forward_sequence(inputs)
            if self.config.pooling == "last":
                return sequence_logits[:, -1, :]
            return sequence_logits.mean(dim=1)


    class GNIBinaryClassifier(GNIClassifier):
        """Backward-compatible two-class model used only by the smoke demo."""

        def __init__(self, config: GNIModelConfig) -> None:
            if config.output_channels != 2:
                raise ValueError("GNIBinaryClassifier requires exactly two output classes")
            super().__init__(config)

else:

    class ReinhardCompression:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for the EMG model") from _TORCH_IMPORT_ERROR


    class GNIClassifier:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for the EMG model") from _TORCH_IMPORT_ERROR


    class GNIBinaryClassifier(GNIClassifier):  # type: ignore[no-redef]
        pass


def save_emg_checkpoint(
    path: str | Path,
    model: "GNIClassifier",
    normalization: Mapping[str, Any],
    training_metadata: Mapping[str, Any] | None = None,
    preprocessing_profile: EmgPreprocessingProfile | None = None,
) -> None:
    if torch is None:
        raise ImportError("PyTorch is required to save an EMG checkpoint") from _TORCH_IMPORT_ERROR
    bound_profile = (
        None if preprocessing_profile is None else preprocessing_profile.bind_normalization(normalization)
    )
    payload = {
        "schema_version": (
            "revo3-emg-checkpoint-v2"
            if bound_profile is None
            else "revo3-emg-checkpoint-v3"
        ),
        "model_config": asdict(model.config),
        "state_dict": model.state_dict(),
        "normalization": {
            "mean": torch.as_tensor(normalization["mean"], dtype=torch.float32),
            "std": torch.as_tensor(normalization["std"], dtype=torch.float32),
        },
        "labels": {index: label for index, label in enumerate(model.config.resolved_labels())},
        "training_metadata": dict(training_metadata or {}),
        "preprocessing_profile": (
            None if bound_profile is None else dict(bound_profile.to_mapping())
        ),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)


def load_emg_checkpoint(
    path: str | Path, map_location: str = "cpu"
) -> Tuple["GNIClassifier", Dict[str, Any]]:
    if torch is None:
        raise ImportError("PyTorch is required to load an EMG checkpoint") from _TORCH_IMPORT_ERROR
    try:
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    except TypeError:  # torch < 2.0 compatibility
        payload = torch.load(Path(path), map_location=map_location)
    if payload.get("schema_version") not in {
        "revo3-emg-checkpoint-v1",
        "revo3-emg-checkpoint-v2",
        "revo3-emg-checkpoint-v3",
    }:
        raise ValueError("Unsupported or missing EMG checkpoint schema")
    config_data = dict(payload["model_config"])
    if "label_names" not in config_data and int(config_data.get("output_channels", 2)) == 2:
        config_data["label_names"] = ("OPEN", "CLOSE")
    config = GNIModelConfig(**config_data)
    model = GNIClassifier(config)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    if payload.get("schema_version") == "revo3-emg-checkpoint-v3":
        raw_profile = payload.get("preprocessing_profile")
        if not isinstance(raw_profile, Mapping):
            raise ValueError("EMG v3 checkpoint lacks preprocessing_profile")
        profile = EmgPreprocessingProfile.from_mapping(raw_profile)
        if profile.channel_count != config.input_channels:
            raise ValueError("EMG checkpoint model/preprocessing channel mismatch")
        profile.bind_normalization(payload["normalization"])
    return model, payload
