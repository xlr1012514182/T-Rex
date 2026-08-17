"""GNI/Nature-style binary EMG model and checkpoint helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

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
    output_channels: int = 2
    dropout: float = 0.1
    reinhard_range: float = 64.0
    reinhard_midpoint: float = 32.0
    pooling: str = "mean"

    @classmethod
    def smoke(cls, input_channels: int) -> "GNIModelConfig":
        return cls(
            input_channels=input_channels,
            conv_output_channels=32,
            kernel_width=9,
            stride=4,
            lstm_hidden_size=32,
            lstm_num_layers=1,
            dropout=0.05,
        )


if nn is not None:

    class ReinhardCompression(nn.Module):
        """Exact operator used in the released Generic Neuromotor Interface."""

        def __init__(self, value_range: float = 64.0, midpoint: float = 32.0) -> None:
            super().__init__()
            self.value_range = float(value_range)
            self.midpoint = float(midpoint)

        def forward(self, inputs):
            return self.value_range * inputs / (self.midpoint + torch.abs(inputs))


    class GNIBinaryClassifier(nn.Module):
        """Reinhard -> Conv1D -> three-layer LSTM -> binary projection.

        The full preset reproduces the released GNI layer dimensions.  GNI
        predicts time-local gesture logits; this project pools the sequence to
        one OPEN/CLOSE decision per window.
        """

        def __init__(self, config: GNIModelConfig) -> None:
            super().__init__()
            if config.output_channels != 2:
                raise ValueError("The Revo3 V1 EMG module requires exactly two output classes")
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

        def forward_sequence(self, inputs):
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
            x = self.post_lstm_layer_norm(x)
            return self.projection(x)  # [B, T', 2]

        def forward(self, inputs):
            sequence_logits = self.forward_sequence(inputs)
            if self.config.pooling == "last":
                return sequence_logits[:, -1, :]
            return sequence_logits.mean(dim=1)

else:

    class ReinhardCompression:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for the EMG model") from _TORCH_IMPORT_ERROR


    class GNIBinaryClassifier:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for the EMG model") from _TORCH_IMPORT_ERROR


def save_emg_checkpoint(
    path: str | Path,
    model: "GNIBinaryClassifier",
    normalization: Mapping[str, Any],
    training_metadata: Mapping[str, Any] | None = None,
) -> None:
    if torch is None:
        raise ImportError("PyTorch is required to save an EMG checkpoint") from _TORCH_IMPORT_ERROR
    payload = {
        "schema_version": "revo3-emg-checkpoint-v1",
        "model_config": asdict(model.config),
        "state_dict": model.state_dict(),
        "normalization": {
            "mean": torch.as_tensor(normalization["mean"], dtype=torch.float32),
            "std": torch.as_tensor(normalization["std"], dtype=torch.float32),
        },
        "labels": {0: "OPEN", 1: "CLOSE"},
        "training_metadata": dict(training_metadata or {}),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)


def load_emg_checkpoint(
    path: str | Path, map_location: str = "cpu"
) -> Tuple["GNIBinaryClassifier", Dict[str, Any]]:
    if torch is None:
        raise ImportError("PyTorch is required to load an EMG checkpoint") from _TORCH_IMPORT_ERROR
    try:
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    except TypeError:  # torch < 2.0 compatibility
        payload = torch.load(Path(path), map_location=map_location)
    if payload.get("schema_version") != "revo3-emg-checkpoint-v1":
        raise ValueError("Unsupported or missing EMG checkpoint schema")
    config = GNIModelConfig(**payload["model_config"])
    model = GNIBinaryClassifier(config)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload

