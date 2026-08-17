# Revo3 V1 EMG binary intent module

This module implements the demo-only `OPEN=0` / `CLOSE=1` path:

```text
multichannel EMG -> GNI-style classifier -> debounced StartIntentEvent / ReleaseEvent
```

The full model follows the released Generic Neuromotor Interface discrete-
gesture architecture: Reinhard compression (`64*x/(32+abs(x))`), Conv1D,
ReLU/dropout/layer normalization, stacked LSTM, layer normalization, and a
projection head.  Only the original nine-class output is replaced by two
classes and time logits are pooled into one decision per labelled window.

The synthetic corpus is an integration fixture, **not evidence of biological
accuracy or user performance**.  It records subject, session and nanosecond
timestamps.  Subjects are assigned as whole groups to train/validation/test;
normalization is fitted only from train-manifest indices.
The training dataset applies the released GNI-style circular electrode
rotation augmentation (default `+/-2` channels) only to the train split.

## Generate a smoke corpus

```powershell
python scripts/revo3_v1_generate_emg.py --output outputs/emg_synthetic --preset smoke
```

Generation requires only NumPy and works when PyTorch is not installed.

## Train

```powershell
python scripts/revo3_v1_train_emg.py `
  --dataset outputs/emg_synthetic `
  --output outputs/emg_smoke_run `
  --preset smoke --epochs 3
```

Use `--preset gni` for the released 512-channel Conv/512-hidden/three-layer
LSTM dimensions.  The smoke preset retains the same layer topology with small
dimensions and one LSTM layer.  Real acquisition must replace the synthetic
NPZ/manifests while preserving `[window, channel, sample]`, label and timestamp
contracts.

The checkpoint contains model configuration, a strict state dict, label map,
and train-only channel normalization.  `StreamingEMGClassifier` consumes
timestamped chunks and `BinaryIntentGate` emits one Start event after stable
CLOSE, then one Release event after stable OPEN; ambiguous or bad-quality EMG
never releases an active task.
