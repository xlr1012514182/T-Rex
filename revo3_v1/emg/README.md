# Revo3 V1 GNI-derived EMG primitive module

The mainline protocol is:

```text
multichannel EMG -> GNI-derived 5-class head -> confidence/margin/quality/dwell gate
                 -> StartIntentEvent / ReleaseEvent
```

The full model follows the released Generic Neuromotor Interface discrete-
gesture architecture: Reinhard compression (`64*x/(32+abs(x))`), Conv1D,
ReLU/dropout/layer normalization, stacked LSTM, layer normalization, and a
projection head. The project head learns `POWER_GRASP`, `PRECISION_GRASP`,
`LATERAL_GRASP`, `RELEASE`, and `REST`; low-margin outputs become `UNKNOWN` and
bad acquisition quality becomes `BAD_SIGNAL`. Time logits are pooled into one
decision per labelled window.

The production acquisition profile is versioned as
`brainco_edu_8ch_250hz_hp40_v1`: 8 named channels, 250 Hz, causal order-4
Butterworth high-pass at 40 Hz, 2 s/500-sample context, and an alternating
12/13-sample inference stride (exactly 50 ms on average). Because the official
GNI model is 16ch/2 kHz, the BrainCo preset explicitly rescales the temporal
stem from kernel/stride `21/10` to `3/1`; it never silently reuses 2 kHz
settings at 250 Hz.

Mainline training requires the official discrete-gesture checkpoint and the
pinned source commit `b6bf250e2be5a67b23488104335373cdb87a15c9`. The strict
migrator inherits only the compatible LSTM and layer-normalization tensors;
the 8-channel stem and five-class head are reinitialized and reported.
Training without that checkpoint is allowed only through the explicit
`--from-scratch-ablation` flag.

The synthetic generator and `BinaryIntentGate` retain the old `OPEN=0` /
`CLOSE=1` path strictly as a runnable integration fixture. In that adapter,
CLOSE maps to POWER_GRASP and OPEN maps to RELEASE; it is not the V1 study
label space.

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

Generation defaults to the frozen five-class, 8-channel, 250 Hz, 2-second
fixture contract. It requires only NumPy and works when PyTorch is not
installed. Add ``--fixture-binary`` only when exercising the legacy two-class
compatibility path.

## Train

```powershell
python scripts/revo3_v1_train_emg.py `
  --dataset outputs/emg_synthetic `
  --output outputs/emg_smoke_run `
  --from-scratch-ablation --allow-window-reset-fallback `
  --preset smoke --epochs 3
```

The command above is a five-class synthetic pipeline smoke and remains an
explicit from-scratch/window-reset ablation. It cannot be presented as the
session-continuous production preprocessing path. Mainline five-class training
with the released GNI initialization is:

```powershell
python scripts/revo3_v1_train_emg.py `
  --dataset <real_profile_bound_dataset> `
  --output outputs/emg_mainline `
  --gni-checkpoint <official_discrete_gestures.ckpt>
```

`windows.npz` must bind sample rate, channel order, profile ID, and a
`preprocessed` flag. The preferred real-data path filters each continuous
session causally before cutting windows and records
`filter_state_provenance=session_continuous_causal_sos_before_windowing` plus
the acquisition-profile fingerprint. Double filtering is rejected.
Independent zero-state window filtering is only an explicit
fixture/calibration fallback.

Use `--preset gni` for the released 512-output-channel Conv/512-hidden/three-layer
LSTM dimensions.  The smoke preset retains the same layer topology with small
dimensions and one LSTM layer.  Real acquisition must replace the synthetic
NPZ/manifests while preserving `[window, channel, sample]`, label and timestamp
contracts.

The checkpoint contains model configuration, a strict state dict, label map,
and train-only channel normalization/profile hashes. Hardware adapters call
`StreamingEMGClassifier.push_many(samples, sample_timestamps_ns, quality)`;
all 20 samples in a typical 250 Hz packet are consumed, so packetization does
not reduce the alternating 12/13-sample (~20 Hz) inference cadence. A gap over
150 ms clears a pending dwell. `MulticlassIntentGate` emits a Start event only after a
grasp class passes `0.80` confidence, `0.20` margin, `0.80` quality for 300 ms;
it emits Release only after the distinct RELEASE class passes `0.90` confidence,
`0.30` margin and `0.80` quality for 500 ms. REST, continued contraction,
UNKNOWN, BAD_SIGNAL, and EMG disconnect never release an active task.

Daily 5–10 minute labelled calibration is available via
`python -m revo3_v1.emg.calibration`. Temperature scaling is the default;
prototype and classification-head-only options are explicit. Every artifact
is bound to base-checkpoint bytes, normalization, preprocessing profile,
channel order, subject/day/session, and rejects training-session leakage. It
does not imply cross-day or clinical performance before participant testing.
