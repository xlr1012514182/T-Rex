# Revo3 V1 evidence and claims boundary

Cutoff: 2026-08-18. The repository audit outcome is **`component-verified`**.
The remote model subtask separately reaches **`runnable-smoke`**. Neither label
is a paper-reproduction, robot-task, training-readiness, safety, or clinical
certification.

## Directly supported

- The official T-Rex source base was reopened at
  `09db9f3b3e3936fb760e67329174bd2bed527a05`.
- Qwen3-VL-2B-Instruct revision
  `89644892e4d85e24eaac8bacfd4f463576704203` loaded on CUDA, generated text,
  and produced one schema-valid `ASK_CLARIFY` planner response.
- Official T-Rex midtrain revision
  `62efb3bcb45a3df0e088c8909d759b582cfb98af` loaded 1,412 source keys with
  zero missing and zero unexpected keys and returned one finite `[16,62]`
  `slow_and_fast` action chunk through the bounded production actor path.
- Local Revo3, camera, U21VT/VisionTouch, EDU EMG, EDU glove, Tianji-loader,
  timestamp, safety, recorder, and collection-orchestrator interfaces pass
  unit or injected-client checks. Hardware templates stay fail-closed.
- Synthetic fixtures preserve controller-boundary `exact_sent_target` labels,
  episode boundaries, and causal timing under their declared mock contracts:
  capture time is no later than the 30 Hz anchor and receive time is no later
  than the controller decision. The exporter independently rechecks both.

## Explicitly not supported

- The official T-Rex `[16,62]` result is not evidence for the local Revo3
  `[16,21]` contract. No 62D-to-21D weight migration is claimed.
- The intended official pretrain-to-Revo3-midtrain path was not loaded,
  trained, or evaluated.
- A schema-valid Qwen response is not object grounding, ambiguity-resolution,
  planner accuracy, or instruction quality evidence.
- Reported GPU memory and single-call durations describe only the bounded
  smoke runs; they are not latency, throughput, edge-deployment, or capacity
  guarantees.
- No physical Revo3, Tianji, camera, U21VT/VisionTouch, EDU EMG, or EDU glove
  stream or write was exercised. No limits, units, calibration, clock domain,
  E-stop, hold path, retargeting, or hardware replay was bench verified.
- Python cannot forcibly stop a permanently blocked vendor-native call. The
  implemented timeout quarantines the episode and records that process/device
  intervention is required; it is not a process-level watchdog or kill
  guarantee. Real operation requires isolated hardware processes, an external
  watchdog, and a physical E-stop.
- No real dataset is approved. Both checked manifests are synthetic fixtures
  with `training_approved=false`, `task_success_defined=false`, and zero
  task-success rates. Exact serialization is not task success.
- No physical task success, model-quality comparison, paper metric, amputee
  study, cross-day stability, cognitive-load reduction, safety, usability, or
  clinical benefit is supported.

## Promotion gates

Before a real-data or Revo3 claim, freeze a new immutable dataset identity and
verify: actual 21-joint order and units, controller-target provenance, causal
capture and receive timestamps, camera identity/calibration, five-finger
tactile identity and units, episode boundaries, human-reviewed RGB evidence,
supervised label replay through the real controller, and a separately defined
task-success protocol. Run the dataset-readiness validator with
`--require-ready`; the synthetic manifests must never be relabeled as real
training evidence.
