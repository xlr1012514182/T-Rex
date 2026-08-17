# Revo3 V1 bounded reproduction audit

## Exact outcome

**`component-verified`** is the only final repository-audit outcome. The local dependency-light path and a separate bounded remote-GPU path support named components only; this audit does not certify the full T-Rex paper protocol, model quality, Revo3 checkpoint compatibility, physical task success, training readiness, clinical performance, or hardware safety.

The bounded chain that passed is:

```text
synthetic five-class EMG edge + mock three-full/center RGB planner response
→ causal multi-rate alignment
→ single Task Executive
→ asynchronous Revo3 21D mock T-Rex policy contract
→ strict T-Rex REP wire-protocol client contract
→ bounded tactile residual
→ 100 Hz single-writer servo and final safety supervisor
→ in-memory Revo backend
```

The VLA observation contract intentionally contains RGB, language, 21D state and five-finger tactile features, but no EMG tensor. EMG is consumed upstream by the planner/executive.

The ZeroMQ client is covered locally with an injected request/reply transport, including malformed replies, task/lease mismatch, server chunk mismatch and timeout recovery. Separately, an official T-Rex midtrain checkpoint ran one bounded production `CascadedServer.predict(slow_and_fast)` call on a remote GPU. That checkpoint returned finite `[16, 62]` actions and therefore verifies only the official dual-hand actor smoke path; it does not verify the local Revo3 `[16, 21]` adapter or the network service boundary.

## Authoritative sources and intentional deltas

| Source | Reopened identity | Local use | Status |
|---|---|---|---|
| [Official T-Rex repository](https://github.com/ZhuoyangLiu2005/T-Rex) | `main@09db9f3b3e3936fb760e67329174bd2bed527a05` | Base repository and slow/fast design | Base aligned; Revo adaptation intentionally diverges |
| [T-Rex paper](https://arxiv.org/abs/2606.17055) | arXiv 2606.17055 | Architecture/tactile-reactive reference | No paper protocol run |
| [Generic neuromotor interface](https://www.nature.com/articles/s41586-025-09255-w) | Nature 645, 702–711 (2025) | EMG topology reference | Explicit 8ch@250 Hz, five-class primitive adaptation; synthetic smoke only |
| [Ask-to-Clarify](https://arxiv.org/abs/2509.15061) | arXiv 2509.15061v3 | Planner states and ambiguity handling | Structural adaptation, not reproduction |
| [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | commit `89644892e4d85e24eaac8bacfd4f463576704203` | Lazy real-planner backend | Exact revision loaded; bounded generation and strict planner-schema smoke passed |
| [TactileReflex](https://arxiv.org/abs/2605.23568) | arXiv 2605.23568 | Bounded residual inspiration | Partial local abstraction only |
| [T-Rex pretrain weights](https://huggingface.co/miniFranka/T-Rex_pretrain_mecka22k_epoch1) | remote model repository | Planned Revo-specific midtrain initialization | Not downloaded or run |
| [T-Rex midtrain weights](https://huggingface.co/miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6) | commit `62efb3bcb45a3df0e088c8909d759b582cfb98af` | Official actor smoke only | Exact revision loaded; 1,412 keys, zero missing/unexpected; one finite `[16,62]` result |

The full expected/observed mapping is in `revo3_v1_reproduction_matrix.csv`.
The concise promotion and exclusion boundary is in
`claims_boundary_20260818.md`.

## Reproduce the bounded evidence

From the repository root:

```powershell
py -3.10 -m pytest -q tests/revo3_v1  # 288 passed

py -3.10 -m pytest -q                 # 425 passed

py -3.10 scripts/revo3_v1_generate_robot_demo.py `
  --output outputs/revo3_audit_mock/dataset `
  --episodes-per-task 1 --frames 20

py -3.10 scripts/revo3_v1_demo.py `
  --task bottle `
  --trace outputs/revo3_audit_mock/demo_bottle.json
```

The four-task smoke results and hashes are preserved in:

- `revo3_v1_test_results.txt`
- `revo3_v1_mock_data_conversion.json`
- `revo3_v1_mock_runtime.json`
- `revo3_v1_mock_controller_replay.json`

The separate remote-GPU evidence is preserved under `remote_gpu_smoke_20260817/`. Its sanitized command record contains no endpoint or credential. The strict Qwen run produced a schema-valid `ASK_CLARIFY` response; this is not object-grounding accuracy. The T-Rex run used the official 62D midtrain embodiment; its approximately 1.22 s single-call timing and observed memory peaks are descriptive smoke measurements, not latency or deployment certification.

Validate the audit matrix:

```powershell
py -3.10 E:\CodexData\skills\robotics-repo-reproduction-audit\scripts\audit_reproduction_matrix.py `
  audit\revo3_v1_reproduction_matrix.csv `
  --outcome component-verified --require-reviewed --json
```

Validate the synthetic manifest schema:

```powershell
py -3.10 E:\CodexData\skills\lehome-imitation-data\scripts\audit_dataset_readiness.py `
  audit\revo3_v1_mock_dataset_readiness.json --json
```

The readiness script reports schema/invariant validity for the generated mock fixture. The manifest itself deliberately records `training_approved=false`; task-success rates are zero because a synthetic in-memory hand has no task-success semantics. The separate `80/80` result proves only exact controller-target serialization through `MockRevoBackend`.

## Hardware adapter status

The working tree now contains opt-in clients and boundaries for BrainCo EDU EMG, BrainCo EDU glove, Revo3 SDK telemetry/commands, fisheye RGB rectification, U21VT/VisionTouch 6D force, a hash-verified Tianji loader, and a one-command collection orchestrator. Unit and injected-client checks may verify these interfaces, but the checked-in templates remain deliberately non-executable. BrainCo glove and EDU EMG are required to use separate `libedu` processes and timestamp-preserving IPC because their callback namespace is process-global.

## Explicitly unsupported

- The Qwen smoke verifies exact-revision loading, generation, and schema parsing only; it does not establish planner grounding, ambiguity-resolution accuracy, or task-selection accuracy.
- The loaded T-Rex midtrain checkpoint is the official dual-hand 62D embodiment. It is not compatible evidence for Revo3 21D, and no Revo-specific action/tactile head was trained or loaded.
- The planned official T-Rex **pretrain → Revo3 midtrain** initialization path was not downloaded, trained, or evaluated.
- Remote CUDA execution occurred only for the bounded model smokes. It is not a local installation claim, a throughput benchmark, or a latency guarantee.
- No BrainCo SDK client, camera, tactile sensor, Tianji arm, or Revo3 hand was physically connected; hardware writes remain disarmed by default.
- Python-side bounded cleanup cannot forcibly terminate a permanently stuck vendor-native call. A timeout quarantines the episode and requires process/device intervention; real operation still requires hardware clients in isolated processes, an external watchdog, and a physical E-stop.
- No U21VT-to-F6 calibration, sensor-unit mapping, real joint limits, current limits or supervised safety envelope was verified.
- No real EMG, amputee participant, cross-day study, accuracy, success rate, latency, cognitive-load, clinical-benefit or willingness-to-use claim is supported.
- The TactileReflex plugin is a bounded Revo synergy residual, not the full paper controller.
- The synthetic RGB frames are colored geometry proxies, not object-recognition evidence.

Before real training, create a separate readiness manifest for the immutable real dataset version and require verified controller-target provenance, calibrated units/joint order, causal observation-action timing, real camera review, and supervised Revo hardware replay.
