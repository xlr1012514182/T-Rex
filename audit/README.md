# Revo3 V1 bounded reproduction audit

## Exact outcome

**`component-verified`** is the only final audit outcome. A dependency-light `runnable-smoke` path supports that conclusion, but this audit does not certify installation of the full T-Rex stack, model-quality evaluation, physical task success, paper reproduction, clinical performance, or hardware safety.

The bounded chain that passed is:

```text
synthetic EMG event + mock RGB planner response
→ causal multi-rate alignment
→ single Task Executive
→ Revo3 21D mock T-Rex policy contract
→ strict T-Rex REP wire-protocol client contract
→ bounded tactile residual
→ final safety supervisor
→ in-memory Revo backend
```

The VLA observation contract intentionally contains RGB, language, 21D state and five-finger tactile features, but no EMG tensor. EMG is consumed upstream by the planner/executive.

The ZeroMQ client is covered with an injected request/reply transport, including malformed replies, task/lease mismatch, server chunk mismatch and timeout recovery. No real GPU server or official checkpoint was invoked, so this is wire-protocol plumbing evidence only.

## Authoritative sources and intentional deltas

| Source | Reopened identity | Local use | Status |
|---|---|---|---|
| [Official T-Rex repository](https://github.com/ZhuoyangLiu2005/T-Rex) | `main@09db9f3b3e3936fb760e67329174bd2bed527a05` | Base repository and slow/fast design | Base aligned; Revo adaptation intentionally diverges |
| [T-Rex paper](https://arxiv.org/abs/2606.17055) | arXiv 2606.17055 | Architecture/tactile-reactive reference | No paper protocol run |
| [Generic neuromotor interface](https://www.nature.com/articles/s41586-025-09255-w) | Nature 645, 702–711 (2025) | EMG topology reference | OPEN/CLOSE adaptation; synthetic data only |
| [Ask-to-Clarify](https://arxiv.org/abs/2509.15061) | arXiv 2509.15061v3 | Planner states and ambiguity handling | Structural adaptation, not reproduction |
| [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | commit `89644892e4d85e24eaac8bacfd4f463576704203` | Lazy real-planner backend | Not downloaded or run |
| [TactileReflex](https://arxiv.org/abs/2605.23568) | arXiv 2605.23568 | Bounded residual inspiration | Partial local abstraction only |
| [T-Rex pretrain weights](https://huggingface.co/miniFranka/T-Rex_pretrain_mecka22k_epoch1) | remote model repository | Planned Revo-specific midtrain initialization | Not downloaded or run |

The full expected/observed mapping is in `revo3_v1_reproduction_matrix.csv`.

## Reproduce the bounded evidence

From the repository root:

```powershell
py -3.10 -m pytest -q tests/revo3_v1

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

## Explicitly unsupported

- No Qwen3-VL weights were downloaded or invoked.
- No official T-Rex checkpoint was loaded, migrated or evaluated.
- No CUDA/GPU execution was performed.
- No BrainCo SDK client or Revo3 hand was connected; hardware writes remain disarmed by default.
- No U21VT-to-F6 calibration, sensor-unit mapping, real joint limits, current limits or supervised safety envelope was verified.
- No real EMG, amputee participant, cross-day study, accuracy, success rate, latency, cognitive-load, clinical-benefit or willingness-to-use claim is supported.
- The TactileReflex plugin is a bounded Revo synergy residual, not the full paper controller.
- The synthetic RGB frames are colored geometry proxies, not object-recognition evidence.

Before real training, create a separate readiness manifest for the immutable real dataset version and require verified controller-target provenance, calibrated units/joint order, causal observation-action timing, real camera review, and supervised Revo hardware replay.
