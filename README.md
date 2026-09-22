<div align="center">

# T-Rex × Revo 3

**EMG-driven intent · Visual task planning · Tactile-reactive dexterous grasping**

[中文](README_ZH.md) · [System architecture](docs/revo3_v1/README.md) · [Training and inference](docs/revo3_v1/TRAINING.md) · [Teleoperation and data collection](teleop_data_collection/README.md) · [Development](docs/DEVELOPMENT.md)

</div>

## Overview

T-Rex × Revo 3 is a modular research implementation for single-hand assistive grasping. It integrates EMG-based grasp recognition, Qwen3-VL visual task planning, a Revo-specific T-Rex action interface, and a safety-aware control runtime into an end-to-end pipeline. Initial Revo 3 real-robot grasping trials have been conducted.

The system separates **what the user intends**, **which target to interact with**, and **how the fingers should move**. EMG provides grasp or release intent, vision supplies the target and task semantics, and the action model generates joint trajectories driven by tactile feedback. Once confirmed, a grasp intent is latched by the system: the user does not need to maintain muscle contraction to hold the object. An explicit release intent triggers controlled opening.

Built on [T-Rex: Tactile-Reactive Dexterous Manipulation](https://github.com/ZhuoyangLiu2005/T-Rex), this project adds Revo 3 single-hand adaptation, EMG–RGB task coordination, single-camera planning, versioned action authorization, and a causally aligned data pipeline.

## Core capabilities

- **Intent–task separation.** Five-class EMG primitive recognition uses confidence, class margin, signal quality, and dwell-time gates; RGB supplies the target object and scene context.
- **Structured visual planning.** Qwen3-VL-2B reads three full frames and the current center view, returns validated task JSON, and requests clarification when the scene is ambiguous.
- **Tactile-reactive actions.** A continuous 21-joint state, 16-step absolute-action chunks, slow/fast inference, native-rate tactile history, and temporal aggregation.
- **A unified task lifecycle.** A single Task Executive manages start, continue, hold, replan, release, and termination. Asynchronous results are bound to a task and version.
- **Multi-rate control.** A 30 Hz action grid and an independently scheduled 100 Hz command writer apply calibrated limits, data-freshness checks, and confirmed stop handling.
- **Traceable training data.** Native-rate recording, causal alignment, exact-sent action labels, separate EMG/VLA exports, and frozen normalization.

## System flow

```text
EMG ──> grasp recognition ──────────────────┐
                                           v
RGB ──> full + center views ──> Qwen Planner ──> Task Executive
                                                    │
                           language + RGB + q[21] + tactile
                                                    v
                                    Revo T-Rex ──> action[16,21]
                                                    │
                                aggregation / interpolation
                                                    │
                            optional bounded tactile correction
                                                    v
                           safety authorization ──> Revo command writer
```

Raw EMG is excluded from VLA inputs and policy-training data. A stable grasp remains in `HOLD` until an explicit release. V1 controls only the hand; arm motion is outside the 21-dimensional action space.

## Task setup

| Task | Grasp primitive | Hand behavior |
|---|---|---|
| `bottle` | `POWER_GRASP` | Grasp and hold a centered bottle |
| `phone` | `PRECISION_GRASP` | Precisely grasp and hold a centered phone |
| `plastic_bag` | `PRECISION_GRASP` | Grasp and hold the bag handles |
| `refrigerator_door` | `LATERAL_GRASP` | Use a lateral grasp to hold the door handle |

The user's arm supplies the displacement for lifting the bag or pulling the door. The optional Tianji teleoperation module is used for data collection and is not part of the online hand-policy action space.

## Getting started

Use **Python 3.10** and run these commands from the repository root. The local workflow uses deterministic simulation backends and requires no hardware, downloaded model weights, or recorded dataset.

### Windows PowerShell

```powershell
py -3.10 -m venv .venv
$Python = (Resolve-Path .\.venv\Scripts\python.exe).Path
& $Python -m pip install --upgrade pip
& $Python -m pip install -r requirements-dev.txt
& $Python scripts/revo3_v1_runtime.py --mode simulation --task all --servo-ticks 120
& $Python -m pytest -q
```

### Linux Bash

```bash
python3.10 -m venv .venv
PYTHON=.venv/bin/python
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -r requirements-dev.txt
"$PYTHON" scripts/revo3_v1_runtime.py --mode simulation --task all --servo-ticks 120
"$PYTHON" -m pytest -q
```

The output includes task-state transitions, policy responses, authorized writes, and shutdown status. See the [development guide](docs/DEVELOPMENT.md) for synthetic-data exercises and single-inference model checks.

## Training and inference

The Revo policy follows **pretrain → W0 → W1 → Revo midtrain → SFT**. Before constructing a command, the launcher checks the data, robot action representation, tactile profile, checkpoint lineage, and normalization. It defaults to a dry run; add `--execute` to run the operation.

```bash
python scripts/revo3_v1_trex.py train --help
python scripts/revo3_v1_trex.py serve --help
python scripts/revo3_v1_runtime.py --help
```

- [Training and inference](docs/revo3_v1/TRAINING.md): data preparation, staged training, tactile encoders, checkpoints, and serving.
- [System architecture](docs/revo3_v1/README.md): Planner, state machine, timing protocols, model identity, and control interfaces.
- [Teleoperation and data collection](teleop_data_collection/README.md): public SDK adapters, device configuration, recording, and data export.

The online mainline uses **Profile A: Force6D + five-finger DIFF**. Data, training, and serving components also provide **Profile B: DIFF-only**; the current online assembly remains fixed to Profile A. Pressure/matrix Profile C is reserved and is not enabled in the current trainer or runtime.

## Repository layout

| Directory | Purpose |
|---|---|
| [`revo3_v1/`](revo3_v1/) | EMG, Planner, task state machine, policy interfaces, runtime, and control |
| [`qwen_vla/`](qwen_vla/) | Qwen-based multi-expert VLA model |
| [`tactile_vqvae/`](tactile_vqvae/) | Temporal tactile tokenizer and training tools |
| [`teleop_data_collection/`](teleop_data_collection/) | Native-rate recording, public hardware adapters, and export |
| [`scripts/`](scripts/) | Runtime, training, inference, and data entry points |
| [`config/`](config/) | Model/control configuration and integration-schema examples |
| [`tests/`](tests/) | Runtime, model-interface, timing, data, and safety tests |
| [`dataset_quickstart/`](dataset_quickstart/) | Upstream T-Rex dataset browsing and replay tools |
| [`hardware_code/`](hardware_code/) | Upstream bimanual hardware reference stack |

## Hardware operation

Before enabling writes, configure device identity, joint conventions, sensor clocks, and calibrated safety limits. SDK calls use explicit interfaces. Retain arming, watchdogs, physical E-stop, and on-site operator supervision; example configurations disable hardware writes by default.

## License and attribution

The project retains its [MIT License](LICENSE). External SDKs, model weights, datasets, and third-party components are governed by their own terms; see [NOTICE](NOTICE.md) and the source records in the data-collection guide. When citing the original T-Rex work, identify this repository as the Revo 3 integration. The original paper's citation is provided below.

## Citation

If this repository is useful, cite the original T-Rex paper and clearly describe this Revo 3 integration as a Revo 3 integration rather than an upstream result.

```bibtex
@misc{trex2026,
  title         = {T-Rex: Tactile-Reactive Dexterous Manipulation},
  author        = {Dantong Niu and Zhuoyang Liu and Zekai Wang and Boning Shao and Zhao-Heng Yin and Anirudh Pai and Yuvan Sharma and Stefano Saravalle and Ruijie Zheng and Jing Wang and Ryan Punamiya and Mengda Xu and Yuqi Xie and Yunfan Jiang and Letian Fu and Konstantinos Kallidromitis and Matteo Gioia and Junyi Zhang and Jiaxin Ge and Haiwen Feng and Fabio Galasso and Wei Zhan and David M. Chan and Yutong Bai and Roei Herzig and Jiahui Lei and Fei-Fei Li and Ken Goldberg and Jitendra Malik and Pieter Abbeel and Yuke Zhu and Danfei Xu and Jim Fan and Trevor Darrell},
  year          = {2026},
  eprint        = {2606.17055},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2606.17055}
}
```
