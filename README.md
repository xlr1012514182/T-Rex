<div align="center">

# T-Rex × Revo 3 V1

### EMG–RGB 任务规划 · 触觉反应式 VLA · Revo 3 单手 21DoF 安全运行时

[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/status-component--verified-yellow)](audit/README.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[English](README_EN.md) · [V1 详细文档](docs/revo3_v1/README.md) · [数采文档](teleop_data_collection/README.md) · [审计证据](audit/README.md) · [上游 T-Rex](https://github.com/ZhuoyangLiu2005/T-Rex)

</div>

> [!IMPORTANT]
> 本仓库是基于上游 **T-Rex: Tactile-Reactive Dexterous Manipulation** 的实验性 Revo 3 适配分支，不是上游作者发布的 Revo 3 官方实现。当前结论严格限定为 **component-verified**：模拟整链、接口、安全边界和官方权重的有界 GPU smoke 已通过；尚未完成真实 Revo 3/U21VT/截肢者闭环验证，不能据此声称真实抓取成功、功能改善或临床安全。

---

## 项目简介

本项目面向 **BrainCo Revo 3 单手 21 自由度灵巧手**，构建一套用于真实截肢者人机协同辅助操作的 V1 软件主线。系统把 EMG 作为上层意图来源，把单相机 RGB 用于补全目标和场景语义，再由 Qwen3-VL Planner 生成规范化语言指令；下游 Revo 专用 T-Rex 只接收语言、RGB、连续手状态与触觉，输出 21 维手部动作。最终动作经过有界触觉残差、100 Hz 唯一写入器与硬安全层后才能发送给灵巧手。

设计原则：

- **EMG 不进入 VLA embedding 或训练样本。** 它只产生抓型/释放原语与边沿事件，避免无预训练 EMG 模态直接扰动 VLA 表征。
- **Planner 是主线，不是模板快路径。** EMG 给出“怎么抓”，RGB 补充“抓什么、抓哪里”，VLM 输出完整指令；固定模板只作为实验基线。
- **单一 Task Executive 是唯一状态权威。** 所有启动、继续、保持、重规划、完成与中止逻辑统一收口。
- **触觉分层控制。** T-Rex 负责语义相关名义动作与中频触觉修正；CAIR/TactileReflex-inspired 插件只提供有界残差；最终安全层拥有否决权。
- **训练与在线 EMG 物理隔离。** 遥操数采的 VLA 投影只包含 robot-side RGB、Revo state、实际下发动作与触觉。

## 系统架构

```mermaid
flowchart TD
    E["BrainCo EDU EMG<br/>8 通道 · 250 Hz"] --> C["GNI-derived 五类分类器<br/>3 种抓型 + RELEASE + REST"]
    C --> G["StartIntent / Release 边沿事件"]
    R["单目鱼眼 RGB"] --> X["显式标定矫正<br/>同一采集生成 full + fixed center"]
    G --> P["Qwen3-VL Planner<br/>EMG 原语 + RGB → instruction"]
    X --> P
    P --> T["Task Executive<br/>WAIT · START · CONTINUE · HOLD<br/>REPLAN · COMPLETE · ABORT"]
    T --> V["Revo 专用 T-Rex<br/>L + RGB + q[21] + tactile → chunk[16,21]"]
    X --> V
    S["Revo state + VisionTouch/U21VT"] --> V
    V --> A["30 Hz slow/fast 调度与时间聚合"]
    A --> F["CAIR / TactileReflex-inspired<br/>有界关节残差"]
    F --> W["100 Hz 唯一写入器 + SafetySupervisor"]
    W --> H["Revo 3 SDK / Mock backend"]
```

视觉主线没有 SAM、目标检测器或实例跟踪器。相机层只执行显式物理标定的鱼眼矫正；Planner 从同一次采集得到 full view 与固定中心视图，并返回严格结构化目标框。系统用帧来源、时间戳、场景签名和结构化结果稳定性做门控，不引入另一套视觉模型。

## V1 固定任务

| CLI 名称 | 动作原语 | 规范化任务 | 手部边界 |
|---|---|---|---|
| `bottle` | `POWER_GRASP` | 抓住并稳定保持居中的瓶子 | 只负责抓持 |
| `phone` | `PRECISION_GRASP` | 精细抓住并保持居中的手机 | 只负责抓持 |
| `plastic_bag` | `PRECISION_GRASP` | 抓住塑料袋提手并保持 | 抬升由用户手臂完成 |
| `refrigerator_door` | `LATERAL_GRASP` | 侧向抓住冰箱门把手并保持 | 拉门由用户手臂完成 |

用户不需要持续收缩肌肉直到任务结束。抓取意图通过后由系统锁存；`REST/UNKNOWN` 不会取消活动任务。稳定 `RELEASE` 触发确定性受控张开，但任何 EMG 意图都不能绕过碰撞、过流、陈旧状态、触觉硬过载、无效 lease 或急停。

## 已实现能力

| 子系统 | 当前实现 |
|---|---|
| EMG | 8ch@250 Hz 五类 GNI-derived 分类、置信度/margin/质量/dwell、流式 Start/Release 事件、subject-exclusive 合成夹具与训练入口 |
| Planner | Qwen3-VL-2B 严格 JSON、三帧 full + 同采集 center、Ask-to-Clarify、LoRA artifact 校验、20 s SLA 与场景稳定性复核 |
| Task Executive | 唯一状态机、任务/版本/lease 锁存、150 ms 原子提交、接触前一次重规划、完成/中止显式确认 |
| T-Rex 适配 | 连续 `q[21]`、绝对动作 `chunk[16,21]`、30 Hz、offset `0/4/8/12` slow/fast refinement、时间聚合 |
| 触觉 | Profile A：16 帧原生 Force6D 历史 + 当前五指 DIFF；Profile B：DIFF-only；Profile C 默认阻断 |
| 控制与安全 | 独立 30 Hz control 与 100 Hz sole-writer、CAIR 有界残差、关节/速度/加速度/电流/碰撞/时效检查、确认式 SoftStop |
| 模型身份 | 服务端自行计算 checkpoint、lineage、normalization、profile 与 joint-order 身份；客户端握手和逐回复复核 |
| 数采 | glove/EMG/camera/Tianji/Revo/touch 原生频率记录、因果 30 Hz 投影、VLA 与 EMG 数据物理分离、exact-sent 动作标签 |
| 真机边界 | BrainCo EDU EMG/glove、Revo3 SDK、VisionTouch Force6D、鱼眼相机与 Tianji 通用适配器均已提供 fail-closed 接口；仍待现场接入 |

### 关键冻结参数

- Planner：`Qwen/Qwen3-VL-2B-Instruct@89644892...`，production `max_new_tokens=384`；128 token 只允许 smoke/消融。
- EMG：8 通道、250 Hz、2 s/500 点窗口，平均 50 ms 更新；抓取 dwell 300 ms，释放 dwell 500 ms。
- Policy：30 Hz、chunk 16（约 0.533 s），`slow_and_fast@0`，`fast@4/8/12`。
- Writer：100 Hz；control watchdog 100 ms；servo input timeout 50 ms；IO close timeout 1000 ms。
- 动作语义：21 维绝对关节目标，仓库内部统一使用 rad/SI；真实 SDK 单位只在 adapter 边界转换。

## 快速开始

### 1. 环境

```bash
conda create -n trex-revo3 python=3.10 -y
conda activate trex-revo3

pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

所有命令都应从仓库根目录执行。Windows 可把 `python` 替换为 `py -3.10`。

### 2. 跑通四任务整链模拟

```powershell
py -3.10 scripts/revo3_v1_runtime.py `
  --mode simulation `
  --task all `
  --servo-ticks 120
```

该命令走与 production 相同的双频 runtime、Task Executive、policy runner、servo 和安全边界，但使用确定性模拟输入/后端。成功输出必须包含四任务的 `START`、policy `READY`、非零授权写入、显式释放和 `shutdown_clean=true`。它只证明软件流程，不代表真实抓取成功。

### 3. 生成并训练合成 EMG 流程夹具

```powershell
py -3.10 scripts/revo3_v1_generate_emg.py `
  --output outputs/emg_synthetic `
  --preset smoke

py -3.10 scripts/revo3_v1_train_emg.py `
  --dataset outputs/emg_synthetic `
  --output outputs/emg_smoke_run `
  --from-scratch-ablation `
  --allow-window-reset-fallback `
  --preset smoke `
  --epochs 3
```

这是五类数据/训练代码的合成 smoke，不代表真人 EMG 泛化能力。旧 OPEN/CLOSE 二分类仅能通过显式 `--fixture-binary` 运行。

### 4. 测试

```powershell
py -3.10 -m pytest -q tests/revo3_v1
py -3.10 -m pytest -q teleop_data_collection/tests
py -3.10 -m pytest -q
```

当前提交验证结果为：Revo3 `289 passed`，数采 `137 passed`，全仓 `426 passed`。两个 warning 来自环境依赖弃用提示，不是测试失败。

## Revo 专用训练与推理

推荐主线从官方无触觉 pretrain 开始，重建 Revo 相关 state/action/tactile/DIFF/VQ 分支，再执行 `W0 → W1 → Revo midtrain-like → SFT`。官方 62 维双手 midtrain 只作为异构迁移消融，**禁止切片或补零成 21 维主线权重**。

```bash
hf download miniFranka/T-Rex_pretrain_mecka22k_epoch1 \
  --local-dir /checkpoints/trex_pretrain

hf download Qwen/Qwen3-VL-2B-Instruct \
  --revision 89644892e4d85e24eaac8bacfd4f463576704203 \
  --local-dir /checkpoints/qwen3-vl-2b-8964489
```

受审计 launcher 默认只做 dry-run。以下为 W0 命令骨架；真实执行还必须提供通过 readiness/replay/split/normalization/profile 检查的数据和 artifact：

```bash
python scripts/revo3_v1_trex.py train \
  --base-model /checkpoints/qwen3-vl-2b-8964489 \
  --checkpoint /checkpoints/trex_pretrain/checkpoint-0-610000 \
  --checkpoint-id miniFranka/T-Rex_pretrain_mecka22k_epoch1 \
  --resume-kind official_pretrain \
  --stage w0 \
  --data-json /data/revo3/revo3_trex_train.json \
  --conversion-manifest /data/revo3/revo3_trex_train_manifest.json \
  --readiness-manifest /data/revo3/revo3_vla_readiness.json \
  --development-data-json /data/revo3/revo3_trex_development.json \
  --development-conversion-manifest /data/revo3/revo3_trex_development_manifest.json \
  --development-readiness-manifest /data/revo3/revo3_vla_development_readiness.json \
  --tactile-profile profile_a_force6d_diff \
  --tactile-profile-manifest /data/revo3/tactile_profile.json \
  --output-dir /runs/revo3 \
  --run-name revo3_v1 \
  --num-processes 1
```

确认打印出的底层命令后，只有显式加入 `--execute` 才启动训练。normalization、Revo VQ 与 DIFF encoder 只能由 `MIDTRAIN_TRAIN` 拟合；SFT/development/locked test 必须复用同一冻结 artifact，避免数据泄漏。

启动 Revo checkpoint 服务同样默认 dry-run：

```bash
python scripts/revo3_v1_trex.py serve \
  --base-model /checkpoints/qwen3-vl-2b-8964489 \
  --checkpoint /runs/revo3/revo3_v1/checkpoint-X-Y \
  --stats-path /data/revo3/revo3_trex_midtrain_train_statistics.json \
  --stats-artifact-path /data/revo3/revo3_trex_midtrain_train_statistics_artifact.json \
  --identity-manifest-out /data/revo3/revo3_trex_server_identity.json \
  --cuda 0 \
  --port 5555
```

生产 runtime 先做 fail-closed 装配验证：

```bash
python scripts/revo3_v1_runtime.py \
  --mode production \
  --control-config config/revo3_v1_control.json \
  --runtime-config /data/revo3/runtime.production.json \
  --bindings-factory my_hardware.bindings:build_bindings \
  --validate-only
```

production 省略 `--servo-ticks` 时持续运行，直到 SIGINT/SIGTERM 或 supervisor 请求停止；退出路径必须完成双循环收拢、确认式 SoftStop、policy/planner/backend/IO 有界关闭，否则不会报告 clean shutdown。

完整训练数据契约、阶段参数和 artifact lineage 请阅读 [V1 详细文档](docs/revo3_v1/README.md) 与 [训练数据审计](docs/revo3_v1/TRAINING_DATA_AUDIT.md)。

## 遥操数采与硬件接入

独立目录 [`teleop_data_collection/`](teleop_data_collection/) 提供：

- BrainCo EDU EMG、BrainCo EDU glove、MANUS、RGB/鱼眼相机、Revo3/U21VT/VisionTouch 与 Tianji 的 source/backend 边界；
- native-rate 原始流与 30 Hz `latest-not-after` 因果 anchor；
- glove→Revo 与 6DoF wrist→Tianji 的可注入 retarget/IK 接口；
- `requested → authorized → exact_sent` 命令 receipt，只有成功写入的 `exact_sent` 名义目标可成为 VLA 动作标签；
- master episode 到 Revo VLA 数据、EMG 分类数据的 allowlist 物理分离导出；
- 默认禁止硬件写、SDK/插件哈希验证、capability/arming/watchdog/SoftStop/原子发布与 quarantine。

真实接入前仍需提供设备身份、关节顺序/限位、U21VT 五指 SN 与 Force6D 模型、相机 K/D/new_K、Tianji 当前 SDK/ABI、腕部 6DoF 来源、URDF/工具/负载参数、物理急停与台架批准。示例配置故意不可直接执行。

## 仓库结构

```text
revo3_v1/                 EMG、Planner、Executive、Policy、Runtime、安全与数据契约
scripts/                  EMG、训练、serve、runtime 与数据转换入口
config/                   冻结控制参数、T-Rex、production/hardware 示例与 schema
teleop_data_collection/   遥操、原生多模态数采、真机适配边界与导出
docs/revo3_v1/            V1 详细设计和训练数据审计
audit/                    复现实证、GPU smoke、矩阵、readiness 与声明边界
qwen_vla/                 上游/适配后的 Qwen VLA 模型代码
tactile_vqvae/            触觉 tokenizer/VQ-VAE
tests/                    Revo3 V1 测试
```

## 验证证据与限制

| 项目 | 已验证 | 不能推出 |
|---|---|---|
| 本地整链 | 四任务 mock runtime、双频调度、状态机、动作授权、释放与 clean shutdown | 真实抓取成功率 |
| Qwen GPU smoke | 固定 revision 可生成严格 schema 的 `ASK_CLARIFY`；观察峰值约 4540 MiB | 真实目标 grounding 或 planner 准确率 |
| 官方 T-Rex GPU smoke | 官方 midtrain 完整 tactile `slow_and_fast` 返回有限 `[16,62]`；峰值约 8674 MiB | Revo `[16,21]` checkpoint 可用性 |
| 数据/安全 | 因果对齐、exact-sent provenance、split/normalization lineage、fail-closed 单测 | 数据已获准训练或用户安全 |
| 硬件代码 | injected/fake client 与静态 SDK 边界 | 真实设备、固件、延迟或急停已验收 |

审计详情见 [`audit/README.md`](audit/README.md) 和 [`audit/revo3_v1_plan_alignment_20260818.md`](audit/revo3_v1_plan_alignment_20260818.md)。

---

## 上游 T-Rex

本分支保留并改造了原始 T-Rex 实现。关于 100 小时触觉反应式数据、异步 Mixture-of-Transformers、时序触觉 VQ-VAE、12 项任务实验以及论文级结论，请查阅并引用上游工作。

- [上游仓库](https://github.com/ZhuoyangLiu2005/T-Rex)
- [项目主页](https://tactile-reactive-dexterous.github.io/)
- [论文](https://arxiv.org/abs/2606.17055)
- [数据集](https://huggingface.co/datasets/zekaiwang/trex_dataset)
- [官方预训练权重](https://huggingface.co/miniFranka/T-Rex_pretrain_mecka22k_epoch1)
- [官方 midtrain 权重](https://huggingface.co/miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6)

## 许可证

本仓库采用 [MIT License](LICENSE)。可选硬件 SDK、下载的模型与数据集、外部源码仍分别受其自身许可证和再分发条款约束。

## 引用

如果本仓库对你的工作有帮助，请引用原始 T-Rex 论文，并将本 Revo 3 集成明确描述为实验性 fork，而非上游论文结果。

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
