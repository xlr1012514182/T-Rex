# Revo 3 数据、训练与推理

[项目首页](../../README_ZH.md) · [架构参考](README.md) · [数采与导出](../../teleop_data_collection/README.md) · [开发检查](../DEVELOPMENT.md)

本指南对应 Revo 单手 21 维绝对关节动作。EMG 分类、Planner LoRA、触觉编码器与 VLA policy 分开训练，再由运行时组合。

## 环境与入口

本地运行和单元测试使用 `requirements-dev.txt`。完整模型训练使用 Python 3.10、PyTorch 2.6.0 和与运行环境匹配的 CUDA；依赖统一维护在根目录 `pyproject.toml`，安装入口为 `pip install -r requirements.txt`。DeepSpeed 训练环境建议使用 Linux。选择 CUDA wheel 时，以 PyTorch 官方安装说明为准。

```bash
python scripts/revo3_v1_trex.py train --help
python scripts/revo3_v1_trex.py serve --help
python scripts/revo3_v1_train_emg.py --help
python -m revo3_v1.planner.lora_sft --help
```

启动器首先校验输入 artifact，默认只打印构造出的命令；显式 `--execute` 才启动训练或服务。

## 数据契约

- state 为连续 `q_rad[21]`；监督 action 为未来绝对关节目标 `[16,21]`。
- action label 只能是控制器确认的 `accepted_exact_sent_teleop_target`。requested、measured、CAIR residual/authorized command 均不能冒充 nominal policy label。
- 物理单相机产生同 timestamp 的两个输入：`full -> slow/head`、`fixed_center -> fast/wrist`，均为 `384x288`。没有 center view 时失败，不能静默退化成单图。
- 30 Hz action grid、10 Hz anchor、future 16、touch delay `0/4/8/12`。末尾 action/FLARE/授权段不足时剔除 anchor，不做 terminal padding。
- FLARE 在 midtrain/SFT 使用 8 个 future full frames、stride 4、loss `0.5`；W0/W1 关闭。collator 对 Revo 缺 future 立即失败，不复制当前帧。
- raw EMG 递归禁止进入 episode、JSON 和 policy batch。
- camera capture/receive、state、touch、decision/write、controller sequence 与 request SHA 都进入数据契约；验证 `capture <= receive <= decision <= write`、观测 age 和 command latency。
- 可记录 context/pre-roll，但只有显式 `policy_loss_eligible=true` 的连续授权段产生 policy loss；future action 与 FLARE 也必须留在授权段。
- split 在统计量之前固定；manifest 强制 locked test 同时按 day 与 object instance OOD 隔离（day-only 或 object-only 均拒绝），并声明 `12h/4h/2h/2h` 时长目标。normalization 只能由 `MIDTRAIN_TRAIN` 生成一次并冻结为 JSON + companion artifact；SFT、development 和 locked test 不得参与拟合或重算，只能加载并核验完全相同的 statistics/artifact SHA。
- SFT instruction 带 `manual_canonical|frozen_planner` 来源和 Planner revision/hash；SFT 转换强制 Planner 比例在 40%–60%。
- Planner LoRA 的 manifest 固定 OOD 维度为 `object_instance_id + day_id`；OOD 与 train/val/ID-test 在两维上分别不相交，不能用“同物体不同 scene”或“同日不同物体”绕过。

## 触觉 capability profiles

- Profile A：Force6D + native 16-sample history + 五指 DIFF；V1 首选。
- Profile B：DIFF-only；禁止用零 Force6D 伪造输入，训练与 serve statistics 不要求 `tactile_f6` block。
- Force6D-only：只允许显式 ablation。
- Profile C pressure/matrix：当前没有完整 native schema/trainer，所有 loader/launcher fail closed。

Force6D history 由采集端每个 anchor 导出 `[16,5,6]` 原生 ring、严格递增 timestamp 和 native sequence。converter/VQ 先按 sequence 去重，再取真实 native window；禁止从 30 Hz policy 帧切片或重复补齐。训练 temporal jitter 保留逻辑 `[-1,0,+1]`，实际只选 native offsets `[-2,-1,0]`，因此没有 future leakage。DIFF 对每个 delay 保留同一 profile family 的五指图和真实时间戳。

## Revo 专用触觉预训练组件

Force6D VQ-VAE 固定为 per-finger、window 16、native stride 4、codebook 64、embedding 256、EMA `0.99`、commitment `0.25`、magnitude-weighted MSE、global batch 256。训练和归一化只读 manifest 的 MIDTRAIN_TRAIN，development 做验证；SFT 与 locked test 都不打开。

DIFF encoder 使用 `DeformAEInfer` 架构在 Revo 五指 DIFF 上从头训练；Sharpa 权重不允许作为 Revo 主线起点。DIFF 与 VQ 都只打开 MIDTRAIN_TRAIN 做拟合、development 做选择，SFT/locked 不打开。两类 companion artifact 都记录并哈希 train/development episode ID，逐项核验 checkpoint SHA、source split/sensor family、profile、五指/形状、checkpoint/normalization family、capability SHA、split SHA 与 `locked_test_opened=false`。模型 loader 使用 strict key/shape 覆盖；缺 key 或额外 key 都失败。

serve 必须同时提供 frozen statistics 与 companion artifact。launcher 和 `scripts/test.py` 进程各自重新哈希，并要求 artifact、`training_args.json`、`checkpoint_lineage.json` 在 MIDTRAIN source、profile/family、capability/split 和 model SHA 上全部一致；同 shape 但内容不同的 statistics 也会失败。

## 训练阶段与参数组

默认训练主线为：

1. W0：官方 `T-Rex_pretrain` -> Revo graph，1000 steps；不启用 tactile/FLARE。
2. W1：W0 -> Profile A tactile graph，1500 steps；不启用 FLARE。
3. Revo midtrain：W1 -> 22000 steps、最多 3 epochs、FLARE `0.5`。
4. SFT：Revo midtrain -> 6500 steps、最多 3 epochs、FLARE `0.5`。

全局有效 batch 固定 64；tactile dropout `0.10`、state dropout `0.05`。midtrain 参数组使用新 action/state/tactile `1e-4`、tactile expert `3e-5`、action expert 后四层及 inherited boundary `1e-5`、FLARE `5e-6`；SFT 对应降为 `5e-5 / 5e-6 / 5e-6 / 1e-6`。完整定义以 `revo3_v1/policy/training.py` 的 `STAGE_SPECS` 为准，`weight_decay=0.01`。视觉/VLM、Revo VQ 和 DIFF encoder 在 policy midtrain/SFT 中冻结。

官方 pretrain 迁移会强制重建所有 Revo dimension-bound 与 tactile/DIFF/VQ 参数，即使源 tensor 与目标碰巧同形。之后仅允许相邻 stage 声明的新模块：W0->W1 为 tactile 组件，W1->midtrain 为 FLARE；midtrain->SFT 必须 exact。未在 allowlist 中的 missing、unexpected、shape mismatch 都会终止。官方 midtrain 只允许独立 heterogeneous-hand ablation，并先经过显式迁移审查。

## 数据增强

仅 train 开启，development/locked test 关闭。full、center 与同一 sample 的所有 FLARE future 共用一组可复现参数：brightness ±20%、contrast ±15%、saturation ±10%、hue ±0.03、rotation ±3°、translation ≤3%；无 flip、无 random crop。Force6D 噪声只从 train split 的显式 context/no-contact 数据拟合 per-finger/per-axis robust center/scale/covariance，并按同一轴顺序采样；验证集不加噪声。


## 训练所需文件

每次训练提供 base model、父阶段 checkpoint、checkpoint id/resume kind、训练和 development JSON，以及各自 conversion/readiness manifest。两组数据共享冻结 statistics 和 split identity。Profile A 在 W1 及后续阶段还需 Revo 专用 VQ-VAE 和 DIFF encoder 及其 companion artifact。

官方上游 62 维双手权重不能通过切片或补零作为 Revo 21 维策略。主线从官方无触觉 pretrain 开始；官方 midtrain 只用于显式声明的异构迁移消融。

下面是 W0 命令结构。路径是用户准备的本地模型和数据，不是自动下载位置；先检查 dry-run 输出，再按需添加 `--execute`。

```bash
python scripts/revo3_v1_trex.py train \
  --stage w0 --mode main \
  --base-model checkpoints/qwen3-vl-2b \
  --checkpoint checkpoints/trex-pretrain \
  --checkpoint-id miniFranka/T-Rex_pretrain_mecka22k_epoch1 \
  --resume-kind official_pretrain \
  --data-json data/revo/midtrain.json \
  --conversion-manifest data/revo/midtrain_manifest.json \
  --readiness-manifest data/revo/midtrain_readiness.json \
  --development-data-json data/revo/development.json \
  --development-conversion-manifest data/revo/development_manifest.json \
  --development-readiness-manifest data/revo/development_readiness.json \
  --tactile-profile-manifest data/revo/tactile_profile.json \
  --output-dir outputs/revo-w0
```

W1、midtrain、SFT 分别接续父权重，`--resume-kind` 对应使用 `revo_w0 / revo_w1 / revo_midtrain`，不绕过父阶段和 shape/key 检查。各阶段具体参数与 global batch 限制在启动器和训练器中同时校验。

## 推理服务

```bash
python scripts/revo3_v1_trex.py serve \
  --base-model checkpoints/qwen3-vl-2b \
  --checkpoint checkpoints/revo-sft \
  --stats-path data/revo/statistics.json \
  --stats-artifact-path data/revo/statistics_artifact.json \
  --identity-manifest-out outputs/revo-server-identity.json
```

服务启动前核对 checkpoint、lineage、统计量和传感器 profile。在线运行时使用服务生成的身份清单握手；仅形状相同不代表权重、触觉标定或归一化相容。

## 其他训练组件

- [EMG](../../revo3_v1/emg/README.md)：GNI 兼容层迁移、五类原语训练与日常校准。
- [触觉 VQ-VAE](../../tactile_vqvae/README.md)：原生历史编码、重建及离散码。
- DIFF encoder：`python scripts/train_revo_deform_ae.py --help`。
- Planner：`revo3_v1/planner/training.py` 校验三帧全图加中心图、标注和 OOD 隔离，`lora_sft.py` 执行 LoRA 训练。

合成数据与小规模流程训练集中在[开发指南](../DEVELOPMENT.md)，与真实训练入口分开使用。

## 上游工具

根目录 shell wrapper 从脚本位置解析项目路径，把参数转交给相应 Python CLI；不激活固定 Conda 环境，也不依赖作者机器目录。`scripts/train.sh` / `test.sh` 保留通用 T-Rex 训练与服务入口；Revo 训练推荐使用上述校验式 launcher。
