# Revo3 V1 数据、训练与迁移契约审计

审计日期：2026-08-18。范围仅覆盖 Revo policy 数据、T-Rex 训练、触觉 VQ/DIFF 组件与 checkpoint lineage；不构成真机安全批准，也不声称已在真实 Revo3/U21VT 数据上得到成功率。

## 已冻结并接入代码的主线

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

## T-Rex 训练 lineage 与优化参数

唯一主线为：

1. W0：官方 `T-Rex_pretrain` -> Revo graph，1000 steps；不启用 tactile/FLARE。
2. W1：W0 -> Profile A tactile graph，1500 steps；不启用 FLARE。
3. Revo midtrain：W1 -> 22000 steps、最多 3 epochs、FLARE `0.5`。
4. SFT：Revo midtrain -> 6500 steps、最多 3 epochs、FLARE `0.5`。

全局有效 batch 固定 64；tactile dropout `0.10`、state dropout `0.05`。optimizer groups 为新 action/state `1e-4`、tactile expert `3e-5`、action expert 后四层 `1e-5`、FLARE `5e-6`，`weight_decay=0.01`。视觉/VLM、Revo VQ 和 DIFF encoder 在 policy midtrain/SFT 中冻结。

官方 pretrain 迁移会强制重建所有 Revo dimension-bound 与 tactile/DIFF/VQ 参数，即使源 tensor 与目标碰巧同形。之后仅允许相邻 stage 声明的新模块：W0->W1 为 tactile 组件，W1->midtrain 为 FLARE；midtrain->SFT 必须 exact。未在 allowlist 中的 missing、unexpected、shape mismatch 都会终止。官方 midtrain 只允许独立 heterogeneous-hand ablation，并先经过显式迁移审查。

## 数据增强

仅 train 开启，development/locked test 关闭。full、center 与同一 sample 的所有 FLARE future 共用一组可复现参数：brightness ±20%、contrast ±15%、saturation ±10%、hue ±0.03、rotation ±3°、translation ≤3%；无 flip、无 random crop。Force6D 噪声只从 train split 的显式 context/no-contact 数据拟合 per-finger/per-axis robust center/scale/covariance，并按同一轴顺序采样；验证集不加噪声。

## 当前仍需真实证据，代码会 fail-fast

- Revo3/U21VT capability manifest、关节/单位/限位和 camera calibration SHA。
- 真实 20 小时 corpus、严格 split 时长与 OOD 隔离结果。
- controller receipt、action replay、RGB/label 人工复核和 readiness approval。
- 真实 native Force6D ring 与五指 DIFF 时钟/skew/age 统计。
- Revo VQ 和 DIFF 从头训练后的 checkpoint/artifact，以及 W0->W1->midtrain->SFT 实际 lineage。
- 真机 replay、安全 envelope、触觉阈值和用户实验均不在本审计批准范围内。

## 自动验证

在审计环境运行：

```powershell
D:\Embodied_AI_Master_Degree\python3_10_4\python.exe -m pytest -q
```

最终整仓结果：`425 passed`；其中 `tests/revo3_v1` 为 `288 passed`，
`teleop_data_collection/tests` 为 `137 passed`。两个环境警告分别来自旧版
`optree` 和 TensorFlow 对 distutils 的弃用提示。
