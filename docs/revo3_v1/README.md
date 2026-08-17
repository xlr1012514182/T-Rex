# Revo 3 单手 21DoF 辅助操作 V1

本目录说明当前分支中已经实现的 Revo 3 V1 代码边界、接口、模拟运行方式，以及从 T-Rex 预训练权重开始训练 Revo 专用权重的路线。

> **当前验证等级：可运行的模拟管线与组件级接口验证。** 代码可以用合成 EMG、合成 RGB/状态/触觉和 Mock VLM、Mock T-Rex、Mock Revo 后端跑通整条链路；尚未在真实 Revo 3、真实 U21VT 触觉、真实截肢者或真实 Qwen/T-Rex 权重闭环上验证。模拟运行成功不代表抓取成功、功能改善或临床安全。证据矩阵和被阻塞项见 [`audit/README.md`](../../audit/README.md)。

## 1. V1 的固定范围

V1 只控制一只 Revo 3 手，不控制机械臂，也不负责人体手臂的移动。限定任务为：

| CLI 任务名 | Planner 规范化指令 | 手部职责 |
|---|---|---|
| `bottle` | `Grasp the centered bottle with a power grasp and hold it securely.` | 抓住并保持瓶子 |
| `phone` | `Grasp the centered phone with a precision grasp and hold it securely.` | 抓住并保持手机 |
| `plastic_bag` | `Grasp the handles of the centered plastic bag and lift it.` | 抓住袋提手；抬升位移由用户手臂完成 |
| `refrigerator_door` | `Grasp the refrigerator door handle and pull the door open.` | 抓住门把手；拉门位移由用户手臂完成 |

V1 的研究主线是：EMG 只表达 `OPEN/CLOSE` 动作意图，RGB 补全目标和任务语义，T-Rex 只接收规范化语言、RGB、Revo 连续状态和触觉，并输出 21 维手关节动作。

```text
多通道 EMG
  -> GNI-style 二分类器
  -> 去抖后的 CLOSE / OPEN 事件 --------------------------+
                                                            |
随手单相机 -> 上游鱼眼矫正后的 RGB -> Ask-to-Clarify Planner |
                                   -> 目标类别、结构化框、指令 |
                                                            v
                                         单一 Task Executive
                                   WAIT / START / CONTINUE / HOLD
                                   REPLAN / COMPLETE / ABORT
                                                    |
                                                    v
          RGB + Language + q[21] + tactile[5,6]/history[16,5,6]
                           -> Revo 专用 T-Rex -> chunk[16,21]
                                                    |
                TactileReflex-inspired 有界关节残差（可插拔）
                                                    |
                            关节/步长/电流/碰撞/过载安全裁剪
                                                    |
                                     Revo 3 SDK / Mock backend
```

这里**没有 SAM、检测器或像素级跟踪器**。Planner 的 VLM 自己返回当前 RGB 中的归一化 `xyxy` 目标框；`VisualGate` 只检查连续两次结构化输出的类别、面积、中心和变化是否稳定，不读取分割掩码，也不做像素/实例级跟踪，只保留上一条结构化决策用于门控稳定性检查。鱼眼到普通 RGB 的矫正网络同样不在本仓库中，必须由相机采集层在进入 Planner 前完成。

## 2. 已实现模块

| 模块 | 代码 | 当前职责 |
|---|---|---|
| EMG 数据与分类 | `revo3_v1/emg/` | 生成合成 `OPEN=0/CLOSE=1` 数据，训练 GNI-style 二分类器，流式输出去抖事件 |
| RGB+意图 Planner | `revo3_v1/planner/` | Ask-to-Clarify 风格的严格 JSON 规划；Mock 后端或延迟加载的 Qwen3-VL-2B 后端 |
| 视觉就绪判定 | `revo3_v1/planner/visual_gate.py` | 校验 VLM 结构化框，不做检测、分割或跟踪 |
| 单一任务权威 | `revo3_v1/executive/` | 锁存任务、版本化 lease、处理启动/继续/保持/重规划/释放/中止 |
| 多速率对齐 | `revo3_v1/timing/` | 低频或有理数 GCD 网格上的因果 latest-not-after 取样 |
| Revo T-Rex 边界 | `revo3_v1/policy/` | `[21]` 状态、`[16,21]` 绝对关节目标、slow/fast 缓存和时间聚合 |
| Revo 硬件边界 | `revo3_v1/revo/` | 21DoF 规范顺序、SI 单位、Mock 后端、SDK 适配器、最终安全授权 |
| 触觉历史与反射 | `revo3_v1/tactile/` | `[5,6]` 已标定 F6 特征契约、16 帧真实历史、有界抓持残差 |
| 机器人合成数据 | `revo3_v1/data/` | 四任务 robot-only 合成 episode 和 T-Rex JSON 转换；明确不含 EMG |
| 整链模拟 | `revo3_v1/demo.py` | 将以上真实接口接到 Mock Planner、Mock T-Rex 和 Mock Revo |
| 受审计训练入口 | `scripts/revo3_v1_trex.py`、`config/revo3_v1_trex.json` | 先校验检查点、真实数据和 Revo 契约，默认只打印命令 |

## 3. EMG 二分类与事件语义

完整 `gni` 配置沿用 Generic Neuromotor Interface 离散手势网络的主要层次和尺寸：

```text
Reinhard 压缩 64*x/(32+|x|)
-> Conv1D(in_channels -> 512, kernel=21, stride=10)
-> ReLU + Dropout(0.1) + LayerNorm
-> 3 层 LSTM(hidden=512)
-> LayerNorm -> 2 类投影 -> 时间维均值池化
```

项目只把原方法的多手势输出改成 `OPEN/CLOSE` 两类。GNI 官方离散手势网络输出时间局部 logits；本项目的时间均值池化是面向二分类窗口标签的适配，不是原论文原样输出。训练默认使用 AdamW、`lr=5e-4`、`weight_decay=1e-4`、梯度裁剪 `0.5`，并只在训练集做电极环形旋转增强 `±2` 通道。`smoke` 配置保留相同拓扑，但把 Conv/LSTM 缩小到 32 维以便 CPU 快速验证。

合成数据包含 subject、session 和纳秒时间戳，并按 subject 整体划分 train/val/test；归一化只从 train manifest 拟合。它只用于检验数据和训练管线，不能代表真实人的肌电分布。

流式门控默认值为：

- `CLOSE` 概率至少 `0.80` 且持续 `300 ms`，产生一次 `StartIntentEvent`；
- 活动任务中 `OPEN` 概率至少 `0.90` 且持续 `500 ms`，产生一次 `ReleaseEvent`；
- 信号质量低于 `0.80` 时清空待定 dwell，但不会擅自释放已锁存任务；
- `REST/UNKNOWN` 不会结束活动任务。

EMG 的最高优先级是**任务意图层**：已确认的 `CLOSE` 可以启动并锁存任务，`OPEN` 可以请求受控张开；它不能绕过碰撞、过流、触觉硬过载、急停、陈旧状态或无效 lease 等最终安全否决。

### 3.1 生成与训练合成 EMG

```powershell
py -3.10 scripts/revo3_v1_generate_emg.py `
  --output outputs/emg_synthetic `
  --preset smoke

py -3.10 scripts/revo3_v1_train_emg.py `
  --dataset outputs/emg_synthetic `
  --output outputs/emg_smoke_run `
  --preset smoke `
  --epochs 3
```

真实数据应替换 `windows.npz` 和 manifests，同时保持信号形状 `[window, channel, sample]`、标签、subject/session 和单调时间戳契约。生成论文形状的合成夹具使用 `--preset gni-shape`，训练完整网络使用 `--preset gni`；它不是轻量 smoke，训练成本明显更高。

## 4. Ask-to-Clarify Planner 与视觉门

`AskToClarifyPlanner` 接收已经锁存的 `CLOSE` 和一张或多张矫正 RGB。它要求后端只返回 `revo3_planner_v1` JSON：任务枚举、归一化框、置信度、目标是否存在/接近/兼容、歧义状态、澄清问题及最终指令。自由文本、额外尾随内容、未知任务和不一致框面积会被拒绝。

V1 借鉴 Ask-to-Clarify 的“先识别歧义、必要时多轮澄清，再提交动作任务”结构，但**没有复现该论文的完整训练、扩散执行器或实验结果**。当前真实后端替换为：

- `Qwen/Qwen3-VL-2B-Instruct`；
- 固定 revision `89644892e4d85e24eaac8bacfd4f463576704203`；
- `do_sample=False`，默认最多生成 256 tokens；
- 首次 `generate()` 才加载模型和处理器。

真实后端最小 API 示例：

```python
import time
from PIL import Image

from revo3_v1.planner import (
    AskToClarifyPlanner,
    PlannerRequest,
    Qwen3VLBackend,
)

planner = AskToClarifyPlanner(Qwen3VLBackend())
decision = planner.plan(
    PlannerRequest(
        emg_action="CLOSE",
        images=(Image.open("rectified_rgb.png").convert("RGB"),),
        timestamp_ns=time.monotonic_ns(),
    )
)
print(decision.status, decision.instruction, decision.bbox)
```

这段真实 Qwen 路径目前只有接口实现，尚未在本项目四任务数据上做 SFT、标定或成功率评估。`MockPlannerBackend` 更弱：它根本不检查像素，只返回固定/脚本化结构化响应，因此只能用于 CI 和管线 smoke，不能替代视觉评测。

`VisualGate` 的代码默认门限为：框面积占图像至少 `0.15`，中心到图像中心距离不超过 `0.25`，Planner 置信度至少 `0.75`，连续 `2` 个递增时间戳的结果通过；相邻中心跳变不超过 `0.18`，面积相对变化不超过 `0.75`。这些都是待相机视场和用户操作数据标定的初值，不是安全或临床阈值。

## 5. 单一 Task Executive

`TaskExecutive` 是唯一任务状态权威，外部只消费以下七类输出：

| 输出 | 典型运动指令 | 含义 |
|---|---|---|
| `WAIT` | `NONE`，或无活动任务时受控张开 | 尚未同时满足稳定 EMG、Planner、视觉和时效条件 |
| `START` | `POLICY` | 创建任务 lease、锁存规范化指令、清空旧策略缓存 |
| `CONTINUE` | `POLICY` 或 `CONTROLLED_OPEN` | 继续当前策略，或执行显式释放后的确定性张开 |
| `HOLD` | `HOLD_POSITION` | 稳定抓持、安全暂挂或等待新的 Planner 结果 |
| `REPLAN` | `HOLD_POSITION` | 无进展时保持位置、失效旧 chunk 并请求一次重规划 |
| `COMPLETE` | `NONE` | 受控释放已完成，等待上层确认并调用 `reset_terminal()` |
| `ABORT` | `SAFE_STOP` | 硬安全、版本、时间戳或完成监视失败；故障锁存 |

关键语义：

1. `CLOSE` 通过后被系统锁存，用户随后放松导致的 `REST` 不会取消任务，也不要求持续收缩。
2. 启动仍需 Planner 与视觉门共同就绪，以及 camera/state/touch 时间戳新鲜。
3. 默认 camera/state/touch TTL 分别为 `100/50/150 ms`；Planner TTL 为 `1 s`，策略 TTL 为 `750 ms`。
4. `GRASP_STABLE` 或 `TASK_SUCCESS` 会进入 `STABLE_HOLD`；当前实现不会据此自动张开。
5. 稳定 `OPEN/RELEASE` 进入确定性 `CONTROLLED_RELEASE`，不再让 VLA 产生相互矛盾的张开 chunk；默认释放超时 `3 s`。
6. `NO_PROGRESS` 最多触发一次 `REPLAN`；第二次仍无进展会中止。
7. task id、task version、lease id、指令哈希和运行时版本指纹必须全部匹配，旧任务的异步策略结果会被拒绝。

当前仓库只定义了 `CompletionState` 输入契约，没有实现经过真机验证的 Completion Monitor。真实系统必须提供抓持稳定、无进展、失败和已释放的可靠判定；在此之前不能把模拟中的 `COMPLETE` 当作任务成功证明。

## 6. 多速率因果对齐

`CausalTimestampAligner` 支持两种网格：

- `lowest`：使用所有配置流中的最低采样率；
- `gcd`：对有理数频率求最大公约数。

每个 anchor 只选择 `sample.timestamp <= anchor` 的最新样本，禁止用未来样本做最近邻或插值。每路还独立检查最大允许 age。模拟配置为 EMG `50 Hz`、相机 `30 Hz`、Revo state `100 Hz`、touch `120 Hz`，GCD 触发网格为 `10 Hz`；这只负责跨模态会合，不把电机控制降成 10 Hz。

触觉缓冲区保存最后 16 个**真实新帧**。重复的 timestamp/sequence 不会被伪造为新历史；乱序或同一时钟下内容冲突会被拒绝。真实部署应让每个采集线程在采样时使用同一单调时钟打戳，网络接收时间不能冒充传感器采样时间。

## 7. Revo 专用 T-Rex 动作接口

当前固定契约为：

- 连续手状态 `q_rad: [21]`；
- T-Rex 输出绝对未来关节目标 `q_target_rad: [16,21]`；
- 当前触觉 `tactile_f6: [5,6]`；
- 触觉历史 `tactile_history_f6: [16,5,6]`；
- 语言是 Planner 锁存的 instruction；
- `PolicyObservation` 中**不存在 EMG tensor/token/embedding**。

控制频率固定为 `30 Hz`，完整执行 16 步 chunk，约 `0.533 s`。chunk offset `0` 执行 `slow_and_fast`，offset `4/8/12` 用新触觉做 `fast` refinement，其余步不重新推理。slow 和三个 fast 版本按 T-Rex main 的时间聚合方式合并；当前 `k=0`，即对覆盖当前步的有效版本做算术平均。task/version/instruction/lease 不一致、fast 先于 slow、offset 逆序或陈旧 chunk 都会被拒绝。

当前原始 F6 路径使用最新 `[5,6]` 触觉。16 帧历史已经在运行接口、数据接口和 server buffer 中保留，但只有在启用 Revo 专用 embedded VQ-VAE 时才用于离散触觉 code。官方 T-Rex VQ-VAE 面向两手 `10x6=60` 维输入，不能截断、补零或静默复用于 Revo `5x6=30` 维；若要启用 `--use_tactile_vqvae 1`，必须先训练并提供 Revo 专用 VQ-VAE、配置和归一化统计。

## 8. TactileReflex-inspired 插件和最终安全权威

`TactileReflexPlugin` 是受 TactileReflex 启发的 Revo 关节协同残差，不是该 IROS 工作的完整复现。它没有实现论文的双视触觉图像、全部三通道 proxy、控制器和实验协议，也没有在 U21VT 上标定。

当前插件只在收到新的触觉帧时更新；pre-contact 和 release 阶段残差归零。接触后可按噪声 median/MAD 滞回判定接触，根据目标法向力做小幅 tighten/loosen，超过 protect threshold 时回退，超过 hard-overload threshold 时交给安全层否决。`slip_enabled` 默认关闭。模拟值包括每步最大残差 `0.003 rad`、累计最大 `0.035 rad`，都必须由真实传感器噪声和台架试验重新标定。

动作唯一写入路径是：

```text
T-Rex 名义 q + 有界触觉 residual
-> SafetySupervisor
-> RevoCommand
-> RevoBackend.write_command
```

`SafetySupervisor` 对关节上下限、单步变化、电流、状态 age、碰撞、急停、触觉硬过载和 lease 做最终授权。`SafetyEnvelope.demo()` 使用 `[-pi, pi]` 和极宽电流阈值，仅供 Mock；绝不能用于真手或用户实验。

## 9. 模拟数据与整链 smoke

### 9.1 生成 robot-only Revo 数据并转换为 T-Rex JSON

```powershell
py -3.10 scripts/revo3_v1_generate_robot_demo.py `
  --output outputs/revo3_robot_demo `
  --episodes-per-task 2 `
  --frames 48
```

生成内容包括四任务的 RGB、`state_rad[21]`、控制器绝对目标 `action_target_rad[21]`、`tactile_features[5,6]`、严格时间戳、metadata 和统计文件。动作标签来源明确为 `controller_target`，不是 RL policy 内部量或从 state 推导的伪标签。JSON converter 要求输入已经因果重采样到 30 Hz，并把末尾不足 16 步的 chunk 用最后一个控制目标补齐。

合成 robot 数据明确写入 `contains_emg=false`。真实 VLA 数采同样只记录机器人侧 RGB、Revo state、控制器 action target、touch、instruction 和时间戳；EMG 应放在独立用户日志中，用于 Planner/Task Executive 研究，不进入 T-Rex 训练样本。

### 9.2 跑四任务整链模拟

```powershell
py -3.10 scripts/revo3_v1_demo.py --task bottle
py -3.10 scripts/revo3_v1_demo.py --task phone
py -3.10 scripts/revo3_v1_demo.py --task plastic_bag
py -3.10 scripts/revo3_v1_demo.py --task refrigerator_door
```

默认模拟稳定抓持后注入一个受控 `OPEN`，最终输出 `COMPLETE`。若只验证锁存和稳定保持：

```powershell
py -3.10 scripts/revo3_v1_demo.py `
  --task bottle `
  --hold `
  --trace outputs/revo3_bottle_trace.json
```

模拟使用的 `MockPlannerBackend`、`MockTReXBackend` 和 `MockRevoBackend` 都是确定性桩。输出中的 `commands_written` 只说明命令经过接口与安全管线，不能解释为真实抓取次数或成功率。

### 9.3 组件测试

```powershell
py -3.10 -m pytest -q tests/revo3_v1
```

当前审计环境结果为 `95 passed`；唯一警告来自环境中过旧的 `optree`，不属于 Revo3 代码失败。

## 10. 从 T-Rex pretrained 开始训练 Revo 权重

### 10.1 推荐主线

1. 用 BrainCo Revo3 遥操作/模仿采集真实 robot-only episode，保存相机原始采样时间、Revo state、实际下发的绝对关节目标、原始触觉和任务 instruction。
2. 对各传感流做因果对齐，生成 30 Hz 的 Revo JSON 训练记录；划分必须按 episode/session，而不是随机拆 frame。
3. 从官方 `miniFranka/T-Rex_pretrain_mecka22k_epoch1` 开始，构建 `action_dim=21`、`action_chunk=16`、`tactile_num_fingers=5` 的模型。
4. 精确形状相同的 backbone/MoT 参数可以载入；与 Revo 维度不兼容的 action/state/tactile 输出层跳过并重新初始化，禁止 pad/truncate。
5. 使用 `resume_source=pretrain`，重新初始化 tactile expert，在 Revo RGB+state+F6+action 数据上做项目自己的 stage-2 cascaded **midtrain-like** 适配。它调用的是本仓库 main 的训练路径；上游论文尺度 midtrain loader 位于 `full-pipeline` 且假设 62 维双手机器人，因此这里不宣称复现论文 midtrain。
6. 当前先走 raw F6：`use_tactile_vec=1`、`use_tactile_deform=0`、`use_tactile_vqvae=0`。Revo 专用 VQ-VAE 是后续独立训练项，不复用官方双手 VQ。
7. 官方 midtrained 权重只作为异构迁移消融。原始官方 midtrain 即使显式确认也会被 launcher 拒绝；必须先产生 Revo 兼容的 `training_args.json` 和 `revo3_migration.json`，证明已移除双手 embedded VQ 路径、重建 21 维输出头并通过迁移测试。

### 10.2 启动模板

先把官方 pretrain 仓库下载到本地，例如：

```bash
hf download miniFranka/T-Rex_pretrain_mecka22k_epoch1 \
  --local-dir /checkpoints/trex_pretrain

hf download Qwen/Qwen3-VL-2B-Instruct \
  --revision 89644892e4d85e24eaac8bacfd4f463576704203 \
  --local-dir /checkpoints/qwen3-vl-2b-8964489
```

官方 pretrain 仓库的 `model.pt` 位于下载目录下的
`checkpoint-0-610000/`，所以 `--resume_checkpoint` 要指向该子目录，不能只指向模型仓库根目录。

正式入口是 `scripts/revo3_v1_trex.py`，冻结参数来自 `config/revo3_v1_trex.json`。它默认 **dry-run**：先验证所有证据并打印底层 `accelerate` 命令，只有再次加入 `--execute` 才真正启动。当前入口只接受经过审查的 Revo JSON；上游 main 的 LeRobot converter 仍是 62 维双手契约，20 小时数据迁移到 LeRobot 必须等专用 21 维 converter 通过 schema、replay 和时间对齐门后再开放。

训练前必须同时提供：

- 官方 pretrain 子目录及准确 checkpoint id；
- 真实 `revo3-trex-json-v1` 数据和同名 statistics；
- converter 生成的 `revo3-trex-conversion-v1` manifest；
- `revo3-vla-readiness-v1` readiness manifest，其中必须明确 `dataset_kind=real_robot`、`ready_for_training=true`、`synthetic_fixture=false`、`contains_emg=false`、`action_label_source=controller_target`、时间对齐和 replay gate 均通过；
- conversion manifest 所指 source root 下至少两个真实 episode，且各自 `meta.json` 明确 `synthetic_fixture=false`、`contains_emg=false`。

合成数据即使手工伪造 readiness 也会因 source episode metadata 被拒绝。训练 dry-run 命令为：

```bash
python scripts/revo3_v1_trex.py train \
  --base-model /checkpoints/qwen3-vl-2b-8964489 \
  --checkpoint /checkpoints/trex_pretrain/checkpoint-0-610000 \
  --checkpoint-id miniFranka/T-Rex_pretrain_mecka22k_epoch1 \
  --data-json /data/revo3/revo3_trex_train.json \
  --conversion-manifest /data/revo3/revo3_trex_train_manifest.json \
  --readiness-manifest /data/revo3/revo3_vla_readiness.json \
  --output-dir /runs/revo3 \
  --run-name revo3_v1 \
  --num-processes 1
```

人工复核打印出的命令、数据和设备资源后，原命令末尾加 `--execute`。冻结初始值为：stage 2、10 epochs、batch/GPU `2`、gradient accumulation `8`、AdamW `lr=1e-4`、`weight_decay=0.01`、warmup ratio `0.03`、图像 `384x288`、tactile MLP intermediate size `1536`、cascade `10/6`、tactile dropout `0.1`、validation ratio `0.1`。这些是首轮工程起点，不是已证明的最优配置。

当前 Revo 数据没有经过验证的 FLARE 未来帧目标，因此冻结配置明确关闭 FLARE。只有在真实数据 loader、图像命名、未来帧目标和时间对齐有验证产物后才能打开。

训练完成后可以用当前 ZMQ server 的 Revo 参数启动真实模型进程：

```bash
python scripts/revo3_v1_trex.py serve \
  --base-model /checkpoints/qwen3-vl-2b-8964489 \
  --checkpoint /runs/revo3/revo3_trex_midtrain_like/revo3_v1/checkpoint-X-Y \
  --stats-path /runs/revo3/revo3_trex_midtrain_like/revo3_v1/checkpoint-X-Y/stats_data.json \
  --cuda 0 \
  --port 5555
```

`serve` 同样默认 dry-run，并要求 checkpoint 带有 processor、`model.pt` 和匹配的 `training_args.json`；复核后加入 `--execute`。它拒绝含 EMG 或不符合 `21/16/5` 固定契约的 checkpoint。

仓库已提供与当前 `scripts/test.py` 严格对齐的 `ZmqTReXBackend`。它在 slow-and-fast 请求中发送 PNG RGB、instruction、`state[21]` 和 `tactile[5,6]`，在 fast 请求中只发送新触觉与 server chunk id；回复必须是有限的 `[16,21]` 绝对关节目标。它还校验 task/version/instruction/lease、本地缓存和 server chunk id，超时后销毁并重建 REQ socket，同时清空旧缓存。

```python
from revo3_v1.policy import (
    TReXPolicyRunner,
    TReXRevoPolicyAdapter,
    ZmqTReXBackend,
)

wire = ZmqTReXBackend(endpoint="tcp://127.0.0.1:5555", timeout_ms=5000)
runner = TReXPolicyRunner(TReXRevoPolicyAdapter(wire))
# 每个 30 Hz tick 将对齐后的 PolicyObservation 交给
# runner.infer_if_due(...)，再用 runner.target_for_step(...)取聚合目标。
```

V1 的正常调度只使用 offset 0 的 `slow_and_fast` 和 offset 4/8/12 的 `fast`。当前 server 在启用 cascaded tactile 时，纯 `slow` 有意返回空 action，因此客户端会拒绝该回复；纯 `slow` 只能与 server `--disable_tactile 1` 一起使用。真实 server 回复仍必须经过同一个 task lease、时间聚合、触觉 residual 和最终安全写入路径；当前 ZMQ 单元测试使用注入的传输桩，尚未运行真实 GPU server。

## 11. 真机接入前仍需提供和标定的信息

以下信息缺一不可；当前代码不会猜测它们：

1. **SDK 版本与 client 实例。** `BrainCoSDKBackend` 优先调用官方的原子批量反馈 `revo3_get_motor_status_data`，并单独读取电机 status；旧 wrapper 才回退到 position/velocity/current 分别读取。写入使用 `revo3_set_all_motor_positions`，碰撞优先使用批量查询。真实方法名和签名必须按实际安装的 `bc-revo3-sdk` 版本通过 CapabilityProbe 再次核验。
2. **硬件写入双重解锁。** 默认 `allow_hardware_write=False` 且 `capability_probe_confirmed=False`；只有完成急停、空载、限位、单位和低速台架检查后才可同时设为 `True`。官方单次 position/servo 写入不会替本项目在每个周期更新状态和碰撞判定；所有真实写入必须经过 `RevoCommandPipeline`。
3. **21 关节清单与方向。** 当前内部顺序来自 Revo3 retargeting 分支并由 hash 固定。还需确认左右手、零位、正方向、机械限位和每关节最大步长。
4. **单位。** 内部统一为 rad、rad/s、A。SDK position 默认按 degree，velocity 默认按 rpm 并使用 `rpm*2π/60`，current 默认按 mA→A；官方 Python 资料对 current 的 A/mA 表述存在冲突，所以必须在 CapabilityProbe 中实测，然后只在 Adapter 边界指定一次转换，禁止重复转换。
5. **U21VT 原始触觉 schema。** 当前策略契约需要每指 6 维 Force6D，共 `[5,6]`。必须提供真实字段、轴定义、单位、有效标志、频率和采样时间。若设备不是原生 Force6D，需要一个经过标定的 raw-touch-to-F6 profile；不能把未知原始通道直接 reshape 成 `[5,6]`。
6. **触觉噪声与反射阈值。** 分别采集无接触、静态接触、滑移、正常抓持和安全过载前的数据，拟合每指 median/MAD、目标力、protect 和 hard-overload；重新测量 closing synergy。
7. **相机标定。** 提供内参、畸变模型、随手外参、真实矫正网络/权重、输出分辨率和采样时间。Planner 和 VisualGate 只接受已经矫正的 RGB。
8. **EMG 采集参数。** 提供通道顺序、电极旋转关系、采样率、增益/单位、丢包/质量指标、受试者/session id 和同步时钟；真实训练必须做跨天、跨 session 和 subject 独立评估。
9. **Completion Monitor。** 定义每个任务的稳定抓持、释放完成、无进展和失败条件，并记录来源时间戳；不能在真实系统中硬编码模拟事件。
10. **逐级安全验收。** 先离线 replay，再无负载台架、软物体、健康受试者监督测试，最后才进入获批的截肢者研究。任何人体实验还需要伦理、知情同意、急停和责任人现场监督。

## 12. 来源与版本

- T-Rex 论文：[T-Rex: Tactile-Reactive Dexterous Manipulation](https://arxiv.org/abs/2606.17055)
- T-Rex 官方仓库：[ZhuoyangLiu2005/T-Rex](https://github.com/ZhuoyangLiu2005/T-Rex)
- 本项目 fork：[xlr1012514182/T-Rex](https://github.com/xlr1012514182/T-Rex)
- T-Rex pretrained：[miniFranka/T-Rex_pretrain_mecka22k_epoch1](https://huggingface.co/miniFranka/T-Rex_pretrain_mecka22k_epoch1)
- T-Rex midtrained 消融候选：[miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6](https://huggingface.co/miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6)
- Qwen Planner：[Qwen3-VL-2B-Instruct 固定 revision](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/tree/89644892e4d85e24eaac8bacfd4f463576704203)
- Ask-to-Clarify：[arXiv:2509.15061](https://arxiv.org/abs/2509.15061)
- GNI 论文：[A generic non-invasive neuromotor interface for human-computer interaction](https://www.nature.com/articles/s41586-025-09255-w)
- GNI 官方数据/代码：[facebookresearch/generic-neuromotor-interface](https://github.com/facebookresearch/generic-neuromotor-interface)
- TactileReflex：[arXiv:2605.23568](https://arxiv.org/abs/2605.23568)
- BrainCo Revo3 SDK：[BrainCoTech/brainco-revo3-sdk](https://github.com/BrainCoTech/brainco-revo3-sdk)
- Revo3 retargeting：[BrainCoTech/Revo-Retargeting `revo3_retargeting`](https://github.com/BrainCoTech/Revo-Retargeting/tree/revo3_retargeting)
- Revo3 ROS 2：[BrainCoTech/brainco_revo3_ros2](https://github.com/BrainCoTech/brainco_revo3_ros2)

## 13. 明确不支持的结论

当前实现不能用于宣称：已经复现 T-Rex 或 TactileReflex 论文结果；官方 T-Rex 触觉权重已兼容 Revo；Qwen Planner 已在四任务达到可靠 grounding；合成 EMG 可迁移到真实截肢者；真实 U21VT 已等价为 Force6D；真机闭环安全；或系统已经改善日常功能、认知负担、跨天稳定性和用户接受度。上述结论都需要独立真实数据、固定协议、真机视频/日志和人体研究证据。
