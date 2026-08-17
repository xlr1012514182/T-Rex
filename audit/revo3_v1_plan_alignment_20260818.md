# Revo 3 V1 功能对齐审计（2026-08-18）

## 审计边界

本审计把以下三份本地方案当作需求证据，并按“较新的明确决定覆盖较旧草案”的规则解释：

- `E:/BrainCo_Intern/第二周方案笔记.txt`
- `E:/BrainCo_Intern/V1暂定方案_第二周_补充版.txt`
- `E:/BrainCo_Intern/T-Rex_Revo3适配方案.txt`

T-Rex 的模型、训练和 slow/fast 协议事实以官方 `main` 为准；`full-pipeline` 只作为主分支缺失的训练实现参考。本轮只审查无需真实硬件即可验证的功能、数据契约和失败边界，不把 Mock、合成数据或官方异构 checkpoint 的 GPU smoke 当作 Revo 真机、任务成功或临床有效证据。

## 冻结主线与代码证据

| 需求 | 当前代码证据 | 功能结论 |
|---|---|---|
| EMG 仅产生抓型/释放原语，不进入 VLA | `revo3_v1/emg/`、`revo3_v1/executive/`；`PolicyObservation` 与 Revo dataset 均无 EMG；exporter 强制 `contains_emg=false` | 已实现并 fail closed |
| BrainCo 8ch@250Hz 显式适配 GNI，而不是补零伪造 16ch@2kHz | `emg/preprocessing.py`、`migration.py`、`streaming.py`；2 秒/500 点窗口，12/13 点交替步长，迁移报告逐 key 记录 | 组件级验证 |
| 五类主线与显式人工复核数据 | `POWER_GRASP/PRECISION_GRASP/LATERAL_GRASP/RELEASE/REST`；`recording/export_emg.py`；二分类只保留显式 fixture 入口 | 已对齐 |
| 五类合成 CLI 能独立跑通生成→训练 smoke | 生成器默认 8ch@250Hz/2s/五类并写 profile metadata；训练 smoke 必须显式 `from-scratch + window-reset fallback` | 已实跑 1 epoch；仅流程夹具 |
| Planner 是必经主线，输入锁存原语 + 最近三帧 full + 当前 center | `planner/schema.py`、`planner/planner.py`、`planner/async_worker.py` | 已实现；真实 LoRA 数据/权重待提供 |
| Qwen Planner 严格 JSON、原语不变、一次 repair、LoRA lineage | `planner/backends.py`、`artifacts.py`、`training.py`、`lora_sft.py` | 组件级验证；基础模型只允许 smoke 消融 |
| 单相机、确定性 full/center、无 SAM/检测/跟踪 | `revo3_v1/vision/`、`planner/visual_gate.py`；两路同 capture timestamp、384x288 | 已实现 |
| 所有任务门控统一到一个 Task Executive | `revo3_v1/executive/task_executive.py` 对外仅输出 WAIT/START/CONTINUE/HOLD/REPLAN/COMPLETE/ABORT | 已实现 |
| StartIntent 锁存；REST 不取消；Release 抢占但不能绕过 Safety | `MulticlassIntentGate`、`TaskExecutive` | 已实现 |
| 150 ms 原子提交、旧 Planner/旧 lease/旧版本结果拒绝 | `TaskExecutive` candidate signature、`validate_policy_response`、async worker generation | 已实现 |
| 接触前最多一次 REPLAN；接触后无进展不得重规划 | `CompletionStatus.CONTACT_ESTABLISHED`、`TaskExecutive` PRECONTACT/CONTACT 分支 | 已实现 |
| 稳定抓持进入 HOLD，只有显式 Release 才受控张开 | `revo3_v1/revo/completion.py`、`TaskExecutive`、`RevoServoExecutor` | 已实现；阈值待真机标定 |
| COMPLETE/ABORT 后统一清除任务局部状态 | `DoubleRateRuntimeService.acknowledge_terminal()` 在下一新鲜 control commit 原子复位 Executive、EMG、Planner、Completion、policy、action clock 与 Servo；ABORT 另需 operator/bench 确认和 confirmed SoftStop | 已实现；终态不自动吞掉，ABORT 不自动开手 |
| servo IO / runtime shutdown 有界且 fail-closed | 配置冻结 control watchdog / servo IO / IO close 为 100/50/1000 ms；servo timeout/error 取消并 join 双循环、TaskExecutive ABORT、confirmed SoftStop；IO close timeout 和未确认 stop 标记 intervention/unclean | 已实现；真实 SDK 仍需满足 cancellation-safe RuntimeIO 契约并经台架验证 |
| T-Rex 输入为 full/center + language + q[21] + 当前 profile 触觉，无 EMG | `policy/contracts.py`、`zmq_backend.py`、`scripts/test.py` | 已实现 |
| state/action 为连续绝对关节角 `[21]` / `[16,21]` | `revo/contracts.py`、`policy/contracts.py`、dataset/launcher | 已实现，禁止截取官方 `[16,62]` |
| Profile A 使用真实原生 `[16,5,6]` Force6D history + DIFF | recorder exporter、`data/native_touch.py`、`tactile_vqvae/data/revo3.py` | 已实现数据/运行契约；真实传感器待接入 |
| Profile B 为 DIFF-only，禁止假 Force6D | `policy/tactile_profile.py`、contracts、dataset/server/launcher | 已实现 |
| Profile C 不得 reshape 成 `[5,6]` | 当前 trainer/runtime 对未经验证的 pressure/matrix adapter fail closed | 正确阻断，依赖真实 schema 后再实现 |
| Revo VQ 与 DIFF encoder 从头训练，不能复用 Sharpa 权重 | VQ/Deform 训练入口与 artifact validator 固定 sensor family、5 指、capability/split/hash | 已实现入口与证据门 |
| 官方 Pretrain 是主线，官方 Midtrain 仅异构消融 | `config/revo3_v1_trex.json`、`scripts/revo3_v1_trex.py` | 已实现 |
| action/state/tactile embodiment 边界必须显式重建 | `policy/checkpoint.py` allowlist 报告；同形 tactile-specific key 也强制重置 | 已实现 |
| 30Hz action、chunk16、offset 0/4/8/12、完整执行与 temporal aggregation | `policy/schedule.py`、aggregation、runner、server/client | 已对齐官方 main |
| GPU 推理不得阻塞控制线程，旧结果必须丢弃 | `policy/async_runner.py`、`planner/async_worker.py` | 已实现有界后台 worker |
| 30Hz 目标插值到 100Hz，只有一个 motor writer | `revo/servo.py`、`revo/pipeline.py` | 已实现依赖注入执行层 |
| T-Rex tactile expert 输出完整 continuation action，不是外部 residual | 模型 flow 路径；外部 residual 仅 `tactile/reflex.py` | 已分离 |
| CAIR 为 12Hz、有界 synergy residual；load relief/anti-slip 默认关闭 | `tactile/synergy.py`、`tactile/reflex.py`、control config | 组件级验证；真实阈值待标定 |
| state/collision/stall/current/status/touch/lease/version 最终安全否决 | `revo/safety.py`、`pipeline.py`、`servo.py` | Mock/fake 边界已验证；真机值待冻结 |
| ABORT、GPU/EMG/touch 故障不得自动开手 | Task Executive + Servo 的 SAFE_STOP/HOLD 分支 | 已实现 |
| 原始 native-rate 先落盘，再因果生成 30Hz hand-only view | `teleop_data_collection/recording/` | 已实现；未来观测与 receive-after-decision 均拒绝 |
| action label 只能来自成功写入的 `exact_sent_target`，CAIR 不进入模仿标签 | recorder receipt、Revo exporter allowlist | 已实现 |
| 10Hz train anchors、未来 16 个 30Hz action、无 terminal/history/FLARE padding | `data/trex_json.py`、trainer | 已实现 |
| 先按 episode/day/object instance/operator 切分，统计仅来自 train | `data/splits.py`、normalization/VQ dataset | 已实现数据门 |
| W0/W1/midtrain/SFT 与 FLARE、dropout、分组学习率冻结值 | `policy/training.py`、launch config、`scripts/train.py` | 已编码；尚未用真实 Revo 数据训练 |
| SFT 语言约 50% manual + 50% frozen Planner | converter 的 instruction source 统计与比例门 | 已实现 |
| 四个限定任务的 Planner/原语/模拟流程 | bottle / phone / plastic_bag / refrigerator_door 的共享 task mapping 与 mock E2E | 组件级验证 |

## 明确不属于本轮“代码缺失”的阻塞项

- 真实 Revo 3 的关节顺序、限位、状态位、碰撞、SoftStop、单位与总线频率台架确认。
- 五个 VisionTouch SN、Force6D 模型、轴向/单位、DIFF schema、跨指时间偏差和噪声阈值。
- 真实鱼眼标定、相机 identity、曝光/模糊阈值。
- 真实 EMG 电极/通道顺序、连续多日数据、官方 GNI checkpoint 以及日校准效果。
- 真实 Planner 标注语料与 LoRA adapter。
- 从官方 tactile-free Pretrain 重新训练出来的 Revo midtrain/SFT checkpoint。
- Tianji 当前厂商 SDK/ABI、腕部 6DoF tracker、IK、payload/COM、急停和现场安全批准。
- 真实任务、用户实验、临床改善或论文指标。

## 声明边界

本轮允许的最高结论是 `component-verified` / `runnable mock integration`。官方 Qwen 与官方 T-Rex midtrain 的远程 GPU smoke 只证明各自官方形态能加载和做一次有界推理；它不证明官方 `[16,62]` 输出可用于 Revo `[16,21]`，也不证明 Revo 训练或整机闭环已经完成。
