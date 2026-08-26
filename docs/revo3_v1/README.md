# Revo 3 单手 21DoF 辅助操作 V1

本目录说明当前分支中已经实现的 Revo 3 V1 代码边界、接口、模拟运行方式，以及从 T-Rex 预训练权重开始训练 Revo 专用权重的路线。

> **当前验证等级：可运行的模拟管线与组件级接口验证。** 代码可以用合成 EMG、合成 RGB/状态/触觉和 Mock VLM、Mock T-Rex、Mock Revo 后端跑通整条链路；尚未在真实 Revo 3、真实 U21VT 触觉、真实截肢者或真实 Qwen/T-Rex 权重闭环上验证。模拟运行成功不代表抓取成功、功能改善或临床安全。证据矩阵和被阻塞项见 [`audit/README.md`](../../audit/README.md)。

## 1. V1 的固定范围

V1 只控制一只 Revo 3 手，不控制机械臂，也不负责人体手臂的移动。限定任务为：

| CLI 任务名 | Planner 规范化指令 | 手部职责 |
|---|---|---|
| `bottle` | `Grasp the centered bottle with a power grasp and hold it securely.` | 抓住并保持瓶子 |
| `phone` | `Grasp the centered phone with a precision grasp and hold it securely.` | 抓住并保持手机 |
| `plastic_bag` | `Grasp the centered plastic bag handles using a precision grasp and hold them securely.` | 抓住袋提手；抬升位移由用户手臂完成 |
| `refrigerator_door` | `Grasp the centered refrigerator door handle using a lateral grasp and hold it securely.` | 抓住门把手；拉门位移由用户手臂完成 |

V1 的研究主线是：EMG 只表达 `POWER_GRASP / PRECISION_GRASP / LATERAL_GRASP / RELEASE` 动作原语（内部另有 `REST / UNKNOWN / BAD_SIGNAL`），RGB 补全目标和任务语义，T-Rex 只接收规范化语言、RGB、Revo 连续状态和触觉，并输出 21 维手关节动作。

```text
多通道 EMG
  -> GNI-derived 五类头（3种抓型 + RELEASE + REST）
  -> 置信度/margin/质量/dwell 后的 Start/Release 事件 -----+
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
| EMG 数据与分类 | `revo3_v1/emg/` | GNI-derived 五类原语头和流式事件；旧二分类仅保留为模拟兼容夹具 |
| RGB+意图 Planner | `revo3_v1/planner/` | Ask-to-Clarify 风格的严格 JSON 规划；三帧因果缓冲和单 worker 异步 Qwen 调用；Mock 或 Qwen3-VL-2B 后端 |
| 视觉就绪判定 | `revo3_v1/planner/visual_gate.py` | 校验 VLM 结构化框，不做检测、分割或跟踪 |
| 单一任务权威 | `revo3_v1/executive/` | 锁存原语/任务、150 ms 原子提交、版本化 lease、处理启动/继续/保持/接触前一次重规划/释放/中止；只消费 Revo 唯一 CompletionMonitor 的状态 |
| 多速率对齐 | `revo3_v1/timing/` | 低频或有理数 GCD 网格上的因果 latest-not-after 取样 |
| Revo T-Rex 边界 | `revo3_v1/policy/` | `[21]` 状态、`[16,21]` 绝对关节目标、slow/fast 缓存和时间聚合 |
| Revo 硬件边界 | `revo3_v1/revo/` | 21DoF 规范顺序、SI 单位、Mock 后端、SDK 适配器、最终安全授权 |
| 触觉历史与反射 | `revo3_v1/tactile/` | `[5,6]` 已标定 F6 特征契约、16 帧真实历史、有界抓持残差 |
| 机器人合成数据 | `revo3_v1/data/` | 四任务 robot-only 合成 episode 和 T-Rex JSON 转换；明确不含 EMG |
| 整链模拟 | `revo3_v1/demo.py` | 将以上真实接口接到 Mock Planner、Mock T-Rex 和 Mock Revo |
| 受审计训练入口 | `scripts/revo3_v1_trex.py`、`config/revo3_v1_trex.json` | 先校验检查点、真实数据和 Revo 契约，默认只打印命令 |

## 3. EMG 原语分类与事件语义

BrainCo 主线不是把 GNI 的 2 kHz 参数静默套到 250 Hz。生产 profile 固定为 8ch@250 Hz、40 Hz 四阶因果高通、2 s/500 点窗口、12/13 点交替步长（平均 50 ms）。主干保持 GNI 的 512 维 Conv、3 层 512 维 LSTM，但输入 stem 显式按采样域改为 kernel/stride `3/1`：

```text
Reinhard 压缩 64*x/(32+|x|)
-> Conv1D(8 -> 512, kernel=3, stride=1)
-> ReLU + Dropout(0.1) + LayerNorm
-> 3 层 LSTM(hidden=512)
-> LayerNorm -> 5 类投影 -> 时间维均值池化
```

主线学习 `POWER_GRASP / PRECISION_GRASP / LATERAL_GRASP / RELEASE / REST`；`UNKNOWN` 由低置信或低 top1-top2 margin 得到，`BAD_SIGNAL` 由采集质量门得到。官方 GNI checkpoint 迁移器只继承形状兼容的 LSTM/LayerNorm，8 通道 stem 与五类 head 重新初始化，并把官方仓库 commit、checkpoint SHA 和逐 key 报告写入训练 lineage；没有官方 checkpoint 时必须显式选择 from-scratch 消融。GNI 官方离散手势网络输出时间局部 logits；本项目的时间均值池化是窗口级原语适配，不是原论文原样输出。旧 `OPEN/CLOSE` 两类合成器和 `BinaryIntentGate` 只用于模拟 smoke。

合成数据包含 subject、session 和纳秒时间戳，并按 subject 整体划分 train/val/test；归一化只从 train manifest 拟合。它只用于检验数据和训练管线，不能代表真实人的肌电分布。

流式门控默认值为：

- 任一抓取原语置信度至少 `0.80`、top1-top2 margin 至少 `0.20` 且持续 `300 ms`，产生一次 `StartIntentEvent`；
- 活动任务中 `RELEASE` 置信度至少 `0.90`、margin 至少 `0.30` 且持续 `500 ms`，产生一次 `ReleaseEvent`；
- 信号质量低于 `0.80` 时清空待定 dwell，但不会擅自释放已锁存任务；
- 相邻有效更新间隔超过 `150 ms` 时清空待定 dwell，掉包不能靠首尾时间跨越阈值；
- `REST/UNKNOWN` 不会结束活动任务。

EMG 的最高优先级是**任务意图层**：已确认的
`POWER_GRASP / PRECISION_GRASP / LATERAL_GRASP` 可以启动并锁存任务，
`RELEASE` 可以请求受控张开；它不能绕过碰撞、过流、触觉硬过载、
急停、陈旧状态或无效 lease 等最终安全否决。

### 3.1 生成与训练合成 EMG

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

生成器默认输出五类、8ch@250 Hz、2 秒原始窗口的**合成流程夹具**；上面的
`--from-scratch-ablation --allow-window-reset-fallback` 仅用于验证五类训练代码，
并明确采用逐窗口零状态滤波，不能冒充真实 session 连续因果预处理。若需要回归旧二分类接口，
生成和训练命令都必须显式加 `--fixture-binary`。

真实数据的 `windows.npz` 还必须记录 250 Hz、严格 8 通道顺序、profile ID、`preprocessed` 和滤波 provenance。主线先按 session 连续执行因果高通，再切 500 点窗口；禁止对已预处理数据重复滤波。硬件 packet 必须调用 `push_many` 并提供每个 sample 的单调时间戳，20 点/80 ms packet 仍会产生全部 12/13 点步长推理。每日校准只更新 temperature/prototype/分类头并输出独立的 checkpoint/profile/channel/subject/day/session 绑定 artifact，不改 T-Rex。

## 4. Ask-to-Clarify Planner 与视觉门

`AskToClarifyPlanner` 接收已经锁存的抓取原语、最近三帧 `full_view` 和同时间戳的当前 `center_view`。它要求后端只返回 `planner_v1` JSON：原语、目标/部位/抓型、归一化 `cxcywh` 区域、三帧就绪数、置信度、可执行性、歧义、有限 reason code 和最终指令。原语改变、抓型改变、自由文本、未知目标或超过 25 token 的指令会触发且只触发一次修复请求；第二次仍不合法返回 `INVALID`，由 Task Executive 保持而不启动。

生产调用必须通过 `PlannerContextBuffer` 和 `AsyncPlannerWorker`：只接收严格递增、已矫正的三次采集，当前 center 与最新 full 来自同一 sequence/capture timestamp；Qwen 在唯一有界后台 worker 执行，不阻塞 servo/control loop。结果携带 generation、锁存 event ID、primitive、task version 和 request timestamp，任何一项变化或结果过期都会被丢弃。worker 不是第二个 Gate，最终是否 START/REPLAN 仍只由 Task Executive 决定。

V1 借鉴 Ask-to-Clarify 的“先识别歧义、必要时多轮澄清，再提交动作任务”结构，但**没有复现该论文的完整训练、扩散执行器或实验结果**。当前真实后端替换为：

- `Qwen/Qwen3-VL-2B-Instruct`；
- 固定 revision `89644892e4d85e24eaac8bacfd4f463576704203`；
- `do_sample=False`；production 固定 `max_new_tokens=384`。远程 exact-revision
  实测中 128 会截断严格 JSON，只允许作为显式 smoke/ablation；
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

planner = AskToClarifyPlanner(Qwen3VLBackend(
    adapter_path="outputs/planner_lora/adapter",
    production=True,
))
decision = planner.plan(
    PlannerRequest.from_aligned_views(
        primitive="POWER_GRASP",
        full_view_history=(full_t2, full_t1, full_t),
        center_view=center_t,
        full_view_timestamps_ns=(t2, t1, t),
        center_timestamp_ns=t,
    )
)
print(decision.status, decision.instruction, decision.bbox)
```

无 adapter 的 Qwen 后端被显式标记为 `base_model_smoke_ablation`；`production=True` 会拒绝启动。LoRA artifact 固定 base model/revision、processor schema/fingerprint、训练数据 audit hash 和 adapter 文件 hash，runtime 加载时逐项核验，其组合 hash 应写入 `RuntimeVersions.planner_revision`。当前仍未在本项目四任务真实数据上给出成功率证据。`MockPlannerBackend` 更弱：它根本不检查像素，只能用于 CI 和管线 smoke。

`VisualGate` 的代码默认门限为：框面积至少 `0.15`，中心距离不超过 `0.25`，Planner 置信度至少 `0.75`，且 Planner 在最近三帧中至少报告 `2` 帧就绪。它不运行检测、分割或跟踪器；跨调用只做结构化结果一致性检查。这些都是待真实数据标定的初值。异步 Planner 的 request/capture 时间与 produced/completed 时间分开保存：默认 SLA `20 s`、produced freshness `1 s`、source 最大年龄 `18 s`、pending intent `22 s`。即使结果刚生成，旧 source、旧 event/generation/task version 或与当前 full+center 低分辨率场景签名差异超过 `0.08` 都会被丢弃并重新规划，绝不会 START。

`ASK_CLARIFY` 时 Task Executive 始终输出 `WAIT/NONE`，不会无回答自动重提或执行。上层只能读取 `service.clarification_token`，再调用 `service.submit_clarification(answer, token=token)`；空答案、超过 256 字符、旧 event/generation/task version 都被拒绝。合法回答使用下一帧新鲜 RGB 进入新 generation，随后仍需完整 Planner、视觉、原子提交和安全门。

`revo3_v1.planner.training` 冻结并审计 Planner LoRA 配置：Qwen3-VL-2B、`rank=16`、`alpha=32`、`dropout=0.05`、`lr=5e-5`、最多 3 epochs、bf16。manifest 必须显式冻结 `ood_test_isolation=[object_instance_id, day_id]`，且 OOD 的物体实例和采集日分别都与 train/val/ID-test 不相交；“同物体换场景”或“同一天换物体”都仍算泄漏。命令构造前还会要求 `data_origin=real_camera`、每条记录恰有 3 个 full + 1 个 center 实图、时间戳因果对齐、结构化标签合法，且训练负/歧义比例为 30%–40%。缺图、mock/synthetic 标记或 split 泄漏均 fail closed；该构造器不下载模型。

## 5. 单一 Task Executive

`TaskExecutive` 是唯一任务状态权威，外部只消费以下七类输出：

| 输出 | 典型运动指令 | 含义 |
|---|---|---|
| `WAIT` | `NONE` | 尚未同时满足稳定 EMG、Planner、视觉和时效条件；无活动任务的 RELEASE 也不张手 |
| `START` | `POLICY` | 创建任务 lease、锁存规范化指令、清空旧策略缓存 |
| `CONTINUE` | `POLICY` 或 `CONTROLLED_OPEN` | 继续当前策略，或执行显式释放后的确定性张开 |
| `HOLD` | `HOLD_POSITION` | 稳定抓持、安全暂挂或等待新的 Planner 结果 |
| `REPLAN` | `HOLD_POSITION` | 无进展时保持位置、失效旧 chunk 并请求一次重规划 |
| `COMPLETE` | `NONE` | 受控释放已完成；至少对外保持一个 control tick，等待 supervisor 确认安全后调用 `service.acknowledge_terminal(safe_state_confirmed=True)` |
| `ABORT` | `SAFE_STOP` | 硬安全、版本、时间戳或完成监视失败；故障锁存 |

关键语义：

1. 任一抓取原语通过后被系统锁存，用户随后放松导致的 `REST` 不会取消任务，也不要求持续收缩。
2. 启动仍需 Planner 与视觉门共同就绪，以及 camera/state/touch 时间戳新鲜；Planner/视觉结果必须在当前锁存的 StartIntentEvent 之后生成，禁止复用 TTL 内的旧缓存决策。
3. `START` 前需同一候选在默认 `150 ms` 原子提交窗口内保持指令、视觉、profile、版本与安全身份不变；变化会撤销候选。默认 camera/state/touch TTL 分别为 `100/50/150 ms`；Planner TTL 为 `1 s`，最近已接受策略结果的运行时 freshness TTL 为 `750 ms`。GPU 请求超时独立设为 `2 s`；响应相对其观测的允许延迟按 mode 分开：slow/slow-and-fast `1.5 s`、fast `0.5 s`，task lease 为 `3 s`。超预算的同任务响应会被丢弃并进入 HOLD，由既有有界 stale timer 最终裁决，不误报成跨任务 ABORT。
4. `GRASP_STABLE` 或 `TASK_SUCCESS` 会进入 `STABLE_HOLD`；当前实现不会据此自动张开。
5. 稳定 `RELEASE` 进入确定性 `CONTROLLED_RELEASE`，不再让 VLA 产生相互矛盾的张开 chunk；旧二分类 fixture 的 `OPEN` 仅为兼容别名，默认释放超时 `3 s`。
6. `NO_PROGRESS` 只允许在接触前最多触发一次 `REPLAN`；接触建立后同一信号进入 fault-latched `ABORT/SAFE_STOP`，保持而不自动张手。
7. task id、task version、lease id、指令哈希和运行时版本指纹必须全部匹配，旧任务的异步策略结果会被拒绝。
8. camera/touch/policy 首先触发 `HOLD`；touch 持续 stale `500 ms`、camera/运行中 policy 持续 stale `1 s` 后 fault-latched ABORT，state stale 立即 ABORT。首个 T-Rex chunk 单独给 `3 s` startup budget，覆盖已实测的约 `1.22 s` production slow+fast 冷调用，而不会把运行中卡死容忍度放宽。
9. Profile A 需 16 帧 Force6D 历史，B 需五指 DIFF 就绪；Profile C 即使已有 pressure+valid-mask 也默认 fail-closed，只有显式绑定并验证对应 policy adapter 后才允许 START。
10. `COMPLETE/ABORT` 不会自行回到待机。连续 service 通过线程安全的 `acknowledge_terminal` 只登记确认，并在下一次新鲜 30 Hz 快照的原子提交点复位，避免 100 Hz writer 看见“已 reset 但尚无 IDLE decision”的空档。COMPLETE 要求任务后置安全条件已确认；ABORT 还必须有 operator/bench reset 确认且底层 SoftStop 已真实成功，随后才清除 pipeline fault latch。缺任一条件都不能重启，且任何路径都不会自动张手。
11. active task lease 一旦到期即 `ABORT/SAFE_STOP`，不得在控制线程恢复后续租“复活”。30 Hz control heartbeat 超时同样经唯一 writer SoftStop；只有底层 `soft_stop()` 实际返回成功才记录 confirmed shutdown。`runtime.control_watchdog_ms/servo_io_timeout_ms/io_close_timeout_ms` 冻结为 `100/50/1000`，且 servo IO deadline 不得大于 watchdog。`servo_input` 超时/异常会先取消并收拢双循环、锁存 IO intervention、再经唯一 pipeline SoftStop；任一 hard-fault servo result 都同步把 Task Executive 置为可观测 ABORT 后退出，SoftStop 未确认时 shutdown 永不标 clean。

唯一的 `revo3_v1.revo.CompletionMonitor` 已提供可注入的确定性实现：新鲜度与校准 ID 校验、两独立指区接触 dwell（单次 `CONTACT_ESTABLISHED` 边沿）、触觉/名义关节收敛、接触前无进展超时，以及“触觉卸载 dwell + safe-open”共同确认释放。Task Executive 只通过纯状态映射消费它，不再维护第二套完成检测器。阈值仍必须由真实硬件/profile 标定；在此之前不能把模拟中的 `COMPLETE` 当作任务成功证明。

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

主线 Profile A 同时使用最新 `[5,6]` Force6D、五指当前 DIFF `[5,1,240,240]` 和 Revo 专用 embedded VQ-VAE。在线 actor 每次 slow/fast 只接收五指最新 current DIFF 与逐指真实时间戳；训练 JSON 中的 delayed offsets 仍用于监督/增强，但不是在线强制依赖，旧 delayed wire 字段只做兼容忽略。VQ 输入是触觉设备原生 sequence 去重后的 16 个真实样本，不是从 30 Hz policy 帧复制或补齐出来的历史。官方 T-Rex VQ-VAE 面向两手 `10x6=60` 维输入，不能截断、补零或静默复用于 Revo `5x6=30` 维；必须在 Revo 数据上按 `window=16`、`stride=4`、逐指 code、`codebook=64`、`embed=256`、EMA `0.99`、commitment `0.25` 独立重训。DIFF encoder 同样必须从 Revo VisionTouch 数据从头训练，禁止复用 Sharpa encoder。两个组件都由 companion artifact 绑定 checkpoint SHA、传感器/profile/family/capability/split SHA，不能只凭 shape 加载。

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
  --frames 64
```

生成内容包括四任务的 RGB、`state_rad[21]`、控制器已接受且实际写入边界的绝对目标 `action_target_rad[21]`、`tactile_features[5,6]`、原生 16 样本触觉 ring、严格时间戳、receipt 和 metadata。动作标签语义固定为 `accepted_exact_sent_teleop_target`，不是未确认 requested target、实测 state、CAIR 后命令或 RL policy 内部量。JSON converter 要求输入有明确的 30 Hz 因果重采样证据；末尾不足 16 步 action、8 个 stride-4 FLARE future 或任一授权 future 的 anchor 直接剔除，禁止复制/补齐。

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

当前全仓审计结果为 `427 passed`；其中 Revo3 组件 `290 passed`、数采组件
`137 passed`。两个环境警告分别来自旧版 `optree` 和 TensorFlow 对 distutils 的弃用提示，均不属于 Revo3 代码失败。

## 10. 从 T-Rex pretrained 开始训练 Revo 权重

### 10.1 推荐主线

1. 用 BrainCo Revo3 遥操作/模仿采集真实 robot-only episode，保存相机原始采样时间、Revo state、实际下发的绝对关节目标、原始触觉和任务 instruction。
2. 先按 day、object instance、operator、episode 建立 split manifest；locked test 至少按 day/object 做 OOD 隔离。normalization 只允许从 `MIDTRAIN_TRAIN` 拟合一次并冻结为 statistics JSON + companion artifact；SFT、development、locked conversion 必须显式加载并验证同一 statistics/artifact SHA，禁止把 SFT 纳入统计或为每个 split 重算。生成 30 Hz 训练记录时保留真实 camera capture/receive、state、touch、decision/write、controller sequence 和 request SHA，不得伪造为同一 anchor timestamp。
3. 从官方 `miniFranka/T-Rex_pretrain_mecka22k_epoch1` 开始，构建 `action_dim=21`、`action_chunk=16`、`tactile_num_fingers=5` 的模型。
4. 官方 pretrain 到 W0 时只载入语义和形状均兼容的 backbone/MoT 参数。所有 Revo dimension-bound state/action 层和全部 tactile/DIFF/VQ 分支即使碰巧同形也强制重初始化；任何未在 allowlist 声明的 missing、unexpected 或 shape mismatch 都立即失败。
5. 固定 lineage 为 `official_pretrain -> W0(1000 steps) -> W1(1500) -> Revo midtrain(22000, <=3 epochs) -> SFT(6500, <=3 epochs)`。W1 引入 Profile A；midtrain 才启用 FLARE。相邻 Revo stage 必须 strict load 并核对 parent/model/profile/capability/split/artifact SHA，不能跳级。
6. 主线 Profile A 为 `use_tactile_vec=1`、`use_tactile_deform=1`、`use_tactile_vqvae=1`。Profile B（DIFF-only）和 Force6D-only 只作为明确实验分支；Pressure/Matrix Profile C 当前无 native schema/trainer，fail closed。
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
- 真实 `revo3-trex-json-v1` 数据，以及仅由 MIDTRAIN_TRAIN 生成的共享 frozen statistics JSON/companion artifact；
- converter 生成的 `revo3-trex-conversion-v1` manifest；
- `revo3-vla-readiness-v1` readiness manifest，其中必须明确 `dataset_kind=real_robot`、`ready_for_training=true`、`synthetic_fixture=false`、`contains_emg=false`、`action_label_source=controller_target`、时间对齐和 replay gate 均通过；
- conversion manifest 所指 source root 下至少两个真实 episode，且各自 `meta.json` 明确 `synthetic_fixture=false`、`contains_emg=false`。

合成数据即使手工伪造 readiness 也会因 source episode metadata 被拒绝。训练 dry-run 命令为：

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

人工复核打印出的命令、数据和设备资源后，原命令末尾加 `--execute`。冻结全局有效 batch 为 `64`（launcher 按 GPU 数计算 gradient accumulation），AdamW 基础 `lr=1e-4`、`weight_decay=0.01`、warmup ratio `0.03`、图像 `384x288`、tactile MLP intermediate size `1536`、cascade `10/6`、tactile dropout `0.1`、state dropout `0.05`。验证集必须是 split manifest 中独立 development JSON，禁止 frame 随机切分。

W0/W1 关闭 FLARE；Revo midtrain/SFT 固定启用 8 个 stride-4 future full-view、loss weight `0.5`。converter 或 collator 缺任一 future 都拒绝整个 anchor，绝不回退到当前帧。W1 及后续 Profile A 命令还必须显式提供 `--vqvae-checkpoint/--vqvae-artifact` 与 `--deform-encoder-checkpoint/--deform-encoder-artifact`。

训练完成后可以用当前 ZMQ server 的 Revo 参数启动真实模型进程：

```bash
python scripts/revo3_v1_trex.py serve \
  --base-model /checkpoints/qwen3-vl-2b-8964489 \
  --checkpoint /runs/revo3/revo3_trex_midtrain_like/revo3_v1/checkpoint-X-Y \
  --stats-path /data/revo3/revo3_trex_midtrain_train_statistics.json \
  --stats-artifact-path /data/revo3/revo3_trex_midtrain_train_statistics_artifact.json \
  --identity-manifest-out /data/revo3/revo3_trex_server_identity.json \
  --cuda 0 \
  --port 5555
```

`serve` 同样默认 dry-run，并要求 checkpoint 带有 processor、`model.pt`、`training_args.json` 和 `checkpoint_lineage.json`；复核后加入 `--execute`。launcher 与实际 server 进程都会重新计算 statistics、companion artifact 和 checkpoint SHA，并逐项核验 MIDTRAIN-only source split、profile/family、capability/split hash。`--identity-manifest-out` 把这次本地验证得到的 checkpoint、模型配置、training args、lineage、normalization、profile、split/capability 和关节顺序哈希固化成控制端 trust anchor；需通过受控文件传输交给运行时，不能用网络握手返回值反向覆盖它。Profile B 的 statistics 可以且应当不含 Force6D block；只有使用 Force6D vector/VQ/code 的 checkpoint 才强制读取该 block。

仓库已提供与当前 `scripts/test.py` 严格对齐的 `ZmqTReXBackend`。Revo profile 的 slow/slow-and-fast 请求发送同一物理 capture 派生的 full+fixed-center PNG、instruction、`state[21]`、完整 16 帧原生 Force6D history（Profile A）和五指最新 current DIFF；fast 请求继续携带完整因果 Force6D history、每次新到达的 current DIFF 及 server chunk id，绝不由 server 按请求次数伪造历史。回复必须是有限的 `[16,21]` 绝对关节目标，并回显 task/version/instruction/lease/version/timestamp，同时携带服务端启动时从磁盘计算、而非从请求回显的 `server_identity`。控制端在组装 production runtime 前先发 `mode=identity` 做 fail-closed 握手，随后逐动作回复复核相同身份；错误 checkpoint、缺失身份或 normalization/profile/lineage 任一不一致均在缓存动作前拒绝。`RuntimeVersions.policy_revision` 取验证后 `server_identity.identity_sha256`，不再接受运行时 JSON 中任意填写的 `policy_revision`。

```python
from revo3_v1.policy import (
    TReXPolicyRunner,
    TReXRevoPolicyAdapter,
    TReXServerIdentity,
    ZmqTReXBackend,
)
import json

with open("/data/revo3/revo3_trex_server_identity.json", encoding="utf-8") as handle:
    expected_identity = TReXServerIdentity.from_mapping(json.load(handle))
wire = ZmqTReXBackend(
    endpoint="tcp://127.0.0.1:5555",
    timeout_ms=5000,
    image_profile="revo3_full_center_v1",
    tactile_profile="profile_a_force6d_diff",
    expected_server_identity=expected_identity,
)
wire.probe_server_identity()
runner = TReXPolicyRunner(TReXRevoPolicyAdapter(wire))
# 每个 30 Hz tick 将对齐后的 PolicyObservation 交给
# runner.infer_if_due(...)，再用 runner.target_for_step(...)取聚合目标。
```

`revo3-v1-runtime-v1` 的 `artifacts` 还必须声明
`policy_server_identity_manifest`。production factory 读取该文件、执行实时握手并以验证后的服务端身份构造 `RuntimeVersions`；旧 `versions.policy_revision` 字段即使存在也不会被信任。

### 可执行双频 runtime factory

同一个 `OnlineV1Coordinator` 同时用于 mock smoke 和 production assembly：

```bash
python scripts/revo3_v1_runtime.py --mode simulation --task all --servo-ticks 120
python scripts/revo3_v1_runtime.py --mode production \
  --control-config config/revo3_v1_control.json \
  --runtime-config /data/revo3/runtime.production.json \
  --bindings-factory my_hardware.bindings:build_bindings \
  --validate-only
```

simulation 是明确标记的 scripted fixture，但四任务都会真实穿过
`StartIntentEvent -> Planner READY -> Task Executive START -> policy READY -> 100 Hz sole writer -> RELEASE`，报告 observed outputs、policy states 和授权 mock writes；这仍不是任务成功率。runtime 使用独立并发的 30 Hz control 与 100 Hz writer loop，接近共同 deadline 时只让 30 Hz commit 让位，writer 不让位；control 成功 heartbeat 超过保守预算即停止。任一 IO、shape、时间戳、Planner、policy、coordinator 或 servo 异常都先停止/join 两个 loop，再由 `RevoCommandPipeline.abort()` SoftStop。正常退出也 SoftStop；Revo backend-owned executor、Planner worker、T-Rex worker/socket 与 IO 都必须有界关闭，未确认 stop 或任一 close 不干净都会让普通 run 和 `--validate-only` 非零退出。

production 未显式给 `--servo-ticks` 时持续运行，SIGINT/SIGTERM 或 supervisor 的 `request_shutdown()` 都进入相同的 cancel/join -> confirmed SoftStop -> IO/backend/model close 路径；simulation 未指定时才使用 120 tick 的有界 smoke。simulation 的非实时桌面调度放宽了 servo jitter，仅证明功能接线，不能作为生产 2 ms jitter 证据。

production 必须提供非 Mock、`is_hardware=True` 且显式 `production_ready=True` 的 Revo backend，以及非 Synthetic、`is_hardware=True`、`raw_emg_only=True` 且支持 control/servo 并发快照的 `RuntimeIO`。`RuntimeIO.servo_input/close` 必须是真正 async、可取消且在传播 `CancelledError` 前自行清理，不能把无界 vendor SDK 调用直接塞进 event loop，也不能遗留后台 read task。`BrainCoSDKBackend.production_ready` 同时要求硬件写入已武装、capability probe、温度、SoftStop callback 和 non-auto-clear collision profile 均经台架验证，且 backend 未关闭、未超时、无故障/人工干预锁存；缺任一项都会在 assembly 阶段拒绝。唯一 EMG 命令入口是 `poll_emg_packet -> StreamingEMGClassifier -> StreamingEMGEventBridge`；`control_input/servo_input.emg` 必须为 `None`，运行时还会再次检查。factory 严格读取 control 中的全部关键 TTL、SLA、runtime deadline、EMG 门限与 max-replans，要求 EMG v3 checkpoint+每日校准、Planner LoRA、live T-Rex identity manifest/握手和硬件 safety/completion calibration；任一 artifact 缺失即拒绝组装。

V1 的正常调度只使用 offset 0 的 `slow_and_fast` 和 offset 4/8/12 的 `fast`。当前 server 在启用 cascaded tactile 时，纯 `slow` 有意返回空 action，因此客户端会拒绝该回复；纯 `slow` 只能与 server `--disable_tactile 1` 一起使用。真实 server 回复仍必须经过同一个 task lease、时间聚合、触觉 residual 和最终安全写入路径；当前 ZMQ 单元测试使用注入的传输桩，尚未运行真实 GPU server。

## 11. 真机接入前仍需提供和标定的信息

以下信息缺一不可；当前代码不会猜测它们：

1. **SDK 版本与 client 实例。** `BrainCoSDKBackend` 优先调用官方的原子批量反馈 `revo3_get_motor_status_data`，并单独读取电机 status；旧 wrapper 才回退到 position/velocity/current 分别读取。写入使用 `revo3_set_all_motor_positions`，碰撞优先使用批量查询。真实方法名和签名必须按实际安装的 `bc-revo3-sdk` 版本通过 CapabilityProbe 再次核验。
2. **硬件写入双重解锁。** 默认 `allow_hardware_write=False` 且 `capability_probe_confirmed=False`；只有完成急停、空载、限位、单位和低速台架检查后才可同时设为 `True`。官方单次 position/servo 写入不会替本项目在每个周期更新状态和碰撞判定；所有真实写入必须经过 `RevoCommandPipeline`。
3. **21 关节清单与方向。** 当前内部顺序来自 Revo3 retargeting 分支并由 hash 固定。还需确认左右手、零位、正方向、机械限位和每关节最大步长。
4. **单位。** 内部统一为 rad、rad/s、A。SDK position 默认按 degree，velocity 默认按 rpm 并使用 `rpm*2π/60`，current 默认按 mA→A；官方 Python 资料对 current 的 A/mA 表述存在冲突，所以必须在 CapabilityProbe 中实测，然后只在 Adapter 边界指定一次转换，禁止重复转换。
5. **U21VT 原始触觉 schema 与 policy byte identity。** 当前策略契约需要每指 6 维 Force6D，共 `[5,6]`。必须提供真实字段、轴定义、单位、有效标志、频率和采样时间。若设备不是原生 Force6D，需要一个经过标定的 raw-touch-to-F6 profile；不能把未知原始通道直接 reshape 成 `[5,6]`。硬件 calibration 还必须写入小写 64 位 `compatible_policy_tactile_profile_manifest_sha256`，且逐字节等于 live-handshake 后 `TReXServerIdentity.tactile_profile_manifest_sha256`；同样的 `profile_a_force6d_diff` 名称，或把两个不同 hash 一起塞进版本 fingerprint，都不是兼容证明。结构见 [`config/revo3_v1_hardware_calibration.schema.json`](../../config/revo3_v1_hardware_calibration.schema.json)；example 明确带 `example_only=true`，production 会拒绝。
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
