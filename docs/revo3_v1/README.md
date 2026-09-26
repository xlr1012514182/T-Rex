# Revo 3 系统架构与运行接口

[项目首页](../../README_ZH.md) · [训练与推理](TRAINING.md) · [数据采集](../../teleop_data_collection/README.md) · [开发指南](../DEVELOPMENT.md)

## 1. 模块职责

| 模块 | 职责 |
|---|---|
| [`emg`](../../revo3_v1/emg/) | 五类肌电原语、因果预处理、校准和边沿事件 |
| [`planner`](../../revo3_v1/planner/) | Qwen 任务规划、严格 JSON、澄清和异步 worker |
| [`vision`](../../revo3_v1/vision/) | 单次采集派生 full/center 视图及相机健康检查 |
| [`executive`](../../revo3_v1/executive/) | 唯一任务状态权威、原子提交和版本化授权 |
| [`policy`](../../revo3_v1/policy/) | Revo 21 维模型接口、slow/fast 调度、身份检查与时间聚合 |
| [`runtime`](../../revo3_v1/runtime/) | 30 Hz 观测决策和 100 Hz 指令写入调度 |
| [`revo`](../../revo3_v1/revo/) | 通用手部接口、公开 SDK 适配、完成监视与安全授权 |
| [`tactile`](../../revo3_v1/tactile/) | 原生触觉历史、有界关节协同修正 |
| [`data`](../../revo3_v1/data/) | 单手 episode、数据划分、转换与归一化 |

配置来源：[控制参数](../../config/revo3_v1_control.json)、[策略参数](../../config/revo3_v1_trex.json)。运行时校验实际配置，而不是依赖文档示意值。

## 2. EMG 意图通道

生产 profile 为 `brainco_edu_8ch_250hz_hp40_v1`：8 通道、250 Hz、40 Hz 四阶因果高通、500 点/2 秒窗口和 12/13 点交替步长，平均每 50 ms 更新一次。

```text
因果预处理 → Reinhard压缩 → Conv1D(8→512, kernel=3, stride=1)
           → ReLU/Dropout/LayerNorm → 3层LSTM(512)
           → LayerNorm → 5类投影 → 时间均值池化
```

学习类别为 `POWER_GRASP / PRECISION_GRASP / LATERAL_GRASP / RELEASE / REST`。`UNKNOWN` 和 `BAD_SIGNAL` 分别由置信度与信号质量派生。

- 抓取门：confidence ≥0.80、margin ≥0.20、quality ≥0.80，持续 300 ms。
- 释放门：confidence ≥0.90、margin ≥0.30，持续 500 ms。
- 更新间隔超过 150 ms 会清除待定 dwell。
- 抓取事件被锁存，REST/UNKNOWN 不取消任务；释放仍受安全层约束。

模型迁移、训练和校准见 [EMG 文档](../../revo3_v1/emg/README.md)。EMG 数据与 VLA 数据使用独立 schema。

## 3. 单相机与 Planner

相机层完成标定几何矫正后，从同次采集派生两幅 `384×288` RGB：全图 `full` 和固定中心裁剪 `fixed_center`。两者共享 capture timestamp 与 sequence。中心图不是检测框裁剪；视觉链不引入 SAM、检测器或实例跟踪器。

`PlannerContextBuffer` 保留三次严格递增的采集。`AsyncPlannerWorker` 将三帧全图和最新中心图交给 Qwen3-VL-2B；模型和 processor 固定 revision，生产装配要求带哈希与来源信息的 LoRA artifact。生成预算为 384 tokens，温度为 0。

规范输出为 `planner_v1` JSON，包括 primitive、目标类别/部位、抓型、`target_region=[cx,cy,width,height]`、就绪状态、置信度、歧义和最长 25 个英文空格分词的 instruction。旧 `bbox=[x1,y1,x2,y2]` 仅作为兼容字段。

状态为 `READY / ASK_CLARIFY / NOT_READY / INVALID`。协议错误只修复一次。`ASK_CLARIFY` 等待带当前 token 的明确回答：

```python
token = service.clarification_token
service.submit_clarification(answer, token=token)
```

回答必须非空且不超过 256 字符。后续请求仍使用新鲜图像，并经过完整任务提交链。

Planner 结果携带 generation、event、primitive、task version 和来源时间。20 s SLA、1 s 结果 freshness、18 s 来源最大年龄和场景签名检查分别约束不同时间边界。`VisualGate` 检查结构化目标区域：默认面积 ≥0.15、中心距离 ≤0.25、置信度 ≥0.75、最近三帧至少两帧就绪。

## 4. Task Executive

| 输出 | 运动指令 | 语义 |
|---|---|---|
| WAIT | NONE | 等待意图、上下文或澄清 |
| START | POLICY | 创建 lease、锁存指令并清理旧缓存 |
| CONTINUE | POLICY / CONTROLLED_OPEN | 执行策略或显式释放 |
| HOLD | HOLD_POSITION | 保持抓持或暂挂执行 |
| REPLAN | HOLD_POSITION | 接触前的一次重规划 |
| COMPLETE | NONE | 受控释放完成，等待确认 |
| ABORT | SAFE_STOP | 故障锁存，要求确认式停止与复位 |

同一候选需在默认 150 ms 提交窗口中保持指令、视觉、profile、版本和安全身份一致。camera/state/touch 的默认 TTL 分别为 100/50/150 ms。

策略响应必须匹配 task id、task version、lease id、instruction hash 和 runtime fingerprint。首次接受 slow chunk 后才建立执行 epoch，避免模型计算时间消耗掉动作前几步；后续 slow chunk 同样重新对齐执行起点。

稳定抓持进入 HOLD，不自动张开。无进展仅允许接触前重规划一次；接触后无进展触发 ABORT。显式 RELEASE 切换为确定性 CONTROLLED_OPEN，完成监视同时确认触觉卸载与 safe-open，随后进入 COMPLETE。

终态通过 `service.acknowledge_terminal(safe_state_confirmed=True)` 请求复位；ABORT 还要求 operator reset 和真实成功的底层 SoftStop。复位在下一次新鲜 control tick 提交，不直接改变 writer 正在读取的状态。

## 5. 动作与触觉接口

- 状态：`q_rad[21]`。
- 动作：`q_target_rad[16,21]`，未来绝对关节目标，内部单位为 rad/SI。
- Profile A：当前 Force6D `[5,6]`、原生历史 `[16,5,6]`、五指当前 DIFF。
- DIFF 在线对象为 uint8 `[5,240,240]`，模型张量增加单通道维。
- Profile B：DIFF-only，适用于数据/训练/服务组件；在线 factory 固定 Profile A。
- Profile C：pressure/matrix 预留接口，当前训练器和在线装配不启用。

原生历史必须有严格递增的时间戳和 sequence，不能复制 policy 帧补齐。在线 fast 输入使用五指当前 DIFF 及逐指时间戳；训练的 delayed offsets 用于监督，不是额外在线传感器要求。

30 Hz 动作网格内，offset 0 请求 `slow_and_fast`，4/8/12 请求 `fast`；其余时间点消费已有动作。16 步约 0.533 s。`k=0` 时间聚合对覆盖同一动作点的有效预测取算术平均，再供 writer 插值消费。

T-Rex 慢分支缓存部分 flow 结果及上下文，快分支基于该快照和新触觉输出完整动作 chunk；外置插件另行提供有界关节残差，两者不是同一机制。

## 6. 控制与安全

唯一写入路径：

```text
名义动作 + 有界触觉残差 → SafetySupervisor → RevoCommandPipeline → RevoBackend
```

writer 检查实际 telemetry、lease 和 control heartbeat。硬安全覆盖限位、指令变化、电流、速度、加速度、温度、碰撞、触觉硬过载、急停和陈旧数据。触觉插件必须绑定标定参数；默认关闭，不在接触前或释放阶段施加抓持残差。

调度目标不等于神经网络推理 FPS，也不构成操作系统硬实时保证。错过的写入周期被跳过，不补发突发命令。watchdog / servo IO / close 的默认预算为 100/50/1000 ms。只有 SoftStop 和资源关闭被确认，shutdown 才标记 clean。对可能永久阻塞的厂商调用，应使用独立设备进程、外部 watchdog 和物理急停。

## 7. 装配与模型身份

- `build_simulation_runtime`：确定性测试后端，用于本地练习与回归测试。
- `build_production_runtime`：加载 EMG、Planner、策略身份与硬件标定，通过 `ProductionBindings` 接入设备 IO 和 Revo backend。
- 策略服务自行计算 checkpoint、lineage、归一化、profile 和 joint-order 身份。客户端启动时握手，并逐回复核对身份。

参考 [runtime 示例](../../config/revo3_v1_runtime.production.example.json)、[标定 schema](../../config/revo3_v1_hardware_calibration.schema.json) 和[标定模板](../../config/revo3_v1_hardware_calibration.example.json)。复制模板后填写自己的配置，保留默认硬件写入关闭状态，完成设备检查后显式启用。

CLI 入口：`python scripts/revo3_v1_runtime.py --help`。生产模式默认持续运行，显式 `--servo-ticks` 用于有界检查。SIGINT/SIGTERM 被转换为有序关闭请求。
