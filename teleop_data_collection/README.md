# Revo 3 / Tianji 遥操作与原始数据采集

> **当前状态：Mock / component-verified。** 本子项目已经用合成数据验证多频流录制、因果对齐、控制器标签溯源、崩溃边界以及 Revo 3 VLA 白名单导出；它**没有**完成 Tianji 真机接入，也**不代表**真机安全、遥操作质量、四类任务成功率或论文结果已经得到验证。任何硬件写入都必须先完成本文的 capability probe、人工 arming 与安全验收。

## 1. 项目边界

`teleop_data_collection/` 是 T-Rex 主训练代码之外的独立数采子项目。它负责：

- 以各传感器和控制器的原生频率保存相机、Revo 3、触觉、手套、EMG 与 Tianji 数据；
- 保存控制命令从请求、通过安全层到真实写入边界的可审计回执；
- 在 30 Hz 时间轴上建立只向过去取样的因果索引；
- 管理 episode 的开始、提交、失败隔离和派生数据导出；
- 从 master episode 生成只含 Revo 3 单手训练字段的 VLA episode。

它不负责：

- 将机械臂自由度并入当前 21 维手部 VLA action；
- 将 EMG、手套或 Tianji 原始数据直接喂给 T-Rex；
- 用最低采样率或“最大公约数”降频驱动所有硬件；
- 绕过现有 `revo3_v1` 安全链路直接写手；
- 猜测 Tianji SDK 方法、关节顺序、单位或载荷配置；
- 在没有可信 6DoF 手腕位姿时，用手套屈曲值驱动 Tianji；
- 声称 Mock 轨迹是专家示教或任务成功数据。

边界关系如下：

```text
MANUS / BrainCo 手套 ──┐
BrainCo EMG ───────────┤
RGB / Revo / 触觉 ─────┼──> master episode（原生频率、完整审计证据）
Tianji 状态与命令 ─────┘                  │
                                          ├──> Revo 3 VLA allowlist 导出
                                          └──> EMG 分类数据导出（独立 schema）
```

master 是证据总账，不是可直接用于任一模型训练的混合数据集。训练数据必须由明确的 allowlist exporter 物理导出到另一个目录，并再次验证 schema。

## 2. 固定来源、用途与许可证边界

所有外部来源由 [`sources.lock.json`](sources.lock.json) 固定到具体 commit。`scripts/bootstrap_sources.py` 只在被忽略的 `vendor/` 下创建 detached checkout，不会把第三方代码或二进制提交进本仓库。

| 来源 | 固定版本 | 许可证/可信边界 | 本项目用途 |
|---|---|---|---|
| [BrainCo Revo 3 SDK](https://github.com/BrainCoTech/brainco-revo3-sdk) | `7ba96ebd5bee16b730577e2f7c5df98600da53eb` | BrainCoTech 官方组织，但该 commit 根目录无 LICENSE；仅外部使用 | Revo 3 硬件 SDK 参考 |
| [BrainCo Revo 3 ROS 2](https://github.com/BrainCoTech/brainco_revo3_ros2) | `67442f2811680f8666b7d51a22398d68a945941e` | Apache-2.0；BrainCoTech 官方组织 | 消息与控制器契约参考 |
| [Revo-Retargeting](https://github.com/BrainCoTech/Revo-Retargeting/tree/revo3_retargeting) | `83087f108b0b853413842e6e185fe4fb509004c5`，分支 `revo3_retargeting` | 混合包声明，MANUS SDK 另有许可证；必须人工确认 | MANUS→Revo 3 重定向工作区参考 |
| [BrainCo Hand SDK](https://github.com/BrainCoTech/brainco-hand-sdk) | `5c399113efd35f5ce664d5d8f3e8ff750ced5d23` | MIT；BrainCoTech 官方组织 | EDU 手套与 8 通道 EMG 公共数据格式参考 |
| [TJ_FX_ROBOT_CONTRL_SDK](https://github.com/cynthia-you/TJ_FX_ROBOT_CONTRL_SDK) | `747f5d0279a91d85e32d06008665d96886eff438` | 仓库声明 Apache-2.0，但作者/厂商身份未核验 | 只用于历史 `MarvinSDK.h` 语义交叉检查；不是受支持的运行依赖 |
| [wuji-hand-teleop](https://github.com/wuji-technology/wuji-hand-teleop) | `647801345a6a27dec5cbf56280ce63bb8b2f6a32`，tag `v2026.6.13` | 仓库源码 MIT；所带二进制许可证未核验 | 只参考 Tianji 位姿输出集成；不 vendoring 二进制 |

BrainCo 官方仓库可作为对应公开接口的一手来源，但仍应逐仓库检查许可证；Revo 3 SDK 示例仓库在固定 commit 没有根 LICENSE，因此 bootstrap 同样要求人工确认，不能因官方身份推定再分发许可。两个 Tianji/Wuji 来源只是参考证据，**不能替代 Tianji 厂商提供的当前 SDK、头文件、单位说明和安全手册**。仓库只实现了注入式、默认失效闭锁的 Tianji 窄适配层；没有附带或加载未经授权的二进制，缺少真实 SDK、反馈结构与安全验收时仍然无法写真机。

列出锁定来源：

```powershell
py -3.10 teleop_data_collection/scripts/bootstrap_sources.py --list
```

拉取许可证边界清晰的默认来源：

```powershell
py -3.10 teleop_data_collection/scripts/bootstrap_sources.py
```

仅在人工检查并接受混合/未核验许可证后，才可显式拉取相应参考来源：

```powershell
py -3.10 teleop_data_collection/scripts/bootstrap_sources.py `
  brainco_retargeting_revo3 `
  --ack-external-license
```

脚本会核对最终 `HEAD` 是否与 lock 中的 commit 完全一致；已有目录若版本不一致会直接失败，不会自动覆盖。

## 3. 安装

要求 Python `3.10.x`。以下命令从 T-Rex 仓库根目录执行：

```powershell
py -3.10 -m venv .venv-teleop
& .\.venv-teleop\Scripts\python.exe -m pip install --upgrade pip
& .\.venv-teleop\Scripts\python.exe -m pip install -e ".\teleop_data_collection[dev,camera]"
```

本子项目当前会复用仓库根目录下的 `revo3_v1` 数据契约与 Revo 安全写入链。运行源码模块时，应让仓库根目录与子项目 `src` 都可导入：

```powershell
$env:PYTHONPATH="$PWD;$PWD\teleop_data_collection\src"
```

上述环境变量只对当前 PowerShell 会话生效。

## 4. 运行 Mock 端到端数据流

生成全部四种当前限定任务的合成 episode：

```powershell
$env:PYTHONPATH="$PWD;$PWD\teleop_data_collection\src"
& .\.venv-teleop\Scripts\python.exe -m revo3_teleop.cli.mock_demo `
  --output .\outputs\teleop_mock `
  --duration-s 0.5
```

也可只生成指定任务：

```powershell
& .\.venv-teleop\Scripts\python.exe -m revo3_teleop.cli.mock_demo `
  --output .\outputs\teleop_mock_subset `
  --duration-s 0.5 `
  --tasks bottle phone
```

也可直接执行经过 schema 校验的配置文件；其中所有 native rate 会实际驱动合成流，命令行参数可覆盖配置值：

```powershell
& .\.venv-teleop\Scripts\python.exe -m revo3_teleop.cli.mock_demo `
  --config .\teleop_data_collection\configs\mock.json `
  --output .\outputs\teleop_mock_from_config
```

支持的 Mock 任务键为：

- `bottle`：抓瓶子；
- `phone`：握手机；
- `plastic_bag`：提起塑料袋；
- `refrigerator_door`：拉开冰箱门。

安装 entry point 后，也可等价使用虚拟环境中的可执行文件：

```powershell
& .\.venv-teleop\Scripts\revo3-teleop-mock.exe `
  --output .\outputs\teleop_mock `
  --duration-s 0.5
```

Mock 默认保存 30 Hz RGB、100 Hz Revo 状态、120 Hz触觉、120 Hz手套、100 Hz Tianji 状态，以及按 12.5 packet/s 回调的 EMG；每个 EMG packet 含 8 通道 × 20 点，对应 250 Hz 原始采样。Mock 的 Tianji 命令不再手工伪造回执，而是经过 `SyntheticTianjiNativeClient -> TianjiMarvinBackend` 的 capability、连接、递增 feedback serial、三重授权、限位、rad/degree 转换以及 `OnClearSet -> OnSetJointCmdPos_A/B -> OnSetSend` 事务，再保存 backend 产生的回执。但 native client、wrist pose、手套标志和运动结果仍全部是合成夹具，不能作为真机能力或任务成功证明。

真实相机入口为 `RgbCameraSource`：构造时不会打开设备，只有显式 `allow_hardware_start=True` 才能启动注入的 client；可选 `OpenCvCameraClient` 始终把 OpenCV BGR 明确转换成 RGB。分辨率、内参、畸变、颜色转换和 calibration revision 作为 episode 级 metadata 固定，帧级数据只保存 `uint8 RGB[H,W,3]` 与时间/冻结证据。该 source 不包含 SAM、目标检测、分割或跟踪；视觉目标语义仍属于上层 Planner，而不是数采驱动隐式锚定。

## 5. 时间语义：原生频率保存，不做 GCD 降频控制

本项目不采用“所有模态按最大公约数或最低频率运行”的方案。原因是那会丢弃触觉、EMG、关节反馈的原始动态，并可能把低级安全控制降到不可接受的频率。

正确约束是：

1. 每个 source callback 按自己的 native rate 进入独立 stream；
2. 每个 stream 的 `capture_timestamp_ns` 和 `sequence` 必须严格递增；
3. Revo 与 Tianji 的低级控制循环各自按硬件能力运行，不由 30 Hz VLA anchor 降频；
4. 30 Hz 只用于 T-Rex/Revo 3 训练投影和上层决策周期；
5. 对每个 30 Hz anchor，只选择 `capture_timestamp_ns <= anchor_timestamp_ns` 的最近有效样本，即 **latest-not-after**；
6. 找不到过去样本、样本超过 `max_age_ns`、时钟域不可信、stream 无效或命令未成功写入时，该 anchor 必须拒绝，绝不能用未来样本回填。

第 `i` 个 30 Hz anchor 使用绝对时间公式：

```text
anchor_timestamp_ns(i) = epoch_ns + floor(i * 1_000_000_000 / 30)
```

这样不会用反复累加 `33.333 ms` 的方式积累漂移。anchor 文件只保存 native row 的 `(relative_path, row_index, timestamp, sequence)` 引用；原始值仍保留在各自 HDF5 shard 中。

## 6. 控制标签：只接受 controller `exact_sent_target`

训练 action 不是手套角度、重定向器输出、planner nominal target，也不是安全层裁剪前的请求。唯一允许成为行为克隆标签的是：

```text
glove/raw intent
  -> retargeter requested_target
  -> safety authorized_target
  -> awaited hardware/controller write
  -> CommandReceipt.exact_sent_target
  -> 30 Hz anchor action[21]
```

`CommandReceipt` 至少区分：

- `requested_target`：调用者请求；
- `authorized_target`：经过限位、安全裁剪后获准写入；
- `exact_sent_target`：backend 写调用成功返回后，记录的实际发送目标；
- `accepted`：写入是否成功；
- `decision_timestamp_ns` 与 `write_timestamp_ns`；
- `controller_sequence`、单位和 joint-order hash。

只有 `component == "revo_hand"`、`accepted == true`、`exact_sent_target` 存在、维度为 21、单位为 rad、joint-order hash 匹配且回执未被其他 anchor 使用时，才可监督当前 VLA anchor。安全 veto、通信异常或 backend 写失败的回执只保留诊断价值，绝不能进入训练 action。

当前 `TeleopRevoWriter` 复用 `revo3_v1.revo.RevoCommandPipeline` 作为唯一写入路径；`execute()` 等待 backend 写完成后才创建可训练回执。Tianji 命令即便存在于 master，也不会被投影进 Revo 3 的 21 维 action。

## 7. Episode 生命周期与落盘格式

master episode 使用显式事务边界：

```text
master/.inprogress/<episode_id>/   # 正在写入，不可训练
              | commit + fsync/close + atomic replace
              v
master/committed/<episode_id>/     # 只读、可审计、可导出

master/.inprogress/<episode_id>/   # 异常
              | abort
              v
master/quarantine/<episode_id>/    # 保留失败证据，不可训练
```

`CollectionSession` 是统一的 admission/termination authority，但不是硬件调度器。它要求构造时显式注入 `stop_targets`、`revo_hold`、`tianji_soft_stop` 和 `flush`，并为每个 required stream 配置 timeout。无效或超时 source、backend fault、被拒命令、writer 异常都会按以下顺序失效闭锁：

```text
停止生成新 target -> Revo hold -> Tianji soft-stop -> episode quarantine
```

正常停止则执行相同的目标停止/安全动作，再 flush 并原子 commit。30 Hz anchor 只是上层调用 `record_anchor_if_due()` 的索引节拍；`CollectionSession` 不会睡眠、重采样或降低 Revo/Tianji 的底层控制频率。

每个 native stream 使用固定 schema、可扩展、chunked 的 HDF5 shard：

```text
streams/<stream>/index.jsonl
streams/<stream>/shards/shard_000000.h5
streams/<stream>/shards/shard_000001.h5
...
```

默认每 shard 最多 2048 行、LZF 压缩。一次 append 的崩溃边界为：

1. 写完该行所有 header 与 payload dataset；
2. HDF5 flush；
3. 推进 `committed_rows` 并再次 flush；
4. 将跨文件引用追加到 `index.jsonl` 并 fsync。

若 HDF5 已写但 index 追加失败，该行只是不可引用的 orphan，recorder 会进入 fault 状态且不得 commit。公共只读接口：

```python
from revo3_teleop.recording import load_native_payload

payload = load_native_payload(
    episode_root,
    "emg",
    reference_or_index_row,
)
```

调用者不应依赖私有 HDF5 dataset 路径。helper 会复核 stream、row、schema、source、clock、sequence、capture time、`committed_rows` 与 clean-close 标记。

## 8. master、VLA 与 EMG 数据的物理隔离

完整 master 可同时保存：

- RGB；
- Revo 3 状态、触觉与手部命令回执；
- Tianji 状态与机械臂命令回执；
- MANUS 或 BrainCo 手套原始数据；
- BrainCo EMG 原始 packet、掉包和脱落电极证据。

Revo 3 VLA exporter 只能从 `master/committed/` 读取，并将结果写到单独目录，例如：

```text
outputs/teleop_mock/derived/revo3_vla/<episode_id>/
```

当前 VLA allowlist 仅允许：

- RGB 图像；
- `state[21]`；
- 来自 Revo controller receipt 的 `action[21]`；
- Revo 3/U21VT 触觉特征 `[5, 6]`；
- 30 Hz 时间、任务 instruction 和必要 provenance。

它明确排除：

- EMG 原始信号与 EMG 标签；
- MANUS/BrainCo 手套原始数据；
- Tianji 状态、目标和 action；
- retargeter 中间 target；
- 未获准或未写入的手部 target。

导出完成后还会调用现有 `RevoEpisode.load()` 复核 schema，且元数据必须声明 `contains_emg=false`。EMG 二分类数据由 `export_emg_binary_dataset()` 写入另一个物理目录；不得通过“训练时忽略若干 key”的方式直接消费 master。两个派生产品可以共享 episode/session provenance ID，但不能共享一个训练 schema 或被误拼接为联合 VLA token。

EMG exporter 只接受显式、人工复核的 OPEN/CLOSE 时间区间。窗口必须完整落在一个标签区间内、无 lead-off、无超阈值采样间隙；标签绝不由手套轨迹或机器人命令推断。默认输出 1 秒窗口（250 点）和 0.5 秒 stride，并强制以 `subject_id` 为 split 单元：

```python
from revo3_teleop.recording import (
    EMGLabelInterval,
    EMGSessionSpec,
    export_emg_binary_dataset,
)

dataset = export_emg_binary_dataset(
    sessions=(
        EMGSessionSpec(
            episode_root=committed_episode,
            subject_id="participant_001",
            session_id="day01_session01",
            split="train",
            intervals=(
                EMGLabelInterval(
                    start_timestamp_ns=start_ns,
                    end_timestamp_ns=end_ns,
                    label=1,  # CLOSE
                    source="protocol_cue_human_reviewed",
                    human_reviewed=True,
                ),
            ),
        ),
        # val/test 必须来自不同 subject；此处省略。
    ),
    output_root="derived/emg_binary_v1",
)
```

真实训练导出要求 train/val/test 三个 split 均至少一个有效窗口，并拒绝同一受试者跨 split 泄漏。窗口分段同时检查时间间隔、header `dropped_since_previous`、payload `sequence_gap_packets`、无效 packet 和 lead-off；即使 host 重建时间看似连续，也绝不允许窗口跨越显式掉包证据。

## 9. 手套的两条接入路径

### 9.1 MANUS 路径

`ManusRosSource` 是惰性的注入边界，不会自行 import ROS/MANUS，也不会在构造时启动硬件。它解析官方 `ManusGlove` 字段契约并保留 raw nodes、ergonomics 与 raw sensor pose。

重要限制：公开 ROS 消息没有 `Header`，现有 bridge 还会丢弃 MANUS `publishTime`。因此当前 adapter 只能把 host callback arrival 作为时序证据，`device_timestamp_ns=None`，不能伪装成设备时间。只有当集成方提供并核验 `wrist_node_ids`，且该 node 在当前消息中确实存在有限 6DoF pose 时，`provides_wrist_pose` 才能为真。IMU 姿态或任意 skeleton node 不能冒充手腕位姿。

MANUS→Revo 3 可参考 BrainCo 的 `Revo-Retargeting`，但 MANUS SDK 的单独许可、坐标系、左右手、关节顺序、尺度和零位都必须在实际机器上验收。

### 9.2 BrainCo EDU 手套路径

`BrainCoGloveSource` 按公开示例保存：

- 6 路 flex，名义 50 Hz；
- IMU，名义 100 Hz；
- magnetometer，名义 20 Hz。

批回调内较早的行会按名义周期向过去重建时间，但这仍是 host-arrival reconstruction，不是设备时钟。该手套路径固定声明 `provides_wrist_pose=false`：flex、IMU 和磁力计可用于手部重定向研究，但不构成经过验证的 Tianji 末端 6DoF 位姿。

两种 source 都默认 `hardware_autostart=false`；真机启动必须同时注入实际 client/ROS adapter，并显式设置 `allow_hardware_start=True`。

固定版本的 `bc-edu-sdk` 示例通过模块级 `set_*_data_callback` 注册回调，而不是给每个设备实例绑定独立 callback。若 BrainCo 手套与 BrainCo EMG 腕带要在同机同时运行，V1 默认采用两个独立采集进程并通过带时间戳的 IPC 汇入 recorder；在没有设备 ID 路由实测证据前，不把两个设备塞进同一 SDK 全局 callback 进程。

## 10. EMG 采集边界

`BrainCoEduEMGSource` 复现公开 EDU row：

```text
[sequence, lead_off_bits, 8 channels * 20 samples]
```

输出 `signal[8,20]`、250 Hz、每通道 sample timestamp、lead-off mask、sequence gap 与 host callback 证据。设备没有被声称提供可信时间戳；packet 内时间由 callback 末端按 250 Hz 向过去重建，并明确标记 `clock_is_host_reconstruction=1`。

采集时必须：

- 保存原始 packet，不只保存过滤后的 feature；
- lead-off 非零时标记 sample 无效，不把它当作负类；
- 检测 sequence gap；
- 保存受试者、截肢侧、通道布局、电极位置、增益/滤波、校准与 session 信息；
- 将“开/闭合”提示时间、执行窗口和人工质量判定作为独立标签证据；
- 跨天、跨 session、跨受试者划分，避免相邻窗口泄漏。

EMG 可用于上层 intent/planner 和 Task Executive，但不属于本轮 hand-only T-Rex mid-training schema。

## 11. Tianji：无可信 6DoF wrist pose 必须硬阻断

`TianjiMarvinBackend` 已实现为不携带第三方 SDK 的注入式窄边界，并通过 fake-native tests 核对接口语义；这不是经厂商或真机验证的 backend。`hardware.example.json` 使用文档保留网段 IP、占位关节顺序和 `provides_wrist_pose=false`，因此按设计**不可能 arm**。

当前实现的关键不变量为：

- 内部统一 rad、rad/s、Nm；仅在 native 边界转换为 degree、degree/s；
- 每次位置写严格执行 `OnClearSet -> OnSetJointCmdPos_A/B -> OnSetSend`；
- 只有 `OnSetSend` 成功后才产生 `exact_sent_target`；
- 写入必须同时满足 `allow_hardware_write=True`、已确认 capability probe、每次调用匹配 arm token、7 维台架限位存在、反馈 fresh 且 frame serial 已被证明前进；
- `cur_state==100` 或 `err_code!=0` 按故障处理并 best-effort soft-stop；
- `OnEMG_A/B` 在这里明确命名为 `soft_stop()`，避免与表面肌电 EMG 混淆；
- `close()` 只有在 state 0 已确认时才释放连接，否则抛出 `TianjiPhysicalInterventionRequired`，要求现场物理处理。

真实 ctypes feedback 结构必须通过本机厂商 SDK 的 `feedback_buffer_factory` 与 `feedback_decoder` 注入；仓库不会猜结构体布局，也不会复制 Wuji/Tianji 二进制。

机械臂遥操作至少需要一个已校准、带明确坐标系与时钟来源的 6DoF 手腕/控制器位姿。以下情况必须硬阻断 Tianji target 产生和写入：

- 只有 BrainCo flex/IMU/magnetometer；
- MANUS 未配置经过核验的 wrist node；
- tracker 丢失、遮挡、漂移或 pose 超龄；
- quaternion 非有限或未归一化；
- source/robot/base/tool 坐标变换未标定；
- 左右手、单位或关节顺序未知；
- 真实 SDK 的 capability probe 未通过。

硬阻断的含义是“不生成、不授权、不写入 arm target”，而不是发送零向量猜测保持。Revo 手部 raw collection 可以在不驱动 Tianji 的模式下单独进行，但该 episode 必须记录 arm disabled 原因。

## 12. 真机接入的强制安全门

任何真实 Revo 或 Tianji 写入前，至少完成以下逐项证据；缺一项都保持 `allow_hardware_write=false`。

### 12.1 Capability probe

- 记录 SDK/固件/控制器/机器人序列号与哈希；
- 只读查询可用设备、轴数、模式、状态字段和故障码；
- 核对 joint order、正方向、零位、位置/速度/电流/力矩单位；
- 核对同步/异步调用语义、超时、返回码以及“成功返回”是否等于控制器接受；
- 量测命令频率、状态频率、通信延迟、抖动与丢包；
- 确认硬件时间戳是否真实存在、时钟域、分辨率、回绕和重启行为；
- 对 Revo 复核 U21VT 触觉 shape、量纲、饱和、噪声和接触基线；
- 对 Tianji 复核末端/工具坐标、笛卡尔/关节模式和控制模式切换条件。

probe 结果必须形成机器可读 manifest，配置中的占位符全部由证据替换，不能由代码默认值猜测。

### 12.2 Arming 与人工确认

- 程序启动和 source 构造永不自动写硬件；
- `allow_hardware_write=true` 只是一道软件许可，不能替代现场人工 arming；
- arming 前再次确认 capability manifest 与当前连接设备一致；
- 先在无负载、低速、小范围、单轴/单指模式下验证方向；
- 每次新 session、换夹具、换手、换固件或重新上电后重新执行最小 probe；
- 任何 schema/hash/单位不一致立即 disarm。

### 12.3 急停、watchdog 与最终限权

- 必须有操作员可达、独立于 Python 进程的物理急停；
- 通信超时、状态超龄、source 丢失、序号跳变、控制循环停顿触发 watchdog hold/stop；
- 位置、速度、加速度、jerk、电流/力矩、接触力和工作空间都需硬/软限位；
- 检查自碰、手指互碰、手-臂碰撞、环境碰撞和人体安全距离；
- 安全层拥有最终裁剪/拒绝权；EMG 或手套意图不能越过急停、过流、碰撞和硬限位；
- 失败或 veto 的 command receipt 不得成为训练标签。

### 12.4 Payload 与 COM

Tianji 控制器中的 payload 必须包含实际安装的 Revo 3、转接板、腕部相机、支架和随动线缆影响。至少实测并记录：

- 总质量；
- 相对工具法兰的质心 `COM(x,y,z)`；
- 厂商接口要求时的惯量张量；
- 工具坐标、安装姿态和标定版本；
- payload 配置写入结果与回读值。

更换相机、夹具、线缆固定方式或手部版本后应重新标定。被抓物体的质量/质心不能靠未经验证的默认值冒充机器人固定 payload；动态负载策略必须遵循实际 Tianji 控制器能力。

### 12.5 时间戳与时钟

- 每行同时记录 capture/device time、host receive time、clock domain 和 sequence；
- 不得把 callback 到达时间标成设备采集时间；
- 对没有设备时间的源，明确记录重建方法和不确定度；
- 跨机器时使用 PTP/硬触发或经验证的 clock mapping，并保存 offset/drift 估计；
- 设备重启、时间回绕、系统 suspend/resume 或 mapping 跳变必须切新 episode；
- 在因果投影前检查每流严格单调、最大 age、掉包和未来样本；
- 相机 exposure time 优先于解码/回调时间；若取不到，必须明确其替代时间语义。

## 13. 测试与当前证据等级

从仓库根目录运行完整组件测试：

```powershell
$env:PYTHONPATH="$PWD;$PWD\teleop_data_collection\src"
py -3.10 -m pytest -q teleop_data_collection/tests tests/revo3_v1
```

当前测试覆盖的核心不变量包括：

- native stream 严格单调；
- 100 个样本不会生成 100 个文件，而是写入分片 HDF5；
- 30 Hz anchor 不选择未来样本；
- 未被 controller 接受的命令不能监督；
- requested/authorized 与 exact-sent 不一致时，导出 action 仍严格取 exact-sent；
- `.inprogress -> committed` 原子提交与失败 quarantine；
- HDF5 `committed_rows` 和 clean-close 边界；
- VLA allowlist 不含 EMG、Tianji 或手套；
- 导出结果可被现有 `RevoEpisode.load()` 读取；
- EMG 8×20 packet 解析、lead-off/drop/host-clock 重建与人工标签窗口导出；
- BrainCo 6-flex 手套固定声明无 6DoF wrist pose；MANUS 只有显式核验的 wrist node 才声明有；
- Tianji fake-native 的 rad/degree 转换、A/B 侧、写事务顺序、三重 arming、watchdog、状态 100、错误码、soft-stop 与安全关闭。

该证据等级是 **component-verified synthetic fixture**，只说明接口与数据谱系按测试实现。它不说明：

- Tianji 或 Revo 真机 SDK 已成功连接；
- MANUS/BrainCo 手套已完成真实重定向；
- 真实控制频率、延迟与时间同步已达标；
- payload/COM、限位和急停已验收；
- 合成 `exact_sent_target` 等于物理执行状态；
- 抓瓶子、握手机、提袋或开冰箱门在真机上成功；
- 当前数据可用于报告泛化能力、受试者获益或论文级结果。

## 14. 真机阶段的最小升级顺序

1. 获取 Tianji 当前官方 SDK、版本、许可证、安全手册和厂商示例；
2. 对 Revo、Tianji、相机、手套、EMG 分别完成只读 capability probe；
3. 固定 joint/field schema、单位、clock provenance 与设备 manifest；
4. 配置并验证独立急停、watchdog、限位、payload/COM；
5. 仅录状态，不写命令，完成长时 native-rate 落盘与时间漂移测试；
6. 在固定支架和安全空间内完成 Revo 单指/单轴小步写入；
7. 有可信 6DoF wrist pose 后，才进行 Tianji 单轴、小范围、低速写入；
8. 回放审计 controller receipt、视频、状态和时钟，人工批准 episode；
9. 最后才启用完整 glove→retargeter→safety→controller 数采，并分别导出 VLA 与 EMG 数据产品。

在第 1–8 步形成可复核证据前，不应把 `configs/hardware.example.json` 改成可自动 arming 的默认配置，也不应把任何 Mock 成功输出描述为硬件完成度。
