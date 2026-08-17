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

通用外部来源由 [`sources.lock.json`](sources.lock.json) 固定到具体 commit，Tianji 追加检索与可信度边界由 [`tianji_sources.lock.json`](tianji_sources.lock.json) 固定。`scripts/bootstrap_sources.py` 只在被忽略的 `vendor/` 下创建 detached checkout，不会把第三方代码或二进制提交进本仓库。

| 来源 | 固定版本 | 许可证/可信边界 | 本项目用途 |
|---|---|---|---|
| [BrainCo Revo 3 SDK](https://github.com/BrainCoTech/brainco-revo3-sdk) | `7ba96ebd5bee16b730577e2f7c5df98600da53eb` | BrainCoTech 官方组织，但该 commit 根目录无 LICENSE；仅外部使用 | Revo 3 硬件 SDK 参考 |
| [ViTai SDK Release](https://github.com/ViTai-Tech/ViTai-SDK-Release) | `0071c61a1bf13539ab8c12b537a69e11bb78430a` | ViTai 官方公开 SDK 示例；wheel 与逐 SN 加密模型不 vendoring | `VTSDeviceFinder/VTSensor/FORCE6D_VECTOR` API 交叉核对 |
| [BrainCo Revo 3 ROS 2](https://github.com/BrainCoTech/brainco_revo3_ros2) | `67442f2811680f8666b7d51a22398d68a945941e` | Apache-2.0；BrainCoTech 官方组织 | 消息与控制器契约参考 |
| [Revo-Retargeting](https://github.com/BrainCoTech/Revo-Retargeting/tree/revo3_retargeting) | `83087f108b0b853413842e6e185fe4fb509004c5`，分支 `revo3_retargeting` | 混合包声明，MANUS SDK 另有许可证；必须人工确认 | MANUS→Revo 3 重定向工作区参考 |
| [BrainCo Hand SDK](https://github.com/BrainCoTech/brainco-hand-sdk) | `5c399113efd35f5ce664d5d8f3e8ff750ced5d23` | MIT；BrainCoTech 官方组织 | EDU 手套与 8 通道 EMG 公共数据格式参考 |
| [TJ_FX_ROBOT_CONTRL_SDK](https://github.com/cynthia-you/TJ_FX_ROBOT_CONTRL_SDK) | `747f5d0279a91d85e32d06008665d96886eff438` | 仓库声明 Apache-2.0，但作者/厂商身份未核验 | 只用于历史 `MarvinSDK.h` 语义交叉检查；不是受支持的运行依赖 |
| [fiveages-sim/marvin-ros2-control](https://github.com/fiveages-sim/marvin-ros2-control) | `82d836122c3cc4fac1a651fd84acc3591cccc677` | 第三方 wrapper；根目录无 LICENSE、`package.xml` 声明 Apache-2.0；SDK submodule 在公开环境不可访问 | 交叉核对当前 ROS2 wrapper 的调用序列，不能证明厂商 ABI |
| [wuji-hand-teleop](https://github.com/wuji-technology/wuji-hand-teleop) | `647801345a6a27dec5cbf56280ce63bb8b2f6a32`，tag `v2026.6.13` | 仓库源码 MIT；所带二进制许可证未核验 | 只参考 Tianji 位姿输出集成；不 vendoring 二进制 |

BrainCo 官方仓库可作为对应公开接口的一手来源，但仍应逐仓库检查许可证；Revo 3 SDK 示例仓库在固定 commit 没有根 LICENSE，因此 bootstrap 同样要求人工确认，不能因官方身份推定再分发许可。公开检索到的 Tianji/Wuji 来源只是第三方或历史参考证据，**不能替代 Tianji 厂商提供的当前 SDK、头文件、单位说明和安全手册**。更细的 Tianji 版本、submodule 可访问性与可信度记录在 [`tianji_sources.lock.json`](tianji_sources.lock.json)。仓库只实现了注入式、默认失效闭锁的 Tianji 窄适配层；没有附带或加载未经授权的二进制，缺少真实 SDK、反馈结构与安全验收时仍然无法写真机。

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

需要装配 BrainCo 官方 Python SDK 时，再显式安装锁定的可选依赖；普通 Mock、导出和离线测试不需要它们：

```powershell
& .\.venv-teleop\Scripts\python.exe -m pip install -e ".\teleop_data_collection[dev,camera,brainco]"
```

当前适配器固定核对 `bc-revo3-sdk==1.5.1` 和 `bc-edu-sdk==0.5.0`。改变版本必须重新执行接口审计、只读 probe 和台架验收，不能仅关闭版本检查。

Revo3 Ultra VisionTouch 的当前 BrainCo 示例固定安装 `pyvitaisdk4bc==1.0.10`，运行时 import 名为 `pyvitaisdk`。`visiontouch` extra 记录这个精确依赖；如果所用索引不提供该发行包，应按固定 BrainCo commit 中的 `python/install_vts_whl.sh` 从官方 OSS 安装对应平台 wheel，再安装本项目。仓库不分发 SDK wheel，也不分发 `{SN}.onnx.enc` 力模型：

```powershell
& .\.venv-teleop\Scripts\python.exe -m pip install -e ".\teleop_data_collection[dev,camera,brainco,visiontouch]"
```

实现所对齐的精确源码位置是 BrainCo [`vision_touch_window.py@7ba96ebd`](https://github.com/BrainCoTech/brainco-revo3-sdk/blob/7ba96ebd5bee16b730577e2f7c5df98600da53eb/python/gui/vision_touch_window.py) 与 ViTai [`vts_force6d.py@0071c61a`](https://github.com/ViTai-Tech/ViTai-SDK-Release/blob/0071c61a1bf13539ab8c12b537a69e11bb78430a/examples/vts_force6d.py)。本地适配只复现公开调用契约，不复制/分发其 SDK 或模型。

本子项目当前会复用仓库根目录下的 `revo3_v1` 数据契约与 Revo 安全写入链。运行源码模块时，应让仓库根目录与子项目 `src` 都可导入：

```powershell
$env:PYTHONPATH="$PWD;$PWD\teleop_data_collection\src"
```

上述环境变量只对当前 PowerShell 会话生效。

### 3.1 真机 source 装配与只读 capability probe

真机适配分成五个默认关闭的边界：

- `BrainCoRevo3SdkAssembly`：按官方 `bc_revo3_sdk.main_mod` API 枚举并核对唯一 Revo 3、21 维电机状态、碰撞接口、42 维触觉摘要以及可用时的 11 个原始触觉模块；
- `BrainCoEduSdkEMGClient`：按官方 `bc_edu_sdk.main_mod` API 枚举 8 通道 EDU 臂环，并以 250 Hz、每包 20 点配置回调；
- `BrainCoEduSdkGloveClient`：严格对齐固定 [`glove_example.py@5c399113`](https://github.com/BrainCoTech/brainco-hand-sdk/blob/5c399113efd35f5ce664d5d8f3e8ff750ced5d23/python/edu/glove_example.py)，只在显式 probe/确认/stream 三重门禁后枚举唯一 VID 21059、PID 6/2 手套，并配置 6 路 flex 50 Hz、calibrated IMU 100 Hz、calibrated magnetometer 20 Hz；
- `VisionTouchForce6DSource`：严格按显式 `thumb/index/middle/ring/pinky -> SN` 映射枚举五个 U21VT VisionTouch 传感器，逐 SN 校验 `{force_model_dir}/{SN}/{SN}.onnx.enc` 的预期 SHA-256，构造 `VTSensor(config=..., force_model_path=...)`、校准并采集 `FORCE6D_VECTOR`；
- `ProbedOpenCvCameraClient`：核对相机后端、分辨率、帧率、首帧形状和数据类型；client 返回原生 BGR，`RgbCameraSource` 只做有记录、无损的 BGR→RGB 通道重排，不内置 DINO、SAM、目标跟踪或分割。

EDU SDK 的 EMG/手套 `set_*_data_callback` 都属于同一个进程级全局 namespace。两个 concrete client 共用 `brainco_edu_callback_namespace_status()` 所暴露的 ownership guard；同进程同时启动会被明确阻断。若 SDK 卡在 `start_stream`/`stop_stream` 且线程未退出，client 和 `BrainCoGloveSource` 都会保留线程/client 引用及 callback ownership，拒绝第二实例接管，并要求进程/设备干预；不会为了“恢复”而覆盖全局回调。真实 EMG 与手套同步采集采用两个进程，再以各自 host monotonic timestamp 经 IPC 汇入 recorder。

### 3.2 一键装配的采集编排（默认只审计）

[`configs/hardware_collection.example.json`](configs/hardware_collection.example.json) 是完整 episode 装配模板。默认命令只执行静态 readiness audit：不导入现场 assembly plugin，不扫描 SDK，不打开相机/串口，也不写 Revo/Tianji。

```powershell
revo3-hardware-collect `
  --config teleop_data_collection/configs/hardware_collection.example.json
```

模板必须先由现场证据替换所有占位项，包括 Revo/EMG/相机序列号或 probe fingerprint、21 维关节限位与单位验收、鱼眼标定文件哈希、五指 VisionTouch SN 与逐模型 SHA-256，以及启用 Tianji 时的当前 SDK/ABI、7 轴限位、坐标标定、payload/COM、急停和 arm token。装配工厂本身也必须固定模块 SHA-256；`assembly_factory.kwargs` 会原样作为已检查的 JSON keyword arguments 传给 `factory(config, **kwargs)`，不会静默忽略。工厂只能构造尚未连接的依赖，并必须提供显式 `assert_disconnected`、同步 `abort_construction` 与 episode 结束使用的 `close` 回调；若断连断言失败，loader 会先同步撤销构造，再拒绝运行。相机/串口/IPC 的启动只能发生在 orchestration 生命周期内。任何一项缺失时，CLI 会在导入工厂之前阻断。

仅当 readiness 无 blocker，并且操作员同时给出配置内许可和三项命令行许可，才会执行一个 episode：

```powershell
revo3-hardware-collect `
  --config path/to/operator-reviewed-hardware-collection.json `
  --execute-hardware `
  --allow-hardware-connect `
  --allow-hardware-write
```

运行时职责固定为：原生频率 source 入账、独立 source watchdog、30 Hz `latest-not-after` anchor、每个 anchor 对应唯一 accepted Revo `exact_sent_target`、故障时停止 target 生产并依次执行 Revo hold/Tianji soft-stop、episode quarantine，以及成功后生成不含 EMG/手套/Tianji 的 VLA allowlist 视图。readiness 强制 EMG 与配置的 camera/state/tactile 三路都有 required timeout；三路 VLA stream 必须唯一、已进入 anchor、与 source role 一致且有 timeout。`control_step_timeout_ms` 必须不超过总体 `safety_watchdog_budget_ms`；control driver 卡住会触发 fault，而不是停住 watchdog。`shutdown_timeout_ms` 同时约束 sensor task、辅助 source stop 和 dependency close，超时会标记需要进程/设备干预并 quarantine，绝不声称 clean close。close 被放在原子 commit 之前，所以关闭失败不会发布可训练 episode。原始 EMG 只保留在 master；必须另行提供人工复核的 `POWER_GRASP / PRECISION_GRASP / LATERAL_GRASP / RELEASE / REST` 区间，才能导出到独立 `emg_review_root` 数据产品。若手套 IPC 被标为 episode-required，其每个 stream 同样必须配置 watchdog timeout；若明确标为诊断流则不会参与 episode commit 健康门。

BrainCo 手套和 BrainCo EDU EMG 都依赖 `libedu` 的模块级全局 callback，**不能在同一进程同时注册**。启用手套时，配置只接受 `external_timestamped_ipc`：手套 client 在独立采集进程运行，携带明确 clock domain/offset-drift 证据汇入 recorder。公共编排器不会覆盖 callback，也不会猜测 6-flex→21DoF 或 wrist→Tianji 映射；这些映射只能由哈希核验、带标定版本的现场插件提供。

先复制 [`configs/hardware_probe.example.json`](configs/hardware_probe.example.json) 到忽略提交的本地文件，填写设备标签/端口/相机标定文件，将需要检查的 component 设为 `enabled: true`，并将配置内的 `allow_hardware_probe` 设为 `true`。只有配置和命令行双重授权后才会打开设备：

```powershell
$env:PYTHONPATH="$PWD;$PWD\teleop_data_collection\src"
py -3.10 -m revo3_teleop.cli.hardware_probe `
  --config .\outputs\hardware_probe.local.json `
  --output .\outputs\hardware_capability.json `
  --allow-hardware-probe
```

该命令仅生成原子写入的 capability manifest；它强制 `allow_hardware_write: false`，不会发送动作，并明确记录 `probe_only=true`、`hardware_write_performed=false`、`real_hardware_function_verified=false`。probe 成功只证明当次主机上的包版本、设备身份和返回 schema 相容，不证明控制安全、传感单位正确或任务可用。

Revo 3 状态进入内部数据契约前统一使用 SI 单位：角度为 `rad`、角速度为 `rad/s`、电流为 `A`。公开 SDK 对速度/电流的单位证据仍有歧义，因此创建 `Revo3TelemetrySource` 前必须保存与 probe fingerprint、关节顺序 hash 绑定的 `Revo3BenchApproval`，通过受限台架测量选择 `rpm|deg/s` 与 `mA|A`。写入许可还需独立的急停、hold、关节限制和人工 arming；只读 probe 不会生成该许可。

Revo SDK 的普通 Pressure/Matrix 触觉默认保存 42 维 `summary_mn` 和可用时的 11 个原始模块。它们是压力区域汇总，不是六轴力/力矩；即使配置经台架审阅的 `U21VTPressureZoneProjection`，也只输出 `pressure_zones_n[5,6]`，**永远不输出 exporter 保留的 `features`**。

只有独立的 `VisionTouchForce6DSource` 可以生成训练用 `features[5,6]`。五行严格按 `thumb,index,middle,ring,pinky` 排列；每行严格为 `[Fx,Fy,Fz,Mx,My,Mz]`，前三轴单位 N、后三轴单位 Nm。初始化必须满足以下全部条件，否则立即失败而不是补零：五个显式 SN 互不重复且均在 `VTSDeviceFinder.get_sns()` 中；五份加密模型均存在且 SHA-256 与配置一致；SDK 版本为 `1.0.10`；五个 `VTSensor` 均完成 `calibrate()`。采样兼容官方 `extract_force6d_mean` 的两类实际返回：原始 `(6,)` 直接保留；非空 `(...,6)` 对所有 leading 维求 mean。实现额外收紧为最后一维必须**恰好**等于 6，绝不截掉多余 component，也不对缺失 component 补零；空数据、非有限值或过高 rank 都会拒绝整份五指样本。模型 hash、SN 映射、官方源码 commit、聚合规则、轴序与单位写入 episode metadata；每个 sample 以固定宽度的 `[rank,dims...,-1 padding]` 保存五个原始 shape，并保存全指有效位和 host read-start/read-completion 时间。该时间仍不是设备硬件时间。

鱼眼相机始终把未经几何变换的采集帧保存为 `camera_raw`。可选 `OpenCvFisheyeRectifier` 只接受显式物理标定的 `K/D`、人工选择的 `new_K`、输入/输出尺寸和 revision；`FisheyeRectificationConfig.transform_hash()` 将这些内容与插值/边界方式一起固定，派生帧另存为 `camera_rectified`，并保留相同 sequence/capture/device-time 证据和 raw sequence 引用。两条流物理分开，原始帧不会被覆盖；没有标定就禁用 rectifier，不会用 DINO 或其他模型猜校正参数。调用方需把 `RealSensorRunner.camera_episode_metadata` 写入 episode manifest，并在 exporter 中显式选择训练使用 raw 还是 rectified stream。

这些公开接口都没有在当前适配路径中提供可信设备采样时间：相机记录 host read-completion monotonic time，Revo 状态/触觉记录 SDK 调用开始与完成时间，EMG 依据 callback 到达时间、包序号和固定 20 点/250 Hz 结构重建包内时间。所有记录均标明 `device_timestamp_ns=None` 或 host reconstruction，不能被解释成硬件同步时间。

`RealSensorRunner` 仅以相互独立的 native rate 将 `camera_raw`（以及启用时的 `camera_rectified`）、Revo state、普通压力流 `tactile_pressure`、可选 VisionTouch 六维力流 `tactile`，以及**二选一**的 EMG 或 BrainCo glove 原生批送入现有 `CollectionSession.accept_sample`；手套批仍统一经过 `BrainCoGloveSource` 的 row parser 和 host-arrival 时间重建。runner 会拒绝同进程同时传入两个 EDU client。它没有动作命令权限，也不会用最低频率统一降采样。`tactile_episode_metadata` 明确标出只有 VisionTouch 流具备 exporter `features` 资格。Tianji 状态/命令仍由外层注入式 backend 和既有安全事务管理，不由该 runner 导入或实例化。

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
- 仅来自逐 SN 模型校验后的 VisionTouch 六轴力特征 `[5, 6]`；
- 30 Hz 时间、任务 instruction 和必要 provenance。

它明确排除：

- EMG 原始信号与 EMG 标签；
- MANUS/BrainCo 手套原始数据；
- Tianji 状态、目标和 action；
- retargeter 中间 target；
- 未获准或未写入的手部 target。

导出完成后还会调用现有 `RevoEpisode.load()` 复核 schema，且元数据必须声明 `contains_emg=false`。EMG 五类主线数据由 `export_emg_dataset()` 写入另一个物理目录；`export_emg_binary_dataset()` 只保留给旧 OPEN/CLOSE smoke。不得通过“训练时忽略若干 key”的方式直接消费 master。两个派生产品可以共享 episode/session provenance ID，但不能共享一个训练 schema 或被误拼接为联合 VLA token。

EMG exporter 只接受显式、人工复核的五类原语时间区间。窗口必须完整落在一个标签区间内、无 lead-off、无超阈值采样间隙；标签绝不由手套轨迹或机器人命令推断。主线先在每段连续 session 上执行 8ch@250 Hz 的因果预处理，再输出 2 秒窗口（500 点）和 0.5 秒 stride（125 点），并强制以 `subject_id` 为 split 单元：

```python
from revo3_teleop.recording import (
    EMGLabelInterval,
    EMGSessionSpec,
    export_emg_dataset,
)

dataset = export_emg_dataset(
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
                    label=0,  # POWER_GRASP；索引来自冻结的五类词表
                    source="protocol_cue_human_reviewed",
                    human_reviewed=True,
                ),
            ),
        ),
        # val/test 必须来自不同 subject；此处省略。
    ),
    output_root="derived/emg_primitives_v1",
)
```

真实训练导出要求 train/val/test 三个 split 均至少一个有效窗口，并拒绝同一受试者跨 split 泄漏。窗口分段同时检查时间间隔、header `dropped_since_previous`、payload `sequence_gap_packets`、无效 packet 和 lead-off；即使 host 重建时间看似连续，也绝不允许窗口跨越显式掉包证据。

## 9. 手套的两条接入路径

### 9.1 MANUS 路径

`ManusRosSource` 是惰性的注入边界，不会自行 import ROS/MANUS，也不会在构造时启动硬件。它解析官方 `ManusGlove` 字段契约并保留 raw nodes、ergonomics 与 raw sensor pose。

重要限制：公开 ROS 消息没有 `Header`，现有 bridge 还会丢弃 MANUS `publishTime`。因此当前 adapter 只能把 host callback arrival 作为时序证据，`device_timestamp_ns=None`，不能伪装成设备时间。只有当集成方提供并核验 `wrist_node_ids`，且该 node 在当前消息中确实存在有限 6DoF pose 时，`provides_wrist_pose` 才能为真。IMU 姿态或任意 skeleton node 不能冒充手腕位姿。

MANUS→Revo 3 可参考 BrainCo 的 `Revo-Retargeting`，但 MANUS SDK 的单独许可、坐标系、左右手、关节顺序、尺度和零位都必须在实际机器上验收。

### 9.2 BrainCo EDU 手套路径

`BrainCoEduSdkGloveClient -> BrainCoGloveSource` 按固定 BrainCo 官方示例保存：

- 6 路 flex，名义 50 Hz；
- IMU，名义 100 Hz；
- magnetometer，名义 20 Hz。

批回调内较早的行会按名义周期向过去重建时间，但这仍是 host-arrival reconstruction，不是设备时钟。该手套路径固定声明 `provides_wrist_pose=false`：flex、IMU 和磁力计可用于手部重定向研究，但不构成经过验证的 Tianji 末端 6DoF 位姿。

两种 source 都默认 `hardware_autostart=false`；concrete EDU client 构造时不会 import SDK、扫 USB 或打开串口。真机必须先用 `configs/hardware_probe.example.json` 中默认关闭的 `glove` component 生成只读 fingerprint，再人工确认同一 fingerprint，并同时显式放开 concrete client 的 `allow_hardware_stream` 与 source 的 `allow_hardware_start`。

固定版本的 `bc-edu-sdk` 示例通过模块级 `set_*_data_callback` 注册回调，而不是给每个设备实例绑定独立 callback。若 BrainCo 手套与 BrainCo EMG 腕带要在同机同时运行，V1 强制采用两个独立采集进程并通过带时间戳的 IPC 汇入 recorder；共享 ownership guard 与 runner 均会阻断同一进程的组合。在没有设备 ID 路由实测证据前，不把两个设备塞进同一 SDK 全局 callback 进程。

### 9.3 手套到 Revo 3 的唯一可执行边界

`RevoHandRetargeter` 只输出带 `calibration_revision`、`model_revision` 和输入样本 provenance 的 canonical `requested_q_rad[21]`；该结果本身不是硬件命令，也不是训练标签。`RevoGloveTeleopController` 强制把它交给 `TeleopRevoWriter -> RevoCommandPipeline -> safety -> awaited backend write`，只有 accepted receipt 的 `exact_sent_target[21]` 可进入 30 Hz action anchor。

MANUS 官方/外部重定向器通过 `PluginRevoHandRetargeter` 或 `load_revo_hand_retargeter(module:factory, expected_module_sha256=...)` 注入，不复制外部代码。非 mock 写入要求 factory 所在可执行模块的 SHA-256 已核对；直接注入但未绑定 hash 的插件只能配合 `MockRevoBackend`。插件还必须显式声明输入类型、校准版本和模型版本，并且输出恰好 21 维有限 rad target。BrainCo EDU 只有 6 路 flex，`BrainCoSixFlexRetargeter` 在没有显式 `BrainCoSixFlexCalibration` 时直接阻断；禁止把 6 个值重复、插值或静默 padding 成 21 维。显式映射需要 `flex_min/max[6]`、`normalized_to_q_matrix[21,6]`、bias、21 维输出边界及版本，超出标定范围的样本拒绝而非外推。

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

`TianjiMarvinBackend` 已实现为不携带第三方 SDK 的注入式窄边界，并通过 fake-native tests 核对接口语义；这不是经厂商或真机验证的 backend。`hardware.example.json` 使用文档保留网段 IP、占位关节顺序和 `provides_wrist_pose=false`，因此按设计**不可能 arm**，但 `hand_collection.enabled=true`，所以不依赖机械臂的 Revo/手套/EMG 数采仍可继续。

当前实现的关键不变量为：

- 内部统一 rad、rad/s、Nm；仅在 native 边界转换为 degree、degree/s；
- 每次位置写严格执行 `OnClearSet -> OnSetJointCmdPos_A/B -> OnSetSend`；
- 只有 `OnSetSend` 成功后才产生 `exact_sent_target`；
- 写入必须同时满足 `allow_hardware_write=True`、已确认 capability probe、每次调用匹配 arm token、7 维台架限位存在、反馈 fresh 且 frame serial 已被证明前进；
- `cur_state==100` 或 `err_code!=0` 按故障处理并 best-effort soft-stop；
- `OnEMG_A/B` 在这里明确命名为 `soft_stop()`，避免与表面肌电 EMG 混淆；
- `close()` 只有在 state 0 已确认时才释放连接，否则抛出 `TianjiPhysicalInterventionRequired`，要求现场物理处理。

真实 ctypes feedback 结构必须通过本机厂商 SDK 的 `feedback_buffer_factory` 与 `feedback_decoder` 注入；仓库不会猜结构体布局，也不会复制 Wuji/Tianji 二进制。

新增的 `TianjiSdkPluginSpec`/`load_tianji_sdk()` 使用显式 `python.module:factory` 插件：本地插件负责厂商动态库的 ABI、结构体 packing 和 decoder，本仓库会分别核对 client factory、buffer factory、decoder、可选 argument adapter 以及实际 native library 的 SHA-256，并将 decoder 统一成 `states/inputs/outputs` mapping。手腕位姿 provider 与 IK solver factory 也各自要求独立的 module hash。`normalized_mapping` 模式适配已有 Python wrapper；`pointer_decoder` 模式要求同时提供 buffer factory、decoder 及二者的 hash。缺少或不匹配任何可执行插件 hash 都会阻断真实写入。加载插件本身不会 `OnLinkTo`，也不会启动伺服或写命令。

`HardwareReadinessReport.arm_write_ready` 是实际控制权限的硬门，而不只是诊断文本：只有完整 report 无 blocker、SDK/反馈/手腕/IK 插件均已加载并完成 hash 绑定时，装配器才会创建可写 backend 和 runtime。否则即使 JSON 中误设 `allow_hardware_write=true`、环境中存在 token，返回的 backend 仍永久保持 `allow_hardware_write=false`，并且不构造 runtime。

如果拿到的 SDK 只是 `.so/.dll`，可把 `client_factory` 设为 `revo3_teleop.backends.tianji_ctypes:create_ctypes_marvin_client`。该通用 client 只配置历史 header 中最小函数原型，必须在 `client_kwargs` 同时提供实际 `library_path` 和 `acknowledge_historical_abi=true`；config 还必须对同一路径提供 native SHA-256。loader 会复核 client 实际报告的 `library_path` 与被哈希文件完全相同，防止“校验 A、加载 B”。真实 `DCSS` packing 仍必须由当前 SDK 对应的 buffer/decoder 插件提供，不能沿用猜测结构。

完整机械臂目标链已经拆成可替换但不绕过验证的组件：

```text
经核验的 MANUS wrist node / 外部 6DoF tracker
  -> WristPose6D（m + xyzw + frame + clock + calibration revision）
  -> RelativeWristRetargeter（相对位姿、轴映射、平移增益、笛卡尔边界）
  -> 注入式 Tianji IKSolver（真实 URDF/tool model revision）
  -> TianjiJointTargetPlanner（收敛、残差、joint order、pose age）
  -> TianjiMarvinBackend（反馈、限位、watchdog、arm token、exact sent receipt）
```

`ManusWristPoseExtractor` 只有在 node id、左右手、位置到米的比例、frame 和 calibration revision 全部显式配置且 `wrist_node_mapping_verified=true` 时才输出 `WristPose6D`。BrainCo EDU 路径即使配置文件错误声称 `provides_6dof=true`，readiness audit 仍会硬拒绝 Tianji arm target；不能用 IMU 姿态补造缺失的平移三轴。

默认无副作用的装配审计命令为：

```powershell
$env:PYTHONPATH="$PWD;$PWD\teleop_data_collection\src"
py -3.10 -m revo3_teleop.cli.hardware_dry_run `
  --config teleop_data_collection/configs/hardware.example.json
```

该命令默认不 import 插件、不连接设备、不写硬件，并分别输出 `hand_collection_allowed`、`arm_planning_blockers` 和 `arm_write_blockers`。只有希望核对本机插件构造与 method presence 时，才可另加 `--load-sdk-plugin`；这仍不连接机器人。`--load-retargeting-plugins` 只允许使用构造时无硬件副作用的 wrist/IK factory。实际连接、position-state enable 和每次 arm token 授权继续是独立步骤，绝不由 dry-run 自动执行。

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
- Revo 3、普通 Pressure/Matrix 触觉、VisionTouch、OpenCV 相机和 EDU EMG concrete client 默认关闭、双重授权、SDK 版本/API、唯一设备 fingerprint、21/42/11/`5x6` schema 与无设备时间戳语义；
- Revo 3 台架批准绑定 probe fingerprint/关节顺序并完成 `deg -> rad`、`rpm|deg/s -> rad/s`、`mA|A -> A` 转换；42 维压力摘要即使投影也不能生成 `features`；
- VisionTouch 五指显式 SN 映射、逐 SN 加密力模型存在性/SHA-256、SDK API、校准生命周期、原始 `(6,)`/非空 `(...,6)` finite 返回与 leading-axis mean、轴序/单位及任一手指异常时的整样本拒绝；
- 鱼眼 rectifier 绑定 `K/D/new_K`、输入/输出尺寸和版本 hash，保持 `camera_raw` 不变并将派生证据写入独立的 `camera_rectified` 流；
- real-sensor runner 的独立 native rate、EMG callback 入账、故障传播与 camera/EMG 生命周期关闭；
- Tianji fake-native 的 rad/degree 转换、A/B 侧、写事务顺序、三重 arming、watchdog、状态 100、错误码、soft-stop 与安全关闭。
- Tianji 本地 SDK 插件惰性加载、插件文件 SHA-256、加载后保持未连接，以及自定义 7 轴 joint-order receipt hash；
- BrainCo EDU 即使被错误配置成 `provides_6dof=true` 也无法放开 arm planning；
- 经核验 MANUS node 的单位换算、相对 wrist retarget、Cartesian 边界、注入式 IK 收敛/残差/joint-order 与 stale-pose 拒绝。
- BrainCo 6-flex 无映射时拒绝、显式 `[21,6]` 标定映射，以及 MANUS fake retarget 请求必须经过 Revo safety/controller write 后才产生 exact-sent 标签。

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
