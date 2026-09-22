<div align="center">

# T-Rex × Revo 3

**肌电意图驱动 · 视觉任务规划 · 触觉反应式灵巧抓持**

[English](README.md) · [系统架构](docs/revo3_v1/README.md) · [训练与推理](docs/revo3_v1/TRAINING.md) · [遥操数采](teleop_data_collection/README.md) · [开发指南](docs/DEVELOPMENT.md)

</div>

## 项目简介

T-Rex × Revo 3 是面向单手辅助抓持的模块化研究实现，将肌电抓型识别、Qwen3-VL 视觉任务规划、Revo 专用 T-Rex 动作接口与安全控制运行时连接为一条完整链路，已开展 Revo 3 真机抓持初步试验。

系统将“**用户想做什么**”“**操作哪个目标**”和“**手指怎样运动**”分层处理：肌电给出抓取或释放意图，视觉补全目标与任务语义，动作模型生成触觉反馈驱动的关节轨迹。抓取意图经确认后由系统锁存，用户无需持续收缩肌肉来维持抓持；明确的释放意图触发受控张开。

项目基于 [T-Rex: Tactile-Reactive Dexterous Manipulation](https://github.com/ZhuoyangLiu2005/T-Rex)，扩展了 Revo 3 单手适配、EMG–RGB 任务协调、单相机规划、版本化动作授权与因果数据管线。

## 核心能力

- **意图与任务分层**：五类肌电原语识别，结合置信度、类别间隔、信号质量和持续时间门控，由 RGB 补全抓取对象与场景。
- **结构化视觉规划**：Qwen3-VL-2B 读取三帧全图和当前中心图，返回经过校验的任务 JSON；有歧义时请求澄清。
- **触觉反应式动作**：连续 21 维关节状态、16 步绝对动作序列、slow/fast 推理、原生触觉历史与时间聚合。
- **统一任务生命周期**：单一 Task Executive 管理启动、继续、保持、重规划、释放和结束，异步结果绑定任务与版本。
- **多频率控制**：30 Hz 动作网格与独立调度的 100 Hz 指令写入器，统一执行标定限位、数据时效检查和确认式停止。
- **可追溯训练数据**：原生频率录制、因果对齐、实际下发动作标签、EMG/VLA 分离导出与冻结归一化。

## 系统流程

```text
肌电 ──> 抓型识别 ──────────────────────┐
                                        v
RGB ──> 全图 + 中心图 ──> Qwen Planner ──> Task Executive
                                              │
                        语言 + RGB + q[21] + 触觉
                                              v
                              Revo T-Rex ──> action[16,21]
                                              │
                                      时间聚合 / 插值
                                              │
                                     可选有界触觉修正
                                              v
                                  安全授权 ──> Revo 指令写入器
```

原始 EMG 不进入 VLA 输入或策略训练集。稳定抓持保持在 `HOLD`，直到明确释放。V1 只控制手部，手臂运动不属于 21 维动作空间。

## 任务设置

| 任务 | 抓取原语 | 手部动作 |
|---|---|---|
| `bottle` | `POWER_GRASP` | 抓住并保持居中的瓶子 |
| `phone` | `PRECISION_GRASP` | 精细抓住并保持居中的手机 |
| `plastic_bag` | `PRECISION_GRASP` | 抓住并保持塑料袋提手 |
| `refrigerator_door` | `LATERAL_GRASP` | 侧向抓住并保持冰箱门把手 |

提袋和拉门的位移由用户手臂完成。可选 Tianji 遥操模块用于数据采集，不并入在线手部策略动作空间。

## 快速开始

使用 **Python 3.10**，从仓库根目录执行。下面的本地流程使用确定性模拟后端，无需硬件、下载模型权重或准备真实数据集。

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

输出包含任务状态变化、策略返回、授权写入及关闭状态。合成数据练习和模型单次推理检查见[开发指南](docs/DEVELOPMENT.md)。

## 训练与推理

Revo 策略采用 **pretrain → W0 → W1 → Revo midtrain → SFT** 路线。启动器先检查数据、机器人动作表示、触觉 profile、权重来源和归一化，再构造命令；默认 dry-run，添加 `--execute` 后执行。

```bash
python scripts/revo3_v1_trex.py train --help
python scripts/revo3_v1_trex.py serve --help
python scripts/revo3_v1_runtime.py --help
```

- [训练与推理](docs/revo3_v1/TRAINING.md)：数据准备、分阶段训练、触觉编码器、权重和服务。
- [系统架构](docs/revo3_v1/README.md)：Planner、状态机、时间协议、模型身份和控制接口。
- [遥操数采](teleop_data_collection/README.md)：公开 SDK 适配、设备配置、录制与数据导出。

在线主线使用 **Profile A：Force6D + 五指 DIFF**。数据、训练及服务组件也提供 **Profile B：DIFF-only**，当前在线装配仍固定为 Profile A。Pressure/matrix Profile C 为预留接口，当前训练器和运行时不启用。

## 目录导航

| 目录 | 用途 |
|---|---|
| [`revo3_v1/`](revo3_v1/) | EMG、Planner、任务状态机、策略接口、运行时与控制 |
| [`qwen_vla/`](qwen_vla/) | Qwen 多专家 VLA 模型 |
| [`tactile_vqvae/`](tactile_vqvae/) | 时序触觉 tokenizer 与训练工具 |
| [`teleop_data_collection/`](teleop_data_collection/) | 原生数据录制、公开硬件适配与导出 |
| [`scripts/`](scripts/) | 运行、训练、推理和数据入口 |
| [`config/`](config/) | 模型/控制配置与集成 schema 示例 |
| [`tests/`](tests/) | 运行时、模型接口、时间、数据和安全测试 |
| [`dataset_quickstart/`](dataset_quickstart/) | 上游 T-Rex 数据浏览与回放工具 |
| [`hardware_code/`](hardware_code/) | 上游双手机器人硬件参考栈 |

## 硬件运行

启用写入前，先配置设备身份、关节约定、传感器时钟和经过标定的安全限位。SDK 调用通过明确接口接入，保留 arming、watchdog、物理急停与现场操作监督；示例配置默认关闭硬件写入。

## 许可与引用

项目保留 [MIT License](LICENSE)。外部 SDK、模型权重、数据集及第三方组件分别遵循各自条款，见 [NOTICE](NOTICE.md) 和数采文档中的来源记录。引用原始 T-Rex 工作时，请将本仓库标明为 Revo 3 集成；论文引用信息见 [English README](README.md#citation)。
