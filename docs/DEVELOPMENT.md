# 开发与本地验证

[项目首页](../README_ZH.md) · [架构参考](revo3_v1/README.md) · [训练指南](revo3_v1/TRAINING.md)

本页集中说明开发环境、离线示例和回归测试。所有命令从仓库根目录执行。

## 环境分层

| 安装入口 | 用途 |
|---|---|
| `requirements-runtime.txt` | 本地运行时、数据处理与模型接口 |
| `requirements-dev.txt` | 上述依赖加 pytest；用于首页快速开始 |
| `requirements.txt` | 从 `pyproject.toml` 安装完整模型训练依赖 |
| `teleop_data_collection/pyproject.toml` | 独立数采包及按设备选择的 SDK extras |
| `hardware_code/pyproject.toml` | 上游双手硬件栈的独立环境 |
| `dataset_quickstart/pyproject.toml` | 上游 LeRobot 数据集工具的独立环境 |

Python 版本为 3.10。不要将所有硬件、训练和数据浏览环境混装；根据实际入口选择依赖。`requirements-demo.txt` 保留为开发依赖的兼容入口。模型训练依赖含 DeepSpeed，使用独立 Linux/CUDA 环境；CUDA wheel 由部署环境选择。

## 回归检查

```bash
python -m pytest -q
python scripts/check_release.py
```

pytest 覆盖意图门控、任务状态机、异步结果版本、动作与触觉 schema、数据因果性、checkpoint lineage、SDK 注入接口以及硬件写入授权。测试使用临时数据和注入式设备，不要求连接机器人。

`check_release.py` 检查发布文档本地链接、依赖入口、默认硬件写入配置、Notebook 执行输出和发布目录卫生。它不是完整的安全审计或硬件验收工具。

## 本地任务演练

```bash
python scripts/revo3_v1_runtime.py --mode simulation --task all --servo-ticks 120
```

该命令用确定性的模拟 Planner、policy 和设备运行四类任务，报告任务状态、授权写入与安全关闭。它用于检查运行时调度，不加载真实模型，也不作为抓持成功率或控制延迟的测量。

## 合成 EMG 流程

```bash
python scripts/revo3_v1_generate_emg.py --output outputs/emg_synthetic --preset smoke
python scripts/revo3_v1_train_emg.py --dataset outputs/emg_synthetic --output outputs/emg_pipeline --from-scratch-ablation --allow-window-reset-fallback --preset smoke --epochs 3
```

生成器输出五类、8 通道、250 Hz 的带 split 与时间戳数据。上述小模型训练显式使用 from-scratch / window-reset 流程，只检验数据到 checkpoint 的连接。真实 EMG 训练使用 GNI 初始化、连续会话因果预处理及独立会话划分，详见 [EMG 模块](../revo3_v1/emg/README.md)。

## 机器人数据与导出

```bash
python scripts/revo3_v1_generate_robot_demo.py --help
```

该入口生成合成 robot-only episode 并调用转换器，适合检查 schema 和读取链路。真实数采通过 [数采指南](../teleop_data_collection/README.md) 记录 native streams，再调用 `revo3_v1.data.convert_revo_episodes_to_trex_json`，按[训练指南](revo3_v1/TRAINING.md)准备 conversion/readiness manifest。不要混用合成样例和训练数据。

## 模型入口检查

```bash
python scripts/revo3_v1_qwen_smoke.py --help
python scripts/revo3_v1_trex.py train --help
python scripts/revo3_v1_trex.py serve --help
```

Qwen 检查工具按指定模型与可选 LoRA 执行一次加载和生成，记录 revision、输入视图、JSON 解析和显存状态。实际执行需要相应模型文件与运行环境；`--help` 仅显示参数。T-Rex 的 Revo 启动器先验证 artifact，默认不启动训练或服务。请将单模型检查与完整抓持任务评估分开。

## 提交与发布

- 功能变化同时补充回归测试，保持默认硬件写入关闭。
- 公共示例使用相对路径或显式环境变量；本机设备配置、数据和模型放在被忽略的目录。
- Notebook 保留代码与说明，提交前清空输出及执行计数。
- 更新接口时同步首页、架构说明与对应子模块文档；维护第三方来源和许可证。
- 发布包只包含代码、配置模板、文档和必要静态资源；运行生成物不入库。

详细约定见 [CONTRIBUTING](../CONTRIBUTING.md)。
