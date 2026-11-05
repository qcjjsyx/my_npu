# 动态可配置多精度融合 MAC

本项目作为毕业设计推进，研究运行时可配置的多格式乘累加电路，以及计算资源复用、数值精度和硬件代价。主要计算块为 dot16；低精度模式可维护两组独立累加状态。

当前已验证的 SystemVerilog 基线覆盖 FP16、BF16、两种 FP8 及其交叉相乘、INT16/8/4、MXFP8 和 MXINT8 的已定义输入域。Verilog 改写并行推进，尚未完成。

## 从哪里开始

1. 阅读[项目状态](docs/project/status.md)，了解已完成工作、证据及限制。
2. 阅读[架构设计](docs/design/IP_Design.md)与[RTL 实现说明](docs/design/RTL_Design.md)。
3. 查看[硬件验证说明](hardware/verification/README.md)、[数值实验指南](docs/experiments/README.md)。
4. 查看[论文材料](thesis/README.md)；后续架构调整和任务安排由用户确定。

Agent 开始工作前阅读[工作指令](AGENTS.md)。变更历史使用 Git 记录，不另行维护决策、变更或任务台账。

建议从 `NPU/` 根目录启动新会话，确保 agent 加载项目 `AGENTS.md`。文件中的工作指令不替代运行环境权限或自动验证。

## 目录职责

| 目录 | 内容 |
|---|---|
| `hardware/` | SystemVerilog 基线、Verilog 改写、编译清单、独立参考模型与 RTL 验证 |
| `experiments/` | 数值实验 Python 包、测试、配置与批量运行脚本 |
| `docs/` | 项目状态、设计说明、量化调研与实验方法 |
| `thesis/` | 章节初稿及论文支撑报告 |
| `assets/` | 图稿源文件、导出图、生成与检查产物、文档附件 |
| `references/` | [文献](references/papers/README.md)和[开源项目索引](references/projects/README.md) |
| `results/` | 数值运行数据、派生分析、编译与验证日志 |
| `scripts/` | 项目检查和图稿工具 |
| `archive/` | 原始压缩包、工程实践材料、历史交接、旧环境 |

## 安装

以下命令均在项目根目录运行。需要 Python 3.10 以上；本次验证使用 Python 3.13。RTL 仿真需要 `iverilog` 与 `vvp`。

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m mac_experiment version
```

根目录虚拟环境为当前环境；归档的旧虚拟环境仅用于保留迁移前文件，不是运行入口。

## 检查与运行

```sh
# 全部检查：完整性、文档链接、测试、完整 TXT 向量、四类小规模实验
.venv/bin/python scripts/check_project.py

# 只做快速的完整性和链接检查
.venv/bin/python scripts/check_project.py --checks integrity links

# 全部 442 项测试；两套测试可分别指定 hardware/verification/tests 或 experiments/tests
.venv/bin/python -m pytest -q

# 运行已交付的 TXT 向量
.venv/bin/python hardware/verification/tools/run_txt_vectors.py --build results/verification/txt

# 三组小规模位宽实验及分析
.venv/bin/python experiments/scripts/run_widthselect_all.py --plain --plots
```

统一检查的报告保存在 `results/verification/check_<时间戳>/report.json`。数值执行默认写入 `results/numerical/<实验族>/runs/`，派生分析写入相邻 `analysis/`；`--out` 和配置的 `output_root` 可显式指定其他位置。

## 当前证据与限制

2026-10-31 目录重组后的回归：硬件验证 134 项、实验框架 308 项，合计 **442 项通过**；完整 TXT 向量的连续输入、停顿和四类负例检查通过。具体记录见[项目状态](docs/project/status.md)。

目前还没有生产顶层、最终 FP32 转换器、结果端 ready，以及本轮综合/PPA 证据。部分历史正式实验原始数据缺失；早期实验模型与当前 Q2.34 硬件语义不同。有关设计结论必须结合各自的验证范围使用。
