# 数值实验指南

实验包位于 `experiments/src/mac_experiment/`；安装和执行都在项目根目录完成，见[根 README](../../README.md)。当前实验框架测试为 308 项。

## 实验族与默认输出

| 实验族 | 配置与用途 | 原始运行 | 派生分析 |
|---|---|---|---|
| dot_product | `smoke_width.yaml`、`smoke_block.yaml`；固定窗口与块数扫描 | `results/numerical/dot_product/runs/` | 同级 `analysis/alignment_only/`、`analysis/bitmatch/`；基础 analyze 写入源 run 的重算文件 |
| mxfp8 | `mx_smoke.yaml`、`mx_pilot.yaml`；MXFP8 矩阵计算 | `results/numerical/mxfp8/runs/` | 同级 `analysis/` |
| mx_alignment | `mx_align_smoke.yaml`；固定对阶窗口实验 | `results/numerical/mx_alignment/runs/` | 同级 `analysis/` |
| widthselect | `widthselect_*`；对阶、累加及联合位宽扫描 | `results/numerical/widthselect/runs/` | 同级 `analysis/` |

所有 YAML 在 `experiments/configs/` 下。`output_root` 可显式指定，离线分析支持 `--out`；相对路径按当前工作目录解释，因此以下命令从项目根目录运行。

```sh
# 日常小规模实验：每次执行创建运行目录；固定 run_name 的配置重复运行会拒绝覆盖
.venv/bin/python -m mac_experiment run --config experiments/configs/smoke_block.yaml
.venv/bin/python -m mac_experiment run --config experiments/configs/mx_smoke.yaml
.venv/bin/python -m mac_experiment run --config experiments/configs/mx_align_smoke.yaml

# 三个小规模位宽实验，分别执行和离线分析
.venv/bin/python experiments/scripts/run_widthselect_all.py --plain --plots

# 用实际输出的 run 路径替换 <run_id>
.venv/bin/python -m mac_experiment analyze --run results/numerical/dot_product/runs/<run_id>
.venv/bin/python -m mac_experiment plot --run results/numerical/dot_product/runs/<run_id>
.venv/bin/python -m mac_experiment mx-analyze --run results/numerical/mxfp8/runs/<run_id>
.venv/bin/python -m mac_experiment mx-align-analyze --run results/numerical/mx_alignment/runs/<run_id>
.venv/bin/python -m mac_experiment width-analyze --run results/numerical/widthselect/runs/<run_id> --plots
```

也可用 `.venv/bin/python scripts/check_project.py --checks smoke` 执行四类独立小规模运行、分析和绘图。它使用独立名称，不覆盖已有结果。正式配置可能耗时数小时，不属于日常检查。

## 模型职责和口径

- 实验包采用精确 dyadic 表示与明确舍入策略，分析对阶、累加、位匹配和 ULP 等指标；配置和 manifest 描述具体实验语义。
- 硬件逐笔预言机独立位于 `hardware/verification/reference/`，采用当前 Q2.34/bias127 合同；它不依赖本实验包。实验中的 FP32 或早期 `hwacc-v1` 累加器不能替代这个模型。
- 主误差口径区分量化误差、对阶损失、累加损失；非有限样本按显式状态报告。未配置精度阈值时只报告结果，不自行认定某个位宽合格。
- 对齐与累加的具体方法见[基础设计](design.md)、[MX 语义](mx_semantics.md)、[固定对阶](mx_alignment.md)、[位宽选择](mac_width_selection.md)和[历史验证记录](verification.md)。

## 已保留和缺失的数据

[历史位宽分析](../../results/numerical/widthselect/analysis/)保留汇总、manifest 和图片，对应[支撑报告](../../thesis/supporting_reports/results_width.md)。[Q_S_shared.csv](../../results/numerical/mx_alignment/imports/Q_S_shared.csv)为原无扩展名 CSV 的保留副本；原数值运行日志在 `results/numerical/history/logs/`。

旧 README 提及的正式 W/B run、MX 正式输入、正式位宽原始 run 及 `archive/legacy_20260916_190730/` 在当前资料中均未找到，现有 `models.zip` 也不含这些数据。无法用现有资料完整重放相关正式实验。历史 manifest 中的原始路径和 digest 保持原值。

`mx_align_prefix_replay.yaml` 需要缺失的 MX 正式输入，当前不可直接运行；可用 fresh 输入的 `mx_align_smoke.yaml` 检查功能。本次新生成的小规模实验不补写为旧正式实验。

旧完整 README 的原文保存在[迁移前文本快照](../../archive/snapshots/pre_reorganization_text_20261031.tar.gz)，其运行状态和目录描述作为历史记录使用。
