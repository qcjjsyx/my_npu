# 数值实验指南

实验包位于 `experiments/src/mac_experiment/`；安装和执行都在项目根目录完成，见[根 README](../../README.md)。旧实验保留在 `experiments/tests/legacy/`，新局部对阶实验测试在 `experiments/tests/local_alignment/`。

## 实验族与默认输出

| 实验族 | 配置与用途 | 原始运行 | 派生分析 |
|---|---|---|---|
| dot_product | `smoke_width.yaml`、`smoke_block.yaml`；固定窗口与块数扫描 | `results/numerical/dot_product/runs/` | 同级 `analysis/alignment_only/`、`analysis/bitmatch/`；基础 analyze 写入源 run 的重算文件 |
| mxfp8 | `mx_smoke.yaml`、`mx_pilot.yaml`；MXFP8 矩阵计算 | `results/numerical/mxfp8/runs/` | 同级 `analysis/` |
| mx_alignment | `mx_align_smoke.yaml`；固定对阶窗口实验 | `results/numerical/mx_alignment/runs/` | 同级 `analysis/` |
| local_alignment | `local_alignment/smoke.yaml`、`formal_template.yaml`；折叠系数完整乘积的局部 floor/RNE 探索 | `results/numerical/local_alignment/runs/` | 每个 run 内 JSON/CSV |
| widthselect | `widthselect_*`；对阶、累加及联合位宽扫描 | `results/numerical/widthselect/runs/` | 同级 `analysis/` |

旧 YAML 在 `experiments/configs/legacy/` 下，新实验配置在 `experiments/configs/local_alignment/` 下。`output_root` 可显式指定，离线分析支持 `--out`；相对路径按当前工作目录解释，因此以下命令从项目根目录运行。

```sh
# 日常小规模实验：每次执行创建运行目录；固定 run_name 的配置重复运行会拒绝覆盖
.venv/bin/python -m mac_experiment run --config experiments/configs/legacy/smoke_block.yaml
.venv/bin/python -m mac_experiment run --config experiments/configs/legacy/mx_smoke.yaml
.venv/bin/python -m mac_experiment run --config experiments/configs/legacy/mx_align_smoke.yaml

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
- 硬件逐笔预言机独立位于 `hardware/verification/reference/`，采用旧基线 Q2.34/bias127 合同（该合同不适用于 2026-11-05 新架构，见[已确认约束](../project/decisions.md)）；它不依赖本实验包。实验中的 FP32 或早期 `hwacc-v1` 累加器不能替代这个模型。
- 主误差口径区分量化误差、对阶损失、累加损失；非有限样本按显式状态报告。未配置精度阈值时只报告结果，不自行认定某个位宽合格。
- 对齐与累加的具体方法见[基础设计](design.md)、[MX 语义](mx_semantics.md)、[固定对阶](mx_alignment.md)、[位宽选择](mac_width_selection.md)和[历史验证记录](verification.md)。

## 新局部对阶实验：local_alignment

此实验保留算术 floor 截断和逐项 RNE 两种比较能力；当前设计已在[已确认约束](../project/decisions.md)中选定局部截断及各格式窗口，历史比较结果保留原语义。只隔离局部对阶损失：完整精确乘积、非零 MAX、有限窗口逐项舍入、精确整数求和、应用各自 MX scale，随后精确 dyadic 累计。**不包含配对 RNE、状态 RNE、RTL 或 PPA 验证。**

```sh
.venv/bin/python -m mac_experiment local-align-run --config experiments/configs/local_alignment/smoke.yaml --quiet
.venv/bin/python -m mac_experiment local-align-replay --run results/numerical/local_alignment/runs/<run_id> --index 0
.venv/bin/python -m pytest -q experiments/tests/local_alignment
```

`formal_template.yaml` 提供完整候选模板，默认检查不会运行它。可改 `seeds`、`sequences`、`lengths`、`formats`、`F`、`rounding`、`distributions`、`output_root`。长度限制为 32 的倍数且不超过 16384；F 候选默认 `[0,1,2,3,4,6,8,12,16]`，长度默认 `[32,128,512,2048,8192,16384]`。完整模板组合较多，运行时间和精确分位数频次表内存随工作量及不同误差值数量增长，正式运行前需按实际资源调整序列数；本次仅执行 smoke。

原六种模式为 `mxfp8_e4m3fn`、`mxfp8_e5m2`、`mxfp8_e4m3fn_e5m2`、`mxfp8_e5m2_e4m3fn`、`fp16`、`bf16`；另有显式选择的 `bf16_postfold` 乘后折叠探索，默认模式列表仍为原六种。FP8 只测 MX。每种分布、seed、序列先生成同一份有限 FP32 A/B，所有格式从这批源量化；同一序列的多个长度使用同一最大长度源的前缀。复用旧六种固定分布定义及采样器；`uniform` 是有限 FP32 编码均匀抽样，`positive` 是正有限非零 FP32 编码均匀抽样，均不等同于实数区间均匀分布。

FP16/BF16 使用逐元素 RNE，保留 subnormal，量化溢出饱和。MX 对 A/B 分别沿 K 每 32 项取最大绝对值，选择覆盖该最大值的最小二进制幂 scale（E4M3FN 上限 448，E5M2 上限 57344），指数钳位至 E8M0 `[-127,127]`；全零块 scale=1。按选定 scale 量化元素，采用 RNE、保留 subnormal、溢出饱和。`quantization.json` 独立记录两侧输入 saturated（RNE 后编码超过最大有限值才钳位）、非零归零、输出 subnormal 与 scale 钳位计数，统计范围为最大长度输入且不重复计入不同候选。MAC 误差参考从量化编码及 scale 的精确乘积出发，源 FP32 到目标格式的量化损失不混入算术误差。

| 路径 | 对阶系数容器 C | 指数粒度 g | 乘积偏置 b | 局部 K |
|---|---|---|---|---|
| MXFP8 | signed16 | 4 | 36 | 16 |
| FP16 | signed30 | 4 | 50 | 8 |
| BF16 | signed18 | 1 | 268 | 16 |
| BF16 乘后折叠 | signed21（signed18 完整乘积重编码） | 4 | 268 | 16 |

按已确认解码及折叠，乘积为 `P_i × 2^(g T_i-b)`。从非零乘积求 `Tmax`，`d_i=g(Tmax-T_i)`，`n_i=floor(P_i×2^(F-d_i))` 或同式逐项 RNE，局部贡献 `Q=sum(n_i)×2^(g Tmax-b-F+h)`，其中 MX 的 `h=sA+sB`，普通模式为零。floor 是有符号算术右移，因此长距离右移负非零值为 -1；不采用向零截断。全零组输出零，零不参与 MAX。每项窗口宽度 C+F，局部和宽度 C+F+ceil(log2 K)，均检查有符号范围，不回绕。这里是结构位宽记录，不是综合面积测量。

每次运行使用时间与随机后缀建立独立目录，拒绝复用同名目录。输出包括：

- `config.json`、`metadata.json`：解析后的配置、seed、算法版本、代码哈希、Git HEAD/工作区状态、采样分布定义和统计口径。
- `sources.jsonl.gz`：原始 FP32 位模式、各目标量化编码、两侧 scale 和量化事件，`source_id` 可关联所有候选。
- `samples.jsonl.gz`：逐序列/长度/候选的精确 S、T、L1，最终 FP32 位模式、饱和标志和 ULP。
- `summary.json`、`summary.csv`：逐分布、模式、长度、F、舍入方式独立统计。逐项 changed/zeroed 的 rate 分母是非零乘积；局部及长点积 T-S 包含 exact-match、绝对误差均值/最大值/p50/p99、有符号均值、误差正负计数与比例、绝对误差/L1 均值/最大值/p50/p99。L1 为零单列，不进入归一化分母。分位数按完整样本的最近秩法精确计算，数值展示转换为 binary64。
- `representatives.json`：每桶最大绝对 T-S 的代表序列（相同误差取首个）及第一个 changed 局部块的逐项 P/T/shift/n trace；全程无 changed 时保留最后块。`local-align-replay` 从保存编码重算，不重新采样。

最终 S 和 T 分别仅做一次 FP32 RNE、有限值饱和、渐进下溢；精确零输出 +0，非零微小负值舍入为零时保留 -0。最终指标包含 bitmatch、ULP、输出绝对误差、有符号零不匹配。饱和样本的匹配单列，另给排除任一侧饱和后的 bitmatch 计数/分母/rate，避免共同饱和造成虚假精度结论；精确 T-S 始终保留。无验收阈值，不自动判定某个候选合格。

### BF16 乘后指数折叠候选

`bf16_postfold` 是独立实验模式，已选为当前BF16局部对阶方案，见[已确认约束](../project/decisions.md)；原 `bf16` 实验模式和外部 BF16 解码保持不变，供历史对照。乘法仍先生成 signed18 完整乘积 `P=mA*mB`，原乘积指数和 `T=EA+EB`；随后精确保值重编码 `r=T mod 4`、`Q=floor(T/4)`、`P'=P<<r`。先在 signed18 检查完整乘积，再扩展到 signed21 保存重编码系数，满足 `P×2^(T-268)=P'×2^(4Q-268)`。21 位是乘后系数容器，**不表示改成21位乘法器或新增PPA结论**。

局部仅从非零 `P'` 选择最大粗指数 `Qmax`，然后延伸 F 位并按粗指数差对阶：`n=floor(P'×2^(F-4(Qmax-Q)))`，局部贡献为 `sum(n)×2^(4Qmax-268-F)`。也可通过通用配置探索 RNE；本轮候选配置专用 floor。全零组贡献为零，归约继续使用精确整数和，不增加配对或状态舍入。

新配置 `experiments/configs/local_alignment/bf16_postfold.yaml` 设置3个 seed、每 seed 64条序列、六种分布及全部长度，`F=11`、`rounding=[floor]`。它是正式工作量模板，默认检查不运行它；若只做定向 smoke，应先将序列数、长度和分布减小。用户已有 `formal_template.yaml` 保持原值。

比较相同32位对阶窗口时，原 BF16 为 `C=18,F=14`，乘后折叠为 `C=21,F=11`，两者局部和均为 signed36。令原非零乘积的最大指数为 `Tmax`，原绝对 LSB 为 `Tmax-282`，新绝对 LSB 为 `4floor(Tmax/4)-279`；两者相差 `3-(Tmax mod 4)` 位。因此重编码本身精确，但同为32位容器时，新模式可能少保留0至3位；`Tmax mod 4=3` 时两者对阶网格相同，逐项结果相同。其他余数下，当各项恰好落在新网格时仍相同，不能仅凭窗口总宽相同推断精度相同。

summary 分别记录 `product_width=18`、`coefficient_width=21`、`window_width=32`、`sum_width=36`。代表 trace 以 P/T 字段保存重编码后的 P'/Q，额外保留 `original_P`、`original_T`、`fold_r`、`multiplication_product_width` 与 `alignment_coefficient_width`，可检查每项精确保值关系；CLI 运行和重放方式沿用上一节。metadata 版本为 `local-alignment-v2`，并保存整个实验包的源码哈希。

## 已保留和缺失的数据

[历史位宽分析](../../results/numerical/widthselect/analysis/)保留汇总、manifest 和图片，对应[支撑报告](../../thesis/supporting_reports/results_width.md)。[Q_S_shared.csv](../../results/numerical/mx_alignment/imports/Q_S_shared.csv)为原无扩展名 CSV 的保留副本；原数值运行日志在 `results/numerical/history/logs/`。

旧 README 提及的正式 W/B run、MX 正式输入、正式位宽原始 run 及 `archive/legacy_20260916_190730/` 在当前资料中均未找到，现有 `models.zip` 也不含这些数据。无法用现有资料完整重放相关正式实验。历史 manifest 中的原始路径和 digest 保持原值。

`mx_align_prefix_replay.yaml` 需要缺失的 MX 正式输入，当前不可直接运行；可用 fresh 输入的 `mx_align_smoke.yaml` 检查功能。本次新生成的小规模实验不补写为旧正式实验。

旧完整 README 的原文保存在[迁移前文本快照](../../archive/snapshots/pre_reorganization_text_20261031.tar.gz)，其运行状态和目录描述作为历史记录使用。
