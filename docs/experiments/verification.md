# 验证记录（2026-09-16）

> 2026-10-31 目录迁移说明：本文保留原实验语义与历史记录。代码和配置现位于 `experiments/`；当前运行方式及缺失的历史数据见[实验指南](README.md)。历史实验结论不直接等同于当前 Q2.34 RTL 的精度证据。

环境：macOS (darwin)，Python 3.13.3，`.venv`（pytest 9.1.1、numpy 2.5.3、matplotlib 3.11.2、
PyYAML 6.0.3）。所有命令在 `/Users/huangyuan/qcjjsyx/NPU` 下执行。

## 1. 归档复核

```
$ .venv/bin/python archive_legacy_20260916_190730/archive_legacy.py --verify archive/legacy_20260916_190730
archive: /Users/huangyuan/qcjjsyx/NPU/archive/legacy_20260916_190730
files in manifest: 466, failures: 0
```

- 归档 preview：19 个顶层条目、466 个文件、31,838,820 字节；
- `--apply` 后逐文件复核 sha256，0 失败；
- 再次 preview：0 个条目（幂等，不会重复归档）；
- 目标目录已存在时拒绝覆盖；`IMPLEMENTATION_BRIEF.md`、`.venv`、`archive/` 不参与归档。

## 2. 单元测试

```
$ .venv/bin/python -m pytest -q
95 passed in 11.21s
```

与实施说明第 8 节逐条对应：

| 要求 | 测试 |
|---|---|
| RNE 正负/halfway、FP32 正常/次正规边界、进位、精确抵消、溢出 | `test_rounding.py`（含 400+ 随机 dyadic 与独立 Fraction oracle 比对） |
| `acc=-1, partial=1+2^-24` → `2^-24` | `test_accumulator.py::test_no_early_rounding_of_partial_sum` |
| 完整乘法与窗口对阶 vs 独立 oracle；窗口总宽关系 | `test_reducer.py`（Fraction oracle 对齐、W+7 容器范围证明、越界抛错） |
| 逐样本精确误差分解恒等式 | 引擎内断言 + `test_decomposition.py::test_decomposition_identity_and_exact_arithmetic` |
| 无丢位样本候选==参考 | `test_decomposition.py::test_no_dropped_bits_case_candidate_equals_reference`（FP8 W=41 全范围无丢位；未把随机样本 W=41 一律当要求） |
| B=1 边界、同 W 不同 driver 一致、前缀一致 | `test_decomposition.py`、`test_experiments.py`、`test_results.py::test_block_run_prefix_stability_across_runs` |
| 七分布编码合法性、PMF 非负归一、频率、cancellation 无拒绝/保幅度、独立 sign 流对照 | `test_sampling.py` |
| 零分母、全零误差、精确 match、p99 插值、事件计数、严格 JSON | `test_metrics.py`、`test_engine_events.py`、`test_results.py` |
| 从逐样本记录重算与首次汇总一致 | `test_results.py::test_tiny_width_run_reanalyze_and_manifest`、`test_reanalysis_detects_tampering` |
| 两格式两实验端到端 smoke，W smoke 覆盖 24 个 W | `test_experiments.py`、`test_cli.py` |
| 实际检查生成图 | `test_plotting.py` + 人工查看 PNG（见第 4 节） |

## 3. smoke 运行

```
$ .venv/bin/python -m mac_experiment run --config configs/smoke_width.yaml
[run] 28/28 samples
[run] completed: results/width_sweep_20260916_112028_e621eea9   (≈1.4 s)

$ .venv/bin/python -m mac_experiment run --config configs/smoke_block.yaml
[run] 28/28 samples
[run] completed: results/block_sweep_20260916_112029_3f6980b6   (≈6.4 s)
```

配置：2 格式 × 7 分布 × 1 seed × 2 样本；W run 固定 B=8 扫描 W=18..41；
B run 固定 W=32、B∈{1,2,4,8,16,32,64,128,256,512,1024}。

| run | 快照记录 | 重算 | 备注 |
|---|---|---|---|
| width_sweep_20260916_112028_e621eea9 | 672 | 一致（0 差异） | 每 cell 24 个 W 点、2 样本 |
| block_sweep_20260916_112029_3f6980b6 | 308 | 一致（0 差异） | 每 cell 11 个 B 前缀、2 样本 |

```
$ .venv/bin/python -m mac_experiment analyze --run results/width_sweep_20260916_112028_e621eea9
re-analysis matches the original summary (672 records)
$ .venv/bin/python -m mac_experiment plot --run results/width_sweep_20260916_112028_e621eea9
wrote 36 files to .../plots
```

smoke 数值检查（抽样，非结论）：

- 所有记录 `status=ok`，无 abort、无非有限、无 window_overflow；
- 事件计数正常出现（如 TF32 positive 的 `alignment_zeroed`、`update_absorbed`）；
- FP8 在 W=41 时 `a_bits == r_bits`（窗口无丢位），符合预期；
- `fp32_baseline` 指标对 W 完全不变（`test_experiments.py` 断言）；
- E_align + E_acc = a_B − S_B 在每个快照上精确成立（引擎断言）。

## 4. 图检查

每个 run 生成 36 个文件（PNG+SVG）：7 指标 × 2 格式 + 2 阶段诊断 × 2 格式。

已人工查看：

- `w_sweep_fp8_e4m3_p99_abs_error.png`：七分布面板、三种比较、图例含分母说明、
  symlog 零值在 0、标题含格式/固定 B/样本数/策略；
- `w_sweep_tf32_stage_E_align.png`：阶段均值/RMS 独立诊断图，有符号值 symlog；
- `b_sweep_tf32_relative_l2.png`：B 轴为 log2（1..1024），全点保留。

绘图只读取 `summary.json`；`test_plotting.py` 覆盖 undefined=null 场景不崩溃。

## 5. 已知限制与未执行项

- 未运行正式大样本实验（`configs/formal_*.yaml` 仅作为计划与默认值，未执行）。
- 未实现 bootstrap 区间、alpha 拟合、ULP 主流程；本轮只交付原始指标曲线。
- 单线程 Python；正式量级约 8–12 小时/实验（见 README 估计）。
- 未提交/推送 Git（目录本身也不是 Git 仓库；`git` 命令受本机 Xcode 许可限制不可用）。
- 对齐策略固定 RNE ties-to-even（接口可替换）；不代表真实硬件确认采用 RNE。
- 历史 e30、accumulator-aware、17-term fused、BF16 路径未实现，也未建立兼容层。

## 6. 非有限样本统计修复（2026-09-17）

修复“非有限样本被排除后，汇总仍显示正常精度”的问题。不改变采样器、对阶模型或单次
FP32 舍入语义，不重新运行正式实验。

### 6.1 修改文件与状态规则

| 文件 | 修改 |
|---|---|
| `version.py` | schema 升为 `mac-exp-v2`（summary 升为 `mac-summary-v2`），新增 `SUPPORTED_SCHEMAS` |
| `metrics/compare.py` | 比较级状态 `ok / partial_nonfinite / no_valid_samples / no_samples`，coverage（n_total/n_valid/n_invalid/invalid_rate），`finite_subset`；阶段累计器同策略 |
| `metrics/aggregate.py` | 按比较对象独立判定有效性；E_acc 完整性判定；cell 级 `status`/`comparison_statuses` |
| `execution/engine.py` | 记录 `a_finite`/`r_finite`/`E_acc_complete`/`E_acc_prefix`/`E_acc_prefix_until_block`；恒等式仅在各项有定义时检查 |
| `results/writer.py` | CSV 增加 `scope`/coverage/note 列，整批与有效子集分行 |
| `results/reanalyze.py` | 支持 v1 记录离线重算（`E_acc_complete` 由候选有限性推导），重算不覆盖 `summary.json` |
| `plotting/*` | 整批 null 用红 ×、子图标题标注失败点数与失败样本数；`finite_subset` 只作虚线诊断曲线 |

规则：

- total 需要候选有限；window_effect 需要候选与参考有限；fp32_baseline 需要参考有限。
  只有候选非有限时 fp32_baseline 仍正常统计；只有参考非有限时 total 仍正常统计。
- 部分无效：整批指标全部 `null` + `partial_nonfinite`，有效子集指标另存并明确标注。
- 全部无效：`no_valid_samples`，`relative_l2` 不再返回 0。
- 全部有效：指标定义与数值与修复前完全一致（下方 6.4 验证）。
- 一旦候选累加出现非有限值，`E_acc` 写 null，`E_acc_prefix`（含截至 block）仅作诊断；
  `E_align` 仍完整保存和统计。

### 6.2 修复前后最小复现

修复前（用户复现）：全无效 → `relative_l2=0.0, status=ok`；1 有效 + 1 无效 →
`match_rate=1` 且状态 ok；溢出后仍保存不完整 E_acc。

修复后（独立输出目录实测）：

```
--- all samples invalid
status=no_valid_samples
relative_l2={'value': None, 'status': 'no_valid_samples'}
coverage={'n_total': 1, 'n_valid': 0, 'n_invalid': 1, 'invalid_rate': 1.0}
finite_subset=None

--- one valid + one invalid
status=partial_nonfinite
relative_l2={'value': None, 'status': 'partial_nonfinite'}
match_rate={'value': None, 'status': 'partial_nonfinite'}
coverage={'n_total': 2, 'n_valid': 1, 'n_invalid': 1, 'invalid_rate': 0.5}
finite_subset=present: computed over the finite subset only; not a whole-batch metric

--- real FP32 overflow: fp32=Fp32(0x7F800000) delta=None overflowed=True
--- E_acc after overflow
status=partial_nonfinite  mean=None rms=None
```

合成 run（独立目录，含 1 个候选非有限样本）经 CLI `analyze` + `plot`：
重算与首次汇总一致；CSV 中 `total` 比较整批行全部 `value=""`、`status=partial_nonfinite`，
`finite_subset` 行带 note；图上整批点为红 ×，子图标题标注
`invalid: total 1/1 x-points (1 invalid samples)`，子集曲线为虚线。

### 6.3 新增/更新测试

```
$ .venv/bin/python -m pytest -q
105 passed in 13.24s
```

- 新增 `tests/test_nonfinite_status.py`（10 个用例）：全有效零误差、1 有效 + 1 候选
  非有限、全部候选非有限、仅候选非有限时 baseline 可统计、仅参考非有限时 total 可统计、
  真实 FP32 溢出（`Fp32Accumulator` + 伪造 reducer 触发引擎路径，校验 `E_acc_prefix` 与
  `E_acc_prefix_until_block`）、零参考与无有效样本区分、严格 JSON/CSV/离线重算/绘图、
  有效指标数值回归（含真实小 run 的独立精确重算）。
- 更新 `test_metrics.py::test_invalid_sample_marks_no_valid_samples`、
  `test_engine_events.py::test_nonfinite_sample_marks_metric_undefined` 以匹配新语义。

### 6.4 对既有全有效数据的影响

用新代码离线重算（不写文件、不覆盖原 summary）并与原 v1 汇总逐值比较：

| run | cells | 比较值 | 重算状态 | 数值差异 |
|---|---|---|---|---|
| width smoke `..._112028_e621eea9` | 336 | 13440 | 1008 ok | 0 |
| block smoke `..._112029_3f6980b6` | 154 | 6160 | 462 ok | 0 |
| formal width `..._152839_ee61f8e7` | 336 | 13440 | 1008 ok | 0 |
| formal block `..._152916_0144683e` | 154 | 6160 | 462 ok | 0 |

即：全部有效数据的指标数值完全不变；新增字段（status/coverage/finite_subset 等）只在
重算输出中出现，`summary.json` 不被覆盖。v1 run 仍可用 `analyze` 重算并写入
`summary_recomputed.json`。

## 7. alignment_only 比较（2026-09-17）

新增第四种比较 `alignment_only`：误差为精确 `E_align = Σ(q_b − s_b) = Q_B − S_B`，
参考为 `S_B`，eta 分母为 `L1_B`。`Q_B` 全程精确、不转 FP32；误差按有符号逐 block 累计，
不是逐 block 绝对误差相加。有效性只看 `E_align` 是否完整，与 FP32 候选/参考是否有限无关。

### 7.1 修改文件

| 文件 | 修改 |
|---|---|
| `events.py`（新） | `EVENT_KEYS` 迁移到无依赖模块；engine 再导出兼容 |
| `metrics/aggregate.py` | `COMPARISONS` 增加 `alignment_only`；`_validated_e_align` 校验 E_align（缺失/null/畸形/`E_align_complete=false` → 无效，不默认成零） |
| `results/enhance.py`（新） | 流式离线增强入口；独立输出目录；`analysis_manifest.json` 记录源身份 |
| `results/schema.py` | `SUMMARY_VERSION = mac-summary-v3`；记录 schema 与引擎数值语义不变 |
| `cli.py` | 新增 `enhance`；重导入移入 handler（离线入口不加载 sampling/execution） |
| `plotting/*` | 第 4 条曲线（紫色三角）；图例/图注含 Q_B 说明 |
| `README.md`、`docs/design.md` | 比较定义、有效性、离线命令、schema 变更记录 |

### 7.2 测试

```
$ .venv/bin/python -m pytest -q
115 passed in 15.32s
```

新增 `tests/test_alignment_only.py`（10 例）：保存的 E_align 等于测试内独立
`Σ(q_b − s_b)`（Sampler + BlockReducer 手工驱动）；FP8 W=41 无对阶损失全零且 match=1；
正负误差有符号累计（含一个逐 block 混合符号、`|Σ| < Σ|·|` 的判别样本，且断言存储值为
有符号和）；与 Fraction 独立手算的 p99/eta_L1/relative_l2/rmse/mean/reference_rms 一致；
FP32 真实溢出（伪造 reducer 触发引擎路径）时 alignment_only 仍 ok 且 mean/RMS 等于
stage E_align；缺失/null/畸形/不完整 E_align 判无效而非零误差；原三种比较数值不变；
JSON/CSV/绘图含新比较；`enhance` 不改源文件、且子进程运行时不加载
`mac_experiment.sampling`/`mac_experiment.execution`。

### 7.3 离线增强结果（正式 B run）

```
$ .venv/bin/python -m mac_experiment enhance \
    --run results/block_sweep_20260916_152916_0144683e
results_enhanced/block_sweep_20260916_152916_0144683e_alignment_only_20260917_091859
```

- 读取 1,540,000 条记录；154 个 cell 的 `alignment_only` 全部 `ok`；
- `mean_signed_error`/`rmse_abs_error` 与 `stage E_align` 的 mean/RMS 逐一相等（0 不一致）；
- 原三种比较 5,082 个数值与源 `summary.json` 完全一致（0 差异）；
- 源 run 四个文件 sha256 在增强前后不变；
- `analysis_manifest.json` 记录源 run 路径、源 schema `mac-exp-v1`、
  `samples.jsonl.gz` 等文件的 sha256 与大小、记录数、分析器版本 0.2.0；
- 输出 36 个图（PNG+SVG），每图 4 条曲线；人工检查 B 轴 log2、紫色三角曲线、
  图例与 Q_B 说明、零值与无效标记均正确。

图位置：`results_enhanced/block_sweep_20260916_152916_0144683e_alignment_only_20260917_091859/plots/`
（例：`b_sweep_fp8_e4m3_p99_abs_error.png`、`b_sweep_tf32_relative_l2.png`）。
按要求本轮只增强正式 B run；正式 W run 未处理（可用同一命令随时增强）。

## 8. FP32/BF16 位级匹配率与 ULP distance（2026-09-20）

### 8.1 离线可行性

现有 v1 逐样本记录已保存精确 `S`（`{"m": 十进制字符串, "e": 指数}`）、`r_bits`（FP32 bit
pattern）、`format`/`distribution`/`B`/`seed`/`sample_id`，字段齐全，**完全离线**，无须重跑
输入生成或 MAC 实验；原始数据只读，派生结果单独输出。无缺失字段、无最小补充方案需求。
`S` 为精确 dyadic，不含符号零信息，规范参考零为 +0（文档明确，不声称恢复原始符号）。

### 8.2 修改文件与命令

| 文件 | 修改 |
|---|---|
| `arithmetic/bf16.py`（新） | BF16 编码、直接 RNE 舍入（次正规、溢出到 Inf、±0） |
| `arithmetic/ulp.py`（新） | 有序索引与精确 ULP distance（±0 合并、仅有限） |
| `metrics/bitmatch.py`（新） | 主/辅助比较的流式累计器（bit match、条件 ULP、分桶、事件、by_seed） |
| `results/bitmatch_analysis.py`（新） | 离线入口：流式读取、独立输出目录、`analysis_manifest.json`、CSV |
| `plotting/bitmatch_plots.py`（新） | 每格式 × FP32/BF16 三张图（rate / 条件 ULP / 分桶） |
| `cli.py` | 新增 `bitmatch` 命令（handler 内延迟导入） |
| `version.py` | 包版本 0.3.0；`schema.py` 增加 `mac-bitmatch-v1` |
| 文档 | README、design.md 第 6 节、本节 |

```bash
.venv/bin/python -m mac_experiment bitmatch --run <run_dir> [--out DIR] [--no-plots]
```

### 8.3 测试

```
$ .venv/bin/python -m pytest -q
141 passed in 16.45s
```

- `tests/test_bf16.py`（16 例）：精确可表示值；ties-to-even 两侧；尾数/指数进位；
  最小次正规、下溢中点、次正规/正规边界；最大有限与溢出边界；±0、Inf、NaN；
  与独立 Fraction oracle 500 组随机交叉验证；**双重舍入反例**
  `S = 1 + 2^-8 + 2^-26`：直接 BF16 → `0x3F81`（1+2^-7），先 FP32 → `0x3F808000`
  再 BF16 → `0x3F80`（1），两者不同；FP32/BF16 ULP distance 基线与格式差异。
- `tests/test_bitmatch.py`（10 例）：全匹配/无不匹配样本（条件值为 null 而非 0）；
  ±0 bit 不匹配但距离 0（`signed_zero_mismatch`）；五类非有限事件分类；
  条件分位数（nearest-rank）与分桶计数；按 B/seed 分组；真实小 run 的独立重算一致；
  W sweep 重复参考不被池化（同一样本各 W cell 独立、r_B 跨 W 不变）；
  严格 JSON/CSV/绘图；源文件 sha256/mtime 不变；子进程验证 `bitmatch` 不加载
  `mac_experiment.sampling`/`mac_experiment.execution`。

### 8.4 正式 B sweep 离线结果

```
$ .venv/bin/python -m mac_experiment bitmatch \
    --run results/block_sweep_20260916_152916_0144683e
results_bitmatch/block_sweep_20260916_152916_0144683e_bitmatch_ulp_20260920_062524   (34.9 s)
```

- 读取 1,540,000 条快照记录（= samples 140,000 × B 11），源文件 sha256 前后不变；
- 全样本均为有限对，五类非有限事件计数全为 0；
- FP32（o32 vs r_B）：总体 bit match 921,449 / 1,540,000 = 59.834%；
  有限不匹配 618,551；分桶 0/1/2/3–4/≥5 = 921449/266421/107670/100110/144350；
  条件 ULP p50/p90/p99/max 随 B 增大（如 B=1024、uniform：6/27/235/74023）；
- BF16（o16 vs r16）：总体 bit match 1,539,914 / 1,540,000 = 99.994%；不匹配仅 86，
  距离 1 的 83 个、距离 2 的 1 个、≥5 的 2 个（B 大时 r_B 与 S_B 跨 BF16 舍入边界）；
- 辅助比较已保存于 summary（未绘图、未与主比较相加）；
- 统计均来自有限样本（`n_finite_pairs` = 全部记录，分母明确写入 CSV/图注）。

图位置：`results_bitmatch/block_sweep_20260916_152916_0144683e_bitmatch_ulp_20260920_062524/plots/`
共 24 个文件（12 图 × PNG+SVG），已人工检查：
`b_sweep_fp8_e4m3_fp32_bitmatch_rate.png`（B=1 零不匹配、随 B 上升）、
`b_sweep_tf32_fp32_conditional_ulp.png`（B=1 空心红圈 "no mismatch samples"，
其余 nearest-rank 分位数）、`b_sweep_fp8_e4m3_bf16_ulp_buckets.png`（分桶比例）。
W sweep 按要求本轮未做位级分析（可用同一命令离线运行）。

## 9. MXFP8 矩阵实验（2026-09-21，历史：v1 输入方案）

> 本节记录已退役的 v1 输入方案（直接独立采样 E4M3 元素编码与 E8M0 scale）的实现、
> 测试与结果。该方案已退出活动入口；历史 run 保持只读，`mx-analyze` 会明确拒绝它们。
> 当前实验见第 10 节。

### 9.1 实现与复用

在现有模块化框架上扩展，不复制独立数值实现：

| 文件 | 作用 |
|---|---|
| `formats/e4m3_element.py`（新） | OCP E4M3 元素编解码：次正规、符号零、RNE、饱和到 448（无 FTZ、无 Inf） |
| `formats/e8m0.py`（新） | E8M0 scale：码 255 保留、精确 `2^(c-127)`、指数截断辅助 |
| `matrix/tensor.py`（新） | `MxTensor`：元素码 + scale 码 + 逻辑形状/分组轴/组长，行主序存储 |
| `matrix/sampling.py`（新） | 元素策略 × scale 策略（复用现有 PMF 与分布定义；不含 cancellation） |
| `matrix/engine.py`（新） | 精确 `S`/`L1`、dot16 精确部分和 + 每次一次 FP32 RNE、paired-prefix 快照 |
| `matrix/quantize.py`（新） | 输出 MX 量化：精确选 scale、饱和、独立/共享 scale、事件计数 |
| `metrics/mx_quant.py`（新） | 组/元素/解码匹配、ULP 条件统计、scale 指数差、每组不匹配直方图 |
| `results/mx_analysis.py`、`results/mx_io.py`（新） | 离线量化分析、严格 JSON/CSV、源文件 sha256、独立输出目录 |
| `plotting/mx_plots.py`（新） | 按分布分面、按 scale 场景分线的 PNG+SVG |
| `experiments/config.py`、`experiments/mx_matrix_sweep.py` | 新 mode `mx_matrix_sweep` 与 driver；旧模式不变 |
| `arithmetic/ulp.py`、`results/schema.py`、`version.py`、`cli.py` | E4M3 ULP 索引、MX schema 常量、`run` 分派、`mx-analyze` |
| `arithmetic/bf16.py` | 修复 `Bf16.from_fp32_bits(-0)`（此前 `Fp32.exact_dyadic()` 丢失零的符号，返回 +0） |

复用点：`Dyadic` 精确算术、`round_dyadic_to_fp32` 整数 RNE、`Fp32Accumulator`、
`ReferencePath`（测试对照）、`ComparisonAccumulator`（`R−S`、`decode(Q_R)−S`）、
`linear_quantile`/`nearest_rank`、`JsonlSink`/严格 JSON、`source_digest`、绘图样式。
旧 dot-product 实验入口与数值语义未改动。

`Bf16.from_fp32_bits(-0)` 回归测试：`tests/test_bf16.py::test_from_fp32_bits_preserves_signed_zero`
（`0x80000000 → 0x8000`，`0x00000000 → 0x0000`）。未做无关重构。

### 9.1.1 审查修订（2026-09-21）：pair 随机流、representation 定义、文档口径

| 问题 | 原因 | 修改 |
|---|---|---|
| P1 不同矩阵对重复 | `mx_stream_rng` 的 key 不含 `pair_index`，`build_sample` 只把它写进 `sample_id`，同一 seed 的所有 pair 生成完全相同的 X/W | `mx_stream_rng` 增加 `pair_index` 并进入 key；`MxElementSampler.sample` / `MxScaleSampler.sample` 增加 `pair_index` 参数；`build_sample` 传入。元素流 label 仍只有分布（不含 scale 场景），激活/权重流仍分离。`MX_STREAM_VERSION` 1 → 2；策略版本 `mx-element-v2` / `mx-scale-v2`；run manifest 新增 `mx_stream_version` 字段；`mx-analyze` 的 `source`/summary 记录 `source_mx_stream_version`（旧 manifest 缺失即 v1，不标注为新版本数据） |
| P2 每组不匹配元素数只算 element code | `MxComparisonAccumulator.add_group` 的直方图只比较 code，scale 不同时也可记录 0 | 明确三种定义：`representation = (scale_code, element_code)`（主指标）、`code_only`（显式命名 `*_code_only_*`）、`decoded_value`。主直方图/整组 bit match 用 representation；scale 不同时该组 32/32 为 representation 不匹配；by-seed 的 `mismatch_elements` 拆成 `representation_mismatch_elements` / `code_only_mismatch_elements` / `decoded_mismatch_elements` / `ulp_mismatch_elements`；summary/analysis 版本升为 `mac-mx-summary-v2` / `mac-mx-analysis-v2`；CSV 与图同步改名，另出 code-only 诊断图；ULP 规则不变 |
| P3 样本量/结论/资源估计 | 输出元素数写错；用相关元素套 3/N 置信上界；正式估计缩放倍数错误 | 本节按实际文件核对数量（9.3），删除置信上界表述（9.5），修正正式模板注释与倍数 |


### 9.2 测试

```
$ .venv/bin/python -m pytest -q
205 passed in 16.6s
```

新增：`test_mx_formats.py`（E4M3/E8M0 编解码与边界、全部 254 个有限码往返、
Fraction oracle 1000 组随机交叉验证、饱和/次正规/±0/NaN、E4M3 ULP 索引）；
`test_mx_quantize.py`（scale 阈值两侧与恰好阈值、上下界截断、全零组、共享 scale
禁止重选、饱和/转零/保留码事件、axis=0 布局、不完整组报错）；
`test_mx_engine.py`（Fraction 精确 oracle、dot16 分别更新 vs 错误 dot32 合并的
E4M3 级反例、两个 dot16 共享组 scale、scale=1 与现有 `ReferencePath` 一致、
paired-prefix 与单独 K 运行一致、更新日志回放、NaN/分组校验；pair 回归：同
seed/pair 重复一致、不同 pair 元素与 scale 不同、K 前缀稳定、改变遍历顺序/增加 pair
不改变已有 pair、scale 场景不改变元素）；
`test_mx_analysis.py`（code 不同解码相同（±0）、scale 不同时主比较不计算 ULP、
空条件为 null、无效组 partial 策略与分母、端到端 run→analysis、离线重算一致、
严格 JSON、拒绝旧 schema、注入编码改变的检测回归；representation 定义：
code 相同 scale 不同 → code-only 0/representation 32、一个 code 不同 → representation 1、
直方图加权总数与元素 representation 不匹配总数一致、零桶等于整组 bit match、
by-seed 与汇总一致、ULP 条件计数不冒充总不匹配）；`test_mx_sampling.py`
（Kmax 增长的输入前缀稳定、pair_index 进入元素与 scale 流、unit 场景 pair 无关且恒为 1、
scale 场景与元素流互不影响、分布互不干扰、合成分布只产生合法有限码、
cancellation 明确拒绝；另有独立验证：
`test_saved_records_match_independent_fraction_oracle` 用测试内独立 E4M3/E8M0 位解码 +
Fraction dot16 递推，逐元素核对保存的 `S`、`L1`、`r_bits`）；`test_mx_config.py`
（MX 配置校验与旧配置回归）。

### 9.3 smoke 与 pilot（已执行）

```bash
.venv/bin/python -m mac_experiment run --config configs/mx_smoke.yaml   # 12 对 × 3 K，即时
.venv/bin/python -m mac_experiment mx-analyze --run results_mx_smoke/mx_smoke
.venv/bin/python -m mac_experiment run --config configs/mx_pilot.yaml   # 28.8 s
.venv/bin/python -m mac_experiment mx-analyze --run results_mx_pilot/mx_pilot   # 3.9 s
```

| 项 | smoke | pilot |
|---|---|---|
| 配置 | M=2,N=32,K=[32,64,128]，2 分布 × 3 场景 × 2 seed，pairs_per_seed=1 | M=4,N=64,K=[32,128,512,2048]，3 分布 × 3 场景 × 2 seed，pairs_per_seed=1 |
| 原始 run | `results_mx_smoke/mx_smoke`（124 KB） | `results_mx_pilot/mx_pilot`（3.4 MB） |
| 记录 | 12 输入对 + 36 快照 | 18 输入对 + 72 快照 |
| 输出元素快照 | 36 × 2 × 32 = 2,304 | 72 × 4 × 64 = **18,432** |
| 输出组快照 | 72 | 576 |
| 分析 cell | 18 | 36 |
| 分析输出（v2 语义） | `results_mx_analysis/mx_smoke_quant_20260921_074436/` | `results_mx_analysis/mx_pilot_quant_20260921_074441/` |

原始 run 由 stream v1 生成（manifest 无 `mx_stream_version`；两份配置 `pairs_per_seed=1`，
因此不受“同 seed 不同 pair 重复输入”问题影响，无需重新生成）。本轮已用新分析器
（`mac-mx-summary-v2` / `mac-mx-analysis-v2`，representation 定义）对它们重新分析，
输出到上表的新目录；早前用 v1 直方图语义生成的
`mx_smoke_quant_20260921_074436/`、`mx_pilot_quant_20260921_074441/` 保留但已被取代。

pilot 主要观测（全部 cell `status=ok`，无无效组、无非有限事件）：

- 独立 scale 下 scale 不匹配率、组 representation `(scale_code, element_code)` 不匹配率、
  元素 representation 不匹配率、解码不匹配率在 **全部 36 个 cell 均为 0**
  （72 快照 / 18,432 元素快照 / 576 组快照；这些计数来自实际文件核对，不是独立样本数）；
  共享 scale 条件 ULP 也无不匹配样本（`no_mismatch_samples`，值为 `null`）。
- 原因与量级一致：`R−S` 累加误差相对 L2 约 `2e-8 .. 4.4e-7`（p99 绝对误差可达
  1e2–1e6，取决于输出幅度），远小于 E4M3 输出的 4 位尾数网格（相对约 2⁻⁴），
  因此 FP32 累加差异几乎不改变输出编码。
- 输出量化本身的总误差 `decode(Q_R) − S` 相对 L2 约 **2.2% .. 2.9%**
  （与 scale 场景、K 弱相关），这是本实验可稳定测量的主效应。
- 图：`results_mx_analysis/mx_pilot_quant_20260921_074441/plots/`（PNG+SVG）；
  已人工检查 `mx_rate_scale_mismatch_rate.png`、
  `mx_rate_group_representation_mismatch_rate.png`、`mx_decoded_decoded_relative_l2.png`、
  `mx_total_error_relative_l2.png`（2.2%–2.9%）、`mx_scale_exponent_diff_positive_rate.png`、
  `mx_group_representation_mismatch_histogram.png`（K=2048，全部落在 0，
  明确标注 `(scale_code, element_code)`）及显式命名的
  `mx_group_code_only_mismatch_histogram.png` 等；零值与 `null` 分别用
  实心点与空心黑圈表示，事件点用红 ×，未混入主曲线。

### 9.4 失配检测能力的验证（合成注入）

pilot 的 0 失配不能区分“真无失配”与“分析器不工作”。已固化为回归测试
`tests/test_mx_analysis.py::test_analysis_detects_injected_code_change`：把某个 `K=32`
快照元素 0 的 `r_bits` 指数整体 +3（并保留其余记录），分析后主比较出现元素
representation 不匹配、共享 scale 辅助比较出现 ULP 不匹配（`n_mismatch > 0`，p50 非 null）；
未改动的 K cell 与干净 run 完全一致。另做过的 1 ULP 注入不改变任何指标（符合预期：
1 ULP 远小于输出网格），两份副本均在临时目录，未进入交付结果。测试同族还覆盖
±0 code 不同但解码相同、scale 不同时主比较不计算 ULP、空条件为 null 等。

### 9.5 独立单元、事件计数与正式规模估计

**不提供置信区间。** 独立重复单位是**输入矩阵对**；同一矩阵对的输出元素共享输入与
scale，不同 K 又是同一最大 K 输入的前缀（同一样本重复出现），因此元素/快照计数高度
相关，不能当作独立 Bernoulli 样本，也不能用 `3/N` 之类的零事件规则给出某个元素失配率
的置信上界。若将来要给区间，必须先定义估计对象（例如“每个矩阵对至少发生一次失配的
概率”与“元素失配率”是两个不同目标）并按其独立单位抽样。

在临时目录执行（未交付）M=8,N=64,K=[4096,16384]、3 分布 × `rare_large`、2 对的探针：
6 个矩阵对 / 12 个快照 / 6144 个元素快照（6 个 cell，每 cell 1024），137 s，3.3 MB。
**描述性结果**：scale、representation、解码不匹配事件数均为 0；`R−S` 相对 L2 约
`3e-8 .. 4.4e-7`。该 0 事件只是有限、相关样本上的观察，不外推为任何概率上界。

正式模板（`configs/mx_formal_template.yaml`，**未执行**）实际为 10 个 K 点
（K=32..16384 倍增）、每 (分布, 场景) 组合 5 seed × 2 pair = **10 对**，共 18 个组合、
**180 对**。资源区分实测与外推：

| 量 | 值 | 来源 |
|---|---|---|
| pilot（M=4,N=64,Kmax=2048，18 对） | 28.8 s，3.4 MB | 实测 |
| 探针（M=8,N=64,Kmax=16384，6 对） | 137 s，3.3 MB | 实测 |
| 正式每对（M=16,N=64,Kmax=16384） | 约 40–45 s，约 1 MB gz | 由探针按 M·N·Kmax 线性外推 |
| 正式 180 对 | 约 2–2.5 h，约 180 MB；分析数分钟、内存数百 MB | 外推 |
| 若改为 500 对/组合（9000 对） | 约 100–125 h，约 9 GB | 外推 = 180 对的 **50 倍**（500/10），不是 10 倍 |

该模板只用于让解码误差/相对 L2 估计达到稳定（pilot 已非零且量级明确）；
它**不承诺**能解析任何极低失配概率——上面 0 事件样本不能支持这种解析度。
是否执行正式实验按本轮交付不自动启动。

### 9.6 修订后的 pair 随机流与 representation 定义验证

小验证 run（`configs/mx_pair_probe.yaml`，M=2,N=32,K=[32,64]、uniform、
unit+narrow、1 seed、**pairs_per_seed=3**）：`results_mx_pair_probe/mx_pair_probe`（40 KB，
6 对 / 12 快照），分析输出 `results_mx_analysis/mx_pair_probe_quant_20260921_071202/`。
从保存的输入文件核对（sha256 前缀）：

```
unit  pair0 x=5fdcb78482f w=cd12868c222 | x_scale=365475d w_scale=e020551
unit  pair1 x=8a0ee66d224 w=b823160df79 | x_scale=365475d w_scale=e020551
unit  pair2 x=ec7125a5c3d w=c9891e9af72 | x_scale=365475d w_scale=e020551
narrow pair0 x=5fdcb78482f w=cd12868c222 | x_scale=dfd881f w_scale=48db62d
narrow pair1 x=8a0ee66d224 w=b823160df79 | x_scale=4ed7024 w_scale=4080970
narrow pair2 x=ec7125a5c3d w=c9891e9af72 | x_scale=d1e41a9 w_scale=b653323
```

即：不同 `pair_index` 的 X/W 元素指纹不同；`unit` 的 scale 与 pair 无关（恒 1.0），
`narrow` 的 scale 随 pair 改变；`unit` 与 `narrow` 在相同 pair 上元素指纹相同
（scale 场景不改变元素编码）。

scale-only 变化的 representation 证据（在真实保存的 group 上把 R 侧 scale 提高一个
指数、保留相同 element codes）：

```
scale mismatch groups: 2 / 2
code-only match: 64 / 64
representation match: 0 / 64
decoded match: 0 / 64
representation histogram nonzero: {'32': 2}
code-only histogram nonzero: {'0': 2}
group bit match: 0  group code-only match: 2
ULP condition elements: 0 no_mismatch_samples
```

即“code 相同、scale 不同”时 code-only 不匹配为 0，representation 为 32/32，
且主 ULP 仍只在共同 scale 网格上计算（条件元素 0）。该证据同时记录在回归测试
`test_same_codes_different_scale_is_representation_mismatch_only` 中。

历史数据影响范围：本轮之前的 MX run 均由 stream v1 生成；其中 `pairs_per_seed=1`
的 smoke/pilot **只需重新分析**（已输出到 9.3 的新目录）；若存在 `pairs_per_seed>1`
的旧 run，其 pair 数据重复，**必须重新生成**才能用于 pair 级结论（本交付中没有此类
已执行 run；`configs/mx_formal_template.yaml` 的 `pairs_per_seed=2` 从未执行）。
旧分析目录不删除、不覆盖，但语义为 v1，当前结论以 v2 目录为准。

固定 `run_name` 的冲突处理：`create_run_dir` 在目录已存在时直接抛 `FileExistsError`
（`error: run directory already exists: ...`），不会覆盖；重复执行同一配置需要改
`run_name`/`output_root` 或先移走旧目录。`mx-analyze` 同样拒绝已存在的输出目录，
因此每次离线分析都会产生新的时间戳派生目录。

### 9.7 当前局限

- 无有限 W 窗口（接口保留后续接入 reducer 的位置，本轮不实现）；矩阵 cancellation
  分布未定义；合成输入不代表真实 LLM 分布；
- 输出量化策略固定为“覆盖组最大幅度的最小二进制幂”，不可切换到其他 MX 量化算法；
- 单线程、分析内存随元素数增长（无外存分位数）；
- 正式模板未执行；探针与合成注入均在临时目录；本轮不提供任何置信区间或失配率上界，
  相关计数是有限、相关样本上的描述性观察；
- 历史 v1 run 保持只读，其二进制文件不做迁移；pair 级结论只使用 stream v2 生成的数据。

## 10. MXFP8 矩阵实验：当前输入方案（2026-09-21）

当前流水线：数值域采样 → FP32 输入矩阵（bits 落盘）→ 按 G32 量化为 MXFP8 →
精确参考与 dot16＋FP32 累加 → MXFP8 输出量化比较。结论口径：测量给定 FP32 数据分布
经 MXFP8 输入量化后，这些 MX 输入在 dot16＋FP32 累加与输出量化中的差异；**不测量**
输入量化相对原始 FP32 矩阵乘法的精度损失。

### 10.1 修改文件与新输入流程

| 文件 | 作用 |
|---|---|
| `matrix/inputs.py`（新） | 四种数值域分布（理论均值 0、RMS 1）、显式 RNE 转 FP32 bits、FP32→G32 MXFP8 输入量化、X/W 分路径事件与输入诊断 |
| `sampling/streams.py` | 新增 `mx_input_stream_rng`（`mac-mx-fp32-stream-v1`） |
| `matrix/engine.py` | `build_sample` 改为 FP32→MX；输入记录含 FP32 bits、分布元数据、输入事件/诊断 |
| `experiments/config.py` | 新配置 `distribution_pairs`（activation/weight 分别配置）；旧键明确报错 |
| `experiments/mx_matrix_sweep.py` | manifest v2：`input_sampling_version` / `input_stream_version` / `input_quantization_policy`；快照带 activation/weight 分组 |
| `results/mx_analysis.py` | 按 activation×weight×K 分组；输入事件/诊断汇总；默认输出根 `results_mx_analysis_v2`；拒绝 v1 run |
| `plotting/mx_plots.py` | 按激活分布分面、权重分布分线；无 scale 场景；新增输入诊断图（RMS、scale 指数直方图） |
| `matrix/sampling.py` | v1 元素/scale 采样模块保留但退役：引擎/配置/分析不再导入，`matrix/__init__` 不再导出 |

精确算术、MX 量化器与矩阵引擎（`evaluate_mx_pair` 的 dot16/FP32/无窗口语义）未重写，
输入量化复用 `quantize_matrix_independent`。

### 10.2 随机流与 schema/策略版本

| 项 | 旧（v1） | 当前 |
|---|---|---|
| schema | `mac-mx-exp-v1` | `mac-mx-exp-v2` |
| 输入 | 元素编码 + E8M0 scale 独立采样 | 数值域 FP32 → G32 MX 量化 |
| 输入流 | `mac-mx-stream-v2`（元素/scale 两套） | `mac-mx-fp32-stream-v1`（`|seed|matrix|label|pair_index|row|value`） |
| 采样/量化版本 | `mx-element-v2` / `mx-scale-v2` | `MX_INPUT_SAMPLING_VERSION = mx-fp32-input-v1`，`MX_INPUT_STREAM_VERSION = 1`，`MX_INPUT_QUANTIZATION_POLICY = min_binary_power_covering_group_max_from_fp32_bits` |
| 分析 | `mac-mx-summary-v2` / `mac-mx-analysis-v2` | `mac-mx-summary-v3` / `mac-mx-analysis-v3` |
| 默认输出根 | `results_mx_analysis` | `results_mx_analysis_v2` |

旧随机流产生的数据绝不标注为新策略：旧 manifest 无 `input_sampling_version`，
`mx-analyze` 报 “retired v1 MX input scheme”；旧分析目录保留但默认排除。

### 10.3 旧方案退出活动入口的验证证据

- `tests/test_mx_config.py::test_mx_config_rejects_legacy_input_scheme_explicitly`：
  含 `scale_scenarios` 或旧 `distributions` 的配置抛 `ValueError`（match “retired”），
  不被静默转换；
- `tests/test_mx_config.py::test_legacy_element_scale_samplers_are_not_called`：
  monkeypatch 旧 `MxElementSampler.__init__` / `MxScaleSampler.__init__` /
  `mx_stream_rng` 为抛错函数后，完整跑一个小 run 仍成功 → 活动路径确实不调用旧入口；
- `tests/test_mx_analysis.py::test_analysis_rejects_legacy_v1_input_runs`：旧 schema 与
  缺 `input_sampling_version` 的 v2-schema run 均被明确拒绝；
- 活动配置目录不再有 `scale_scenarios` / `unit` / `narrow` / `rare_large`
  （`configs/mx_pair_probe.yaml` 已删除；smoke/pilot/formal 全部为 `distribution_pairs`）；
- 自定义分布参数被明确拒绝，而不是记录后不生效：
  `test_mx_config_rejects_custom_distribution_parameters`（配置层）与
  `test_custom_distribution_parameters_are_rejected`（采样器层）断言
  `activation_parameters` / `weight_parameters` 触发 `ValueError`，采样器元数据
  标记 `parameters_configurable=False`，固定使用四种单位 RMS 分布。

### 10.4 测试

```
$ .venv/bin/python -m pytest -q
209 passed in 18.4s
```

新增/改写：`test_mx_inputs.py`（自定义参数拒绝、显式 FP32 RNE 与边界、非有限报错、四种分布参数与
理论均值/RMS、宽容差经验统计、同 pair 复现/不同 pair 不重复、K 前缀稳定、
pair 数量/遍历顺序无关、激活/权重流独立、X/W 轴与 G32/scale 索引、输入量化前缀、
从保存 bits 重量化等于保存编码、全零组规范 scale）；`test_mx_engine.py` 增加
“精确 S 来自 MX 解码值而非原始 FP32”的构造样本（1.1→1.125，S=36 而非 35.2）、
FP32/MX/快照三级前缀一致；`test_mx_analysis.py` 的端到端 run→analysis、离线重算、
严格 JSON、源文件只读（sha256 前后一致）、v1 拒绝；`test_mx_config.py` 新键与旧键报错。
原有 dot-product 测试全部保留通过。

### 10.5 不同 pair、前缀与重量化证据（实测）

`results_mx/mx_smoke`（4 个同分布组合 × 1 seed × pairs_per_seed=2，K=[32,64]）保存的
输入：同一 seed、同分布组合的 pair 0/1 的 FP32 bits 指纹不同（sha256 前 16 位）：

```
gaussian pair 0  x_bits_fp=fc824a0a9e1139fe  w_bits_fp=71c751b378194f8b
gaussian pair 1  x_bits_fp=e6b4905c7e5af8ab  w_bits_fp=2fae7f30fcb65ed8
requantized equals saved MX codes: True True True   (X/W element + X scale fingerprints)
```

前缀稳定性（独立引擎实例，同 seed/pair）：短 K 的 FP32 bits、MX 元素编码、scale 与
长 K 前缀逐行一致，且 K=32 快照（精确 S/L1/r_bits）与长 K 扫描的第一个快照完全相同
（测试 `test_build_sample_pairs_reproducible_distinct_and_fp32_prefix_stable`）：

```
prefix stability: x_bits True  w_bits True  x_mx True  K=32 snapshot equal: True
```

### 10.6 smoke 与计时 pilot（已执行）

```bash
.venv/bin/python -m mac_experiment run --config configs/mx_smoke.yaml        # 8 对，即时
.venv/bin/python -m mac_experiment mx-analyze --run results_mx/mx_smoke
.venv/bin/python -m mac_experiment run --config configs/mx_pilot.yaml        # 24.5 s
.venv/bin/python -m mac_experiment mx-analyze --run results_mx/mx_pilot      # 0.6 s
```

| 项 | smoke | pilot |
|---|---|---|
| 配置 | M=2,N=32,K=[32,64]，4 组合 × pairs_per_seed=2 | M=4,N=64,K=[32,128,512,2048]，4 组合 × 2 seed |
| 原始 run | `results_mx/mx_smoke`（168 KB） | `results_mx/mx_pilot`（6.7 MB，24.5 s） |
| 快照 / 元素快照 | 16 / 1,024 | 32 / 8,192 |
| 分析输出 | `results_mx_analysis_v2/mx_smoke_quant_20260921_074436/` | `results_mx_analysis_v2/mx_pilot_quant_20260921_074441/` |

pilot 观测（全部 cell `status=ok`，无无效组、无非有限事件）：

- 独立 scale 的 scale/组 representation/元素 representation/解码不匹配率在全部 16 个
  cell 均为 0；共享 scale 条件 ULP 无不匹配样本（`null`，`no_mismatch_samples`）。
- `R−S` 累加误差相对 L2 约 `2e-9 .. 8e-8`；`decode(Q_R)−S` 总误差相对 L2 约
  **2.1% .. 2.7%**（输出量化主导）。
- 输入诊断（实际产生，非误差指标）：FP32 RMS 0.99–1.00、解码 MX RMS 0.986–1.001、
  MX 零值比例 0、FP32 最大幅度 5.1（gaussian）/8.6（laplace）/22（outlier）/
  75.6（student_t3，重尾）。
- 图表 `results_mx_analysis_v2/mx_pilot_quant_20260921_074441/plots/` 已人工检查：
  主直方图 `mx_group_representation_mismatch_histogram.png` 用
  `(scale_code, element_code)` 定义、按 activation 分面/weight 分线；另有显式
  `mx_group_code_only_mismatch_histogram.png`；输入诊断图为
  `mx_input_rms_diagnostic.png` 与 `mx_input_scale_exponent_histogram.png`
  （标注为诊断而非误差指标）。

### 10.7 正式模板规模与资源估计

单对实测（临时目录，未交付）：M=16、N=64、K=[16384]（1 快照）55.7 s、8.4 MB，
其中 `inputs.jsonl.gz` 7.7 MB（FP32 bits 为主体）。正式模板为 10 个 K 点、
4 个同分布组合 × 5 seed × 2 对 = 每组合 10 对、共 **40 对**：

| 量 | 估计 | 依据 |
|---|---|---|
| 运行时间 | 约 40 min（约 60 s/对） | 实测 55.7 s/对 + 额外快照写出 |
| 磁盘 | 约 350 MB（约 8.5 MB/对） | 实测 8.4 MB/对，FP32 bits 为主体 |
| 离线分析 | 约 1–2 min；内存数百 MB | pilot 分析 0.6 s / 8,192 元素快照，按 40 对 × 10 K × 1024 元素快照外推 |
| 500 对/组合（2,000 对） | 约 33 h、约 17 GB | 外推 = 40 对的 50 倍 |

该模板用于稳定的解码误差/相对 L2 估计与描述性失配计数；0 事件样本不承诺解析极低
失配概率，也不提供基于元素独立性的置信上界（独立单位是矩阵对）。

### 10.8 当前局限

- 单线程；输入 FP32 bits 落盘使磁盘成为正式规模的主要成本（约 8.5 MB/对）。
- 不加入额外随机通道尺度、block 尺度或真实模型数据；矩阵 cancellation 分布未定义。
- 输出量化策略固定；分析在内存中保留逐元素误差值。
- 不提供置信区间/失配率上界；正式模板未执行。
- 历史 v1 run 只读且默认排除；本轮不将新旧结果直接比较。

## 11. MXFP8 固定对阶精度 K/B 扫描（W_align=21，2026-09-22）

### 11.1 实现与文件

| 文件 | 作用 |
|---|---|
| `matrix/alignment.py`（新） | E4M3 幅值分解 `(sign,t,E)`（次正规、NaN 拒绝）、逐项 RNE 对阶、16 项整数和、`q_b = N_b·2^(Cmax-14-W_align+h)`、28 位容器范围校验、全零/相消区分、逐项 trace |
| `matrix/aligned_engine.py`（新） | S/L1/R 复用现有无窗口定义；新增 T（精确累加 q_b）与 A（q_b 的 FP32 累加）、逐元素对齐误差、事件/直方图；`trace_dot16` 指定 block 导出 |
| `experiments/config.py`、`experiments/mx_alignment_sweep.py`（新） | `mx_alignment_sweep` 模式；`source_run` 复用（schema/身份/维度/分组/指纹/唯一性校验 + 源 S/L1/R 逐元素比对 + paired-prefix 切片 + 源文件 sha256）；fresh 模式生成并保存完整输入 |
| `results/mx_alignment_analysis.py`（新） | 离线比较 T-S/R-S/A-T/A-S/A-R、三条 MX 输出比较（独立+共享）、总误差、representation 转变计数、对阶诊断聚合、按 pair 摘要、输入诊断（pooled RMS、group-weighted scale 直方图） |
| `plotting/mx_alignment_plots.py`（新） | 误差/对阶事件率/转变/总误差图；无空占位面板（无该 activation×weight 组合时不画空线） |
| `cli.py`、`results/schema.py`、`version.py` | `mx-align-analyze`、`mx-align-trace`；`mac-mx-align-v1` / `mac-mx-align-analysis-v1`；包 0.7.0 |

无窗口 `evaluate_mx_pair`、输入流水线、输出量化器、representation/ULP 定义均未改动。
T 不作为默认输出量化路径。

### 11.2 测试

```
$ .venv/bin/python -m pytest -q
244 passed
```

口径备注：旧正式 block run 的固定窗口是 `W = 24`（dot-product 实验量），本轮 MXFP8
对阶量是 `W_align = 21`，两者不是同一个量；旧 run 未被重新解释。

新增 `tests/test_mx_alignment.py`（14 例）：幅值解码/次正规/±0/NaN 拒绝；有符号 RNE
含 exact tie ties-to-even；`Cmax-C_i` 15/16 边界（shift>0 但精确 vs 实际改变）；
单项对阶为零；逐项误差相消使 `q_b == s_b`；全零 dot16 与非零相消区别；28 位容器边界
与越界 `ArithmeticError`；元素域规约与精确缩放域 oracle 严格等价（200 组随机）；
大 W_align 时 `T==S、A==R` 且与无窗口基线逐位一致；MX 次正规不继承旧 FTZ；
同一 G32 两个 dot16 各自选 Cmax；q_b 不预舍入 FP32 的次正规双舍入反例
（正确 A bits=0x2，错误预舍入=0x3）；independent Fraction oracle 对齐/递推/
误差分解/`Σdelta = A-T`；paired-prefix 与单独短 K 一致；W=21 确定性对阶损失。
`tests/test_mx_alignment_analysis.py`（8 例）：转变计数（组/元素）与 ±0；非有限时
T-S 保留而候选/基线各自判无效；复用源端到端 + 源文件 sha256 前后不变 + schema 隔离；
篡改源快照被 "S mismatch" 拒绝；K 子集切片与 cumulative 事件不跨 K 重复计数；
scale-only 组的 representation 32/32、条件 ULP 排除 scale 不同组、`±0` 距离 0；
配置拒绝（W_align 边界、旧键、自定义参数、源身份不匹配）。

### 11.3 smoke 结果（已执行，W_align=21）

```bash
.venv/bin/python -m mac_experiment run --config configs/mx_align_smoke.yaml
.venv/bin/python -m mac_experiment mx-align-analyze --run results_mx_align/mx_align_smoke
```

- run：`results_mx_align/mx_align_smoke`（8 对 × 2 K，148 KB）；
- 分析：`results_mx_align_analysis/mx_align_smoke_fix_analysis_20260922_061110/`（8 cell；修复前旧目录 `mx_align_smoke_analysis_20260922_024014/` 仅作历史）；
- 对阶损失**确实触发**（不是功能未生效）：49,152 个非零乘积中 21 个对阶后整数改变、
  21/3,072 个 dot16 block `q_b != s_b`、7/8 个 cell 至少一个输出元素 `T != S`；
  最坏 `T-S` 相对 L2 `3.4e-8`，`A-T` 相对 L2 `1.0e-8`，`A-R` 相对 L2 `3.3e-8`；
- 本轮 smoke 的 MX 输出转变全部为 `both_match`（32/32 组）：对阶误差远小于 E4M3
  4 位输出网格，**“没有最终 MX 失配”不代表对阶未生效**；
- 图表已人工检查：`mx_align_error_align_T_minus_S_relative_l2.png`、
  `mx_align_rate_qb_diff.png`、`mx_align_rate_products_changed.png` 等按 activation
  分面、weight 分线，仅绘制实际存在的组合；诊断事件注明 cumulative。

### 11.4 复用正式输入的短回放实测（少量指定样本）

**交付的 40 对 K≤64 前缀回放**（`configs/mx_align_prefix_replay.yaml`，只读复用
`results_mx/mx_formal`，未运行完整 10-K 正式实验）：运行 27 s，输出
`results_mx_align/mx_align_prefix_replay`（2.2 MB，40 对 × 2 K = 80 快照），
分析 `results_mx_align_analysis/mx_align_prefix_replay_analysis_20260922_061121/`（8 cell）。
全部 40 对的源 `S/L1/R` 校验通过；对阶损失：3,932,080 个非零乘积中 933 个改变、
7 个对阶为零，930/245,760 个 dot16 block `q_b != s_b`；最坏 `T-S` 相对 L2 `3.3e-8`，
`A-T` 相对 L2 `7.4e-9`；候选与基线 FP32 结果的精确一致率约 99.4%（即对阶已在
FP32 层面改变约 0.6% 输出元素），但 K≤64 时 MX 输出 representation 失配仍为 0
（输出网格吸收了对阶误差）。

对 `results_mx/mx_formal` 的 `mx|gaussian|gaussian|seed1|00000`（M=16、N=64、
K=16384）做一次无窗口基线与一次 W_align=21 候选回放：

```
基线 evaluate 39.5 s（源 S/L1/R 与源快照逐元素一致：True/True/True）
对阶 evaluate 60.1 s
products_total 16,777,216；products_changed 2,563；products_zeroed 37
dot16_total 1,048,576；dot16_qb_diff 2,559；dot16_all_zero 0
T-S 非零输出元素 892/1024；A 与 R 的 FP32 bits 不同元素 67/1024
```

即真实正式输入下对阶误差显著存在，并已在 FP32 结果层面改变 67/1024 个输出元素。

### 11.5 正式规模与资源估计

对阶正式模板 `configs/mx_align_formal.yaml`（**未执行**）：复用 `results_mx/mx_formal`
的 40 对 × 10 K = 400 快照（409,600 元素、12,800 组）：

| 量 | 估计 | 依据 |
|---|---|---|
| 运行时间 | 约 1.1–1.2 h | 实测单对基线 39.5 s + 对阶 60.1 s |
| 输出磁盘 | 数十 MB | 对齐快照含 S/L1/T + 2 个诊断数组（高于基线 run 的 7 MB） |
| 离线分析 | 数分钟，内存数百 MB | pilot 分析亚秒级，按 400 快照 × 1024 元素外推 |

源输入（310 MB gz）只读复用，不重新采样；不提供基于元素独立性的零事件置信上界。

### 11.7 审查修复（2026-09-22）

| 问题 | 修复位置 | 行为变化 |
|---|---|---|
| CSV 目标路径被循环变量遮蔽 | `results/mx_alignment_analysis.py::write_alignment_csv` | 参数改为 `output_path`，循环变量 `quant_path`；始终写调用者指定路径，cwd 不再产生 `Q_S_shared` 等文件 |
| E8M0 scale=1（0x7F）被当成 E4M3 NaN | `AlignedAnalysisBuilder.add_input` | 元素用 E4M3 NaN 规则（0x7F/0xFF），scale 用 E8M0 规则（0..254 合法、0xFF 保留）；畸形数组抛明确 `ValueError`，不产生 TypeError |
| 无效输入被排除后仍报整批有效 | `AlignedCell._count_invalid_pair` | 无效 pair 逐比较累计 invalid 元素/组、transition valid/invalid/total；部分无效时整批指标 null + `finite_subset` |
| 未验证预期快照完整性 | `_expected_identity` + `analyze_mx_alignment_run` | 由 manifest 身份推导预期 `(sample_id,K)` 集合；缺失/多余/重复/未知、字段不符、数组长度错误全部报错；完整性检查先于创建输出目录 |
| 基线校验没覆盖实际保存结果 | `mx_alignment_sweep.py::_compare_saved_snapshot` | 直接校验 `evaluate_mx_pair_aligned` 返回值即将落盘的 S/L1/r_bits；错误带 sample_id/K/元素/字段；不再依赖额外旧引擎重算，也不依赖 A−S 恒等式 |
| 源快照重复被静默覆盖 | `mx_alignment_sweep.py` | 重复 `(sample_id,K)`（同内容或冲突）拒绝；源 manifest 必须 `completed` 且计数与身份集一致；处理完后检查输入/快照迭代器无剩余；合法短前缀仍通过 |
| 分析器源码指纹缺失 | `results/schema.py::source_digest` + analyzer | 新增 `analyzer_source_digest`（analysis_manifest 另含 `numeric_run_source_digest`）；离线分析不导入执行引擎/采样器（`matrix/__init__` 惰性导入） |
| 逐 pair 摘要不足 | `AlignedCell`/`PairSummary` + `merge_from` | 每个 cell 的 `by_pair` 保留数值比较指标/覆盖/`sum_err2`/`sum_ref2`、MX 计数与 ULP 摘要、transition 分母、累计事件；cell 指标由 exact 平方和合并 |
| reuse run 诊断入口 | `cli.py::_cmd_mx_align_trace` | 支持 alignment run 入口（reuse 自动读记录的源目录）、`--source-run` 覆盖（按记录的 sha256/size 校验）、按 run 的 `w_align` 与 K 前缀限制；显式覆盖标注 `w_align_override` |
| `StageAccumulator.merge_from` 字段错误 | `metrics/compare.py` | 原方法误用 `ComparisonAccumulator` 字段（调用即 `AttributeError`）；改为按自身 `total`/`total_sq`/计数精确合并，并加合并等价测试（该方法此前未被使用） |
| 源快照只检查本次请求的 K | `mx_alignment_sweep.py` | 每对矩阵 gather 后校验实际源 K 集合等于源 manifest 声明的完整 `K_list`；缺失或未声明的 K 都报错并指出 `sample_id`/missing/extra；K 子集请求不再掩盖源 run 残缺 |

回归测试（`tests/test_mx_alignment_analysis.py` 新增 10 例、`tests/test_metrics.py` 新增 1 例）：CSV 输出路径与 cwd 干净、
E8M0/E4M3 校验、无效 pair 覆盖、9 种快照完整性破坏（缺失/重复/多余 K/未知 pair/错分布/
错 W_align/数组长度）均失败且不留输出目录、故障注入新引擎 S/L1/r_bits 必须失败且
不得 completed、重复/冲突/多余源记录拒绝、**源每对 K 集合残缺（删全部或单对 K=64）
在只请求 K=32 时也必须拒绝、未声明 K 视为 extra**、分析器 digest 与引擎隔离（子进程
检查 `sys.modules`）、逐 pair 合并与 cell 一致、`StageAccumulator` 合并等价、
trace reuse 入口/源覆盖/身份拒绝/W 覆盖标注/K 前缀越界。
原有 W=21 触发对阶损失的确定性测试保留。

`Q_S_shared` 身份确认：仓库根目录的 `Q_S_shared`（27,883 B，CSV）经逐行多重集比对，
与旧 smoke 分析中 `gaussian/gaussian/K=32` 单 cell 生成的 CSV 完全一致，确认是修复前
变量遮蔽 bug 写错位置的产物；按“不删历史产物”原则原样保留，修复后的分析只写指定
输出目录（已在新生成的两个分析目录中验证，cwd 无新增文件）。

修复后重新生成的分析（独立新目录，源 run 未改动）：

- `results_mx_align_analysis/mx_align_smoke_fix_analysis_20260922_061110/`
  （新运行 `results_mx_align/mx_align_smoke_fix`，run → analyze 全流程）；
- `results_mx_align_analysis/mx_align_prefix_replay_analysis_20260922_061121/`
  （复用 `results_mx/mx_formal` 的短前缀 run 重新分析）。

两者的 `summary.csv`/`analysis_manifest.json` 齐备，`analyzer_source_digest` 与
`numeric_run_source_digest` 分别为分析与数值运行源码；`results_mx/mx_formal` 的
sha256/size 在核验前后不变。此前的 `*_analysis_20260922_0240xx` 目录是修复前产物
（无 `summary.csv`），仅作历史保留。

### 11.6 局限

- 本轮不运行正式复用实验；W_align 固定 21（测试覆盖其他合法 W，未做 W 扫描）；
- 未加入额外随机尺度、有限加法树或新的 FP32 窗口模型；
- 对阶诊断默认只保存聚合量（逐项明细用 `mx-align-trace` 按需导出）；
- 历史 run/分析保持只读，不被覆盖或重新标注；修复前生成的旧分析目录不视为修复后
  产物，当前结论以 2026-09-22 修复后新目录为准。
