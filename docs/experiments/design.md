# 设计说明

> 2026-10-31 目录迁移说明：本文保留原实验语义与历史记录。代码和配置现位于 `experiments/`；当前运行方式及缺失的历史数据见[实验指南](README.md)。历史实验结论不直接等同于旧基线 Q2.34 RTL 的精度证据，也不构成新架构的精度或实现证据；新架构约束见[已确认约束](../project/decisions.md)。

## 1. 数值语义

### 1.1 输入格式

FP8 E4M3：1 符号位、4 指数位（bias 7）、3 尾数位；正常有效数整数 `8+frac`。
`raw_exp == 0` 解码为零（输入 FTZ）；`exp=15, frac=7` 为 NaN；其余 `exp=15` 编码按有限值处理
（最大 448）。

TF32：1 符号位、8 指数位（bias 127）、10 尾数位；正常有效数整数 `1024+frac`。
内部使用 19 位紧凑编码（bit18 符号，bit17..10 指数，bit9..0 尾数），**不是** FP32 bit pattern。
`raw_exp == 0` 为零；`raw_exp == 255` 为特殊值并排除。

两者输入的 FTZ 规则与输出的渐进下溢在 manifest 的 `policies` 中分别记录。

### 1.2 精确乘积与窗口

乘积不做舍入或规格化：`signed_sig_a * signed_sig_b * 2**(C - product_lsb_offset)`，
其中 `C = raw_exp_a + raw_exp_b`。

| 格式 | product_lsb | 窗口 LSB 指数 |
|---|---|---|
| FP8 E4M3 | C − 20 | Cmax − 14 − W |
| TF32 | C − 274 | Cmax − 254 − W |

`Cmax` 为当前 block 非零乘积的最大 C；全部为零时走显式零路径（`window_lsb=None`）。
16 项对齐后使用 W+7 位有符号容器（`reduction.container_bounds`），越界抛
`WindowOverflowError` 并标记该单元格，不 wrap、不饱和。范围证明与测试见
`tests/legacy/test_reducer.py::test_window_container_range_proof`。

### 1.3 单次 FP32 更新

精确量统一为 `Dyadic(integer, exponent)`（值为 `integer * 2**exponent`，规范形式）。

```
s_b = exact_sum(exact_products_b)
q_b = exact_sum(aligned_and_rounded_products_b)
a_b = RN32(exact_value(a_(b-1)) + q_b)      # 候选
r_b = RN32(exact_value(r_(b-1)) + s_b)      # 无窗口损失路径参考
```

- 部分和 `q_b` 为精确 dyadic，**不预先舍入为 FP32**，也不压缩成其他有限精度浮点格式；
- `exact_value(state) + partial` 精确相加后仅做一次 FP32 RNE（含渐进下溢、溢出到 inf）；
- 累加器不作为第 17 项进入 W 窗口，部分和与累加器的对阶不引入第二个实验窗口；
- 所有舍入由整数算法实现（`arithmetic/rounding.py`），不经过 host binary64。

实现位置：`accumulation/accumulator.py`（候选）、`reference/exact.py`（参考路径）。

### 1.4 误差分解

```
E_align = Σ(q_b − s_b)
delta_b = a_b − (a_(b-1) + q_b)
E_acc   = Σ delta_b
```

对每个有限样本精确满足 `a_B − S_B = E_align + E_acc`；引擎在每个快照处用精确 dyadic
断言该恒等式。恒等式只在各项均有定义时检查：候选与参考均有限、且 `E_acc` 完整。

完整性规则：

- `E_align` 只依赖窗口规约结果，只要该 W 未发生窗口溢出就始终完整（即使 a/r 已非有限）。
- 一旦候选累加出现非有限值（FP32 溢出使 `delta_b` 无定义），完整 `E_acc` 无定义：
  快照记录写 `"E_acc": null`、`"E_acc_complete": false`，同时保存明确命名的
  `E_acc_prefix`（溢出前累计值）与 `E_acc_prefix_until_block`（最后一个贡献了有效
  delta 的 block 索引，无则 null）。`E_acc_prefix` 只是诊断，不进入任何整批指标。

### 1.5 比较定义

| 名称 | 实际值 | 参考值 | 备注 |
|---|---|---|---|
| total | a_B | S_B | 候选总误差 |
| window_effect | a_B | r_B | 有限 W 的最终影响，含舍入传播 |
| fp32_baseline | r_B | S_B | 参考基线，不是候选误差的严格下界 |
| alignment_only | Q_B | S_B | Q_B = S_B + E_align，对阶部分和的无限精度累加，最终不转 FP32 |

`alignment_only` 的误差就是精确 `E_align = Σ(q_b − s_b)`：先按有符号误差逐 block 精确累计，
再作为整体与 `S_B` 比较；不得把各 block 的绝对误差相加。`Q_B` 只在需要时由 `S_B + E_align`
恢复（精确 dyadic），绝不舍入到 FP32。有效性独立于 FP32 候选/参考状态：只要 `E_align`
完整（快照记录按构造覆盖其 B 前缀，abort 样本走独立记录；显式 `E_align_complete=false`、
字段缺失、null 或畸形均判无效，不默认成零），即使 a_B 或 r_B 已非有限也可正常统计。
因有效集合与 `stage E_align` 一致，`alignment_only` 的 mean_signed_error / rmse_abs_error
等于该 cell 的 E_align 阶段 mean / RMS。

指标：`p99_abs_error`、`p99_eta_l1`、`relative_l2 = sqrt(ΣE_align²/ΣS²)`、
`rmse_abs_error`、`max_abs_error`、`mean_signed_error`、`match_rate`（精确判断 E_align 是否为零）、
`reference_rms`，以及 p50/p90 附加项；eta 分母为 `L1_B`；零分母与未定义沿用明确规则。

- `eta_l1 = |error| / L1_B`，先逐样本精确计算再取分位数；
- `relative_l2 = sqrt(Σerror² / Σreference²)`（window_effect 的参考为 r_B，另两项为 S_B）；
- p99 采用位置 `(N−1)*0.99` 的线性插值；不同 seed 的逐样本结果先合并再取分位数；
- match 用精确数值比较（±0 视为相等），不经过 binary64；
- 分母全零：误差也为零返回 0，否则 `value=null` + `status=undefined_zero_*`，不使用
  `denominator or 1`；
- 浮点转换只发生在精确误差得到之后；未定义/非有限写 `null` + status，JSON 禁止裸 NaN/Infinity。

阶段统计（E_align、E_acc）报告逐样本均值和 RMS，作为独立诊断，不参与总误差相加。

### 1.6 事件

`window_overflow`、`non_finite_fp32`、`unexpected_zero`（a_b=0 且 r_b≠0，仅诊断）、
`alignment_zeroed`、`update_absorbed`、`all_zero_block`。

每条快照记录同时保存 `events_cumulative`（前缀累计）与 `events_endpoint`（该前缀最后一个
block 内的事件）。汇总只累加 endpoint 事件，累计计数不再相加。窗口溢出与非有限结果标记
该样本/单元格不可作普通精度统计，但仍写入记录、计数，不静默丢弃。

## 2. 采样

七种分布（默认全部）：`positive`、`uniform`、`exponent_gaussian`、`cancellation`、
`laplace`、`student_t3`、`outlier`。

- `positive`：raw exponent 在采样范围内均匀，给定 exponent 的合法 fraction 条件均匀，两侧全正；
- `uniform`：与 positive 相同幅度规则，两侧符号独立公平；
- `exponent_gaussian`：raw exponent 截断离散高斯（FP8 中心 7、sigma 1；TF32 中心 127、sigma 4），
  fraction 条件均匀，符号独立公平；
- `cancellation`：复用 exponent_gaussian 幅度，按精确乘积幅度降序贪心分配乘积符号使其与
  当前和相反（和为零时随机），随机分解到两个输入符号，最后打乱 lane；不拒绝幅度样本；
  记录 `rho`；`sign_mode=independent` 提供同幅度独立随机符号对照（诊断，不新增默认分布）；
- `laplace` / `student_t3` / `outlier`：单位方差连续形状积分成编码 PMF。正正常编码有序值之间
  取算术中点，最小边界向下延伸半个间距（下界钳 0），最大边界向上延伸半个间距；质量为
  `F(u_j) − F(l_j)`，在保留区间内重新归一化，外部尾部质量记录而不压到端点编码；
  上尾使用稳定 SF/级数公式避免 CDF 相减消减。

随机流：`sha256("mac-stream-v1|seed|format|distribution|sample_index|block_index|purpose")`
映射到独立 `random.Random`，purpose ∈ {amplitude, sign, lane}。不使用 Python 随机化 hash；
扩样本数/Bmax 不改变已有前缀；并行调度不影响数据。

PMF 元数据（边界规则、截断支持、目标参数、保留质量、离散 RMS、幅度分位点）写入 manifest。
这些分布是合成形态，不声称代表真实 LLM 张量。

## 3. 模块职责

```
src/mac_experiment/
  formats/      编码集合、精确解码/乘积、窗口常数（无实验循环）
  arithmetic/   Dyadic、Fp32(bit pattern)、整数 RNE 舍入
  reduction/    16 项窗口对齐 + 宽整数求和 + 溢出检测（无格式分支）
  accumulation/ FP32 状态 × 精确部分和 → 一次 RNE
  reference/    独立精确点积、L1、无窗口损失路径
  sampling/     PMF、符号策略、确定性流
  execution/    单样本流式执行与快照（不复制计算核心）
  experiments/  配置校验 + W/B driver（只组织扫描）
  metrics/      流式误差累计与分位数（不依赖格式/采样）
  results/      schema、严格 JSON 读写、离线重算、alignment_only 离线增强
  plotting/     只读 summary 出图
  cli.py        run / analyze / enhance / plot / distributions / version
```

W driver 与 B driver 共用同一 `SampleEngine`；`tests/legacy/test_decomposition.py`
断言同一 (W,B) 在两种 driver 下记录逐字段一致。

`results/enhance.py` 提供离线增强入口：只读逐样本记录（流式），用与首次汇总相同的
`SummaryBuilder` 重建包含 `alignment_only` 的 summary，输出到独立目录
`results_enhanced/<源run_id>_alignment_only_<时间戳>/`（`analysis_manifest.json` 记录源 run、
源 schema、源文件 sha256、记录数与分析器版本），并保证不修改源 run、不导入或执行
采样器与数值引擎。

## 4. 结果 schema（`mac-exp-v2`）

### 4.1 逐样本记录

快照记录字段：`record_type=snapshot`、`mode`、`sample_id`、`seed`、`sample_index`、
`format`、`distribution`、`W`、`B`、`K=16B`、`a_bits`、`r_bits`、`a_finite`、`r_finite`、
`S`、`L1`、`E_align`、`E_acc`（完整时）或 null、`E_acc_complete`、`E_acc_prefix`、
`E_acc_prefix_until_block`、`events_cumulative`、`events_endpoint`、`rho`、
`rho_per_block`、`input_fingerprint`、`status`（`ok` / `nonfinite_fp32`）。

溢出中止样本写 `record_type=abort` 记录（`W`、`B`、原因），计数进入汇总。

### 4.2 有效性判定（按比较对象独立）

| 比较 | 需要的有限值 |
|---|---|
| total | 候选 `a_B`（`S_B` 始终为精确有限 dyadic） |
| window_effect | 候选 `a_B` 与路径参考 `r_B` |
| fp32_baseline | 路径参考 `r_B` |
| alignment_only | 完整的精确 `E_align`（与 a_B、r_B 是否有限无关） |

只有候选非有限时 `fp32_baseline` 与 `alignment_only` 仍照常统计；只有参考非有限时
`total` 与 `alignment_only` 仍照常统计。单元格保留失败状态（`status`、
`comparison_statuses`、`counts.nonfinite`）。

### 4.3 整批 / 有效子集状态

每个比较 payload：

```
{
  "status": "ok" | "partial_nonfinite" | "no_valid_samples" | "no_samples",
  "coverage": {"n_total", "n_valid", "n_invalid", "invalid_rate"},
  "metrics": { ... },            # 整批指标；status != ok 时全部 null + 同一状态
  "reference_rms": {...},
  "finite_subset": {             # 仅 partial_nonfinite 时存在
      "note": "computed over the finite subset only; not a whole-batch metric",
      "coverage": {...}, "metrics": {...}, "reference_rms": {...}
  },
  "counts": {...}
}
```

- 全部有效：`ok`，指标定义与数值与修复前完全一致。
- 部分无效：`partial_nonfinite`，整批指标为 null；有效子集指标单独保存并明确标注。
- 全部无效：`no_valid_samples`，不返回任何指标（尤其 `relative_l2` 不为 0）。
- 无快照记录：`no_samples`。
- 失败样本绝不当作有限误差、绝不被静默丢弃、绝不用替换分母/补零/任意大误差处理。

阶段诊断使用同一策略：`E_align`、`E_acc` 各自有 status/coverage/finite_subset。

### 4.4 汇总、CSV 与离线重算

- `summary.json`：cells 按 (format, distribution, x) 组织，含上述比较 payload、阶段
  诊断、事件与计数；cells 另有 `status` 与 `comparison_statuses`。
- `summary.csv` 为长表，列：format, distribution, x_name, x, comparison, scope
  (`whole_batch` / `finite_subset`), metric, value, status, reference_rms, n_total,
  n_valid, n_invalid, invalid_rate, note。
- 读取缓存前比对 manifest 的完整 config 与 config_hash，不允许仅凭目录名跳过。
- 离线重算：`analyze` 用与首次汇总相同的记录聚合器重建 summary，写入
  `summary_recomputed.json`（不覆盖 `summary.json`）并报告逐字段差异。
- 严格 JSON：禁止裸 NaN/Infinity；未定义写 `null` + status。

### 4.5 schema 变更记录

| 版本 | 变更 |
|---|---|
| `mac-exp-v1` | 初版。非有限样本直接作为 undefined 加入所有比较；无整批/子集区分。 |
| `mac-exp-v2` | 新增 `a_finite`/`r_finite`/`E_acc_complete`/`E_acc_prefix`/`E_acc_prefix_until_block`；比较 payload 新增 `status`/`coverage`/`finite_subset`；cell 新增 `status`/`comparison_statuses`；CSV 新增 `scope`/coverage 列；summary 版本升为 `mac-summary-v2`。v1 记录仍可离线重算（`E_acc_complete` 由记录的候选有限性推导），重算只写 `summary_recomputed.json`。 |
| `mac-summary-v3` | 比较集合增加 `alignment_only`（error = 精确 `E_align`，reference = `S_B`，eta 分母 = `L1_B`）。记录 schema 不变（仍 `mac-exp-v2`）；引擎数值语义与记录字段不变。v1/v2 记录可用 `enhance` 离线增强，输出到独立目录。 |

## 5. 绘图

- 每种格式、每个指标一张图，七种分布为子图；同图显示 total / window_effect / fp32_baseline /
  alignment_only（紫色三角，独立一致样式），图例含参考分母与 Q_B 说明。
- W 图横轴 18..41 全点；B 图 log2 横轴（默认倍增）。
- 整批指标是主曲线；`status != ok` 的点为 `null`，用红色 × 标记并在子图标题注明
  失败 x 点数与失败样本数。`finite_subset` 仅作为虚线诊断曲线（图例明确标注
  "finite subset only"），不得冒充整批曲线。
- 零值画在 0（symlog 或线性，图注写明阈值）；异常事件区间用红色阴影。
- 标题/图注包含格式、固定 B 或 W、样本数、对齐策略与舍入语义。
- 阶段均值/RMS 为独立诊断图，E_acc 非完整处标记为未定义。
- 绘图只读 summary（兼容 v1 汇总，缺失字段按旧 counts 推导），可仅从结果重绘。

## 6. 位级匹配与 ULP distance 离线分析（bitmatch）

对无窗口损失路径的输出转换做位级分析，不涉及候选 a_B。

### 6.1 输出与舍入

```
o32 = RN32(S_B)
o16 = RN_BF16(S_B)          # 直接由精确 dyadic 舍入，禁止 FP32/FP64 中转
r16 = RN_BF16(r_B)          # r_B 为记录的 FP32 bit pattern，按其精确值舍入
d16 = RN_BF16(o32)          # 辅助：双重舍入诊断
```

统一策略：RNE ties-to-even；保留次正规数（无 FTZ）；按目标格式溢出到 ±Inf（不饱和）；
±0 按 IEEE 保留（`r_B = -0` 时 `r16 = -0`）。`S_B` 是精确 dyadic，没有符号零信息，
规范参考零统一为 +0（不声称恢复原始符号）。

### 6.2 主比较与辅助比较

| 比较 | reference | result | 说明 |
|---|---|---|---|
| fp32 | o32 | r_B | 主比较 |
| bf16 | o16 | r16 | 主比较 |
| aux_double_rounding | o16 | d16 | 双重舍入诊断 |
| aux_accum_bf16 | r16 | d16 | 累加路径经 BF16 输出后的变化 |

辅助比较只保存，不取代主比较；其匹配率/距离不得相加声称误差分解。

### 6.3 指标与分母

- `bit_match` / `bit_mismatch`：完整编码比较，严格区分 +0 与 -0；分母 `n_total`。
- `signed_zero_mismatch`：bit 不匹配但 ULP distance 为 0（±0 合并）。
- ULP distance：目标格式有序可表示有限数序列上的步数（精确整数）；相邻为 1，
  跨指数边界按可表示步数，±0 合并；仅在两侧均有限时计算。
- 条件统计（p50/p90/p99/max）：仅有限且 bit 不匹配样本，分母 `n_finite_mismatch`；
  无此类样本时全部为 null + `no_mismatch_samples`，不填 0。分位数用 nearest-rank。
- 距离分桶：0、1、2、3–4、≥5，分母 `n_finite_pairs`（含匹配样本）。
- 特殊事件（每对独立计数）：`reference_finite_result_nonfinite`、
  `reference_nonfinite_result_finite`、`both_inf_same_sign`、`both_inf_opposite_sign`、
  `any_nan`；NaN 编码相同不代表数值正确。
- 分组：按 (format, distribution, B 或 W)；seed 以逐样本距离池化，禁止平均各 seed 分位数。
- W 扫描：每个 W 单独成 cell；同一样本的 S_B/r_B 在 W 间按构造重复，不得合并为独立样本。

### 6.4 离线入口与输出

`results/bitmatch_analysis.py` + `mac-experiment bitmatch`：流式读取记录，不导入/执行采样器
与数值引擎，不修改源 run。输出到独立目录
`results_bitmatch/<源run_id>_bitmatch_ulp_<时间戳>/`：

- `analysis_manifest.json`：源 run 路径、源 schema、源文件 sha256/大小、记录数、
  分析器版本、舍入/ULP/分位数策略、主/辅助比较清单；
- `summary.json` / `summary.csv`：逐 cell、逐比较的计数、匹配率、条件 ULP、分桶、事件、
  by_seed 计数；
- `plots/`：每种输入格式 × 每种主比较（FP32、BF16）三张图：
  bit match/mismatch rate、条件 ULP p50/p90/p99/max、距离分桶比例。

图表注明输入格式、输出比较、固定参数、样本数与分母、RNE/次正规策略、
条件统计口径；零值与"无不匹配样本"分别用 0 和空心红圈表示，不替换为任意小正数。

## 7. MXFP8 矩阵实验（`mac-mx-exp-v2`）

MX 矩阵实验（数值域 FP32 输入 → G32 MXFP8 量化、元素 E4M3、scale E8M0、输入/输出分组 32、
dot16 + FP32 累加、无 W 窗口、输出 MX 量化、paired-prefix K 扫描）使用独立 schema 与
独立命令，不与本文档第 1–6 节的 dot-product 语义和结果混用。完整数值语义、格式依据
（OCP MX / OFP8）、输入流水线与输出量化策略见 `docs/mx_semantics.md`；实测与正式规模
估计见 `docs/verification.md` 第 10 节（第 9 节为历史 v1 输入方案记录）。
固定对阶精度 K/B 扫描（`mac-mx-align-v1`，W_align=21）独立成节：见
`docs/mx_alignment.md` 与 `docs/verification.md` 第 11 节。

模块划分：`matrix/inputs.py`（FP32 采样与输入量化）、`matrix/alignment.py` +
`matrix/aligned_engine.py`（固定对阶与 T/A 轨迹）、`matrix/tensor.py`、
`matrix/engine.py`（精确参考 + dot16/FP32）、`matrix/quantize.py`（输出量化）、
`metrics/mx_quant.py`（量化比较）、`results/mx_analysis.py` + `results/mx_io.py`
（离线分析）、`plotting/mx_plots.py`、`experiments/mx_matrix_sweep.py` +
`config.py` 的 `mx_matrix_sweep` 模式。`matrix/sampling.py` 为已退役的 v1 采样模块，
仅作历史参考，活动路径不再调用。
