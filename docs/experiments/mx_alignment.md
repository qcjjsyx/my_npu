# MXFP8 固定对阶精度 K/B 扫描：数值语义与配置

> 2026-10-31 目录迁移说明：本文保留原实验语义与历史记录。代码和配置现位于 `experiments/`；当前运行方式及缺失的历史数据见[实验指南](README.md)。历史实验结论不直接等同于当前 Q2.34 RTL 的精度证据。

本文档描述 `mx_alignment_sweep` 实验（schema `mac-mx-align-v1`）：在**相同的 MXFP8
输入**上，固定对阶小数位 `W_align = 21`，只扫描 K/B，研究 dot16 内乘积有限精度对阶
误差及其与 FP32 累加、MXFP8 输出量化的相互作用。

结论口径：测量 **MXFP8 输入**在 dot16 有限对阶 + FP32 累加 + MX 输出量化中的差异；
不测量输入量化相对原始 FP32 矩阵乘法的损失。

历史口径说明：旧的 dot-product 正式 block run 曾核查为固定窗口 `W = 24`
（见 README 的 `configs/formal_block.yaml` 注释）；那是旧 FP8 窗口实验的语义，与本轮
MXFP8 对阶实验的 `W_align = 21` **不是同一个量**，旧 run 也未被重新解释为 W_align=21。
本实验的被测实现使用 `W_align` 这一名称以区别于权重矩阵 `W`。

基线语义不变：数值域采样 → 显式 RNE 到 FP32（保存 bits）→ G32 MXFP8 输入量化 →
数值执行 → G32 MXFP8 输出量化与离线分析。E4M3（次正规保留、无 Inf）、E8M0、X 每行
沿 K G32、W 每列沿 K G32、输出每行沿 N G32、MAC 并行度 16（B = K/16）、一个输入 G32
对应两个 dot16 且分别选公共指数、分别更新 FP32；无窗口路径与历史语义不变。

## 1. 对阶模型（严格定义）

对每个有限 E4M3 元素（`matrix/alignment.py`）：

```
raw_exp = 编码 4 位指数, frac = 编码 3 位尾数
E = max(raw_exp, 1)
t = frac            当 raw_exp == 0        （次正规，t >= 1）
t = 8 + frac        当 raw_exp > 0
u = (-1)^sign * t * 2^(E - 10)
raw_exp == 0 且 frac == 0 为零；0x7F/0xFF（NaN）拒绝。
```

对某输出的一个 dot16 中第 i 个乘积：

```
P_i = (-1)^(sign_x XOR sign_w) * t_x * t_w
C_i = E_x + E_w
v_i = P_i * 2^(C_i - 20)          （不含 MX scale）
h   = alpha + beta                （该 G32 组两个输入 scale 的真实指数和）
```

- 对阶整数计算不减 bias（指数差中 bias 抵消），也不引入 `h`（16 项共享）；
- 恢复真实部分和时必须减去 bias 并施加 `h`。

非零乘积中取 `Cmax = max(C_i)`。逐项：

```
n_i = RNE_integer(P_i * 2^(W_align - 6 - (Cmax - C_i)))
    = rne_shift(P_i, shift_i),   shift_i = (Cmax - C_i) + 6 - W_align
shift_i > 0：有符号 RNE ties-to-even 右移；shift_i <= 0：精确左移
N_b = sum_i n_i                     （精确整数和）
q_b = N_b * 2^(Cmax - 14 - W_align + h)
W_align = 21 时：q_b = N_b * 2^(Cmax - 35 + h)
```

`14` 是两个 E4M3 指数 bias 之和。对阶整数容器为 `W_align + 7` 位有符号
（W_align=21 → 28 位，范围 ±(2²⁷−1)）；逐项与整数和都做范围校验，越界抛
`ArithmeticError`，不 wrap/截断/饱和。

不变式与显式路径：

- 不把乘积先舍入成 FP8/FP16/FP32；
- 逐项对阶后求和，不对精确 dot16 总和只舍入一次；加法树各级不再加舍入；
- `q_b` 不预舍入成 FP32；FP32 累加器不作为第 17 项进入对阶窗口；
- 部分和与累加器之间没有第二个有限对阶窗口；
- `q_b` 与当前 FP32 状态精确相加，再做一次 FP32 RNE（保留次正规）；
- 全零 dot16 走显式零路径：`q_b = 0`，`Cmax`/窗口指数为 null；非零乘积相消到
  `N_b = 0` **不**算全零 dot16；
- “右移发生”不等于“数值改变”；实际改变用精确比较（`(n << shift) != P`），
  不是只检查 `shift > 0`。

## 2. 四条轨迹与比较

对每个输出元素：

| 轨迹 | 定义 |
|---|---|
| S | 精确参考：`S = Σ_b s_b`，`s_b = Σ_i v_i · 2^h`（MX 解码值） |
| R | 无对阶损失 FP32 基线：`r_{b+1} = RN32(exact(r_b) + s_b)` |
| T | 对阶后部分和的精确累加：`T = Σ_b q_b`（不转 FP32） |
| A | 有限对阶候选：`a_{b+1} = RN32(exact(a_b) + q_b)` |
| L1 | `Σ_k |decode(X)·decode(W)|` |

比较与参考分母（离线分析，全部精确 dyadic）：

| 比较 | 误差 | 参考 | 含义 |
|---|---|---|---|
| `align_T_minus_S` | T−S | S | 纯对阶累计误差 |
| `baseline_R_minus_S` | R−S | S | 原无窗口 FP32 累加误差 |
| `candidate_A_minus_T` | A−T | T | 候选路径 FP32 累加误差 |
| `candidate_A_minus_S` | A−S | S | 候选输出量化前总误差 |
| `candidate_A_minus_R` | A−R | R | 引入对阶机制后的最终 FP32 变化 |

有限快照逐元素精确验证 `A−S = (T−S) + (A−T)`；分析若不满足直接报错，不输出“成功”。
维护逐更新 delta 时 `Σ_b delta_b = A−T`（测试用独立 oracle 验证）。
`A−R` 不是纯对阶误差；不可假定 `A−S = (T−S) + (R−S)`。
候选或基线非有限时按比较对象分别判有效；**T 与 T−S 仍精确计算**，不因 A 非有限而丢弃。
误差累计不完整时用 `null` 与完整性字段（`coverage`/`finite_subset`），不冒充完整误差。

## 3. 输入复用与运行身份

* 首选 `source_run` 复用已有 `mx_matrix_sweep` run（例如 `results_mx/mx_formal`）保存的
  MX 元素/scale 编码：校验 schema、`input_sampling_version`、分布组合/顺序、seeds、
  `pairs_per_seed`、K 子集、形状/分组轴、指纹、FP32 bits 有限性与记录唯一性；
* 按 paired-prefix 切到请求的 K 前缀（`K_list` 必须是源 `K_list` 的子集）；
* 每个被复用矩阵对的实际快照 K 集合必须等于**源 manifest 声明的完整 `K_list`**：
  缺失 K 或未声明的 K 都报错（`sample_id` + missing/extra），请求子集不会掩盖源 run
  的残缺；与“源 manifest 必须是 completed 且计数与身份集一致”共同构成源完整性检查；
* 重新计算无窗口 `S/L1/R` 并与源快照**逐元素精确比对**，不一致立即报错并给出
  `sample_id/K/元素序号/字段`；
* 源 run、源分析、历史配置只读；新 run 使用独立目录与 schema；
* manifest 记录源 run ID、源 config hash、源文件 sha256+size，不只靠路径；
* fresh 模式（无 `source_run`）用当前输入流水线生成并保存完整输入。

## 4. 输出量化与比较

保持现有精确量化策略（覆盖组最大幅度的最小二进制幂、E4M3 RNE、次正规、饱和、全零组
规范 scale、符号零保留）。对 S、R、A 分别量化得 `Q_S`、`Q_R`、`Q_A`：

- 主比较（各自选 scale）：`Q_A vs Q_S`、`Q_A vs Q_R`、`Q_R vs Q_S`；
- 辅助比较（使用 `Q_S` 选出的同一参考 scale 重量化全部三条，不重选）；
- 输出总误差：`decode(Q_A)-S`、`decode(Q_R)-S`，另保留 `decode(Q_S)-S` 作为理想输出
  量化基准（参考均为 S，etaL1 统一用该输出原 MX 输入乘积 L1）；
- T 不量化为默认主路径。

MX 编码指标沿用：scale mismatch、整组 representation `(scale_code, element_code)`
mismatch、元素 representation mismatch、显式 code-only mismatch、decoded-value
mismatch、每组不匹配元素直方图、scale 指数差、条件 E4M3 ULP（独立比较仅 scale 相同组；
共享比较在有效组；±0 数值距离合并、编码区分；空条件为 null）。新增输出转变计数
（representation 定义，组与元素两个层级）：`both_match`、
`baseline_match_candidate_mismatch`、`baseline_mismatch_candidate_match`、
`both_mismatch_candidate_differs`、`both_mismatch_candidate_same`，各自带分母与比率。

## 4.1 离线分析完整性与错误处理

- **数据结构完整性问题立即报错**：缺失输入记录/张量/指纹、代码数组畸形、样本身份冲突、
  快照缺失/重复/未知，都抛出带 `sample_id`（必要时含 `K`/字段/元素序号）的错误；
  完整性检查在创建输出目录之前完成，失败不会留下半成品目录。
- **预期集合来自 manifest 身份**：由 `distribution_pairs × seeds × pairs_per_seed × K_list`
  推导预期 `sample_id` 与 `(sample_id, K)` 集合，并校验 manifest `status=completed`、
  `counts`/`expected` 与实际记录一致；不靠缩小计数迁就缺失记录。
- 每条快照校验 seed、pair_index、分布、M/N/K/B、`w_align`、`container_bits`、
  `group_length`、`mac_parallelism` 以及 `S/L1/T/r_bits/a_bits`（和可选诊断数组）的长度。
- **明确支持的数值无效**（E4M3 NaN 元素码、E8M0 保留码 0xFF）逐比较累计 invalid 元素/
  组：数值比较按元素、MX 比较按组、transition 记录 valid/invalid/total；部分无效时整批
  指标为 `null`，有限子集放 `finite_subset`，不把有效子集包装成整批结果；T−S 独立有效
  时不因 A/R 非有限而丢弃。
- E8M0 与 E4M3 的合法性判定分开：scale 0..254 合法（`0x7F` 是合法 scale=1），
  `0xFF` 保留；元素 `0x7F/0xFF` 是 NaN。

## 5. 对阶诊断

快照 `events_cumulative`（覆盖到该 K 的全部 block，**cumulative，不跨 K 相加**）：
`products_total`、`products_nonzero`、`products_changed`、`products_zeroed`、
`dot16_total`、`dot16_all_zero`、`dot16_with_changed_product`、`dot16_qb_diff`、
`r_*` / `a_*` 更新与非有限/上溢/吸收计数；`cmax_minus_ci_histogram_cumulative`；
逐元素 `align_abs_sum`（Σ|q_b−s_b|）与 `align_abs_max`（max|q_b−s_b|）。
`T−S` 可由保存的 T 与 S 精确重建。

默认不保存全部乘积/block 轨迹；`mx-align-trace` 按 `sample_id`、输出坐标、block 区间
导出指定 dot16 的逐项明细（P、C、shift、n、changed/zeroed、q_b、s_b、cmax）。

输入诊断（非误差指标）：X/W 分开，pooled RMS 由平方和/元素数计算（不是等权平均），
scale 指数直方图明确标注 group-weighted（每个 scale code 记一次；无编码时为
element-weighted）。修正了旧诊断口径：之前逐元素计数却标成组数的问题不再出现。

## 6. 结果文件、schema 与命令

运行目录 `results_mx_align/<run_id>/`：

- `manifest.json`：schema `mac-mx-align-v1`、`w_align`、`container_bits`、
  `align_policy`、`K_list`、配置与 hash、源身份（reuse 时含文件 sha256/size）；
- `inputs.jsonl.gz`：`align_input_pair`，含分布身份、指纹、输入事件/诊断、源引用
  （reuse 时 `codes_available=false`，编码在源 run）；
- `samples.jsonl.gz`：`align_snapshot`，含 S/L1/T（无损 dyadic）、`r_bits`、`a_bits`、
  事件与诊断、状态；无裸 NaN/Infinity。

离线分析输出 `results_mx_align_analysis/<run_id>_analysis_<时间戳>/`
（`summary.json`、`summary.csv`、`analysis_manifest.json`、`plots/`）。
`summary.source_digest` 是**数值 run** 的源码 digest；
`summary.analyzer_source_digest` 是实际执行离线分析的源码 digest（`
analysis_manifest` 中另有 `numeric_run_source_digest`），二者不混用；分析不导入
执行引擎/采样器（`matrix/__init__` 采用惰性导入）。逐 pair 摘要保留在每个 cell 的
`alignment_diagnostics.by_pair`：五项数值比较的指标/覆盖率/`sum_err2`/`sum_ref2`、
MX 比较计数与 ULP 摘要、transition（valid/invalid/total 分母）、输入事件累计；
cell 指标由逐 pair exact 平方和合并（`merge_from`），不是逐 pair 比值的平均。

```bash
# 小型 smoke（fresh 输入，pairs_per_seed=2）与离线分析
.venv/bin/python -m mac_experiment run --config configs/mx_align_smoke.yaml
.venv/bin/python -m mac_experiment mx-align-analyze --run results_mx_align/mx_align_smoke

# 复用现有 MX 输入 run 的正式模板（未执行）
# .venv/bin/python -m mac_experiment run --config configs/mx_align_formal.yaml
# .venv/bin/python -m mac_experiment mx-align-analyze --run results_mx_align/mx_align_formal

# 指定 dot16 诊断导出（alignment run 入口；reuse 模式自动读取记录的源输入目录）
.venv/bin/python -m mac_experiment mx-align-trace --run results_mx_align/mx_align_prefix_replay \
    --sample-id 'mx|gaussian|gaussian|seed1|00000' --row 0 --col 0 --block-start 0 --block-end 1
# 源目录移动后用 --source-run 显式覆盖；会按记录的文件 sha256/size 校验，不匹配则拒绝
# .venv/bin/python -m mac_experiment mx-align-trace --run <align_run> --source-run <moved_src> ...
# 显式 --w-align 覆盖会在输出中标注 w_align_override=true 与 run_w_align
# block 范围超出该 run 实际执行的 K 前缀会被拒绝
```

## 7. 规模与资源（实测外推）

* 单对实测（复用 `results_mx/mx_formal` 的 gaussian/gaussian seed1 pair0，M=16、N=64、
  Kmax=16384、W_align=21）：无窗口基线校验 39.5 s，对阶候选 60.1 s，合计约 100 s；
  源 `S/L1/R` 与源快照完全一致；
* 正式模板 40 对 × 10 K → 运行约 **1.1–1.2 h**，输出（含 S/L1/T/诊断数组）约
  **数十 MB**；离线分析数分钟、内存数百 MB；
* 本轮只运行 M=2/N=32 的 smoke（数秒），不自动执行正式实验。

## 8. 与旧无窗口实验的可比性

- 无窗口路径的 `S/L1/R` 语义未变；reuse 模式会逐元素校验其与源 run 完全一致，
  因此 `R−S` 指标与旧 MX 实验直接可比；
- 新候选路径单独记录 `T`、`A`、对阶事件与 `A−R`，不改变旧 schema
  （`mac-mx-exp-v2`）的任何数值含义；对齐实验使用独立 schema `mac-mx-align-v1`；
- 输出量化比较复用同一精确量化器与同一组 representation/ULP 定义。

## 9. 当前局限

- 本轮固定 `W_align=21`，不实现/不运行大规模 W 扫描；测试可设其他合法 W（1..60）；
- 未加入额外随机尺度、有限加法树或新的 FP32 窗口模型；
- 正式复用 run 未执行；资源数字为单对实测外推；
- 元素与 K 前缀相关，不提供基于元素独立性的零事件置信上界；正式模板用于稳定误差估计
  与描述性计数；
- 历史 run 与历史分析保持只读，不被覆盖或重新标注。
