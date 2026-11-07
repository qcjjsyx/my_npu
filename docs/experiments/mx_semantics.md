# MXFP8 矩阵实验：数值语义与配置说明

> 2026-10-31 目录迁移说明：本文保留原实验语义与历史记录。代码和配置现位于 `experiments/`；当前运行方式及缺失的历史数据见[实验指南](README.md)。历史实验结论不直接等同于旧基线 Q2.34 RTL 的精度证据。

本文档描述 `mx_matrix_sweep` 实验（schema `mac-mx-exp-v2`）的显式数值语义、格式依据、
输入流水线、量化策略与配置。它是本实验的权威说明；与旧 dot-product 实验
（`mac-exp-*`）的数值结论、schema 和命令完全分离。

当前输入流水线（第 3–4 节）：**数值域采样 → FP32 输入矩阵 → 按 G32 量化为 MXFP8 →
精确参考与 dot16＋FP32 累加 → MXFP8 输出量化比较**。历史 v1 输入方案（直接独立采样
E4M3 元素编码与 E8M0 scale）已退出活动入口，代码保留但不再被运行/配置/分析调用，
其历史结果保持只读且默认排除。

重要免责声明：本实验模拟的是一组**明确写出的数值语义**（dot16 精确部分和 + 每次
FP32 RNE 更新 + 固定输出量化策略），**不声称**代表任何具体 NPU/GPU 硬件内部流水线。
不得用现有 FP32 状态累加行为反推硬件实现。

## 1. 格式依据（authoritative format specifications）

格式规范与本实验选择的量化策略分开描述：

- 元素格式 E4M3 与 scale 格式 E8M0 的定义来自
  Open Compute Project, *OCP Microscaling Formats (MX) Specification v1.0*（2023）：
  E8M0（8 位纯指数、bias 127、码 255 保留为 NaN）与 MXFP8 块大小 32 的约定。
- E4M3 编码规则（bias 7、无 Inf、`S.1111.111` 为 NaN、最大有限幅度 448）来自
  Open Compute Project, *OCP 8-bit Floating Point Specification (OFP8) v1.0*；
  背景见 Micikevicius et al., *FP8 Formats for Deep Learning*, arXiv:2209.05433。
- 本实验的**输出 scale 选择策略**（“覆盖组最大幅度的最小二进制幂”）和**元素饱和
  策略**是实验选择，不是格式规范的一部分，见第 5 节。

### 1.1 E4M3 元素语义（OCP，无 FTZ）

| 编码 | 含义 |
|---|---|
| `raw_exp = 0, frac = 0` | 符号零 `±0`（`0x00` / `0x80`） |
| `raw_exp = 0, frac > 0` | 次正规数，值 `frac × 2⁻⁹` |
| `raw_exp = 15, frac = 7` | NaN（`0x7F` / `0xFF`） |
| 其余 `raw_exp = 15` | 有限正规数，最大幅度 448（`0x7E` / `0xFE`） |
| 正规数 | 值 `(8 + frac) × 2^(raw_exp − 10)` |

E4M3 **没有 Inf**：编码只能饱和到对应符号的最大有限值 448，不存在“溢出到 Inf”。

与现有 `formats/fp8_e4m3.py`（`Fp8E4M3`）的区别：旧适配器是 dot-product 实验的
**输入适配器**，对 `raw_exp == 0` 做 FTZ（一律当零）。MX 元素编解码器
`formats/e4m3_element.py` 显式支持次正规数与符号零（无 FTZ），两者语义不同、互不替代。
测试 `tests/test_mx_formats.py` 覆盖全部 254 个有限码的编解码往返。

### 1.2 E8M0 scale 语义（OCP）

| 编码 `c` | 含义 |
|---|---|
| `0..254` | 精确值 `2^(c − 127)`（`2^-127 .. 2^127`） |
| `255` | 保留（NaN），**不得**当作有限 scale 使用 |

scale 通过精确的 dyadic 指数平移应用：`x · 2^e = Dyadic(x.m, x.e + e)`，
不使用宿主浮点乘法，也不通过 `log2(float(...))` 决定边界。

## 2. 矩阵与逻辑分组

计算 `X[M,K] · W[K,N]`：

| 张量 | 分组轴 | 组内元素 | scale 逻辑形状 |
|---|---|---|---|
| X | 沿 K（axis 1） | 32 | `[M, K/32]` |
| W | 沿 K（axis 0） | 32 | `[K/32, N]` |
| 输出 | 沿 N（axis 1） | 32 | `[M, N/32]` |

约束：`K`、`N` 必须是 32 的正整数倍，`M ≥ 1`；不支持不完整组，非法配置直接报错。
张量 `matrix/tensor.py::MxTensor` 显式保存元素编码、scale 编码、逻辑形状、分组轴、
组长、元素格式与 scale 格式；存储是行主序的便利布局，逻辑分组轴不因存储改变。

## 3. 输入流水线（数值域采样 → FP32 → 输入 MXFP8）与精确参考

当前输入流程（`MX_INPUT_SAMPLING_VERSION = mx-fp32-input-v1`，
`MX_INPUT_STREAM_VERSION = 1`）：

1. 按数值域分布采样 binary64 数值（算法见第 4 节）；
2. 每个样本经**显式整数 RNE** 从其精确 dyadic 值转换为 binary32，记录最终 FP32 bits；
   非有限值直接报错，不裁剪、不替换、不重新抽样；
3. 用现有精确 MX 量化器把 FP32 矩阵按 G32 量化为 MXFP8（第 4.3 节），得到元素编码与
   E8M0 scale 编码；其**精确解码值**记为 `X̃`、`W̃`。

设 `X̃[m,k] = 2^alpha[m,g] · u[m,k]`，`W̃[k,n] = 2^beta[g,n] · v[k,n]`，`g = k // 32`。

- 精确参考（任意精度 dyadic，**使用解码后的 MX 值**，不使用原始 FP32 矩阵，
  也不使用 FP32/FP64/NumPy matmul）：

  `S[m,n] = Σ_k X̃[m,k]·W̃[k,n]`，`L1[m,n] = Σ_k |X̃[m,k]·W̃[k,n]|`

- 第 b 个 dot16：`s_b[m,n] = Σ_{i=0..15} X̃[m,16b+i]·W̃[16b+i,n]`（精确求和）；
  scale 在部分和进入 FP32 累加前生效，且逐元素应用，不提到整个点积之外。
- FP32 路径：`r_0 = 0`，`r_{b+1} = RN32(r_b + s_b)`，最终 `R[m,n] = r_B`。
  每次更新只做一次 RNE，保留 FP32 次正规数（渐进下溢），部分和不预先转 FP32。
- 一个 32 元素输入 scale 组恰好包含两个 dot16：两者共享组 scale，但**分别**更新 FP32，
  不合并成一次 dot32 累加。`tests/test_mx_engine.py` 给出 E4M3 级反例区分两种实现。
- **无有限 W 窗口损失**：本实验不使用旧 reducer 的窗口模型。

结论口径：本实验测量**给定 FP32 数据分布经 MXFP8 输入量化后**，这些 MX 输入在
dot16＋FP32 累加与输出量化中的差异；**不测量**输入量化相对于原始 FP32 矩阵乘法
`X_32 W_32` 的精度损失。输入量化误差不是本次主指标。

## 4. 输入分布、随机流与输入量化

### 4.1 数值域分布（实验定义）

| 名称 | 定义 | 理论均值 | 理论 RMS |
|---|---|---|---|
| `gaussian` | `N(0,1)` | 0 | 1 |
| `laplace` | 位置 0、尺度 `1/√2` | 0 | 1 |
| `student_t3` | 自由度 3 的 t 除以 `√3` | 0 | 1 |
| `outlier` | 0.99 概率 `N(0,1)`、0.01 概率 `N(0,8²)`，整体除以 `√1.63` | 0 | 1 |

激活与权重分别配置（`activation_distribution` / `weight_distribution`），可以是不同分布；
活动配置显式列出 `distribution_pairs`，正式模板默认只跑四种“同分布”组合，不默认执行
所有交叉组合。**自定义分布参数当前不支持**：`activation_parameters` / `weight_parameters`
会被配置加载器与采样器明确拒绝（`ValueError`），不存在“记录参数但不生效”的路径；
四种分布固定为上表的单位 RMS 实验定义。不加入额外随机通道尺度、block 尺度或真实模型数据。不对每个实际矩阵或
每个 G32 组强制归一化为单位 RMS——上述归一化只是分布理论参数，保留组间自然变化与
K 前缀结构。

这些是实验定义，不宣称代表真实 LLM 张量。样本由有限精度算法（binary64 数学函数 +
精确 dyadic RNE 到 FP32）产生，**不描述为数学上精确的连续分布样本**。

### 4.2 随机流与独立矩阵对

key：`mac-mx-fp32-stream-v1|seed|matrix|distribution_label|pair_index|row|value`，
其中 `distribution_label` 为分布名 + 参数的规范 JSON。因此：

- 每个矩阵对由 seed、pair_index、激活/权重标识、分布与参数、行坐标唯一确定；
- `pair_index` 实际进入随机流；相同配置重复生成一致，不同 pair 生成不同非退化矩阵；
- 改变最大 K 不改变较小 K 的前缀；增加 pair 数量或改变遍历顺序不改变已有 pair；
- 激活（`matrix=x`）与权重（`matrix=w`）流独立；不使用 Python 进程随机化 `hash()`；
- 每行按全局列号顺序采样，因此前缀稳定；不假定向量化改变请求形状后前缀不变
  （实现逐行顺序采样并有回归测试）。

### 4.3 输入 FP32 → MXFP8 量化

复用现有精确 MX 量化器：

- X 每行沿 K、G32（axis 1）；W 每列沿 K、G32（axis 0）；E4M3 元素、E8M0 scale；
- scale 为“覆盖组最大幅度的最小二进制幂”；元素 RNE、保留次正规数；全零组用规范 scale；
- 每一步（最大值、阈值、scale 选择、元素舍入）都从 FP32 bits 精确解码出的 dyadic 计算，
  不用近似浮点 `log2`；
- 输入与输出量化复用底层实现，但元数据与事件统计区分 `input_X`、`input_W` 与各输出
  量化路径（`x_input_quantization_events` / `w_input_quantization_events`）。

## 5. 输出 MXFP8 量化（显式实验策略）

## 5. 输出 MXFP8 量化（显式实验策略）

对每个输出组（沿 N 的 32 个元素）：

1. `A = max_j |z_j|`（精确 dyadic）；
2. 取最小整数 `e` 使 `A ≤ 448 · 2^e`（精确整数比较；不使用浮点 log2），
   再限制到 E8M0 有限指数范围 `[-127, 127]`，记录是否上下界截断；
3. 元素编码 `c_j = Q_E4M3(z_j / 2^e)`：RNE ties-to-even、保留次正规数、
   超出有限范围时饱和到对应符号的最大有限值，分别统计超范围、饱和、非零转零事件。

全零组使用规范 scale `e = 0`（码 `0x7F`，值 1.0）。精确参考 `S` 的零规范为 `+0`；
来自 FP32 状态的符号零在量化中保留（`-0` 编码为 `0x80`）。

两种比较：

- **主比较（独立 scale）**：`Q_S = MXQuant(S)`、`Q_R = MXQuant(R)`，各自选 scale；
- **辅助比较（共享精确参考 scale）**：使用 `Q_S` 的 scale 分别量化 `S` 与 `R`，
  禁止重新选 scale；超出范围按约定饱和并记录事件。用于隔离固定网格下的元素编码变化。

非有限输入组（例如 `R` 的 FP32 溢出，或保留 scale 码）标记为无效组并记录原因，
不纳入普通有限误差指标。该策略是显式实验选择，不宣称是所有 MXFP8 硬件的统一量化算法。

## 6. 指标与分母

计数维度必须区分：**矩阵对数量**、**输出组数量**、**元素数量**，每个指标记录分母。
分析版本为 `mac-mx-summary-v2` / `mac-mx-analysis-v2`。

三种不匹配定义必须分开，不得混用：

- **representation（主定义）**：比较 `(scale_code, element_code)` 组合。两组 scale 不同时，
  该组 32 个元素全部为 representation 不匹配（即使部分解码值相同或都为零）；
  “每组不匹配元素数”直方图与“整组 bit match”都用该定义，整组 bit match 等价于
  该组 representation 不匹配数为 0；
- **code_only（显式诊断）**：只比较元素 code、忽略 scale，所有字段显式命名
  `*_code_only_*`，不得当作主不匹配指标，也不得与 representation 相加；
- **decoded_value**：比较精确解码值，独立保留，不得与表示匹配率混淆。

主比较包含：scale 匹配/不匹配计数与比率；整组 representation 匹配/不匹配计数与比率；
元素 code-only / representation 匹配率；解码数值匹配率；每组 representation 不匹配元素
数直方图（0..32，另附显式 code-only 直方图）；scale 指数差 `e_R − e_S` 的直方图与
正/零/负比例；`|decode(Q_R) − decode(Q_S)|` 的 p50/p90/p99/max；解码相对 L2
（参考为 `decode(Q_S)`）；全部量化/非有限事件计数。另有 `R − S` 的既有误差指标
（累加误差）与 `decode(Q_R) − S` 的总误差；若保留 `etaL1`，分母为原始 MX 输入精确
乘积的 `L1`。by-seed 统计分别记录 representation / code-only / decoded / ULP 条件
（`ulp_mismatch_elements`）计数；ULP 条件计数不得冒充全部表示不匹配计数。

ULP distance：

- 共享 scale 比较使用 E4M3 有序有限编码步数；
- 独立 scale 主比较只对 scale 相同的有限组计算，并记录条件分母；
- scale 不同时不计算元素 code 的 ULP；
- `±0` 在数值距离中合并，但 code 匹配区分；
- 条件统计（仅 bit 不匹配元素）报告 p50/p90/p99/max，nearest-rank；
  无样本时全部为 `null`，状态 `no_mismatch_samples`，绝不填 0。

错误分位数用框架既有的线性插值定义（与 dot-product 实验一致）；零分母、空条件样本与
非有限事件都区分处理，JSON 禁止裸 NaN/Infinity。

## 7. K/B 扫描与样本单位

先按最大 K 生成 FP32 矩阵（`X_32` 形状 `M×Kmax`、`W_32` 形状 `Kmax×N`）并按固定 G32
边界量化；小 K 使用 `X_32` 前 K 列、`W_32` 前 K 行及其 MX 编码/scale 前缀，不重新采样、
不重新量化、不重新归一化。`B = K / 16`，归约并行度固定 16；每个 K 的输出单独选 scale。
回归测试覆盖：小 K 单独生成的 FP32 与最大 K 前缀一致；小 K 单独量化与最大 K 的 MX
前缀一致；小 K 单独执行与大 K 扫描中的快照一致。

独立重复单位是**输入矩阵对**。同一矩阵对的不同 K 快照共享输入与 scale，且是前缀关系，
同一矩阵内的输出元素也共享输入；这些元素/快照计数高度相关，不能当作独立 Bernoulli
样本，不能用 `3/N` 之类的零事件规则给出元素失配率的置信上界。本实验只报告观察到的
事件数与描述性比例；若要置信区间，必须先定义估计对象（“每对至少一次失配的概率”与
“元素失配率”是不同目标），再按独立单位抽样。

## 8. 结果文件与离线分析

运行目录 `results_mx/<run_id>/`（当前默认根目录；旧的 `results_mx_smoke/`、
`results_mx_pilot/`、`results_mx_pair_probe/` 属 v1 输入方案，保持只读且默认排除）：

- `manifest.json`：schema `mac-mx-exp-v2`、模式、固定参数、完整配置与 `config_hash`、
  `input_sampling_version` / `input_stream_version` / `input_quantization_policy`、
  策略、格式与输入元数据、期望与实际计数、状态；
- `inputs.jsonl.gz`：每个矩阵对一条记录，包含 X/W 的 **FP32 原始 bits**、MX 元素编码与
  scale 编码、逻辑形状、分组轴/组长、分布与参数、采样/量化版本、输入文件指纹、
  分路径（X/W）输入量化事件与输入诊断；**最大矩阵只保存一次**；
- `samples.jsonl.gz`：每个 (矩阵对, K) 一条快照，含精确 `S`、`L1`、FP32 `r_bits`、
  `K`/`B`/矩阵对 ID、数值事件与状态。

输入诊断（实际 RMS、最大幅度、MX 零值比例、scale 指数直方图）只作为诊断保存，
不是主误差指标。原始 run 只读；`mx-analyze` 离线重算输出量化与指标，输出到独立目录
`results_mx_analysis_v2/<run_id>_quant_<时间戳>/`（`summary.json`、`summary.csv`、
`analysis_manifest.json`、`plots/`）。`analysis_manifest.json` 记录源文件 sha256、
源输入采样/流/量化版本，分析策略与版本（`mac-mx-summary-v3` / `mac-mx-analysis-v3`）。
修改输出量化或绘图不需要重跑 MAC；修改输入分布或输入量化策略需要新运行。
旧 v1 run（schema `mac-mx-exp-v1`，无 `input_sampling_version`）会被 `mx-analyze`
明确拒绝，不会被静默重解释。

## 9. 命令

```bash
# 执行（smoke / pilot / 正式模板；正式模板默认不执行）
.venv/bin/python -m mac_experiment run --config configs/mx_smoke.yaml
.venv/bin/python -m mac_experiment run --config configs/mx_pilot.yaml
# .venv/bin/python -m mac_experiment run --config configs/mx_formal_template.yaml

# 离线输出量化分析 + 绘图（不导入采样器与执行引擎，不修改源 run）
.venv/bin/python -m mac_experiment mx-analyze --run results_mx/<run_id>
.venv/bin/python -m mac_experiment mx-analyze --run <run_dir> --out <dir> --no-plots
```

固定 `run_name` 与已存在的 `mx-analyze --out` 目录都会直接报
`FileExistsError`（不覆盖）；重复执行需要改 `run_name`/`output_root`（或 `--out`）
或先移走旧目录。`analyze`/`enhance`/`bitmatch`/`plot` 只适用于旧 dot-product schema；
对 MX run 会明确报错。旧 v1 输入配置（`scale_scenarios`、旧 `distributions`）会
明确报错而不是被转换成新含义。

## 10. 当前局限

- 单线程 Python；M=16、Kmax=16384 实测约 56 s/矩阵对（单快照），正式 40 对模板
  约 40 min、约 350 MB（见 `docs/verification.md` 第 10 节）。
- 分析在内存中保留逐元素误差浮点值；正式规模内存约数百 MB。
- 未实现有限 W 窗口的 MX 实验；接口（精确 dot16 部分和 + FP32 更新）允许后续接入
  现有 reducer，但本轮不实现。
- 不加入额外随机通道尺度、block 尺度或真实模型数据；矩阵 cancellation 分布未定义。
- 输出量化策略固定为“覆盖组最大幅度的最小二进制幂”，不覆盖其他 MX 量化算法。
- 历史 v1 输入方案（元素/scale 独立采样）已退出活动入口，代码保留仅供历史参考，
  新结果不与其混合；文档只在历史说明中提及。
