# MAC 位宽选择实验（W_align / W_acc）：设计、语义与配置

> 2026-10-31 目录迁移说明：本文保留原实验语义与历史记录。代码和配置现位于 `experiments/`；当前运行方式及缺失的历史数据见[实验指南](README.md)。历史实验结论不直接等同于旧基线 Q2.34 RTL 的精度证据。

本文档定义新实验 `widthselect`（独立 schema `mac-width-v1`，独立结果目录
`results_width/`）：在**本次重新生成**的合成输入上，选择 dot16 内部乘积对阶位宽
`W_align` 与累加器操作数位宽 `W_acc`，使长点积的最终 FP32 输出满足可配置的
bit match / ULP 精度要求。

**与历史实验的隔离（强制）**：正式入口不读取、不复用、不混入任何历史实验的
输入、原始结果或分析数据；没有 `source_run` 字段，配置中出现即报错。只允许复用
代码（`Dyadic`、`rne_shift`、`round_dyadic_to_fp32`、E4M3/E8M0 编解码、
MX 输入量化器、`fp32_ordered_index` 等）。新实验使用独立随机流命名空间
（`mac-width-fp32-stream-v1`），即使 seed/分布相同，也不会产生与历史 run 相同的
输入文件，从数据上保证“重新生成”。

## 0. 结论口径

- 主参考是 **O = RN32(S)**：S 由目标格式解码值精确 dyadic 求和得到，**不含**原始
  FP32 输入到目标格式的量化损失；实验测量的是 MAC 数值路径（对阶 / 累加）的误差。
- K/B 是压力维度，不是优化目标；恒有 **B = K/16**（dot16 块数）。
- 主指标为 FP32 32-bit 编码匹配比例与 ULP 距离；L2 只作辅助。
- 用户未给阈值时只报告结果，不自行宣布某个位宽“可接受”。位宽推荐只表述为
  “在已测试位宽、分布、K 与阈值下的最小通过候选”，不外推、不假设单调性。

## 1. 输入与参考值

### 1.1 输入格式

| 名称 | 定义 | scale | 分组 |
|---|---|---|---|
| `mx_fp8` | E4M3 元素（OCP MX，次正规保留，无 Inf） | E8M0（bias 127，0xFF 保留） | A 每行沿 K G32；W 每列沿 K G32 |
| `tf32` | 1+8+10 位，bias 127；**保留 FP32 指数范围与次正规** | 无 | 无（不分组） |

统一流水线（两种格式共享同一份本次新生成的 FP32 输入）：

```
按固定分布合成数值（6 种）
  → 每个数值经精确 dyadic 做一次显式 RNE 到 binary32（保存 bits）
  → 量化为目标输入格式（MXFP8 G32 / TF32 RNE）
  → MAC 实验
```

- 不再提供独立 FP8 编码采样入口；`matrix/sampling.py` 的 v1 方案不参与本实验。
- `M`、`N` 自由；`Kmax` 与每个 K 快照必须是 16 的倍数；含 `mx_fp8` 时必须是 32 的
  倍数（保证 G32 完整组）；`B = K/16` 恒成立。

### 1.2 六种固定参数分布

沿用历史四种固定参数生成器（逐字节复用代码，不重新定义、不改语义）：

| 名称 | 定义 | 理论均值 | 理论 RMS |
|---|---|---|---|
| `gaussian` | Box–Muller `sqrt(-2 ln u1) cos(2π u2)` | 0 | 1 |
| `laplace` | 逆 CDF，b = 1/√2 | 0 | 1 |
| `student_t3` | `Z / sqrt(G1²+G2²+G3²)`（等价 t3/√3） | 0 | 1 |
| `outlier` | 0.99·N(0,1) + 0.01·N(0,64)，除以 √1.63 | 0 | 1 |

新增两种固定参数分布，**直接定义在有限 binary32 编码集上**（非连续形状、非目标格式
编码）：

| 名称 | 定义 | 理论均值 | 理论 RMS |
|---|---|---|---|
| `uniform` | 所有有限 binary32 编码等概率：符号公平 + 31 位幅度字段在 `[0, 0x7F7FFFFF]` 均匀（含 ±0、次正规、正常值；排除指数字段 255） | 0 | 闭式值（量级 ~1e37，非单位 RMS） |
| `positive` | 同一编码均匀规则，符号固定 0 且幅度非零：**每个值都是正的有限 binary32 数**（含正次正规，无零） | 闭式值（~1e36） | 闭式值 |

两者已是精确 FP32 编码，因此统一流水线中的“显式 RNE 到 FP32”步骤对它们恒等（由
metadata 记录）；历史四种仍走“数值生成 → 精确整数 RNE”。两者不做事先归一化
（符合“不做逐矩阵或逐组强制归一化”）。

默认六种**同分布组合**：`gaussian|gaussian`、`laplace|laplace`、
`student_t3|student_t3`、`outlier|outlier`、`uniform|uniform`、`positive|positive`。
不做逐矩阵、逐组或逐 pair 强制归一化。任何自定义分布参数
（`activation_parameters` / `weight_parameters`）一律显式报错。

### 1.3 随机流

命名空间 `mac-width-fp32-stream-v1`，键包含
`seed | matrix | distribution_label | pair_index | row | col | purpose`：

- `matrix` ∈ {`x`,`w`}，与坐标一起完全区分每个元素 → 扩展 Kmax 或 pair 数量不改变
  任何已有元素（前缀稳定、pair 数无关）；
- 每个元素独立流（逐元素抽取与顺序无关），pair_index 区分同一 seed 的不同矩阵对；
- 不使用 Python 的进程随机 `hash()`；
- 与历史 `mac-mx-fp32-stream-v1` 键空间不同，新输入与历史数据不重合。

### 1.4 MXFP8 输入量化（沿用已验证策略）

覆盖组最大值的**最小二进制幂** scale（E8M0，越界记录 clamp）、元素 RNE ties-to-even、
次正规保留、饱和到 ±448、全零组规范 scale `0x7F`。A 按行沿 K 分组（axis=1）、
W 按列沿 K 分组（axis=0）。全部边界比较与舍入在精确 dyadic 上完成。

### 1.5 TF32 输入量化（本轮明确策略）

**保留 FP32 指数范围，有效数 RNE 到 11 位有效精度（1 隐含 + 10 显式尾数）**，
`不可用截断冒充`：

- 编码：`sign(1) | raw_exp(8, bias 127) | frac(10)`；正常值
  `±(2^10+frac)·2^(raw_exp−137)`；
- 次正规（`raw_exp=0, frac>0`）：`±frac·2^(−136)`（与 FP32 同构但尾数更窄）；
  零保留符号（`±0`）；
- 舍入：在精确 dyadic 上做一次 RNE ties-to-even；尾数进位时向指数进位；
- 溢出：RNE 后 `biased ≥ 255` 时按用途区分（两条路径都是显式策略，均记事件）：
  - **输入量化（本实验使用的策略）**：饱和到 `±` 最大有限 TF32（`raw_exp=254,
    frac=1023`），保证**每个记录在案的输入都有限**，精确 dyadic 引擎绝不会收到特殊码；
    事件 `element_overflow_saturated` 写入输入记录并由离线分析汇总；
  - 通用转换 `round_dyadic_to_tf32(..., overflow="inf")`：给出 `±Inf` 特殊码。
  该边界对有限 FP32 输入可达：舍入中点是 `2^128 − 2^116`，高于该点的 FP32 值（含 FP32
  最大有限值 `(2−2^-23)·2^127`）会触发上述策略；实现与测试覆盖该边界；
- 解码：`raw_exp=0` → `±frac·2^(−136)`；否则 `±(2^10+frac)·2^(raw_exp−137)`；`raw_exp=255` 拒绝。

**与历史 `formats/tf32.py` 的区别**：历史 `from_fp32_bits` 仅截断低位且 FTZ，是旧
dot-product 适配器；本轮是新的 RNE 量化器，历史模块不改动。策略标识
`fp32_significand_rne_11bit_kept_subnormal_input_overflow_saturated` 写入 manifest。

### 1.6 精确参考与主指标参考

对目标格式的精确解码值 $\widetilde A, \widetilde W$：

```
s_b      = Σ_{i=0..15} Ã[m,16b+i] · W̃[16b+i,n]      精确 dyadic，16 项和精确
S        = Σ_b s_b                                   精确 dyadic
L1       = Σ_k |Ã[m,k] · W̃[k,n]|                     精确 dyadic
O        = RN32(S)                                   主参考 FP32 编码
```

乘积、`s_b`、`S`、`L1` 全部任意精度精确；MX 的公共 scale 作为 dyadic 指数平移精确
施加（P_i 整数域求和后再施加，见 §3.1）。主指标不含 FP32 原始输入 → 目标格式的
量化损失；输出格式仅 FP32，本轮不做 MXFP8/BF16 输出量化。

## 2. W_align 严格定义（核对历史，并给出 TF32 推广）

历史定义在 `matrix/alignment.py`（schema `mac-mx-align-v1`）中固定为 E4M3 语义，
本轮**保留该定义与公式**，逐位核对如下（对 G32 组内一个 dot16、公共 scale 指数
`h = alpha + beta`）：

```
E = max(raw_exp, 1);  t = frac (raw_exp=0，次正规) 或 8+frac (raw_exp>0)
u = (-1)^sign · t · 2^(E-10)
P_i = (-1)^(sign_x⊕sign_w) · t_x · t_w ;  C_i = E_x + E_w
v_i = P_i · 2^(C_i - 20)        （20 = E4M3 乘积 LSB 偏移）
Cmax = max C_i（非零乘积）
n_i = RNE_int(P_i · 2^(W_align - 6 - (Cmax - C_i)))     （6 = 20 - 14）
N_b = Σ_i n_i（精确整数）;  q_b = N_b · 2^(Cmax - 14 - W_align + h)
```

对阶整数容器 `W_align + 7` 位有符号，逐项与整数和都做范围校验，越界抛错不 wrap。
`W_align` 与 `W_acc` 位宽计数方式不同，**禁止共用一个含义模糊的 `W` 字段**；
配置中分别使用 `W_align_list` / `W_acc_list`。

**格式参数化（本轮新增，用于 TF32 输入）**：对阶窗口以“乘积有效数宽度 p”为参数，
只改常数、不改策略：

```
k_shift = product_lsb_offset - window_offset = p - 2      (E4M3: 6, TF32: 20)
shift_i = (Cmax - C_i) + k_shift - W_align
q_b     = N_b · 2^(Cmax - window_offset - W_align + h)    (TF32 的 h = 0)
```

TF32 的元素分解沿用同一形式：`E = max(raw_exp,1)`、`t = frac`（次正规）或
`2^10+frac`，`u = t·2^(E-137)`，从而 `v_i = P_i·2^(C_i-274)`、`window_offset=254`、
`product_lsb_offset=274`、`k_shift=20`。容器上界与 E4M3 相同：单项
`|n_i| < 2^(W_align+2)`，16 项和 `< 2^(W_align+6)`，`W_align+7` 位有符号足够。
测试用独立 oracle 验证“E4M3 常数下与历史 `align_dot16` 完全一致”。

**MXFP8 公共 scale 在 dot16 整数求和后生效**：一个 G32 组仍对应两个独立 dot16，
各自选 `Cmax`、各自恢复 `q_b`，不合并为 dot32。

## 3. 实验一：扫描 W_align（本实现）

对每个输出元素、每个 dot16，按 §2 得到 `q_b(W_align)`；随后**精确累加**：

```
T(W) = Σ_b q_b(W)             精确 dyadic，绝不逐 b 转 FP32
Y_align(W) = RN32(T(W))       仅在请求的 K 快照处生成
```

- 禁止每次累加后舍入为 FP32；`T` 是唯一状态；
- 快照只“读”当前精确 `T` 并做一次 RN32，**不改变继续累加的精确状态**（测试验证：
  带中间快照的完整运行与只跑终点的运行，所有快照 `T` 完全相同）；
- 每个 K 快照保存：`S`、`L1`（与该宽度无关，单独记录一次）、`O_bits`、该宽度的
  `T` 与 `Y_bits`；
- 辅助数值 `T−S` 与 `decode(Y_align)−S` 由保存字段精确重建（离线分析），其中
  `decode` 是 FP32 编码的精确 dyadic 值（±0 保留符号）。

记录组织：每个 `(pair, K, format)` 一条 base 快照（`S/L1/O_bits/status`），每个
`(pair, K, format, W_align)` 一条宽度快照（`T/Y_bits/事件/status`），分析按
`(sample_id, K, format)` 连接，避免宽度维度重复存储 S/L1/O。

## 4. 实验二：扫描 W_acc（已确认并实现，`hwacc-v1`）

语义：dot16 用精确部分和 `s_b`（不引入 W_align 损失），累加状态使用自定义格式
`C_0 = 0`，`C_{b+1} = HWAdd_{W_acc}(C_b, s_b)`，仅在全部累加完成或 K 快照处
`Y_acc = RN32(decode(C_B))`；禁止每步转 FP32，禁止“精确相加后转 FP32”冒充。
位宽约定（用户给定，不可改）：两个整数加法器操作数各 `W_acc` 位有符号（含符号）；
完整加法结果 `W_acc+1` 位有符号（含符号）；状态指数字段 8 位；
**状态逻辑位宽 = W_acc + 9**（例：W_acc=30 → 39 位）。不额外重复符号位、不加入
隐含有效位，不把有符号整数有效数编码与 IEEE sign-magnitude 混为一谈。

### 4.1 数值格式（7 项，已由用户全部确认；实现见 `widthselect/acc_format.py`）

以下为**已确认的方案**，编号对应任务书 7 项；`widthselect/acc_format.py` 按此实现，
并用独立 Fraction oracle 与针对性语义测试覆盖。

**F1. 有符号有效数编码与解码公式（建议）**
- 状态有效数 `M`：`W_acc+1` 位**二补码**有符号整数，MSB 即符号位；无隐含位、
  无独立符号字段。加法器操作数为 `W_acc` 位二补码。
- 状态指数码 `E`：8 位无符号；`e = E − 126`。
- **解码：`value(M,E) = M × 2^(E − 126 − W_acc)`**（有效数 LSB 权重 = `2^(e−W_acc)`）。
- 选择二补码的理由：整数加减、进位与抵消语义直接；选择 `−W_acc` 偏移的理由：
  `e≈ceil(log2|value|)`，动态范围与 W_acc 解耦。
- 例（W_acc=30）：`E=126, M=2^29..2^30−1` → `|value|∈[2^-1,1)`；`E=254` → 上界
  `M·2^98 ≈ 2^128`，与 FP32 溢出阈值对齐；`E=0, M=1` → 最小非零 `2^-156`
  （低于 FP32 最小次正规 `2^-149`）。

**F2. 8 位指数的偏置、范围、保留码、次正规、零与溢出（建议）**
- 偏置 **126**：`E ∈ [0,254]` 为有限码，`e ∈ [−126,128]`；选择 126（而非 127）使
  最大可表示幅度 `≈2^128` 与 FP32 上溢边界重合，对任意 W_acc 成立。
- `E = 255` **保留为“溢出/无效”标记**，正常运算绝不产生；解码该状态在最终转换中
  按 `M` 的符号给出 `±Inf`，并单独累计 `acc_overflow` 事件。
- **次正规（渐进下溢）**：没有额外编码；`E=0`（`e=−126`）且 `|M| < 2^(W_acc−1)`
  的“非规格化有效数”即次正规状态，值小到 `2^(−126−W_acc)`，精度渐进降低；
  无需特殊标志位。
- **零**：`M=0` 规范零，规范码 `E=0`；状态格式只有一种零，**没有有符号零**
  （与 IEEE 不同；输出 `Y_acc` 的 ±0 差异只在最终 RN32 处按普通 FP32 统计）。
- **上溢**：规格化后需要的 `e > 128` → 置 `E=255` 保留标记（符号由 `M` 的 MSB
  保留），并累计事件；不饱和、不 wrap。溢出后继续累加仍在标记态传播（可选：直接
  拒绝该 pair，由配置 `nonfinite_policy` 决定）。

**F3. 公共指数选择（建议）**
- 对两个操作数值 `v`（状态精确 dyadic；incoming `s_b` 或 joint 的 `q_b` 精确
  dyadic）分别定义

  ```
  fit(v) = 最小整数 w，使 RNE(v / 2^w) ∈ [−2^(W_acc−1), 2^(W_acc−1)−1]
  w*     = max(fit(v_state), fit(v_in))
  ```

- `fit` 必须搜索到**最细合法窗口**：二补码窗口的负端比正端多一个可表示值
  `−2^(W_acc−1)`，因此取到首个可行窗口后还要检查再细一档是否仍合法（至多再进一档；
  实现与定向测试覆盖 `−2^k` 及其邻值、舍入中点）；
- 若某个 `RNE(v/2^w*)` 恰好越出正端 `+2^(W_acc−1)`（舍入边界），则 `w* += 1`
  重算（最多迭代 1 次，终止性可由整数界证明）；
- 两个操作数都按 `w*` 对齐：`n_i = RNE(v_i / 2^w*)`，落到 `W_acc` 位窗口内；
- `E_code` 编码 `w*`：`decode` 要求 `w* = e_state − W_acc`（步进状态时同步更新）。

**F4. 对阶窗口之外低位的处理（建议）**
- 统一采用 **RNE ties-to-even（对数值本身，符号对称）**：`round(m·2^d)` 到最近整数、
  平局取偶；负数不是算术右移的 floor，也不是向零截断，而是 `sign × RNE(|x|)`。
- 实现直接复用 `rne_shift`（对数值的 RNE，已有独立 oracle 测试）。
- **GRS 辅助**（可选实现）：对二补码算术右移的余数 `r = m mod 2^s ≥ 0`
  取 `G`=第 `s−1` 位、`R`=第 `s−2` 位、`S`=其余位 OR；当 `G & (R|S|lsb)` 时加 1。
  该规则与数值 RNE 严格等价（包含负数）。
- 本轮不使用截断模式；若后续需要，作为独立配置项并单独标注。

**F5. W_acc+1 位状态有效数进入 W_acc 位操作数窗口（建议）**
- 状态规格化后（`|M| ∈ [2^(W_acc−1), 2^W_acc]`）其操作数形式必然需要右移 ≥1 位进入
  `W_acc` 位窗口；右移由 `w*` 统一决定，**被 RNE 丢弃的状态 LSB 就是每次更新的
  信息损失点**（典型 1 位；若 `s_b` 更大迫使更粗窗口则更多；若 `s_b` 很小被吸收为 0，
  状态仍丢 1 位）；
- 状态保 `W_acc+1` 位相当于 **1 位 guard bit**：它参与本次加法，但在下一次进入
  操作数窗口时被舍入掉；不跨更新携带未计入宽度；
- **没有零检测旁路**：即使一个 dot16 的精确部分和 `s_b = 0`，状态仍按同一条
  `fit → w* → RNE` 路径进入 `W_acc` 位窗口（记录 `acc_input_zero` 事件）；因此
  “加精确零”和“加被吸收为零的极小非零数”走同一条损失路径，不产生未写入策略的
  寄存器保持特例（`width_hwacc_v1` 无该优化）；
- 若状态已非规格化（如抵消后的 `E>0` 小值），`fit` 可能精确落入窗口、丢失 0 位。

**F6. 加法后的进位、抵消与规格化（建议）**
- `N = n_1 + n_2`：`W_acc+1` 位二补码，**精确、无溢出**（两个 `W_acc` 位操作数之和
  必落在 `[−2^W_acc, 2^W_acc−2]`）；进位由第 `W_acc+1` 位（符号位）吸收，不 wrap；
- 抵消允许；规格化：整数左移直到最高两位不同（无冗余符号位），同时 `e` 递减
  （`E` 码减小）；左移无损，且**不得声称恢复任何已丢弃位**；
- 若 `E` 降到 0（`e=−126`）仍有冗余符号位，则保留非规格化状态（渐进下溢），停止左移；
- `N=0` → 规范零；
- 加法本身不再舍入（舍入只发生在 F4/F5 的操作数对齐）；规格化后检查 `e>128` →
  F2 上溢路径。

**F7. GRS 的存储属性（建议）**
- GRS 只作为**一次 HWAdd 内的组合逻辑辅助位**（移位网络中产生、舍入后即丢弃），
  **不是持久状态**；状态逻辑宽度严格保持 `W_acc+9`，不增加任何未计入的隐藏寄存器位。

**硬件路径**（显式暴露、诊断工具可见）：
`拆解(状态/输入) → 选公共指数 w* → 有限窗口对阶(RNE/GRS) → 整数加减 →
规格化(左移/E 递减) → 状态编码`。Python 大整数只承载位向量，每一步有明确宽度约束；
不保存精确残差到后续更新。

### 4.2 实验二输出与比较（已实现）

- 每个 K 快照保存自定义状态字段（`E_code`、`M`（十进制字符串）、`W_acc`）、
  精确解码值 `C_exact` dyadic 与 `Y_acc = RN32(decode(C_B))` bits；
- 主比较 `Y_acc vs O`（bit match + ULP）；辅助 `decode(C_B)−S`（有限状态；
  溢出标记单独计数）、`decode(Y_acc)−S`；
- `E=255` 标记单独统计（`state_marker_count`）；不把非有限静默排除；
- 累加事件（cumulative，不跨 K 相加）：`acc_updates`、`acc_input_zero`、
  `acc_input_absorbed`、`acc_state_bits_dropped`、`acc_input_bits_dropped`、
  `acc_carry`、`acc_cancellation`、`acc_overflow`、`acc_subnormal`、
  `acc_marker_propagated`。

## 5. 联合验证（可选，已实现）

两项隔离实验筛出候选后，显式配置少量组合（`joint_pairs`）：

```
C_{b+1} = HWAdd_{W_acc}(C_b, q_b(W_align))
```

最终 `RN32(decode(C_B))` 与 O 比较；每个组合单独记录状态/解码值/事件；
**单独合格不代表组合合格**，联合结果单独报告，不并入实验一/二的主表。

## 6. 指标与分析口径

每个 `(format, distribution, 位宽, K/B)` 独立统计；独立重复单位是**矩阵对**，
不把元素当独立 Bernoulli 样本、不给出 3/N 零事件上界。

主指标（与 O 比较）：

- 完整 32-bit 编码匹配数 `n_bit_match`、总分母 `n_total`（含非有限）、有效（双方有限）
  分母 `n_finite_pairs` 与匹配比例；
- 有限对的 ULP 分桶：0、1、2、3–4、≥5（分母 = `n_finite_pairs`，离散整数，不做插值）；
- 有限且 bit 不匹配样本的条件 ULP p50/p90/p99/max（nearest-rank）；无条件样本 → `null`
  并带 `no_mismatch_samples` 状态；
- 非有限计数：`reference_finite_candidate_nonfinite`、`reference_nonfinite_candidate_finite`、
  双方 ±Inf 同号/异号、`any_nan`、`acc_overflow`（实验二）；
- `signed_zero_mismatch`：双方有限、数值 ULP 距离 0 但编码不同（±0）——bit match 区分
  ±0，ULP 距离合并 ±0。

辅助数值（明确分母与有效样本范围）：

- `abs_error = |decode(Y) − S|`：有限候选样本上的 p50/p90/p99/max 与均值；
- `etaL1 = |decode(Y) − S| / L1`：仅 `L1 ≠ 0` 的有限候选样本，`L1=0` 记为
  `undefined_zero_l1`；
- `relative_l2 = sqrt(Σ err² / Σ S²)`：有限候选样本的精确平方和之比（分母 = 这些
  元素的 `Σ S²`，不静默替换）；
- 非有限候选不入 aux 分母，但计入主指标分母并单独计数。

阈值（可选，`pass_criteria`）：`min_bit_match_rate`、`max_ulp_p99`、`max_ulp_max`、
`nonfinite_policy ∈ {reject, ignore}`。未配置时分析只报告，不输出推荐；配置时输出
每个 `(format, distribution, K)` 的 `minimal_passing_candidate`（最小通过位宽），
并附“仅在已测试位宽、分布、K 与阈值内成立”的限定说明。不假设匹配率随位宽单调。

主图（`width-analyze --plots`，输出 `<analysis>/plots/`；matplotlib 可选依赖）：

- **每个合成分布一张图**，文件名 `<分布>_metrics.png`（同分布组合取单名，如
  `gaussian_metrics.png`；交叉组合如 `gaussian_laplace_metrics.png`）；
- 图内网格：**行 = 指标**（bit match 比例、条件 ULP p99、条件 ULP max），**列 = 输入
  格式**（`mx_fp8` / `tf32`；单格式 run 自动退化为一列）；
- **横轴 = `B = K/16`**（log2 轴，刻度为实际 B 值），顶部副轴标注对应 `K = 16B`；
- **每条曲线一个位宽**：align 图为 `W_align`，acc 图为 `W_acc`，联合图为
  `(W_align, W_acc)` 组合；曲线按位宽升序，图例在图底部；
- 空条件（无 bit 不匹配样本）的 ULP 点**画在 y=0 并用空心 marker 标出**（绘图约定，
  保证每位宽曲线连续），面板标注 `hollow = no mismatch samples (drawn at 0)` 并给出
  有限对/非有限计数；`summary.json`/CSV 中这些分位数仍为 `null`，不写 0；
  bit match 曲线若其输出**全部非有限**（匹配来自 `±Inf == ±Inf`）则画虚线并在面板中
  点名，避免把 1.0 误读为有限精度全对。

## 7. 数据、追踪与复现

运行目录 `results_width/<run_id>/`（schema `mac-width-v1`，run_id 前缀
`width_align_` / `width_acc_` / `width_joint_`）：

- `manifest.json`：schema、run_kind、mode、config + config_hash、源码 digest、
  `M/N/K_list/B_list/W_align_list/W_acc_list/formats/distribution_pairs/seeds/pairs_per_seed`、
  全部数值策略标识（输入、TF32、MX、对阶、参考定义）、`pass_criteria`、expected/counts、
  `status`；
- `inputs.jsonl.gz`：每个矩阵对**本次生成**的 `x_fp32_bits`/`w_fp32_bits`（两格式共享）
  与各格式编码（MX 元素/scale 码 + 指纹；TF32 codes + 指纹）、输入事件/诊断；
- `samples.jsonl.gz`：实验一的 base 快照（`S/L1/O_bits`）与候选快照（`T/Y_bits/事件`）；
  实验二候选快照（`C_state/C_exact/Y_bits/事件`）；联合候选快照（`w_align/w_acc/
  C_state/C_exact/Y_bits/事件`）；
- 不默认保存全部更新轨迹；`width-replay` 按 pair、输出坐标、K、位宽小规模重放，
  导出实验一的对阶位移/被丢弃位/整数操作数/加法结果/精确 T，以及实验二/联合的
  每次更新：状态前值、两操作数、丢位标志、整数和、规格化位移、指数与状态后值。

离线分析输出 `results_width_analysis/<run_id>_analysis_<时间戳>/`：
`summary.json`、`summary.csv`、`analysis_manifest.json`（含源文件 sha256/size）、
逐 pair 指标（`by_pair`）与 cell 指标（由逐 pair 精确平方和合并，不是比值平均）。
分析只读源 run，不导入执行引擎/采样器。

## 8. 测试与交付

重点测试（见 `tests/legacy/test_widthselect_*.py`）：

- 新随机流：复现、pair 区分、前缀稳定、扩展 pair 数/坐标不改变已有输入；与历史
  MX 流不重合（不同命名空间抽样若干坐标即可）；
- 两种格式共享同一份新 FP32 输入；MX 量化与既有量化器逐位一致；
- TF32 RNE：独立 Fraction oracle 穷举/随机比对（进位、次正规、溢出、±0、平局取偶）；
- 实验一：逐 block `q_b` 与历史 `align_dot16` 在 E4M3 常数下完全一致（随机 oracle）；
  `T` 精确、无逐步 FP32 舍入、快照不扰动状态；前缀运行 = 独立短运行；
- 小位宽穷举 oracle（独立 Fraction）；
- 指标：bit match、ULP 分桶/条件分位数、空条件为 null、非有限/±0 统计正确；
- 新结果与历史严格隔离（schema 拒绝、无 source_run、目录分离）。

交付：设计文档（本文）、实现、测试、smoke 配置、正式配置模板、运行/分析命令。

## 9. 计算规模估算

单对（M×N 输出、Kmax、|W_align_list| = nw）的精确工作量：
`M·N·(Kmax/16)·16·(1 + nw)` 次乘积/对阶项操作（S/L1 一次 + 每个宽度一次对阶），
外加每个 block 的精确 dyadic 求和的分配开销。

- smoke（M=2,N=32,K≤64,nw=2）：数千次项操作，秒级（实测实验一 0.6 s、分析 0.13 s，
  实验二/联合同量级）；
- 模板（M=8,N=32,K≤4096,nw=4，12 对）：`8·32·256·16·5 ≈ 5.2M` 项操作/对 → 十余秒
  每对（Python 精确算术经验值，参考历史实测 ~100 s/对@16×64×16384）；实验二的
  每次 HWAdd 含大整数对齐/规格化，单次成本与一个对阶段相当量级；
- 若放大到历史正式规模（16×64×16384、40 对、nw=5），单对约 5–10 分钟量级，
  总计数小时；本轮不自动启动正式扫描。

## 10. 运行 / 分析 / 重放命令

```bash
# 实验一 smoke（fresh 输入，含 mx_fp8 + tf32 两个格式）
.venv/bin/python -m mac_experiment width-run --config configs/widthselect_smoke.yaml

# 实验二 smoke（W_acc 扫描）
.venv/bin/python -m mac_experiment width-run --config configs/widthselect_acc_smoke.yaml

# 联合 smoke（仅显式列出的组合）
.venv/bin/python -m mac_experiment width-run --config configs/widthselect_joint_smoke.yaml

# 离线分析（bit match / ULP / 辅助误差 / 阈值推荐）
.venv/bin/python -m mac_experiment width-analyze --run results_width/widthselect_smoke
.venv/bin/python -m mac_experiment width-analyze --run results_width/widthselect_acc_smoke
# 加图：每个分布一张 <分布>_metrics.png（行=指标，列=格式，横轴 B/K，曲线=位宽）
.venv/bin/python -m mac_experiment width-analyze --run results_width/widthselect_smoke --plots

# 诊断重放：某个 pair、输出坐标、K、位宽
.venv/bin/python -m mac_experiment width-replay --run results_width/widthselect_smoke \
    --sample-id 'width|gaussian|gaussian|seed1|00000' --row 0 --col 0 --K 32 --w-align 21
.venv/bin/python -m mac_experiment width-replay --run results_width/widthselect_acc_smoke \
    --sample-id 'width|gaussian|gaussian|seed1|00000' --row 0 --col 0 --K 64 --w-acc 16 --format tf32
.venv/bin/python -m mac_experiment width-replay --run results_width/widthselect_joint_smoke \
    --sample-id 'width|gaussian|gaussian|seed1|00000' --row 0 --col 0 --K 32 \
    --w-align 21 --w-acc 16 --format tf32

# 正式模板（均未执行；一文件一 mode，按需分别运行，保持两项实验隔离）
# .venv/bin/python -m mac_experiment width-run --config configs/widthselect_align_formal_template.yaml
# .venv/bin/python -m mac_experiment width-run --config configs/widthselect_acc_formal_template.yaml
# .venv/bin/python -m mac_experiment width-run --config configs/widthselect_joint_formal_template.yaml

# 一键批量运行三个隔离实验（align → acc → joint，各自 run + analyze，带进度条）
.venv/bin/python scripts/run_widthselect_all.py                    # smoke 预设（秒级）
.venv/bin/python scripts/run_widthselect_all.py --preset formal    # 三份正式模板（未执行）
.venv/bin/python scripts/run_widthselect_all.py --overwrite        # 重跑：先删配置 run_name 对应的 run 目录
.venv/bin/python scripts/run_widthselect_all.py --skip-analysis --plain   # 只跑实验 / 纯文本进度
.venv/bin/python scripts/run_widthselect_all.py --plots             # 分析同时出图
.venv/bin/python scripts/run_widthselect_all.py \
    --align-config configs/widthselect_smoke.yaml \
    --acc-config configs/widthselect_acc_smoke.yaml \
    --joint-config configs/widthselect_joint_smoke.yaml
```

批处理脚本只做编排：三个实验仍是独立 run/schema/结果目录，输入各自重新生成；
`--overwrite` 仅删除配置 `run_name` 指向且 `manifest.json` 为 `mac-width-v1` 的目录，
分析输出写到 `<output_root>/../results_width_analysis/<run_id>_analysis_<时间戳>/`。

## 11. 与历史实验的隔离与不变性

- 只新增独立模块 `src/mac_experiment/widthselect/`；`matrix/engine.py`、
  `matrix/alignment.py`、`matrix/inputs.py`、`results/schema.py`（除新增 hash 函数）、
  `formats/*` 的既有语义一律不改；
- 历史 schema（`mac-exp-*`、`mac-mx-exp-v2`、`mac-mx-align-v1`）的 run 不能被本
  实验入口读取；本实验 schema 也不能被历史分析器读取；
- 新增的 `uniform` / `positive` 只存在于本实验的分布表，历史配置的分布集合不变。

## 12. 确认与实现状态

1. **W_acc 数值格式 7 项**（§4.1 F1–F7）：已由用户全部确认；
   `widthselect/acc_format.py` 已实现（`WIDTHSELECT_ACC_FORMAT_STATUS =
   "hwacc_v1_confirmed_implemented"`），独立 Fraction oracle 与针对性语义测试
   （零/大指数差/抵消/进位/正负边界/低位丢弃/指数边界/无隐藏精度）通过；
2. `uniform` / `positive` 的定义：已在用户澄清后修正为有限 FP32 编码域均匀
   （`uniform` 覆盖全部有限编码；`positive` 为符号固定 0、幅度非零的正有限编码），
   已实现并测试；理论矩用闭式精确计算（`uniform` 的分母为幅度编码数，不再多乘 2）；
3. TF32 次正规 `frac·2^(−136)`；输入量化溢出**饱和**到 ±最大有限 TF32 并记
   `element_overflow_saturated`（通用转换仍保留 ±Inf），已实现并测试；
4. 审查修复（已实现并加入定向测试）：
   - `fit` 取最细合法窗口（二补码负端多一个可表示值），`W_acc=8, −257` → 步长 2、
     `−128`；序列 `2, −257` 得 `−254`；
   - 累加器**无零检测旁路**：`s_b=0` 仍走状态→操作数窗口的 RNE 窄化，策略与硬件
     假设写入 F5/`acc_format.metadata()`；
   - `width-analyze --plots` 输出主图：每个合成分布一张 `<分布>_metrics.png`，
     网格行=指标（bit match / 条件 ULP p99 / 条件 ULP max）、列=格式，横轴 `B=K/16`
     （log2，顶部副轴标 K），每条曲线一个位宽/组合；
   - 图面标注增强：空条件 ULP 点画在 0（空心 marker，曲线连续）并标注
     `no mismatch samples (drawn at 0)` + 有限/非有限计数（数据层仍为 `null`）；
     bit match 曲线全部非有限时画虚线并点名（`uniform`/`positive` 即此情形：
     编码域均匀导致精确结果远超 FP32 有限范围，`O/Y` 均为 ±Inf）。

三个 mode 均已可运行；本轮只执行 smoke（实验一/二/联合），正式扫描未启动。
