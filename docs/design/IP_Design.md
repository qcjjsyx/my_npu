# 多精度融合 MAC IP 电路架构设计

> 适用范围：本文描述 2026-10-31 及以前的旧 SystemVerilog dot16 基线，包括 Q2.34 累加、FTZ/Inf 写回、MX 输入指数 1…254 边界、dot16 归约与 26-bit 输入容器。它是旧基线的设计合同，不是 2026-11-05 新架构的合同；新架构已确认约束与未冻结项见 [decisions.md](../project/decisions.md)。新 Verilog 实现不默认沿用本文的数值语义、接口或流水结构。

旧基线版本已将 ALIGN、PSUM、MX 和 ACC RTL 迁移到 Q2.34/bias127，并以独立任意精度参考模型验证。其接口和实现边界见§7.9、§8.8、§9.12；本轮运行证据见 hardware/verification/README.md，不构成综合或时序证据。

## 1. 多精度融合方案

### 1.1 支持的数值格式与计算模式

本 IP 面向 Tensor Core/NPU 多精度矩阵乘加运算，对外支持 FP16、BF16、FP8 E4M3、FP8 E5M2、INT16、INT8 和 INT4。E4M3 固定采用无 Inf、最大有限值为 448 的 E4M3FN 编码，不支持其他 E4M3 变体。

SRAM 保存原始紧凑编码。数据读出后，在 MAC 内部入口先按源格式处理特殊值，再将 FP16/BF16 转换为 FP19 内部表示，将 E4M3/E5M2 转换为 FP9 内部表示，随后构造有限补码操作数并广播或分发至 16 个物理乘法前端。

乘法前端实际支持五种计算模式：

| 计算模式 | 输入来源 | 每个操作数的计算 payload | 单前端乘法吞吐 |
|---|---|---|---|
| FP19 | FP16、BF16 | 1 个 signed12 尾数 | 1 个完整乘积/拍 |
| FP9 | E4M3、E5M2 | 2 个 signed5 尾数 | 2 个完整乘积/拍 |
| INT16 | INT16 | 1 个 signed16 整数 | 2 项部分积/拍，2 拍完成四项部分积 |
| INT8 | INT8 | 2 个 signed8 整数 | 2 个完整乘积/拍 |
| INT4 | INT4 | 2 个 signed4 整数 | 2 个完整乘积/拍 |

FP19、FP9 是自定义内部表示，不新增独立的外部输入格式。整数采用对称量化，合法输入范围为：

~~~text
INT16：[-32767, 32767]
INT8 ：[-127, 127]
INT4 ：[-7, 7]
~~~

`16'h8000`、`8'h80` 和 `4'h8` 不属于合法量化输入，必须由系统输入合同规定其拒绝、钳位或转换行为，不能依赖乘法器静默处理。

16 个物理乘法前端在无停顿条件下提供：

~~~text
FP19：每拍形成 1×dot16 的乘积集合
FP9 ：每拍形成 2×dot16 的乘积集合
INT8：每拍形成 2×dot16 的乘积集合
INT4：每拍形成 2×dot16 的乘积集合
INT16：每两拍形成 1×dot16 的完整乘积贡献
~~~

上述是乘法资源吞吐，不等同于已经保证相同的 ACC 连续更新吞吐。最终吞吐还取决于累加反馈、旁路、停顿和事务调度。

### 1.2 物理乘法资源与模式映射

每个物理乘法前端固定包含两颗有符号乘法器：

| 物理资源 | 输入端口 | 完整物理输出 |
|---|---|---|
| MUL12 | signed12 × signed12 | signed24 |
| MUL9 | signed9 × signed9 | signed18 |

完整物理乘积与数值有效 payload 必须区分。合法操作数下的映射及数学最小位宽如下：

| 模式 | MUL12 | MUL9 | 单乘积最大绝对值 | 单乘积数学最小 signed 位宽 |
|---|---|---|---:|---:|
| FP19 | 单个 FP19 尾数乘法 | 隔离 | 4190209 | 23 |
| FP9 | lane0 | lane1 | 225 | 9 |
| INT8 | lane0 | lane1 | 16129 | 15 |
| INT4 | lane0 | lane1 | 49 | 7 |

FP19 尾数最大绝对值为 2047，FP9 为 15。INT8、INT4 分别受对称量化范围限制。RTL 首版保留 MUL12 的 signed24 输出和 MUL9 的 signed18 输出；若后级缩窄 payload，必须检查被删除高位全部等于保留结果的符号位，不执行数值截断、舍入或饱和。

FP19 的 signed12 操作数直接进入 MUL12。FP9、INT8、INT4 的各 lane 显式符号扩展到对应物理端口宽度。FP19 模式下 MUL9 输入执行 operand isolation，其独立结果寄存器停止无效更新。

双 lane 模式中，lane0 归属 group0，lane1 归属 group1。两个乘积不在乘法前端局部相加，不执行宽乘积重构。16 个前端的同名 lane 分别形成两个独立 dot16。

INT16 的两拍映射见 1.5 节；其两条乘积输出是同一个 output group 的不同部分积，不得按双输出模式解释。

### 1.3 浮点尾数的补码表示

当前基线只采用有符号乘法器。合法整数直接使用补码，浮点数则在 MAC 内部入口将符号与非负尾数幅值转换为显式补码。

浮点尾数按以下顺序构造：

~~~text
先形成非负幅值：{hidden, fraction}
再在最高位补 0，形成正的有符号数
最后根据原始 sign 进行条件取负
~~~

FP19 的非负幅值为 unsigned11，补码尾数为 signed12；FP9 的非负幅值为 unsigned4，补码尾数为 signed5。

~~~systemverilog
logic        fp19_sign, fp9_sign;
logic [10:0] fp19_mag_u11;
logic [3:0]  fp9_mag_u4;
logic signed [11:0] fp19_mag_pos_s12, fp19_sig_s12;
logic signed [4:0]  fp9_mag_pos_s5, fp9_sig_s5;

always_comb begin
    fp19_mag_pos_s12 = $signed({1'b0, fp19_mag_u11});
    fp19_sig_s12 = fp19_sign ? -fp19_mag_pos_s12 : fp19_mag_pos_s12;
    fp9_mag_pos_s5 = $signed({1'b0, fp9_mag_u4});
    fp9_sig_s5 = fp9_sign ? -fp9_mag_pos_s5 : fp9_mag_pos_s5;
end
~~~

这些 signed 位宽已包含补码符号位及隐藏位对应的信息，不包含额外的舍入 Guard bit。负数的位模式不是 sign、hidden、fraction 的直接拼接。条件取负在上述合法幅值范围内精确，不需要额外扩位或饱和。

乘法前端只接收有限补码 payload，不重复执行浮点符号到补码的转换。乘积可以直接进入补码对阶与归约通路。入口条件取负、模式 mux 和广播网络的时序代价需要一起评估。

### 1.4 FP19/FP9 格式转换与源格式信息

#### 1.4.1 FP19 内部表示

FP19 字段容器为 1-bit sign、8-bit exponent、10-bit fraction。对已处理为有限数的输入进行以下精确扩展：

~~~text
FP16 → FP19：
    sign19 = sign16
    exp19  = {3'b000, exp16[4:0]}
    frac19 = frac16[9:0]

BF16 → FP19：
    sign19 = sign_bf16
    exp19  = exp_bf16[7:0]
    frac19 = {frac_bf16[6:0], 3'b000}
~~~

FP16 不在入口加 112，不统一到 BF16 的 bias。BF16 尾数整数扩大 2^3，同时 fraction 位数从 7 增至 10，数值保持不变。

#### 1.4.2 FP9 内部表示

FP9 字段容器为 1-bit sign、5-bit exponent、3-bit fraction。有限输入扩展规则为：

~~~text
E4M3 → FP9：
    sign9 = sign_e4m3
    exp9  = {1'b0, exp_e4m3[3:0]}
    frac9 = frac_e4m3[2:0]

E5M2 → FP9：
    sign9 = sign_e5m2
    exp9  = exp_e5m2[4:0]
    frac9 = {frac_e5m2[1:0], 1'b0}
~~~

上述转换保留源格式 bias，不进行舍入。FP9 虽具有 E5M3 字段宽度，但不是具有单一固定 bias 的通用外部浮点格式。

FP19/FP9 字段容器与乘法前端 payload 是不同表示。字段容器的 exponent 保留原始编码；乘法 payload 使用有效指数 Eeff 和补码尾数。特别是 subnormal 在字段容器中 exponent 为 0，而在计算 payload 中 Eeff 为 1。

#### 1.4.3 支持的源格式组合

每个事务固定 A/B 源格式组合，归约组内不混合不同组合。支持的浮点组合如下：

| A 源格式 | B 源格式 | 计算模式 | bias_A | bias_B | bias_sum |
|---|---|---|---:|---:|---:|
| FP16 | FP16 | FP19 | 15 | 15 | 30 |
| BF16 | BF16 | FP19 | 127 | 127 | 254 |
| E4M3 | E4M3 | FP9 | 7 | 7 | 14 |
| E5M2 | E5M2 | FP9 | 15 | 15 | 30 |
| E4M3 | E5M2 | FP9 | 7 | 15 | 22 |
| E5M2 | E4M3 | FP9 | 15 | 7 | 22 |

FP16×BF16 不支持。整数仅支持同格式相乘。混合 FP8 仍按 A/B 各自源格式解码，再共同使用 FP9 尾数乘法。

入口保留 `src_fmt_A/src_fmt_B`，后续随事务传递源格式信息或等价的 bias 信息。若采用 `bias_sum`，支持集合为 {14, 22, 30, 254}，数值字段使用 unsigned8 足够；其生成必须先扩位相加，不能使用窄源指数位宽相加后再赋值。

MX 统一进行 bias 修正。同一组各乘积的 bias_sum 恒定，因而 ALIGN 可以直接比较有效指数和。运行中修改全局格式配置不能改变已进入流水的事务解释。

### 1.5 INT16 两拍部分积输出与累加合同

#### 1.5.1 输入拆分及物理映射

任意合法 signed16 输入满足：

~~~text
A_L = unsigned8(A[7:0])
A_H = signed8(A[15:8])
A   = A_L + 2^8 × A_H
~~~

低片进入有符号乘法器前必须零扩展成正的 signed9，高片按 signed8 解释后符号扩展。B 使用相同规则。

~~~systemverilog
logic signed [15:0] a_s16, b_s16;
logic signed [8:0] a_l_s9, b_l_s9;
logic signed [7:0] a_h_s8, b_h_s8;

always_comb begin
    a_l_s9 = $signed({1'b0, a_s16[7:0]});
    b_l_s9 = $signed({1'b0, b_s16[7:0]});
    a_h_s8 = $signed(a_s16[15:8]);
    b_h_s8 = $signed(b_s16[15:8]);
end
~~~

| phase | MUL12 输出 | MUL9 输出 | PSUM 去向 |
|---|---|---|---|
| 0 | A_L×B_L | A_H×B_L | 分别进入低树、高树 |
| 1 | A_L×B_H | A_H×B_H | 分别进入低树、高树 |

MUL12 输入显式扩展到 signed12；MUL9 输入显式扩展到 signed9。低片的最高有效符号位是补入的 0，高片扩展复制其原始符号位。

| 部分积 | 数值范围 | 数学最小表示 | 首版物理结果 |
|---|---:|---|---|
| A_L×B_L | 0～65025 | unsigned16 或 signed17 | MUL12 signed24 |
| A_L×B_H | −32640～32385 | signed16 | MUL12 signed24 |
| A_H×B_L | −32640～32385 | signed16 | MUL9 signed18 |
| A_H×B_H | −16256～16384 | signed16 | MUL9 signed18 |

M 级直接输出这两项部分积，不在前端计算 T0、T1，也不重构完整 signed32 乘积。两拍绑定同一输入向量、tag 和 ACC0 目标；第一拍接受后，尚未使用的输入分片必须保持至第二拍被接受。phase 及数值必须共同响应停顿。

#### 1.5.2 两树归约和跨 limb 重构

16 个前端对应部分积分别进入两颗树：

$$
L_0=\sum_{i=0}^{15}A_{L,i}B_{L,i},\qquad
H_0=\sum_{i=0}^{15}A_{H,i}B_{L,i}
$$

$$
L_1=\sum_{i=0}^{15}A_{L,i}B_{H,i},\qquad
H_1=\sum_{i=0}^{15}A_{H,i}B_{H,i}
$$

PSUM 两颗树独立归约，形成内部 signed28 组合结果，并在 PSUM 内显式重构：

$$
T_0=L_0+(H_0\ll8),\qquad T_1=L_1+(H_1\ll8)
$$

进入 ACC 的两拍贡献分别为 T0 和 T1×2^8。因此完整 dot16 为：

$$
D=T_0+(T_1\ll8)
$$

两颗树之间没有隐式 carry；跨 limb 加法发生在 PSUM 内的 merge CPA。INT16 的两树属于同一 group，不执行独立的双输出累加。

| 节点 | 数学最小表示 | 物理接口或安全实现 |
|---|---|---|
| L0 | unsigned20 或 signed21 | 低树 signed28，低 21 bit 为 signed payload |
| H0、L1、H1 | signed20 | 对应树 signed28，低 20 bit 为 signed payload |
| T0 | signed28 | PSUM 重构结果 |
| T1 | signed27 | PSUM 重构结果，尚未施加最终 2^8 权重 |
| T1×2^8 | signed35 | 统一扩展至 signed36 后送 ACC |
| 完整 D | signed35 | 数学结果，不是 INT32 架构状态 |

PSUM 若直接对两个 signed28 内部 tree 输出进行重构，先分别符号扩展至 signed37，再将高树结果左移 8 位并相加；signed37 足以覆盖任意 signed28 接口值的该表达式。按合法 payload 缩窄实现时必须保留数值范围约束和符号冗余检查。T0/T1 缩窄只删除冗余符号位，不进行饱和或舍入。

#### 1.5.3 signed36 工作状态与最终 INT32 饱和

INT16 架构状态为 INT32，但两拍运算的加法器和跨拍工作状态均使用 signed36：

$$
C_{mid}=\operatorname{sext}_{36}(C_{old})+\operatorname{sext}_{36}(T_0)
$$

$$
C_{wide}=C_{mid}+\left(\operatorname{sext}_{36}(T_1)\ll8\right)
$$

$$
C_{new}=\operatorname{sat}_{INT32}(C_{wide})
$$

phase0 保存完整中间和，不截断、不饱和，不提交完整事务结果。phase1 完成后统一饱和到 [-2^31, 2^31−1]。输入 INT16 的对称量化范围不改变 INT32 结果的饱和边界。

整数 ACC0可使用一个 signed36 工作寄存器：phase0 写入 C\_mid；phase1 将饱和后的 INT32 结果符号扩展后写回。该寄存器在 phase0 后只表示事务中间状态，输出结果有效信号只能在 phase1 完成后提交。不得在两拍之间让其他事务覆盖该工作状态。

饱和判断在 signed36 域进行：大于 2147483647 时输出 `32'h7fffffff`，小于 −2147483648 时输出 `32'h80000000`，否则取低 32 bit；后一分支的高四位必须为 bit31 的符号扩展。第二拍位移在 PSUM 内先扩展至 signed36，再左移 8 位，禁止对 signed27 的 T1 直接移位后依赖赋值扩展。公式表示该贡献的数值权重；MX 和 ACC 不重复移位。

仅扩大加法器、不扩大跨拍保存路径，不能满足本合同。phase1 必须使用已包含 phase0 贡献的工作状态，ACC 反馈旁路或停顿机制在后续章节展开。

### 1.6 当前架构约束汇总

计算流水固定为：

~~~text
M → ALIGN → PSUM → MX → ACC
~~~

PSUM 固定包含两颗物理 limb tree，各自产生 signed28 内部组合结果。PSUM 最终只有一个 56-bit 打包输出寄存器：DUAL_OUTPUT 的 [27:0]/[55:28] 分别保存 group0/group1；WIDE_SUM 先在 PSUM 内完成重构，再将完整 signed36 结果符号扩展到56位（旧37位视图的bit36为冗余符号）。两种打包方式由随事务传递的模式区分，不得直接拼接原始 limb 代替重构。当前未切拍参考结构在内部tree输出与最终输出寄存器之间不另加寄存器；物理实现可按4.8节评审增加内部流水，不改变PSUM最终56-bit打包接口和重构归属。

本设计的 Q 格式整数位数包含符号位。浮点尾数绝对值小于 2，乘积绝对值小于 4，16 项和的绝对值小于 64，因此采用 1-bit 符号、2-bit 乘积整数位及 4-bit 归约增长位。归约增长位不等同于舍入 Guard bit。

| 模式 | PSUM 组织 | 数值结果与有效位段 | ACC 状态 |
|---|---|---|---|
| INT4 | DUAL_OUTPUT，每组一树 | 每树低 [11:0] 为 signed12 接口 payload；数学最小 signed11，bit11 为冗余符号位 | ACC0、ACC1 独立 |
| INT8 | DUAL_OUTPUT，每组一树 | 每树低 [18:0] 为 signed19 payload | ACC0、ACC1 独立 |
| FP9 | DUAL_OUTPUT，每组一树 | 每树全部 [27:0] 为 signed28 Q7.21 | ACC0、ACC1 独立 |
| INT16 | WIDE_SUM，两拍 | 两树每拍重构，输出 T0 或 T1×2^8；完整 dot16 为 signed35 | 仅 ACC0 |
| FP19 | WIDE_SUM | 两树重构 signed36 Q7.29，非 FP32 编码 | 仅 ACC0 |

INT4 的 dot16 最大绝对值为 784，INT8 为 258064。整数 tree 输出的其余高位均为 payload 符号扩展，不是无定义保留位。

FP19 的 29-bit 小数精度对应单项对阶数数学最小 signed32、归约结果 signed36；FP9 的 21-bit 小数精度对应单项数学最小 signed24、归约结果 signed28。ALIGN 每前端有两套并行电路，末端 lane0 寄存器为 signed32、lane1 为 signed24。FP9 lane0 仅符号扩展，不改变 21 位小数；FP19 拆片在 PSUM 内进行。

不同 output group 不得共享数据、carry、scale metadata、tag 或累加状态；公共格式配置不改变各组事务信息独立有效的要求。单输出模式下 ACC1 必须支持时钟门控或 operand isolation。

浮点及 MXINT8 输入/状态统一 signed36 Q2.34 + 8-bit bias127 E，value=M×2^(E−161)。ACC 保留对阶 tail，完整37位和经最终 RNE/规格化写回；ACC 写回下溢 FTZ、上溢保留有符号 Inf，MX 输入指数越界暂缓处理。整数仍为 INT32。完整长点积后再转换 FP32；MX 精确重定标，不重复 PSUM 重构或 INT16 权重。

## 2. 非规格化数与特殊值识别处理

本章定义 SRAM 原始紧凑编码到乘法前端有限补码 payload 的转换。输入 NaN/Inf 必须在普通乘法和指数比较之前处理；长点积结束后的输出转换器执行目标格式 Zero/Subnormal/Normal/Inf 分类，不能替代入口转换。ACC 有限计算溢出后保存逻辑有符号 Inf，下溢写规范零，见§9.3。

### 2.1 原始浮点输入分类

分类必须按 A/B 各自源格式执行，不能先扩展为 FP19/FP9，再按扩展后的指数宽度判断特殊值。

| 输入格式 | exponent 位数 | fraction 位数 | bias |
|---|---:|---:|---:|
| FP16 | 5 | 10 | 15 |
| BF16 | 8 | 7 | 127 |
| E4M3 | 4 | 3 | 7 |
| E5M2 | 5 | 2 | 15 |

FP16、BF16、E5M2 使用以下规则：

| exponent | fraction | 类别 |
|---|---|---|
| 全 0 | 全 0 | Zero |
| 全 0 | 非 0 | Subnormal |
| 非全 0、非全 1 | 任意 | Normal |
| 全 1 | 全 0 | Inf |
| 全 1 | 非 0 | NaN |

E4M3 固定采用 E4M3FN，规则单独定义：

~~~text
exp=0，frac=0：Zero
exp=0，frac≠0：Subnormal
exp=15，frac=7：NaN，原始编码为 8'h7f 或 8'hff
其他非零指数编码：Normal，包括 exp=15 且 frac≤6
Inf：不存在
最大有限幅值：exp=15，frac=6，对应 448
~~~

E4M3 的 exponent 全 1 不能整体视为特殊值。所有 NaN 均转为零，普通计算通路不区分 qNaN/sNaN。

### 2.2 非规格化数、有效指数与数值合同

#### 2.2.1 Preserve 与有限数表示

subnormal 固定 Preserve，不提前规格化、不执行 FTZ。特殊值替换后，对有限源格式字段构造：

~~~text
Normal：hidden=1，Eeff=e_raw，nz=1
Subnormal：hidden=0，Eeff=1，nz=1
Zero：hidden=0，Eeff=0，nz=0
~~~

将 fraction 按 1.4 节扩展，再形成 `mag={hidden, fraction}` 和补码 `sig_tc`。非零有限数满足：

$$
x=sig_{tc}\times2^{Eeff-bias_{src}-F}
$$

其中 FP19 的 F=10，FP9 的 F=3。FP16 subnormal 转为 FP19 后 Eeff 仍为 1，保留源 bias=15，不加 112。BF16 低位补 3 个零、E5M2 低位补 1 个零均精确保值，无入口舍入。

保留 subnormal 不需要入口 LZC 和可变规格化左移器。但小数在后续有限 ALIGN 窗口中仍可能丢失低位，Preserve 不代表后续对阶无限精确。

#### 2.2.2 M、ALIGN、MX 指数合同

M 级计算整数尾数乘积及未扣 bias 的有效指数和：

$$
P_{int}=sig_A\times sig_B,\qquad E_{sum}=Eeff_A+Eeff_B
$$

| 模式 | Eeff 接口 | E_sum 数学最小及 RTL 加法宽度 |
|---|---|---|
| FP19 | unsigned8 | unsigned9 |
| FP9 | unsigned5 | unsigned6 |

相加前分别零扩展一位，禁止丢失最高进位。真实乘积 LSB 指数定义为：

$$
e_{prod}=E_{sum}-bias_A-bias_B-F_A-F_B
$$

$$
value=P_{int}\times2^{e_{prod}}
$$

`E_sum` 不是无 bias 的真实指数。本文不再使用遗漏 bias 却直接与真实 value 等同的 e_p0 合同。

每个 output group 独立选择非零乘积的 E_sum_max：

$$
d_i=E_{sum,max}-E_{sum,i}
$$

同一事务固定源格式组合，bias 总和及 F_A+F_B 均恒定，因此该距离与真实指数对阶距离相同。两个 output group 分别保留自己的 MAX、valid 和 scale metadata。

设 F_Q 为归约结果的小数位数，FP19 为 29，FP9 为 21，则对阶后部分和的目标数值为：

$$
S_Q\approx\sum_i P_{int,i}\times2^{F_Q-F_A-F_B-d_i}
$$

固定小数提升量 FP19 为 9 bit，FP9 为 15 bit。等式中的近似仅表示逐项 ALIGN RNE 引入的误差，具体 Guard/Round/Sticky 与舍入位置在 ALIGN 章节定义。所有物理左移必须先扩位。

MX 扣除源 bias 并应用组合尺度指数：

$$
e_{q,lsb}=E_{sum,max}-bias_A-bias_B+combined\_scale\_exp-F_Q
$$

$$
value\approx S_Q\times2^{e_{q,lsb}}
$$

无额外尺度时 combined_scale_exp=0。各尺度字段的编码、范围及指数安全位宽在 MX 章节冻结；减法前必须将 unsigned E_sum 和 bias 转换到足够宽的 signed 中间域，不允许直接在 unsigned 域减出负数。

e_q_lsb 是 PSUM 的真实 LSB 指数；MX 转换后字段 E 采用 bias127，Q2.34 的真实 LSB 指数为 E−161。PSUM 的 e_q_lsb 与 E 不能混用，精确转换见§8.4。

#### 2.2.3 零乘积与全零组

~~~text
product_nz = (sig_A != 0) && (sig_B != 0)
~~~

零乘积不参与有效指数 MAX，不能用其占位指数抬高整组对阶基准。全零组输出零数值、确定的指数占位值及 group_nz=0；占位指数不承担非零数值语义。nz 可从补码 payload 派生，不强制占用 26-bit 数值容器中的位。

### 2.3 NaN、Inf 与 Zero 的转换规则

输入策略固定为：

~~~text
NaN  → +0
+Inf → 源格式最大正有限数
-Inf → 源格式最大负有限数
+0/-0 → +0
~~~

“同格式最大有限数”指原始输入格式，而不是 FP19/FP9 容器可容纳的最大字段值。必须先按源格式替换，再扩展字段和构造补码。

| 源格式 | Inf 替换后的 exponent | 替换后的 fraction | 符号 |
|---|---:|---|---|
| FP16 | 30 | 10-bit 全 1 | 保留 |
| BF16 | 254 | 7-bit 全 1 | 保留 |
| E5M2 | 30 | 2-bit 全 1 | 保留 |

E4M3 没有 Inf 编码，不设置虚构的 Inf 分支；其 NaN 按 2.1 节识别。NaN 与输入正负零统一输出：

~~~text
sign=0，sig_tc=0，Eeff=0，nz=0
~~~

补码数值不保留负零，不增加 zero_sign sideband。该入口规则不替代最终输出转换对下溢零符号的处理。ACC 精确零统一存为 C_sig=0、E_C=0。

入口可以产生可选诊断信息 `nan_seen`、`pos_inf_seen`、`neg_inf_seen`。这些字段不参与有限数计算，不占用数值 payload，也不能与 `align_sticky` 混用；若保留诊断状态，必须按 output group 隔离。

### 2.4 特殊值处理位置与方案比较

#### 2.4.1 MAC 内部入口处理：当前基线

~~~text
SRAM：原始紧凑编码
    ↓
MAC 内部共享入口
    源格式解包与分类
    NaN/Inf 策略替换、零规范化
    subnormal Preserve
    FP19/FP9 字段扩展
    hidden、Eeff 与补码构造
    26-bit payload 打包
    ↓
广播或分发至 16 个乘法前端输入寄存器
    ↓
M → ALIGN → PSUM → MX → ACC
~~~

共享粒度由实际操作数复用决定。同一个广播操作数只转换一次；不同操作数需要匹配输入吞吐的并行转换资源。16 个前端并不意味着 A/B 都相同，不能用单操作数吞吐的共享转换器无代价地替代 16 路不同输入的转换。

处理位于 MAC 内部，但不在每个乘法前端重复实现完整浮点分类器。MAC 具有原始格式入口，乘法前端内部只有有限补码计算语义。

#### 2.4.2 MAC 外部处理：保留的集成候选

该方案在 SRAM 读出后的 MAC 外部共享路径完成同样的分类、替换、Preserve、字段扩展和补码转换，然后将有限 payload 送入 MAC。

数值规则与内部方案完全相同。MAC 外部必须保证源格式/bias、数值、valid、scale 和 group metadata 同拍。转换逻辑可跨多个 MAC 共享，但 MAC 本身依赖外部输入合同。

SRAM 在两种方案下均保存原始紧凑编码，不保存展开的补码 payload。两种入口方案均不在 ACC 内做目标格式特殊值分类；该分类位于长点积完成后的输出转换边界，其系统集成位置另行确定。

#### 2.4.3 位置方案的 PPA 与复杂度

| 比较项 | MAC 内部共享入口 | MAC 外部共享入口 |
|---|---|---|
| MAC 边界输入 | 原始紧凑编码 | 有限补码 payload |
| 独立复用性 | 入口语义完整 | 依赖外部转换模块 |
| 分类逻辑共享 | 可在内部广播前共享 | 可跨 MAC 共享 |
| 广播成本 | 取决于实际拓扑和展开位置 | 取决于实际拓扑和展开位置 |
| M 级时序 | 取决于入口寄存器位置 | 取决于入口寄存器位置 |
| 验证重点 | 内部转换、分发及流水一致性 | 跨模块合同与配置一致性 |

不能仅凭处理在 MAC 内部就判定面积更大或 M 级更慢。PPA 由转换实例数、广播扇出、缓冲、寄存器位置和布线共同决定。当前 RTL 固定采用内部入口方案，外部方案只作为系统集成比较候选。

### 2.5 入口转换逻辑、寄存器与 metadata

#### 2.5.1 组合逻辑关系

对每个独立源操作数，入口执行：

~~~text
src_fmt_A/B
    → 源字段解包及分类规则选择
    → 最大有限值常量选择
    → 特殊值替换 mux
    → fraction 扩展、hidden/Eeff 构造
    → 幅值零扩展及按 sign 条件取负
    → payload 打包
~~~

FP19 构造 unsigned11 `{hidden, frac19}`，FP9 构造 unsigned4 `{hidden, frac9}`，再按 1.3 节转换为补码。源格式 exponent=0 且 fraction≠0 的有限输入始终使用 hidden=0、Eeff=1。

逻辑上的先后顺序必须满足先按源格式分类和替换、再形成有限计算表示；不要求各概念步骤之间新增寄存器。综合可以在保持行为一致的前提下合并比较器和 mux。

#### 2.5.2 输入寄存器与同拍关系

入口转换放在 M 输入寄存器之前。该输入准备边界不额外命名为第六个计算阶段，但 SRAM 到 ACC 的总延迟和入口时序必须实际计入。若需要额外准备寄存器，须在整体流水章节明确其延迟与握手，不能隐式增加。

与数值共同锁存并流水传递的语义包括：

~~~text
valid、compute_mode
源格式或等价 bias 信息
output-group 归属、事务 tag
scale metadata
INT16 phase（在两拍调度边界生成）
~~~

具体 metadata 编码位宽随顶层接口和 MX scale 合同冻结，不占用数值容器保留位。流水停顿时数值与 metadata 一起保持；模式切换不能让新配置解释旧事务。不同 output group 的有效状态、tag 和 scale 生命周期独立。

乘法前端仅进行 payload 解包、显式扩位和模式路由，不再次条件取负，不再次判断原始 NaN/Inf。

### 2.6 26-bit 统一乘法前端输入格式

每个物理乘法前端分别接收 A、B 两个 26-bit 容器，数值输入总宽度为 52 bit，不包括 valid、格式、scale、tag 和 group metadata。

~~~systemverilog
logic [25:0] a_payload, b_payload;
~~~

整个容器是 unsigned packed container，只有取出的具体尾数或整数 lane 具有 signed 属性。

| 模式 | 指数字段 | 数据字段 | 保留位 |
|---|---|---|---|
| FP19 | [23:16]=Eeff[7:0] | [11:0]=signed12 sig | [25:24]、[15:12] |
| INT16 | 无 | [15:0]=signed16 | [25:16] |
| FP9 | [25:21]=exp1，[20:16]=exp2 | [9:5]=signed5 sig1，[4:0]=signed5 sig2 | [15:10] |
| INT8 | 无 | [15:8]=signed8 lane1，[7:0]=signed8 lane0 | [25:16] |
| INT4 | 无 | [7:4]=signed4 lane1，[3:0]=signed4 lane0 | [25:8] |

FP9 的 exp1/exp2 均为 Eeff，而不是未处理的原始 exponent：

~~~text
exp1 ↔ sig1 → lane1 → MUL9  → group1
exp2 ↔ sig2 → lane0 → MUL12 → group0
~~~

INT8/INT4 同样采用低位 lane0、高位 lane1。该映射在全部 16 个前端保持一致。

发送端将保留位置零，接收端不将其解释为数值。例如：

~~~systemverilog
logic [7:0] fp19_exp_u8;
logic signed [11:0] fp19_sig_s12;
logic [4:0] fp9_exp0_u5, fp9_exp1_u5;
logic signed [4:0] fp9_sig0_s5, fp9_sig1_s5;
logic signed [15:0] int16_s16;
logic signed [7:0] int8_lane0, int8_lane1;
logic signed [3:0] int4_lane0, int4_lane1;

always_comb begin
    fp19_exp_u8  = a_payload[23:16];
    fp19_sig_s12 = $signed(a_payload[11:0]);
    fp9_exp0_u5  = a_payload[20:16];
    fp9_exp1_u5  = a_payload[25:21];
    fp9_sig0_s5  = $signed(a_payload[4:0]);
    fp9_sig1_s5  = $signed(a_payload[9:5]);
    int16_s16   = $signed(a_payload[15:0]);
    int8_lane0  = $signed(a_payload[7:0]);
    int8_lane1  = $signed(a_payload[15:8]);
    int4_lane0  = $signed(a_payload[3:0]);
    int4_lane1  = $signed(a_payload[7:4]);
end
~~~

B 采用相同解包。禁止把 FP19 的 `[15:0]` 整体解释为 signed16，保留位零不能替代负数的符号扩展。

各模式扩展到物理乘法端口时，使用显式符号复制，例如 FP9 lane0 进入 MUL12 为 `$signed({{7{fp9_sig0_s5[4]}}, fp9_sig0_s5})`；INT16 的低分片先补 0 再扩位，高分片复制原始符号。

逻辑上可按指数/数据组织两个寄存器 bank：

~~~systemverilog
logic [9:0]  exp_bank_q;   // payload[25:16]
logic [15:0] data_bank_q;  // payload[15:0]
~~~

整数模式下指数 bank 可停止更新；此时 bank 的旧值必须被模式选择隔离，不得进入整数数值路径。如果对外重组完整 payload，当前模式的保留位仍显式输出零。FP19 的指数高两位无效，INT4 只使用数据低八位。

时钟门控采用规范的 clock-enable 或 ICG，不能通过组合逻辑直接生成时钟。无效结果寄存器停止更新和乘法器输入隔离分别控制寄存器与组合逻辑的翻转。

### 2.7 20-bit 紧凑输入格式

在显式携带 Eeff 和补码尾数的字段组织下，20 bit 足以覆盖所有计算模式：

| 模式 | 有效信息位数 | 20-bit 映射 |
|---|---:|---|
| FP19 | 20 | [19:12]=Eeff，[11:0]=signed12 sig |
| FP9 | 20 | [19:15]=exp1，[14:10]=exp2，[9:5]=sig1，[4:0]=sig2 |
| INT16 | 16 | [15:0]=signed16 |
| INT8 | 16 | [15:8]=lane1，[7:0]=lane0 |
| INT4 | 8 | [7:4]=lane1，[3:0]=lane0 |

未使用位为零。FP9 的 lane 与 exponent 配对保持不变。20 bit 是当前显式字段组织的充分宽度，不宣称是所有重编码方式的信息论下限。

紧凑格式不减少数值精度。相对 26-bit，每个操作数减少 6 bit，容器宽度减少约 23.1%。若 16 个前端各保留 A/B 本地输入寄存器，则一个流水边界名义减少：

$$
16\times2\times6=192
$$

个寄存器位。实际收益还受常量位优化、共享寄存器、mux 和广播缓冲影响。

其代价是字段用途随模式变化，例如 `[15:12]` 在 FP19 中属于指数，在 INT16 中属于整数数据。模式相关解包可能增加布线交叉和输入选择路径。

26-bit 与 20-bit 是综合期架构选择，不是运行时计算模式。同一实例不通过运行时模式切换物理端口宽度。

### 2.8 输入格式的 PPA 与实现约束

| 比较项 | 26-bit 基线 | 20-bit 紧凑候选 |
|---|---|---|
| 数值精度 | 无损 | 无损 |
| 指数与数据平面 | 固定分离 | 模式相关重叠 |
| 输入总线名义宽度 | 52 bit/前端 | 40 bit/前端 |
| 解包与布局 | 较直接 | 模式选择及交叉布线可能增加 |
| 整数指数门控 | 独立 10-bit bank | 需要按实际字段用途组织 |
| 面积与功耗 | 寄存器、广播宽度较大 | 需验证缩窄收益是否覆盖附加逻辑 |

当前首版固定采用 26-bit。后续仅在综合及布局布线证明有整体收益时，再选择 20-bit 实现。若采用紧凑传输、入口展开，也必须明确新增适配边界。

重点评估分类、特殊值替换、条件取负、广播和 MUL12 模式选择形成的路径；不能仅根据乘法器面积判断整体 PPA。需验证 FP19 模式 MUL9 隔离、整数指数 bank 门控，以及数据与 metadata 的同步停顿。

本章冻结入口数值规则和接口字段。ALIGN 舍入细节、FP19 limb 映射、scale 编码及 ACC 反馈时序在后续章节评审；这些细节不得改变本章已定义的源格式语义、独立 output group、INT16 最终饱和和指数合同。

## 3. 低位宽格式的并行度与归约组织

本章定义 FP9、INT8 和 INT4 模式下，16 个物理乘法前端、两个 product lane、两颗 PSUM tree 与两个累加状态之间的对应关系。

三种模式均采用 DUAL_OUTPUT：

~~~text
16 个 lane0 乘积 → tree0 → group0 → ACC0
16 个 lane1 乘积 → tree1 → group1 → ACC1
~~~

每个 output group 独立形成一个 dot16。两个 group 不执行数值合并，不共享 carry、对阶基准、尺度 metadata 或累加状态。

### 3.1 乘积 lane、dot 与 output group 的定义

#### 3.1.1 物理前端与 product lane

定义 i=0…15 为物理乘法前端编号，g=0,1 为低位宽模式的 lane 编号，同时对应 output group 编号。A[i,g]、B[i,g] 为该位置的数学操作数，P[i,g] 为对应乘积：

$$
P[i,g]=A[i,g]\times B[i,g]
$$

| lane | 物理乘法器 | 乘积物理端口 | 归约树 | 累加状态 |
|---|---|---|---|---|
| lane0 | MUL12 | signed24 | tree0 | ACC0 |
| lane1 | MUL9 | signed18 | tree1 | ACC1 |

MUL12 与 MUL9 的物理位宽不同，但在 FP9、INT8、INT4 模式下均能精确容纳其 lane 的完整乘积，因此两个 group 的数值精度相同。lane 表示独立乘积位置，不表示宽乘积的高低分片。

#### 3.1.2 dot16 与归约事务

一次低位宽归约事务包含每组最多 16 个有效乘积位置：

$$
D_g=\sum_{i=0}^{15}P[i,g]
$$

整数模式精确计算该和。浮点模式先对阶，再在规定的小数精度下归约，可能产生对阶误差。

dot16 表示一次归约的最大项数，不表示整个矩阵乘法的 K 维长度。更长的点积通过多次事务更新对应 ACC 状态。

例如 C[m,n]=Σ_k A[m,k]B[k,n] 中，一个 dot16 可以对应 16 个 K 位置。当前保持独立操作数接口，不固定两个 group 对应的矩阵坐标，也不要求两个 lane 共用 A 或 B。具体输出坐标映射和广播拓扑由后续系统集成确定，不改变本章数值合同。

#### 3.1.3 output group 的含义

每个 output group 包含：

~~~text
归约数值、该组有效状态
该组浮点对阶基准
该组 scale metadata
该组 tag 或目标输出标识
对应 ACC 状态及其更新控制
~~~

相同格式配置可以共同控制两个 group，但不能据此合并它们的数值或事务状态。两个 group 的 tag、scale 数值即使相同，也必须保持各自的有效性和生命周期。

#### 3.1.4 每组 dot16 的 scale 合同

每个 group 的一个 dot16 始终对应同一组 A/B scale，归约后由 MX 统一应用组合尺度。两个 group 的 scale 可以不同，跨事务的 scale 也可以改变，但 scale metadata 必须与该事务的数值共同流水传递。

不允许将属于不同 scale 的子块直接混入同一组 dot16，再在 MX 统一应用单个 scale。若系统数据跨越 scale 边界，应在上游组织为满足本合同的独立归约事务。

scale 的编码、范围及 MXINT8 尺度处理细节在 MX 章节定义。本章的普通 INT8/INT4 整数累加公式使用未附加尺度变换的整数部分和；MXINT8按第八章赋予尺度并使用第九章TC_FLOAT状态。

### 3.2 FP9 并行与归约组织

#### 3.2.1 输入字段与乘法映射

FP8 输入经过第二章处理后，以 FP9 计算 payload 进入乘法前端：

| group | 尾数字段 | 指数字段 | 乘法器 |
|---|---|---|---|
| group0 | [4:0]，signed5 | [20:16]，unsigned5 Eeff | MUL12 |
| group1 | [9:5]，signed5 | [25:21]，unsigned5 Eeff | MUL9 |

A、B 使用相同 lane 布局。E4M3×E4M3、E5M2×E5M2、E4M3×E5M2 和 E5M2×E4M3 均使用该映射。源格式组合在事务内固定，入口分别按 A/B 源格式转换。

$$
P_{int}[i,g]=sig_A[i,g]\times sig_B[i,g]
$$

$$
E_{sum}[i,g]=Eeff_A[i,g]+Eeff_B[i,g]
$$

尾数乘积数学最小为 signed9，物理输出仍为 signed24、signed18；E_sum 使用 unsigned6，相加前两侧分别零扩展至 6 bit。

#### 3.2.2 两组独立对阶

每个位置的非零乘积标志为：

$$
nz[i,g]=(sig_A[i,g]\ne0)\land(sig_B[i,g]\ne0)
$$

若后续增加位置 mask，还需与对应位置的有效 mask 相与。每个 group 独立执行：

$$
E_{max,g}=\max_{i:nz[i,g]=1}E_{sum}[i,g]
$$

$$
d[i,g]=E_{max,g}-E_{sum}[i,g]
$$

不得对两个 group 的 32 个乘积共同求 MAX，否则一组较大的指数会压缩另一组的有效对阶窗口，使两个独立输出产生数值耦合。全零组按第二章输出零数值和确定的指数占位值。

#### 3.2.3 Q7.21 归约

FP9 两个尾数各有三位小数，尾数整数乘积具有六位小数，归约目标保留 21 位小数，因此固定提升量为 21−3−3=15 bit。

对阶后单项整数表示定义为：

$$
X[i,g]=\mathcal{A}_{21}(P_{int}[i,g],d[i,g])
\approx P_{int}[i,g]\times2^{15-d[i,g]}
$$

其中 A_21 表示第六章定义的有限精度对阶操作。单项数值采用 signed24 足够；必须先扩展再左移，不能在 signed9 乘积上直接左移 15 位。

~~~text
signed9 乘积
    → 显式符号扩展
    → 小数位提升与对阶
    → signed24 单项数值
    → 符号扩展至 tree 的输入宽度
~~~

两颗树分别计算：

$$
S_g=\sum_{i=0}^{15}X[i,g]
$$

每颗树输出 signed28 Q7.21，全部 [27:0] 属于数值结果。

| 节点 | 数学最小位宽 | RTL 安全处理 | 接口或寄存器 |
|---|---|---|---|
| 原始尾数乘积 | signed9 | 保留物理输出；缩窄检查符号冗余 | signed24 / signed18 |
| E_sum | unsigned6 | 输入先零扩展至 6 bit 再相加 | 每项 6 bit |
| 对阶后单项数值 | signed24 | 位移前扩展 | 送树前扩展到所选物理输入宽度 |
| dot16 数值和 | signed28 | 按归约层次扩位，或在 28-bit 域归约 | signed28 |

Guard、Round、Sticky 等附加信息不计入上述数值 payload，具体组织在第六、七章确定。不能简单假定各乘积 sticky 做 OR 后就足以保证带正负抵消的 dot16 正确舍入。

负补码的算术右移不等于向零截断，具体误差和舍入语义必须在 ALIGN 章节明确。

#### 3.2.4 MX 与 ACC 对应关系

两个 group 分别生成：

$$
e_{q,lsb,g}=E_{max,g}-bias_A-bias_B+combined\_scale\_exp_g-21
$$

$$
value_g\approx S_g\times2^{e_{q,lsb,g}}
$$

~~~text
tree0 的 S0、Emax0、scale0 → MX group0 → ACC0
tree1 的 S1、Emax1、scale1 → MX group1 → ACC1
~~~

两个 ACC 均保存 signed36 Q2.34 和8位 bias127 指数。PSUM signed28 Q7.21 需经 MX 补8个低零位后输入 ACC；各组 Emax、scale 和状态独立。

### 3.3 INT8 并行与归约组织

#### 3.3.1 输入及资源映射

每个前端的 A、B payload 分别包含：

~~~text
[7:0]  ：signed8 lane0 → MUL12 → group0
[15:8] ：signed8 lane1 → MUL9  → group1
~~~

合法输入范围为 [-127,127]，单乘积最大绝对值为 127²=16129，数学最小为 signed15。物理输出仍为 signed24、signed18；若提取低 15 bit，必须确认其余高位全部为 bit14 的符号扩展。

#### 3.3.2 两组精确 dot16

整数不需要浮点指数 MAX 或可变对阶。ALIGN 保留事务流水关系，通过模式 mux 将整数乘积送入各自归约通路，并隔离无效浮点指数逻辑。

$$
S_g=\sum_{i=0}^{15}P[i,g],\qquad |S_g|\le16\times16129=258064
$$

signed19 足以精确表示结果。

| 节点 | 数学最小位宽 | 最终物理表示 |
|---|---|---|
| 单乘积 | signed15 | MUL12 signed24 / MUL9 signed18 |
| 两项和 | signed16 | 依归约树实现扩位 |
| 四项和 | signed17 | 同上 |
| 八项和 | signed18 | 同上 |
| 十六项和 | signed19 | PSUM signed28 |

表中分层表示数值增长关系，不要求物理实现采用四层串联 CPA；具体 CSA 或其他结构在第七章确定。每次相加前须扩展到该级安全宽度，不能依赖窄表达式溢出后再赋值。

每颗 PSUM 输出规定：

~~~text
[18:0]  ：signed19 有效 payload
[27:19] ：bit18 的符号扩展
~~~

PSUM 不执行饱和或舍入。缩窄输出只允许删除经检查的冗余符号位。

#### 3.3.3 每次更新后的 INT32 饱和

普通 INT8 累加中，signed19 部分和与旧 INT32 状态相加的数学完整结果需要 signed33；RTL 安全加法采用 signed36，两侧先显式符号扩展：

$$
C_{wide,g}=\operatorname{sext}_{36}(C_{old,g})+\operatorname{sext}_{36}(S_g)
$$

$$
C_{new,g}=\operatorname{sat}_{INT32}(C_{wide,g})
$$

每次有效的完整 dot16 更新后，ACC0、ACC1 分别饱和到 [-2^31, 2^31−1]。无效 group 不更新状态。不得在树内部提前饱和，也不将饱和延后到整个长点积结束。

饱和比较在 signed36 域进行：超出正边界输出 `32'h7fffffff`，低于负边界输出 `32'h80000000`，其余情况取低 32 bit。若工作寄存器为 36 bit，提交后写回 INT32 结果的符号扩展。

### 3.4 INT4 并行与归约组织

#### 3.4.1 两 lane 映射

~~~text
[3:0] ：signed4 lane0 → MUL12 → group0
[7:4] ：signed4 lane1 → MUL9  → group1
~~~

每个前端每拍仅计算两个 INT4 乘积，16 个前端形成两个 dot16。当前基线不采用在单个乘法器中额外打包多个 INT4 乘积的编码，也不将单个物理乘法器解释为多个可独立取出的乘积通路。

#### 3.4.2 数值位宽与接口冗余

合法输入范围为 [-7,7]，单乘积最大绝对值为 49，数学最小为 signed7。dot16 最大绝对值为 16×49=784，数学最小为 signed11。

| 节点 | 数学最小位宽 |
|---|---|
| 单乘积 | signed7 |
| 两项和 | signed8 |
| 四项和 | signed9 |
| 八项和 | signed10 |
| 十六项和 | signed11 |

相加前显式扩展至对应安全宽度。物理乘积仍保留 signed24/signed18，PSUM 输出仍为 signed28；按第一章约定，有效接口保留 signed12：

~~~text
[10:0]  ：数学结果所需的 signed11
[11]    ：bit10 的冗余符号位
[27:12] ：bit11 的符号扩展
~~~

合法结果应满足以下符号冗余检查：

~~~systemverilog
psum_q[27:11] == {17{psum_q[10]}}
~~~

signed12 是接口选择，不代表该模式需要 12 位才能避免溢出。INT4 归约不执行数值截断、舍入或饱和。

#### 3.4.3 每次更新后的 INT32 饱和

按 signed12 接口提取 S_g，并与旧 INT32 状态分别符号扩展至 signed36 后相加，执行与 3.3.3 节相同的 INT32 饱和。该加法数学完整结果需要 signed33，signed36 为安全实现宽度。

ACC0、ACC1 分别在每次有效完整 dot16 更新后提交饱和结果；无效 group 不更新。长点积采用逐事务饱和，不等同于无限宽累加后只饱和一次。

### 3.5 乘积局部相加与独立输出的边界

#### 3.5.1 禁止跨 group 局部相加

当前基线禁止将 P[i,0] 与 P[i,1] 相加后作为一个归约输入。它们属于两个独立输出，合并会破坏 ACC0/ACC1 的数值语义；相同指数或 scale 也不构成合并条件。

允许的局部归约发生在同一 group 内不同前端之间，例如 P[0,g]+P[1,g]、P[2,g]+P[3,g]。

#### 3.5.2 浮点局部归约的共同数值基准

FP9 中，同一 group 的原始整数尾数乘积在指数不同时不能直接相加。若 value0=P0×2^e0、value1=P1×2^e1，且 e0≠e1，则 P0+P1 不能表示两者数值之和。

当前基线在 ALIGN 确定组内共同基准并对阶后，再进入 PSUM 归约。共同 scale 可以延后至 MX 应用，是因为本组所有项具有相同尺度，不是因为忽略了各项数值单位。

#### 3.5.3 DUAL_OUTPUT 与 WIDE_SUM

| 模式 | tree0 的角色 | tree1 的角色 | 跨树重构 |
|---|---|---|---|
| FP9、INT8、INT4 | group0 完整归约 | group1 完整归约 | 无 |
| INT16 | 同一结果的低路部分积和 | 同一结果的高路部分积和 | PSUM 内显式重构 |
| FP19 | 同一结果的一个 limb | 同一结果的另一个 limb | PSUM 内按 23-bit limb 权重重构 |

tree0/tree1 是物理资源名称，group0/group1 是数值归属。仅在 DUAL_OUTPUT 模式下两者一一对应。INT16/FP19 占用两树时，tree1 数据仍属于 group0，不能触发 ACC1 更新。

### 3.6 对整体 MAC 架构的影响

#### 3.6.1 路由与模式 mux

~~~text
MUL12[0…15]
    → lane0 模式适配
    → FP9 对阶或整数直通
    → tree0 → MX group0 → ACC0

MUL9[0…15]
    → lane1 模式适配
    → FP9 对阶或整数直通
    → tree1 → MX group1 → ACC1
~~~

模式 mux 选择格式适配及 DUAL_OUTPUT/WIDE_SUM 的数据解释，不增加运行时任意 lane 交换或跨 group 归约。保留独立操作数接口不意味着需要增加内部交叉交换网络。

MUX、valid 和 metadata 必须使用同一事务的模式信息，不能直接使用已切换的新配置。

#### 3.6.2 寄存器与物理接口边界

本章不增加功能阶段。数据仍通过M、ALIGN、PSUM、MX、ACC五个功能阶段；物理流水划分按4.8节确定。两个 PSUM 输出可用以下 DUAL_OUTPUT 逻辑视图表示：

~~~systemverilog
logic signed [27:0] psum_data_q [0:1];
logic        [1:0]  psum_group_valid_q;
~~~

FP9 每组还保留：

~~~systemverilog
logic [5:0] psum_emax_q [0:1];
logic [1:0] psum_has_nonzero_q;
~~~

上述psum_data_q仅为56-bit寄存器在DUAL模式下的两个signed28逻辑视图，不额外增加寄存器；WIDE模式按第七章提取完整signed36结果并检查符号扩展。psum_has_nonzero 表示是否存在参与 MAX 的非零输入乘积，不等于归约结果非零；正负项抵消可令结果为零，而该标志仍为 1。

tag 和 scale 的具体位宽由顶层接口及 MX 章节确定，数值、有效状态、指数及 metadata 必须共同锁存和保持。

#### 3.6.3 有效事务、零数据与尾块处理

必须区分有效事务中的数值零、无效 group，以及 dot16 中不参与计算的位置。数值全零不自动等价于没有事务，不能仅凭 data=0 推导 ACC 写使能。

首版由上游将不足16项的尾块补零至16项，并保留两个group的独立valid；若只有一个输出有效，另一组不更新ACC。不增加逐位置mask。

另一种实现是每组增加 16-bit 位置 mask，在乘法或归约入口屏蔽无效位置，FP9 同时禁止其参与 MAX。该方案支持更灵活的位置屏蔽，但增加控制位和屏蔽逻辑。

当前已确认上游补零、独立 group valid，不增加逐位置 mask；后者仅作为未来扩展候选。

#### 3.6.4 双组握手与停顿

两个group共用流水前进/停顿节拍，各自保留独立valid和ACC写使能：

- 两组一起随流水前进或保持；
- 无效 group 不更新 ACC，也不因自身无效而阻塞；
- 任一有效 group 暂时无法接收时，共享流水事务整体保持；
- ACC 更新必须由一次实际提交事件触发，不能因保持 valid 而重复累加。

该方案避免独立队列与重新配对逻辑，同时保持数值状态隔离。若要求独立接收、独立停顿和独立提交，则需增加各组握手及必要缓冲；数值独立不自动要求控制完全解耦。

当前已确认共同停顿方案；独立操作数接口及两组数值独立性不受该选择影响。

#### 3.6.5 时序、门控与 PPA

重点评估两组 FP9 指数 MAX、对阶距离及可变移位路径，每树16输入归约、必要的WIDE重构再到56-bit打包寄存器的路径，以及 MX 到 ACC 的反馈更新能力。两组 metadata 布线必须与数据路径一起评估。

固定 lane 到 tree 的映射避免额外跨组交换网络。整数模式可隔离浮点 MAX 与对阶移位逻辑；INT4 冗余高位可作为综合优化依据，但不能将负数高位强制清零。

无效 group 可停止结果寄存器更新并隔离组合输入。单输出模式关闭 ACC1 时，不能同时关闭 WIDE_SUM 仍需使用的 tree1。

### 3.7 当前冻结的并行度方案

| 模式 | 单前端每拍输出 | 16 前端归约组织 | PSUM 结果 | ACC |
|---|---|---|---|---|
| FP9 | 2 个完整乘积 | 2×dot16 | 两个 signed28 Q7.21 | ACC0、ACC1 |
| INT8 | 2 个完整乘积 | 2×dot16 | 每树低 19 bit 有效 | ACC0、ACC1 |
| INT4 | 2 个完整乘积 | 2×dot16 | 每树低 12 bit 接口有效，数学最小 11 bit | ACC0、ACC1 |
| FP19 | 1 个完整乘积 | 1×dot16，WIDE_SUM | 重构 signed36 Q7.29 | 仅 ACC0 |
| INT16 | 2 项部分积 | 两拍形成 1×dot16 | 输出 T0、T1×2^8 | 仅 ACC0 |

已冻结规则：

- FP9、INT8、INT4 双输出，固定 lane0/MUL12 到 group0，lane1/MUL9 到 group1，不跨 group 合并乘积。
- 两颗树在 DUAL_OUTPUT 模式下分别归约 16 项；FP9 每组独立求 MAX、对阶并携带 scale。
- 保持独立操作数接口，不固定矩阵坐标或广播映射。
- 每个 group 的单个 dot16 使用同一组 scale；两个 group 可以不同，跨事务变化与数值共同流水传递。
- 整数 PSUM 精确归约，不提前饱和。INT8、INT4 每次完整 dot16 更新后分别饱和到 INT32；INT16 保留 signed36 跨拍中间状态，第二拍完成后统一饱和。

已确认上游补零、独立 group valid、共同流水停顿、ALIGN 逐项 RNE 及 E8M0 scale。第四至九章定义具体电路；已接受同一ACC隔若干拍更新，允许多周期实现；具体周期数和寄存器位置尚未冻结。

## 4. 整体 MAC 电路架构

本章及后续章节展开新数值方向：上游补零、独立 group valid、共同停顿不变；ALIGN 逐项 RNE，lane0/lane1 为32/24位；PSUM 内完成重构和 INT16 权重，输出56位容器。浮点/MXINT8 输入及反馈统一 Q2.34/bias127；tail 参与最终 RNE，写回采用 FTZ/Inf，MX 输入指数越界暂缓。RTL 已迁移，MX 输入指数越界仍不定义恢复数值。

前三章的尾块、停顿、舍入及scale合同由本章至第九章展开为电路。前三章关于普通 INT8 的 INT32 累加规定不适用于 MXINT8：二者复用乘法计算模式，但采用不同的数值解释和累加状态类型。

当前以功能正确性优先，没有目标频率要求。M → ALIGN → PSUM → MX → ACC为五个功能阶段，不限定为五个物理周期。已接受同一ACC隔若干拍完成一次更新，可采用多周期实现和反压；具体切拍位置、周期数及启动间隔依据实现结果确定，不承诺每拍更新。

### 4.1 顶层功能与接口边界

#### 4.1.1 数值输入及入口转换

建议原始紧凑输入接口为：

~~~systemverilog
logic [15:0] raw_a [0:15];
logic [15:0] raw_b [0:15];
~~~

每个前端对应独立 A/B，不固定矩阵坐标或广播关系。

| 源格式 | 原始 16-bit 容器 |
|---|---|
| FP16、BF16、INT16 | 全部 16 bit |
| FP8、INT8 | [7:0] 为 lane0，[15:8] 为 lane1 |
| INT4 | [3:0] 为 lane0，[7:4] 为 lane1，[15:8] 置零 |

SRAM 保持原始紧凑编码，MAC 内部入口按照第二章执行分类、有限数替换、Preserve、字段扩展和尾数补码转换。转换后保存：

~~~systemverilog
logic [25:0] m_operand_a [0:15];
logic [25:0] m_operand_b [0:15];
~~~

输入准备寄存器保存展开数值和对应 metadata，不计作额外计算阶段，但其物理延迟必须计入 SRAM 到输出的总延迟。INT16 可复用该寄存器保持两拍所需输入。

不足 16 项的尾块由上游补零；本版不增加逐位置 mask。双输出中仅一组有效时，由 group_valid 屏蔽另一组，不能仅以零数据替代无效事务。

#### 4.1.2 控制和结果接口

| 信号 | 宽度/属性 | 语义 |
|---|---|---|
| clk、rst_n | 各 1 bit | 时钟、复位 |
| in_valid、in_ready | 各 1 bit | 完整输入事务握手 |
| src_fmt_a、src_fmt_b | 各 unsigned3 | 七种源格式的枚举，保留编码非法 |
| group_valid | 2 bit | 两组数值和状态更新的有效掩码；WIDE仅允许01，DUAL允许01/10/11 |
| acc_first | 2 bit | 每组首次更新标记，由运行时配置或上层调度器提供；对应组以零作为本事务旧累加值 |
| tag[g] | 每组 TAG_W bit | 随事务传播的输出标识 |
| mx_enable | 1 bit | FP8/INT8 的 MX 数值语义选择 |
| scale_a[g]、scale_b[g] | 每组各 unsigned8 | 原始 E8M0 编码 |
| out_valid、out_ready | 各 1 bit | 结果包握手 |
| out_group_valid | 2 bit | 结果包内有效组 |
| out_data[g] | 浮点快照需36 bit；整数32 bit | 完整 Q2.34 系数，不能截为32位；ACC 状态观察端口分别提供 float_sig[35:0] 与 int32[31:0] |
| out_exp[g] | 每组 unsigned8 | 浮点 biased E；整数占位0；零 E=0、Inf E=255；Inf 符号由独立 signed2 端口输出 |
| out_tag[g] | 每组 TAG_W bit | 与结果对应的标识 |
| out_acc_kind | 1 bit | 本包 INT32/TC_FLOAT 类型；TC_FLOAT 指自定义补码浮点状态 |

输出包是每次完整dot16提交后的内部状态快照，不是最终格式转换结果。浮点out_data完整提取signed36 Q2.34再结合out_exp解释；整数忽略指数。系统负责识别长点积的最后一次提交，并仅对所需快照执行最终格式转换。当前不新增acc_last、目标格式枚举或转换器握手端口，其集成协议尚待确定；不能把这些未定接口误当作ACC已实现能力。中间快照遵守相同反压协议，不得被后续状态更新覆盖。

TAG_W 为综合参数，由系统标识空间确定；tag 不表示按 tag 索引的累加寄存器文件。当前只有 ACC0/ACC1 两个累加上下文，不能仅凭不同 tag 交错维护更多输出。

本版不支持外部任意初值装载。新输出使用 acc_first 初始化为零。group_valid=0 的组忽略其 acc_first；两组均无效的空包不应作为计算请求发送。

WIDE/DUAL只定义数值组织和累加路由：FP19及INT16采用WIDE，56-bit容器保存一个完整贡献（signed36 数学结果符号扩展），送ACC0；INT16每phase各有一个贡献，并非phase0就产生完整dot16结果。FP9/MXFP8、INT8/MXINT8、INT4采用DUAL，低28位和高28位分别是group0/group1的数值容器，各自送ACC0/ACC1；28位容器不表示各模式都需要28位数学有效精度。两数不互相相加。

保留group_valid作为运行时有效控制。WIDE输入只允许2'b01，tree1参与重构不代表group1有效；DUAL通常为2'b11，也允许2'b01或2'b10，仅更新有效组。无请求使用in_valid=0，不发送group_valid=2'b00空包。入口验证上述合法组合，无效组忽略其acc_first和tag，不以数值补零代替无效控制。

mx_enable 只在 FP8 或 INT8 输入组合下合法，不为 FP19、INT4、INT16 隐式增加尺度模式。compute_mode 和 acc_kind 由源格式与 mx_enable 派生，不允许上游分别给出相互矛盾的选择。

#### 4.1.3 长点积调度、首次更新和模式切换

reset 清除流水 valid、INT16 调度状态、结果 valid 和 ACC 有效状态。未初始化的组必须以 acc_first=1 启动。acc_first 随事务进入 ACC 才生效，不得在入口提前清除仍被旧事务使用的状态。

当前调度基线为：同一ACC必须完成一个长点积后才开始另一个长点积，不支持在同一ACC上交错执行不同长点积，也不提供中间上下文保存/恢复接口。允许在长点积内暂停发射，暂停期间保留ACC状态。每个长点积的源格式组合、compute_mode和acc_kind保持一致；scale仍可按既有合同在dot16事务间变化。

acc_first是首次更新的内部语义，不要求在运行时配置之外增加重复控制。若上层每次dot16均附带配置，从该配置提取每组first；若配置只在长点积开始时读取一次，由上层调度器按事务进度生成first。first随该事务锁存并流水传递，不能在ACC执行时读取可变化的全局配置，也不能让长点积配置中的first=1持续作用于所有dot16。标记推进依据实际事务握手，不依据经过的时钟数。

例如1024项点积拆为64个dot16：第1个事务acc_first=1，第2～64个为0；下一个独立长点积的第1个事务重新置1。长点积结束后无需额外清零周期，ACC可保留最终值；新长点积首次更新时选择零旧加数。DUAL保留逐组acc_first[1:0]；两组同时开始时，可由同一个配置first生成两组标记。INT16仅phase0使用acc_first，phase1强制为0并延续中间状态。

模式或源格式组合切换由上层同步机制负责排空，具体见4.5.4节；MAC不强制提供idle端口或独立排空状态机。上层完成受当前模式影响的长点积、接收旧事务结果并确认排空后再切换。新长点积均重新初始化；INT32/TC_FLOAT类型改变时不得直接重解释旧状态，普通INT8与MXINT8同样遵守此规则。从WIDE返回DUAL时，ACC1即使保留旧数据，其新长点积首次更新也必须使用acc_first=1。本版不支持在同一长点积中混合不同源格式。

### 4.2 五级数据通路

~~~text
原始紧凑输入
    → MAC入口转换/保持寄存器
    → M → ALIGN → PSUM → MX → ACC
    → 带 valid 的内部状态快照包
    → 长点积结束后，由系统选择最终快照进行目标格式转换
~~~

| 阶段 | 主要组合逻辑 | 末端保存内容 |
|---|---|---|
| M | 模式 mux、两颗乘法器、指数加法 | 完整乘积、指数和、非零标志、metadata |
| ALIGN | 两组独立 MAX、两套距离/移位/RNE，整数旁路 | 16×(signed32+signed24)、Emax、metadata |
| PSUM | FP19 拆片、两树及 CPA、WIDE merge、INT16 phase 权重 | 56-bit 模式打包的完整部分和、metadata |
| MX | 56-bit 模式解包、bias/scale 修正 | Q2.34/整数贡献、biased E、metadata |
| ACC | 补码累加、内部规格化/RNE，普通整数饱和 | 自定义浮点或整数状态及快照包；无目标格式分类 |

每一级的数据、valid 和 metadata 使用一致的更新条件。INT16的两个phase是依次流经同一流水的两份部分积事务，不增加功能阶段；发射和完成之间可因物理流水或反压间隔多个时钟。phase0 只更新 ACC 工作状态，phase1 才提交完整 INT16 结果。

### 4.3 各模式的数据流与资源复用

| 外部运算语义 | compute_mode | ALIGN/PSUM | MX | ACC |
|---|---|---|---|---|
| FP16/BF16 | FP19 | 单组对阶、两树 limb 并重构 Q7.29 | bias 修正 | TC_FLOAT ACC0 |
| FP8 | FP9 | 两组独立对阶/归约 | 每组 bias 修正 | TC_FLOAT ACC0/1 |
| MXFP8 | FP9 | 与 FP9 相同 | 再应用 E8M0 组合指数 | TC_FLOAT ACC0/1 |
| 普通 INT8 | INT8 | 整数直通、两组归约 | 符号扩展 | INT32 ACC0/1，逐次饱和 |
| MXINT8 | INT8 | 与普通 INT8 相同 | M=S<<17，E=s_A+s_B+132 | TC_FLOAT ACC0/1 |
| INT4 | INT4 | 整数直通、两组归约 | 符号扩展 | INT32 ACC0/1，逐次饱和 |
| INT16 | INT16 | 两拍部分积、两树归约、重构及第二拍权重 | 完整贡献直通 | signed36 工作状态，最终 INT32 |

MXINT8 不增加新的物理乘法模式，也不使用 FP9 的逐项浮点对阶器。它先精确求整数编码乘积和，再由 MX 赋予真实指数；不得先执行普通 INT8 的 INT32 饱和再转换为浮点。

### 4.4 级间寄存器与位宽总表

| 边界 | 数值接口 | 属性和有效用途 |
|---|---|---|
| 入口→M | 16×(26+26) bit | unsigned 容器，解包字段按第二章解释 |
| M→ALIGN | 16×(24+18) bit | signed 完整乘积 |
| M→ALIGN 指数 | 每前端 9+6 bit | unsigned；lane0 兼容 FP19，lane1 用于 FP9 |
| M→ALIGN 非零标志 | 每前端 2 bit | 浮点 MAX 的有效候选控制 |
| ALIGN→PSUM | 16×(32+24) bit | lane0 signed32、lane1 signed24；拆片在 PSUM 内 |
| PSUM→MX | 56 bit unsigned 容器 | DUAL 两个 signed28；WIDE signed36 数学结果符号扩展到56位，bit36冗余 |
| MX→ACC group0 | signed36 | 浮点 Q2.34；整数按其整数单位 |
| MX→ACC group1 | signed36 | 浮点 Q2.34；普通整数符号扩展 |
| MX→ACC 指数 | 每组 unsigned8 | bias127；MX 输入越界暂缓，普通整数隔离 |
| ACC 整数工作状态 | 每组 signed36 | ACC0 支持 INT16 跨拍；ACC1 支持普通整数 |
| ACC 浮点状态 | 每组 signed36+unsigned8 | Q2.34 + bias127，共44位；控制独立 |
| 输出快照包 | 浮点36-bit数据+8-bit指数，整数32-bit | 配合 out_group_valid/out_acc_kind；不是最终目标格式编码 |

浮点两组输入均为 signed36 Q2.34；普通整数按原路径提取整数贡献。PSUM 的37位兼容视图不能作为新的浮点 ACC 状态。

数学最小位宽、算术安全中间值和最终接口宽度在各级分别列出。接口表不包含 tag/scale/control 总位数；这些字段按照本章定义逐项显式声明，不塞入数值保留位。

### 4.5 控制、有效位与元数据流水

共同控制不等于两组共享数值。一个事务包携带独立的 group_valid[1:0]、acc_first[1:0]、tag[0:1]、scale_a/b[0:1]，以及本事务公共的 compute_mode、acc_kind、mx_enable、源格式/bias 和 INT16 phase。

#### 4.5.1 输入保持和发射事件

入口使用一个操作数保持槽，in_fire=in_valid&&in_ready 时接收原始输入的转换结果。握手未完成时，上游必须保持数据及 metadata 稳定。

M 发射仅在下游允许流水前进且保持槽有效时发生。非 INT16 发射一次后释放槽；INT16 phase0 发射后保持槽不变，下一次成功发射 phase1 后释放。可在释放边沿同时补入新事务，但不能提前覆盖旧输入。

INT16 的两拍必须连续占用本槽的发射机会，允许因反压间隔多个时钟，不能插入另一个事务。phase 推进以实际发射为准，不以时钟数计数。

#### 4.5.2 单周期 ACC 的等价控制模型

为说明提交语义，单周期等价模型可定义 out_space=!out_valid||out_ready，pipe_step=out_space。pipe_step=1 时，各计算级同步推进；ACC 只消费当前 MX valid 一次，并按 group_valid 更新对应状态。

完成的事务覆盖或填入输出寄存器；无完成结果时清除已被接收的旧 out_valid。INT16 phase0 消费后不产生 out_valid。pipe_step=0 时，数据、valid、metadata、ACC 状态及 phase 全部保持。

该模型便于功能验证，不冻结物理单周期时序。实际 in_ready 还取决于保持槽能否接收新输入，不能直接等于 out_ready。

#### 4.5.3 多周期 ACC 的兼容要求

若 ACC 实现为多周期，必须区分 acc_accept、acc_busy、acc_commit：在 acc_accept 时只捕获一次 MX 事务和旧状态快照，忙期间不再次消费相同 MX 数据，在 acc_commit 时只更新一次架构/工作状态。

M/ALIGN/PSUM/MX 可共同停顿，但 ACC 已启动运算的内部状态机必须能独立前进；不得用上游 pipe_step 直接门控 ACC 内部计算，否则会因 acc_busy 导致死锁。完成结果需要稳定保持到输出空间可用。

首版不要求两个 group 独立握手。有效组作为同一个事务包原子接收；无效组不构成等待条件，不执行初始化或写回。reset 中断并清除整个未完成事务，不支持仅取消 INT16 的一个 phase 后继续使用中间状态。

#### 4.5.4 上层同步与排空确认

上层负责停止新事务发射、跟踪未完成事务并确认排空。MAC保留正常执行所需的级间valid、握手、ACC忙状态和INT16 phase控制，不要求将其额外汇总成对外idle信号，也不在MAC内增加长点积长度计数器或模式切换排空状态机。

在未被reset或异常中断的正常执行中，每次完整输入事务接收，必须且仅对应一次完整输出包握手：

~~~text
in_fire  = in_valid && in_ready
out_fire = out_valid && out_ready
outstanding_next = outstanding + in_fire - out_fire
~~~

outstanding由上层维护，表示已接收而输出尚未被接收的完整事务包数。计数器按最大在途事务容量选宽，同拍输入/输出事件净变化为0，检查不得上溢或下溢。DUAL无论一组或两组有效均按一个包计数；INT16输入只计一次，phase0不产生完成事件，phase1对应的输出握手才减一。算术完成或acc_commit本身不替代out_fire。

排空顺序为：上层停止对该MAC提交新事务，保持下游能够接收结果；MAC继续处理已接受事务；上层在停止提交且outstanding=0后确认排空，再更新模式并恢复提交。若有多个提交来源，计数和停止发射必须覆盖全部来源。排空不表示长点积必然完成，长点积完成由上层长度/调度信息判断；未完成长点积暂停时不能仅因outstanding=0就允许其他长点积覆盖同一ACC。

输出valid建立前，对应ACC状态更新必须已经完成；最后一个输出被接收后，不得仍有该事务遗留的架构状态写入。reset中断事务时，上层计数、MAC及相关输出队列按系统协议协调清除；异常取消/恢复必须另有同步处理，不能继续等待已取消事务的正常输出。

若out_ready来自上层结果FIFO，out_fire代表结果已移交该FIFO，MAC可据此被上层判定排空。后续存储写回或最终格式转换的完成由更高层同步跟踪；接收方必须保存数据及其解释所需的类型、指数和标识，不能由新配置重新解释旧结果。

### 4.6 双 output group 与累加状态隔离

逻辑上分别维护 ACC0、ACC1 的整数工作寄存器、补码浮点尾数/指数寄存器、有效/类型状态和输出标识。公共模式不得覆盖另一组的有效性或尺度。

状态存储是否在整数/浮点模式之间复用属于综合优化；行为模型先分开描述。两个组之间不共享加法 carry、舍入决策、饱和判断或错误恢复状态。

INT16/FP19 的 tree1 是 group0 的 limb 数据，不产生 group1 ACC 写使能。INT16 phase0 与 phase1 之间不得让其他事务覆盖 ACC0 工作状态。tag 仅验证和标识事务，不增加隐藏的多上下文能力。

### 4.7 时序划分、时钟门控与 operand isolation

预计最长组合路径分别位于 ALIGN 的 MAX→距离→移位→RNE，PSUM 的压缩网络→tree CPA→WIDE merge CPA，以及浮点 ACC 的对阶→加法→规格化→最终舍入。当前没有工艺或频率指标，不据此承诺单周期反馈吞吐。

仅对无效数值路径执行 operand isolation；寄存器使用规范的 clock enable 或 ICG。入口、级间 valid 和多周期控制逻辑不能因数据门控而停止必要的状态推进。

FP19 关闭 MUL9，但两颗 PSUM tree 仍工作；单输出模式关闭 ACC1，不关闭服务 WIDE_SUM 的高树。普通 INT8、MXINT8、INT4、INT16 隔离浮点指数 MAX 和对阶器；MXINT8 的 TC_FLOAT ACC 仍需开启。

### 4.8 功能阶段与物理流水的实现方法

#### 4.8.1 已确认约束与实现自由度

五个功能阶段用于描述算术职责和数据解释，不要求每个模块恰好在一个时钟周期内完成。已接受最终ACC状态隔若干拍更新；首版可采用多周期ACC及共同反压，保持每次dot16按顺序累加、每次更新RNE的现有合同。无需为了每拍更新引入多上下文或改变求和顺序。当前不指定两拍、三拍等固定周期数。

物理流水寄存器应依据目标工艺/器件、频率、面积和功耗预算，以及综合与布局后的路径延迟确定。ALIGN、PSUM和ACC可能需要切拍，M和MX是否能单拍完成也须验证。功能模块边界不必与全部寄存器边界重合；内部切拍不能改变ALIGN逐项RNE、PSUM精确归约、MX指数赋值及ACC逐次RNE的位置语义。本文原有tree/merge之间无寄存器的描述作为未切拍参考结构，不作为所有物理实现的强制限制。

#### 4.8.2 前向流水的候选切分

以下仅为时序探索位置，不是已冻结的拍数，也不保证任一行能满足某个目标频率：

| 功能块 | 可评估的寄存器位置 | 必须同时保存的内容 |
|---|---|---|
| M | 乘法结果边界；必要时乘法器内部 | 乘积、指数、非零标志和事务metadata |
| ALIGN | MAX之后，或移位与RNE之间 | 原乘积/工作数、Emax/距离、丢失位信息及metadata |
| PSUM | CSA层间、tree CPA之后、merge之前 | 完整sum/carry行或完整tree结果；WIDE两limb必须同事务对齐 |
| MX | 指数修正与贡献寄存器边界 | 完整贡献及匹配的真实LSB指数、metadata |
| ACC输入准备 | 新贡献非零/位置检测及精确工作表示之后 | 准备后的新贡献、指数、valid、first及tag |

前向事务之间没有累加反馈依赖时，增加流水深度可以增加延迟而仍保持每拍接收一个事务。例如三拍ALIGN可在不同物理级同时处理三个相邻事务；这里的“延迟三拍”不等于“每三拍才接收一个事务”。整数旁路也必须保持正确的事务顺序和metadata对齐，首版可采用与浮点路径匹配的延迟。

MX较轻时，可评估与相邻物理级合并，或将指数修正并行提前计算并延迟对齐；不改变其逻辑职责。保留独立边界也可能更便于验证和布线，不要求所有物理级延迟完全相等。新增寄存器须计入数据/metadata存储、时钟功耗和停顿控制，不能只比较组合逻辑面积。

#### 4.8.3 ACC反馈与握手

同一ACC满足C[n+1]=RNE_Q2.34(C[n]+S[n])，下一次更新依赖上一次已规格化并舍入的状态。简单地在反馈路径中插入寄存器，不能默认维持每拍更新。可把仅依赖新贡献的准备工作前移，缩短反馈路径；使用近似或重排加法则属于另行评审的数值变更。

首版可采用以下保守控制方式：acc_accept时捕获一次事务及旧状态，忙期间不接收下一次依赖更新；完成后若输出槽不可用则保持结果；槽可用时acc_commit更新ACC并保存完整快照。下游out_fire负责确认交付，与内部提交是不同事件。INT16 phase0只更新工作状态，不需要产生快照；phase1读取该工作状态后完成最终提交。DUAL有效组全部完成后共同提交一个输出包，无效组不构成等待条件。

M/ALIGN/PSUM/MX在ACC无法接收时可共同保持，ACC内部状态机仍独立前进。若某实现将反馈结果在L拍后提供给下一次更新，且无其他开销，则同一上下文的启动间隔至少为L拍；输出等待和控制气泡可能使其更长。缓冲只能吸收短时速度差，不能消除前端与ACC长期吞吐不匹配。

#### 4.8.4 工程评估顺序

先实现与当前逐dot16参考模型一致的功能和握手，再结合目标频率进行综合，识别真实关键路径，调整内部寄存器及物理周期。每个候选实现分别报告：无停顿延迟、启动间隔、同一ACC持续更新吞吐、反压行为、寄存器/算术资源数量及面积功耗。切拍后验证停顿下无丢失/重复、first仅生效一次、INT16 phase配对、DUAL共同提交，以及4.5.4节上层排空计数的一致性。具体物理周期数仍待实现评估。

## 5. 乘法单元的架构设计

### 5.1 MUL 之前的入口分类与格式转换电路

本节展开第二章规则，电路位于MAC内、M输入保持寄存器之前；M本体只接收有限补码。输入是16组 raw_a/raw_b[15:0]及源格式、group_valid和metadata，输出是16组A/B各26-bit payload。

#### 5.1.1 解包、分类比较器及替换 mux

每个独立源元素配置字段选择器及比较逻辑：

| 格式 | sign | exponent | fraction | Inf替换幅值编码 |
|---|---|---|---|---|
| FP16 | raw[15] | raw[14:10] | raw[9:0] | 16'h7bff |
| BF16 | raw[15] | raw[14:7] | raw[6:0] | 16'h7f7f |
| E4M3 | byte[7] | byte[6:3] | byte[2:0] | 无Inf分支 |
| E5M2 | byte[7] | byte[6:2] | byte[1:0] | 8'h7b |

源格式case内产生 exp_zero、exp_all_ones、frac_zero。FP16/BF16/E5M2：is_inf=exp_all_ones&&frac_zero，is_nan=exp_all_ones&&!frac_zero；E4M3：is_inf=0，is_nan=(exp==15)&&(frac==7)。is_zero=exp_zero&&frac_zero，is_sub=exp_zero&&!frac_zero。不能用扩展后的Eeff或FP19/FP9容器检测特殊值。

替换mux优先级：无效组→全零；NaN或±0→sign/exp/frac全零；Inf→保留原sign、选择表中源格式最大有限字段；其他有限输入→原字段直通。可选nan_seen/inf_seen由原始字段产生，仅随有效事务保存作诊断。

#### 5.1.2 有限字段到补码及payload

替换后的exp/frac再次提供zero/subnormal判定，或用与mux选择一致的分类控制：zero为H=0/Eeff=0，subnormal为H=0/Eeff=1，normal为H=1/Eeff=exp。按1.4节对指数高位补零及fraction低位补零，形成unsigned11/unsigned4幅值；再分别补一个正符号位、按sign条件取负，得到signed12/signed5。取负前后均不舍入、不改变bias。

按2.6节打包，组合过程先把26-bit容器赋零，再写有效字段。INT16/INT8/INT4经模式mux直接写原补码字段，绕过浮点比较器和条件取负。非法最小负整数编码仍属于输入合同违例，不默认钳位。

bias_sum由源格式组合查表得到unsigned8，独立于payload保存；源格式组合合法性、mx_enable合法性与当前事务一起检查。E8M0=255属于独立尺度合同检查，不套用元素NaN→0规则。

#### 5.1.3 并行数量、寄存器与门控

FP19每拍最多32个独立源元素（16组A/B），FP9每拍最多64个（16组A/B各两个lane）。首版独立接口必须提供相应转换吞吐；共享入口表示逻辑模块边界，不是用一个标量转换器串行处理全部元素。允许每个A/B端口在互斥模式下复用一套FP19或两套FP9转换逻辑；确定广播拓扑后才能按相同源操作数合并实例。

转换结果在in_fire时与模式、bias、scale、tag、valid、acc_first一起进入入口保持寄存器。该寄存器之后才是M输入mux和乘法器。INT16两次发射期间保持内容；停顿不能只保持数值而更新metadata。无效组数值输入隔离，控制握手继续推进。

关键路径为字段mux→源分类→替换mux→H/Eeff与扩展→条件取负→打包/分发→保持寄存器。独立接口当前不承诺广播扇出和面积收益；若增加入口流水寄存器，需显式更新端到端延迟和握手，不能用五逻辑阶段名称隐藏新增周期。

### 5.2 乘法单元接口与寄存器定义

每个前端的主要逻辑端口如下，时钟/复位、valid 和 metadata 与这些数值端口共同组成完整流水接口：

~~~systemverilog
logic [25:0] operand_a, operand_b;
logic [2:0]  compute_mode;
logic        int16_phase;
logic signed [23:0] product12;
logic signed [17:0] product9;
logic [8:0] exp_sum0;
logic [5:0] exp_sum1;
logic [1:0] product_nz;
~~~

M 末端寄存器保存完整 signed24/signed18 乘积，不提前饱和、舍入或打包成更窄公共算术数。输入 26-bit 容器始终按第二章分字段解包。

### 5.3 多精度操作数路由

| 模式 | MUL12 A/B | MUL9 A/B |
|---|---|---|
| FP19 | signed12 sig_A/sig_B | 0/0 |
| FP9 | lane0 signed5，各自符号扩展 | lane1 signed5，各自符号扩展 |
| INT8（含 MXINT8） | lane0 signed8，各自符号扩展 | lane1 signed8，各自符号扩展 |
| INT4 | lane0 signed4，各自符号扩展 | lane1 signed4，各自符号扩展 |
| INT16 phase0 | A_L/B_L | A_H/B_L |
| INT16 phase1 | A_L/B_H | A_H/B_H |

MUL12 的 mux 输出必须为 signed12，MUL9 为 signed9。INT16 低片先零扩展成 signed9，再适配端口；高片先按 signed8 提取，再符号扩展。

~~~systemverilog
// 举例：INT16 phase0 的物理端口，a_s16/b_s16 为已声明的 signed16。
mul12_a = $signed({4'b0000, a_s16[7:0]});
mul12_b = $signed({4'b0000, b_s16[7:0]});
mul9_a  = $signed({a_s16[15], a_s16[15:8]});
mul9_b  = $signed({1'b0, b_s16[7:0]});
~~~

示例赋值置于组合过程的对应 case 分支。所有分支先设置默认零值，非法模式不驱动未定义操作数，避免锁存器。禁止直接将低片 `$signed(a_s16[7:0])` 当作 A_L。

### 5.4 有符号部分积生成

~~~systemverilog
logic signed [11:0] mul12_a, mul12_b;
logic signed [8:0]  mul9_a, mul9_b;
logic signed [23:0] mul12_p;
logic signed [17:0] mul9_p;

assign mul12_p = mul12_a * mul12_b;
assign mul9_p  = mul9_a  * mul9_b;
~~~

乘法结果直接赋给完整宽度 signed 信号，不将乘法嵌入窄函数返回值、窄拼接或改变表达式上下文宽度的逻辑。M 末端在实际流水更新事件保存结果。

FP19 模式将 MUL9 两输入置零，并关闭其无效结果寄存器写使能。整数模式直接复用合法补码，不增加绝对值转换。

### 5.5 FP19 直接乘法与 INT16 两拍调度

FP19 使用 MUL12 直接完成 signed12×signed12，不再分成 INT16 式部分积。

INT16 保持同一组 A/B 和 metadata，依次发射 LL/HL 与 LH/HH：

~~~text
保持槽有效，phase0：等待 M 发射机会
phase0 发射成功：phase←1，操作数保持
停顿：phase、操作数、tag 均保持
phase1 发射成功：释放保持槽
~~~

两拍不能使用不同事务的 A/B。acc_first 在 phase0 按原值传递，phase1 强制为 0。最终左移 8 bit 在 PSUM 执行，M 不计算 T0/T1，也不提前重构完整 INT16 乘积。

### 5.6 FP9/INT8/INT4 低位宽乘法映射

低位宽模式固定两路，没有局部 dot2 或运行时 lane 交换。缩窄到数学 payload 前需要以下符号冗余检查：

| 模式 | 有效乘积 | MUL12 被删除高位 | MUL9 被删除高位 |
|---|---|---|---|
| FP9 | signed9 | [23:9] 应等于 bit8 扩展 | [17:9] 应等于 bit8 扩展 |
| INT8/MXINT8 | signed15 | [23:15] 应等于 bit14 扩展 | [17:15] 应等于 bit14 扩展 |
| INT4 | signed7 | [23:7] 应等于 bit6 扩展 | [17:7] 应等于 bit6 扩展 |

FP19 有效乘积为 signed23，MUL12 bit23 应等于 bit22。INT16 的各部分积位宽按第一章检查。这些是验证输入/实现合同的断言，不是检查失败后自动饱和的数值功能。

### 5.7 指数生成与输出字段组织

FP19：

$$
E_{sum,0}=\operatorname{zext}_{9}(E_A)+\operatorname{zext}_{9}(E_B)
$$

FP9：

$$
E_{sum,g}=\operatorname{zext}_{6}(E_{A,g})+\operatorname{zext}_{6}(E_{B,g})
$$

FP9 lane0 的 6-bit 和零扩展到公共 9-bit 接口，lane1 保留 6 bit。普通整数和 MXINT8 不生成浮点元素指数，输出占位零并隔离指数路径。

product_nz 由对应两个尾数非零与有效组共同生成，不必等待乘法结束后再比较乘积。FP19 的 lane1 非零标志固定为零，不能被 WIDE_SUM 的高树使用需求误解释为第二个浮点乘积。

### 5.8 流水、关键路径与 PPA

M 的数据关键路径为操作数 mux→乘法器，指数加法可并行完成。符号扩展主要使用连线，非法格式检测和特殊值替换在入口完成，不重复放入 M。

INT16 优先复用入口保持寄存器，避免为两拍再复制整组 A/B。完整输出寄存器便于验证，后续可根据静态模式约束综合删除冗余位，但不能以运行时打包改变 signed 语义。

## 6. 对阶电路的架构设计

### 6.1 指数表示与指数合同

浮点真实乘积为：

$$
value=P_{int}\times2^{E_{sum}-bias_A-bias_B-F_A-F_B}
$$

ALIGN 比较 E_sum，MX 扣 bias 并应用尺度。固定源格式组合及组内固定尺度保证对阶距离正确。目标小数位为 FP19 的 29 bit、FP9 的 21 bit，固定提升量分别为 9 bit、15 bit。

本版明确采用逐项 RNE 后求和：每项先量化为指定定点精度，PSUM 精确求这些量化值之和，ACC 保留对阶 tail，对与旧状态的和执行一次 Q2.34 RNE；最终输出格式转换是长点积结束后的独立步骤。该行为不是原始 dot16 无限精度求和后仅舍入一次。

#### 6.1.1 signed33→signed32 的重定标推导

源尾数各含10位小数，|P|≤2047²=4190209<2^22。旧项 X_old=RNE(P×2^(10−d))，真实 LSB 为 Emax−bias_sum−30。新项直接计算 X_new=RNE(P×2^(9−d))，真实 LSB 为 Emax−bias_sum−29：量化前系数减半、LSB 权重加倍，数值相同。

最大 |P<<9|=2145387008<2^31，故新项 signed32 Q3.29 足够；16项最大绝对和为34326192128<2^35，故 PSUM signed36 Q7.29 足够。符号正负对称。d≥32 时幅值严格小于0.5，RNE为零；每项误差≤0.5个新LSB，16项误差≤8个新LSB。

新 RNE 必须直接作用于乘积提升后的值；不能截掉旧项一位，也不能旧 RNE 后再 RNE。反例 P=11,d=12：旧 RNE(2.75)=3，再 RNE(3/2)=2；新直接 RNE(1.375)=1。P=3,d=10：新+1.5应为+2，简单截断会得到+1。正负 tie 和这些反例由当前 Python 测试验证。

以上证明了缩放、范围和指数补偿；它是降低一位 ALIGN 小数精度的新量化合同，不是与旧量化合同逐位等价。RTL 已应用该推导，逐项和级联测试对照新参考模型。

### 6.2 每 lane 乘积指数生成与输入检查

M 已生成指数和，ALIGN 不重复计算。FP19 使用 16 个 signed23 有效乘积及 unsigned9 E_sum；FP9 每组使用 16 个 signed9 有效乘积及 unsigned6 E_sum。

完整物理乘积先检查符号冗余再提取 payload。非零标志与数据同拍进入 MAX；上游补零位置自然不参与 MAX。MXINT8 走整数直通，不按 FP9 构造隐藏位或元素指数。

### 6.3 每 output group 最大指数选择

采用四层二输入比较选择网络，节点数量依次为 8、4、2、1。每个节点携带 has_nonzero 与 max_exp。

~~~text
两侧均无效：输出 has_nonzero=0，max_exp=0
仅一侧有效：选择该侧
两侧均有效：选择较大指数，平局可固定选择任一侧
~~~

FP19 仅 group0 启用 MAX；FP9 两组各一套。禁止 32 个乘积共用一个全局 MAX。全零组保存 group_has_nonzero=0 和 Emax=0，所有对阶输出置零。

group_has_nonzero 不等于归约结果非零：有效正负项可能完全抵消。

### 6.4 对阶距离计算与显式位宽扩展

对有效非零项：

$$
d_i=E_{max}-E_i
$$

| 模式 | 有效指数差上界 | 数学最小距离宽度 | RTL 实现 |
|---|---:|---:|---|
| FP19 | 506 | unsigned9 | MAX 和 E 均零扩展至 10 bit 后减法，检查非负并保留低 9 bit |
| FP9 | 58 | unsigned6 | 扩展至 7 bit 后减法，检查非负并保留低 6 bit |

同组同拍 MAX 必须不小于当前有效项。零项直接输出零，不使用占位指数生成无意义位移。

构造提升后 Y：

$$
Y_i=P_{int,i}\ll K
$$

| 模式 | 有效乘积 | K | Y 数学宽度/接口 |
|---|---|---:|---|
| FP19 | signed23 | 9 | signed32 |
| FP9 | signed9 | 15 | signed24 |

先符号扩展到 32/24 bit 再移位。尾数有效范围保证结果不溢出，不增加额外隐藏舍入位。

### 6.5 多精度对阶窗口与移位器组织

每个前端物理配置两套独立的距离计算、移位及 RNE 电路：lane0 为 32-bit 路径，lane1 为 24-bit 路径。全 MAC 共 16 条 lane0 和 16 条 lane1 路径，FP9 两路同拍对阶，禁止 lane1 通过模式 mux 或时间复用占用 lane0 移位器。只有 lane0 自身的 FP19/FP9 两种互斥模式允许共享同一 32-bit 移位器。FP9 在 lane0 中先把 signed24 Y 符号扩展为 signed32，不额外左移8位；有效量化宽度仍按 W=24 判断。lane1 的 FP9 使用独立 24-bit 移位器。

对 W=32 或 W=24 的 Y：

~~~text
d=0：不移位、不量化
0<d<W：算术右移，保留被移除位供 RNE
d≥W：在当前合法输入范围下 RNE 结果为零
~~~

先比较完整距离与 W，再将有限范围内的距离送入 barrel shifter。禁止仅截取距离低位，否则大距离会模回绕。

移位器无效输入可以隔离，但必须同时清除该项的有效性及量化诊断输出。

### 6.6 逐项 RNE、Guard、Round 与 Sticky

对 0<d<W，定义向下取整商 q=Y>>>d。Y 的低 d 位是相对于 q 的非负余数，因此补码可以直接使用以下公式：

~~~text
G = Y[d-1]
R = d≥2 ? Y[d-2] : 0
S = d≥3 ? OR(Y[d-3:0]) : 0
inc = G AND (R OR S OR q[0])
X = q + inc
~~~

例如 −1.5 的 q 为 −2，余数恰好一半且 q 为偶数，不增加；−0.5 的 q 为 −1，余数恰好一半且 q 为奇数，增加为 0。公式对正负数一致实现 ties-to-even。

q 先符号扩展到 W+1 bit，inc 零扩展到 W+1 bit，再相加；检查冗余符号位后保留 W bit。d=0 时 X=Y、inexact=0；d≥W 时 X=0，inexact=(Y!=0)。

实现 G/R/S 时不能使用非法或可变宽度的 part-select。可综合实现可用常量上界循环，对索引满足 j<d−2 的位做 OR，G/R 使用受分支保护的动态 bit-select；或采用分级移位器同步合并 lost 标志。必须先处理 d=0/1 等边界，避免 unsigned 的 d−2 下溢。

对 0<d<W，inexact 为原始低 d 位的 OR。组内 OR 得到 align_inexact，只作为量化误差诊断，不得把它作为一个数值 sticky 注入 PSUM/ACC。

每项量化误差不超过半个输出 LSB，16 项和的绝对误差上界为 8 个该 LSB；抵消时相对误差仍可能很大。该精度合同不由 ACC 最终舍入补救。

### 6.7 对阶输出寄存器

~~~systemverilog
logic signed [31:0] align_lane0_q [0:15];
logic signed [23:0] align_lane1_q [0:15];
~~~

| 模式 | lane0 寄存器 | lane1 寄存器 |
|---|---|---|
| FP19 | signed32，单项含29位小数 | 无效，关闭数值更新并隔离 |
| FP9 | signed24 RNE 结果符号扩展至32位，仍含21位小数 | signed24 RNE 结果，含21位小数 |
| INT8/MXINT8、INT4 | MUL12 结果符号扩展至32位 | MUL9 结果符号扩展至24位 |
| INT16 | LL/LH 符号扩展至32位 | HL/HH 符号扩展至24位 |

整数旁路不执行浮点 RNE。FP9 lane0 检查 [31:24] 为 bit23 的符号扩展。寄存器总数值位宽为16×56=896 bit；不能以两个 signed24 tree 输入寄存器替代这一边界。

FP19 拆片只发生在 PSUM 输入组合逻辑中。ALIGN 末端同时保存 group_valid、WIDE/DUAL 模式、phase、Emax、bias/scale、tag、acc_first 和可选 align_inexact，统一停顿。

### 6.8 时序与移位器资源复用

主要路径为每组 MAX→各项指数差→独立可变移位→RNE 修正。lane0 的 FP19/FP9 模式共享不涉及 lane1；两条路径同时有输入时必须同拍产生结果。FP19 limb 拆分在下一级 PSUM 内，主要为连线。门控只关闭当前模式不使用的路径，不能关闭 FP9 的第二路。

未规定目标频率，不隐式将 MAX 或 RNE 拆成新流水级。需要调整阶段边界时，应保持五逻辑阶段及事务 metadata 一致，并单独评审物理延迟。

## 7. 部分和电路的架构设计

### 7.1 两颗物理 limb tree 的接口

PSUM 从 ALIGN 寄存器接收16个 signed32 lane0 和16个 signed24 lane1。模式适配后，每颗 tree 的逻辑输入为16个 signed24 组合项；接入固定28-bit DesignWare 输入总线前，逐项按其有符号语义扩展到 signed28。FP19 低 limb 先零扩展成正的 signed24，高 limb 符号扩展成 signed24：

~~~systemverilog
logic signed [23:0] term [0:15];
logic signed [27:0] tree_sum; // PSUM 内部组合信号，不是独立级间寄存器
~~~

16 个任意 signed24 的真实和落在 [-2^27, 2^27−16]，signed28 足够。两颗 tree 的内部计算互不传递 carry。其28-bit输出先进入模式选择/重构逻辑，最终才写入56-bit PSUM输出寄存器。DUAL_OUTPUT 的 lane0 先检查高8位符号冗余再取 signed24；lane1 直接进入 tree1。INT16 同样提取合法部分积的 signed24 表示。

### 7.2 物理输入、有效 payload 与输出位宽

| 模式/树 | 单项有效表示 | 16 项和数学表示 | 物理输出及有效段 |
|---|---|---|---|
| FP9 | signed24 | signed28 | 全部 [27:0] |
| INT8/MXINT8 | signed15 | signed19 | [18:0]，其余符号扩展 |
| INT4 | signed7 | signed11 | 接口 [11:0] 为 signed12；bit11 为冗余符号 |
| INT16 phase0 低树 | signed17 非负 | unsigned20 或 signed21 | [20:0]，其余符号扩展 |
| INT16 其他部分积树 | signed16 | signed20 | [19:0]，其余符号扩展 |
| FP19 低树 | unsigned23 | unsigned27 或 signed28 | [26:0] 为幅值，bit27=0 |
| FP19 高树 | signed9 | signed13 | [12:0]，其余符号扩展 |

每个输入先适配 signed24，再扩展到内部算术宽度。低片的零扩展是数值操作，不能被高位未定义值或负数解释替代。

### 7.3 DUAL_OUTPUT 模式的数据组织

FP9、INT8/MXINT8、INT4 分别计算：

$$
sum_0=\sum_i term_{0,i},\qquad sum_1=\sum_i term_{1,i}
$$

两树之间没有加法、共同饱和或跨组舍入。每组的 sum、valid、tag、scale、Emax、acc_first 随本组数值一起保存。普通整数和 MXINT8 的树结果相同，区别由后续 acc_kind 和尺度解释决定。

### 7.4 WIDE_SUM 模式的 limb 拆分与重构合同

#### 7.4.1 FP19

对 ALIGN 的每个 signed32 X，在 PSUM 内拆为 unsigned23 L_i=X[22:0]、signed9 H_i=X[31:23]，满足 X=(H_i<<23)+L_i：

~~~systemverilog
low_term_s24  = $signed({1'b0, x_s32[22:0]});
high_term_s24 = $signed({{15{x_s32[31]}}, x_s32[31:23]});
~~~

低片零扩展、高片符号扩展后分别进 tree0/tree1。低片最大位数 B 满足 B+4+1≤28，故 B=23；不能把低28位全部当作单项低片。X=−1 时低片为2^23−1、高片为−1，重构仍为−1。

$$
L=\sum_i L_i,\qquad H=\sum_i H_i,\qquad S_{36}=(H\ll23)+L
$$

低树完整保留 unsigned27 的和：

~~~text
[22:0]  ：低片求和后的低23位
[26:23] ：归约增长产生的高位
[27]    ：0，非负signed结果的符号位
~~~

[26:23] 不得删除，也不需要传入高树。它们与高树移位结果在 PSUM 的显式 merge CPA 中自然合并。禁止将两个树结果直接拼接，或只取低树 [22:0] 再拼高树。

#### 7.4.2 INT16

每拍低树归约 LL 或 LH，高树归约 HL 或 HH，PSUM 执行 T=L+(H<<8)，并为 phase1 的 T1 再施加整体 2^8 权重。两拍的所有部分积均属于 group0。

FP19 和 INT16 的 limb 位移不同，必须根据随事务传递的模式选择 23 或 8，不能共用一个固定移位常量。

#### 7.4.3 FP19 低23位拆分的重构

低树取 unsigned27 L，高树检查并取 signed13 H。新数学结果为 signed36 Q7.29。先在 signed37 安全域计算 `(sext37(H)<<23)+zext37(L)`，检查 bit36=bit35 后取 signed36；这一步只删除冗余符号位，没有数值舍入。低树完整 [26:0] 参与相加。

#### 7.4.4 INT16 重构和第二拍权重

两条 signed28 物理结果先符号扩展到 signed37，再执行 T=L+(H<<8)。phase0 检查并提取 signed28 T0，phase1 检查并提取 signed27 T1。

~~~text
phase0 contribution36 = sext36(T0)
phase1 contribution36 = sext36(T1) << 8
~~~

第二拍整体移位只在 PSUM 执行一次。ACC 接收的 contribution 已带权重，不重复左移。旧 PSUM 的37位兼容视图可保留冗余符号；新 MX→ACC 整数输入提取 signed36，数值权重不变。

### 7.5 归约树结构与有符号扩展

首版 RTL 采用两颗 DesignWare `DW02_tree`，每颗设置16个输入、28-bit输入宽度，并接一颗28-bit `DW01_add` 完成末级进位传播。每个 `DW02_tree` 的 `INPUT` 总线宽448 bit，`OUT0`、`OUT1` 各28 bit；对应的 `DW01_add` 输入这两行，`CI=0`，`SUM` 是 tree 的 signed28 精确结果。RTL 将 `verif_en` 显式设为1；该参数控制 DesignWare 的验证模式，不改变本节的数值合同。`CO` 不参与结果；合法输入范围保证28-bit模和就是所需 signed28 结果，`CO` 不作为 signed overflow 标志使用。

~~~text
逻辑数据流：16 个操作数 → DW02_tree 两个 carry-save 行 → DW01_add CPA → signed28 和
~~~

逻辑上每个 3:2 压缩节点计算：

~~~text
s = a XOR b XOR c
c = ((a AND b) OR (a AND c) OR (b AND c)) << 1
~~~

输入位串按28-bit 二进制补码解释；压缩行按28-bit模表示，移出的 bit28 不属于输出行。对本 IP 的合法项，最终数学和由输入范围证明能容纳于 signed28。

不能要求每一级 CSA 的最高 carry 为零，也不能将 CSA 中间行当作已完成的 signed 部分和检查符号冗余。最终 CPA 输出才是普通 signed28 和。

`DW02_tree` 是本版选定的实现结构；换成其他综合等价归约网络仍属于实现选择，前提是不改变模式适配、精确归约结果、位宽和级间接口。该选择不冻结厂商以外的内部压缩拓扑，也不构成新的数值合同。

### 7.6 carry 边界与符号冗余检查

在对应有效模式下检查：

~~~text
INT8/MXINT8：sum[27:19] 全部等于 sum[18]
INT4：sum[27:11] 全部等于 sum[10]
FP19低树：sum[27]=0
FP19高树：sum[27:13] 全部等于 sum[12]
INT16 phase0低树：sum[27:20] 全部为0（unsigned20有效、signed21接口）
INT16其他树：sum[27:20] 全部等于 sum[19]
~~~

这些检查不增加跨树修正功能。FP19 低树 [26:23] 是合法增长位，不能把它们误判为溢出。PSUM 不执行饱和、舍入或量化截断。

### 7.7 56-bit PSUM 输出寄存器与元数据

~~~systemverilog
logic [55:0] psum_d, psum_q; // unsigned 打包容器
always_comb begin
    psum_d = '0;
    if (wide_sum_mode)
        psum_d = {{19{wide_sum_s37[36]}}, wide_sum_s37};
    else
        psum_d = {tree1_sum_s28, tree0_sum_s28};
end
~~~

WIDE 模式 wide_sum_s37 已由 PSUM 完成重构：FP19 为 signed36 Q7.29，符号扩展到37位容器；INT16 为符号扩展后的 T0 或 T1×2^8。高[55:37]固定为bit36的符号扩展。只有重构之后才允许下游取低37位，禁止拼接原始limb后取低37位。

DUAL 模式 [27:0] 为 group0、[55:28] 为 group1，各自按 signed28 解包。无效组可以保持其半寄存器或写零，但 valid=0 必须隔离旧数据。WIDE 写入必须覆盖全部56位，不能因 group1_valid=0 而禁止高半寄存器更新。

数值与 metadata 同步寄存。WIDE 两树来自同一事务，只输出 group0_valid，group1_valid=0；tree1 活动不代表 ACC1 有效。DUAL 各组独立保留 Emax、scale、tag 和 acc_first。

当前 `mac_psum_stage` 对无效 DUAL group 写入零，并仍以 `group_valid` 隔离该数据。PSUM 使用一个弹性输出寄存器：`space=!o_psum_valid||i_psum_ready`，ALIGN 输入握手为 `i_align_valid&&o_align_ready`；满槽可在同一拍消费旧结果并装入新结果。停顿期间56-bit数据及其 metadata 一起保持。`align_inexact` 随事务透传，仅供诊断，不进入数值计算。复位为低有效同步复位。

两颗28-bit tree输出在当前未切拍参考结构中为内部组合节点；后续可按4.8节在PSUM内部增加物理流水寄存器，须明确保存内容和延迟。56-bit是物理接口宽度，DUAL有两个28-bit数值，WIDE数学贡献为signed36；旧37位视图的bit36为冗余符号。

### 7.8 时序、门控与 PPA

FP19/INT16 两树都工作；低位宽 DUAL_OUTPUT 中无效组可隔离本树输入并关闭结果更新。不能因为 FP19 关闭 MUL9 就关闭高树。

统一 24→28-bit 接口覆盖所有模式，窄有效位可由综合静态优化。动态门控不能把负数符号扩展高位强制为零。WIDE 关键路径为模式适配→DW02_tree→DW01_add→merge CPA→打包寄存器，相比原来把 merge 放在 MX，会加长 PSUM 路径。无目标频率时先保证功能；后续可评估其他等价压缩实现，或显式评审增加物理周期。不能偷偷在tree和merge之间新增寄存器。

### 7.9 RTL 实现与验证边界

当前 `align_lane0_t` 为 signed32；FP19 ALIGN 执行 P<<9 后逐项 RNE。PSUM 高 limb 为 signed9、高树为 signed13，在 signed37 域合并低23位 limb 后检查冗余并提取 signed36 Q7.29。独立 PSUM、真实 ALIGN→PSUM 及完整链路均对照新模型。

Icarus 使用 hardware/verification/tb/dwbb_sim_models.sv 的 test-only 功能模型验证 tree 封装；不是 Synopsys DesignWare 模型、综合或时序证据。本轮执行结果见 hardware/verification/README.md。

## 8. MX 电路的架构设计

### 8.1 功能边界与公共表示

MX 接收56-bit PSUM、mode/phase、group_valid、Emax、bias、E8M0 scale、first、tag。浮点输出统一为 signed36 Q2.34 系数 M 和 8-bit bias127 指数 E：

~~~text
value = M × 2^(E − 127 − 34)
~~~

MX 不重新重构 WIDE limb，不重复 INT16 phase1 权重，不饱和普通整数以外的路径，不在此计算 MXINT8 的最高有效位位置 k。浮点 MX 输入无需先规格化；E 不是最终 FP32 输出字段。

### 8.2 解包与整数路径

DUAL 的低/高28位分别提取 S0/S1。WIDE 的 signed36 数学贡献符号扩展到56位；保留旧37位视图时 bit36 必须等于 bit35。FP19 归约为 Q7.29；INT16 的 phase0 T0 和 phase1 T1<<8 已在 PSUM 完成。

普通 INT8/INT4 仍分别检查 signed19/signed12，再送 INT32 饱和 ACC；INT16 提取 signed36 送两 phase 路径。普通整数指数占位为0，不应用 E8M0。

### 8.3 MXINT8：signed19 Q2.17 到 Q2.34

合法整数编码 q∈[−127,127]，元素值为 q×2^(s−6)。单项乘积范围 ±16129 可放 signed15，dot16 范围 ±258064 可放 signed19。令 S_int 为精确整数和：

~~~text
value = S_int × 2^(s_A+s_B−12)
M = S_int << 17                 // signed19 解释成 Q2.17，补17个低零位
E = s_A+s_B+132
acc_kind = TC_FLOAT
~~~

保值证明：17+(s_A+s_B+132)−127−34=s_A+s_B−12。M 为 signed36，无最高位检测 k，无 INT32 预饱和。零和也处理 valid、first、tag。

### 8.4 FP9 与 FP19 转换证明

| 模式 | PSUM 系数与格式 | MX Q2.34 系数 M | 数学 E |
|---|---|---|---|
| FP9 | signed28 Q7.21 S | S<<8 | Emax−bias_A−bias_B+132 |
| MXFP8 | 同 FP9 | S<<8 | Emax−bias_A−bias_B+132+s_A+s_B |
| FP19 | signed36 Q7.29 S，推导见§6.1 | S（不移位） | Emax−bias_A−bias_B+132 |
| MXINT8 | signed19 整数和，重解释 Q2.17 | S<<17 | s_A+s_B+132 |

FP9 原真实 LSB 指数 e=Emax−bias_A−bias_B−21+s_A+s_B。先将 signed28 系数解释为 Q2.26，E=e+26+127；补8个零转到 Q2.34 后，8+E−161=e。Emax=32、bias_A=bias_B=7、scale=0 时 e=−3、E=150。

FP19 新 ALIGN 的 F=29；S 的真实指数为 Emax−bias_sum−29。重解释为 Q2.34 时系数原样保留，E=Emax−bias_sum−29+34+127=Emax−bias_sum+132。相对旧 ALIGN 的 F=30，真实 LSB 补偿是+1；相对新 Q7.29，Q2.34 指数补偿是+5。这两项不能重复相加。

上述保值是相对于各自 PSUM 的精确二进制值；不宣称新 FP19 逐项 RNE 结果等于旧33位 ALIGN 的结果。

### 8.5 E8M0 与指数范围

scale_a/b 是原始 unsigned8 E8M0 字节，有限 raw∈[0,254] 解码为 s=raw−127。raw=0 表示2^−127，raw=255 的有效事务策略未冻结。无尺度模式不使用这些字节。两组 scale 和 Emax 分别计算，不允许跨组共享 MAX。

非零源项的数学 MX E 范围如下；这不是可直接编码的8位字段范围：

| 模式 | 数学 E 范围 |
|---|---|
| FP16 | 104…162 |
| BF16 | −120…386 |
| E4M3×E4M3 | 120…148 |
| E5M2×E5M2 | 104…162 |
| 混合 E4M3/E5M2 | 112…155 |
| MXFP8（所有支持组合） | −150…416 |
| MXINT8 | −122…386 |

Emax 全零占位0不在上述非零范围内。零状态 M=0/E=0。数学指数运算保留符号及进位，不能直接取低8位；Python `Mx.exp` 暂存完整数学 E。按用户要求，非零 MX 输入 E 不在1…254时暂报 `UnfrozenContract`，不重定标或给出越界结果；零贡献不依赖其占位指数。

用户已确认：内部正常指数 E=1…254 对应 −126…127；ACC 写回下溢 FTZ 为0，上溢保留有符号 Inf，零采用 M=0/E=0。E=255 用于逻辑 Inf；不保存 subnormal 状态。MX 未规格化输入 E 越界按用户要求暂不考虑，不能沿用 signed12 状态覆盖全范围。

### 8.6 group、流水与 metadata

group0→ACC0、group1→ACC1；FP19/INT16 仅 group0。每组独立传递 E、M、scale、Emax、first、tag。共同 ready/valid 控制整包消费；无效组不写状态。MX 弹性槽的停顿保持、满槽消费/补入和同步复位语义保持。

### 8.7 舍入边界

FP9 和 MXINT8 补零、FP19 重解释均精确；MX 不执行丢位舍入。ALIGN 已完成每项 RNE，诊断 inexact 不能作为数值 sticky 注入 ACC。ACC 对阶 tail 参与和的最终 RNE，见§9。

### 8.8 RTL 接口与暂缓域

`mac_mx_stage.sv` 的两组 `o_mx_contribution0/1` 均为 signed36；`o_mx_exp0/1` 均为 unsigned8 bias127，删除旧 lsb_exp 命名及类型。普通整数保持整数系数和 E=0。浮点零贡献统一输出 M=0/E=0；Python MX 保留的数学占位 E 不影响零值，testbench 在编码边界规范为0。

指数运算先在 signed13 中完成。非零贡献 E 不在1…254时，仅在有效握手触发仿真合同断言；该输入域仍暂缓，不定义综合后恢复行为。不得将输出字段的低8位视为该非法事务有意义的回绕值。

## 9. 累加电路的架构设计

### 9.1 两组状态与公共操作数

每组浮点输入和状态均为 signed36 Q2.34 M + unsigned8 bias127 E，value=M×2^(E−161)，共44位有限数数值字段，当前 RTL 另存每组 signed2 Inf 符号。FP9、FP19、MXINT8 均使用此表示；普通 INT8/INT4 和 INT16 的架构状态仍为 INT32。

ACC0/ACC1 独立，不跨组比较指数、相加或共享 tail。first 仅在该组实际消费时选零旧值；无效组保持；有效零贡献仍提交 metadata。INT16 仅 phase0 使用 first，phase1 完成才产生完整事务快照。

正常非零写回采用 2^34≤|M|<2^35；signed36 负端点 −2^35 可作为输入，经精确重定标成为 −2^34、E+1。零为 M=0/E=0，无负零。Inf 是独立逻辑特殊状态：E=255，符号由 `infinity_sign=±1` 明确携带，参考模型的 sig=0 仅为占位，不代入有限数公式；RTL 使用独立 `o_acc_float_inf_sign[g]` signed2 字段：0为有限/零，+1为+Inf，−1为−Inf；Inf 时 float_sig=0、float_exp=255，不挤占有限系数。

### 9.2 整数 INT32 路径

普通INT8/INT4执行：

$$
W_g=\operatorname{sext}_{36}(C_{old,g})+\operatorname{sext}_{36}(S_g),\qquad
C_{new,g}=\operatorname{sat}_{INT32}(W_g)
$$

数学完整结果需要signed33，安全加法及工作寄存器采用signed36。INT4接口signed12、INT8接口signed19先显式扩位。每次有效dot16更新后饱和，无效组不更新。

INT16接收PSUM已重构并施加phase权重、经MX直通的贡献。公共signed37输入检查bit36=bit35后提取signed36：

~~~text
phase0：work36 ← sext36(C_old) + contribution0
        不截断、不饱和、不提交
phase1：wide36 ← work36 + contribution1
        C_new ← sat_INT32(wide36)
        提交，并将C_new符号扩展回work36
~~~

phase1绝不再次左移；phase0使用acc_first，phase1读取配对的中间状态。完整dot16加旧INT32可由signed36容纳。饱和在signed36域比较36'sd2147483647与-36'sd2147483648，越界输出32'h7fffffff/32'h80000000，其余取低32位并检查高4位为bit31符号扩展。MXINT8选择TC_FLOAT，不经过本节饱和。

### 9.3 指数边界与有限数工作域

内部正常指数 E=1…254、bias127，系数精度为 Q2.34。最终 RNE 及进位修正后，E<1 时 FTZ 写 M=0/E=0；E>254 时保存有符号 Inf（E=255）。E=0 不保存非零 subnormal；当前输入源特殊值仍先按第二章转换为有限值。

旧 Inf 加任意合法有限贡献仍为原符号 Inf，first 先清除旧状态，因而可以从 Inf 重新开始。MX 输入只接有限贡献，不新增 Inf 输入通路或 Inf−Inf/NaN 规则。非零 MX 输入 E 超出1…254仍暂报 `UnfrozenContract`，与 ACC 写回越界已定义的 FTZ/Inf 行为区分。

BF16 乘积和 MX scale 的合法源输入可超出此范围。8-bit 状态不能覆盖原 signed12 方案的所有数学值；不得继续引用旧 signed12/signed10 的全范围估计证明新状态安全。指数计算使用完整数学整数，最后才能根据冻结规则编码。

### 9.4 对阶保留 tail，不分别舍入操作数

令共同指数 Ec=max(E_old,E_new)，零加数不参与指数选择。非零加数在 Ec 网格上的精确系数分解为：

~~~text
d = Ec − E
q = floor(M / 2^d)
r = M − q×2^d，0 ≤ r < 2^d
tail = r / 2^d
~~~

q 可用 signed36 表示。负数同样采用 floor 商和非负余数。两路 tail 必须参与最后舍入，不能先各自 RNE 到 signed36 再相加。任意深抵消时仅保留一个 OR sticky 未必足够；参考模型保存精确有理 tail，不据此冻结硬件 GRS 位数。

### 9.5 36+36→37 位完整和

Z=q_old+q_new 以 signed37 完整保存，精确结果为 (Z+tail_old+tail_new)×2^(Ec−161)。即使 Z=0，也必须检查 tail，不能误判抵消为零。按 Ec=max(E_old,E_new) 的分解至少一路 tail 为0；另一侧余数仍可能改变最终 RNE 或深抵消结果，硬件需要在所选扩展窗口内证明其保留规则。

可以在36位操作数之后增加低位工作位，或以37位和加额外尾部信息完成计算；补零只改变工作网格，不能单独替代舍入。新增工作位不计入36位架构状态。

### 9.6 最终一次 RNE 与规格化

对含 tail 的精确和确定规格化位置，再按目标 Q2.34 网格执行 ties-to-even。floor 商 q 和余数 r、除数 D 的进位条件为 2r>D 或 (2r=D 且 q 为奇数)。先决定进位，再舍去低位；不是直接截断原结果。

RNE 产生 |M|=2^35 时，精确右移一位并令 E+1，不再舍入一次。同号增长、异号深抵消和负二次幂都需验证。精确零写 M=0/E=0；进位和左规格化后按§9.3执行 FTZ/Inf，禁止静默回绕。

参考模型以最细输入网格的任意精度整数精确求和，再执行一次 RNE；同时提供 signed36 操作数、signed37 full_sum 和精确 tail 的诊断记录，用于核对算术关系。它不模拟物理移位器或周期。

#### 9.6.1 当前 RTL 的三位工作尾部及充分性

`mac_acc_float_prepare36` 先在 ACC 内对两路有限系数精确重定标，使幅值位34为最高有效位；不在 MXINT8 计算 k。任意非零 signed36 系数可精确左移到该位置；负端点 −2^35 只需精确右移1位、E+1。临时指数保留 signed11，不在此执行 FTZ；合法编码输入对应的临时 E 范围为−33…255。

在规格化幅值后补3个低零位，再按新的共同指数进行 magnitude shift-right-jam：右移结果 bit0 合并所有更低位的 OR。它不是逐操作数 RNE，也不把每路先舍入到36位。重新施加符号后，两路 signed39（36+3）相加为 signed40（37+3），保留核心37位和及3个低位工作位。整数分支同样补3个零复用本组加法器，结果取[39:3]后进行原 INT32 饱和或 INT16 中间保存。

充分性分为两种情况：

- 规格化指数差 d≤3：补3零后的右移完全精确，因此可能的深抵消不丢位，结果可以任意左规格化。
- d≥4：较小幅值小于较大幅值最小值的1/8，异号相减的幅值仍大于7/8个规格化单位，最多左规格化1位。最终 RNE 至少丢掉2个低位工作位；右移 jam 保留相对于该舍入网格的低于半、恰好半、高于半关系。加法或减法均不因压缩最底部信息而越过 RNE 判决边界。

最终取和的幅值完成一次 RNE，再恢复符号；舍入进位只做精确重定标。该实现与参考模型的 floor 商/精确 Fraction tail 中间编码不同，比较目标是最终状态及精确数学合同；不要求两个内部 trace 逐位相同。定向距离、逐位抵消、正负 tie 与随机状态序列用任意精度模型作独立预言机。此范围论证和仿真不等于形式验证或时序收敛。

### 9.7 写回、快照与最终转换

仅实际 commit 更新有效组；first、tag、group_valid 与该事务同步。浮点快照必须完整保留36位 M 与8位 E，不能用旧32位 out_data 截掉系数高位。Python Snapshot.data 对有限浮点保存36位补码编码，整数仍为32位编码。无效组字段为 None；有效零为 data=0/E=0；有效 Inf 为 data=None/E=255/infinity_sign=±1，有限值 infinity_sign=0。Inf 的 data=None 防止把占位误当零；RTL 状态观察口以 float_sig=0/float_exp=255/float_inf_sign=±1 输出 Inf；系统最终输出包和格式转换器仍未实现。

输出保持槽在反压期间稳定，不被后续写回覆盖。每个 dot16 提交一次快照；INT16 phase0 不提交、phase1 提交。仍由系统选取完整长点积的最后快照进行最终转换，不新增 acc_last 或目标格式握手协议。

### 9.8 长点积结束后的输出转换合同

本节是ACC之后的数值边界合同，不把转换器塞回每次ACC更新反馈路径，不新增第六个MAC计算阶段。转换器的目标格式集合、系统放置位置和控制接口未冻结；以下对支持Inf的二进制浮点目标定义通用行为，不自动扩展到没有Inf的E4M3输出或整数输出转换。

对最终 Inf 直接输出对应符号 Inf；对有限非零C_sig：输出符号为C_sig[35]，可在转换器中先符号扩展到signed37再条件取负，得到unsigned36幅值。这是最终封装使用的幅值转换；ACC 的局部幅值规格化另见§9.6.1。输入真实LSB指数为E_C−127−34。

目标有p位有效精度、最小正常指数emin、最大正常指数emax时：

$$
e_{target}=\max(E_C-127-(p-1),\ e_{min}-(p-1))
$$

按e_target相对E_C−127−34的位置，一次性执行目标精度RNE；精度更宽时精确补零，不恢复此前丢失信息。不能先按normal精度舍入再执行subnormal二次舍入。完成进位规格化后分类：

| 目标舍入结果 | 输出行为 |
|---|---|
| Normal | 编码目标sign/exponent/fraction |
| Subnormal | FTZ为零，保留原非零结果符号 |
| 非零输入舍入为零 | 保留原非零结果符号 |
| 舍入溢出 | 对支持Inf的目标输出带符号Inf |
| 输入内部精确零 | 输出+0 |

边界经RNE变成最小normal时保留normal。以FP32输出为例，p=24、emin=−126、emax=127、bias=127，最小subnormal量化步长2^-149；输出转换与 ACC 内部扩展精度不同；ACC 已明确使用 FTZ/Inf，但系数精度仍为 Q2.34；不能以最终 FP32 转换结果替代内部反馈。输入subnormal Preserve与最终输出FTZ是两个不同边界。

目标含NaN编码也不意味着本内部有限状态会生成NaN。没有Inf编码的输出格式溢出策略、浮点到整数的转换范围、目标格式选择及最终完成标志仍需另行确认。可选output_ftz/overflow诊断只在转换器产生，IEEE异常标志和tininess时点未冻结。

### 9.9 门控、反馈与物理周期

保留原有共同握手、ACC0/ACC1 独立性和 INT16 配对语义。单输出关闭 ACC1 写使能；MXINT8 虽使用整数乘法模式仍启用浮点累加。多周期实现必须捕获一次新旧操作数，在 commit 前阻止同组依赖更新读取旧状态；内部忙状态在上游停顿期间仍须前进。

反馈路径为指数比较、保留 tail 的对阶、37位完整和、规格化及最终 RNE、指数检查、状态写回。当前采用三位低位工作位和独立signed2 Inf字段，保持一次接受/时钟的组合反馈实现；物理切拍与PPA仍待评估，无目标工艺/频率下不宣称单周期收敛、面积或时序结果。

### 9.10 两两预加候选（未采用）

先把两个新贡献预加、再与反馈状态相加会改变有限精度舍入顺序。该方案没有进入当前参考模型；采用前必须另行定义工作位宽、tail 和舍入位置，并验证与顺序 Q2.34 更新的差异。旧27位预加分析不能作为本方案的数值批准。

### 9.11 位宽检查与验证边界

| 节点 | 当前方向/数学位宽 |
|---|---|
| FP19 ALIGN | signed32 Q3.29；直接 RNE(P<<9,d) |
| FP19 PSUM | signed36 Q7.29 |
| FP9 PSUM→MX | signed28 Q7.21 → 补8零 signed36 Q2.34 |
| MXINT8 单项/dot16→MX | signed15/signed19 → 补17零 signed36 Q2.34 |
| 两路浮点操作数 | 各 signed36 Q2.34，保留对阶 tail |
| 浮点完整整数和 | signed37，加精确尾部信息 |
| 浮点写回 | signed36 + unsigned8 bias127；一次最终 RNE |
| INT16 中间/普通整数 | signed36 工作域；最终 INT32 饱和 |

新模型测试覆盖转换不变量、FP19 重量化、正负 tie、归约边界、37位和、tail/抵消/RNE、first、双组及 metadata。ACC 写回测试 FTZ/Inf、Inf保持及first清除；MX 输入越界测试显式阻断，不把阻断当成硬件恢复策略。当前运行证据和未实现边界记录在 verification/README.md；不引用历史测试数量作为此次验证结果。

### 9.12 当前 RTL 状态接口与旧合同隔离

原 signed28/27位有效精度 + signed12 无偏规格化指数合同已被替换。旧 `signed37 sig + signed12 lsb_exp` ACC36 候选也不是 Q2.34：它分别将操作数 RNE 到共同36位窗口，再写回37位状态，丢位位置与当前 tail 后最终 RNE 不同。

`hardware/rtl/systemverilog/mac_acc_stage.sv`、`mac_acc_float_prepare36.sv` 和 `tb_mac_acc_stage.sv` 已改为当前 Q2.34 合同。`update_float_acc36_candidate()` 继续明确拒绝旧接口；所有现行 ACC 数值测试调用 `update_float_acc36()`。旧格式的 skip/xfail 标记已移除，测试失败不再被预期差异掩盖。

ACC 仍为写状态的 sink：`o_mx_ready=i_rstn`，实际 MX 握手提交，INT16 phase0仅保存中间值，phase1才输出更新 pulse。保持原有无最终结果ready端口的接口，不宣称具备最终输出包反压。新增全链路 TB 在 MX→ACC 连接处同步门控 valid/ready，验证流水反压和复位取消；该门控只在测试环境，未改变生产 ACC 的握手。

本轮普通及 SYNTHESIS 宏编译结果仅是 Icarus 编译检查；宏不会调用综合工具。真实 DesignWare 库、综合、面积、时序以及最终 FP32 转换仍无本轮验证证据。
