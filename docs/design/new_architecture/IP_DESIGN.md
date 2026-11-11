# 新架构 IP 设计：入口转换与共享乘法前端

状态日期：2026-11-12。本文只记录已确认的入口转换、乘法前端和局部归约接口；累加器位级实现、最终输出转换、局部 MAX/对齐器细部及物理流水不在本文范围。

本文是实现说明，不是已验证结论。新架构 RTL、端到端仿真、综合、时序和 PPA 尚未开始；数值合同以[已确认约束](../../project/decisions.md)为准。旧 [IP 设计](../IP_Design.md)和 [RTL 设计](../RTL_Design.md)描述旧 SystemVerilog dot16/Q2.34 基线，不是新架构合同。

## 1. 转换器边界

独立输入转换器只实现一份模块逻辑，负责原始编码分类、特殊值映射、subnormal 精确保留、FP8/FP16 指数折叠、补码系数构造、FP4 的 `q=2x` 和 INT4 符号扩展。它不生成、不应用 MX scale。

| 端口 | 宽度 | 说明 |
|---|---:|---|
| `i_raw` | 16 | 一个 raw operand word |
| `i_fmt` | 3 | FP4/E4M3/E5M2/BF16/FP16/INT4/INT8/INT16 |
| `i_valid` | 1 | 输入有效 |
| `o_fields` | 22 | 一个前端一侧的转换后字段 bundle |
| `o_valid` | 1 | 字段有效 |

- 转换器不例化在 MAC 内，也不与 MAC 端口物理连接；MAC 入口是乘法前端。
- 转换器单独测试；MAC 测试直接使用转换后的 22-bit 字段向量，不重复分类 raw/fmt。
- 一个 raw word 的打包：FP4 为 4×4-bit，按 `g0p0,g0p1,g1p0,g1p1` 从低到高；FP8/INT8 为 2×8-bit，低字节 g0、高字节 g1；BF16/FP16/INT16 为 1×16-bit 的 g0。
- 尾段补零由外部完成，前端不设逐元素 valid；转换器与 MAC 之间没有 scale 连线，scale 随 MAC 事务 meta 传递。

## 2. 22-bit 字段 bundle

每个前端每侧一个 22-bit bundle；未用高位为 0。

| 模式 | 布局 |
|---|---|
| FP8 | `[10:0]={Q0,M0}`，`[21:11]={Q1,M1}` |
| FP4 | `[4:0]=g0p0`，`[9:5]=g0p1`，`[14:10]=g1p0`，`[19:15]=g1p1` |
| INT4 | 同 FP4，q 符号扩展到 5-bit |
| INT8 | `[7:0]=g0`，`[15:8]=g1` |
| BF16 | `[16:0]={E,m}` |
| FP16 | `[17:0]={Q,M}` |
| INT16 | `[15:0]=q` |

FP8 的 E4M3/E5M2 在转换器中统一为 `M/Q`；MAC 只使用一种 FP8 mode。E4M3×E5M2 与 E5M2×E4M3 通过交换两侧 fmt 支持，不增加 MAC 内格式标签。

## 3. 16 前端 group 与乘法块映射

每个物理前端配置 2 颗 signed9×signed9（W0/W1）和 2 颗 signed5×signed5（S0/S1）。16 前端共 32 颗宽块、32 颗小块。

| 模式 | W0 | W1 | S0 | S1 | 完整乘积/前端（对） | K/G |
|---|---|---|---|---|---|---:|
| FP4 | g0p0 | g1p0 | g0p1 | g1p1 | 4 | 32/2 |
| INT4 | g0p0 | g1p0 | g0p1 | g1p1 | 4 | 32/2 |
| FP8 | g0 | g1 | 空闲 | 空闲 | 2 | 16/2 |
| INT8 | g0 | g1 | 空闲 | 空闲 | 2 | 16/2 |
| BF16 | g0 | 空闲 | 空闲 | 空闲 | 1 | 16/1 |
| FP16 | 偶前端 p00，奇前端 p10 | 偶前端 p01，奇前端 p11 | 空闲 | 空闲 | 1/对 | 8/1 |
| INT16 | 偶前端 p00，奇前端 p10 | 偶前端 p01，奇前端 p11 | 空闲 | 空闲 | 1/对 | 8/1 |

- FP4/INT4 使用全部四颗块；W0/S0 固定 g0，W1/S1 固定 g1。
- BF16 只使用 W0，保持一组 dot16；不启用 32 乘积/拍或 G=2。
- FP16/INT16 每对前端同拍形成四部分积；W1 在该模式下不再表示 group1，S0/S1 空闲。FP16 低段 unsigned8、高段 signed7；INT16 低段 unsigned8、高段 signed8；低段零扩展，高段符号扩展。
- FP16/INT16 完整重构：`P=p00+2^8×(p01+p10)+2^16×p11`；FP16 用 signed30，INT16 用 signed32。偶前端计算 p00/p01，奇前端计算 p10/p11；移位前扩展，四部分积同拍形成，不使用两拍 phase。
- 前端输出：FP8 为 P signed16+T unsigned4；BF16 为 P' signed21+Q unsigned7；FP16 为 P signed30+T unsigned4；INT16 为 P signed32；FP4 为 P signed9；INT4 为 P signed8；INT8 为 P signed16。`T/Q=0` 不代表零乘积，零判定使用 `P!=0`。

## 4. 局部归约

### 4.1 树资源

归约使用 **2 棵 16 输入、32-bit 输入、36-bit 输出的加法树**和 **1 个 41-bit 最终加法器**。不使用 38-bit 宽树或 4 棵树的两级方案。

| 模式 | TreeA | TreeB | 最终加法 | N |
|---|---|---|---|---|
| FP4 | packed `p0/p2` | packed `p1/p3` | A+B，切分并修正 g0 | g0 14，g1 14 |
| INT4 | packed `p0/p2` | packed `p1/p3` | A+B，切分并修正 g0 | g0 12，g1 12 |
| FP8 | g0 对阶系数 | g1 对阶系数 | 无 | 36/36 |
| INT8 | g0 乘积 | g1 乘积 | 无 | 19/19 |
| BF16 | g0 对阶系数 | 空闲 | 无 | 36 |
| INT16 | g0 完整乘积 | 空闲 | 无 | 34 |
| FP16 | `CL` 低 19 位 | `CH` 高 19 位 | `low+(high<<19)` | 41 |

FP8/BF16/FP16 的逐项 MAX 和对齐/截断在树前完成；FP4/INT4/INT8/INT16 没有 MAX/对齐，直接进入树或先做 packing。树和最终加法器可以继续切拍，物理寄存器边界未冻结。

### 4.2 FP16 19/19 limb

```text
C = C30 + F8 = 38 bit signed
CL = C[18:0]   unsigned19
CH = C[37:19]  signed19
C  = CL + 2^19·CH

N = ΣCL + 2^19·ΣCH
```

TreeA/TreeB 各归约 8 个 limb；逻辑输出各 22 bit，最终加法器 41 bit。低 limb 的和可以自然向高位进位，最终加法器吸收；不是只保留 19 bit。

### 4.3 FP4/INT4 offset-binary packing

记 `p0=W0(g0)`、`p1=S0(g0)`、`p2=W1(g1)`、`p3=S1(g1)`：

```text
L=9（FP4）或8（INT4），W=L+5，β=2^(L-1)
p0o=p0^β，p1o=p1^β
A_i={p2_i,5'b0,p0o_i}
B_i={p3_i,5'b0,p1o_i}
S=ΣA_i+ΣB_i
g0=signed(S[W-1:0]^2^(W-1))
g1=signed(S[W+L+4:W])
```

低场加 β 后为非负数，32 项偏置总和为 `32β=2^(W-1)`，由最终低场 MSB 取反精确消除；中间 5 个 0 位是求和增长位。packed word 宽度为 `L+W`，按高场 `p2/p3` 的符号扩展到 32-bit 树输入；树输出 36-bit。不能直接用补码 `{p2,5'b0,p0}`，也不能只向 guard 位做符号扩展，否则负乘积的偏移/借位会污染高场。

| 模式 | L | β | W | packed word | Tree 输出 | 最终 S | g0/g1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP4 | 9 | 256 | 14 | 23 | 27 | 28 | 14/14 |
| INT4 | 8 | 128 | 13 | 21 | 25 | 26 | 13/13 |

INT4 的 g0/g1 逻辑 N 为 12 bit，13-bit 字段含一个冗余符号位。

### 4.4 BF16 postfold

BF16 在 W0 的完整 signed18 乘积后执行：

```text
T = EA + EB                 // unsigned9
r = T mod 4
Q = T >> 2                  // unsigned7
P' = signext21(P) << r
if (P == 0) begin P'=0; Q=0; end
```

postfold 在局部 MAX 前，属于乘法功能级；物理寄存器可以放在 P 后或 P' 后，当前不冻结。

## 5. 归约输出

每个 group 输出一条逻辑记录：

```text
N[40:0]        signed，局部精确和，符号扩展到 41 位
e_loc[9:0]     signed，不含 MX scale
h[8:0]         signed，h=sA+sB；非 MX 为 0
group
ctx_id[CTX_W-1:0]
first, last, valid
```

- N 的逻辑宽度：FP4 14、INT4 12、FP8 36、INT8 19、BF16 36、FP16 41、INT16 34；统一符号扩展到 41 位后送 MX/ACC。
- `e_loc` 按 [decisions.md](../../project/decisions.md) 的 P1-3 表取值，不含 h；MX 级计算 `e=e_loc+h`，只加一次。`h` 用 signed9，`e_loc/e` 用 signed10。
- 普通 INT4/INT8/INT16 绕过 MX，直接进入整数累加路径；FP4/MXFP4 使用 `e_loc=-2`，MXINT8 使用 `e_loc=-12` 与 h 进入浮点路径。
- `N=0` 仍为有效贡献，`last` 仍有效；零贡献规范为 `N=e_loc=h=0`。零乘积判定使用 `P!=0`，不能用 `T/Q==0`。
- G=2 时 FP4/INT4 的最终 packed 加法切出 g0/g1 两条记录；FP8/INT8 两棵树各一条；BF16/INT16/FP16 一条。两条记录固定送两条 ACC 通路，不跨 group 合并。
- `CTX_W`、first/last 精确语义、输出物理打包和握手仍未冻结；旧 PSUM 的 56-bit 打包不继承。

## 6. 尚未冻结

- 加法器内部电路（CSA 或平衡树）、MAX/对齐器分配、路由和时钟门控。
- 物理流水级数、寄存器插入位置和同状态调度接口。
- `CTX_W`、first/last 精确语义、输出打包、ready/反压和顶层握手。
- RTL 实现、RTL 共仿、完整合法输入域验证、16384 项数值验证、综合、时序和 PPA。

## 7. 参考与证据

- 数值合同与已确认接口：[decisions.md](../../project/decisions.md)。
- 旧基线设计：[IP_Design.md](../IP_Design.md)、[RTL_Design.md](../RTL_Design.md)。
- 验证入口：[hardware/verification/README.md](../../../hardware/verification/README.md)。
- 当前证据状态：[status.md](../../project/status.md)。

本文的接口和归约结构来自 2026-11-12 的讨论确认；其中 FP4/INT4 packing 和 FP16 19/19 limb 重构只做过 Python 精确模型随机核对与字段枚举，不是 RTL 或综合证据。
