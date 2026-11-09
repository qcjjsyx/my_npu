---
title: NVDLA FP16/INT8/INT16 数值通路与位宽调研
subtitle: 面向多精度 MAC 的源码级参考分析
date: 2026-11-08
status: 调研稿
scope: 只读源码调研；仅描述 NVDLA 事实
---

# NVDLA FP16/INT8/INT16 数值通路与位宽调研

> 本文只记录 NVDLA 本地仓库的实际实现，不包含与本项目数值合同的映射、评价或迁移建议。  
> 本文没有运行 RTL 仿真、综合或 STA；所有手算例子均为静态推导，仅用 Python 做过常数核对。

## 阅读约定

| 标签 | 含义 |
|---|---|
| `[CODE]` | 由当前工作区 RTL 代码直接支持的事实 |
| `[CONFIG]` | 受 `nv_full` 配置或启用路径限制的事实 |
| `[INFER]` | 由位宽、结构或连线推断，但不是单行代码的直接声明 |
| `[UNVERIFIED]` | 代码可见但未通过仿真/形式验证闭环确认 |

源码位置使用绝对路径和行号；行号按 2026-11-08 当前工作区 `vmod` 源码。  
NVDLA 工作区有 769 个文件显示 modified，但 `git diff -w` 只有 3 行插入、0 行删除，属于空白/行尾差异；本文结论按当前工作区源码。

---

## 摘要

1. **NVDLA 本地 `nv_full` 配置支持 FP16、INT16、INT8，不支持 FP8、FP4、BF16 或 MX。** 仓库中没有这些格式的 MAC/CACC RTL。
2. **完整阵列由 2 个 CMAC 组成**，每个 CMAC 含 8 个 `CORE_mac`，每个 `CORE_mac` 含 64 个 16×16 乘法器。FP16 每拍 1024 个乘积，INT8 每拍 2048 个乘积。
3. **FP16 在 MAC 内部不是标准 IEEE 字段直接相乘。** 输入被重编码为 `sign + 15-bit 重编码幅值`，指数只保留 `E[4:2]` 三位，低 2 位指数折进尾数左移。
4. **乘积保持 carry-save。** Booth 部分积压缩成 32-bit `res_a/res_b` 两向量；对阶时对两个向量同时右移，按 4-bit 粒度截断；没有逐项 RNE。
5. **DC 模式的 64 项局部归约在 MAC 内完成。** CSA 树之后，`sop_0` 处有一次 46-bit CPA，形成 44-bit MAC 输出：6-bit 指数 + 38-bit 补码和；CACC 取其低 36 bit，要求被丢弃高位为符号扩展。
6. **CACC 是 48-bit 状态**：40-bit 尾数 + 8-bit 指数。它只归一化新来的 SOP，旧状态保持归一化；加法保留 3 个低位工作位；状态更新和最终 FP32 都是 最近舍入（平局远离零），不是 RNE。
7. **CACC 没有“相邻配对 RNE”级。** 每个 SOP 直接更新一次状态；这与“两份贡献先配对 RNE、再状态 RNE”的两级结构不同。
8. **CACC 的 32 地址 × 4 phase 是 SRAM 的 bank 分时访问方式，不是 128 份存储或 128 个累加器。** 存储本体是 8 bank × 32 row × 16 lane；FP16 使用 4 个 48-bit bank，INT8 使用 8 个 bank 的 128 个 34-bit lane。

---

## 1. 版本、配置与核查边界

### 1.1 仓库与配置

| 项目 | 值 | 来源 |
|---|---|---|
| 分支 | `nvdlav1` | 本地仓库 |
| commit | `8e06b1b` | `git log -1` |
| VERSION | `NVDLA_OS_INITIAL` | [VERSION](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/VERSION) |
| README 版本 | v1.0，非可配置 full-precision 版本 | [README.md:15](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/README.md#L15-L18) |
| 配置 | `nv_full` | [Makefile:17](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/Makefile#L17-L17) |
| 原子阵列参数 | `MAC_ATOMIC_C_SIZE_64`、`MAC_ATOMIC_K_SIZE_32` | [nv_full.spec:16](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/spec/defs/nv_full.spec#L16-L17) |

**分析对象是 `vmod/nvdla` 下生成后的 RTL**，不是 `cmod` 的 C-model/模板，也不是只根据 spec 推断。`nv_full` 同时启用了 `WINOGRAD_ENABLE`、`WEIGHT_COMPRESSION_ENABLE` 等特性；本文涉及的 64 项 DC SOP 结论针对非 Winograd 的 DC 路径，Winograd 只作为配置限制提及。

### 1.2 实际支持格式

| 格式 | 是否支持 | 证据 |
|---|---|---|
| FP16 | 是 | `cfg_is_fp16`，[MAC_mul.v:15](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_MAC_mul.v#L15-L17) |
| INT16 | 是 | `cfg_is_int16`，[NV_NVDLA_CMAC_CORE_mac.v:15](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_mac.v#L15-L17) |
| INT8 | 是 | `cfg_is_int8` |
| FP8 E4M3/E5M2 | 否 | 本地 `vmod/nvdla/cmac`、`cacc` 无对应 RTL |
| FP4 / MXFP4 | 否 | 同上 |
| BF16 | 否 | 同上 |
| MXFP8 / MXINT8 / E8M0 scale | 否 | 同上 |

`README` 的 2048 个 8-bit MAC / 1024 个 16-bit MAC 与 RTL 计数一致：2 个 CMAC × 8 core × 64 个 16×16 乘法器 = 1024 个乘法器；INT8 每个乘法器出 2 个乘积，2048 个。[README.md:15](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/README.md#L15-L18)

---

## 2. 阵列组织与数据流总览

### 2.1 两个 CMAC、八个 core、六十四乘

- 顶层有两个 `NV_NVDLA_partition_m`，每个含一个 `NV_NVDLA_cmac`。[NV_nvdla.v:2523](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/top/NV_nvdla.v#L2523-L2523)、[:2817](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/top/NV_nvdla.v#L2817-L2817)、[NV_NVDLA_partition_m.v:635](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/top/NV_NVDLA_partition_m.v#L635-L635)
- 每个 CMAC 有 8 个 `CORE_mac`，每个 64 个乘法器，物理输出 176 bit = 4×44 bit；DC 模式只使能第一份 SOP，其余三份仅在 Winograd 模式使能。[NV_NVDLA_CMAC_core.v:2059](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cmac/NV_NVDLA_CMAC_core.v#L2059-L2304)
- 两个 CMAC 共 16 个 core；DC 模式每个 core 仅第一份 SOP 有效，共 16 个 FP16 SOP/拍。4×44-bit 是兼容 Winograd 的物理输出容器。[NV_NVDLA_CACC_calculator.v:10068](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cacc/NV_NVDLA_CACC_calculator.v#L10068-L10080)
- CACC 有 64 个 FP cell、64 个 INT16 cell、64 个 INT8 cell；这些是物理实例数量；DC 模式有效输入为 FP16/INT16 每拍 16 路、INT8 每拍 32 路，不能按全部实例数解释持续有效输出。[NV_NVDLA_CACC_calculator.v:8972](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cacc/NV_NVDLA_CACC_calculator.v#L8972-L8985)

### 2.2 数据流简图

```text
CSC: stripe 调度、CBUF 读、dat/wt/mask、stripe_st/end
   │
   ├─ CORE_active: A 复制到 8 个 CORE_mac；B 由 wt_sel 分时写入 per-core 寄存器
   ▼
[ per-core A 1024b, B 1024b ]                 ← 8 个 CORE_mac
   │
   ├─ CORE_MAC_exp: 64 个 q_sum → max tree → 64 个 shift code [reg]
   ├─ CORE_mac: 64 个 Booth 16×16 → 2 个 carry-save 向量
   │      └─ exp 右移 → CSA 树 → sop_0 46-bit CPA → DC: 1 个有效 44-bit SOP [reg]
   ▼
[ mac_out_data[175:0] = 4×44-bit；sop_exp[5:0]；pvld ]
   │
   ├─ rt_out
   ▼
CACC：assembly buffer SRAM（8 bank × 32 row × 16 lane）
   ├─ 物理 64 FP / 128 INT 路；DC 有效 16 FP / 32 INT8：44-bit SOP + 48-bit 状态
   │     → 归一化新 SOP → 对阶 → 加法 + 3 tail → 归一化 → 最近舍入（平局远离零）
   ▼
[ dlv_data 8×512b ] → cacc2sdp_valid/ready → SDP
   ↳ accu2sc_credit → CSC
```

CACC 的 assembly buffer 是 32×768/544 的 RAM bank；见 [NV_NVDLA_CACC_assembly_buffer.v:275](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cacc/NV_NVDLA_CACC_assembly_buffer.v#L275-L370)。

---

## 3. FP16 输入重编码

### 3.1 原始字段

FP16 原始 16 bit：

```text
bit15    : sign
bit14:10 : exp[4:0]
bit9:0   : frac[9:0]
```

`CORE_active` 对每个 lane 取出 sign/exp/frac，并检测 NaN：

```text
in_dat_nan = (exp == 5'h1f) && (frac != 0)
```

见 [NV_NVDLA_CMAC_CORE_active.v:28240](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_active.v#L28240-L28240)。

### 3.2 重编码为 3-bit 指数 + 15-bit 幅值

```text
q = exp[4:2]          // 3-bit 指数码
r = exp[1:0]          // 低 2 位指数
mts_ori = normal ? {2'b1, frac} : {1'b0, frac, 1'b0}
mts_sft = mts_ori << r
op_a_dat = {sign, mts_sft[14:0]}
```

- `normal` 指 `exp != 0`；hidden=1，`mts_ori = 1024 + frac`。
- `exp == 0` 时 hidden=0，使用 `{0, frac, 0}`，不执行 FTZ。
- `mts_sft` 左移 `exp[1:0]` 位，把低 2 位指数折进幅值。

代码位置：[active:28312](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_active.v#L28312-L28312)（`in_dat_exp` 取 `exp[4:2]`）、[:28383](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_active.v#L28383-L28383)（`in_dat_norm`）、[:29214](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_active.v#L29214-L29214)（`mts_ori`）、[:29230](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_active.v#L29230-L29230)（`mts_sft`）。

### 3.3 数值公式

对正常数：

```text
x = (-1)^sign × m × 2^(4q − 25)
```

其中 `m = (1024 + frac) << r`，`q = exp[4:2]`，`r = exp[1:0]`。

因为 `1024 = 2^10`，FP16 的值是 `(1 + frac/1024) × 2^(E−15)`；把 `m` 左移 `r` 后，`x = m × 2^(4q−25)`。

### 3.4 特殊值与不足

| 输入 | 实际行为 | 证据 |
|---|---|---|
| +0/−0 | `exp=0, frac=0`，幅值为 0；零乘积不参与 MAX | [active:28383](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_active.v#L28383-L28383) |
| subnormal | `exp=0, frac≠0`，hidden=0，幅值 `{0,frac,0}`；不 FTZ | [active:29214](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_active.v#L29214-L29214) |
| NaN | `exp=31, frac≠0`；64 lane 中任一 NaN，`dat_pre_nz` 全清，SOP 被 `sop_nan` 覆盖 | [active:28240](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_active.v#L28240-L28240)、[:29935](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_active.v#L29935-L29935)、[MAC_nan:1120](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_MAC_nan.v#L1120-L1123) |
| Inf | 没有独立 Inf 分支；`exp=31, frac=0` 仍按 `in_dat_norm=1` 走有限数 hidden 路径 | [active:28383](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_active.v#L28383-L28383) |

**静态推导：** subnormal 的 q=0、m=2*frac，代入同一尺度得到 `(-1)^sign*frac*2^-24`，因此入口重编码精确保值；这不表示后续对阶和累加没有精度损失。

### 3.5 为什么用 3-bit 指数

- 64 个指数和、max 树和 shift code 都从 5-bit 降到 3/4-bit，减少面积。
- 低 2 位指数折进 15-bit 幅值左移，正常数不丢信息。
- 移位粒度变成 4 bit：`q` 差 1 = 真实指数差 4 = 右移 4 bit。
- 对阶向量宽度为 32 bit，最大非清零右移量为 28 bit；差 ≥8 个 q 单位（≥32 bit）直接清零。

---

## 4. 乘法前端

### 4.1 Booth 与 carry-save

- 乘法器端口 16-bit：`op_a_dat[15:0]`、`op_b_dat[15:0]`。
- FP16 下 bit15 是符号，bits14:0 是 15-bit 重编码幅值。
- `fp16_sign = op_a[15] ^ op_b[15]`。[MAC_mul:374](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_MAC_mul.v#L374-L374)
- 16-bit 操作数用 radix-4 Booth 生成 8 个部分积；每个部分积按 2 位权重装入 17-bit `sel_data`，负项用 `sel_inv` 补 1。[MAC_mul:320](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_MAC_mul.v#L320-L340)、[:492](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_MAC_mul.v#L492-L520)
- 8 个部分积先分两颗 5:2 CSA 树压缩，再在 level1 合并成 32-bit `res_a`/`res_b` 两个向量。`res_a/res_b` 是经过模式选择、FP16 对阶及有效门控后的输出，不能无条件写成其和等于原始乘积；`dbg_booth_result = res_a + res_b` 只是调试和。[MAC_mul:540](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_MAC_mul.v#L540-L590)、[:630](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_MAC_mul.v#L630-L665)

### 4.2 对阶移位直接作用于 carry-save 向量

```text
pp_fp16_0_sft = (exp_sft[3]) ? 32'b0 : (pp_out_l1n0_0 >> {exp_sft[2:0], 2'b0})
pp_fp16_1_sft = (exp_sft[3]) ? 32'b0 : (pp_out_l1n0_1 >> {exp_sft[2:0], 2'b0})
```

见 [MAC_mul:591](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_MAC_mul.v#L591-L610)。

- 此处使用逻辑右移，低位直接丢弃，没有逐项 RNE；符号语义还依赖 pp_sign_tag、符号树和末级 patch，不能将两向量当作独立补码操作数做算术右移。分别移位的结果也不能直接等同于完整乘积先相加后截断。
- `exp_sft[3]` 时不仅两向量清零，`pp_sign_tag` 也清零，乘积不留下符号残余。[MAC_mul:607](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_MAC_mul.v#L607-L610)

### 4.3 部分积重构与元素归约是两件事

- Booth 部分积 → `res_a/res_b` 是**一个元素乘积**的 carry-save 重构。
- DC 模式的 64 个乘法器 → 1 个有效 SOP 是**不同元素乘积**的归约；其余 SOP 输出受 Winograd 使能控制。
- NVDLA 不把每个元素乘积先重构成一个单数再归约；它把 carry-save 向量保留到归约树，最后 `sop_0` 再一次 CPA。

---

## 5. 指数选择与对阶

### 5.1 q_sum 与 exp_max

```text
q_sum = qA + qB           // 4 bit，0..14
exp_max = max(q_sum[0..63])  // 一个 CORE_mac 一个 4-bit max
```

- `exp_sum_00[3:0] = dat_cur_exp[2:0] + wt_cur_exp[2:0]`。[MAC_exp:807](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_MAC_exp.v#L807-L807)
- 一级 16 个 4-input max，二级 1 个 16-input max。[MAC_exp:968](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_MAC_exp.v#L968-L1020)、[:1304](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_MAC_exp.v#L1304-L1316)
- 4 个 SOP 共用一个 `exp_max`：`sop_exp = {pp_exp_d3, 2'b0}`。[mac:10376](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_mac.v#L10376-L10376)、[:10386](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_mac.v#L10386-L10435)

### 5.2 exp_sft 的单位与移位

```text
exp_sft = exp_max − q_sum   // 4 bit，q 单位差
shift_bits = {exp_sft[2:0], 2'b0}  // 0,4,8,...,28
```

| exp_sft（q 单位） | exp_sft[3] | 低 3 位 | 实际 bit 移位 | 结果 |
|---:|---:|---:|---:|---|
| 0 | 0 | 0 | 0 | 不移位 |
| 1 | 0 | 1 | 4 | 右移 4 |
| 2 | 0 | 2 | 8 | 右移 8 |
| 7 | 0 | 7 | 28 | 右移 28 |
| 8 | 1 | 0 | 32 | 清 0 |
| 14 | 1 | 6 | 56 | 清 0 |

`exp_sft[3]` 不是“28 bit 之外的额外窗口”，而是“移位量 ≥32 bit，32-bit 乘积被完全移出，直接置零”。低 3 位负责 0/4/…/28，bit3 负责 32/…/60，清零是统一的截断策略。

### 5.3 示例：1.0×1.0 与 2.0×1.0

**静态推导，未仿真。**

- 1.0：`E=15, q=3, r=3, m=8192`。
- 2.0：`E=16, q=4, r=0, m=1024`。

```text
1.0×1.0: m=8192×8192=2^26, q_sum=6
2.0×1.0: m=1024×8192=2^23, q_sum=7
exp_max=7
1.0×1.0: exp_sft=1 → 右移 4 → 2^22
2.0×1.0: exp_sft=0 → 保持 2^23
和 = 2^23 + 2^22 = 3×2^22
qmax=7 的尺度 = 2^(4×7−50)=2^−22
结果 = 3×2^22 × 2^−22 = 3.0
```

---

## 6. 局部 SOP 归约

### 6.1 结构

- 每个 `CORE_mac` 有 64 个乘法器；DC 路径继续经过 level1 至 level4 合并为一个 SOP，后三份输出在非 Winograd 模式强制为零。不能把中间四组结果当作四份独立 dot16。
- 证据：`NV_NVDLA_CMAC_CORE_mac.v` 的 DC 树级说明为 16→8→4→2→1（7680、8119、9385、9988 行），末级选择见 10009 行起，输出使能见 10400–10435 行。Winograd 分组另论。
- 每个乘积的 carry-save 向量先符号扩展到 36-bit，进入 level0 的 8-input/36-bit CSA 树（每棵树 4 个乘积），再逐级合并。[mac:4935](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_mac.v#L4935-L4945)、[:9987](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_mac.v#L9987-L10080)

### 6.2 46-bit CPA 与 44-bit 输出

```text
sop_0 = pp_out_l4n0_0_d3 + pp_out_l4n0_1_d3 + pp_sign_patch_0
```

这是一次 46-bit 的 3 输入 CPA（两个 carry-save 向量 + 符号修补常数）。因此 MAC 输出的 SOP 是单一有符号数，不是把 carry-save 直接交给 CACC。[mac:10360](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_mac.v#L10360-L10365)

```text
mac_out_data[43:0] = {sop_exp[5:0], sop_0[37:0]}
```

46-bit `sop_0` 只取 `[37:0]` 装入 44-bit `mac_out_data`。高 8 位是设计上可丢弃的符号/修补位；这不是低位舍入，而是容器缩窄。[mac:10386](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_mac.v#L10386-L10435)

44-bit 字段：

```text
[43:38] = 6-bit 指数
[37]    = 符号
[36:0]  = 补码和的其余 37 位（与 bit37 合为 signed38）
```

CACC 断言 `di_mant_pre[37:35]` 必须是全 0 或全 1；真正有效的是 `di_mant[35:0]` 36-bit 尾数。[CALC_fp:219](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cacc/NV_NVDLA_CACC_CALC_fp_48b.v#L219-L231)、[:759](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cacc/NV_NVDLA_CACC_CALC_fp_48b.v#L759-L759)

### 6.3 位宽增长

| 节点 | 宽度 | 说明 |
|---|---:|---|
| 一个 16×16 乘积 | 32 | carry-save，实际有效约 28 bit |
| level0 树输入 | 36 | 32-bit 乘积符号扩展到 36 |
| level0 树输出 | 36 | 4 个乘积一组 |
| level1/2 树 | 42/46 | 分组/符号增长 |
| `sop_0` | 46 | 两向量 + 符号修补，46-bit CPA |
| MAC 输出 | 44 | 6-bit exp + 38-bit 符号扩展和 |
| 有效尾数 | 36 | CACC 的 `di_mant[35:0]` |

> 不能用物理容器宽度直接推断有效精度或余量。DC 是 64 项归约；CSA 的补偿表示也需要单独解释，撤回此前基于 dot16 的“留 3 bit 余量”结论。

---

## 7. CACC 累加

### 7.1 状态格式

CACC 的 FP 状态是 48 bit：

```text
oi_expn = oi_pd[47:40]   // 8-bit 指数
oi_mant = oi_pd[39:0]    // 40-bit 有符号尾数
```

见 [CALC_fp:219](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cacc/NV_NVDLA_CACC_CALC_fp_48b.v#L219-L231)。

- 尾数是 40-bit 有符号容器；写回 `partial_mant_nmlz[39:0]`，指数 `partial_result_expo[7:0]`。
- 最终 FP32 转换取 `partial_mant_abs[38:15]` 作为 24-bit 有效字段；绝对 bit 位置依赖代码的归一化和固定 bias，本文不把它解释为标准 FP32 的隐藏位/指数格式。
- 因此“40-bit 状态”是容器宽度，不是 40-bit 有效精度；最终只保留 24-bit 有效字段。

对有限状态可推得解码公式 `[INFER]`：`value = signed(mant40) × 2^(exp8−101)`。CMAC 的补偿后 SOP 系数 D 对应 `D×2^(di_expn−50)`；新输入转为 `M=D×2^(lead+4)`、`E=di_expn+47−lead`，因此 `M×2^(E−101)=D×2^(di_expn−50)`。零和 NaN 编码另行处理。这解释固定偏置 47，并非仅指数差有意义；此推导未替代对 CMAC 符号补偿和整条路径的仿真。

### 7.2 新 SOP 归一化

```text
di_lead_num = 冗余符号位数（正数为前导 0，负数为前导 1）
di_manm    = di_mant << di_lead_num
di_mans    = di_manm
di_expm    = di_expn + 47 − di_lead_num
```

见 [CALC_fp:327](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cacc/NV_NVDLA_CACC_CALC_fp_48b.v#L327-L335)。

- `+47` 是固定 bias，把 36-bit 尾数放到 40-bit 状态的指数标度；真正有意义的是 `di_expm` 之间的差。
- 只归一化新来的 SOP；旧 `oi_mant` 保持归一化，不再扫描。

**1.0×1.0 的 SOP：**

```text
di_expn = 24        // q_sum=6, sop_exp=4×6
di_mant = 2^26
di_lead_num = 8
di_expm = 24 + 47 − 8 = 63
di_mant_ext = 2^26 << 8 << 4 = 2^38
```

**0.5×1.0 的 SOP：**

```text
di_expn = 24
di_mant = 2^25
di_lead_num = 9
di_expm = 24 + 47 − 9 = 62
di_mant_ext = 2^25 << 9 << 4 = 2^38
```

两者 `di_expn` 都是 24，但 CACC 通过 `di_lead_num` 把 0.5 的 `di_expm` 变成 62，比 1.0 小 1。新 SOP 归一化后的 40-bit 工作尾数 `di_mant_ext` 都是 `2^38`；这不等于旧状态的绝对尾数位值。

### 7.3 max、对阶、加法、3 个 tail 位

```text
max_expo = max(di_expo, oi_expo)
di_expo_shift = max_expo − di_expo
oi_expo_shift = max_expo − oi_expo

di_mant_ext = {di_mans, 4'b0}   // 36-bit 扩到 40-bit
oi_mant_ext = oi_mans           // 旧状态本来就是 40-bit

di_mant_shift = signed(di_mant_ext) >>> di_expo_shift
oi_mant_shift = signed(oi_mant_ext) >>> oi_expo_shift
```

见 [CALC_fp:338](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cacc/NV_NVDLA_CACC_CALC_fp_48b.v#L338-L359)。

加法：

```text
mant_sum[40:0] = signed(di_aligned_mant) + signed(oi_aligned_mant)
mant_sum_total[43:0] = {mant_sum[40:0], in_mant_cut[2:0]}
```

`in_mant_cut` 来自被右移的较小操作数：右移丢掉的低位压缩成 3 bit，最高位作 guard，其余位提供 round/sticky 信息。[CALC_fp:348](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cacc/NV_NVDLA_CACC_CALC_fp_48b.v#L348-L359)

### 7.4 归一化与 最近舍入（平局远离零）

```text
mant_sum_ncmp[41:0] = {mant_sum_scmp[43:3], |mant_sum_scmp[2:0]}
mant_sum_effect = mant_sum_trans[41:2]   // 40-bit
guard  = mant_sum_trans[1]
sticky = mant_sum_trans[0]
mant_sum_point5 = guard & (~sign | sticky)
mant_sum_round  = mant_sum_effect + point5
```

见 [CALC_fp:405](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cacc/NV_NVDLA_CACC_CALC_fp_48b.v#L405-L418)、[:497](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cacc/NV_NVDLA_CACC_CALC_fp_48b.v#L497-L520)。

- 正数：`~sign=1`，`guard=1` 就加 1；恰好一半也进位。
- 负数：`~sign=0`，只有 `guard & sticky` 才加 1；恰好一半不进位。
- 这不是 ties-to-even。

最终 FP32 又是另一处 最近舍入（平局远离零）：

```text
fp32_mant_point5 = partial_mant_abs[14]
fp32_mant_round  = fp32_mant_effect + fp32_mant_point5
fp32_expo_round  = partial_expo + 64 + carry/neg2
```

见 [CALC_fp:640](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cacc/NV_NVDLA_CACC_CALC_fp_48b.v#L640-L662)。

### 7.5 写回与最终输出

状态写回：

```text
partial_result_out = {partial_result_expo[7:0], partial_result_mant[39:0]}
abuf_wr_data       = 16 × 48-bit
abuf_wr_addr       = calc_addr_out
abuf_wr_en         = calc_wr_en_out
```

FP 路径在 `calc_valid_d4` 写回；`calc_wr_en_out` 由 d4 选择。[NV_NVDLA_CACC_calculator.v:11435](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cacc/NV_NVDLA_CACC_calculator.v#L11435-L11438)、[:13179](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cacc/NV_NVDLA_CACC_calculator.v#L13179-L13187)

第一次贡献：

```text
in_op_valid = 0 时，in_mask_op = 0
oi_mant = 0, oi_expo = 0
```

第一次更新就是 `0 + 新 SOP`。[CALC_fp:169](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cacc/NV_NVDLA_CACC_CALC_fp_48b.v#L169-L171)

最后一次贡献：

```text
calc_channel_end 时：
  calc_wr_en = 0       // 不写回部分和
  calc_dlv_valid = 1   // 产生最终输出
  calc_dlv_en 选择最终结果
```

见 [calculator:2808](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cacc/NV_NVDLA_CACC_calculator.v#L2808-L2810)、[:3298](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cacc/NV_NVDLA_CACC_calculator.v#L3298-L3307)。

### 7.6 累加例子：1.0 + 0.5，1.0 + 1.0

**静态推导，未仿真。以下用归一化浮点模型说明相对行为；固定 +47/+64 bias 和绝对尾数 bit 位置不展开。**

记 1.0 的归一化状态为 `(E0, M0)`，`M0` 是归一化尾数。0.5 的 SOP 归一化后写成 `(E0−1, M0)`：同样的归一化尾数，指数低 1。

代码里可以核对的相对量是：

```text
1.0×1.0: di_expn=24, di_mant=2^26, di_lead_num=8, di_expm=63
0.5×1.0: di_expn=24, di_mant=2^25, di_lead_num=9, di_expm=62
```

两者 `di_expm` 差 1；绝对 63/62 不是 IEEE 指数，而是 CACC 固定 bias 后的内部值。

**加 0.5：**

```text
旧状态：expo=E0,   mant=M0
新 SOP：expo=E0−1, mant=M0
max_expo = E0
0.5 右移 1 位：M0 >> 1 = 0.5M0
旧 1.0 不移位：M0
和 = M0 + 0.5M0 = 1.5M0
```

结果在归一化区间内，状态：

```text
expo = E0
mant = 1.5M0
```

数值 = **1.5**。

**加 1.0：**

```text
旧状态：expo=E0, mant=M0
新 SOP：expo=E0, mant=M0
和 = 2M0
```

`2M0` 超出归一化尾数范围时，归一化右移 1 位，指数加 1：

```text
expo = E0+1
mant = M0
```

数值 = **2.0**。

> 这是相对指数/尾数示例，不用于核对 CACC 状态的绝对 bit 值。绝对位值需要把固定 bias 和状态归一化位置一起逐位展开，本文不声称做过这类逐位核对。

### 7.7 32 地址 × 4 phase 的详细说明

这一节解释 CACC partial-sum SRAM 的组织。单 bank 有 32 行；bank 与 row 共同选择不同存储位置，不能把四 phase 理解为不对应独立存储的纯延迟。

#### 7.7.1 四个词

| 词 | 含义 |
|---|---|
| **lane** | 一个输出通道的 partial sum；FP16/INT16 是 48 bit，INT8 有效是 34 bit |
| **bank** | 16 个 lane 拼成一个宽 SRAM 字；4 个 768-bit bank + 4 个 544-bit bank |
| **row** | 每个 bank 的 32 个地址（32 个字） |
| **phase** | 同一 row 上分 4 拍，轮流选不同 bank 的时隙 |

物理存储：

```text
8 bank × 32 row × 16 lane = 4096 个 lane 槽位
```

但 FP16 只用 4 个 bank：

```text
4 bank × 32 row × 16 lane = 2048 个 48-bit partial sum
```

INT8 用 8 个 bank：

```text
8 bank × 32 row × 16 lane = 4096 个有效 34-bit partial sum
```

#### 7.7.2 一个 bank 的内部

宽 bank：

```text
bank b:
  row 0   : [lane0 48b][lane1 48b]...[lane15 48b]  = 768 bit
  row 1   : ...
  ...
  row 31  : ...
```

窄 bank：

```text
bank b:
  row 0   : [lane0 34b][lane1 34b]...[lane15 34b]  = 544 bit
  ...
  row 31  : ...
```

代码中的 RAM：

```text
nv_ram_rws_32x768  u_accu_abuf_0 ... u_accu_abuf_3
nv_ram_rws_32x544  u_accu_abuf_4 ... u_accu_abuf_7
```

见 [NV_NVDLA_CACC_assembly_buffer.v:275](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cacc/NV_NVDLA_CACC_assembly_buffer.v#L275-L370)。

- 4 个 768-bit bank：每个字 16×48，供 FP16/INT16 的 64 lane。
- 4 个 544-bit bank：每个字 16×34，供 INT8 的另外 64 lane。
- INT8 共 128 lane：宽 bank 中使用 34-bit 有效值，窄 bank 也是 34-bit 字。

#### 7.7.3 7-bit 计数器如何切

```text
accu_cnt[6:2] = 5-bit row 地址，32 行
accu_cnt[1:0] = 4 个 phase，0/1/2/3
```

代码：

```text
accu_addr = (cfg_winograd) ? accu_cnt[4:0] : accu_cnt[6:2];
accu_rd_sel_w = 4'h1 << accu_cnt_w[1:0];
```

见 [NV_NVDLA_CACC_assembly_ctrl.v:723](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cacc/NV_NVDLA_CACC_assembly_ctrl.v#L723-L730)。

非 Winograd 模式下，同一 row 保持 4 拍：

```text
cycle:         t0 t1 t2 t3 | t4 t5 t6 t7
accu_cnt[1:0]: 0  1  2  3 | 0  1  2  3
accu_addr:     A  A  A  A | B  B  B  B
```

`accu_cnt` 仅在 `layer_st | accu_valid[8]` 时更新；更新时遇 layer_st/stripe_end 清零，否则递增。上图仅适用于连续有效且没有重置边界的情况；128 是计数位置数，不保证恰好 128 个物理时钟。使能见 assembly_ctrl.v:804–813。

#### 7.7.4 phase 选哪些 bank

`is_x1/x2/x4/x8` 的定义：

```text
x1 = ~int8 & ~winograd   // 普通 FP16/INT16
x2 = int8 & ~winograd    // 普通 INT8
x4 = ~int8 & winograd
x8 = int8 & winograd
```

见 [assembly_ctrl:423](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cacc/NV_NVDLA_CACC_assembly_ctrl.v#L423-L444)。

bank mask：

```text
x8: 8'hff          // 8 个 bank 全选
x4: 8'h0f          // bank 0..3
x2: {2{sel}}       // 低 4 位一份，高 4 位一份
x1: {4'b0, sel}    // 只选 bank 0..3 中的一个
```

见 [assembly_ctrl:739](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cacc/NV_NVDLA_CACC_assembly_ctrl.v#L739-L742)。

`sel` 是 one-hot：

```text
phase0: 0001
phase1: 0010
phase2: 0100
phase3: 1000
```

**x1（FP16/INT16）每个 phase 选 1 个 bank：**

```text
phase0 -> bank0（16 lane）
phase1 -> bank1（16 lane）
phase2 -> bank2（16 lane）
phase3 -> bank3（16 lane）
```

四次连续有效访问覆盖的存储 lane 槽位（不是每拍有效输出数）：

```text
4 bank × 16 lane = 64 个 FP16/INT16 存储 lane 槽位
```

**x2（INT8）每个 phase 选 2 个 bank：**

```text
phase0 -> bank0 + bank4
phase1 -> bank1 + bank5
phase2 -> bank2 + bank6
phase3 -> bank3 + bank7
```

四次连续有效访问覆盖的存储 lane 槽位（不是每拍有效输出数）：

```text
8 bank × 16 lane = 128 个 INT8 存储 lane 槽位
```

`accu_ram_sel` 对 x1/x2 进一步门控，`accu_rd_en` 再把 valid/channel_st 合进去。[assembly_ctrl:758](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cacc/NV_NVDLA_CACC_assembly_ctrl.v#L758-L768)、[:774](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cacc/NV_NVDLA_CACC_assembly_ctrl.v#L774-L776)

#### 7.7.5 phase 与计算流水：已知连接及待核查关系

CACC 的流水：

```text
读旧状态
  → 归一化新 SOP
  → 对阶/加法 + 3 tail
  → 归一化/最近舍入（平局远离零）
  → 写回
```

代码里对应 `calc_valid_d0..d4`；FP 在 d4 写回，写地址是 `calc_addr_out`，写使能是 `calc_wr_en_out`。[calculator:3309](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cacc/NV_NVDLA_CACC_calculator.v#L3309-L3309)、[:11435](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cacc/NV_NVDLA_CACC_calculator.v#L11435-L11438)、[:13187](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cacc/NV_NVDLA_CACC_calculator.v#L13187-L13187)

在连续有效情况下，row 地址每四次计数更新变化一次，各 phase 选择不同 bank。但仅凭这一关系，不能证明“四 phase 恰好匹配四级流水”或“保证同 bank 不会同拍读写”。还需逐拍列出读请求、RAM 返回、计算 valid、写回 bank/row 和下一次同状态读取。本文保留 bank 选择事实，撤回上述因果推断；精确反馈间隔尚待核查。

#### 7.7.6 容易误解的点

1. **32×4 不是单个 RAM 的 128 行。** 存储是 32 row × 8 bank × 16 lane；4 phase 是访问时序。
2. **不是 4 个累加器。** 硬件算术实例数、逻辑输出状态数和 bank 数不同；4 phase 轮流选择不同 bank，各 bank/row/lane 对应独立存储槽。
3. **32 不是 K=32。** 32 来自 5-bit SRAM 地址；逻辑上对应什么由 CSC/CACC 的 stripe/通道控制决定，不能单从 SRAM 断言。
4. **Winograd 模式不同。** `accu_addr=accu_cnt[4:0]`，低 2 位变成地址，不是 phase；bank mask 固定为 `0x0f`/`0xff`。
5. **不是每拍更新同一状态。** 同一 row 保持 4 拍，同一地址不是每拍读改写；CACC 是多 bank/多 phase 的流水累加。

---

## 8. 整数路径

### 8.1 INT8

- 每个 16-bit 操作数打包两个 8-bit lane；一个 16×16 乘法器在 INT8 模式下出 2 个乘积，`res_a` 和 `res_b` 的低/高 16 位分别保存两个 lane 的 carry-save 分量，不能把 res_a 单独当作乘积。[MAC_mul:365](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_MAC_mul.v#L365-L366)、[:630](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_MAC_mul.v#L630-L647)
- 每个 INT8 SOP 是 22-bit 有符号二补码；44-bit FP16 容器拆成两个 22-bit 字段：`{sop_0[45:24], sop_0[21:0]}`，中间 bits23:22 是间隔/填充。[mac:10386](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_mac.v#L10386-L10400)、[calculator:2898](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cacc/NV_NVDLA_CACC_calculator.v#L2898-L2899)
- CACC INT8 cell：`in_data[21:0]`，`in_op[33:0]`，和 `i_sum_pd[34:0]`，输出 partial 34-bit；每次更新若 35-bit 和溢出 34-bit，会饱和到 34-bit 端点。[CALC_int8:96](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cacc/NV_NVDLA_CACC_CALC_int8.v#L96-L98)、[:172](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cacc/NV_NVDLA_CACC_CALC_int8.v#L172-L172)、[:193](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cacc/NV_NVDLA_CACC_CALC_int8.v#L193-L204)
- 最终 32-bit：`i_sft = (partial << 16) >>> cfg_truncate`；`i_point5 = guide & (~sign | sticky)` 做 最近舍入（平局远离零）；再判饱和。[CALC_int8:209](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cacc/NV_NVDLA_CACC_CALC_int8.v#L209-L221)
- `cfg_truncate` 来自 `reg2dp_clip_truncate`，是 5-bit 可编程右移；默认 0，但不是“精确累加”合同的一部分。[assembly_ctrl:1670](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cacc/NV_NVDLA_CACC_assembly_ctrl.v#L1670-L1673)

### 8.2 INT16

- CACC INT16 cell：`in_data[37:0]`，`in_op[47:0]`，partial 48-bit；它把 48-bit 拆成低 32-bit + 高 6-bit + 符号位，低 33-bit 求和、高 17-bit 求和，再重组。[CALC_int16:116](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cacc/NV_NVDLA_CACC_CALC_int16.v#L116-L121)、[:380](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cacc/NV_NVDLA_CACC_CALC_int16.v#L380-L411)
- 最终 32-bit 同样用 `cfg_truncate` 右移、最近舍入（平局远离零）、饱和。[CALC_int16:449](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cacc/NV_NVDLA_CACC_CALC_int16.v#L449-L461)
- **INT16 最高位保留。** `op_a_cur_dat = {(~cfg_is_fp16_d1[0] & op_a_dat[15]), op_a_dat[14:0]}`，B 同理（MAC_mul.v:273–274）。INT16 时 FP16 标志为 0，bit15 不被清零；FP16 才清除幅值路径中的符号位。此前“INT16 bit15 被清零”的判断错误。完整有符号乘法解释还须连同 Booth 编码和末级 INT16 patch 展开，不能仅由 fp16_sign=0 推断无符号运算。

---

## 9. 逐级数值表

### 9.1 FP16（DC、非 Winograd）

| 阶段/信号 | 物理位宽 | 符号及字段 | 数值公式/权重 | 合法范围 | 扩位、移位或舍入 | 源码 |
|---|---:|---|---|---|---|---|
| 原始 FP16 lane | 16 | sign1 / exp5 / frac10 | IEEE half 原始码 | E=0..31 | 无 | [active:28312](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_active.v#L28312-L28312) |
| `in_dat_exp` | 3 | exp[4:2] | q = floor(E/4) | 0..7 | 低 2 位折进尾数 | [active:28312](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_active.v#L28312-L28312) |
| `mts_sft` / `op_a_dat` | 15 / 16 | 15-bit 幅值 + sign | normal `(1024+frac)<<(E&3)`；subnormal `(2*frac)<<0` | 0..16376 | 左移 0..3，精确 | [active:29214](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_active.v#L29214-L29214)、[:29230](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_active.v#L29230-L29230) |
| Booth 部分积 | 8×17 | radix-4，±0/1/2×B | 权重 2^(2i)，负项加 1 | — | 符号反相 | [MAC_mul:320](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_MAC_mul.v#L320-L340)、[:492](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_MAC_mul.v#L492-L520) |
| `res_a/res_b` | 2×32 | carry-save | 含模式选择、对阶及门控；须连同符号补偿解释 | 实际 28-bit | 5:2 CSA，32-bit 容器 | [MAC_mul:540](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_MAC_mul.v#L540-L590) |
| 对阶后乘积 | 32 | carry-save | `>> 4*(qmax−q)` | 0..28 | `delta≥8` 清零；截断 | [MAC_mul:591](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_MAC_mul.v#L591-L610) |
| level0/1/2 CSA | 36/42/46 | 有符号 sum/carry | 64 项 DC 分组归约 | 36-bit 有效和 | 符号扩展；树内不舍入 | [mac:4935](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_mac.v#L4935-L4945)、[:9987](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_mac.v#L9987-L10080) |
| `sop_0` | 46 | 有符号和 | 两向量 + 符号修补常数 | 范围待结合补偿表示推导 | 46-bit CPA | [mac:10360](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_mac.v#L10360-L10365) |
| `mac_out_data` | 44 | exp6 + sum38 | `{sop_exp, sop_0[37:0]}` | 有效 36-bit 尾数 | 46→38 缩窄；不是低位舍入 | [mac:10386](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_mac.v#L10386-L10435) |
| CACC 输入 | 44 | exp6 / sign1 / mant36 | `di_expn/di_mant` | — | 解包，bits37:35 为符号副本 | [CALC_fp:219](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cacc/NV_NVDLA_CACC_CALC_fp_48b.v#L219-L231) |
| 旧状态 | 48 | exp8 + mant40 | 归一化部分和 | 0..255 | 已归一化 | [CALC_fp:226](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cacc/NV_NVDLA_CACC_CALC_fp_48b.v#L226-L229) |
| 加法 | 40+3 | 40-bit 尾数 + GRS | `mant_sum_total[43:0]` | — | 对阶到 max_expo | [CALC_fp:332](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cacc/NV_NVDLA_CACC_CALC_fp_48b.v#L332-L359) |
| 状态写回 | 48 | exp8 + mant40 | normalize + 最近舍入（平局远离零） | — | 40-bit 归一化 | [CALC_fp:497](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cacc/NV_NVDLA_CACC_CALC_fp_48b.v#L497-L520) |
| FP32 输出 | 32 | sign1/exp8/frac23 | FP32 | — | 40→24 最近舍入（平局远离零） | [CALC_fp:640](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cacc/NV_NVDLA_CACC_CALC_fp_48b.v#L640-L662) |

### 9.2 INT8 / INT16

| 阶段 | INT8 | INT16 | 源码 |
|---|---:|---:|---|
| 原始输入 | 8-bit 二补码 | 16-bit | [active:27876](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_active.v#L27876-L27876)、[:28145](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_active.v#L28145-L28145) |
| 局部 SOP | 22-bit 有符号 | 38-bit 有符号 | [mac:10386](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_mac.v#L10386-L10400) |
| 部分和状态 | 34-bit 有符号 | 48-bit 有符号 | [CALC_int8:96](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cacc/NV_NVDLA_CACC_CALC_int8.v#L96-L98)、[CALC_int16:116](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cacc/NV_NVDLA_CACC_CALC_int16.v#L116-L121) |
| 最终输出 | 32-bit，`cfg_truncate` 右移 + 最近舍入（平局远离零） + 饱和 | 同左 | [CALC_int8:209](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cacc/NV_NVDLA_CACC_CALC_int8.v#L209-L221)、[CALC_int16:449](/Users/huangyuan/qcjjsyx/NPU/references/projects/NVDLA-hw/vmod/nvdla/cacc/NV_NVDLA_CACC_CALC_int16.v#L449-L461) |
| 16 项完整补码输入域所需 | signed20 | signed36 | 最小负数平方分别为 2^14、2^30；16 项上界为 2^18、2^34 |
| 16384 项完整补码输入域所需 | signed30 | signed46 | 正上界分别为 2^28、2^44；不含额外初值 |
| DC 局部归约（完整补码输入域） | 64 项，上界 2^20，需 signed22 | 64 项，上界 2^36，需 signed38 | 与 22/38-bit SOP 容器一致 |

> 上界按完整二补码输入域计算，包括最小负数；不能把本项目排除最小负数的对称输入域套给 NVDLA。整数容器宽度直接决定可精确表示的整数范围，不应因输入位宽较小就否认状态的有效存储位数。

---

## 10. 手算例子

### 10.1 FP16 对阶：1.0×1.0 与 2.0×1.0

已写在 5.3 节，结论：

```text
1.0×1.0 → 2^26, qsum=6, exp_sft=1 → 2^22
2.0×1.0 → 2^23, qsum=7, exp_sft=0 → 2^23
和 = 3×2^22，qmax=7 的尺度 2^-22 → 3.0
```

### 10.2 INT8：16 项 −127×127 的算术示例（非 DC SOP）

**静态推导，未仿真。**

```text
单项：-127 × 127 = -16129
16 项：-16129 × 16 = -258064
22-bit 二补码：0x3C0FF0（bit21=1）
CACC 符号扩展至 34-bit：0x3FFFC0FF0
```

若在数学上累计 1024 组这样的 16 项和（K=16384；不是 DC 的实际分组）：

```text
-258064 × 1024 = -264257536
34-bit 二补码：0x3F03FC000
取低 32-bit：0xF03FC000
```

这说明：

- 本例的 16 项结果（−258064）可放入 22-bit 容器，但不是 NVDLA DC SOP 的范围上界；
- 本例数值可放入 signed29；若覆盖包含 −128 的完整 INT8 域，16384 项需 signed30，相对 34-bit 状态只有 4 位范围余量；
- 34→32 的截断只是去掉高 2 位符号扩展，但 NVDLA 还会根据 `cfg_truncate` 做右移和 最近舍入（平局远离零），不能把它等同于“精确 INT32”。

### 10.3 CACC：1.0 + 0.5，1.0 + 1.0

已写在 7.6 节。要点（相对指数模型，不展开固定 bias/绝对 bit 值）：

```text
1.0 SOP: di_expn=24, di_mant=2^26, di_lead_num=8, di_expm=63
0.5 SOP: di_expn=24, di_mant=2^25, di_lead_num=9, di_expm=62
两者 di_expm 差 1；归一化尾数都在同一 40-bit 标度

1.0 + 0.5:
  0.5 右移 1 → 0.5M0；和 1.5M0；状态 (E0, 1.5M0)

1.0 + 1.0:
  和 2M0；若超出归一化范围，右移 1 → M0；状态 (E0+1, M0)
```

---

## 11. 证据矩阵与未决问题

### 11.1 主要结论与证据等级

| 编号 | 结论 | 证据 | 等级 |
|---|---|---|---|
| C1 | `nv_full` 只支持 FP16/INT16/INT8 | `cfg_is_fp16/int16/int8`；无 FP8/FP4/BF16/MX RTL | `[CODE][CONFIG]` |
| C2 | 2 CMAC ×8 core ×64 个 16×16 乘法器 | 顶层、CMAC core 实例计数 | `[CODE]` |
| C3 | FP16 重编码为 3-bit q + 15-bit 重编码幅值 | `in_dat_exp`、`mts_sft` | `[CODE]` |
| C4 | 乘积 32-bit carry-save，无逐项 RNE | `res_a/res_b`、移位 | `[CODE]` |
| C5 | 一个 core 一个 `exp_max`，4 个 SOP 共享 | `exp_max_l1_0`、`sop_exp` | `[CODE]` |
| C6 | `exp_sft` 是 q 单位差，4-bit 粒度，≥32 bit 清零 | `exp_sft`、`{exp_sft[2:0],2'b0}` | `[CODE]` |
| C7 | 64 项 DC SOP；`sop_0` 46-bit CPA；MAC 输出 44-bit | `sop_0`、`mac_out_data` | `[CODE]`；DC 路径及输出门控已核查，完整补偿等式待补 `[INFER]` |
| C8 | CACC 状态 48-bit = 40-bit 尾数 + 8-bit 指数 | `oi_mant/oi_expn` | `[CODE]` |
| C9 | 状态更新和最终 FP32 是 最近舍入（平局远离零），不是 RNE | `mant_sum_point5`、`fp32_mant_point5` | `[CODE]` |
| C10 | CACC 无 pair 级，每个 SOP 更新一次 | 输入只有 `in_data`，无 pair 寄存器 | `[CODE]` |
| C11 | 32×4 phase 是 bank 分时访问，不是 128 份存储 | `accu_cnt/accu_addr/accu_rd_mask` | `[CODE]` 仅限物理选择；逻辑映射与反馈周期待核查 |
| C12 | INT8 22/34/32；INT16 38/48/32 | `CALC_int8/int16` | `[CODE]` |
| C13 | INT16 保留输入 bit15；FP16 清除幅值输入的 bit15 | MAC_mul.v:273–274 | `[CODE]`；完整 Booth/patch 推导待补 |
| C14 | subnormal 入口重编码精确保值，完整路径未仿真 | `in_dat_norm`、`mts_ori` | `[INFER]` |
| C15 | NaN 是块级传播；Inf 无独立路径 | `in_dat_nan`、`sop_nan` | `[CODE]` |

### 11.2 未决或受限

- **subnormal 的精确缩放**：RTL 保留 hidden=0 的幅值，不 FTZ；其重编码尺度可由 `(2*frac)×2^-25=frac×2^-24` 解释，但完整路径尚未仿真。
- **INT16 前端展开**：已确认 bit15 保留，仍需将 Booth 部分积及末级符号补偿写成完整等式。
- **CACC 32 地址的逻辑映射**：物理上是 32 个 row；逻辑上对应 K-stripe 还是输出组，由 CSC/CACC 控制决定，本文没有完整追踪。
- **精确读-写延迟**：`calc_valid_d0..d4` 和 RAM retiming 存在，但未仿真确认从读请求到写回的具体周期数。
- **FP8/FP4/BF16/MX**：本地 NVDLA 没有这些路径；任何比较只能是结构/位宽层面，不能给出 NVDLA 的 FP8/FP4 数值合同。
- **Winograd**：`nv_full` 启用了 Winograd，树的 mux 和 `ps_out_l2` 会改变分组；本文的 64 项 DC SOP 只用于 DC 路径。

### 11.3 本文没有做的事

- 没有运行 RTL 仿真、综合或 STA。
- 没有用 NVDLA C-model 交叉验证 RTL。
- 没有把 NVDLA 的结构映射为本项目的实现方案。
- 没有根据模块数量或位宽乘积给出面积/频率结论。

---

## 附录 A：关键源码索引

| 内容 | 文件 |
|---|---|
| 顶层/两个 partition_m | `vmod/nvdla/top/NV_nvdla.v` |
| CMAC core / 8 个 Core_mac / rt_out | `vmod/nvdla/cmac/NV_NVDLA_CMAC_core.v` |
| 16×16 Booth 乘法器 | `vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_MAC_mul.v` |
| 指数 max / shift | `vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_MAC_exp.v` |
| 64 乘积 CSA 树 / sop_0 | `vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_mac.v` |
| FP16 输入重编码 / NaN | `vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_active.v` |
| CACC FP cell | `vmod/nvdla/cacc/NV_NVDLA_CACC_CALC_fp_48b.v` |
| CACC INT8/INT16 cell | `vmod/nvdla/cacc/NV_NVDLA_CACC_CALC_int8.v`、`NV_NVDLA_CACC_CALC_int16.v` |
| CACC calculator / 32×4 phase | `vmod/nvdla/cacc/NV_NVDLA_CACC_calculator.v`、`NV_NVDLA_CACC_assembly_ctrl.v` |
| CACC assembly buffer | `vmod/nvdla/cacc/NV_NVDLA_CACC_assembly_buffer.v` |
| 输出 delivery / credit | `vmod/nvdla/cacc/NV_NVDLA_CACC_delivery_ctrl.v` |

## 附录 B：术语

| 术语 | 含义 |
|---|---|
| SOP | sum of products；一个输出通道的局部乘积和 |
| lane | 一个输出通道的 partial sum 存储槽 |
| bank | 一组 16 lane 的 SRAM 宏 |
| row | 每个 bank 的 32 个地址之一 |
| phase | 同一 row 上分 4 拍访问不同 bank 的时隙 |
| carry-save | 用 sum/carry 两个向量表示一个数，暂不进位 |
| CPA | carry-propagate adder，真正做进位的加法器 |
| 最近舍入（平局远离零） | 恰好一半时向幅值增大方向进位，不是 ties-to-even |
| guard/sticky | 舍入判断用的保护位和“低位是否有 1” |
| q | `exp[4:2]`，以 4 bit 为一档的指数码 |
