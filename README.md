# 面向大语言模型推理的多精度融合 MAC

**Multi-Precision Fused MAC for LLM Inference** — 一个运行时动态可配置的多精度融合乘累加（MAC）硬件核心（Verilog-2001 RTL），面向大语言模型推理场景，重点优化 OCP Microscaling（MXFP8 / MXFP4 / MXINT8）与低比特浮点/整数的计算密度、资源共享及长点积累加精度。

本项目包含完整的架构规格、16 个 Verilog-2001 RTL 模块、独立 Python 位级/周期级参考模型、Icarus Verilog 逐拍共仿验证套件、数值精度实验框架、电路微架构图稿，以及基于 SMIC 28 nm 标准单元库的三轮 Synopsys DC 逻辑综合数据。当前新架构与 Verilog RTL 已定型冻结。

---

## 核心特性

- **10 种运行时可配置模式**：一套核心统一支持 **FP4 / MXFP4 (E2M1)**、**FP8 / MXFP8 (E4M3FN、E5M2 及双向混乘)**、**INT4 / INT8 / MXINT8 / INT16**、**BF16** 与 **FP16**，无需为不同格式堆叠独立计算核。
- **全链路深度资源复用**：
  - **乘法前端**：16 组共享前端（每前端 `2 × signed9×9 + 2 × signed5×5`），覆盖从 64 路 4-bit 并行乘积到 8 路 16-bit 四部分积重构；
  - **局部对阶与归约**：FP8/FP16 乘前指数折叠 + BF16 乘后折叠（postfold），统一共享 32 条 4-bit 步进对阶移位器与 2 棵 16→36-bit 平衡归约树（通过 Offset-Binary Packing 并行归约双组 FP4/INT4，通过 19/19 limb 拆分复用处理 38-bit FP16）；
  - **长点积累加**：双路 45-bit 补码主加法器在浮点/MX 与普通整数间共享。
- **严谨的有限精度数值设计**：基于同源数值实验与误差界推导，选定 `p_acc = 40`（`signed41` 补码尾数 + `signed10` 指数）、`2-bit 真实低位 (G, R) + 1-bit sticky` 直接补码累加数据通路，支持长达 `L_max = 16,384` 的长点积无内部溢出累加与加后 RNE。
- **高均衡度 11 级流水与持续吞吐**：`E0 → E11` 固定 11 拍流水、三段式 ACC（`D = 3`）配合每路 3 组独立上下文实现 `II = 1` 交错持续供数；在 28 nm 超频压力测试中，前/中/后 5 条核心流水级在 `0.520 ~ 0.543 ns` 同步触顶，无明显单点短板。
- **高算力密度**：SMIC 28 nm（TT 0.9 V 25 °C，零线载）逻辑综合在 **1.25 GHz** 实现 Setup / Hold / DRC 全零违例，面积仅 **`0.0381 mm²`**（FP4 峰值算力 **160 GOPS** / **4.20 TOPS/mm²**），物理极限约 **1.43 GHz**。

---

## 支持格式与峰值吞吐

以 16 个物理乘法前端、合法多上下文交错（`II = 1`、`1 MAC = 2 Ops`）为基准：

| 格式模式 | MX 块缩放 | 每拍归约宽度 × 组数 (`K / G`) | 完整乘积 / 拍 | 乘法前端资源映射 | 峰值算力 @ 1.00 GHz | 峰值算力 @ 1.25 GHz |
| :--- | :---: | :---: | :---: | :--- | :---: | :---: |
| **FP4 (E2M1) / INT4** | FP4 支持 | `32 / 2` | `64` | 每前端全部 4 颗块并行（无闲置） | `128 GOPS` | `160 GOPS` |
| **FP8 (E4M3/E5M2) / INT8** | 均支持 | `16 / 2` | `32` | 每前端 2 颗 `9×9`（含 E4M3×E5M2 双向混乘） | `64 GOPS` | `80 GOPS` |
| **BF16** | — | `16 / 1` | `16` | 每前端 1 颗 `9×9` + 乘后折叠（postfold） | `32 GOPS` | `40 GOPS` |
| **FP16 / INT16** | — | `8 / 1` | `8` | 两前端配对，4 颗 `9×9` 同拍重构 4 部分积 | `16 GOPS` | `20 GOPS` |

> 注：OCP MX 模式采用 32 元素缩放块与 8-bit E8M0 共享 scale（`0x00…0xFE`），MX scale 在局部归约后的独立 MX 级通过无损指数加法精确应用；普通 INT4/INT8/INT16 路径在 45-bit 累加器中精确求和，MXINT8 计入 `2^-12` 隐含权重并走浮点累加通路。

---

## 微架构概览

独立输入转换器（`mac_input_convert`）位于核心之外，负责原始编码解码、subnormal 精确保留、特殊值有限化映射及 FP8/FP16 乘前指数折叠，输出统一 22-bit 字段 bundle。`mac_core` 内部采用固定 11 级流水（低有效异步复位、门控时钟）：

```text
  22-bit Bundles + Mode/MX/Scale/Ctx
                  │
     ┌────────────▼────────────┐
     │  E0→E1   R_IN → R_MUL   │  16 前端共享乘法 (32× 9×9 + 32× 5×5) & 共享 9-bit 指数加
     │  E1→E2   R_MUL → R_PROD │  FP16/INT16 低 8 位旁路四部分积重构 & BF16 postfold
     │  E2→E3   R_PROD → R_MAX │  双组 16 输入非零 MAX 树 (A: u7, B: u4) + 逐槽非零标志
     │  E3→E4   R_MAX → R_ALIGN│  8×38b + 24×32b 4-bit 步进对阶截断 / FP4·INT4 Offset-Binary Packing
     │  E4→E6   R_TREE_LO / HI │  两棵 16→36b 平衡归约树（两拍切分）
     │  E6→E7   R_TREE → R_SUM │  41-bit 最终合并（FP16 19/19 limb 拼接 / FP4·INT4 偏置消除）
     │  E7→E8   R_SUM → R_MX   │  MX scale 应用 (e_loc + h) + 符号异或前导检测无损左移规格化
     │  E8→E9   ACC S1 (R_ACC1)│  3 上下文槽组合读 + 指数比较 + 对阶与尾部摘要 (G, R, S)
     │  E9→E10  ACC S2 (R_ACC2)│  双路共享 45-bit 补码主加法器（浮点/MX 与整数复用）
     │ E10→E11  ACC S3 → WB    │  规格化 + RNE 舍入 + 重规格化并行指数预计算 + 状态写回
     └────────────┬────────────┘
                  ▼
     3-Slot Context Handshake (o_state0/1 [50:0], o_ctx_free/done, o_pipe_empty)
```

### 电路微架构图集

完整 Draw.io 源文件位于 `assets/diagrams/source/new_architecture/`，高清导出图如下：

| 模块 / 流水段 | 架构图 |
| :--- | :--- |
| 独立输入转换器 (`mac_input_convert`) | [input-converter.drawio.png](assets/diagrams/exports/new_architecture/input-converter.drawio.png) |
| 共享乘法前端与重构 (`E0 → E2`) | [shared-multiply.drawio.png](assets/diagrams/exports/new_architecture/shared-multiply.drawio.png) |
| 非零 MAX 树与局部对阶 (`E2 → E4`) | [local-alignment.drawio.png](assets/diagrams/exports/new_architecture/local-alignment.drawio.png) |
| 平衡归约树与最终合并 (`E4 → E7`) | [local-reduction.drawio.png](assets/diagrams/exports/new_architecture/local-reduction.drawio.png) |
| MX Scale 与无损规格化 (`E7 → E8`) | [mx-normalization.drawio.png](assets/diagrams/exports/new_architecture/mx-normalization.drawio.png) |
| 双路三段补码累加器 (`E8 → E11`) | [acc-datapath.drawio.png](assets/diagrams/exports/new_architecture/acc-datapath.drawio.png) |

<details>
<summary>点击展开预览核心数据通路图（共享乘法 & 补码累加器）</summary>

![Shared Multiply Datapath](assets/diagrams/exports/new_architecture/shared-multiply.drawio.png)
![ACC Datapath](assets/diagrams/exports/new_architecture/acc-datapath.drawio.png)

</details>

---

## 28 nm 逻辑综合结果

基于 **SMIC 28 nm HKC RVT** 标准单元库（Synopsys DC `V-2023.12-SP3`，`TT 0.9 V 25 °C`，零线载模型，`150 ps` Setup / `50 ps` Hold Uncertainty，ICG 门控覆盖率 `99.72%`），完整归档于 `results/synthesis/dc/`（本地归档目录，未入 Git 追踪）：

| 综合版本 / 频点 | 目标频率 | 关键路径 | WNS / TNS | 逻辑级数 | Hold / DRC | 总面积 (`µm²`) | FF + ICG | 默认估算功耗 (`mW`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `01_baseline` (初版 RTL) | `500 MHz` | `1.558 ns` | `+0.279 / 0` | 52 | `0 / 1` (3ps trans) | `36,670.8` | `6,743 + 219` | `4.794` |
| `02_opt_rtl` (冻结 RTL) | `500 MHz` | `1.514 ns` | `+0.322 / 0` | 52 | `0 / 1` (3ps trans) | **`34,484.1`** | `5,989 + 195` | **`4.454`** |
| `03_freq_sweep` (`1p000ns`) | **`1.00 GHz`** | `0.854 ns` | `0.000 / 0` | 34 | `0 / 1` (3ps trans) | **`36,660.0`** | `5,981 + 195` | `8.994` |
| `03_freq_sweep` (`0p800ns`) | **`1.25 GHz`** | `0.638 ns` | **`0.000 / 0`** | **31** | **`0 / 0`** | **`38,094.6`** | `5,981 + 195` | `11.454` |
| `03_freq_sweep` (`0p700ns`) | **`1.43 GHz`** | `0.541 ns` | `0.000 / 0` | 36 | `0 / 4` (fanout≤31) | `40,005.8` | `5,989 + 195` | `13.414` |

- **微架构优化收益**：同条件 500 MHz 下，`02_opt_rtl` 通过乘积槽合并（统一 `16×36-bit`、`q1/r_max_b` 收窄至 4-bit）、移位器收窄及 ACC S2/S3 逻辑精简，净减 **754 个 FF 与 24 个 ICG**，总面积下降 **`5.96%`**、估算功耗下降 **`7.08%`**；在 **1.00 GHz** 下面积（`36,660.0 µm²`）仍略低于初版 500 MHz 基线，实现等面积 2× 主频。
- **极限频率与流水均衡性**：在 `0.600 ns`（WNS `-0.103 ns`）与 `0.550 ns`（WNS `-0.141 ns`）超频扫描下，物理极限锁定在 **`~1.43 GHz`（`0.691 ~ 0.703 ns`）**，且 `E0→E1`、`E2→E3`、`E7→E8`、`E8→E9`、`E10→E11` 五条核心流水级在 `0.520 ~ 0.543 ns` 同步触顶。
- **数据口径说明**：上述结果属 TT 典型角、RVT 单阈值、零线载、无 SAIF/VCD 翻转率标注的逻辑综合数据，不代表 SS 慢角或布局布线后签核结果。

---

## 快速开始与验证

### 环境要求

- Python `3.10+`（本机验证环境：Python `3.13`）
- Icarus Verilog（`iverilog`、`vvp`），用于 RTL 仿真

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m mac_experiment version
```

### 运行验证与实验

```sh
# 1. 运行硬件与参考模型验证（233 项，含新架构转换器穷举、前向链与 MAC 核心共仿）
.venv/bin/python -m pytest -q hardware/verification/tests

# 2. 运行全部测试（硬件验证 + 数值实验框架）
.venv/bin/python -m pytest -q

# 3. 检查项目文档链接
.venv/bin/python scripts/check_project.py --checks links

# 4. 运行局部对阶数值探索（smoke 配置）
.venv/bin/python -m mac_experiment local-align-run --config experiments/configs/local_alignment/smoke.yaml --quiet

# 5. 运行旧 SystemVerilog 基线交付向量
.venv/bin/python hardware/verification/tools/run_txt_vectors.py --build results/verification/txt
```

### 已完成验证摘要

- **独立输入转换器**：穷举验证 8 种源格式全部 16-bit raw 码字（`524,288` 组输入）及异步复位/气泡时序；
- **MAC 前向链与完整核心**：独立 Python 周期参考模型与 Verilog-2001 RTL 逐位、逐拍共仿通过，覆盖全部 10 种合法 `mode/mx` 组合、`D = 3` 三槽交错调度、异步复位、异常调度负例断言及 **16,384 项定向长点积**；
- **数值精度评估**：覆盖 4 种分布（Gaussian / Laplace / Student-t3 / outlier）、6 种路径与长度（最高 `L = 16,384`）的局部对阶与 `p_acc` 累加精度同源实验（138,240 条记录）。

---

## 仓库结构

```text
NPU/
├── hardware/
│   ├── rtl/verilog/new_arch/      # 定型新架构 Verilog-2001 源码（15 核心模块 + 1 输入转换器）
│   ├── rtl/systemverilog/         # 旧 dot16/Q2.34 SystemVerilog 基线（保留作功能参考）
│   ├── filelists/                 # 新旧架构 RTL 编译文件清单
│   └── verification/              # 独立 Python 位级/周期参考模型、Icarus 共仿与回归测试
├── experiments/                   # 多精度数值建模、局部对阶与累加精度实验 Python 包
├── docs/
│   ├── design/new_architecture/   # 新架构 IP 完整设计规格（IP_DESIGN.md）
│   ├── project/                   # 项目状态（status.md）与已确认设计决策（decisions.md）
│   └── experiments/               # 数值实验方法与精度口径说明
├── assets/diagrams/               # Draw.io 微架构图源与高清 PNG 导出
├── thesis/                        # 毕业论文初稿与实验支撑报告
├── references/                    # 参考文献（含 OCP MX v1.0 规范）与开源参考项目索引
├── results/                       # 数值实验输出、验证日志与 28 nm DC 综合归档
└── scripts/                       # 项目检查与图稿辅助脚本
```

---

## 文档导航

- **新架构核心文档**：[IP 设计规格](docs/design/new_architecture/IP_DESIGN.md) · [已确认设计决策](docs/project/decisions.md) · [项目状态与综合记录](docs/project/status.md)
- **验证与编译入口**：[硬件验证说明](hardware/verification/README.md) · [RTL 编译清单](hardware/filelists/README.md) · [数值实验指南](docs/experiments/README.md) · [累加精度实验](docs/experiments/accumulation_precision.md)
- **论文与参考资料**：[论文材料](thesis/README.md) · [参考文献](references/papers/README.md) · [开源项目索引](references/projects/README.md)
- **旧基线参考文档**：[旧基线架构设计](docs/design/IP_Design.md) · [旧基线 RTL 实现说明](docs/design/RTL_Design.md)（描述历史 SystemVerilog dot16/Q2.34 基线，不构成新架构合同）
- **协作约定**：[Agent 工作指令](AGENTS.md)

---

## 设计范围与已知边界

- **模块边界**：`mac_core` 聚焦内层 `II = 1` 高频乘累加核心，输入解码/折叠由独立 `mac_input_convert` 模块前置完成，输出止于 51-bit ACC 上下文状态交接（最终低频 FP32/INT32 转换与量化由外部承担）。
- **调度前提**：持续 `II = 1` 满吞吐依赖外部提供合法的 `D = 3` 多上下文交错调度（至少 3 个独立输出上下文轮转）及 MX 块边界对齐补零。
- **语义边界**：元素输入采用面向推理的有限化映射（NaN→0，±Inf→有限最大值，subnormal 精确保留）；当前硬件验证聚焦 RTL 功能正确性与 TT 零线载逻辑综合，SS 慢角、LEC/GLS、带 SAIF 波形功耗及后端布局布线未纳入本轮实测。
