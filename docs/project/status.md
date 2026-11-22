# 项目状态

状态更新日期：2026-11-22。下文旧基线证据日期为2026-10-31。

本文记录新架构的设计、RTL 验证与数值实验证据，以及旧 SystemVerilog dot16/Q2.34 基线的历史实现与验证状态。新架构已审核设计约束、验证边界及后续候选见[已确认约束](decisions.md)。

## 新架构设计与证据

- **新架构低有效异步复位**：MAC 核心的流水数据和控制、批次寄存器及全部 ACC 槽在 rst_n 拉低时清零，生命周期置 FREE。沿前清零、事务取消和释放后重启的定向测试 5 项通过，更新后的硬件验证回归 233 项通过。复位释放由上层在 clk 域同步控制。见[复位约定](../design/new_architecture/IP_DESIGN.md#83-复位暂停与容量边界)及[验证说明](../../hardware/verification/README.md#新架构-mac-核心与-acc)。

- **阶段 3 MAC 核心**：[顶层 RTL](../../hardware/rtl/verilog/new_arch/mac_core.v)及[独立周期参考](../../hardware/verification/reference/new_arch/mac_core.py)已完成。40 项共仿、648 项全量回归通过。共仿覆盖双路 ACC、E0→E11 的固定 11 个时钟间隔、D=3 三槽交错、完成状态交接、复位和异常调度断言。10 种合法 `mode/mx` 组合均验证 16384 项定向长点积。见[核心与 ACC 验证](../../hardware/verification/README.md#新架构-mac-核心与-acc)。

- **阶段 2 MAC 前向链**：[RTL](../../hardware/rtl/verilog/new_arch/mac_forward.v)及[独立 Python 参考](../../hardware/verification/reference/new_arch/forward.py)已完成。29 项 Verilog-2001 共仿通过，逐拍检查 E0→E8 的固定八个时钟间隔、各级有效数据和元数据，覆盖 10 种合法 `mode/mx` 组合。阶段 2 全量回归 608 项通过。见[前向链验证](../../hardware/verification/README.md#新架构-mac-前向链)。

- **阶段 1 独立输入转换器**：[RTL](../../hardware/rtl/verilog/new_arch/mac_input_convert.v)及[独立 Python 参考](../../hardware/verification/reference/new_arch/input_convert.py)于 2026-11-17 完成，阶段 1 初版全量回归 579 项通过。2026-11-18 将输出寄存器改为低有效异步清零，由上层在转换器时钟域同步释放复位。更新后 Verilog-2001 共仿 9 项、硬件验证回归 163 项通过。共仿覆盖 524,288 组 `fmt/raw` 的 22 位输出，以及格式切换、一级 valid、气泡保持、沿间复位清零和释放后恢复采样。复现命令见[硬件验证说明](../../hardware/verification/README.md#新架构独立输入转换器)。

- **2026-11-14用户整体审核通过**：[IP_DESIGN](../design/new_architecture/IP_DESIGN.md)作为新架构Verilog实现与功能验证的设计基线，包含接口、数值、资源、周期及审查修订。数值精度采用误差指标报告与对照分析，不设统一通过阈值。功能验收要求RTL逐位符合数值合同。

- 2026-11-14审查修订：明确Eeff、MAC有效位切片、scale宽域临时量及逐输入合法性检查；补充每前端R_IN→R_MUL共享9-bit无符号指数加法器、BF16指数和E1写入R_MUL并于E1→E2执行postfold、非零MAX树A组unsigned7与B组unsigned4比较；采用低8位旁路重构和MX符号异或检测/负端修正，补充X隔离及限定浮点非旁路的T!=-1断言。当前仍为三段ACC、D=3；四段ACC仅为实际时序紧张后的候选。

- MAC 核心止于 ACC 状态交接。E11 的 last 写回使上下文变为 DONE，状态留在原槽至指定读取握手，释放后的下一拍可复用。G=2 两路独立累计并共同交接，每路状态 51 位。外部提供合法调度，并在三个槽均已释放且流水排空后切换模式。
- 已有[局部对阶实验](../experiments/README.md)和[累加精度实验](../experiments/accumulation_precision.md)的数值模型与运行结果。局部实验用后续精确累计隔离局部损失；累加实验用精确dyadic相加后RNE评价状态精度。二者的配置、覆盖范围及结论边界见各自说明与[已确认约束](decisions.md)，不能替代有限工作位电路或全部合法输入域的验证。
- 新架构的 GLS、综合、STA 和 PPA 仍待实施。
- [新架构 IP 设计文档](../design/new_architecture/IP_DESIGN.md)覆盖总体接口、独立转换、乘法配对连接、MAX/对阶资源、两树归约、MX、ACC、完成读取、物理寄存器及验证场景。核心采用入口 E0 至 E11 写回的固定延迟、8 条 38-bit 加 24 条 32-bit 对阶器、两棵分拍平衡树、低有效异步复位和无效读取清零。

后续架构调整和任务顺序由用户确定。以下保留设计审核期间的模型验证记录。

本轮设计稿检查（2026-11-12）：文档链接检查通过；既有累加模型定向测试 `-k 't0 or t2 or t5'` 为 **14 passed、6 deselected**，日志见[acc_directed.log](../../results/verification/ip_design_review_20261112/acc_directed.log)。这只复核贡献规格化、ACC边界和范围，不覆盖新增前向流水或接口。未运行小p穷举、代表轨迹重放、正式数值实验、RTL或PPA；本轮未新增验证代码。

## 旧基线已实现、已验证

| 内容 | 当前证据 | 边界 |
|---|---|---|
| 输入转换、乘法、ALIGN、PSUM、MX、ACC | 15 个生产 SystemVerilog 文件；与 RTL 设计文档指纹一致 | 16 对输入，已定义格式与合法输入域 |
| 硬件数值和事务验证 | 本次硬件测试 134 项全部通过 | 参考模型采用精确整数和 Fraction；握手由 RTL 测试覆盖 |
| 数值实验框架 | 本次实验测试 308 项全部通过 | 合成输入、明确的格式和误差口径；不等于模型任务精度 |
| 完整 TXT 交付 | 打包 345,168 条；MAC 9,385 步，连续和停顿两种调度通过 | 四类负例命中指定断言；停顿测试记录 6,739 个入口受阻周期 |
| 目录迁移 | 原有 12,486 个文件/符号链接已登记并核对 | 路径适配文件单独标注；外部项目、向量、RTL、历史数据保持内容 |

迁移后在新根目录环境执行 `python -m pytest -q -ra -p no:cacheprovider`，结果为 **442 passed in 53.12s**。这包括 134 项硬件验证和 308 项实验框架测试，没有将活动 Verilog 改写计作已通过实现。完整验收信息见[迁移验证记录](verification_20261031.md)。

## 旧基线开发中

`hardware/rtl/verilog/` 根层四个文件为旧基线的未完成改写。乘法模块只有未完成的接口，端口声明存在语法错误。转换与打包模块尚未接入回归。编译入口分别见[编译清单](../../hardware/filelists/README.md)。

## 旧基线实现与证据缺口

- 当前完整链路由 testbench 连接；没有生产顶层封装、最终 FP32/其他目标格式转换器或结果端 ready。
- MX 非零贡献的数学指数 E 超出 1…254 仍是暂缓定义域。仿真断言用于拒绝该输入，不代表综合后的恢复策略。
- ACC 写回已定义 FTZ/Inf；这与 MX 输入越界边界不同，不能混为一种策略。
- PSUM 仿真采用 DesignWare 功能替身；本次没有真实库验证、综合、STA、面积、功耗或布局结果。
- 暂未形成旧基线 Q2.34 语义下的完整精度与 PPA 权衡评估。早期 `hwacc-v1` 实验与旧基线实现的系数、指数及舍入过程存在差异，不能直接把旧位宽结论当作旧基线 RTL 的精度保证。
- 图稿和章节已有草稿，但图中的输出保持槽等描述仍需在工程实现完善后复核。

## 数据溯源

已保留的位宽结果在[分析目录](../../results/numerical/widthselect/analysis/)，对应[支撑报告](../../thesis/supporting_reports/results_width.md)。这批历史分析的 manifest、汇总和图像保留原始字节。

原先 `models/results/`、`models/results_width/`、`models/results_mx/`、`models/results_mx_align/` 和 `models/archive/` 在本次迁移前均不存在。`models.zip` 也不含这些原始运行数据或旧模型归档。现有资料无法完整重放有关正式实验。本次新产生的小规模运行单独使用 `migration_check_` 标识，不能替代缺失的历史正式数据。

## 文档的使用顺序

[已确认约束](decisions.md)是新架构已审核范围与验证边界的入口。[RTL 实现说明](../design/RTL_Design.md)和[架构设计](../design/IP_Design.md)描述旧 SystemVerilog 基线。[硬件验证说明](../../hardware/verification/README.md)给出新旧两套实现各自的执行入口与证据。发生差异时，先核对对应 RTL、独立模型和测试，再修订文档。涉及新架构语义时以新 IP 设计、已确认约束和用户后续确认为准。

[历史 ACC 交接说明](../../archive/handoffs/ACC36_REDESIGN_HANDOFF.md)记录迁移前的阶段决定，其中的未完成状态已被后续工作部分取代。
