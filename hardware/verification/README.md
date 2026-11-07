# 旧基线 MAC Q2.34 参考模型与验证

> 适用范围：本文记录旧 SystemVerilog dot16/Q2.34 基线。文中的 Q2.34 累加、FTZ/Inf 写回、MX 输入指数 1…254 边界及 dot16 事务语义只适用于该基线，不是 2026-11-05 新架构的数值合同；新架构已确认范围与未冻结项见[已确认约束](../../docs/project/decisions.md)。

旧基线设计见 [IP_Design.md](../../docs/design/IP_Design.md)。第二阶段已迁移 ALIGN、PSUM、MX、ACC RTL 及配套 testbench。独立 Python 数值模型保持不变，所有阶段与完整链路对照该模型；原旧格式 skip/xfail 已移除。

## 目录重组后运行（2026-10-31）

从项目根目录执行 `.venv/bin/python -m pytest -q hardware/verification/tests`；根 `pyproject.toml` 已配置参考模型和工具的导入路径。Python 测试位于 `tests/`，SV testbench 与 DesignWare 仿真替身位于 `tb/`，运行工具位于 `tools/`。

此次新环境回归中硬件 134 项通过，连同实验框架 308 项共 442 项通过。现有完整 TXT 向量连续和停顿测试、四类负例均通过。下文 2026-10-07/09 的记录为原始历史记录，所述数值合同只对旧基线继续适用。SystemVerilog、参考模型、testbench 和原始向量文件内容保持一致。

## 旧基线数值合同（Q2.34）

`reference/mac_contract.py` 使用精确整数和 `Fraction`，不用 Python float，不复用 `experiments/` 的 FP32 实验累加器，不模拟时钟。入口分类→M→逐项 ALIGN RNE→精确 PSUM→MX 精确重定标→保留对阶 tail 的 ACC 最终一次 RNE。

| 路径 | 部分和与 MX | 浮点指数 E（bias127） |
|---|---|---|
| FP9/MXFP8 | signed28 Q7.21 S，M=S<<8 | Emax−bias_A−bias_B+132+s_A+s_B |
| FP19 | 直接 RNE(P<<9,d) 得 signed32 Q3.29，PSUM signed36 Q7.29，M=S | Emax−bias_A−bias_B+132 |
| MXINT8 | signed15 乘积、signed19 dot16；Q2.17→Q2.34，M=S<<17；不求 k | s_A+s_B+132 |

无 MX scale 时 s_A=s_B=0；启用时 scale_a/b 为 E8M0 原始字节，s=raw−127，不能把字节当有符号指数。有限浮点值为 `M × 2^(E−127−34)`；FP9 的 Emax=32、bias=7/7、scale=0 对应 E=150。

FP19 新量化精度的范围、重定标和误差证明见设计文档§6.1.1：不是对旧 signed33 寄存器简单删位，不是旧 RNE 后再右移/舍入，也不承诺新旧已量化项逐位等价。

`FloatAcc36State(sig, exp, infinity_sign=0)` 使用 signed36 Q2.34。`trace_float_acc36_update()` 返回两路 signed36 floor 对阶操作数、signed37 完整整数和、两路精确非负 tail 及最终状态。tail 参与相加后的最终 RNE；深抵消时不丢弃余数。RTL 在 ACC 内先精确规格化系数，再采用3个低位工作位及 shift-right-jam，signed39→signed40 相加后一次 RNE；范围证明见设计文档§9.6.1。参考模型不复用硬件算法。

旧基线用户已确认的写回策略：正常 E=1…254；下溢 FTZ 为规范零 M=0/E=0；上溢保留有符号 Inf（E=255）；旧 Inf 加合法有限贡献保持，first 可清除。模型用 `infinity_sign=±1` 保存 Inf 符号，sig=0 仅为占位，不能代入有限公式。RTL 输出独立 signed2 `o_acc_float_inf_sign`（0、+1、−1），Inf 时系数为0、E=255。有限数仍完整保留36位系数。

**旧基线暂缓项：**按用户要求不处理非零 MX 输入公式 E 越界。`Mx.exp` 保留完整数学 E，不截成8位；ACC 遇到非零输入 E 不在1…254时报 `UnfrozenContract`，不自行重定标、回绕或生成 Inf/FTZ。零贡献忽略占位 E。此诊断与已定义的 ACC 写回 FTZ/Inf 是不同边界。

旧27位 ACC 模型已移除。旧 signed37 + signed12 lsb_exp 的 `update_float_acc36_candidate()` 接口明确报退役，不能静默把旧参数当 biased E；新接口为 `update_float_acc36()`。

## 事务和接口

- 源格式编码 FP16=0、BF16=1、E4M3=2、E5M2=3、INT16=4、INT8=5、INT4=6；compute_mode FP19=0、FP9=1、INT16=2、INT8=3、INT4=4；acc_kind INT32=0、TC_FLOAT=1。
- dot16、TAG_W 显式参数、入口整数最小负编码保留、E8M0=255 有效事务策略未定义等约束保持。
- 普通 INT8/INT4 每次 dot16 后 INT32 饱和；INT16 phase0 保存 signed36 中间值，不提交/饱和，phase1 完成后饱和和提交。权重仅由 PSUM 施加一次。
- ACC0/ACC1 独立；无效组保持；first 在实际消费时选择零旧值。模型先验证双组浮点更新再一起写入，暂缓的越界路径不会留下半包提交。
- `Snapshot.data` 对有限浮点保存完整36位补码编码，对整数保存32位编码。零为 data=0/E=0；Inf 为 data=None/E=255/infinity_sign=±1；无效组所有字段为 None。必须结合 group_valid、kind 与 infinity_sign 解释，不能继续使用旧32位浮点快照。
- Python 不验证周期握手；RTL 测试负责 ready/valid、反压保持、同拍消费/补入、同步复位、INT16 phase 顺序和 metadata。ACC TB 验证 valid 气泡；新增完整链路 TB 验证真实 MX→ACC 接受、上游反压和复位取消。ACC 仍无最终结果ready端口，不宣称最终输出反压已实现。

## 测试组织

| 测试文件 | 当前用途 |
|---|---|
| `test_mac_contract.py` | 独立 Q2.34 定向模型，3,000组 Fraction RNE 对照、1,000组 FP19 误差检查 |
| `test_mac_operand_pack.py`、`test_mac_mul_frontend.py`、`test_mac_entry_m_stage.py` | 入口分类、乘法、phase 与 metadata |
| `test_mac_align_units.py`、`test_mac_align_stage.py`、`test_mac_m_align.py` | 32/24位 ALIGN、MAX、正负 RNE、各模式/握手及非法输入 |
| `test_mac_psum_stage.py`、`test_mac_align_psum.py` | signed32/Q7.29 归约、limb 和真实 ALIGN→PSUM |
| `test_mac_mx_stage.py` | PSUM→MX 系数、8位biased E、E=150、MXINT8边界、反压、非法scale及暂缓指数域断言 |
| `test_mac_acc_stage.py` | Q2.34状态逐笔比较，37+3位求和/RNE、逐位抵消、FTZ/Inf、first、INT32饱和、INT16两phase、气泡、复位与非法输入 |
| `test_mac_q234_pipeline.py` | 原始输入→M→ALIGN→PSUM→MX→ACC，10种模式组合、800笔事务，停顿/metadata/phase/复位取消 |

全部比较为正常断言，不保留旧格式 xfail 或 skip。MX 的零贡献在8位物理编码边界规范成 E=0；非零输入 E 必须在1…254。随机向量限制在该已定义域，另外明确测试越界断言，不给暂缓域赋予回绕/FTZ/Inf 恢复语义。

Icarus 的 PSUM 测试使用 `dwbb_sim_models.sv` 中仅供仿真的 DesignWare 功能替身，不能作为 Synopsys 实际库、面积或时序验证。

## 本轮执行记录（2026-10-07）

以下为当前 RTL 和 testbench 的本轮执行结果；9月30日的参考模型阶段结果不作为当前 RTL 证据。

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=hardware/verification:hardware/verification/tools .venv/bin/python -m pytest \
  -q -ra -p no:cacheprovider hardware/verification/tests
```

- 完整回归：**127 passed in 8.34s**，无 skip/xfail。
- ACC testbench 对照独立模型比较 **15,067** 个更新向量；完整链路比较10种模式组合的800笔事务。
- ACC 普通编译及定义 `SYNTHESIS` 宏的编译均通过；完整链路额外通过 `iverilog -g2012 -DSYNTHESIS` 编译。类型包独立编译与仿真通过。
- 独立 Python 参考模型与本轮修改前备份逐字节一致，没有为适配 RTL 改弱模型。
- 当前环境无 Verilator、Yosys 或 dc_shell；未执行实际综合、面积或时序验证。定义 `SYNTHESIS` 宏仅是编译检查，DesignWare 仍使用仿真功能替身。

## TXT 测试向量交付（2026-10-09）

当前向量位于 `verification/vectors/`，固定seed `0x2342026`、TAG_W=4。

| 文件 | 内容 |
|---|---|
| `pack_vectors.txt` | 345,168条打包模块输入/期望输出，含FP16/BF16全部原始编码、两种FP8全部byte组合、合法INT16编码穷举，以及组屏蔽定向/随机测试 |
| `mac_inputs.txt` | 9,385个显式phase输入步骤；每步15字段头＋16行26位operand对 |
| `mac_expected.txt` | 每步ACC消费后的完整状态及提交信息，共9,284次提交；包含INT16 phase0不提交时的保持状态 |
| `mac_intermediate.txt` | 每步M、ALIGN、PSUM和MX期望值，用于逐级比较 |
| `mac_invalid_*.txt` | 4个独立非法输入用例；不指定数值结果，要求命中对应RTL断言 |
| `vectors_readme.txt` | 完整字段顺序、位宽、seed、覆盖统计、重现命令和数据文件SHA-256 |

所有主数据文件只包含固定宽度十六进制字段，负数使用相应位宽的补码，便于 `$fscanf` 读取。MAC输入是直接构造的26位操作数，参考入口为 `reference/packed_mac.py`，不调用raw转换或 `enter()`。定向 Emax=32/bias_sum=14/E=150 向量属于计算接口的合成输入，不冒充可由E4M3原始编码直接得到的输入；随机FP19/FP9字段按源格式构造。

MAC序列编号只是标识，不是复位命令。全文件开头复位一次，新序列依靠first开始累加，未选中的组和数值bank继续保持历史值。INT16显式提供phase0和phase1；期望文件保留每个phase之后的状态及update信号。当前终点为Q2.34 ACC状态，没有FP32输出转换。

`tb_mac_packed_txt.sv` 直接实例化16个 `mac_mul_frontend`，增加测试专用单槽M输出寄存器，再连接真实ALIGN→PSUM→MX→ACC。不编译/实例化 `mac_entry_m_stage`、打包模块或原始格式转换器。MX→ACC反压门只用于测试，valid/ready同时受控。`tb_pack_txt.sv` 则单独实例化一个打包模块。

从仓库根目录重现交付向量并执行RTL检查：

```sh
python3 hardware/verification/tools/generate_vectors.py --exhaustive-pack --stress
python3 hardware/verification/tools/run_txt_vectors.py --build /tmp/mac-txt-build
```

生成器仅依赖Python标准库；RTL检查需要 `iverilog` 和 `vvp`。`--out` 可指定输出目录，runner用 `--vectors` 指向该目录。去掉 `--exhaustive-pack --stress` 可生成较小的日常回归集；`--seed` 和 `--random-count` 控制随机序列。当前TXT MAC TB每次最多载入32,768个phase步骤。

本轮实际执行结果：

- 打包 TXT：345,168条全部通过。
- MAC TXT：9,385步在连续输入和气泡/停顿两种时序下均通过，逐级检查数据和metadata；停顿模式发生6,739个入口受阻周期。
- 缺少first、scale255、非零MX指数越界、无phase0的phase1四类负例，分别命中指定断言。
- 新增测试还验证可复现生成、手算转换/舍入值、直接packed入口不调用raw转换、phase配对和越界拒绝；故意破坏ACC期望值时TB确实报错。
- 完整回归：`134 passed in 9.20s`。运行命令同前述全量pytest命令。
- 15个生产RTL文件指纹与设计文档快照一致，原有 `mac_contract.py` 未修改。新增的是向量生成器、packed参考入口、TXT testbench与运行器。
- 本轮未执行综合/面积/时序分析；DesignWare仍使用仿真替身。
