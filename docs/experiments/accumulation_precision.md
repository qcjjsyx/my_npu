# 交错累加状态精度实验方案

本方案研究新架构逐份累加的精度，不是旧 widthselect 的操作数预舍入实验。架构合同见[已确认约束](../project/decisions.md)。正式实验由用户运行；代码开发只执行定向测试和微型检查，不自动运行正式模板。结果待运行后审查，不预设候选通过。

## 问题与数值合同

初始比较点为 p_acc=[40,44,48]；配置支持任意不小于40的整数精度，当前正式模板扩展为 [40,42,44,46,48]。p_acc 是对阶主窗口的幅值位数，包含最高有效位、不含符号位，同时是规格化写回精度。状态 C=M*2^(H-(p_acc-1))，非零时 2^(p_acc-1)<=abs(M)<2^p_acc；M 为 signed(p_acc+1)，H 为 signed10 实际最高位指数，无 bias。零统一 M=H=0。

新贡献保留全部局部系数，不先独立舍入。逐次更新 C_next=RNE_p(C+v)，无配对，无跨更新残差。主窗口之外仍需本次运算的保护和尾部信息；本实验用精确 dyadic 相加后整数 RNE 建立数值参考，不声称已经验证某个固定 GRS 宽度或 RTL 对阶实现。40 位起点可覆盖当前 FP16 signed41 合法局部贡献的幅值精度；物理加法器总位宽不能直接称为40位。

每次非零写回检查 H 在 [-512,511]，越界报错且不得产生成功完成的实验标记；不擅自设置内部 FTZ、饱和或 Inf。最终 FP32 使用 RNE、有限值饱和、渐进下溢，精确零为+0，非零舍入为零保留符号。普通整数不参与 p_acc 扫描，精确整数合同不改变。

## 固定局部路径及覆盖范围

第一版完整实现已有 local_alignment 支持的六条当前路径：四种 MXFP8 组合（E4M3FN/E5M2 同格式及双向交叉）、FP16、BF16 postfold。固定 floor：MXFP8 F=16、K=16；FP16 F=8、K=8；BF16 postfold F=11、K=16。原 bf16 未折叠模式不是本实验候选。

直接复用 local_alignment 的解码、量化、完整乘积和局部归约，保持历史代码和输出语义。普通 FP8、FP4/MXFP4、MXINT8 的随机输入链尚未包含在本版覆盖中，不以这六条路径结果宣称全部模式通过；后续补齐需单独安排。加法核心本身通过精确贡献构造覆盖小整数和不同指数。

## 数据及规模

正式模板：seeds=[1,2,3]，每种子64条序列；分布 gaussian、laplace、student_t3、outlier；长度 [128,512,1024,2048,8192,16384]；p_acc=[40,42,44,46,48]。每桶192条，P99为经验分位数，不代表极端概率保证。正式模板共768条同源最大长度输入，六种模式共4608条编码序列，五个精度及六种长度共产生138240条结果记录；不预估运行时间。

同一分布/种子/序列只生成一份最大长度 FP32 A/B；所有格式同源量化，同一模式的所有 p 候选复用同一编码及 scale，各长度取同一量化长序列前缀。不对每个长度重新计算 MX scale。MX 块长32，两侧 scale 关联沿用当前合同。记录量化事件但不把量化误差计入 MAC 误差。

微型配置使用一个种子、一条序列、一种分布、长度17和32，覆盖六条路径和三个候选，共36条结果，同时检查尾段补零；不能代替正式结果。运行器将全部样例记录和各桶代表轨迹保存在内存中后生成汇总，正式配置增加样本量时内存随记录数增长。

## 参考链和指标

S 为编码输入及 scale 的精确乘积总和；V 为固定局部贡献的精确总和；C_p 为逐份有限精度累计结果；L1=sum(abs(a_i*b_i))。记录 V-S（局部）、C_p-V（新增状态）、C_p-S（总内部）三组误差，及绝对误差/L1；L1=0单列。保留精确 dyadic 值以支持重算，展示统计可转 binary64。

Y_ref=Q32(S)，Y_local=Q32(V)，Y_p=Q32(C_p)。分别统计 Y_p 对 Y_ref、Y_p 对 Y_local 的 bit-match、ULP p50/p99/max；饱和样例的匹配与非饱和匹配分开，非饱和 ULP 分位数单列；记录 subnormal/零输出及符号零差异。不得因共同饱和推断内部误差小。近抵消用 abs(S)/L1 描述，避免用接近零的 S 作唯一归一化分母。

记录每序列状态更新数、非零贡献未改变状态的次数、非零状态指数最小/最大值。每桶保留最大 abs(C_p-V) 的代表，保存编码及精确状态轨迹供重放；不需为全部样例保存全部轨迹。不存在统一验收阈值，各候选精度的取舍由用户审查结果后决定。

## 实现与验证

新增独立 accumulation_precision 实验模块、配置及 CLI，不修改旧 hw_add 数值语义。复用 Dyadic、量化和统计辅助能力。新参考用整数幅值 RNE；定向测试用独立 Fraction/整数商余数参考验证，包含正负半ULP、奇偶、舍入进位、深抵消、零、吸收、指数极值和越界。

验证零贡献保持、分段保存恢复与连续累计一致、独立上下文交错与各自连续执行一致；检查外部补零尾段和正确更新次数。来源编码保存后可脱离随机生成器重放；检查 summary 与 samples 重算一致及不同候选同源。正式运行只写新目录，不覆盖历史产物。

输出 config.json、metadata.json（合同版本、代码哈希、Git状态、分布及统计口径）、sources.jsonl.gz、samples.jsonl.gz、summary.json/csv、representatives.json、quantization.json。COMPLETE.json 成功标记仅在所有文件完整写出后生成；失败或中断目录不能视为有效完整运行。

## 运行与结果审查

以下命令将在本轮代码实现后验证。从项目根目录使用 .venv：

```sh
# 定向测试
.venv/bin/python -m pytest -q experiments/tests/accumulation_precision
# 用户可先运行微型配置
.venv/bin/python -m mac_experiment acc-precision-run --config experiments/configs/accumulation_precision/smoke.yaml
# 正式实验由用户启动
.venv/bin/python -m mac_experiment acc-precision-run --config experiments/configs/accumulation_precision/formal_template.yaml
# 用实际运行目录替换 <run_id>
.venv/bin/python -m mac_experiment acc-precision-replay --run results/numerical/accumulation_precision/runs/<run_id> --index 0
```

用户运行后提供运行目录。审查先确认完成状态、合同版本、配置、源码指纹和样例数量，再检查同源关系、量化与饱和分层、误差分解及代表重放，最后比较各长度和格式的各候选精度收益。实验只提供数值证据，不作为 RTL、时序、SRAM组织或 PPA 证据。
