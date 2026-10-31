# MAC RTL 详细设计说明书

**版本：V0.1 / 待审查初版**  
**编写日期：2026-10-08**  
**实现基线：项目根目录下 `hardware/rtl/systemverilog/` 的15个 SystemVerilog 文件，文件指纹见附录C。**

本文描述当前代码已实现的功能、组合逻辑、寄存器、接口和约束，供公司内部移植、代码规范化和继续开发使用。代码事实是本文的依据；本文不要求读者依赖 `IP_Design.md`，不描述 Python 数值模型，也不将此前仿真结果作为本版验证结论。文中“要求”表示合法调用条件，不等同于芯片中有对应错误恢复电路。

## 阅读导航

- 第1～3章：实现范围、模块组织、公共类型和数据表示。
- 第4章：15个源文件的逐项详细设计。
- 第5～6章：跨模块计算过程、时序、复位和控制协议。
- 第7章：SV testbench、依赖和集成方法。
- 第8章：当前限制、代码注释差异和规范化修改注意事项。
- 附录A：全部模块的原始端口声明及接口阅读约定。
- 附录B：当前仿真断言原文索引。
- 附录C：源文件指纹与本文检查范围。

## 1. 实现范围

本设计接收16对原始操作数，根据格式执行逐项乘法、块内对阶、dot16归约、MX缩放及跨事务累加。FP16/BF16、INT16使用单个逻辑组；FP8、INT8、INT4可以使用两个独立逻辑组。物理 lane0/lane1 与逻辑 group0/group1 并非在所有模式下一一对应。

| 原始格式 | 编码 | 计算模式 | 合法 A/B 组合 | group_valid | MX使能 | ACC类型 |
|---|---:|---|---|---|---|---|
| FP16 | 0 | FP19 | FP16×FP16 | 01 | 0 | TC_FLOAT |
| BF16 | 1 | FP19 | BF16×BF16 | 01 | 0 | TC_FLOAT |
| E4M3FN | 2 | FP9 | E4M3FN/E5M2任意两者 | 01、10、11 | 0或1 | TC_FLOAT |
| E5M2 | 3 | FP9 | 同上 | 同上 | 0或1 | TC_FLOAT |
| INT16 | 4 | INT16 | INT16×INT16 | 01 | 0 | INT32 |
| INT8 | 5 | INT8 | INT8×INT8 | 01、10、11 | 0或1 | MX=0时INT32；MX=1时TC_FLOAT |
| INT4 | 6 | INT4 | INT4×INT4 | 01、10、11 | 0 | INT32 |

整数输入的最小负编码保留：INT16不允许−32768，INT8不允许−128，INT4不允许−8。因此有效范围分别为±32767、±127、±7。INT4原始16位容器的高8位必须为零。浮点入口把NaN置零、可表示的Inf替换为源格式同符号最大有限数，保留subnormal；ACC计算产生的Inf则作为特殊累加状态保留。这是两个不同处理位置。

当前 `hardware/rtl/systemverilog/` 没有整体顶层模块。完整链路在 `tb_mac_q234_pipeline.sv` 中连接；它不是可直接用于产品的顶层。当前ACC输出内部状态及更新通知，没有FP32/其他目标格式输出转换器，没有结果端ready，没有异常状态寄存器或软件寄存器接口。所有时序模块只有一个 `i_clk` 时钟域。

## 2. 模块组织和流水结构

### 2.1 集成连接关系

```text
16对 raw A/B + 格式/组/first/tag/scale
        │
        ▼
mac_entry_m_stage
  32 × mac_operand_pack（每个含1个FP19转换器和2个FP9转换器）
  converted operand holding slot
  16 × mac_mul_frontend（每个 MUL12 + MUL9）
  M output slot
        │ 16×(product0/product1/exp/nz) + txn_meta + tag
        ▼
mac_align_stage
  2 × mac_align_max16
  16 × mac_align_lane0 → 各1个 mac_align_rne(W=32)
  16 × mac_align_lane1 → 各1个 mac_align_rne(W=24)
  ALIGN output slot
        │ 16×(signed32, signed24) + txn_meta + align_meta + tag
        ▼
mac_psum_stage
  2 × DW02_tree(16,28,1) + 2 × DW01_add(28)
  wide-mode merge / dual-mode pack
  PSUM output slot（56位）
        ▼
mac_mx_stage
  unpack / scale exponent / coefficient rescale
  MX output slot（两组36位系数 + 两组8位指数）
        ▼
mac_acc_stage
  2 × mac_acc_float_prepare36
  每组整数/浮点复用39→40位加法逻辑
  规格化/RNE 或 INT32饱和
  两组ACC状态 + INT16中间状态 + update通知
```

上图前后级连接由集成环境完成。各stage并不在前一个stage中实例化；stage内部的子模块关系才是实际RTL层级。两个组共享事务握手、mode和acc_kind，不能在同一事务中选择不同计算模式；每组的first、tag、scale和累加数值独立。

### 2.2 存储边界和容量

| 位置 | 存储内容 | 容量/说明 |
|---|---|---|
| entry holding slot | 转换后的32个operand、txn_meta、两个tag、phase | 一笔原始事务；INT16保留到phase1发射 |
| M output slot | 16路乘积、指数、nz及metadata | 一个phase/结果包 |
| ALIGN output slot | 16路32/24位对阶值、metadata | 一个结果包 |
| PSUM output slot | 56位打包结果、metadata | 一个结果包 |
| MX output slot | 两组系数/指数、精简metadata | 一个结果包 |
| ACC | 持久累加状态和单周期更新指示 | 不是带ready的输出队列 |

没有停顿时，普通模式入口最多每拍接收一笔；INT16每笔占两次M发射，稳态最多每两拍接收一笔。ACC对合法输入每拍都可以接受一个MX包，但INT16仅在phase1通知提交。实际频率和时序裕量未由上述结构推断或保证。

## 3. 公共类型与数值表示

### 3.1 类型、参数和枚举

`mac_types_pkg` 定义 `DOT_N=16`、`OPERAND_W=26`、`PSUM_W=56`、`ACC_OP_W=36`、`ACC_FLOAT_SIG_W=36`、`ACC_TAIL_W=3`、`ACC_WORK_W=39`。其中部分端口仍直接使用 `[0:15]`，所以DOT_N不是已完成端到端可配置的参数。

`compute_mode_t`：FP19=0、FP9=1、INT16=2、INT8=3、INT4=4，宽3位。`acc_kind_t`：INT32=0、TC_FLOAT=1，宽1位。`src_fmt_t`宽3位，编码见第1章。各时序stage的 `TAG_W` 无默认值，实例化必须显式提供正整数。MAX的EXP_W实际取9/6；RNE的W实际取32/24；没有承诺任意参数组合都可用。

| 类型 | 实际声明/位宽 | 使用含义 |
|---|---|---|
| operand_t | unsigned26 | 模式相关打包容器，不能整体当一个数 |
| product12_t / product9_t | signed24 / signed18 | 物理乘法器完整输出 |
| align_lane0_t / align_lane1_t | signed32 / signed24 | ALIGN物理输出 |
| psum_packed_t | unsigned56 | WIDE符号扩展或DUAL双组打包 |
| contribution0_t / contribution1_t | signed36 / signed36 | 浮点Q2.34或普通整数贡献 |
| acc_exp_t | unsigned8 | 浮点biased E；普通整数用0 |
| int_work_t / acc_sig_t | signed36 / signed36 | INT16中间整数 / 浮点系数 |
| m_lane_t | packed59 | product0/1、exp_sum0/1、product_nz的类型组合 |
| align_lane_t | packed56 | 32位lane0与24位lane1 |

后两个结构体是公共类型，当前stage接口采用分离数组端口，不能据结构体声明误认为总线端口已经打包成此类型。

### 3.2 metadata逐字段说明

| 结构体 | 字段 | 位宽 | 含义与消费位置 |
|---|---|---:|---|
| txn_meta_t | mode / acc_kind | 3 / 1 | 计算路由 / ACC状态类型 |
| txn_meta_t | bias_sum | 8 | 两源格式bias之和，MX计算E |
| txn_meta_t | group_valid / acc_first | 2 / 2 | 有效组 / 新累加序列首包；first只在有效组解释 |
| txn_meta_t | mx_enable | 1 | FP9或INT8使用scale |
| txn_meta_t | scale_a / scale_b | 各16 | packed `[1:0][7:0]`，索引是group再byte |
| txn_meta_t | int16_phase | 1 | 0/1阶段标志 |
| align_meta_t | emax0 / emax1 | 9 / 6 | 组内非零浮点乘积的最大有效指数和 |
| align_meta_t | group_nz | 2 | MAX是否有非零候选；不代表PSUM非零 |
| align_meta_t | inexact | 2 | 各组逐项ALIGN丢位标志的OR |
| mx_meta_t | mode / acc_kind / group_valid / acc_first / int16_phase | 3/1/2/2/1 | 从txn_meta复制的控制字段 |
| mx_meta_t | group_nz / inexact | 2/2 | ALIGN诊断信息，ACC不将其注入算术路径 |

三个结构体总宽分别为50、19、13位。tag不在结构体内，各级以 `[0:1]` 两个TAG_W位元素单独传递。tag用于标识事务，不参与选择累加器地址；物理只有group0和group1两组状态。

### 3.3 operand的26位布局

| 格式 | 系数字段 | 指数字段 | 其余位 |
|---|---|---|---|
| FP16/BF16 | [11:0] signed12 | [23:16] unsigned8 | 0 |
| 两个FP8 | group0:[4:0] signed5；group1:[9:5] signed5 | group0:[20:16]；group1:[25:21]，各5位 | 0 |
| INT16 | [15:0] signed16 | 无 | 0 |
| 两个INT8 | group0:[7:0]；group1:[15:8] | 无 | 0 |
| 两个INT4 | group0:[3:0]；group1:[7:4] | 无 | 0 |

FP8原始输入低byte为group0、高byte为group1；INT4仅用低byte中的两个nibble。无效组系数及其指数字段清零。

### 3.4 数值位宽链

本文Qm.n中的m包含符号位。浮点系数与指数必须一起解释。

| 路径 | 乘积有效系数 | ALIGN | PSUM | MX→ACC |
|---|---|---|---|---|
| FP19 | signed23，20个小数位，装入signed24 | `RNE(P×2^9 / 2^d)`，signed32 Q3.29 | signed36 Q7.29，符号扩展到56 | M=S，E=Emax−bias_sum+132 |
| FP9 | signed9，6个小数位，装入24/18 | `RNE(P×2^15 / 2^d)`，signed24 Q3.21，lane0扩为32 | 每组signed28 Q7.21 | M=S×2^8，E=Emax−bias_sum+132+scale_sum |
| INT8 | signed15，有效范围±16129 | 整数旁路 | signed19，有效范围±258064 | 普通整数符号扩展；MXINT8为M=S×2^17，E=132+scale_sum |
| INT4 | signed7，有效范围±49 | 整数旁路 | signed11，有效范围±784 | 整数符号扩展到36 |
| INT16 | 各phase的低/高部分积 | 整数旁路 | 含phase权重的WIDE整数 | signed36整数贡献 |

MXINT8的S在重定标时按Q2.17解释；E中的补偿保证真实整数值乘scale不变。普通整数从不使用浮点Q2.34解释。

浮点有限值为 `V=M×2^(E−161)`。规范非零写回满足 `2^34≤abs(M)<2^35`、E=1…254。零为M=0/E=0/inf_sign=0；Inf为M=0/E=255/inf_sign=±1。signed2的+1编码01，−1编码11，10不是已定义状态。ACC输入只接受有限贡献，非零M要求E=1…254；输入没有独立Inf标志。

## 4. 逐模块详细设计

本章给出端口的功能分组与实现说明；全部精确端口名、方向、类型和数组维度在附录A逐模块列出。除显式参数外，没有隐含可配置端口。

### 4.1 mac_types_pkg.sv：公共定义

该文件没有硬件实例和状态。常量、枚举、数据结构见第3章。`legal_src_pair`允许同格式或两种FP8混合；`legal_mx_enable`进一步限制MX只用于FP8/INT8。`mode_for`将FP16/BF16统一映射为FP19，两种FP8统一为FP9；非法输入的default返回FP19仅用于组合完备赋值，不是容错策略。

`acc_kind_for`为FP19、FP9及MXINT8选择TC_FLOAT，其余为INT32。`is_wide_mode`识别FP19/INT16；`is_float_mode`只识别FP19/FP9，因此MXINT8在M/ALIGN仍走整数路径。`lane1_active`在INT16时取group0，在DUAL模式取group1，FP19固定0。`source_bias`返回FP16/E5M2的15、BF16的127、E4M3的7；`bias_sum_for`先用9位相加再取8位，合法最大值254。

### 4.2 mac_fp19_convert.sv：FP16/BF16转换

组合模块，输入16位raw及src_fmt，输出8位有效指数和signed12补码系数，没有valid。处理顺序为拆字段、分类与替换、构造幅值并施加符号。

FP16使用sign[15]、exp[14:10]、fraction[9:0]；BF16使用sign[15]、exp[14:7]、fraction[6:0]并在fraction低端补3零。NaN和正负零得到系数0/指数0。FP16 Inf替换为E=30、fraction=1023；BF16 Inf替换为E=254、fraction=`7'h7f<<3`，均保留源符号。

有限非零数的有效指数为源E（E=0时改为1）；hidden位在源E非零时为1，subnormal时为0。11位幅值 `{hidden,fraction10}` 前补0扩成signed12，再按符号取负。BF16补零保持源精度，不生成额外有效信息。输出真实值为 `sig×2^(exp_eff−source_bias−10)`。不支持格式输出0，但合法性由entry在握手时检查。

### 4.3 mac_fp9_convert.sv：FP8转换

组合模块，输入8位raw及src_fmt，输出5位有效指数和signed5系数。E4M3FN使用sign[7]、E[6:3]、fraction[2:0]；E=15且fraction=7是NaN，E=15且fraction≤6仍为有限数，没有Inf编码。E5M2使用E[6:2]、fraction[1:0]，fraction低端补1零；E=31/fraction=0的Inf替换为E=30、扩展fraction=6。

NaN和零清为0；subnormal设置exp_eff=1且hidden=0。幅值为 `{hidden,fraction3}`，补正号后按源sign取负。真实值为 `sig×2^(exp_eff−source_bias−3)`。混合FP8的两源bias由entry分别计算，不共用错误的单一bias。

### 4.4 mac_operand_pack.sv：输入打包与隔离

每个A/B原始元素各用一个实例。先根据格式、group_valid隔离输入：未选中FP19转换器或无效组看到raw=0；两个FP9转换器分别看到低/高byte或0。三个转换器并行组合，随后按第3.3节位图写payload，默认全零，整数模式直接复制对应位段。

本模块不判断A/B格式组合、不检查整数保留码，也不存储metadata；entry在实际输入接受时承担这些检查。`group_valid[1]`在WIDE模式没有额外数据意义，必须由上级遵守01约束。隔离是组合选择零，不是显式时钟门控单元。

### 4.5 mac_mul_frontend.sv：单路共享乘法前端

每个dot元素实例化一次，共16个。两个物理乘法器输出signed24及signed18，乘法赋值使用完整结果宽度；本模块没有寄存器、乘积相加或phase权重左移。

| 模式 | MUL12输入 | MUL9输入 | active |
|---|---|---|---|
| FP19 | 两个signed12系数 | 0×0 | 仅group0 |
| FP9 | group0 signed5扩展至12 | group1 signed5扩展至9 | 对应group_valid |
| INT8 | group0 signed8扩展至12 | group1 signed8扩展至9 | 对应group_valid |
| INT4 | group0 signed4扩展至12 | group1 signed4扩展至9 | 对应group_valid |
| INT16 phase0 | unsigned A_lo × unsigned B_lo | signed A_hi × unsigned B_lo | 两路都由group0控制 |
| INT16 phase1 | unsigned A_lo × signed B_hi | signed A_hi × signed B_hi | 两路都由group0控制 |

INT16低byte补正号后再扩位，高byte符号扩展。FP19指数和为9位；FP9每组指数和6位，group0物理端口再补零到9位。`product_nz`由两个源系数均不为0生成。整数模式指数和与nz固定0（不代表整数乘积一定为0）。`o_product*_active`控制父级物理乘积寄存器的写使能。

### 4.6 mac_entry_m_stage.sv：入口保持与M寄存器

输入为16对16位raw、两源格式、group_valid、first、两个tag、MX使能及两组scale；输出M结果数组、txn_meta和tag。端口握手分别是 `i_in_valid/o_in_ready` 与 `o_m_valid/i_m_ready`。

关键寄存器为 `slot_a_q/slot_b_q`、`slot_meta_q/slot_tag_q`、`slot_valid_q/phase_q`，以及乘积/指数/nz寄存器、`m_meta_q/m_tag_q/o_m_valid`。组合组装in_meta；32个pack转换器供holding slot采样；16个mul从slot取数。

```text
m_space     = !o_m_valid || i_m_ready
issue       = i_rstn && slot_valid_q && m_space
release_slot= issue && (mode != INT16 || phase_q)
o_in_ready  = i_rstn && (!slot_valid_q || release_slot)
in_fire     = i_in_valid && o_in_ready
```

时钟边沿先按issue向M寄存器写旧slot的数据，slot控制的代码优先级为：in_fire装入新事务并phase=0；否则release_slot清valid；否则INT16 phase0的issue将phase设1。非阻塞赋值保证同拍旧事务出槽、新事务入槽互不覆盖。空slot收到输入不会同拍穿透到M。

INT16 phase1的M metadata把acc_first清为00，并设置int16_phase=1；phase0保持原first。仅active的乘积寄存器写入；指数/nz仅在浮点模式更新。输出乘积以o_m_valid和组/模式屏蔽，无效输出为0；指数/nz还以浮点模式屏蔽。meta/tag端口直接连寄存器，valid=0时不可当有效事务使用。

同步复位清slot_valid、phase、o_m_valid、M metadata/tag，未清slot数据和各乘积寄存器。入口断言仅在in_fire检查格式组合、MX使能、group_valid、scale=255、保留整数码及INT4高byte。TAG_W在仿真初始化检查。

### 4.7 mac_align_max16.sv：最大指数树

组合四层比较选择树，16→8→4→2→1。输入16个EXP_W位指数和16位nz掩码；输出最大指数及has_nonzero。叶节点的无效指数强制0，节点valid为两个子节点valid的OR。若只有一个有效子节点选择它；两个都有效时用`>=`优先左节点，因此相同指数保留较低索引对应值。未输出索引。全零组输出has_nonzero=0、max_exp=0。

组0实例EXP_W=9，组1为6；两组独立比较，零乘积不能抬高Emax。整数模式父级把候选nz全部置0。

### 4.8 mac_align_rne.sv：逐项对阶舍入

组合参数化模块，输入signed W位Y、9位完整距离d、window24选择；输出signed W位X、inexact和overflow。limit为24或W。当前W=32/24，不承诺W小于3等参数情况。

- d=0：X=Y，inexact=0。
- d≥limit：X=0，inexact=(Y≠0)。此路径依赖合法输入幅值窗口；window24不是允许任意W位幅值的通用模式。
- 0<d<limit：`q=Y>>>d` 得到向负无穷取整商；`lost=Y<<(W−d)` 把丢弃位搬至高端。guard=lost[W−1]，round=lost[W−2]，sticky为其余位OR。`inc=guard&&(round||sticky||q[0])`，扩为W+1位计算q+inc，再取低W位。inexact为lost非零。

使用floor商与非负低位余数使负数tie也正确：Y=3,d=1得到2；Y=5,d=1得到2；Y=−3,d=1得到−2；Y=−5,d=1得到−2。overflow在中间距离路径比较rounded的两个最高符号位，由父级断言；不是饱和功能。

完整9位距离先与limit比较，不能先截成移位器控制宽度，避免大指数差回绕成小移位。当前文件头仍有旧“W=33”注释，实际实例与声明为32，详见第8章。

### 4.9 mac_align_lane0.sv：主对阶通路

输入mode、active、signed24乘积、9位指数和/Emax、product_nz；输出signed32对阶值和诊断标志。FP19把有效signed23乘积用 `{product[22:0],9'b0}` 扩为32位Y；被去掉的product[23]是冗余符号，父级断言保证与bit22相等。d为Emax−exp_sum的9位结果，减法先扩为10位。

FP9使用 `{product[8:0],15'b0}` 形成signed24，再符号扩展至32；指数差使用扩展到7位的6位减法，RNE窗口选择24。这里没有额外再左移8位，补8零发生在MX。

整数模式Y/d清零使舍入逻辑静止，结果直接把signed24乘积扩展到32。浮点product_nz=0时结果及inexact强制0；inactive时同样输出0。负指数差由父级对合法非零项检查，本模块本身不提供负差恢复。

### 4.10 mac_align_lane1.sv：第二对阶通路

输入signed18乘积和6位指数和/Emax，输出signed24。仅FP9构造 `{product[8:0],15'b0}` 并使用独立的24位RNE实例。FP19保持0；INT16/INT8/INT4把signed18扩展为24后旁路。active由父级使用lane1_active确定，因此INT16时本lane服务group0高段，不能当group1结果。

### 4.11 mac_align_stage.sv：MAX、逐项对阶与输出槽

实例化两棵MAX树、16个lane0和16个lane1。MAX候选掩码需要组有效、浮点模式及product_nz；group1只在FP9使用指数树。输出为16对32/24位值、原txn_meta、新增align_meta和tag。

一个弹性输出槽：`space=!o_align_valid||i_align_ready`，`o_m_ready=i_rstn&&space`，`fire=i_m_valid&&o_m_ready`。space时更新valid；fire时采样metadata、Emax、has_nz和inexact的逐组OR，并只更新active lane寄存器。无效lane物理寄存器保持旧值，输出由valid/group/mode屏蔽成0。复位清valid、metadata和tag，不清lane数组。

断言检查合法mode/group、FP19 signed23冗余位、FP9 signed9冗余位、nz与实际乘积一致、非零项Emax不小于exp_sum、舍入无溢出及FP9 lane0高位符号扩展。

### 4.12 mac_psum_stage.sv：部分和归约与WIDE重组

输入ALIGN两路16项、txn_meta/align_meta/tag；输出56位PSUM和随行信息。每棵树将16个语义signed24项扩展到signed28，拼为448位输入；两个 `DW02_tree(16,28,1)` 分别产生sum/carry两行，两个 `DW01_add(28)` 完成末级进位传播。树间没有交叉进位。

**FP19**：对每个signed32 X，令L=unsigned X[22:0]、H=signed X[31:23]，恒等式 `X=(H<<23)+L`。低树累计L为unsigned27，高树累计H为signed13。重组时低树结果[26:0]全量保留，高树[12:0]符号扩展，两者均进入signed37中间量：`S37=(Hsum<<23)+Lsum`。检查bit36=bit35后取signed36，再符号扩展至56位。不能丢掉低树增长位[26:23]。

**DUAL FP9/INT8/INT4**：lane0[23:0]送树0，lane1送树1，无效组输入0。输出 `[27:0]=group0 sum`，`[55:28]=group1 sum`。INT8仅需signed19，INT4仅需signed11，高位由断言确认是冗余符号。

**INT16**：两棵树都是group0部分积。令两树结果L/H，则T=L+(H<<8)。phase0输出T，phase1输出T<<8；都使用signed37临时量，输出符号扩展至56位。phase0低树是unsigned20、高树signed20，T可装signed28；phase1两树signed20，未加外层权重的T可装signed27。外层权重只在这里施加一次。

输出采用单槽弹性握手，space=`!o_psum_valid||i_psum_ready`，输入接受条件为i_align_valid且o_align_ready。复位清输出数据、valid、两类metadata和tag；停顿时全部保持。树末级CO不是饱和控制，高位正确性依赖范围与断言。

### 4.13 mac_mx_stage.sv：解包和指数重定标

输入56位PSUM、两类metadata/tag；输出两个signed36贡献、unsigned8指数、mx_meta/tag。无树重组、无INT16二次权重、无舍入、无饱和，也不计算MXINT8最高有效位位置。

WIDE模式取PSUM[35:0]作为group0贡献，group1为0。FP9每个signed28段低端补8零。INT8取每段低19位，MX使能时补17零，否则符号扩展。INT4当前代码取每段低12位再扩为36；合法数实际只需signed11，额外位是已检查的冗余符号位。

只有MX使能且有效的FP9/INT8组消费scale。每个raw scale先正扩到signed9再减127；两者相加到signed10。其余scale调整为0。FP19/FP9指数为Emax−bias_sum+132+scale_sum；MXINT8为132+scale_sum。计算使用signed13，普通整数E保持0。

贡献为0时输出E=0；非零时取exp_s13[7:0]进入8位接口。在合法浮点输入域内E必须为1…254。仿真在接受PSUM时检查全宽exp_s13，越界终止；综合后切片仍存在，不能把物理低8位行为写成已定义的回绕策略，外部必须限制输入域。

输出是单个弹性槽，系数、指数、精简metadata和tag在psum_fire时一起写入，停顿一起保持，复位全部清零。group_nz/inexact只是转发，不作为舍入补偿。断言还检查WIDE组掩码、acc_kind、scale255和PSUM高位冗余。

### 4.14 mac_acc_float_prepare36.sv：累加对阶准备

组合模块。输入old/new系数及指数、first，输出两路signed39工作操作数和signed11共同指数。first只把旧系数视为0；不直接改变存储状态。Inf优先处理位于父模块，本模块只准备有限数运算。

1. 对signed36系数取unsigned36幅值，包含−2^35的幅值2^35，不在signed36中求正值。
2. 用逐bit扫描获取最高1的位置lead。非零幅值左移34−lead，使最高1在bit34；仅lead=35时右移1，此时只可能是−2^35，右移精确。指数同步变为 `E'=E+lead−34`。
3. 非零操作数之间取较大E'为common_e；只有一个非零时用该指数；均为0时取0。中间E'可超出8位正常编码域，不能在这里提前FTZ。
4. 规格化幅值补3个低位零，按common_e−E'调用shift_right_jam。d=0直接取值；0<d<39右移，并将所有丢弃位OR并入bit0；d≥39仅保留“原值是否非零”在bit0。
5. 对工作幅值恢复正负号，输出39位有符号操作数及common_e。

此处jam不是各操作数分别RNE。三位尾部与预规格化配合：指数差≤3时补零足以使对阶精确，包括深抵消；差≥4时小幅值小于大幅值的1/8，相减最多左规格化1位，最终仍至少丢弃2个低工作位，jam用于保留半ULP判定需要的信息。父级在相加后执行唯一一次ACC RNE。

### 4.15 mac_acc_stage.sv：累加状态与提交

输入为MX valid、两组贡献/指数、mx_meta和tag。`o_mx_ready=i_rstn`，不存在正常运行下的内部反压。`mx_fire=i_mx_valid&&o_mx_ready`。输出分两类：update_valid/group_valid/kind/tag表示刚接受并提交的事务；state_valid/kind、int32、float_sig/exp/inf_sign是持续可读状态。

**共享加法**：每组有独立加法路径，组内整数/浮点通过mux选择操作数。浮点来自prepare；整数把signed36旧值/新值各补3个低零。两路signed39先符号扩展到40相加。`adder_sum_s37=adder_sum[39:3]`是整数准确结果；浮点保留全部40位供舍入，不能先丢尾部再相加。

**整数旧值**：INT16 phase1选择int16_mid_q；其他情况first=1选择0，否则选择该组o_acc_int32并符号扩展。新值是对应MX贡献。饱和比较signed37结果与2147483647/−2147483648，越界分别输出7fffffff/80000000，范围内取低32位。

**浮点写回**：取40位和的无符号幅值；若old_inf_sign非零且first=0，直接保留旧Inf。否则和为0输出规范零。非零和最高1位置为lead，shift=lead−34，候选E=common_e+shift−3。shift>0时右移得到候选系数，丢弃位对齐到40位高端，以guard、其余位OR和保留位LSB执行RNE；shift≤0精确左移。若舍入进位使bit35置1，再右移1并E+1。最终E>254生成同符号Inf，E<1清零FTZ，其他情况恢复符号、写36位M及8位E。没有subnormal写回状态。

**INT16控制**：phase0保存未饱和signed36结果至int16_mid_q，pending置1，不更新公开ACC状态，不产生update_valid；phase1从mid继续加，INT32饱和写group0，pending清0，并将饱和值符号扩展留在mid。其他组不更新。phase0和phase1之间可插valid气泡，但不得插入其他模式或另一phase0。

**一般提交**：每个非INT16合法mx_fire，按group_valid分别更新state_valid/kind和所选数值bank；未选bank保留旧值。例如整数事务不清旧float寄存器，消费者必须查看state_kind，不能同时解释两个bank。无效组保持。update_valid每拍默认0、update_group_valid默认00，真正提交时置位；连续提交时valid可以连续为1。update_kind/tag在没有提交时保持上次值，应以update_valid限定使用。

复位清所有公开状态、tag和INT16中间状态。mode历史 `acc_mode_q` 只在非SYNTHESIS构建中存在，用于检查换模式前first；它不是综合后硬件的运行时保护。first并非异步清零，只有有效消费才改变相应状态。

## 5. 跨模块数值实例和不变量

### 5.1 FP19缩放与指数

转换后系数有10个小数位，乘积P有20个小数位。ALIGN增加9位得到29个小数位，故单项signed32为Q3.29；16项增长4位得到signed36 Q7.29。PSUM的真实值为 `S×2^(Emax−bias_sum−29)`。令ACC表示为 `M×2^(E−161)`，取M=S即要求 `E=Emax−bias_sum+132`。不能将Q7.29的字面小数位直接作为Q2.34的相同指数，差异由E补偿。

FP16 1×1：系数1024×1024，P=2^20，Emax=30，bias_sum=30。d=0时单项X=2^29；MX取M=2^29/E=132，其值仍为1。若16项均相同，S=2^33，同一E对应16；ACC再规格化为M=2^34/E=131。

### 5.2 FP9和组独立性

FP9乘积有6个小数位，ALIGN补15位得到21位小数；PSUM值为 `S×2^(Emax−bias_sum−21+scale_sum)`。MX取M=S×2^8，E=Emax−bias_sum+132+scale_sum，代回ACC公式得到相同值。

Emax=32、两源bias均7、scale_sum=0时E=150。group0/group1的Emax和scale_sum分别计算；公共bias_sum来自整笔事务的两源格式。某组所有非零项相互抵消时group_nz仍可为1，但S=0，因此MX最终输出M=0/E=0。

### 5.3 MXINT8重定标

合法单项最大幅值127×127=16129，16项最大幅值258064，分别装signed15/signed19。取M=S×2^17、E=132+scale_sum，则 `M×2^(E−161)=S×2^scale_sum`。scale raw=127代表指数调整0，不是raw=0。该阶段不按S最高1的位置调整E；ACC内部准备阶段才会精确规格化M。

### 5.4 INT16分解与饱和位置

令A=A_lo+256A_hi、B=B_lo+256B_hi，其中低byte为unsigned，高byte为signed。每个dot项的phase0形成 `A_lo B_lo +256 A_hi B_lo = A B_lo`；phase1形成 `256(A_lo B_hi+256 A_hi B_hi)=256 A B_hi`。PSUM在每个phase内先对16项归约，再应用相同权重，ACC将两phase加回旧状态。

例A=−1（A_lo=255、A_hi=−1），B=257（B_lo=1、B_hi=1）：phase0=−1，phase1=−256，总和−257。phase0不得提前INT32饱和，否则跨phase抵消可能错误；phase1不得再次左移8位。

### 5.5 ACC进位、抵消与first

两个M=2^34/E=127相加代表1+1，写回M=2^34/E=128。M相反且E相同相加得到M=0/E=0。旧状态为Inf时有限更新保持Inf；first=1时忽略旧状态，有限贡献重新形成状态。group_valid=10时group0不改变，即使输入总线上group0 first或数据发生变化。

## 6. 时序与协议

### 6.1 无停顿边沿示例

以下T0表示输入首次在该上升沿握手，表内内容是边沿后寄存器状态；复位已释放，链路初始空且各级就绪。

| 边沿 | 普通事务 | INT16事务 |
|---|---|---|
| T0 | entry slot装入 | entry slot装入，phase0 |
| T1 | M结果有效 | M phase0有效；slot转phase1 |
| T2 | ALIGN有效 | ALIGN phase0；M phase1；slot释放 |
| T3 | PSUM有效 | PSUM phase0；ALIGN phase1 |
| T4 | MX有效 | MX phase0；PSUM phase1 |
| T5 | ACC状态更新、update有效 | ACC保存phase0 mid，无update；MX phase1 |
| T6 | 后续事务继续 | ACC提交phase1、update有效 |

普通事务从T0接受到T5提交相隔5个周期；INT16到T6相隔6个周期。停顿时相应寄存器边界推迟，不能把上述数值当任意输入序列的固定延迟。

### 6.2 弹性槽协议

ALIGN/PSUM/MX均使用 `space=!out_valid||downstream_ready`。out_valid=1且ready=0时不得更新该槽的数据、metadata、tag及valid。槽内旧数据被消费的同拍，可以接受并替换为下一包。空间存在但输入valid=0时清输出valid，数值寄存器通常保持旧值；只有valid为1时才解释为事务。

entry因INT16双发射具有额外release条件，不能直接改成普通单槽模板。ACC本身始终接受，完整链路的测试反压由testbench外加的admission gate产生；这不证明生产ACC有结果反压接口。

### 6.3 复位、气泡与状态生命周期

所有时序块为 `always_ff @(posedge i_clk)` 中判断 `!i_rstn`，因此是低有效同步复位。ready组合逻辑受i_rstn门控，拉低复位会使ready立即低，但寄存器直到时钟上升沿才清除。至少覆盖一个有效时钟边沿才能完成状态复位。

entry/M和ALIGN有未复位数据数组，但valid清0，数据输出有掩码；后续合法消费前不得直接取这些内部数组。PSUM、MX和ACC的数据与控制均有显式复位赋值。复位取消未完成事务，包括INT16 phase0；复位后phase1不能继续先前事务。

首次使用、切换compute_mode或切换acc_kind前，有效组必须给first。FP16↔BF16都映射FP19，当前ACC不保存源格式；不要误写成硬件检查每种源格式切换。tag不参与状态合法性判断或bank选择。INT16 pending为全局单份，phase之间不允许其他模式更新，哪怕只更新group1。

## 7. Testbench与公司集成

### 7.1 SV文件用途

| 文件（位于 hardware/verification/tb） | DUT/组织 | 激励与主要检查 |
|---|---|---|
| tb_mac_types_pkg.sv | package | 内建检查类型宽度、映射函数和组选择 |
| tb_mac_operand_pack.sv | operand_pack及转换器 | +VEC，逐项组合输出比对 |
| tb_mac_mul_frontend.sv | mul_frontend | +VEC，模式/phase、乘积、指数、nz/active |
| tb_mac_entry_m_stage.sv | entry/M | 内建时序激励，holding slot、phase、停顿与连续发射 |
| tb_mac_entry_m_invalid.sv | entry/M | +CASE，非法输入预期触发断言 |
| tb_mac_align_units.sv | MAX/RNE单元 | +VEC，移位舍入和最大指数 |
| tb_mac_align_stage.sv | ALIGN | +VEC，可用+BAD注入非法条件；数据/metadata和停顿 |
| tb_mac_m_align.sv | entry/M→ALIGN | 内建事务与phase/停顿检查 |
| tb_mac_psum_stage.sv | PSUM | +VEC，树重组/打包、握手、复位 |
| tb_mac_align_psum.sv | ALIGN→PSUM | +VEC，真实相邻stage数值与保持 |
| tb_mac_mx_stage.sv | PSUM→MX | +VEC，两级结果、指数、反压及复位 |
| tb_mac_acc_stage.sv | ACC | +VEC，逐笔状态/更新检查；+BAD检查非法ACC调用 |
| tb_mac_q234_pipeline.sv | 完整链路 | +VEC，原始事务、提交、tag、phase、停顿及在途复位 |
| dwbb_sim_models.sv | DW02_tree / DW01_add替身 | 功能仿真依赖，不是testbench顶层，不得进入综合列表 |

向量文件的期望值来自外部文件，SV本身不是独立完整数值oracle。本文仅描述SV消费协议，不规定向量生成器。公司内部若采用新验证平台，可保留相同字段和检查语义，替换激励来源。

### 7.2 ACC与全链路向量接口

`tb_mac_acc_stage` 每行25个十六进制字段，顺序如下：

```text
mode kind group_valid first phase
contribution0 contribution1 exp0 exp1 tag0 tag1
expected_update_valid expected_update_group_valid expected_update_kind
expected_state_valid expected_state_kind0 expected_state_kind1
expected_int0 expected_int1
expected_float_sig0 expected_float_exp0 expected_float_sig1 expected_float_exp1
expected_inf0 expected_inf1
```

每行实际上连续写在一行，以空白分隔。负数使用对应宽度的补码位模式。BAD=1在首向量移除first；2/3修改exp0为0/255；4注入没有phase0的INT16 phase1。向量必须配合所需的非法前提，例如指数测试需要非零浮点贡献。每10笔插入valid气泡，检查无重复提交及状态保持。

`tb_mac_q234_pipeline` 文件首行是十进制 `fmt_a fmt_b mx_enable transaction_count`。每笔事务先给15个十六进制字段：group_valid、first、tag0、tag1、scale_a0、scale_a1、scale_b0、scale_b1、expected_kind、group0 data/E/inf、group1 data/E/inf；再跟16行raw_a/raw_b。data根据kind解释为整数或浮点系数；只对有效组解释。

全链路驱动在输入ready前保持当前事务；MX→ACC处测试门同时限制向ACC的valid和向MX的ready，避免一侧认为接受而另一侧没有接受。检查总接受数、MX消费数（INT16为输入数两倍）、最终提交数和停顿确实发生；复位测试额外阻塞在途事务后复位，检查无旧事务泄漏。该门不属于产品RTL。

### 7.3 编译顺序和依赖

建议顺序：package；FP19/FP9 convert；operand_pack；mul_frontend；entry；align_max16/align_rne/align_lane0/align_lane1；align_stage；DesignWare库模型；psum；mx；acc_float_prepare36；acc_stage；所选testbench。所有stage实例保持一致TAG_W。

当前PSUM实例化的是 `DW02_tree` 和 `DW01_add`。本地 `dwbb_sim_models.sv` 用串行3:2压缩实现功能替身，不能据此推断实际供应商IP的拓扑、面积或延迟。公司环境应接入授权DesignWare实现及匹配工具配置，避免真实库与同名替身重复编译。

以下只是可复现的基本类型包检查入口，本文编写过程未把它当作完整功能验证：

```sh
iverilog -g2012 -s tb_mac_types_pkg -o /tmp/mac_types_pkg.vvp \
  hardware/rtl/systemverilog/mac_types_pkg.sv hardware/verification/tb/tb_mac_types_pkg.sv
vvp /tmp/mac_types_pkg.vvp
```

公司顶层需要连接上述stage的valid/ready、所有数值数组及metadata/tag，不能遗漏align_meta中的Emax信息或txn_meta中的bias_sum。ACC结果消费者需能接受连续update；如需输出反压，必须另外设计缓存及其满条件与前级控制，不能只给update_valid外接ready后丢弃脉冲。

## 8. 限制与规范化注意事项

1. **指数合法域**：MX非零贡献的数学E超出1…254仍是暂缓定义域。断言会终止仿真，综合代码未实现饱和、预规格化或错误返回。该限制也可能影响FP19极端输入，不能宣称所有源格式有限编码组合都可完整通过现有MX接口。
2. **ACC范围策略**：已实现写回FTZ/Inf；输入转换的NaN/Inf策略与其不同；没有NaN累加状态和Inf输入端口。
3. **接口边界**：当前没有产品顶层、目标输出编码转换、结果ready、CSR、CDC或完整时序约束交付。
4. **综合与物理结果**：本文为代码分析，没有本轮综合、STA、面积、功耗或新仿真通过率结论。组合扫描、移位器、MAX及ACC反馈路径应由公司工具实际评估。
5. **参数化**：DOT_N和位宽并非全面自由可配；存在literal16/32/36及固定切片。不能只改package常量就认为完成参数化。
6. **仿真保护与硬件保护**：`ifndef SYNTHESIS`内检查、mode历史和参数断言在综合配置下不存在。上游合法性约束必须在集成接口保留。
7. **已发现注释差异**：`mac_align_rne.sv`头注释仍写W=33，实际lane0实例为32；MX/ACC部分INT4诊断文字写signed12，实际范围断言要求signed11冗余。本文使用代码表达式和实际连接解释，这些注释后续规范化时应同步修正，本次文档不修改RTL。
8. **规范待提供**：没有公司具体代码规范文本，本文不认定always_comb/always @*选择、命名、声明位置、复位覆盖等写法是否合规。后续应按公司规则审查，而不是在文档中假定合规。

规范化重构时重点保持：INT16无符号低byte；FP19低树增长位；各切片前的符号冗余检查；FP9与MXINT8固定补零；独立组Emax/scale；ACC先保留tail后一次RNE；phase0不饱和；同拍消费/补入的优先级；无效组保持；first仅在实际消费时生效。任何增加流水级或ready缓冲的改动都必须重新检查延迟、phase和metadata配对。

## 附录A. 全部模块端口声明

以下直接摘录本版源文件的模块声明，保留精确类型、方向、数组维度和参数，不引入二次手工转录的位宽差异。`i_*`为输入、`o_*`为输出，语义以前文模块节为准。所有stage的时钟/复位含义相同；组合模块没有隐藏时钟。未指定signed的logic向量为无符号；package类型的signed属性见第3章。

### A.mac_acc_float_prepare36

源文件：[mac_acc_float_prepare36.sv](../../hardware/rtl/systemverilog/mac_acc_float_prepare36.sv)。

```systemverilog
module mac_acc_float_prepare36 (
  input logic i_first,
  input mac_types_pkg::acc_sig_t i_old_sig,
  input mac_types_pkg::acc_exp_t i_old_exp,
  input mac_types_pkg::acc_sig_t i_new_sig,
  input mac_types_pkg::acc_exp_t i_new_exp,
  output logic signed [38:0] o_old_operand,
  output logic signed [38:0] o_new_operand,
  output logic signed [10:0] o_window_exp
);
```

### A.mac_acc_stage

源文件：[mac_acc_stage.sv](../../hardware/rtl/systemverilog/mac_acc_stage.sv)。

```systemverilog
module mac_acc_stage #(parameter integer TAG_W) (
  input logic i_clk, i_rstn, i_mx_valid,
  output logic o_mx_ready,
  input mac_types_pkg::contribution0_t i_mx_contribution0,
  input mac_types_pkg::contribution1_t i_mx_contribution1,
  input mac_types_pkg::acc_exp_t i_mx_exp0, i_mx_exp1,
  input mac_types_pkg::mx_meta_t i_mx_meta,
  input logic [TAG_W-1:0] i_mx_tag [0:1],
  output logic o_acc_update_valid,
  output logic [1:0] o_acc_update_group_valid,
  output mac_types_pkg::acc_kind_t o_acc_update_kind,
  output logic [TAG_W-1:0] o_acc_update_tag [0:1],
  output logic [1:0] o_acc_state_valid,
  output mac_types_pkg::acc_kind_t o_acc_state_kind [0:1],
  output logic signed [31:0] o_acc_int32 [0:1],
  output mac_types_pkg::acc_sig_t o_acc_float_sig [0:1],
  output mac_types_pkg::acc_exp_t o_acc_float_exp [0:1],
  output logic signed [1:0] o_acc_float_inf_sign [0:1]
);
```

### A.mac_align_lane0

源文件：[mac_align_lane0.sv](../../hardware/rtl/systemverilog/mac_align_lane0.sv)。

```systemverilog
module mac_align_lane0 (
  input  mac_types_pkg::compute_mode_t i_mode,
  input  logic i_active,
  input  mac_types_pkg::product12_t i_product,
  input  logic [8:0] i_exp_sum,
  input  logic i_product_nz,
  input  logic [8:0] i_emax,
  output mac_types_pkg::align_lane0_t o_aligned,
  output logic o_inexact,
  output logic o_round_overflow
);
```

### A.mac_align_lane1

源文件：[mac_align_lane1.sv](../../hardware/rtl/systemverilog/mac_align_lane1.sv)。

```systemverilog
module mac_align_lane1 (
  input  mac_types_pkg::compute_mode_t i_mode,
  input  logic i_active,
  input  mac_types_pkg::product9_t i_product,
  input  logic [5:0] i_exp_sum,
  input  logic i_product_nz,
  input  logic [5:0] i_emax,
  output mac_types_pkg::align_lane1_t o_aligned,
  output logic o_inexact,
  output logic o_round_overflow
);
```

### A.mac_align_max16

源文件：[mac_align_max16.sv](../../hardware/rtl/systemverilog/mac_align_max16.sv)。

```systemverilog
module mac_align_max16 #(
  parameter integer EXP_W
) (
  input  logic [EXP_W-1:0] i_exp [0:15],
  input  logic [15:0] i_nz,
  output logic [EXP_W-1:0] o_max_exp,
  output logic o_has_nonzero
);
```

### A.mac_align_rne

源文件：[mac_align_rne.sv](../../hardware/rtl/systemverilog/mac_align_rne.sv)。

```systemverilog
module mac_align_rne #(
  parameter integer W
) (
  input  logic signed [W-1:0] i_y,
  input  logic [8:0] i_distance,
  input  logic i_window24,
  output logic signed [W-1:0] o_x,
  output logic o_inexact,
  output logic o_overflow
);
```

### A.mac_align_stage

源文件：[mac_align_stage.sv](../../hardware/rtl/systemverilog/mac_align_stage.sv)。

```systemverilog
module mac_align_stage #(
  parameter integer TAG_W
) (
  input  logic i_clk,
  input  logic i_rstn,
  input  logic i_m_valid,
  output logic o_m_ready,
  input  mac_types_pkg::product12_t i_m_product0 [0:15],
  input  mac_types_pkg::product9_t i_m_product1 [0:15],
  input  logic [8:0] i_m_exp_sum0 [0:15],
  input  logic [5:0] i_m_exp_sum1 [0:15],
  input  logic [1:0] i_m_product_nz [0:15],
  input  mac_types_pkg::txn_meta_t i_m_meta,
  input  logic [TAG_W-1:0] i_m_tag [0:1],

  output logic o_align_valid,
  input  logic i_align_ready,
  output mac_types_pkg::align_lane0_t o_align_lane0 [0:15],
  output mac_types_pkg::align_lane1_t o_align_lane1 [0:15],
  output mac_types_pkg::txn_meta_t o_align_meta,
  output mac_types_pkg::align_meta_t o_align_ameta,
  output logic [TAG_W-1:0] o_align_tag [0:1]
);
```

### A.mac_entry_m_stage

源文件：[mac_entry_m_stage.sv](../../hardware/rtl/systemverilog/mac_entry_m_stage.sv)。

```systemverilog
module mac_entry_m_stage #(
  parameter integer TAG_W
) (
  input  logic i_clk,
  input  logic i_rstn,
  input  logic i_in_valid,
  output logic o_in_ready,
  input  logic [15:0] i_raw_a [0:mac_types_pkg::DOT_N-1],
  input  logic [15:0] i_raw_b [0:mac_types_pkg::DOT_N-1],
  input  mac_types_pkg::src_fmt_t i_src_fmt_a,
  input  mac_types_pkg::src_fmt_t i_src_fmt_b,
  input  logic [1:0] i_group_valid,
  input  logic [1:0] i_acc_first,
  input  logic [TAG_W-1:0] i_tag [0:1],
  input  logic i_mx_enable,
  input  logic [7:0] i_scale_a [0:1],
  input  logic [7:0] i_scale_b [0:1],

  output logic o_m_valid,
  input  logic i_m_ready,
  output mac_types_pkg::product12_t o_m_product0 [0:mac_types_pkg::DOT_N-1],
  output mac_types_pkg::product9_t o_m_product1 [0:mac_types_pkg::DOT_N-1],
  output logic [8:0] o_m_exp_sum0 [0:mac_types_pkg::DOT_N-1],
  output logic [5:0] o_m_exp_sum1 [0:mac_types_pkg::DOT_N-1],
  output logic [1:0] o_m_product_nz [0:mac_types_pkg::DOT_N-1],
  output mac_types_pkg::txn_meta_t o_m_meta,
  output logic [TAG_W-1:0] o_m_tag [0:1]
);
```

### A.mac_fp19_convert

源文件：[mac_fp19_convert.sv](../../hardware/rtl/systemverilog/mac_fp19_convert.sv)。

```systemverilog
module mac_fp19_convert (
  input  logic [15:0] i_raw,
  input  mac_types_pkg::src_fmt_t i_src_fmt,
  output logic [7:0] o_exp_eff,
  output logic signed [11:0] o_sig_tc
);
```

### A.mac_fp9_convert

源文件：[mac_fp9_convert.sv](../../hardware/rtl/systemverilog/mac_fp9_convert.sv)。

```systemverilog
module mac_fp9_convert (
  input  logic [7:0] i_raw,
  input  mac_types_pkg::src_fmt_t i_src_fmt,
  output logic [4:0] o_exp_eff,
  output logic signed [4:0] o_sig_tc
);
```

### A.mac_mul_frontend

源文件：[mac_mul_frontend.sv](../../hardware/rtl/systemverilog/mac_mul_frontend.sv)。

```systemverilog
module mac_mul_frontend (
  input  mac_types_pkg::operand_t i_operand_a,
  input  mac_types_pkg::operand_t i_operand_b,
  input  mac_types_pkg::compute_mode_t i_mode,
  input  logic [1:0] i_group_valid,
  input  logic i_int16_phase,
  output mac_types_pkg::product12_t o_product0,
  output mac_types_pkg::product9_t o_product1,
  output logic [8:0] o_exp_sum0,
  output logic [5:0] o_exp_sum1,
  output logic [1:0] o_product_nz,
  output logic o_product0_active,
  output logic o_product1_active
);
```

### A.mac_mx_stage

源文件：[mac_mx_stage.sv](../../hardware/rtl/systemverilog/mac_mx_stage.sv)。

```systemverilog
module mac_mx_stage #(
  parameter integer TAG_W
) (
  input  logic i_clk,
  input  logic i_rstn,
  input  logic i_psum_valid,
  output logic o_psum_ready,
  input  mac_types_pkg::psum_packed_t i_psum_data,
  input  mac_types_pkg::txn_meta_t i_psum_meta,
  input  mac_types_pkg::align_meta_t i_psum_ameta,
  input  logic [TAG_W-1:0] i_psum_tag [0:1],

  output logic o_mx_valid,
  input  logic i_mx_ready,
  output mac_types_pkg::contribution0_t o_mx_contribution0,
  output mac_types_pkg::contribution1_t o_mx_contribution1,
  output mac_types_pkg::acc_exp_t o_mx_exp0,
  output mac_types_pkg::acc_exp_t o_mx_exp1,
  output mac_types_pkg::mx_meta_t o_mx_meta,
  output logic [TAG_W-1:0] o_mx_tag [0:1]
);
```

### A.mac_operand_pack

源文件：[mac_operand_pack.sv](../../hardware/rtl/systemverilog/mac_operand_pack.sv)。

```systemverilog
module mac_operand_pack (
  input  logic [15:0] i_raw,
  input  mac_types_pkg::src_fmt_t i_src_fmt,
  input  logic [1:0] i_group_valid,
  output mac_types_pkg::operand_t o_payload
);
```

### A.mac_psum_stage

源文件：[mac_psum_stage.sv](../../hardware/rtl/systemverilog/mac_psum_stage.sv)。

```systemverilog
module mac_psum_stage #(
  parameter integer TAG_W
) (
  input  logic i_clk,
  input  logic i_rstn,
  input  logic i_align_valid,
  output logic o_align_ready,
  input  mac_types_pkg::align_lane0_t i_align_lane0 [0:15],
  input  mac_types_pkg::align_lane1_t i_align_lane1 [0:15],
  input  mac_types_pkg::txn_meta_t i_align_meta,
  input  mac_types_pkg::align_meta_t i_align_ameta,
  input  logic [TAG_W-1:0] i_align_tag [0:1],

  output logic o_psum_valid,
  input  logic i_psum_ready,
  output mac_types_pkg::psum_packed_t o_psum_data,
  output mac_types_pkg::txn_meta_t o_psum_meta,
  output mac_types_pkg::align_meta_t o_psum_ameta,
  output logic [TAG_W-1:0] o_psum_tag [0:1]
);
```

## 附录B. 仿真断言索引

以下逐条摘录RTL中的 `$fatal` 消息，用于公司环境定位现有检查，不代表对所有非法调用都有完整防护。行号对应附录C的代码快照；正文说明其主要触发条件。检查一般由输入fire限定，参数检查位于initial；全部处于非SYNTHESIS分支。

| 文件:行 | 诊断消息 |
|---|---|
| mac_acc_stage.sv:195 | `TAG_W must be explicit and positive` |
| mac_acc_stage.sv:200 | `ACC requires at least one valid group` |
| mac_acc_stage.sv:204 | `ACC received an illegal mode` |
| mac_acc_stage.sv:207 | `floating mode requires TC_FLOAT` |
| mac_acc_stage.sv:210 | `integer mode requires INT32` |
| mac_acc_stage.sv:214 | `TC_FLOAT is not defined for this mode` |
| mac_acc_stage.sv:217 | `wide ACC mode requires group_valid=01` |
| mac_acc_stage.sv:221 | `INT8 group0 contribution must be signed19` |
| mac_acc_stage.sv:223 | `INT8 group1 contribution must be signed19` |
| mac_acc_stage.sv:227 | `INT4 group0 contribution must be signed12` |
| mac_acc_stage.sv:229 | `INT4 group1 contribution must be signed12` |
| mac_acc_stage.sv:234 | `INT16 phase0 arrived before previous phase1` |
| mac_acc_stage.sv:236 | `INT16 phase1 arrived without phase0` |
| mac_acc_stage.sv:238 | `INT16 phase1 must not reapply acc_first` |
| mac_acc_stage.sv:242 | `INT16 phase0 requires acc_first for an invalid/noninteger state` |
| mac_acc_stage.sv:245 | `non-INT16 update interleaved between INT16 phases` |
| mac_acc_stage.sv:250 | `ACC state requires acc_first before mode/type change or initial use` |
| mac_acc_stage.sv:257 | `integer group0 sum exceeds signed36 work range` |
| mac_acc_stage.sv:260 | `integer group1 sum exceeds signed36 work range` |
| mac_acc_stage.sv:267 | `ACC nonzero input exponent boundary deferred` |
| mac_align_stage.sv:121 | `TAG_W must be explicit and positive` |
| mac_align_stage.sv:129 | `invalid ALIGN mode or group_valid` |
| mac_align_stage.sv:136 | `FP19 ALIGN input redundancy/nonzero/distance` |
| mac_align_stage.sv:144 | `FP9 lane0 ALIGN input redundancy/nonzero/distance` |
| mac_align_stage.sv:150 | `FP9 lane1 ALIGN input redundancy/nonzero/distance` |
| mac_align_stage.sv:153 | `FP9 lane0 ALIGN result sign redundancy` |
| mac_entry_m_stage.sv:177 | `TAG_W must be explicit and positive` |
| mac_entry_m_stage.sv:182 | `illegal source-format combination or MX enable` |
| mac_entry_m_stage.sv:185 | `invalid group_valid` |
| mac_entry_m_stage.sv:189 | `E8M0 255 has no defined valid-transaction policy` |
| mac_entry_m_stage.sv:195 | `reserved INT16 input` |
| mac_entry_m_stage.sv:201 | `reserved INT8 input` |
| mac_entry_m_stage.sv:205 | `INT4 raw upper byte must be zero` |
| mac_entry_m_stage.sv:210 | `reserved INT4 input` |
| mac_mx_stage.sv:172 | `TAG_W must be explicit and positive` |
| mac_mx_stage.sv:180 | `invalid MX mode or group_valid` |
| mac_mx_stage.sv:183 | `WIDE MX requires group_valid=01` |
| mac_mx_stage.sv:187 | `MX scale is only defined for FP9 and INT8` |
| mac_mx_stage.sv:191 | `MX acc_kind does not match mode and scale enable` |
| mac_mx_stage.sv:195 | `FP19 PSUM sign extension` |
| mac_mx_stage.sv:199 | `INT16 PSUM signed36 redundancy` |
| mac_mx_stage.sv:204 | `INT8 group0 PSUM signed19 redundancy` |
| mac_mx_stage.sv:207 | `INT8 group1 PSUM signed19 redundancy` |
| mac_mx_stage.sv:213 | `INT4 group0 PSUM signed12 redundancy` |
| mac_mx_stage.sv:216 | `INT4 group1 PSUM signed12 redundancy` |
| mac_mx_stage.sv:224 | `MX active group requires finite E8M0 scales` |
| mac_mx_stage.sv:230 | `MX input exponent boundary deferred group0` |
| mac_mx_stage.sv:232 | `MX input exponent boundary deferred group1` |
| mac_psum_stage.sv:219 | `TAG_W must be explicit and positive` |
| mac_psum_stage.sv:227 | `invalid PSUM mode or group_valid` |
| mac_psum_stage.sv:230 | `WIDE PSUM requires group_valid=01` |
| mac_psum_stage.sv:236 | `PSUM lane0 signed24 input redundancy` |
| mac_psum_stage.sv:240 | `INT8 PSUM lane0 signed15 input redundancy` |
| mac_psum_stage.sv:243 | `INT8 PSUM lane1 signed15 input redundancy` |
| mac_psum_stage.sv:247 | `INT4 PSUM lane0 signed7 input redundancy` |
| mac_psum_stage.sv:250 | `INT4 PSUM lane1 signed7 input redundancy` |
| mac_psum_stage.sv:255 | `INT16 phase0 low-limb input must be unsigned16` |
| mac_psum_stage.sv:257 | `INT16 high-limb input signed16 redundancy` |
| mac_psum_stage.sv:260 | `INT16 phase1 low-limb input signed16 redundancy` |
| mac_psum_stage.sv:267 | `FP19 low tree is not unsigned27` |
| mac_psum_stage.sv:269 | `FP19 high tree signed13 redundancy` |
| mac_psum_stage.sv:271 | `FP19 reconstructed signed36 overflow` |
| mac_psum_stage.sv:276 | `INT16 phase0 low tree unsigned20 overflow` |
| mac_psum_stage.sv:278 | `INT16 phase1 low tree signed20 redundancy` |
| mac_psum_stage.sv:281 | `INT16 high tree signed20 redundancy` |
| mac_psum_stage.sv:284 | `INT16 phase0 result signed28 redundancy` |
| mac_psum_stage.sv:287 | `INT16 phase1 result signed27 redundancy` |
| mac_psum_stage.sv:292 | `INT8 PSUM signed19 tree overflow` |
| mac_psum_stage.sv:297 | `INT4 PSUM signed11 tree overflow` |

## 附录C. 源码快照与文档检查

以下SHA-256用于定位本版解释的实际代码。本文编写只新增设计文档，没有修改RTL、testbench或数值模型。源码变化后应更新正文及本表。testbench只作结构和接口说明，未在本轮重新运行功能回归；历史通过记录不作为本版的新验证证据。

| RTL文件 | 行数 | SHA-256 |
|---|---:|---|
| [mac_acc_float_prepare36.sv](../../hardware/rtl/systemverilog/mac_acc_float_prepare36.sv) | 72 | `4c099ff03e2e99603724eef4477f4a4d1367c19c57bf1f732d4318a81785ef43` |
| [mac_acc_stage.sv](../../hardware/rtl/systemverilog/mac_acc_stage.sv) | 272 | `4ddee3d1a26e2cee9f6ef567a5a4c663b67ccee82bea15f09d728dd1a80affb5` |
| [mac_align_lane0.sv](../../hardware/rtl/systemverilog/mac_align_lane0.sv) | 79 | `11e07d314fa575b462a5cd8cedeac41f433c618c72dc5cc6cba2c5f4da70aafd` |
| [mac_align_lane1.sv](../../hardware/rtl/systemverilog/mac_align_lane1.sv) | 60 | `988627c4b9acb0c25328a2e6e385352706ac15da75085fe33cc49c3a3cce5d9f` |
| [mac_align_max16.sv](../../hardware/rtl/systemverilog/mac_align_max16.sv) | 35 | `e09713b7b02933e7ce693adaa7f5c39fd94cbf54000b6131f3b4ce8c589c52fd` |
| [mac_align_rne.sv](../../hardware/rtl/systemverilog/mac_align_rne.sv) | 66 | `dbf10ac44a87493c5ad102e37b93065b6e7595cc8d7e67ac9b8ba18083275f9a` |
| [mac_align_stage.sv](../../hardware/rtl/systemverilog/mac_align_stage.sv) | 159 | `6479cf1e8b6bfa0126bbfcf6f7286c5f0f95b32eba7e2ea35b3e99b9d0f44a92` |
| [mac_entry_m_stage.sv](../../hardware/rtl/systemverilog/mac_entry_m_stage.sv) | 218 | `06326f06713ac69c8ab4bb0d2972be47c9c4bb600fc96448e36a55069e32e526` |
| [mac_fp19_convert.sv](../../hardware/rtl/systemverilog/mac_fp19_convert.sv) | 90 | `ea74a53da59594b47dcc26b7f9c4e1b58d9b816762d834f9b43a949f4e4ea105` |
| [mac_fp9_convert.sv](../../hardware/rtl/systemverilog/mac_fp9_convert.sv) | 86 | `9e877b04a2180a764eda8cd55894d54a7521379b7ae62780644127cf39ca282d` |
| [mac_mul_frontend.sv](../../hardware/rtl/systemverilog/mac_mul_frontend.sv) | 145 | `ee22abdaf40616afa5d4c8851243bf8d3e772eaa1ed3dabdbcaf2bd516b82887` |
| [mac_mx_stage.sv](../../hardware/rtl/systemverilog/mac_mx_stage.sv) | 237 | `8cc057a0220c49a9cf3411ff13b8f4d111d9ec8ddc32ebf2e5af5d6643b05ae9` |
| [mac_operand_pack.sv](../../hardware/rtl/systemverilog/mac_operand_pack.sv) | 80 | `4a75037036bca8a8ea63851483a2997442cd6e7344210bd949bbeb6114441c08` |
| [mac_psum_stage.sv](../../hardware/rtl/systemverilog/mac_psum_stage.sv) | 304 | `b49d6e05c5fbd2353a3af5acc922504ad6a51d114105cdccb171c1cf987c1b1c` |
| [mac_types_pkg.sv](../../hardware/rtl/systemverilog/mac_types_pkg.sv) | 180 | `90e9a576d32aa7bdef2d0be6c641155082df8b332f432acc678820732f316886` |

初版审查重点：整体集成边界是否符合公司交接方式；各模块算法、位宽与状态更新描述是否充分；是否需要按公司模板调整端口表、时序图或章节编号；暂缓域与后续开发边界是否清楚。公司规范化修改应作为后续代码变更单独验证，不能用本文替代lint或综合检查。
