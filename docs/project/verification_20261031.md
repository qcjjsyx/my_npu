# 目录迁移验证记录

日期：2026-10-31，Asia/Shanghai。使用新建的根目录 Python 3.13 虚拟环境，已完成可编辑安装；旧环境保留在归档。

| 检查 | 结果 |
|---|---|
| 全部测试 | 442 passed in 53.12s；硬件验证 134、实验框架 308 |
| 完整打包 TXT | 345,168 条通过 |
| MAC TXT | 9,385 步，连续与停顿两种调度均通过；停顿模式 6,739 个入口受阻周期 |
| 非法输入 | 缺少 first、scale255、非零 MX 指数越界、缺少 phase0 均命中指定断言 |
| 生产文件清单 | 类型包 TB 使用 `systemverilog.f` 编译、运行通过 |
| CLI | `python -m mac_experiment` 与已安装的 `mac-experiment version` 均可运行 |
| 实验到图表 | dot_product、mxfp8、mx_alignment、widthselect 四类小规模执行、分析、PNG 绘图均通过 |
| 默认输出 | 四类原始 run 和三类独立分析目录符合新的 `results/numerical/<实验族>/` 布局 |
| 额外离线分析 | dot_product 的 enhance、bitmatch 默认输出进入各自 analysis 子目录 |
| 图稿工具 | 导出脚本帮助入口通过；此次未重绘或覆盖原图 |

四类实验与本地链接的[机器报告](../../results/verification/check_20261031_183148_650797/report.json)保存了实际输出目录。测试完成输出和 TXT 结果分别记录在[回归摘要](../../results/verification/regression_20261031.log)、[TXT 日志](../../results/verification/txt_20261031.log)。这些结果文件属于本地产物，可重新生成。

迁移完整性由[逐文件清单](migration_manifest.json)和统一检查入口核对：生产 RTL、活动 Verilog、硬件参考模型、SV testbench、TXT 数据、PDF、历史结果和压缩包保持原始内容；路径适配与文档更新单独登记。外部仓库少量 Git 缓存及 IDE 元数据变化另行记录，未执行暂存、检出或提交。

此记录仅证明目录迁移后的功能与运行路径，没有新增综合、PPA、真实 DesignWare 库或 Verilog 改写的完成证据。新生成的 `migration_check_` run 是迁移检查，不能替代缺失的历史正式实验数据。
