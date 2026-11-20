# 编译清单

从项目根目录使用文件清单，文件中的路径均相对项目根目录。

- `systemverilog.f`：15 个已验证生产源文件，类型包先编译。
- `verilog.f`：旧基线的 4 个未完成改写文件。
- [`new_arch_convert.f`](new_arch_convert.f) 包含独立输入转换器生产源码，使用 `iverilog -g2001` 编译。共仿入口见[硬件验证说明](../verification/README.md#新架构独立输入转换器)。
- [`new_arch_mac.f`](new_arch_mac.f) 包含完整 MAC 核心，使用 `iverilog -g2001` 编译。`new_arch_mac_core` 是完整顶层，`new_arch_mac_forward` 可作为前向子模块单独编译。执行方法见[核心共仿](../verification/README.md#新架构-mac-核心与-acc)。

各清单独立编译。旧 SystemVerilog 基线的 [DesignWare 仿真替身](../verification/legacy_sv/tb/dwbb_sim_models.sv)按测试需要单独加入。真实综合使用对应厂商库。

例如，编译并运行类型包测试：

```sh
iverilog -g2012 -s tb_mac_types_pkg -o /tmp/mac_types.vvp \
  -f hardware/filelists/systemverilog.f \
  hardware/verification/legacy_sv/tb/tb_mac_types_pkg.sv
vvp /tmp/mac_types.vvp
```
