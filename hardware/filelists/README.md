# 编译清单

从项目根目录使用文件清单，文件中的路径均相对项目根目录。

- `systemverilog.f`：15 个已验证生产源文件，类型包先编译。
- `verilog.f`：4 个正在推进的 Verilog 改写文件；尚不能完整编译或替代基线。

两套清单不可混编。生产清单不包含 DesignWare 仿真替身；Icarus 验证时单独加入 `hardware/verification/tb/dwbb_sim_models.sv`，真实综合时应使用对应厂商库。

例如，编译并运行类型包测试：

```sh
iverilog -g2012 -s tb_mac_types_pkg -o /tmp/mac_types.vvp \
  -f hardware/filelists/systemverilog.f \
  hardware/verification/tb/tb_mac_types_pkg.sv
vvp /tmp/mac_types.vvp
```
