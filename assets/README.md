# 图稿与附件

| 位置 | 用途 |
|---|---|
| [主电路图](diagrams/source/各模块电路图.drawio) | 可编辑的多页 draw.io 源文件 |
| [整体架构描述](diagrams/source/mac_overall_architecture.json)及[样式](diagrams/source/mac_overall_architecture.style.json) | 整体架构图的源描述 |
| [整体架构预览](diagrams/exports/mac_overall_architecture.html) | 现有 HTML 预览；PNG/SVG 与之同目录 |
| `diagrams/exports/` | 根目录原有六张命名导出图和整体架构导出图 |
| `diagrams/generated/` | 分页 draw.io、SVG/PNG/PDF、历史视觉检查与 receipt |
| `figures/` | 量化粒度示意图和生成提示词 |
| `attachments/` | 项目总结引用的附件 |

图稿在本次整理中仅迁移，没有重绘。现有图稿可能与当前代码有差异，例如 ACC 输出保持槽与最终 ready 协议，论文引用前应复核。

[图稿构建脚本](../scripts/diagrams/build_module_circuits.py)会更新主源图并保存绘制前备份；[导出脚本](../scripts/diagrams/export_module_circuits.py)用于导出分页图与检查。导出依赖本机 draw.io、Node 和 draw.io 工具脚本，具体路径可通过环境变量配置。历史 receipt 中的绝对路径保留为原生成记录，不作为当前运行路径。
