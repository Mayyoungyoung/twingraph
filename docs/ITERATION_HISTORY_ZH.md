# TwinGraph 迭代记录

仓库保留各阶段的报告与当前源码。2026-09-24 清理了旧版原始采集、录像和历史价值模型文件；报告中的历史结论仍按原实验边界解读。

| 版本 | 主要变化与实际结论 | 报告 |
|---|---|---|
| V2–V5 | 完整候选程序表示、图谱输入及早期价值排序实验；其局部或早期系统标签不能与当前完整功能验收标签混用。 | [V2](../NEXT_ROUND_REPORT.md)、[V3](../VALUE_GRAPH_REPORT.md)、[V4](../VALUE_V4_REPORT.md)、[V5](../VALUE_V5_REPORT.md) |
| V6–V14 | 装配与功能流程、候选覆盖、视觉和排序机制的迭代诊断。 | [V6](evidence/value_v6/README.md)、[V9](evidence/value_v9_functional_assembly/REPORT_ZH.md) |
| V15–V17 | 完整任务候选池与最终验收；严格运行时下的自然候选多数全负。 | [V15](evidence/value_v15_full_system/REPORT_ZH.md)、[V16](evidence/value_v16_natural_coverage/README.md) |
| V18 | 通用原子技能计划进入物理执行；挡板局部任务有自然混合池，不能当作完整装配正例。 | [V18](evidence/value_v18_atomic_flow/README_ZH.md) |
| V19 | 各阶段自然覆盖和控制器瓶颈诊断；局部成功不进入整任务价值标签。 | [V19](evidence/value_v19_stage_matrix/README_ZH.md) |
| V20 | 固定随机布局、完整任务标签、预定数据折与同池对照协议。首批 20 个布局只有 1 个自然混合池；第二批依用户要求中止。当前没有符合门槛的完整任务价值检查点或 Top-4 对照。 | [V20](evidence/value_v20_full_flow/README_ZH.md) |

归档清理只影响当前文件树。旧文件仍可从 Git 历史找到；此整理没有重写远端提交历史。运行时必需的装配控制器权重保留在 `simbench/assembly/checkpoints/`，历史价值模型和原始样本不再随当前版本发布。
