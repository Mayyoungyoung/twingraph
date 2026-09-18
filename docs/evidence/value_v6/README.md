# TwinGraph v6 价值模块正式证据

本目录是远程正式实验的可提交证据包。完整原始数据（约 716 MB）和系统状态轨迹（约 3.1 GB）保留在远程主机 `/home/jia/twingraph-v6-20260917/results/v6/formal`，未复制进 Git；`audit.json`、`evaluation.json`、`system_batch.json` 和 `system_case_summary.json` 提供逐组哈希与汇总审计。

正式数据审计：train/val/test = 64/16/16，96 个场景，1,152 条候选程序，3,456 次完整物理执行，0 个未分类失败。3 个初始 240 秒截尾尝试原样保留在远程 `censored/`，同种子恢复采集使用 600 秒上限；最终 96 组均通过 source/input/graph/paired-trial 校验。

评估按验证集配置均值 Brier 选择模型和视角，阈值按验证集 hard-forecast Brier（经验成功率目标）冻结，测试集不参与选择。最终模型为 `both_43`（双视角 RGB-D），测试准确率 64.58%、平衡准确率 64.83%、ROC-AUC 0.7048、Brier 0.1894、Top-4 命中 75.00%。

系统实验包含 71400–71411 共 12 个配置；Top-4 与全量 12 条策略分别实际完成 30 次目标调用（另有 2 个配置因没有满足双策略共同接受规则而 abstain），成功率分母固定为 36 次目标试验。Top-4 平均决策/端到端耗时 674.53/835.90 秒，目标成功率 69.44%；全量策略为 1678.99/1819.34 秒、75.00%。完整中文报告和图见 [`report/report.md`](report/report.md) 与 [`report/overview.png`](report/overview.png)。

评估更正和系统适配层修复记录见 [`../../value-v6-evaluation-amendment.md`](../../value-v6-evaluation-amendment.md)。修复只改动 `scripts/` 评估/批处理适配，不改动 `simbench/value`、`simbench/assembly` 或 `simbench/core` 的冻结采集源码。

实现、输入输出、采集协议和 Top-4 最终选择规则见 [`value_module_summary_zh.md`](value_module_summary_zh.md)。71400 案例的 12 个候选视频、4 个 Top 视频和实际 target 执行视频均为独立 MP4，见 [`videos/separate/`](videos/separate/)。
