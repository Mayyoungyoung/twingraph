# 候选多样性审计

v7 每场景固定 12 个候选，候选生成只使用冻结观测、固定工艺先后关系和几何/IK 约束。候选的可执行差异包括：

- 合法装配顺序（carriage → end_stop → 两销 → handle；两销可交换）；
- 每个零件的抓取 yaw、抓取高度、抓取力、clearance 和速度；
- 擦拭方向/遍历顺序、目标力和持续时间；
- 把手安装后功能测试的独立双向行程验收。

候选 ID、哈希、状态摘要和视频字幕不作为模型输入。`stage_v7.build_pool()` 对 `(order, choices, wipe_variant, wipe_force, wipe_duration, stroke_minimum)` 做语义去重，并输出 `structure_branches`、`grasp_branches`、`wipe_path_branches` 和 `executable_semantic_unique`。目前开发 smoke 已产生不同顺序、抓取分支和擦拭分支；正式汇总待完整采集后写入本文件的附表。

禁止通过 rollout 结果删候选或隐藏成功保底方案；几何拒绝需记录在 counts 中。
