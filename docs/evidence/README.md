# 验证记录

- current-tests.txt：清理后的 16 项测试，在全新导出的源码目录中运行。
- clean-checkout.json：独立检出运行测试、完整 120 步装配和固定镜头录制检查；没有复制被忽略的 results。
- assembly.json：当前精选完整视频对应的验收记录。
- assembly-robustness.json：历史 8 次小扰动整链回归，包含参与过调试的种子。
- insertion-training.json：24 回合行为克隆训练报告，包含当时较短执行预算的开发评估；不是最终几何的评估成绩。
- insertion-evaluation.json：最终几何、1000 步预算、种子 991235 的 12 个新起点评估。
- demo-audit.json：独立演示文件哈希、可视化和首/中/末帧检查。
- cube-grasp.json：小方块三组件抓取例子的真实执行指标。

历史 JSON 中的 source_directory/checkpoint 字段用于记录当时来源，可能指向已清理的中间目录。当前模型位于 simbench/assembly/checkpoints，成品视频位于 docs/demos。实验数据不等于现实机器人性能或终态价值模块成绩。
