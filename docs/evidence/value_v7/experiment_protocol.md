# value-v7 实验协议

## 冻结任务

任务范围：`clean_assemble_five_parts_and_post_handle_bidirectional_stroke`。每场景 12 个候选，每候选重复执行完整任务；成功标签为：

`cleaning_pass AND assembly_pass AND functional_test_pass AND final_release_and_retraction_pass`。

清洁阈值固定为残留污迹比例 ≤5%，并在工具移开后验收。滑动测试要求正、反两个方向均达到最小行程 0.08 m，横向偏移 <6 mm，峰值接触力 ≤14 N。

## 数据划分与扰动

- 训练/验证覆盖 L0/L1；L2 只留作测试。
- 默认正式目标 train=128、val=32、test=64，候选重复数为 3/3/5；资源不足时先执行开发冒烟并标记 pending，不填充未运行结果。
- L0 供料平面 ±12 mm、偏航 ±7°；L1 ±25 mm/±15°；L2 ±40 mm/±25°。
- 观测位置和偏航噪声、摩擦、质量、阻尼、执行器增益均写入 request/trial/observation manifest。
- twin 与 target 使用不同 trial namespace；target 不复用 twin 终态或结果。

## 视角与模型

四种输入共享场景、观测、候选、标签、训练预算和 80×80 RGB-D：none（无图像）、task、top、both。每种视角使用种子 17、29、43；只按验证 Brier 选 checkpoint 和阈值，held-out test 不调参。

## 系统基线

Random-1、Value-Top1、Random-Top4+Twin、Heuristic-Top4+Twin、Value-Top4+Twin（四视角）和 Full-12+Twin。接受规则固定为 twin 3 次中至少 2 次成功，abstain 计入全场景失败率；target 评价使用独立重复，不是连续重试。
