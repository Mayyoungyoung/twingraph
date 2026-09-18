# 插销成功标准：有效插入

本版本不再把销的世界坐标和倾角当作主成功条件。主谓词是
`pin_inserted_in_hole`，在 `simbench/value/pin_geometry.py` 中实现，独立于
RGB-D 检测器。

## 几何定义

v7 导向环的 CAD 参数为：导向内半径 5.5 mm、销轴半径 3.3 mm、导向长度
8 mm。有效插入要求销的轴段与孔轴共同穿过圆柱形有效孔道，并且沿孔轴的
有效嵌入深度至少 6 mm；同时要有足够的径向间隙，不能只是碰孔口或搭在孔边。
孔入口由视觉估计的挡块位姿加已知 CAD 孔偏移得到，而不是固定世界坐标。

判定按几何采样处理轻微倾斜，不强求名义中心误差或 3° 姿态阈值。销在孔旁
但高度较低、只碰孔口、被孔边托住或已经掉出孔均失败。夹爪仍接触时只记录
`inserted_while_held`，松爪并短暂 settle 后才记录主指标
`inserted_after_release`；滑动后另记录 `retained_after_stroke`。

## 链路

`stage_v5.py` 的 v7 pin 分支使用 `acceptance="pin_inserted"` 和
`inspect_pin_inserted`；`goal_check.py` 使用同一谓词；独立评估器
`evaluate_pin_context` 只在验收边界读取 MuJoCo 几何，不进入感知或价值输入。
普通零件的位姿、支撑、碰撞和抓取失败条件保持不变。

## 已验证与未完成

`tests/test_v7_image_perception.py` 覆盖：正常插入、允许倾斜、孔边/低位假插入、
松爪后掉出以及滑动保持的阶段区分。最新的真实 MuJoCo 开发轨迹
(`scene=1200,candidate=08fafc1b0668beb1f7c5`) 中，两枚销在松爪后均通过该几何谓词：
左销有效深度约 7.91 mm、最大径向偏差约 0.82 mm，右销有效深度约 7.90 mm、
最大径向偏差约 0.34 mm，并在往复滑动后的独立检查中保持。此前只在孔口产生约 8 N
接触并丢失夹持的轨迹仍按执行失败保留，没有改写标签。后续记录以
`execution_results.csv` 和原始 `result.json` 为准。
