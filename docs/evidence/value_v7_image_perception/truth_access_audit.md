# 真值访问审计

|位置|当前状态|用途|
|---|---|---|
|`rgbd_perception.py`|已隔离|纯 RGB-D、标定、CAD 模板；无 session/MjData/qpos/xpos/xquat 参数或导入|
|MuJoCo renderer|保留|只负责由物理状态生成 RGB-D；相机外参/内参用于渲染标定|
|`stage_v7.install_visual`|已接入|在决策边界生成冻结观测，并保存 detector/value 分辨率、标定版本和哈希|
|`observe_execution_pose`|正式路径为视觉|`rgbd_geometry` 时调用重新渲染和检测；旧的直接 pose 分支只作为显式 legacy 诊断|
|`full_task_v7._clean/_update_execution_targets/_stroke`|已改为视觉定位|工具、挡块/孔位、把手分别在执行节点重定位|
|`pin_geometry.evaluate_pin_context`|仅独立验收|读取实际接触、轴、孔几何判定插入；不能作为感知输入或价值特征|
|controller 的 `align_axis/stream_part/press`|仍有仿真反馈依赖|使用末端运动学、接触力、抓取接触和物理对象反馈完成仿真闭环；这些是模拟触觉/物理反馈，不等同真机传感器|
|独立最终评估器|保留真值访问|只在试验结束或验收边界计算阶段标签，不反哺在线控制|

仍未解决的边界是：当前仿真接触反馈和渲染本身天然访问物理引擎状态；真机等价
传感器尚未实现。控制器中的物理反馈调用没有被伪装成“全视觉”。正式视觉核心
不会在检测失败时切换到 oracle；会返回 invalid 并停止/重新观测。
