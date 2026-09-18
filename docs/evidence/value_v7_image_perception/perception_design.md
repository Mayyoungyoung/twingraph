# RGB-D 几何感知设计

正式 backend 名称为 `rgbd_geometry`。每个决策边界从 MuJoCo 相机渲染一组同步
RGB 和米制深度，再下采样同一时刻的图像给价值模块（价值输入仍为 80×80）；
检测器使用 640×480、相机内参、标定外参和静态 CAD 外观/尺寸模板。

`simbench/value/rgbd_perception.py::estimate_scene` 是纯函数，签名只接收
`rgbd_frames, camera_calibration, object_templates, previous_estimates`，不接收
session、MjData 或任何物体真值。它完成：桌面/背景深度分离、容差颜色和连通区域
提取、有效深度反投影、区域点云 PCA 轴向估计、CAD 参考点偏移、工作区几何约束和
质量/残差输出。实例由当前图像和宽工作区先验匹配；不是像素查表或 body/geom ID
分割。不可见或深度无效返回 `valid=false`，不隐式回退到 oracle。

当前模板是闭集模拟感知代理，颜色和尺寸来自固定 CAD 外观，适度容忍光照变化；
它不是训练好的开放世界 RGB-D 网络，也不声称完成真实相机检测。遮挡严重、物体
离开工作区或颜色变化超出模板时会受控失败或要求重新观测。`stage_v7.refresh_visual_observation`
在抓取前、零件放置后、挡块确定孔位前、把手滑动前再次采图。

检测器输出包含类别、视觉 track id、mask/框面积、位置、任务相关轴向、质量、
来源视角、时间戳/观测年龄和 valid/occluded 状态。检测坐标到抓取参考点的
`reference_offset_m` 是静态 CAD 变换，不是当前场景位姿。

离线审计脚本 `scripts/evaluate_v7_rgbd_perception.py` 在没有把真值传入检测器的
情况下运行；真值只由独立评估器事后读取来计算误差。
