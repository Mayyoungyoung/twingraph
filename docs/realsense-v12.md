# 白色打印件与 RealSense 接口

用户已确认零件全白，并进一步说明实际使用腕部双目深度相机、尚未手眼标定；
主仿真实验按腕部 RGB-D 实现。材料尚未确认。新版白色场景及
CAD 深度几何检测使用同一输入接口；颜色不能充当零件类别标签。
真实腕部相机尚未接入本次运行，不能把理想仿真深度精度称为实拍精度。

腕部相机的点云变换为 `T_base_optical(t) = T_base_hand(q(t)) @ T_hand_optical`。
仿真使用声明的相机安装外参与机器人编码器 FK；相机采集时刻与关节状态必须匹配。
外置录像镜头仅用于展示。下列 RealSense 工具是通用对齐帧适配器，其单次采集
`world_from_optical` 参数不应被误用为移动腕相机在所有时刻恒定的世界外参。
实机部署时仍须测量安装外参并同步每帧的机器人状态；这不是当前仿真实验的阻碍。

`simbench/value/realsense_v12.py` 接收真实对齐 RGB-D 和外部标定：

- 通过设备 SDK 获取实际深度比例，转换为米，不假设所有整数都是毫米。
- 深度对齐 RGB，再依 RGB 内参处理畸变；深度重采样保留空洞，避免在边缘
  插值出不存在的深度。普通 Brown 使用 OpenCV；modified/inverse Brown
  使用 RealSense SDK 的投影建立整流映射，避免混用畸变公式。其他未支持
  的畸变模型会明确拒绝，需先在 SDK 中整流。
- RealSense optical 为 x 向右、y 向下、z 向前；检测器原有相机系为
  x 向右、y 向上、z 向后。显式转换 `diag(1,-1,-1)`，并检查刚体外参。
- `world_from_optical` 必须来自实际相机到机器人/桌面世界坐标的标定，
  不能填入仿真相机的位姿，也不默认单位阵。

这些采集操作依据官方 [深度对齐示例](https://github.com/realsenseai/librealsense/blob/master/wrappers/python/examples/align-depth2color.py)。
坐标与畸变约定依据官方 [投影说明](https://github.com/realsenseai/librealsense/wiki/Projection-in-RealSense-SDK-2.0)。
本轮对坐标轴、深度比例、无效深度及外参校验做了软件测试，未连接真实设备。

在已安装 `pyrealsense2` 且连接相机的机器中，从仓库根目录运行：

```bash
python scripts/capture_realsense_v12.py --world-from-optical-json measured_camera_extrinsics.json --out results/realsense_capture --infer
```

外参文件格式为 `{"world_from_optical": [[...], [...], [...], [...]]}`；
矩阵表示米制光学坐标到世界坐标的变换。此处不提供虚构的数值标定。
可加 `--serial` 选择设备，或 `--bag` 使用实际录制的 RealSense 文件。

输出包含对齐帧、内外参、原始深度比例与时间信息，后续可离线重放：

```bash
python scripts/capture_realsense_v12.py --replay results/realsense_capture --out results/realsense_replay --infer
```

检测失败应保留未知状态，触发换视角/重新观测；实际深度空洞、反光、
遮挡和相机外参误差，需要真实录制样本进一步验证，不能用仿真真值补齐。
