# 11 个原子技能：独立近景视频

本目录的 11 段视频于 2026-09-08 重新录制。固定正面偏侧近景，1600 × 1000、25 fps；底部显示当前原子技能、标记含义与实时结果。路径等空间线条加粗约 2.8 倍。没有网页前端或镜头切换。

| 原子技能 | 视频 | 时长 | 观察重点 |
|---|---|---:|---|
| 检测 | [detect](atoms/detect.mp4) | 6.00 秒 | 三个对象的半透明检测球 |
| 物体位姿估计 | [estimate_pose](atoms/estimate_pose.mp4) | 6.00 秒 | 物体位置与 RGB 坐标轴 |
| 抓取位姿估计 | [estimate_grasp](atoms/estimate_grasp.mp4) | 6.00 秒 | 两种真实抓法及接近方向 |
| 路径规划 | [plan_path](atoms/plan_path.mp4) | 6.00 秒 | 三条保留的候选路径，加粗显示 |
| 移动 | [move](atoms/move.mp4) | 9.84 秒 | 沿规划路径绕过障碍物，叠加实际轨迹 |
| 抓取 | [grasp](atoms/grasp.mp4) | 6.48 秒 | 方块两侧闭爪，显示真实接触点 |
| 放置 | [place](atoms/place.mp4) | 4.48 秒 | 在支撑面释放方块，检查释放后状态 |
| 插入 | [insert](atoms/insert.mp4) | 18.24 秒 | 销轴进入通孔，旁侧显示实际推进量 |
| 压靠 | [press](atoms/press.mp4) | 6.00 秒 | 小位移压靠与实时接触力条 |
| 测量 | [measure](atoms/measure.mp4) | 6.00 秒 | 实际导轨间隙、引线和放大示意 |
| 检查 | [inspect](atoms/inspect.mp4) | 6.00 秒 | 插销装配位置误差与倾角验收 |

计算类技能展示真实输出，展示时间不推进物理仿真。检测后端目前是仿真位姿观测；插入使用已有行为克隆策略。准备动作通过原控制器完成，在该原子技能开始前结束。抓取和放置分别展示闭爪与支撑位置释放；接近、上下移和搬运统一属于移动。

[运行和重录](../setup.md) · [原子技能定义](../atomic-skills.md) · [验证与抽帧检查](../evidence/atomic-closeup/README.md)

此前的 [完整装配](assembly.mp4)、[连续方块抓取示例](cube_grasp.mp4) 与 [兼容组件回归录像](COMPONENTS.md) 单独保留，不计为新的原子技能。
