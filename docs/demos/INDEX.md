# 当前视频：完整任务与 10 个原子技能

## 完整场景

| 视频 | 内容 | 时长 |
|---|---|---:|
| [成功](full_success.mp4) | 擦拭底座 → 滑台装配 → 行程与终态验收 | 236.48 秒 |
| [失败](full_failure.mp4) | 同一场景，右侧插销目标偏移 8 mm，真实落座检查失败后停止 | 176.28 秒 |

1920 × 1200、25 fps，固定正面略向下镜头，完整过程明确标注 2× 播放。没有切镜头。失败是标注清楚的偏差测试，不是伪造失败画面或修改验收阈值。

## 原子技能独立视频

| 技能 | 观看重点 |
|---|---|
| [检测 / detect](atoms/detect.mp4) | 半透明目标球 |
| [物体位姿估计 / estimate_pose](atoms/estimate_pose.mp4) | 物体坐标轴 |
| [抓取位姿估计 / estimate_grasp](atoms/estimate_grasp.mp4) | 两种抓取候选 |
| [路径规划 / plan_path](atoms/plan_path.mp4) | 多条规划路线 |
| [移动 / move](atoms/move.mp4) | 避障移动与实际轨迹 |
| [抓取 / grasp](atoms/grasp.mp4) | 真实双侧夹持 |
| [放置 / place](atoms/place.mp4) | 支撑面释放 |
| [插入 / insert](atoms/insert.mp4) | 插销入孔近景 |
| [压靠 / press](atoms/press.mp4) | 接触压靠 |
| [擦拭 / wipe](atoms/wipe.mp4) | 学习轨迹、真实表面接触与覆盖 |

擦拭为本轮新增的 1920 × 1200、1× 近景演示（18.80 秒）。其余九段是已验证且接口未改变的独立近景演示。检测后端为仿真位姿观测，插入近景使用旧行为克隆策略；完整场景选用已有接触反馈插销控制。

测量与检查已移出原子技能列表；[测量示例](feedback/measure.mp4)、[验收示例](feedback/inspect.mp4) 仅作为辅助工具资料保留。

[技能图谱](../skill-graph/README.md) · [当前验证](../evidence/surface-assembly/README.md) · [重录说明](../setup.md) · [兼容组件](COMPONENTS.md)
