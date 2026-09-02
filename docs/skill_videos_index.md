# 原子技能演示视频索引（Skill Videos Index）

> 与 `docs/skill_inventory.md` 一一对应；由 `simbench/record_skills.py` 生成。视频文件位于本目录（results/，不入库）。

| 技能 | 类别 | 粒度 | 视频路径 | 演示内容概述 | 录制结果 |
|---|---|---|---|---|---|
| `detect` | exec | 原子 | results/skill_videos/exec_detect.mp4 | 感知零件位姿（单次判定） | 已录制 |
| `inspect` | exec | 原子 | results/skill_videos/exec_inspect.mp4 | 落座质量判定 OK/NG（NG 紧容差 + OK 规格容差两例） | 已录制 |
| `move` | exec | 原子 | results/skill_videos/exec_move.mp4 | EEF 航点移动到目标点 | 已录制 |
| `descend` | exec | 原子 | results/skill_videos/exec_descend.mp4 | 保持 xy 垂直下降到目标 z | 已录制 |
| `grip_open` | exec | 原子 | results/skill_videos/exec_grip_open.mp4 | 张开夹爪到全开 | 已录制 |
| `grip_close` | exec | 原子 | results/skill_videos/exec_grip_close.mp4 | 闭合夹爪咬合零件（含举升效果验证） | 已录制 |
| `release` | exec | 原子 | results/skill_videos/exec_release.mp4 | 张开夹爪原地释放持件 | 已录制 |
| `lift_verify` | exec | 原子 | results/skill_videos/exec_lift_verify.mp4 | 举升校验：判零件被抬升达标（纯判定） | 已录制 |
| `push` | exec | 原子 | results/skill_videos/exec_push.mp4 | 并指刃状推动零件 5cm | 已录制 |
| `transport` | exec | 组合 | results/skill_videos/exec_transport.mp4 | 持件沿航点搬运（组合，视频演示其效果） | 已录制 |
| `plan_grasp_pose` | plan | 原子 | results/skill_videos/plan_plan_grasp_pose.mp4 | 多候选抓取位姿生成/评分/选优（EEF 走向所选位姿可视化） | 已录制 |
| `plan_path` | plan | 原子 | results/skill_videos/plan_plan_path.mp4 | 多候选路径规划 + 碰撞门（EEF 沿所选航点行进可视化） | 已录制 |
| `retreat_lift` | trans | 原子 | results/skill_videos/trans_retreat_lift.mp4 | 垂直抬升退避 0.12m | 已录制 |
| `pre_align` | trans | 原子 | results/skill_videos/trans_pre_align.mp4 | 持件对中到目标 xy（3cm 偏置收敛） | 已录制 |
| `pull` | ext | 原子 | results/skill_videos/ext_pull.mp4 | 抓持后沿方向拖拽并释放 | 已录制 |
| `wipe` | ext | 原子 | results/skill_videos/ext_wipe.mp4 | 并指压表+力带扫掠擦拭 | 已录制 |
