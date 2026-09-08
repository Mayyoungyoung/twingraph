# 兼容组件录像（不作为平级原子技能计数）

这些是原有控制和可视化的回归示例，任务层清单见 [11 项技能](../atomic-skills.md)。

# 底层组件独立演示

30 个底层组件，分别使用方块、障碍物、单孔插销座和短导轨。每段为固定 MuJoCo 视角，底部显示技能名称与标记说明。

[小方块完整抓取示例](cube_grasp.mp4) · [实现与重录说明](../standalone-skills.md)

规划、检测与验收技能展示计算结果，不额外驱动机械臂。青线为计划，黄线为实测运动；绿色通常为目标或指垫接触，橙色为外部接触。检测目标与坐标轴采用各自的颜色标记，以视频底部说明为准。

| 技能 | 独立场景 | 视频 | 时长 |
|---|---|---|---|
| 检测零件 | 三个彩色方块 | [observe_parts](skills/observe_parts.mp4) | 4.8 秒 |
| 估计零件位姿 | 三个彩色方块 | [estimate_pose](skills/estimate_pose.mp4) | 4.8 秒 |
| 生成抓取候选 | 三个彩色方块 | [propose_grasps](skills/propose_grasps.mp4) | 4.8 秒 |
| 选择可达抓取 | 三个彩色方块 | [select_grasp](skills/select_grasp.mp4) | 4.8 秒 |
| 规划避让搬运路径 | 方块与被动障碍物 | [plan_transfer](skills/plan_transfer.mp4) | 4.8 秒 |
| 规划直线操作路径 | 单个 40 mm 方块 | [plan_linear](skills/plan_linear.mp4) | 4.8 秒 |
| 执行关节搬运路径 | 方块与被动障碍物 | [execute_joint_path](skills/execute_joint_path.mp4) | 9.1 秒 |
| 执行直线操作路径 | 单个 40 mm 方块 | [execute_cartesian_path](skills/execute_cartesian_path.mp4) | 12.3 秒 |
| 张开夹爪 / 释放 | 单个 40 mm 方块 | [open_gripper](skills/open_gripper.mp4) | 3.5 秒 |
| 接近抓取位姿 | 单个 40 mm 方块 | [approach](skills/approach.mp4) | 8.0 秒 |
| 接触反馈夹持 | 单个 40 mm 方块 | [close_gripper](skills/close_gripper.mp4) | 5.8 秒 |
| 验证夹持接触 | 单个 40 mm 方块 | [verify_grasp](skills/verify_grasp.mp4) | 4.8 秒 |
| 保持夹持并抬升 | 单个 40 mm 方块 | [lift](skills/lift.mp4) | 7.7 秒 |
| 受控下移至预接触高度 | 单个 40 mm 方块 | [lower](skills/lower.mp4) | 8.4 秒 |
| 空夹爪退出操作区 | 单个 40 mm 方块 | [retreat](skills/retreat.mp4) | 5.2 秒 |
| 机械臂回位 | 单个 40 mm 方块 | [home](skills/home.mp4) | 5.5 秒 |
| 调整末端朝向 | 单个 40 mm 方块 | [orient_wrist](skills/orient_wrist.mp4) | 6.7 秒 |
| 持物轴线精对准 | 单个插销与孔座 | [align_axis](skills/align_axis.mp4) | 3.6 秒 |
| 规划接触插入参数 | 单个插销与孔座 | [plan_insertion](skills/plan_insertion.mp4) | 4.8 秒 |
| 沿导轨约束插入 | 单个滑块与短导轨 | [slide_insert](skills/slide_insert.mp4) | 16.9 秒 |
| 接触保护下降 | 单个插销与孔座 | [guarded_descent](skills/guarded_descent.mp4) | 6.4 秒 |
| 肩面压靠到位 | 单个插销与孔座 | [press_seat](skills/press_seat.mp4) | 5.3 秒 |
| 检查零件装配位置 | 单个插销与孔座 | [inspect_seat](skills/inspect_seat.mp4) | 4.8 秒 |
| 沿已装导轨运动 | 单个滑块与短导轨 | [move_constrained](skills/move_constrained.mp4) | 6.1 秒 |
| 验证产品往复行程 | 单个滑块与短导轨 | [verify_stroke](skills/verify_stroke.mp4) | 4.8 秒 |
| 测量导轨剩余间隙 | 单个滑块与短导轨 | [measure_clearance](skills/measure_clearance.mp4) | 4.8 秒 |
| 规划有预算的接触恢复 | 单个插销与孔座 | [plan_recovery](skills/plan_recovery.mp4) | 4.8 秒 |
| 从接触中撤回 | 单个插销与孔座 | [retract_contact](skills/retract_contact.mp4) | 4.3 秒 |
| 接触引导螺旋寻孔 | 单个插销与孔座 | [spiral_search](skills/spiral_search.mp4) | 8.2 秒 |
| 学习策略插销 | 单个插销与孔座 | [learned_insert](skills/learned_insert.mp4) | 17.5 秒 |

小方块完整抓取示例串联接近、闭爪和抬升三个底层组件，不计为新的底层组件。

[30 个技能缩略图](contact_sheet.jpg) · [文件与执行检查记录](../evidence/demo-audit.json)

这些视频解释现有实现组件，不代表 30 个平级任务技能。任务级划分见 [架构设计](../architecture.md)。
