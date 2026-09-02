# Task A 两类原子动作与故障注入设计

> 2026-09-01。范围：仅 Task A（gearbox）。B/C 任务的计划与执行路径保持原状。

## 1. 原子动作的两类划分

执行器（`simbench/executor.py`）用两张分派表组织原子动作，步骤结果
记录 `kind` 标记；`TaskPlan` 格式不变，B/C 计划无需任何修改。

### 1.1 规划类动作（`_PLAN_SKILLS`，kind="plan"）

只做计算、不驱动物理仿真，产出"计划工件"写入
`executor.state["plans"]`，可失败（失败原因记录在故障日志）：

| 动作 | 输入 | 产出工件 | 失败条件 |
|------|------|----------|----------|
| `plan_grasp_pose` | `detect_part` 的观测 | `plans/grasp`：抓取位姿 {pos, yaw, outer_d, grasp_dz, approach_yaw, confidence} | 漏检/估计失败（重试 1 次） |
| `plan_path` | 语义起终点 + style | `plans/<as>`：waypoints + valid + style | 线段-AABB 碰撞命中（真实几何检查）、随机不可达/预测碰撞（换 clear 风格重规划 1 次） |

### 1.2 执行类动作（`_SKILLS`，kind="exec"）

驱动物理仿真并返回结果，消耗对应工件：

| 动作 | 说明 | 消耗工件 |
|------|------|----------|
| `detect_part` | 零件检测（噪声/漏检/误检）→ `state["percepts"]` | — |
| `move_to` | 按计划航点执行移动（末端加移动残差） | path 计划 |
| `grasp` | 按估计位姿抓取（无工件时退回旧的检测+估计内联行为） | grasp 计划 |
| `transport` | 持件搬运（低速 carry，滑脱注入点，搬运后校验仍持件） | path 计划 |
| `place` / `insert` | 放置 / thread 装配（配方与旧基线一致） | — |
| `inspect` | 质量检测（传感器噪声 + spec 判定 → verdict） | — |
| `nudge` | 条件纠偏 push（状态标志已 OK 时跳过） | — |
| `scatter_parts` | 来料散布（seed 化初始位姿 jitter） | — |
| 原有 `settle`/`home`/`release`/`push` 等 | 归入执行类，语义不变 | — |

每阶段循环：
`detect_part → plan_grasp_pose → plan_path → move_to → grasp →
plan_path(carry) → transport → place/insert → settle`，
S2 后加中段 `inspect`（shaft 对中，超差触发 `nudge` 纠偏），
链尾加 S9 终检 `inspect`（retainer/cover 就位 + pin 啮合）。

## 2. 成功/失败模拟机制（`simbench/faults.py`）

`FailureModel(profile, seed)`：单一随机源（`np.random.default_rng(seed)`），
同 (profile, seed) 完全可复现。档位：`none | mild | default | strong`。

| 失败源 | 默认档参数 | 注入点与机制 |
|--------|-----------|--------------|
| 抓取位姿估计误差 | noise_std=1.5mm（yaw×10） | `detect_part` 高斯噪声 → 偏心抓取 → 真实物理后果 |
| 检测漏检 | p=0.04 | `detect_part` 返回未找到（重试 1 次） |
| 检测误检 | p=0.03 | 检测位姿叠加 2-6mm 随机偏移（错误特征锁定） |
| 路径规划不可达 | p=0.02 | `plan_path` 判定不可达（重规划 1 次） |
| 路径规划碰撞 | p=0.03 | `plan_path`：预测碰撞 + 真实线段-AABB 检查（tray 壁/sleeve） |
| 移动到位偏差 | std=0.8mm | `move_to` 末端航点叠加残差 |
| 抓取滑脱 | p=0.04 | `transport` 中放松手指咬合 → 零件真实滑出（物理掉落，搬运后校验） |
| 初始散布 | std=1mm | `scatter_parts` 对零件初始位姿做 seed 化 jitter |

所有事件写入 `faults.log`（源 + 详情），随运行数据落盘
`fault_log.json`，运行结束打印按源计数摘要，失败可归因。
`fail_mode="continue"` 保留：前段失败级联影响后续阶段（future-value
研究目标）。`--noise` 旧旋钮保留并映射为噪声覆盖值。

## 3. 运行方式

```bash
conda activate turbovla-libero
cd /home/jia/RAL

# 确定性基线（无故障注入）：8 装配阶段 + S9 终检，9/9
MUJOCO_GL=egl python -m simbench.run_task --task A --fault-profile none

# 默认故障档：多 seed 呈现成功/失败混合（seed 决定随机量，可复现）
MUJOCO_GL=egl python -m simbench.run_task --task A --seed 0   # 或 1,2,3...

# 强故障档（演示失败路径）
MUJOCO_GL=egl python -m simbench.run_task --task A --fault-profile strong --seed 3

# 录屏（子任务标注）
MUJOCO_GL=egl python -m simbench.run_task --task A --record results/simbench/taskA_faults_demo.mp4
```

数据落盘：`results/simbench/data/<task>_s<scene>_seed<seed>_noise<noise>_<ts>/`
（plan.json / steps.json / stages.json / trace.npz / meta.json / fault_log.json）。

## 4. 已验证的物理基线注意事项

- 抓取下降 tol 保持旧 GRASP_TOL=4mm（仅 gear/cover 沿用 1.5mm 旧值）：
  1.5mm 深降 + transport 额外搬运腿会让重件在释放时被 pads 带翻
  （实测 shaft 释放后 17° 倾斜，连锁打翻全链）。
- 环类零件（gear/spacer/bearing/retainer）摩擦改为 0.6（原默认 1.0），
  使滑脱失败物理可见；`none` 档回归验证 9/9 无回归。
