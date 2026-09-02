# LIBERO Phenomenon Verification Report

## Symbolic Equivalence ≠ Future Success: Evidence from LIBERO Benchmark

**Date**: 2026-08-21  
**Experiment**: `experiment_demo_replay.py` (v3)  
**Environment**: LIBERO_10 + Franka Panda + MuJoCo 2.3.2

---

## 一、研究目标

验证核心现象：

> **"当前任务成功" ≠ "当前任务完成后的状态对未来任务同样友好"**

即：对于同一个已完成子任务，不同但都满足 symbolic goal 的中间终态，会导致不同的后续任务成功率（future success spread）。
一个 subtask 即使满足当前阶段的成功条件，也可能落入一个对未来任务不友好的 terminal state；不同的规划、抓取和执行过程会产生不同的合法 terminal states，而这些 terminal states 对后续任务的可完成性具有不同的 Future Value
---

## 二、实验设计

### 2.1 任务选择

**Primary Task**: LIBERO_10 Task 4 — `LIVING_ROOM_SCENE5`
- Language: "put the white mug on the left plate and put the yellow and white mug on the right plate"
- 子目标 1 (SG1): `porcelain_mug_1` on `plate_1` (left plate)
- 子目标 2 (SG2): `white_yellow_mug_1` on `plate_2` (right plate)
- Demo: 275 steps, SG1@116, SG2@225

### 2.2 关键发现：仿真非确定性与 State Save/Restore

在实验过程中发现了一个关键技术问题：

**HDF5 中保存的 state 不能直接用于 suffix replay**。原因：
- 从 HDF5 state[0] 开始 replay prefix，得到的实际仿真状态与 HDF5 中保存的 state[116] 不一致
- 最大位置差异可达 1.66 单位（经过 116 步后）
- 直接使用 HDF5 state[116] 进行 suffix replay 会导致 SG2 失败

**解决方案**：采用 "replay prefix → save state → restore → replay suffix" 流程
1. 从 state[0] replay prefix 到 sg1_step
2. 保存当前仿真状态 (`baseline_prefix_state`)
3. 从该状态 replay suffix → SG2 = True ✓

### 2.3 State 结构

84-dim state array: `[qpos(44), qvel(39), time(1)]`

**Free joint qpos format**: `[qw, x, y, z, qx, qy, qz]` (7 DOF)

| Component | Indices | Description |
|-----------|---------|-------------|
| Robot | qpos[0:9] | 7 joints + 2 gripper |
| porcelain_mug_1 | qpos[9:16] | [qw,x,y,z,qx,qy,qz] |
| red_coffee_mug_1 | qpos[16:23] | |
| white_yellow_mug_1 | qpos[23:30] | |
| plate_1 | qpos[30:37] | |
| plate_2 | qpos[37:44] | |
| qvel | qvel[44:83] | Velocity per DOF |
| time | state[83] | Simulation time |

**Critical bug fixed**: 初始代码误将 `state[9]`（qw，四元数实部）当作 x 坐标修改。正确索引为 `state[10]=x`, `state[11]=y`, `state[12]=z`。

### 2.4 候选终态生成

**方法**: Direct qpos perturbation + physics settle
1. 从 baseline_prefix_state 出发
2. 直接修改 mug1 的 qpos（x, y 坐标）
3. 让物理仿真 settle 50 步
4. 验证 SG1 谓词（`on`）
5. 保存完整仿真状态

**参数**:
- 候选数量: 20
- 扰动半径: 0.03（均匀分布在圆盘内）
- 物理 settle: 50 步

**关键优势**: 所有候选状态直接从仿真中获取，SG1 验证保证通过（20/20 通过）。

### 2.5 Noise Ablation 设计

对 suffix actions 添加高斯噪声：

| Level | noise_std (position) | noise_std (rotation) |
|-------|---------------------|---------------------|
| No noise | 0.0 | 0.0 |
| Low | 0.005 | 0.0025 |
| Medium | 0.01 | 0.005 |
| High | 0.02 | 0.01 |

---

## 三、实验结果

### 3.1 Task 4: Noise Ablation Results

**配置**: 20 candidates × 10 rollouts per noise level

| Noise Level | Spread | Mean | Std | Min | Max | Baseline SG2 |
|-------------|--------|------|-----|-----|-----|-------------|
| No noise (0.0) | **1.0000** | 0.2150 | 0.3978 | 0.0 | 1.0 | ✅ True |
| Low (0.005) | **1.0000** | 0.2100 | 0.3885 | 0.0 | 1.0 | ✅ True |
| Medium (0.01) | **1.0000** | 0.1950 | 0.3721 | 0.0 | 1.0 | ✅ True |
| High (0.02) | **1.0000** | 0.2100 | 0.3885 | 0.0 | 1.0 | ✅ True |

### 3.2 噪声对成功候选的影响

C1（最接近 baseline 的候选）在不同噪声下的成功率：

| Noise | C1 Success Rate |
|-------|----------------|
| 0.0 | 1.00 (10/10) |
| 0.005 | 0.90 (9/10) |
| 0.01 | 0.70 (7/10) |
| 0.02 | 0.90 (9/10) |

**观察**: 噪声确实产生了非二元的成功率（0.70-0.90），但波动方向不完全单调（High noise 反弹到 0.90）。

### 3.3 候选终态分布

只有 ~4/20（21.5%）的候选终态能够成功完成后续任务。其余候选虽然满足 SG1（mug 在 plate 上），但 demo replay 的 suffix 动作无法完成 SG2（pick mug2 放到 plate2）。

**原因分析**: 
- Demo replay 是开环控制（open-loop），动作序列对初始状态高度敏感
- 改变 mug1 位置后，机器人的预定轨迹可能与 mug1 发生碰撞，或物理状态偏离过大
- 这说明：**即使当前子目标满足，不同终态对后续任务的"兼容性"差异巨大**

---

## 四、关键发现

### ✅ Phenomenon CONFIRMED

1. **Spread = 1.0**: 在所有噪声级别下，不同的 SG1-满足状态导致 SG2 成功率从 0% 到 100% 的巨大差异
2. **SG1 100% 验证通过**: 所有 20 个候选终态都满足 "mug on plate" 的 symbolic goal
3. **SG2 差异巨大**: 仅 ~21.5% 的候选终态能成功完成后续任务

### ⚠️ 噪声未显著改变 Spread

Noise ablation 显示 spread 在所有噪声级别保持 1.0。这是因为：
- Demo replay 是开环控制，对初始状态敏感
- 噪声主要影响"已经能成功"的候选（C1 从 1.0 → 0.7-0.9）
- 对于"本来就不能成功"的候选，噪声无法改变其命运

### 🔬 仿真非确定性是关键挑战

- 不同任务的仿真非确定性程度不同
- Task 4: replay 发散约 1.66（116步后），但 suffix replay 仍可工作
- Task 7: replay 发散约 0.35（115步后），但已导致 SG1 失败
- 这表明 demo replay 方法的可靠性高度依赖任务特性

---

## 五、多任务扩展（Phase 6）

### Task 7 尝试

**Task**: `LIVING_ROOM_SCENE1` — "put both the alphabet soup and the cream cheese box in the basket"
- 子目标 1: `alphabet_soup_1` in `basket_1_contain_region`
- 子目标 2: `cream_cheese_1` in `basket_1_contain_region`

**结果**: 失败。Prefix replay 从 state[0] 到 step 115 后，soup 位置偏移达 0.35 单位，导致 SG1 验证失败。

**原因**: 仿真非确定性在此任务中更为严重，可能因为：
- 物体更多（5 个物体 vs 4 个）
- 接触交互更复杂
- Basket 的物理特性（碰撞体积）不同

**结论**: Demo replay 方法不适用于所有 LIBERO 任务。需要更鲁棒的控制器（如 OSC + 视觉反馈）来实现多任务扩展。

---

## 六、方法局限与改进方向

### 当前方法的局限

1. **开环 demo replay**: 对初始状态极度敏感，导致大部分候选失败
2. **仿真非确定性**: 不同任务的 replay 质量差异大
3. **二元结果**: 大部分候选要么全成功要么全失败，缺少连续的成功率分布

### 改进方向

1. **闭环控制器**: 使用 OSC_POSE 控制器 + 物体位姿估计，实现更鲁棒的 pick-and-place
2. **视觉反馈**: 利用 LIBERO 的 camera observations 进行闭环控制
3. **更多候选**: 使用不同 demo 或不同控制策略生成更多样的候选终态
4. **更大噪声范围**: 测试更细粒度的噪声级别，找到 spread 从 0 到 1 的过渡区间

---

## 七、结论

**核心现象已验证**: 在 LIBERO benchmark 的 Task 4 中，不同但都满足 symbolic goal 的中间终态，确实导致后续任务成功率出现巨大差异（spread = 1.0）。

**关键指标**:
- Current success (SG1): 100% (20/20 candidates)
- Future success (SG2): 21.5% mean, spread = 1.0
- Baseline SG2: 100% (True)

**意义**: 
- Symbolic equivalence does NOT guarantee equal future recoverability
- 在任务规划中，仅满足 symbolic goal 是不够的
- 需要选择"对未来任务友好"的终态，而非仅满足当前 symbolic constraint

---

## 附录：实验代码

- 主脚本: `experiment_demo_replay.py`
- 环境: `turbovla-libero` conda environment
- 依赖: libero, robosuite, mujoco 2.3.2, h5py, numpy

### 运行命令

```bash
conda activate turbovla-libero
python experiment_demo_replay.py
```

### 关键函数

| 函数 | 功能 |
|------|------|
| `run_experiment()` | 主实验流程，支持多任务参数化 |
| `generate_candidates()` | 直接 qpos 扰动 + 物理 settle 生成候选 |
| `replay_actions()` | Demo action replay + 可选高斯噪声 |
| `find_subgoal_boundary()` | 在 demo states 中定位子目标完成时刻 |
| `get_obj_qpos_idx()` | 动态查找物体在 state array 中的 qpos 索引 |