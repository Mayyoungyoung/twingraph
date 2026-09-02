# Task C（Fixture Loading）Phase 0 可行性设计

> 状态：**设计稿，待用户确认**
> 依据：`RESEARCH_SPEC.md` 第三、五、九、十、十一节；`poc_taskc_fixture.py`（纯 MuJoCo 物理机制 POC，无机器人）
> 架构基准：Task A gearbox 链（`assembly/gearbox_*.py`，已收官，作为 Task A 保存不动）

---

## 1. 现状审查结论（Task A gearbox 模式 → Task C 复用清单）

### 1.1 Task A 已收官，代码保持不变

gearbox 链（8 stages：housing→shaft→gear→spacer→bearing→cover→latch→test）M1–M5 全部完成，
`assembly/` 下文件即最终资产，本任务不改动任何 gearbox 文件。

### 1.2 gearbox 分层架构（Task C 直接复用的骨架）

| 层 | gearbox 文件 | 职责 | Task C 对应物 |
|---|---|---|---|
| Arena | `gearbox_arena.py` | 桌面 + 固定托盘/套筒 + 参考 site；**名义坐标唯一真相** | `FixtureArena(TableArena)`：夹具底座 + 双销 + 压板（含 actuator）+ 工件垫块 + 参考 site |
| Objects | `gearbox_objects.py` | `CompositeObject` 原语几何，clearance/深度全可配 | `WorkpieceObject`（网格切孔板，POC 技术） |
| Env | `gearbox_env.py` | 零件注册、`stage_metrics(k)` 终态指标、`stage_success(k)` 宽松谓词 | `FixtureAssemblyEnv(SingleArmEnv)` |
| Skills | `gearbox_skills.py` | `grasp_standing_part` + 各 stage 放置技能 + `_drop_at` 核心 | S1–S8 技能（复用 `goto`/`goto_safe`/`_goto_pose`/`_drop_at`/`close_full` 模式） |
| Planners | `gearbox_planners.py` | `Plan` dataclass + A1–A5 五变体，仅改 transport 参数 | P1–P4（规格书 C.5） |
| Experiment | `experiment_gearbox.py` | 配对设计、边界 state 保存、`restore()` 配方、future rollout、CSV | 原样复制框架 |
| Analyze | `analyze_gearbox.py` | 阶段成功率/终态 spread/V 统计 | 适配新指标 |

### 1.3 零改动复用的基础设施

- `env_adapter.py`（LIBERO 兼容适配层，`RobotSkills` 免改）
- `robot_skills.py`（OSC 闭环、夹爪、停滞检测、perception/action noise 钩子）
- `experiment_gearbox.py::restore()` 配方（`ctrl/qfrc_applied/xfrc_applied/act/mocap` 清理 +
  `controller.goal_pos/goal_ori` 重置 + 夹爪 `current_action` 重映射）——**原则 6 要求的 restore determinism 已解决**
- `robosuite_macros.SIMULATION_TIMESTEP = 0.005` 提速手法（Task C 需重新评估，见 §7.5 风险 1）

### 1.4 已有 Task C 物理 POC（`poc_taskc_fixture.py`，纯 MuJoCo 无机器人）

2026-08-26 复跑 6 case（dx/dy/yaw 注入，全部落入 S2 宽松谓词范围）：

```
 dx    dy  yaw |  pinA  pinB |  res_x  res_y res_yaw |    op
 0.0   0.0  0.0 | 10.98 10.98 |  -0.01  -0.18    0.01 |  0.18
 0.8   0.0  0.0 | 11.09 11.05 |  -1.66  -0.02   -0.35 |  1.70   <- 唯一分化 case
 1.2   0.6  1.0 | 10.98 10.98 |   0.08  -0.04   -0.34 |  0.16
 0.0   1.0  2.0 | 10.98 10.98 |  -0.01   0.00   -0.04 |  0.00
-1.0   0.5 -1.5 | 11.00 10.98 |  -0.00  -0.01   -0.01 |  0.01
 1.5  -0.8  2.5 | 10.96 11.17 |   0.01  -0.05   -0.14 |  0.08
```

**实测结论（与 POC 注释叙事有出入，如实记录）**：

- ✅ 全链物理成立：S2 放置误差确实能传播为 op_err 分化（0.18mm vs 1.70mm，10×）；
- ❌ **双销自定心效应强**：6 case 中 5 个（含 yaw 2~2.5°、dy 1mm）被销**拉回对齐**（op_err < 0.2mm），双销 qpos 均 FULL（~11mm）——注释声称的 "PARTIAL 捕获带 ±0.3mm/±1°" **未复现**，实际捕获/拨正范围大得多（销顶倒角 + 孔壁 jag 的拨正作用）;
- ⚠️ case 2（dx=0.8mm）的分化机制是 **clamp 推偏**而非 PARTIAL：销已将板拉回至 -0.27mm，但 −x 侧压板（中心 x=−18mm，半宽 14mm，恰好压在销 A 上方）下压时把板推向 −x 至 −1.66mm → op_err 1.70mm。这是一个**新的、真实的传播通道**（clamp×locator 相互作用），但只对特定初始误差触发；
- ⚠️ case 2 的 settle 阶段出现反常漂移（注入 +0.8mm → settle 后 −0.10mm，横向漂移 0.9mm），其他 case 漂移 <0.1mm——网格孔 jag 直接参与动力学，0.8mm 恰好落在 jag 不对称位置；
- ✅ 压板力控制（目标低 0.5mm + 接触力 >5N 停止）与销 0.2N 力限行为符合设计（无顶飞/穿透）；
- 已知问题：`poc_taskc_diag_drift.py` 确认的 settle 期 -y 漂移量级 <0.1mm（除 case 2），可接受但需在 robosuite 集成后复测。

**Phase 0 待办（对 §7.4 风险的直接响应）**：参数扫描（dx∈{0.5,0.8,1.2,1.6}mm × yaw∈{0,1,2,3}°）确认分化触发边界；若自定心过强（分化 case 占比过低），调整销几何/力限或 clamp 布局使 PARTIAL 通道稳定出现。

---

## 2. 几何设计（尺寸 / 材质 / 参考系）

### 2.1 全局参考

- 桌面顶 `table_top_z = 0.8`（robosuite 默认，gearbox 同）；`table_full_size=(0.8, 0.8, 0.05)`
- 夹具中心 `fixture_xy = (0.15, 0.0)`（同 gearbox 托盘位，Panda 工作空间内）
- **名义坐标唯一真相原则**（同 gearbox）：planner/技能只瞄准固定名义点，绝不读取工件实时位姿

### 2.2 Workpiece（矩形加工工件）

| 项 | 值 | 说明 |
|---|---|---|
| 外形 | 70 × 40 × 10 mm（半尺寸 0.035/0.020/0.005） | 铝，density 2700 → 75.6 g / 0.74 N |
| 定位孔 | A(−0.025, 0)、B(+0.025, 0)，r=4.3mm，通孔（z 向） | 孔距 50mm，孔轴 = 板厚度方向 |
| 孔几何 | 网格切割（POC 技术）：bulk 2mm + 孔周 0.5mm fine ring | jag band ~0.35mm << 捕获带 |
| **Phase 0 优化项** | 孔壁改 16 段 box 环（内径 4.3mm），板主体 2mm 网格在孔区留方洞 | geom 数 ~3400 → ~730；jag ~0.08mm；接触稳定、dt 可放宽。**实现前先跑 quick test，失败则沿用 POC 网格** |
| 名义操作点 | 工件局部 (0.008, 0.009)（POC OP_NOM） | 挂在工件上的 site；世界名义操作点 = fixture_center + R(yaw=0)·OP_NOM |

### 2.3 Locator pins（双定位销，fixture 内 actuator 驱动）

| 项 | 值 | 说明 |
|---|---|---|
| 直径 | r=3.8mm（对孔 4.3mm → 径向间隙 0.5mm） | 捕获带 ±0.3mm |
| 结构 | 主圆柱 + 顶部倒角（防卡缘），复位时顶面与板面齐平 | POC 几何 |
| 关节 | slide joint（z 轴），range 0..0.02，damping 0.3，frictionloss 0.02 | 真实接触，无 teleport |
| actuator | position，kp=250，ctrlrange 0..0.02，**forcerange ±0.2N** | 0.2N < 板重 0.74N → 未对齐时退缩 |
| 名义升程 | 11mm（穿透 10mm 板厚） | S3 执行参数（固定，非 planner 旋钮） |

### 2.4 Clamp（压板）

| 项 | 值 | 说明 |
|---|---|---|
| 压板 | 28 × 14 × 6 mm box，位于 −x 侧 | 压板只压 −x 边（POC） |
| 关节 | slide joint（z 轴），range −0.028..0.004，damping 3.0，frictionloss 0.1 | 真实接触 |
| actuator | position，kp=600，forcerange ±12N | 力控下压：目标低于接触面 0.5mm → 伺服饱和于有界压力，接触力 >5N 停止（POC `clamp_down`） |
| 设计意图 | **clamp 不能完全消除位姿误差**：FULL 定位 → 残余 ~0.1mm 级；PARTIAL → 残余保留部分放置误差 | 残余由 MuJoCo 接触自然产生，无人工公式 |

### 2.5 Fixture 底座与 cavity

- 底座板 + 4 矮墙（POC 布局）：内腔约 78 × 46 mm（x 间隙 ±4mm、y 间隙 ±3mm）
  - x 间隙 ±4mm、y ±3mm **明显大于**销捕获带 ±0.3mm → 满足"局部成功区域宽、未来可行区域窄"（§7）
  - yaw 自由范围约 ±4~6°（角点碰墙极限），与规格书 S2 谓词 yaw ±3~5° 一致（pilot 校准）
- **墙高调整建议：8mm → 6mm**（板厚 10mm，墙高 6mm 时板顶露出 4mm，S8 可垂直抓取；yaw 约束由间隙几何决定，与墙高无关）

### 2.6 Tool（EEF 工具附件）

- 圆柱 r=4mm、长 25mm，**挂载在 PandaGripper 的 `eef` body 下（fixed joint）**，尖端超出指面底端 ~15mm
  - 已验证：`PandaGripper` 继承 `MujocoXML`，可在 `eef` body 下 append body + fixed joint + geom（`mujoco.xml` 中 fixed joint 无自由度，跟随 eef）
  - contype/conaffinity：仅与工件碰撞（操作接触），不与夹具/桌面碰撞
- 语义：只模拟接触/压入/打标，不模拟切削

### 2.7 工件存放（S1 抓取源）

- **推荐方案：垫块平放**——板平放于桌面 15mm 高垫块上；S1 夹爪垂直下降，指面夹持 10mm 板厚（指面底端距垫块顶 ≥7mm，不碰垫块）；S2 保持水平搬运 + 垂直下放，**全程无需翻转**
- 备选方案：竖立槽（板竖直插在桌面槽中，S1 复用 `grasp_standing_part`，S2 需 90° 翻转放置）——风险高，不推荐

---

## 3. 关节设计汇总

| 关节 | 类型 | 范围 | actuator | 力限 | 驱动方 | 阶段 |
|---|---|---|---|---|---|---|
| `wp_free` | free | — | — | — | 机器人 | S1/S2/S8 |
| `pinA_j` / `pinB_j` | slide(z) | 0..0.02 | position kp=250 | ±0.2N | fixture 执行器 | S3 |
| `clamp_j` | slide(z) | −0.028..0.004 | position kp=600 | ±12N | fixture 执行器 | S4/S7 |
| tool | fixed(挂 eef) | — | — | — | 随 eef | S5/S6 |

- fixture 关节（pins/clamp）由 **robosuite Arena 层注入**（已验证 `MujocoXML.actuator` 可 append，`merge()` 时随 arena 合并）
- 所有 fixture 关节 qpos 随 sim state 保存/恢复（原则 5/6）；`ctrl` 需在 `restore()` 时重设（配方已有）
- **无 teleport**：全部状态由闭环执行 + 真实接触产生

---

## 4. Stage 定义、执行者与谓词

| # | Stage | 执行者 | Local success（宽松谓词） | 记录指标（终态） |
|---|---|---|---|---|
| S1 | Grasp Workpiece | 机器人 | 抓持稳定：lift 30mm 后 `is_grasping` 且工件位姿变化 < 阈值 | grasp offset (dx,dy,dyaw)、lift 后工件位姿 |
| S2 | Place Workpiece into Fixture | 机器人 | 工件中心在 cavity 内：\|x\|<2mm、\|y\|<2mm、\|yaw\|<5°、z 在板面 ±1mm（pilot 校准） | x_err、y_err、yaw、z |
| S3 | Engage Dual Locators | fixture（pin A→B 依次 ramp） | **primary engaged**：pin_A 进入深度 ≥5mm 且板不倾覆（tilt<8°）；记录 FULL/PARTIAL/FAILED | pin_A_depth、pin_B_depth、分类、板位姿变化、接触配置 |
| S4 | Clamp Workpiece | fixture（力控下压） | clamped：clamp 接触力 >2N 且板 z 位移收敛 <1mm | **x_residual、y_residual、yaw_residual**（与 S2 对比）、clamp 力 |
| S5 | Move Tool to Nominal Operation Point | 机器人 | 工具尖端 xy 距名义操作点 <2mm、z 到位 | 工具名义定位误差（≈0，非研究通道） |
| S6 | Execute Operation | 机器人 | 局部执行成功：工具接触工件 + 名义下压深度达到 | **operation_error**（工具接触点投影 vs 工件真实 op_point site 的水平距）、contact_force、quality=f(op_err, force)、operation_success |
| S7 | Release Clamp | fixture（clamp 归位） | clamp 回到初始位（qpos>0.002）且接触力 <0.5N | clamp qpos、释放后板位姿（回弹量） |
| S8 | Unload Workpiece | 机器人 | 工件中心距 cavity 中心 >30mm（已取出） | 取出后工件位姿、是否中途掉落 |

**关键设计点（C.4 核心链）**：
```
S2 Place → S3 Locator → S4 Clamp → residual pose(x/y/yaw) → S5/S6 Tool 对准名义点 → operation quality
```
- S5/S6 工具**只瞄准固定世界名义点**（工件名义位姿下的操作点投影）；工件真实残留位姿由 S2→S3→S4 接触产生 → `operation_error` 是物理残差，不是"忘了做坐标变换"
- S3 不设"至少一销进入即成功"：primary（pin A）为最低门槛，同时用 pin_B_depth 区分 FULL/PARTIAL → 残余 mobility 差异（mobility 的直接测量见 §7.6，Phase 1 标定）

---

## 5. Candidate 生成机制（对照规格书第五节）

所有候选均由**真实闭环执行**产生，初始状态、subgoal、几何全部相同；无 qpos teleport、无 endpoint 修改。

### TYPE I — Planner-induced（S2 place 的 transport 参数；对应规格书 C.5）

| Planner | 参数变体（gearbox 映射） | 物理通道 |
|---|---|---|
| P1 Shortest Cartesian | drop_dz=0.004、align_xy=False、grip 低位（A1 映射） | 释放高度低 → 指面拖拽/接触 → 放置偏差 |
| P2 Min Joint Motion | safe_z 取最低走廊（A2 映射） | 低空搬运 → 接近角差异 → 放置偏差 |
| P3 Clearance-aware | safe_z=1.06、drop_dz=0.030（A4 映射） | 高空 + 大落差 → 撞击/回弹 → 放置偏差 |
| P4 Smoothness-aware | 名义参数（A5 映射，M2 验证过的基准） | 基准：放置最接近名义点 |

- 全部走 OSC_POSE（**论文诚实声明**：无 joint-space 控制；P2 用 waypoint 高度差异近似关节运动差异——与 gearbox A2 同一手法）
- `Plan` dataclass 直接复用（drop_dz/gain/tol/safe_z/align_xy/grip_dy/grip_dz）

### TYPE II — Execution-induced（固定 P4）

- 抓取感知噪声 `noise_std`：0.002 / 0.004（gearbox 验证过的档位）
- 抓取位置变体 `grip_dy`：±1mm（夹持横向偏移 → 搬运后放置偏差）
- 全部候选仍须通过 S2 宽松谓词

### TYPE III — Combined

- P1×N0/N1/N2、P3×N0/N1/N2（规格书示例）

**S3/S4/S7 fixture 执行参数固定**（不变量）——研究的是"放置残差如何经定位/夹紧自然传播"，不是 fixture 控制差异。

---

## 6. Planner 实现可行性

- **高**：全部旋钮（drop_dz/gain/tol/safe_z/align_xy/grip_dy/grip_dz）已在 `gearbox_skills.py` 的 `_drop_at`/`goto_safe`/`grasp_standing_part` 中实现并验证；Task C 只换目标坐标与释放几何
- S6 下压：复用 `_goto_pose`（位姿闭环）+ 深度判据（工具尖端 z），新增小段"接触力读取"逻辑（`mujoco.mj_contactForce`，POC 已验证）
- S3/S4/S7 fixture 执行：把 POC 的 `ramp`/`clamp_down` 移植为 env 方法（actuator ctrl 直接写入 `sim.data.ctrl`）

---

## 7. Reachable Terminal-State 分析（规格书第十一节，最重要的判据）

### 7.1 结构：局部成功区 >> 未来可行区 ✓

```
         S2 Local Success Region (谓词 ±2mm / ±5°)
     ┌────────────────────────────────────┐
     │                                    │
     │   Future-good (双销 FULL, 捕获带)    │
     │   ┌──────────────┐                 │
     │   │ ±0.3mm / ±1° │                 │
     │   └──────────────┘                 │
     │      Future-bad (PARTIAL/FAILED)   │
     └────────────────────────────────────┘
```
- R_2（S2 成功终态）：x/y ∈ [−2,2]mm、yaw ∈ [−5°,5°]（谓词内，cavity 物理允许更大）
- future-good（双销 FULL + 自定心回中）：**实测范围远大于注释声称的 ±0.3mm**（6 case 中 5 个被拉回，含 yaw 2.5°）——捕获带边界**未知，待参数扫描**；但已确认存在触发区（dx≈0.8mm → op_err 1.70mm）
- 面积比取决于触发区宽度：若触发区为 ~±0.5mm 窄带 → 比 ~(4/1)² ≈ 16×；若自定心覆盖几乎全部 R_2 → 分化不足（UNSUITABLE 风险）——**这是 Phase 0 扫描首先要回答的问题**

### 7.2 传播链机制（POC 实测，2026-08-26 复跑）

| S2 终态 | S3 结果 | S4 后残余 | S6 op_err |
|---|---|---|---|
| 名义附近 / 多数偏差 case（含 yaw≤2.5°） | 双销 FULL + 自定心拉回 | ~0.1mm 级 | <0.2mm（夹具精度级） |
| 特定误差（实测 dx≈0.8mm 触发） | 双销 FULL 但板未完全回中（−0.27mm） | **clamp 推偏至 −1.66mm** | 1.70mm（放大 ~10×） |
| 更大偏差（>2mm / >5°，未测） | 双销 PARTIAL/FAILED（机制存疑，待扫描） | ? | ? |

**设计意图修正**：原设计假定"PARTIAL 定位 → 残余保留"是主通道；实测显示"**clamp × locator 相互作用推偏**"（case 2）才是当前几何下实际出现分化的通道，且自定心会治愈多数误差。两条通道都是真实接触产物、都满足研究叙事（G_2=1 而 op_err 分化），但 **PARTIAL 通道需要重新标定才能稳定出现**（见 §7.4 风险 2）。

### 7.3 判据检查（规格书第九节 A–F）

| 判据 | 状态 | 依据 |
|---|---|---|
| A 存在 ≥2 自然终态 cluster | ⚠️ 待扫描 | POC 6 case 中 1/6 产生 op_err 1.70mm vs 其余 <0.2mm（双峰雏形）；但自定心效应强，需参数扫描确认触发区宽度与真实执行可达性 |
| B 全部通过 G_i | ✓ 设计保证 | 谓词 ±2mm/±5° >> 捕获带 |
| C downstream value 分化 | ⚠️ 待 pilot | POC 已实测双峰雏形（0.18 vs 1.70mm，10×）；真实执行下是否稳定分化待扫描/耦合 pilot |
| D 无 endpoint 修改 | ✓ | planner 只改 transport 参数 |
| E 执行可重复 | ✓ 配方已有 | `restore()` + OSC 残留清理（M4 验证过） |
| F ≥5 stages | ✓ | 8 stages |

### 7.4 风险清单（pilot 必须回答的问题）

1. **接触/性能风险（高）**：网格板 ~3400 geom × robosuite 0.005s dt → 可能极慢/不稳定。缓解：孔壁 16 段环优化（§2.2，先 quick test）；dt 用 0.001–0.002 兜底（速度代价 ~3–5× gearbox）
2. **双销自定心过强（高，POC 实测）**：6 case 中 5 个被销拉回对齐 → 分化触发区可能很窄，planner/噪声 spread 落不进去 → 无分化（T3/T9 教训）。缓解：① 参数扫描定位触发边界（dx≈0.8mm 附近）；② 减小销倒角/增大销摩擦使 PARTIAL 更易卡缘；③ 调整 clamp 位置使其与 locator 的相互作用对更多误差触发（case 2 机制）；④ 所有缓解都需保持"真实接触"原则
3. **S2 spread 覆盖风险（高）**：cavity 引导 + 精确技能可能使所有 planner 终态过窄。缓解：cavity 间隙 ±3–4mm（> 触发区 2–4×）；planner 用 no-align/drop 变体；pilot 先扫 S2 终态 spread
4. **网格 jag 引入的噪声（中，POC 实测）**：case 2 settle 漂移 0.9mm 疑似 jag 不对称所致 → 既可能成为 spread 来源（可用），也可能掩盖机制（需区分）。缓解：扫描时记录 settle 漂移；孔壁环优化（§2.2）可消除 jag 不确定性
5. **S1 抓取稳定性（中）**：75g 板、10mm 厚度夹持，静摩擦临界。缓解：指面中心对齐板中心、夹持后先微抬验证；必要时垫块开槽让指面伸入
6. **S8 几何（低）**：墙高 6mm 时垂直抓取几何分析可行（指面在 y=±21mm 不与 ±27mm 墙碰撞），pilot 确认
7. **S6 接触力读数（低）**：`mj_contactForce` 接口 POC 已验证

### 7.5 性能预算（估）

- 网格方案：~3400 geom、dt 0.001 → 单 stage 约数分钟（M4 全链 30 链 × 8 stage 预计 10h+，需优化）
- 孔壁环方案：~730 geom、dt 0.003–0.005 → 与 gearbox 同量级（全链实验可接受）

---

## 8. 实现顺序（对照规格书第十节）

1. ✅ 审查 repository（本文档）
2. `FixtureArena`（含 pins/clamp actuator 注入，§3 已验证 API）
3. `WorkpieceObject`（孔壁环 quick test → 决定孔几何方案）
4. Tool 挂载（gripper `eef` body 注入，§2.6 已验证）
5. `FixtureAssemblyEnv` + S1–S8 skills（复用 gearbox 模式）
6. 谓词 + `stage_metrics`
7. **Phase 0 门禁：名义 demo 全链 8/8 通过**
8. 几何 pilot：① 参数扫描（dx/yaw 网格，POC 层）确认分化触发边界与通道（PARTIAL vs clamp 推偏）；② S2 终态 spread 扫描（drop/grip/noise 旋钮）→ 确认真实执行可达触发区 + settle 漂移量级
9. restore determinism 检查（同一终态 → 两次 future rollout 结果一致）
10. 通过后才进入 coupling pilot（10–30 candidates）→ planner matrix → noise → combined → long-horizon

---

## 9. 需要用户确认的决策点

| # | 决策点 | 推荐 | 备选 |
|---|---|---|---|
| 1 | 工件存放（S1 抓取源） | 垫块平放 + 垂直抓取（无翻转） | 竖立槽 + 90° 翻转放置 |
| 2 | cavity 墙高 | 6mm（S8 垂直抓取） | 8mm（POC 原样，S8 侧向抓取） |
| 3 | 孔几何 | 孔壁 16 段环（性能，先 quick test） | POC 网格切孔（已验证，慢） |
| 4 | planner 数量 | 规格书 4 个（P1–P4） | 5 个（加 min-time，同 gearbox A3） |
| 5 | S6 下压深度 / 力阈值 | pilot 标定 | — |
| 6 | fixture 位置 | (0.15, 0)（同 gearbox 托盘） | 其他 |

---

## 10. 交付物与下一步

- 本设计经用户确认后，按 §8 顺序进入实现；每个 phase 完成时按规格书第十二节输出：
  做了什么 / 为什么 / trials / 成功率 / 终态分布 / planner 是否产生不同轨迹 / downstream 是否分化 / 是否通过 / 下一步 / 失败标记
- 若 pilot 发现终态被压缩或 future value 恒定 → 停止并标记 **UNSUITABLE / NEEDS REDESIGN**（规格书原则 7）
