# simbench 原子技能清单（Skill Inventory）

> 由 `python -m simbench.skill_inventory` 从 `SkillRegistry` 契约自动生成；请勿手工编辑。

分层可组合技能库：**L0** 原语（`MjContext` / `CartesianController` / `Gripper`）→ **L1** 原子技能（执行/规划/过渡/扩展四类）→ **L2** 契约与门控（`SkillSpec` + `SkillRegistry.run`）→ **L3** 组合（`run_chain` 与 planner/executor 的 plan→exec 工件握手）。

**技能总数：22**（原子 15 + 组合 7）　执行类：13　规划类：2　过渡类：4　扩展类：3

**粒度约定**：`原子` = 单一最小职责（一次夹爪动作 / 一段运动 / 一次判定 / 一次计算），内部不含多步工作流；`组合` = 由多个原子编排的复合行为（保留给 A/B/C 生产链的实测稳定配方），其分解见下方「组合技能分解」。组合只允许出现在组合层（run_chain / planner），不得注册成 atomic 名字。

## 汇总表

| 技能名称 | 类别 | 粒度 | 功能描述 | 输入 | 输出 | 实现方式 | 依赖项 | 已封装 | 验证方式 / 成功标准 |
|---|---|---|---|---|---|---|---|---|---|
| `descend` | 执行类 | 原子 | 垂直下降（最小原子）：保持当前 xy，把 EEF 下降到目标 z（或给定 xyz）。接近后/放置前的精降原语，是 grasp/place/insert 分解中的下降步。 | `to_z`, `to`, `gain`, `tol` | `reached`, `eef_z` | 运动规划 | CartesianController.move_eef | 是 | EEF z 到达 to_z（以 reached 判定） |
| `detect` | 执行类 | 原子 | 零件检测：在失败模型下感知零件位姿（位置+偏航），可注入高斯噪声/漏检/误检，输出观测供 plan_grasp_pose 消费。是执行链的感知入口。 | `part`, `faults`, `step` | `found`, `pos`, `yaw`, `outcome` | 规则脚本 | perception.detect_part, faults.FailureModel(可选) | 是 | 检测到则 outcome!=miss 且 pos 非空 |
| `grasp` | 执行类 | 组合<br>↳ detect→plan_grasp_pose→approach→grip_close→retreat_lift→lift_verify | 抓取：检测/估计（或消费 plan_grasp_pose 工件）→ 接近→ 下降 → 闭爪 → 举升校验的闭环抓取。前置零件存在，后置以举升量判定抓稳。是操作链的核心执行技能。 | `part`, `grasp_pose`, `press`, `lift`, `tol`, `squeeze`, `approach_yaw` | `ok`, `held_dist` | 规则脚本 | perception.estimate_grasp_pose, motion.move_eef, Gripper.close_on_part, plan_grasp_pose(可选) | 是 | 零件被抬升 >= VERIFY_LIFT（以 ok 判定） |
| `grip_close` | 执行类 | 原子 | 闭合夹爪（最小原子）：仅将手指闭到指定开度/压入量/接触力，不做接近或感知。三种口径：给 part/outer_d 按外径+press 闭合并过冲咬合；给 span 闭到该 pad 间距；给 force_stop 闭到 pad 接触力达标。即“grasp 只负责闭合夹爪”的那个原子。 | `part`, `outer_d`, `span`, `press`, `force_stop` | `span`, `pad_force` | 规则脚本 | Gripper.close_on_part, Gripper.close_to_span, perception.part_meta(可选) | 是 | 手指已闭合（以 span/pad_force 度量） |
| `grip_open` | 执行类 | 原子 | 张开夹爪（最小原子）：仅将手指伺服到全开，不做任何移动或感知。释放/接近前的手指原语。 | `max_steps` | `span` | 规则脚本 | Gripper.open | 是 | pad 间距达到全开（以 span 度量） |
| `insert` | 执行类 | 组合<br>↳ pre_align→descend(thread+抗卡摆动)→release | 插入装配：mode='press' 到点压入并保压；mode='thread' 底部导向螺旋下降（比例纠偏+抗卡摆动），把持件装入孔/轴。前置仍持件。是插销/套轴等装配的执行原语（学习版见扩展类 peg_insert mode='policy'）。 | `part`, `mode`, `target_eef`, `ref_axis`, `to_z`, `half` | `ok`, `bot_z` | 优化 | manipulation.insert, perception.part_meta, CartesianController.move_eef | 是 | 零件底面到达 to_z 附近（thread 模式） |
| `inspect` | 执行类 | 原子 | 质量判定（原子）：对（已就位的）零件做一次几何检查并给出 OK/NG 结论，不驱动任何运动。kind='seat' 测 xy/z 落位误差与倾斜角；kind='pin_engage' 测底部进入孔的深度与孔轴偏差。可叠加传感器测量噪声。检查的是“装配质量”（落座/压入是否达标），不是“零件存在性”（那是 detect）。 | `part`, `kind`, `target_xy`, `target_z`, `xy_tol`, `z_tol`, `tilt_tol`, `axis_xy`, `top_ref_z`, `half`, `min_depth`, `bot_off_tol`, `noise_std` | `ok`, `tilt`, `xy_err`, `z_err`, `depth`, `bot_off` | 规则脚本 | MjContext.obj_pos, MjContext.obj_tilt, MjContext.obj_axis | 是 | metrics 含判定量且 ok 反映容差判定 |
| `lift_verify` | 执行类 | 原子 | 举升校验（最小原子，纯判定）：比较零件当前 z 与参考 z，确认已被抬升 >= min_lift，判抓取是否抓稳。不驱动任何运动，是 grasp 分解的收尾校验原语。 | `part`, `z_ref`, `min_lift` | `ok`, `z_now`, `lift` | 规则脚本 | MjContext.obj_pos | 是 | z_now - z_ref >= min_lift（以 ok 判定） |
| `move` | 执行类 | 原子 | 末端移动：按规划航点风格（direct/safe_z/clearance）把 EEF 移动到 world xyz 目标，可选梯形速度顺滑。是执行类的通用位移原语，消费 plan_path 的航点风格。 | `to`, `style`, `tol`, `gain`, `max_speed`, `smooth` | `reached`, `eef` | 运动规划 | motion.move_eef, planning.plan_path, CartesianController.move_eef | 是 | EEF 与目标距离 <= tol（以 reached 判定） |
| `place` | 执行类 | 组合<br>↳ transport→pre_align→descend→release→retreat_lift | 放置：把持件搬运到目标 xy/z 并释放落位。基于持件时实测的 EEF-零件偏移推导落点，抵消抓取检测噪声；支持对中/低空搬运/腕部扶正/多种释放策略。前置仍持件，后置以落位 xy 容差判定。 | `part`, `at`, `target_z`, `align`, `live_align`, `release`, `carry_style`, `low_carry` | `ok`, `pos` | 规则脚本 | manipulation.place, motion.move_eef, Gripper.open | 是 | 零件 xy 与目标距离 < 0.012m（以 ok 判定） |
| `push` | 执行类 | 原子 | 推动：并指成刃，从零件背后沿方向闭环推抵，实时监测零件随动位移，卡滞即停（不硬磨）。用于纠偏/送料等非抓持的水平位移。以位移达请求量 70% 判定成功。 | `part`, `delta`, `to_target`, `push_z`, `speed` | `ok`, `moved` | 规则脚本 | manipulation.push, motion.move_eef, Gripper | 是 | 零件沿方向位移 >= 70% 请求量（以 ok 判定） |
| `release` | 执行类 | 原子 | 释放（最小原子）：张开夹爪让持件原地落下并静置，不做抬升退避（退避见过渡类 retreat_lift）。是 place 分解的释放原语。 | `settle_steps` | `ok`, `span` | 规则脚本 | Gripper.open, settle | 是 | 夹爪全开（以 span 度量） |
| `transport` | 执行类 | 组合<br>↳ move(低速航点执行)→持件校验(谓词，滑脱即败) | 搬运（组合）：把持件沿给定航点（或到目标点的规划航点）低速平移到目标上方悬停，供后续 place/insert 精降；全程监控仍持件，滑脱即失败。由原子 move（航点执行）与持件校验组合而成。 | `part`, `waypoints`, `to`, `style`, `carry_speed`, `tol` | `ok`, `held` | 运动规划 | CartesianController.move_eef, motion.move_eef, plan_path(可选) | 是 | 搬运后 EEF-零件 xy 距离 < 0.03m（以 held 判定） |
| `plan_grasp_pose` | 规划类 | 原子 | 抓取位姿估计（多候选）：由检测观测+零件元数据生成 n_yaw 个接近角候选抓取位姿，按接近线碰撞/零件类型做可行性过滤，按行程+偏航失配+碰撞惩罚评分，排序选最优作为抓取工件（保留全部候选分支用于归因/数据导出）。 | `part`, `detect_result`, `noise_std`, `n_yaw`, `obstacles`, `yaw` | `grasp_pose`, `n_candidates`, `candidates` | 运动规划 | planning.grasp_pose_candidates, perception.part_meta, perception.detect | 是 | 至少一个可行候选（否则 ok=False，估计失败） |
| `plan_path` | 规划类 | 原子 | 路径规划（多候选）：由起点到目标生成多个航点候选（高度变体×侧向 via 偏移），对每个候选做线段-AABB 碰撞门过滤，按路径长+航点数评分，选最优可行路径；全碰撞时换 style 重规划。产出 waypoints 工件供 move/transport 消费。 | `to`, `cur`, `style`, `lift`, `safe_z`, `obstacles`, `via_offsets` | `waypoints`, `style`, `score`, `n_candidates`, `candidates` | 运动规划 | planning.path_candidates, planning.check_path, planning.seg_aabb_hit | 是 | 至少一个无碰撞候选（否则 ok=False，需重规划） |
| `approach` | 过渡类 | 组合<br>↳ move(悬停) -> descend(精降) | 抓取前接近：先到抓取点上方悬停，再垂直下降至抓取高度（不闭合）。与 grasp(approach_direct=True) 组合即标准两段式抓取接近，避免 grasp 内部 safe_z 二次抬升的下降-上升-抽搐。 | `part`, `grasp_pos`, `hover_lift`, `tol` | `eef_z` | 规则脚本 | perception.estimate_grasp_pose, motion.move_eef | 是 | — |
| `pre_align` | 过渡类 | 原子 | 下降前预对齐：在当前位置高度上，将持件（或空爪 EEF）对中到目标 xy，供后续垂直下降/放置直接衔接（gain/迭代沿用 place 实测配方）。 | `at`, `part`, `tol`, `max_iters` | `resid` | 规则脚本 | CartesianController.move_eef | 是 | — |
| `retreat_lift` | 过渡类 | 原子 | 放置/释放后垂直抬升退避：从当前 EEF 位置爬升指定高度并静置，避免 pads 扫掠已就位零件。 | `height`, `tol`, `settle_steps` | `lift_m` | 规则脚本 | motion.move_eef, settle | 是 | EEF 爬升量 >= height（skill 以 lift_m 度量判定） |
| `return_home` | 过渡类 | 组合<br>↳ grip_open(open_gripper=True 时)→move(可选抬升)→关节回扫 HOME | 回初始位：开爪（可选）+ 关节空间回 HOME。可选先爬升到安全高度再回扫，规避关节回扫扫飞已装配件（Task B 实测教训）。 | `open_gripper`, `retreat_z` | `eef` | 规则脚本 | Gripper.open, CartesianController.home, motion.move_eef | 是 | — |
| `peg_insert` | 扩展类 | 组合<br>↳ grasp→insert(thread/press) 或 policy(RL/IL 逐步循环)→release | 插销：把持件插入指定孔口。mode='press' 为到点压入，mode='thread' 为底部导向螺旋下降（抗卡摆动，规则闭环），mode='policy' 由学习策略（模仿学习/强化学习）输出插入动作序列，输入孔心 xy 与孔底 z，奖励显式考虑接触、卡滞与姿态偏差。 | `name`, `hole_xy`, `to_z`, `mode`, `policy`, `max_steps` | `depth_m`, `bot_z` | 规则脚本/强化学习/模仿学习 | grasp, manipulation.insert, skills.learned.insert_env, skills.learned.load_policy (mode=policy) | 是 | 零件底面到达 to_z 附近（以 depth_m 判定） |
| `pull` | 扩展类 | 原子 | 拉动：抓持零件后沿指定方向向机器人侧拖拽（与 push 反向的闭环位移技能）。若未持件则先轻咬合抓取并验证举升；拖动全程监控零件随动，滑脱即停。 | `name`, `delta`, `to_target`, `release`, `speed` | `moved_m` | 规则脚本 | grasp, manipulation._drive_along, Gripper | 是 | 位移 >= 70% 请求量（以 moved_m 判定） |
| `wipe` | 扩展类 | 原子 | 擦拭：并指成刃，轻压贴面后在目标面上沿直线扫掠，全程监控 pad 接触力处于目标力带（既保证贴实又不压穿/挑翻被擦面），可作触垫预清洁等语义。 | `at`, `direction`, `length`, `z`, `f_band` | `force_mean`, `force_max`, `swept_m` | 规则脚本 | Gripper.close_to_span, motion.move_eef | 是 | 接触力处于力带且扫掠完成（以 force_max 判定） |

## 组合技能分解（composite → 原子链）

组合技能保留实测稳定的内部配方（A/B/C 生产链使用）；其等价原子链（可由 `run_chain` 编排）如下：

- **`approach`** → move(悬停) -> descend(精降)
- **`grasp`** → `detect` → `plan_grasp_pose` → `approach` → `grip_close` → `retreat_lift` → `lift_verify`
- **`insert`** → `pre_align` → descend(thread+抗卡摆动) → `release`
- **`peg_insert`** → `grasp` → insert(thread/press) 或 policy(RL/IL 逐步循环) → `release`
- **`place`** → `transport` → `pre_align` → `descend` → `release` → `retreat_lift`
- **`return_home`** → grip_open(open_gripper=True 时) → move(可选抬升) → 关节回扫 HOME
- **`transport`** → move(低速航点执行) → 持件校验(谓词，滑脱即败)

## 技能明细（契约）

### 执行类（exec）

#### `descend`
- **功能描述**：垂直下降（最小原子）：保持当前 xy，把 EEF 下降到目标 z（或给定 xyz）。接近后/放置前的精降原语，是 grasp/place/insert 分解中的下降步。
- **实现方式**：运动规划
- **粒度**：原子
- **失败策略**：retry
- **是否已封装**：是
- **依赖项**：CartesianController.move_eef
- **输入**：
    - `to_z`：目标 world z（保持 xy）
    - `to`：可选完整 xyz 目标
    - `gain`：P 增益
    - `tol`：到位容差 (m)
- **输出**：
    - `reached`：是否到位
    - `eef_z`：终态 EEF z
- **前置条件**：场景已加载，机器人可控
- **后置条件 / 成功标准**：EEF z 到达 to_z（以 reached 判定）

#### `detect`
- **功能描述**：零件检测：在失败模型下感知零件位姿（位置+偏航），可注入高斯噪声/漏检/误检，输出观测供 plan_grasp_pose 消费。是执行链的感知入口。
- **实现方式**：规则脚本
- **粒度**：原子
- **失败策略**：retry
- **是否已封装**：是
- **依赖项**：perception.detect_part, faults.FailureModel(可选)
- **输入**：
    - `part`：零件名（body）
    - `faults`：可选 FailureModel（漏检/误检/噪声源）
    - `step`：可选步序号（供故障模型归因）
- **输出**：
    - `found`：是否检测到
    - `pos`：world xyz 观测
    - `yaw`：偏航观测 (rad)
    - `outcome`：ok\|miss\|false
- **前置条件**：`part_exists()`
- **后置条件 / 成功标准**：检测到则 outcome!=miss 且 pos 非空

#### `grasp`
- **功能描述**：抓取：检测/估计（或消费 plan_grasp_pose 工件）→ 接近→ 下降 → 闭爪 → 举升校验的闭环抓取。前置零件存在，后置以举升量判定抓稳。是操作链的核心执行技能。
- **实现方式**：规则脚本
- **粒度**：组合
- **原子分解**：detect → plan_grasp_pose → approach → grip_close → retreat_lift → lift_verify
- **失败策略**：retry
- **是否已封装**：是
- **依赖项**：perception.estimate_grasp_pose, motion.move_eef, Gripper.close_on_part, plan_grasp_pose(可选)
- **输入**：
    - `part`：零件名
    - `grasp_pose`：可选预估计抓取位姿工件
    - `press`：压入量 (m)
    - `lift`：举升高度 (m)
    - `tol`：下降容差 (m)
    - `squeeze`：力反馈深咬 (N)
    - `approach_yaw`：接近前腕部偏航
- **输出**：
    - `ok`：举升校验通过
    - `held_dist`：EEF-零件距离 (m)
- **前置条件**：`part_exists()`
- **后置条件 / 成功标准**：零件被抬升 >= VERIFY_LIFT（以 ok 判定）

#### `grip_close`
- **功能描述**：闭合夹爪（最小原子）：仅将手指闭到指定开度/压入量/接触力，不做接近或感知。三种口径：给 part/outer_d 按外径+press 闭合并过冲咬合；给 span 闭到该 pad 间距；给 force_stop 闭到 pad 接触力达标。即“grasp 只负责闭合夹爪”的那个原子。
- **实现方式**：规则脚本
- **粒度**：原子
- **失败策略**：continue
- **是否已封装**：是
- **依赖项**：Gripper.close_on_part, Gripper.close_to_span, perception.part_meta(可选)
- **输入**：
    - `part`：可选零件名（用其外径闭合）
    - `outer_d`：可选外径 (m)
    - `span`：可选目标 pad 间距 (m)
    - `press`：压入量 (m)
    - `force_stop`：可选：闭到该接触力 (N)
- **输出**：
    - `span`：闭合后 pad 间距 (m)
    - `pad_force`：pad 接触力 (N)
- **前置条件**：—
- **后置条件 / 成功标准**：手指已闭合（以 span/pad_force 度量）

#### `grip_open`
- **功能描述**：张开夹爪（最小原子）：仅将手指伺服到全开，不做任何移动或感知。释放/接近前的手指原语。
- **实现方式**：规则脚本
- **粒度**：原子
- **失败策略**：continue
- **是否已封装**：是
- **依赖项**：Gripper.open
- **输入**：
    - `max_steps`：开爪 slew 最大步数
- **输出**：
    - `span`：张开后 pad 间距 (m)
- **前置条件**：场景已加载，夹爪可控
- **后置条件 / 成功标准**：pad 间距达到全开（以 span 度量）

#### `insert`
- **功能描述**：插入装配：mode='press' 到点压入并保压；mode='thread' 底部导向螺旋下降（比例纠偏+抗卡摆动），把持件装入孔/轴。前置仍持件。是插销/套轴等装配的执行原语（学习版见扩展类 peg_insert mode='policy'）。
- **实现方式**：优化
- **粒度**：组合
- **原子分解**：pre_align → descend(thread+抗卡摆动) → release
- **失败策略**：abort
- **是否已封装**：是
- **依赖项**：manipulation.insert, perception.part_meta, CartesianController.move_eef
- **输入**：
    - `part`：零件名
    - `mode`：press\|thread
    - `target_eef`：press 模式目标 EEF xyz
    - `ref_axis`：thread 模式参考轴 (callable/xy)
    - `to_z`：thread 模式零件底面目标 z
    - `half`：零件半高
- **输出**：
    - `ok`：插入完成
    - `bot_z`：终态零件底面 z
- **前置条件**：`held_part()`
- **后置条件 / 成功标准**：零件底面到达 to_z 附近（thread 模式）

#### `inspect`
- **功能描述**：质量判定（原子）：对（已就位的）零件做一次几何检查并给出 OK/NG 结论，不驱动任何运动。kind='seat' 测 xy/z 落位误差与倾斜角；kind='pin_engage' 测底部进入孔的深度与孔轴偏差。可叠加传感器测量噪声。检查的是“装配质量”（落座/压入是否达标），不是“零件存在性”（那是 detect）。
- **实现方式**：规则脚本
- **粒度**：原子
- **失败策略**：continue
- **是否已封装**：是
- **依赖项**：MjContext.obj_pos, MjContext.obj_tilt, MjContext.obj_axis
- **输入**：
    - `part`：被检零件名
    - `kind`：seat\|pin_engage
    - `target_xy`：seat: 名义 world xy
    - `target_z`：seat: 可选名义 z
    - `xy_tol`：seat: xy 容差 (m)
    - `z_tol`：seat: z 容差 (m)
    - `tilt_tol`：倾斜容差 (deg)
    - `axis_xy`：pin_engage: 孔轴 xy
    - `top_ref_z`：pin_engage: 孔口参考 z
    - `half`：pin_engage: 零件半高
    - `min_depth`：pin_engage: 最小啮合深度 (m)
    - `bot_off_tol`：pin_engage: 底端偏移容差 (m)
    - `noise_std`：可选测量噪声 std (m)
- **输出**：
    - `ok`：判定结论
    - `tilt`：倾斜角 (deg)
    - `xy_err`：seat: xy 误差 (m)
    - `z_err`：seat: z 误差 (m)
    - `depth`：pin_engage: 啮合深度 (m)
    - `bot_off`：pin_engage: 底端偏移 (m)
- **前置条件**：`part_exists()`
- **后置条件 / 成功标准**：metrics 含判定量且 ok 反映容差判定

#### `lift_verify`
- **功能描述**：举升校验（最小原子，纯判定）：比较零件当前 z 与参考 z，确认已被抬升 >= min_lift，判抓取是否抓稳。不驱动任何运动，是 grasp 分解的收尾校验原语。
- **实现方式**：规则脚本
- **粒度**：原子
- **失败策略**：continue
- **是否已封装**：是
- **依赖项**：MjContext.obj_pos
- **输入**：
    - `part`：零件名
    - `z_ref`：抬升前参考 z (m)
    - `min_lift`：最小抬升量 (m)
- **输出**：
    - `ok`：抬升达标
    - `z_now`：当前零件 z
    - `lift`：实测抬升量
- **前置条件**：`part_exists()`
- **后置条件 / 成功标准**：z_now - z_ref >= min_lift（以 ok 判定）

#### `move`
- **功能描述**：末端移动：按规划航点风格（direct/safe_z/clearance）把 EEF 移动到 world xyz 目标，可选梯形速度顺滑。是执行类的通用位移原语，消费 plan_path 的航点风格。
- **实现方式**：运动规划
- **粒度**：原子
- **失败策略**：retry
- **是否已封装**：是
- **依赖项**：motion.move_eef, planning.plan_path, CartesianController.move_eef
- **输入**：
    - `to`：目标 world xyz
    - `style`：航点风格
    - `tol`：到位容差 (m)
    - `gain`：P 增益
    - `max_speed`：速度上限 (m/s)
    - `smooth`：梯形顺滑
- **输出**：
    - `reached`：是否到位
    - `eef`：终态 EEF xyz
- **前置条件**：场景已加载，机器人可控
- **后置条件 / 成功标准**：EEF 与目标距离 <= tol（以 reached 判定）

#### `place`
- **功能描述**：放置：把持件搬运到目标 xy/z 并释放落位。基于持件时实测的 EEF-零件偏移推导落点，抵消抓取检测噪声；支持对中/低空搬运/腕部扶正/多种释放策略。前置仍持件，后置以落位 xy 容差判定。
- **实现方式**：规则脚本
- **粒度**：组合
- **原子分解**：transport → pre_align → descend → release → retreat_lift
- **失败策略**：continue
- **是否已封装**：是
- **依赖项**：manipulation.place, motion.move_eef, Gripper.open
- **输入**：
    - `part`：零件名
    - `at`：目标 world xy
    - `target_z`：可选目标 z
    - `align`：落前对中
    - `live_align`：活体比例对中
    - `release`：释放策略
    - `carry_style`：搬运航点风格
    - `low_carry`：低空搬运
- **输出**：
    - `ok`：落位 xy 在容差内
    - `pos`：终态零件 xyz
- **前置条件**：`held_part()`
- **后置条件 / 成功标准**：零件 xy 与目标距离 < 0.012m（以 ok 判定）

#### `push`
- **功能描述**：推动：并指成刃，从零件背后沿方向闭环推抵，实时监测零件随动位移，卡滞即停（不硬磨）。用于纠偏/送料等非抓持的水平位移。以位移达请求量 70% 判定成功。
- **实现方式**：规则脚本
- **粒度**：原子
- **失败策略**：continue
- **是否已封装**：是
- **依赖项**：manipulation.push, motion.move_eef, Gripper
- **输入**：
    - `part`：零件名
    - `delta`：位移 (dx,dy)
    - `to_target`：可选绝对目标 xy（优先）
    - `push_z`：推动线 world z
    - `speed`：推动速度
- **输出**：
    - `ok`：位移 >= 70% 请求量
    - `moved`：实测位移向量
- **前置条件**：`part_exists()`
- **后置条件 / 成功标准**：零件沿方向位移 >= 70% 请求量（以 ok 判定）

#### `release`
- **功能描述**：释放（最小原子）：张开夹爪让持件原地落下并静置，不做抬升退避（退避见过渡类 retreat_lift）。是 place 分解的释放原语。
- **实现方式**：规则脚本
- **粒度**：原子
- **失败策略**：continue
- **是否已封装**：是
- **依赖项**：Gripper.open, settle
- **输入**：
    - `settle_steps`：释放后静置步数
- **输出**：
    - `ok`：已释放
    - `span`：张开后 pad 间距
- **前置条件**：—
- **后置条件 / 成功标准**：夹爪全开（以 span 度量）

#### `transport`
- **功能描述**：搬运（组合）：把持件沿给定航点（或到目标点的规划航点）低速平移到目标上方悬停，供后续 place/insert 精降；全程监控仍持件，滑脱即失败。由原子 move（航点执行）与持件校验组合而成。
- **实现方式**：运动规划
- **粒度**：组合
- **原子分解**：move(低速航点执行) → 持件校验(谓词，滑脱即败)
- **失败策略**：continue
- **是否已封装**：是
- **依赖项**：CartesianController.move_eef, motion.move_eef, plan_path(可选)
- **输入**：
    - `part`：零件名
    - `waypoints`：航点列表 (world xyz)
    - `to`：可选：目标 xyz（缺 waypoints 时按 style 规划）
    - `style`：航点风格
    - `carry_speed`：搬运速度上限
    - `tol`：航点到位容差
- **输出**：
    - `ok`：到达且仍持件
    - `held`：搬运后是否仍持件
- **前置条件**：`held_part()`
- **后置条件 / 成功标准**：搬运后 EEF-零件 xy 距离 < 0.03m（以 held 判定）

### 规划类（plan）

#### `plan_grasp_pose`
- **功能描述**：抓取位姿估计（多候选）：由检测观测+零件元数据生成 n_yaw 个接近角候选抓取位姿，按接近线碰撞/零件类型做可行性过滤，按行程+偏航失配+碰撞惩罚评分，排序选最优作为抓取工件（保留全部候选分支用于归因/数据导出）。
- **实现方式**：运动规划
- **粒度**：原子
- **失败策略**：retry
- **是否已封装**：是
- **依赖项**：planning.grasp_pose_candidates, perception.part_meta, perception.detect
- **输入**：
    - `part`：零件名
    - `detect_result`：可选外部观测 (found,pos,yaw,outcome)
    - `noise_std`：检测噪声 std
    - `n_yaw`：候选接近角数量（2=旧两轴）
    - `obstacles`：可选静态障碍 AABB 列表
    - `yaw`：可选指定偏航
- **输出**：
    - `grasp_pose`：选中的抓取工件 {pos,yaw,outer_d,...}
    - `n_candidates`：候选数
    - `candidates`：候选分支摘要
- **前置条件**：`part_exists()`
- **后置条件 / 成功标准**：至少一个可行候选（否则 ok=False，估计失败）

#### `plan_path`
- **功能描述**：路径规划（多候选）：由起点到目标生成多个航点候选（高度变体×侧向 via 偏移），对每个候选做线段-AABB 碰撞门过滤，按路径长+航点数评分，选最优可行路径；全碰撞时换 style 重规划。产出 waypoints 工件供 move/transport 消费。
- **实现方式**：运动规划
- **粒度**：原子
- **失败策略**：retry
- **是否已封装**：是
- **依赖项**：planning.path_candidates, planning.check_path, planning.seg_aabb_hit
- **输入**：
    - `to`：目标 world xyz
    - `cur`：可选起点（默认 EEF）
    - `style`：direct\|safe_z\|clearance
    - `lift`：抬升量 (m)
    - `safe_z`：clearance 绝对高度
    - `obstacles`：静态障碍 AABB
    - `via_offsets`：侧向 via 偏移候选
- **输出**：
    - `waypoints`：选中路径航点
    - `style`：路径风格
    - `score`：路径评分
    - `n_candidates`：候选数
    - `candidates`：候选分支+碰撞命中摘要
- **前置条件**：场景已加载，机器人可控
- **后置条件 / 成功标准**：至少一个无碰撞候选（否则 ok=False，需重规划）

### 过渡类（trans）

#### `approach`
- **功能描述**：抓取前接近：先到抓取点上方悬停，再垂直下降至抓取高度（不闭合）。与 grasp(approach_direct=True) 组合即标准两段式抓取接近，避免 grasp 内部 safe_z 二次抬升的下降-上升-抽搐。
- **实现方式**：规则脚本
- **粒度**：组合
- **原子分解**：move(悬停) -> descend(精降)
- **失败策略**：continue
- **是否已封装**：是
- **依赖项**：perception.estimate_grasp_pose, motion.move_eef
- **输入**：
    - `part`：目标零件名
    - `grasp_pos`：可选：预计算抓取点（plan_grasp_pose 工件），缺省时现场估计
    - `hover_lift`：悬停高度 (m)
    - `tol`：下降容差 (m)
- **输出**：
    - `eef_z`：下降后 EEF 高度
- **前置条件**：零件存在
- **后置条件 / 成功标准**：—

#### `pre_align`
- **功能描述**：下降前预对齐：在当前位置高度上，将持件（或空爪 EEF）对中到目标 xy，供后续垂直下降/放置直接衔接（gain/迭代沿用 place 实测配方）。
- **实现方式**：规则脚本
- **粒度**：原子
- **失败策略**：continue
- **是否已封装**：是
- **依赖项**：CartesianController.move_eef
- **输入**：
    - `at`：目标 xy (world)
    - `part`：可选：被持零件名（None=对齐 EEF 本身）
    - `tol`：收敛容差 (m)
    - `max_iters`：比例纠偏最大迭代数
- **输出**：
    - `resid`：收敛后的 xy 残差 (m)
- **前置条件**：场景已加载
- **后置条件 / 成功标准**：—

#### `retreat_lift`
- **功能描述**：放置/释放后垂直抬升退避：从当前 EEF 位置爬升指定高度并静置，避免 pads 扫掠已就位零件。
- **实现方式**：规则脚本
- **粒度**：原子
- **失败策略**：continue
- **是否已封装**：是
- **依赖项**：motion.move_eef, settle
- **输入**：
    - `height`：抬升高度 (m, 默认 0.07)
    - `tol`：抬升到位容差 (m)
    - `settle_steps`：抬升后静置步数
- **输出**：
    - `lift_m`：实测爬升量 (m)
- **前置条件**：场景已加载，机器人可控
- **后置条件 / 成功标准**：EEF 爬升量 >= height（skill 以 lift_m 度量判定）

#### `return_home`
- **功能描述**：回初始位：开爪（可选）+ 关节空间回 HOME。可选先爬升到安全高度再回扫，规避关节回扫扫飞已装配件（Task B 实测教训）。
- **实现方式**：规则脚本
- **粒度**：组合
- **原子分解**：grip_open(open_gripper=True 时) → move(可选抬升) → 关节回扫 HOME
- **失败策略**：continue
- **是否已封装**：是
- **依赖项**：Gripper.open, CartesianController.home, motion.move_eef
- **输入**：
    - `open_gripper`：回位前是否开爪
    - `retreat_z`：可选：先平移到该绝对 z 高度再回扫
- **输出**：
    - `eef`：回位后 EEF 位置
- **前置条件**：场景已加载
- **后置条件 / 成功标准**：—

### 扩展类（ext）

#### `peg_insert`
- **功能描述**：插销：把持件插入指定孔口。mode='press' 为到点压入，mode='thread' 为底部导向螺旋下降（抗卡摆动，规则闭环），mode='policy' 由学习策略（模仿学习/强化学习）输出插入动作序列，输入孔心 xy 与孔底 z，奖励显式考虑接触、卡滞与姿态偏差。
- **实现方式**：规则脚本/强化学习/模仿学习
- **粒度**：组合
- **原子分解**：grasp → insert(thread/press) 或 policy(RL/IL 逐步循环) → release
- **失败策略**：abort
- **是否已封装**：是
- **依赖项**：grasp, manipulation.insert, skills.learned.insert_env, skills.learned.load_policy (mode=policy)
- **输入**：
    - `name`：被插零件名（已由 grasp 抓持）
    - `hole_xy`：孔心 world xy
    - `to_z`：零件底面目标 z（孔底/座面）
    - `mode`：press \| thread \| policy
    - `policy`：可选：学习策略对象（mode='policy' 时必填）
    - `max_steps`：策略/闭环最大步数
- **输出**：
    - `depth_m`：实测插入深度 (m)
    - `bot_z`：终态零件底面 z
- **前置条件**：`held_part()`
- **后置条件 / 成功标准**：零件底面到达 to_z 附近（以 depth_m 判定）

#### `pull`
- **功能描述**：拉动：抓持零件后沿指定方向向机器人侧拖拽（与 push 反向的闭环位移技能）。若未持件则先轻咬合抓取并验证举升；拖动全程监控零件随动，滑脱即停。
- **实现方式**：规则脚本
- **粒度**：原子
- **失败策略**：continue
- **是否已封装**：是
- **依赖项**：grasp, manipulation._drive_along, Gripper
- **输入**：
    - `name`：被拉零件名
    - `delta`：位移向量 (dx, dy)
    - `to_target`：可选：绝对目标 xy（优先于 delta）
    - `release`：结束后是否原位释放
    - `speed`：拖拽速度 (m/s)
- **输出**：
    - `moved_m`：零件沿方向实测位移 (m)
- **前置条件**：零件存在
- **后置条件 / 成功标准**：位移 >= 70% 请求量（以 moved_m 判定）

#### `wipe`
- **功能描述**：擦拭：并指成刃，轻压贴面后在目标面上沿直线扫掠，全程监控 pad 接触力处于目标力带（既保证贴实又不压穿/挑翻被擦面），可作触垫预清洁等语义。
- **实现方式**：规则脚本
- **粒度**：原子
- **失败策略**：continue
- **是否已封装**：是
- **依赖项**：Gripper.close_to_span, motion.move_eef
- **输入**：
    - `at`：擦拭区域中心 xy
    - `direction`：扫掠方向单位向量
    - `length`：扫掠长度 (m)
    - `z`：EEF 扫掠高度 (world z，通常=面高+pad 偏移)
    - `f_band`：(f_lo, f_hi) 目标接触力带 (N)
- **输出**：
    - `force_mean`：扫掠全程平均接触力 (N)
    - `force_max`：扫掠峰值接触力 (N)
    - `swept_m`：实测扫掠位移 (m)
- **前置条件**：场景已加载
- **后置条件 / 成功标准**：接触力处于力带且扫掠完成（以 force_max 判定）

## 学习类技能说明（扩展类插销）

`peg_insert` 提供三种 `mode`：`press`（到点压入）、`thread`（底部导向螺旋下降 + 抗卡摆动，规则闭环，A/B/C 生产链使用）、`policy`（学习策略逐步驱动 EEF）。`policy` 模式由 `simbench/skills/learned/` 实现：

- **观测**（`insert_env.build_obs`，8 维，场景无关）：销底相对孔心的横向偏差(2)、销底距座面高度(1)、销轴姿态偏差(2)、销与环境接触力/卡滞信号(1)、EEF 相对孔心横向偏差(2)。
- **动作**（`insert_env.action_to_delta`，3 维）：EEF xyz 增量（每步 ≤1mm）。
- **奖励**：基于势函数的下降/对齐整形 − 接触力(卡滞)惩罚 − 姿态偏差惩罚 − 步长惩罚 + 落座奖励；显式覆盖“接触、卡滞、姿态偏差”反馈。
- **训练**（`train_insert.py`）：`--algo ppo`（强化学习，torch 自包含 PPO+GAE）与 `--algo bc`（模仿学习，克隆脚本专家）。产出 `.pt` 检查点，`peg_insert(mode='policy')` 自动加载 `checkpoints/peg_insert.pt`。
