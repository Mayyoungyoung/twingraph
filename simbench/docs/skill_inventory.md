# simbench 原子技能清单（Skill Inventory）

> 由 `python -m simbench.skill_inventory` 从 `SkillRegistry` 契约自动生成；请勿手工编辑。

分层可组合技能库：**L0** 原语（`MjContext` / `CartesianController` / `Gripper`）→ **L1** 原子技能（执行/规划/过渡/扩展四类）→ **L2** 契约与门控（`SkillSpec` + `SkillRegistry.run`）→ **L3** 组合（`run_chain` 与 planner/executor 的 plan→exec 工件握手）。

**技能总数：17**　执行类：8　规划类：2　过渡类：4　扩展类：3

## 汇总表

| 技能名称 | 类别 | 功能描述 | 输入 | 输出 | 实现方式 | 依赖项 | 已封装 | 验证方式 / 成功标准 |
|---|---|---|---|---|---|---|---|---|
| `detect` | 执行类 | 零件检测：在失败模型下感知零件位姿（位置+偏航），可注入高斯噪声/漏检/误检，输出观测供 plan_grasp_pose 消费。是执行链的感知入口。 | `part`, `faults`, `step`, `retries` | `found`, `pos`, `yaw`, `outcome` | 规则脚本 | perception.detect_part, faults.FailureModel(可选) | 是 | 检测到则 outcome!=miss 且 pos 非空 |
| `grasp` | 执行类 | 抓取：检测/估计（或消费 plan_grasp_pose 工件）→ 接近→ 下降 → 闭爪 → 举升校验的闭环抓取。前置零件存在，后置以举升量判定抓稳。是操作链的核心执行技能。 | `part`, `grasp_pose`, `press`, `lift`, `tol`, `squeeze`, `approach_yaw` | `ok`, `held_dist` | 规则脚本 | perception.estimate_grasp_pose, motion.move_eef, Gripper.close_on_part, plan_grasp_pose(可选) | 是 | 零件被抬升 >= VERIFY_LIFT（以 ok 判定） |
| `insert` | 执行类 | 插入装配：mode='press' 到点压入并保压；mode='thread' 底部导向螺旋下降（比例纠偏+抗卡摆动），把持件装入孔/轴。前置仍持件。是插销/套轴等装配的执行原语（学习版见扩展类 peg_insert mode='policy'）。 | `part`, `mode`, `target_eef`, `ref_axis`, `to_z`, `half` | `ok`, `bot_z` | 优化 | manipulation.insert, perception.part_meta, CartesianController.move_eef | 是 | 零件底面到达 to_z 附近（thread 模式） |
| `inspect` | 执行类 | 质量检测：对（已就位的）零件做 seat 几何判定——测量 xy 位置误差、z 高度误差与倾斜角，与容差比较给出 OK/NG 结论，可叠加传感器测量噪声。用于阶段终检或中段纠偏触发。 | `part`, `target_xy`, `target_z`, `xy_tol`, `z_tol`, `tilt_tol`, `noise_std` | `xy_err`, `z_err`, `tilt`, `ok` | 规则脚本 | MjContext.obj_pos, MjContext.obj_tilt | 是 | metrics 含 xy_err/tilt 且 ok 反映容差判定 |
| `move` | 执行类 | 末端移动：按规划航点风格（direct/safe_z/clearance）把 EEF 移动到 world xyz 目标，可选梯形速度顺滑。是执行类的通用位移原语，消费 plan_path 的航点风格。 | `to`, `style`, `tol`, `gain`, `max_speed`, `smooth` | `reached`, `eef` | 运动规划 | motion.move_eef, planning.plan_path, CartesianController.move_eef | 是 | EEF 与目标距离 <= tol（以 reached 判定） |
| `place` | 执行类 | 放置：把持件搬运到目标 xy/z 并释放落位。基于持件时实测的 EEF-零件偏移推导落点，抵消抓取检测噪声；支持对中/低空搬运/腕部扶正/多种释放策略。前置仍持件，后置以落位 xy 容差判定。 | `part`, `at`, `target_z`, `align`, `live_align`, `release`, `carry_style`, `low_carry` | `ok`, `pos` | 规则脚本 | manipulation.place, motion.move_eef, Gripper.open | 是 | 零件 xy 与目标距离 < 0.012m（以 ok 判定） |
| `push` | 执行类 | 推动：并指成刃，从零件背后沿方向闭环推抵，实时监测零件随动位移，卡滞即停（不硬磨）。用于纠偏/送料等非抓持的水平位移。以位移达请求量 70% 判定成功。 | `part`, `delta`, `to_target`, `push_z`, `speed` | `ok`, `moved` | 规则脚本 | manipulation.push, motion.move_eef, Gripper | 是 | 零件沿方向位移 >= 70% 请求量（以 ok 判定） |
| `transport` | 执行类 | 搬运：持件沿给定航点（或到目标点的规划航点）低速平移到目标上方悬停，供后续 place/insert 精降。全程监控仍持件，滑脱即失败。衔接规划与放置的搬运腿。 | `part`, `waypoints`, `to`, `style`, `carry_speed`, `tol` | `ok`, `held` | 运动规划 | CartesianController.move_eef, motion.move_eef, plan_path(可选) | 是 | 搬运后 EEF-零件 xy 距离 < 0.03m（以 held 判定） |
| `plan_grasp_pose` | 规划类 | 抓取位姿估计（多候选）：由检测观测+零件元数据生成 n_yaw 个接近角候选抓取位姿，按接近线碰撞/零件类型做可行性过滤，按行程+偏航失配+碰撞惩罚评分，排序选最优作为抓取工件（保留全部候选分支用于归因/数据导出）。 | `part`, `detect_result`, `noise_std`, `n_yaw`, `obstacles`, `yaw` | `grasp_pose`, `n_candidates`, `candidates` | 运动规划 | planning.grasp_pose_candidates, perception.part_meta, perception.detect | 是 | 至少一个可行候选（否则 ok=False，估计失败） |
| `plan_path` | 规划类 | 路径规划（多候选）：由起点到目标生成多个航点候选（高度变体×侧向 via 偏移），对每个候选做线段-AABB 碰撞门过滤，按路径长+航点数评分，选最优可行路径；全碰撞时换 style 重规划。产出 waypoints 工件供 move/transport 消费。 | `to`, `cur`, `style`, `lift`, `safe_z`, `obstacles`, `via_offsets` | `waypoints`, `style`, `score`, `n_candidates`, `candidates` | 运动规划 | planning.path_candidates, planning.check_path, planning.seg_aabb_hit | 是 | 至少一个无碰撞候选（否则 ok=False，需重规划） |
| `approach` | 过渡类 | 抓取前接近：先到抓取点上方悬停，再垂直下降至抓取高度（不闭合）。与 grasp(approach_direct=True) 组合即标准两段式抓取接近，避免 grasp 内部 safe_z 二次抬升的下降-上升-抽搐。 | `part`, `grasp_pos`, `hover_lift`, `tol` | `eef_z` | 规则脚本 | perception.estimate_grasp_pose, motion.move_eef | 是 | — |
| `pre_align` | 过渡类 | 下降前预对齐：在当前位置高度上，将持件（或空爪 EEF）对中到目标 xy，供后续垂直下降/放置直接衔接（gain/迭代沿用 place 实测配方）。 | `at`, `part`, `tol`, `max_iters` | `resid` | 规则脚本 | CartesianController.move_eef | 是 | — |
| `retreat_lift` | 过渡类 | 放置/释放后垂直抬升退避：从当前 EEF 位置爬升指定高度并静置，避免 pads 扫掠已就位零件。 | `height`, `tol`, `settle_steps` | `lift_m` | 规则脚本 | motion.move_eef, settle | 是 | EEF 爬升量 >= height（skill 以 lift_m 度量判定） |
| `return_home` | 过渡类 | 回初始位：开爪（可选）+ 关节空间回 HOME。可选先爬升到安全高度再回扫，规避关节回扫扫飞已装配件（Task B 实测教训）。 | `open_gripper`, `retreat_z` | `eef` | 规则脚本 | Gripper.open, CartesianController.home, motion.move_eef | 是 | — |
| `peg_insert` | 扩展类 | 插销：把持件插入指定孔口。mode='press' 为到点压入，mode='thread' 为底部导向螺旋下降（抗卡摆动，规则闭环），mode='policy' 由学习策略（模仿学习/强化学习）输出插入动作序列，输入孔心 xy 与孔底 z，奖励显式考虑接触、卡滞与姿态偏差。 | `name`, `hole_xy`, `to_z`, `mode`, `policy`, `max_steps` | `depth_m`, `bot_z` | 规则脚本/强化学习/模仿学习 | grasp, manipulation.insert, skills.learned.insert_env, skills.learned.load_policy (mode=policy) | 是 | 零件底面到达 to_z 附近（以 depth_m 判定） |
| `pull` | 扩展类 | 拉动：抓持零件后沿指定方向向机器人侧拖拽（与 push 反向的闭环位移技能）。若未持件则先轻咬合抓取并验证举升；拖动全程监控零件随动，滑脱即停。 | `name`, `delta`, `to_target`, `release`, `speed` | `moved_m` | 规则脚本 | grasp, manipulation._drive_along, Gripper | 是 | 位移 >= 70% 请求量（以 moved_m 判定） |
| `wipe` | 扩展类 | 擦拭：并指成刃，轻压贴面后在目标面上沿直线扫掠，全程监控 pad 接触力处于目标力带（既保证贴实又不压穿/挑翻被擦面），可作触垫预清洁等语义。 | `at`, `direction`, `length`, `z`, `f_band` | `force_mean`, `force_max`, `swept_m` | 规则脚本 | Gripper.close_to_span, motion.move_eef | 是 | 接触力处于力带且扫掠完成（以 force_max 判定） |

## 技能明细（契约）

### 执行类（exec）

#### `detect`
- **功能描述**：零件检测：在失败模型下感知零件位姿（位置+偏航），可注入高斯噪声/漏检/误检，输出观测供 plan_grasp_pose 消费。是执行链的感知入口。
- **实现方式**：规则脚本
- **失败策略**：retry
- **是否已封装**：是
- **依赖项**：perception.detect_part, faults.FailureModel(可选)
- **输入**：
    - `part`：零件名（body）
    - `faults`：可选 FailureModel（漏检/误检/噪声源）
    - `step`：可选步序号（供故障模型归因）
    - `retries`：漏检重试次数（默认 0）
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

#### `insert`
- **功能描述**：插入装配：mode='press' 到点压入并保压；mode='thread' 底部导向螺旋下降（比例纠偏+抗卡摆动），把持件装入孔/轴。前置仍持件。是插销/套轴等装配的执行原语（学习版见扩展类 peg_insert mode='policy'）。
- **实现方式**：优化
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
- **功能描述**：质量检测：对（已就位的）零件做 seat 几何判定——测量 xy 位置误差、z 高度误差与倾斜角，与容差比较给出 OK/NG 结论，可叠加传感器测量噪声。用于阶段终检或中段纠偏触发。
- **实现方式**：规则脚本
- **失败策略**：continue
- **是否已封装**：是
- **依赖项**：MjContext.obj_pos, MjContext.obj_tilt
- **输入**：
    - `part`：被检零件名
    - `target_xy`：名义 world xy
    - `target_z`：可选名义 world z
    - `xy_tol`：xy 容差 (m)
    - `z_tol`：z 容差 (m)
    - `tilt_tol`：倾斜容差 (deg)
    - `noise_std`：可选测量噪声 std (m)
- **输出**：
    - `xy_err`：xy 误差 (m)
    - `z_err`：z 误差 (m)
    - `tilt`：倾斜角 (deg)
    - `ok`：判定结论
- **前置条件**：`part_exists()`
- **后置条件 / 成功标准**：metrics 含 xy_err/tilt 且 ok 反映容差判定

#### `move`
- **功能描述**：末端移动：按规划航点风格（direct/safe_z/clearance）把 EEF 移动到 world xyz 目标，可选梯形速度顺滑。是执行类的通用位移原语，消费 plan_path 的航点风格。
- **实现方式**：运动规划
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

#### `transport`
- **功能描述**：搬运：持件沿给定航点（或到目标点的规划航点）低速平移到目标上方悬停，供后续 place/insert 精降。全程监控仍持件，滑脱即失败。衔接规划与放置的搬运腿。
- **实现方式**：运动规划
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
