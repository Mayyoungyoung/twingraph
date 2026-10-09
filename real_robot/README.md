# Panda 真机执行接口

本目录与冻结的 `simbench` 分开，保持 V33 实验运行时指纹不变。代码匹配 SSH 901 上实际检查到的接口：系统 Python 3.8、`panda_py 0.8.1`、`panda_py.Panda('192.168.1.2')`、`libfranka.Gripper` 和 `CartesianImpedance`。不需要 ROS 话题，导入及默认 dry-run 均不连接机器人。没有自动解锁、FCI 激活、recover、回 HOME、仿真关节直接回放或异常后自动松开夹爪。

## 当前交付边界

`panda_adapter.py` 已实现可执行的有限指令控制器：

| 指令 | 真机行为及验收 |
|---|---|
| `move_tcp` | 低速 Cartesian 阻抗运动；五次平滑曲线限定参考速度和加速度，实际 EE 位姿到达才通过 |
| `move_path_tcp` / `contact_path_tcp` | 有限完整 Cartesian 路线；逐点实测跟踪、全程力/力矩与时间限制，可检查整段实际插入深度 |
| `contact_move_tcp` | 沿标定轴低速插入；实际到达位姿且实际投影深度达到 `min_progress_m`，力/力矩超限立即失败 |
| `guarded_move_tcp` | 沿标定轴接近，连续实际反力样本达到阈值并满足最小深度才检测为接触；随后停止进一步前进 |
| `press_tcp` | 检测上述实际接触后固定有限参考位置，在指定实际力范围内保持 `hold_s`；失去接触不会继续盲推 |
| `gripper_open` / `gripper_grasp` | 调用实际夹爪 API，检查返回值及测量宽度；抓取还要求 `is_grasped` |
| `verify_pose` | 检查实际机械臂位姿 |
| `verify_observation` | 等待独立视觉或人工实际测量文件，检查物体位姿、插入深度等数值与实际功能验收 |

软件力/工作空间检查用于初次调试，不能代替机器人自带安全控制、实体障碍物检查与现场急停。当前未运行过真机运动，因此没有真机成功率。仿真成功不能证明实际装配成功。

编译器会从完整仿真 TCP 曲线生成整套阶段命令供审查。连续同阶段/同技能/同调用上下文的微步 servo 合并为一个完整曲线，避免把数千个仿真微步当成数千次真机指令。仿真 joint 路径使用 `control_samples` 的完整 EE 曲线，丢弃仿真关节角；几何简化最多 0.5 mm / 0.005 rad，并保留有实际尺度的往返/探测动作，新的时间参数由低速曲线计算。仿真路径需要实际坐标/工具变换及整条机械臂碰撞复核。采样缺失时保留 `unbound_joint_route`，不会把起终点直连成假通过路径；完整采样证明 TCP 静止的仿真 nullspace 动作只提案为待复核的位姿检查，丢弃关节姿态变化。未审查的接触控制、夹爪参数与阶段观测会使 `hardware_executable=false`。仿真嵌套 move/joint 包装事件只保留内层实际 primitive，避免执行两次。

## 两个 Python 环境

在服务器整理后的项目目录运行：

```bash
cd /home/jia/twingraph/code
# 方案生成、价值 Top3、独立孪生精验：Python 3.10 / Torch / MuJoCo
/home/jia/twingraph/runtime/twin/bin/python -m scripts.prepare_real_experiment_v34 --help
# Panda 原生库：系统 Python 3.8
/usr/bin/python3 -m real_robot.panda_adapter --help
```

只读取真机初始状态，不启动控制器、不移动：

```bash
/usr/bin/python3 -m real_robot.panda_adapter capture-state --hostname 192.168.1.2 --output results/panda_initial_state.json
```

该文件记录实际 `q`、`dq`、`O_T_EE`、FCI `F_T_EE`、已配置总负载、夹爪宽度及抓持状态。将实际初始关节和开口录入实际场景，再针对该场景重新生成/验证方案。机器人必须处于已由操作者正常使能的空闲状态；采集状态命令本身没有启用 FCI 的行为。

## 场景、坐标与工具

`config/nominal_layout_4000.json` 是实际摆放建议，坐标为 Panda 原生基座系；其数值来自 CAD 支承面及冻结 seed 4000。`docs/real_robot_coordinates_v34.md` 区分零件 CAD 原点、桌面支承面、夹取点及装配目标。此处默认机器人基座安装面与桌面齐平，孪生的 `T_base_world` 为平移 `[+0.56, 0, -0.8]`。V33 guide_base 使用 6 mm 支承垫，直接放桌面则所有对应装配目标需要下移 6 mm 并重验。

`panda_calibration_template.json` 中身份矩阵工具变换与工作空间仅是待填写占位值：

- `T_base_world`：孪生世界 → Panda 原生基座；实际场景的 `T_world_base` 必须是它的逆。
- `T_sim_eef_tcp`：仿真 `grip_site` → 任务采用的真实 TCP；仿真 grip_site 不等于 FCI EE。
- `T_ee_tcp`：FCI EE → 实际任务 TCP；发送的控制目标为 `T_base_world @ T_world_tcp @ inv(T_ee_tcp)`。
- `expected_F_T_EE` 与 `expected_total_payload_mass_kg`：与当前 FCI 工具/负载配置一致，程序不自行写入未知工具或负载。
- `workspace_base`：实际 EE 和 TCP 都必须位于经现场复核的范围内；本检查不等价于整条机械臂避障。

实际测量完成后填写带时区的 `measured_at_utc`、各项验证标志、阶段接触/夹爪配置及真实测量源。初始 `q` 必须与孪生记录相差不超过 0.02 rad、`dq` 不超过 0.01 rad/s，夹爪初始宽度相差不超过 2 mm 且没有已抓持物体；即使 EE 位姿相同，另一肘部构型也会被拒绝。

## 从孪生参考编译与检查

先将模板复制到本次试验专用标定文件，保留原始模板。以下路径是一次试验示例：

```bash
trial=results/real_v34_release_20261009
/usr/bin/python3 -m real_robot.panda_adapter compile --reference "$trial/reference_schedule.json" --calibration real_robot/config/panda_calibration_template.json --output "$trial/hardware_schedule_review.json"
/usr/bin/python3 -m real_robot.panda_adapter run --schedule "$trial/hardware_schedule_review.json" --calibration real_robot/config/panda_calibration_template.json --evidence-root "$trial" --report "$trial/hardware_dry_run.json"
```

初次输出有完整命令提案和明确阻塞项。每个 primitive 的接触/夹爪参数由真实测量给出，不能直接把仿真的 `force` 字段当作 `libfranka.Gripper.grasp` 的力。标定文件支持：

- `reviewed_event_ids`：已实际复核的原始 primitive ID。
- `reviewed_stage_ids`：整阶段路线已复核；可按阶段名称批量声明当前 trace 的逐段复核，仍需最终精确 schedule 的碰撞审查摘要。
- `joint_route_bindings[event_id]`：`{"reviewed": true, "use_trace_cartesian_proposal": true}` 使用经复核的完整 Cartesian 曲线；或显式 `commands` 覆盖该段路径。
- `contact_profiles[event_id]`：`reviewed:true` 以及接触指令类型、低速、实际进给轴、插入深度、接触阈值/保持力区间等字段。
- `gripper_profiles[event_id]`：`reviewed:true`、实际 `width_m`、`speed_m_s`，抓取另需 `force_n`。
- `stage_bindings[stage_id]`：可用经测量/复核的 `commands` 替换整阶段策略，保留阶段实际验收。
- `observation_checkpoints[stage_id]` 与 `final_postconditions`：实际物体姿态、深度范围及功能验收要求。

阶段输入应来自装配后的实际复测；目标发生变化时，更新场景/阶段命令并重验，不能把仿真中的反馈绑定结果冒充真实视觉反馈。

合并后的 servo ID 使用 `first_id_to_last_id`。长插入轨迹可选择 `contact_path_tcp` 并配置 `progress_axis_world` 与实测 `min_progress_m`；压装保持可改为 `press_tcp` 并提供实际接近目标、轴、触发力、保持区间和时长。完整路径的默认 waypoint 验收为 0.5 mm / 0.005 rad，避免较大的终点容差把细路径拐点省略。每条完整曲线都有独立有限超时，较长路径需要按现场复核结果划为合理阶段，不能无限提高超时或力阈值。

过长自由移动会自动分为最长 90 秒的有限段，保留完整路线和参考连续性；接触曲线保持为整段，使实际插入深度始终相对于该段的实际测量起点。初始调试模板明确提案最多 120 秒/指令、3600 秒/完整程序，均为硬上限且需要随标定一起现场复核。库默认值仍为 90 秒/1800 秒。编译输出的 `required_limits` 告诉操作者低速完整程序的实际有限预算；超过硬上限就需要改写并重验阶段方案。

## 独立阶段测量文件

控制器在阶段结束保持当前位姿，等待外部测量进程生成该 checkpoint 的 JSON。操作员也可以通过独立测量工具提供当前测量；代码没有自动视觉识别或读取夹具测头的实现。文件必须在 checkpoint 开始之后测量，默认仅接受 2 秒内的结果，并且绑定当前 task/command/source。

```json
{
  "task_id": "real_v34_release_20261009",
  "command_id": "stage_002_pin_left_observed",
  "source": "operator_measured_fixture",
  "measured_at_utc": "2026-10-09T15:20:30+08:00",
  "objects": {"pin_left": {"pose_world": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}},
  "scalars": {"pin_left_depth_m": 0.046},
  "flags": {"pin_left_bridge_passed": true}
}
```

示例矩阵/时间仅说明格式，必须替换为当前实测。`expected_objects` 使用孪生世界系；若测量在 Panda 基座系，需要用 `inv(T_base_world)` 转换。深度与功能字段要由实测/人工检查提供，不能从目标值复制。最终 `task_accepted=true` 仅在最后独立测量 checkpoint 通过时写入，运动队列完成本身不等于装配成功。

## 显式执行

完成实际标定及路线复核、重新编译并验证实际场景后，把该 schedule 的 `canonical_sha256` 写入专用标定文件的 `collision_review_schedule_sha256`。再次 dry-run 会给出与 schedule+calibration 同时绑定的 arm token。只有显式提供 `--execute --arm 'EXECUTE_PANDA:...'` 才会连接并运动；修改任何指令或标定值都会改变 token。

```bash
/usr/bin/python3 -m real_robot.panda_adapter run --schedule "$trial/hardware_schedule.json" --calibration "$trial/panda_calibration_measured.json" --evidence-root "$trial" --report "$trial/hardware_execution.json" --execute --arm 'EXECUTE_PANDA:<本次 dry-run 输出的 16 位摘要>'
```

程序先检查证据文件内容 hash，再检查 request/layout/candidate/plan/runtime/twin 的交叉绑定，要求孪生 `valid=true, success=true` 且没有超时、实际 layout 与初始机器人状态已测量。然后检查真实起始关节/速度/夹爪、工具负载及 EE 位姿，最后才启动控制器。错误、超时、反馈停止更新、力/力矩超限、夹爪失败或验收失败均停止流程，输出失败日志；失败后不自动重试、恢复或松开可能掉落的工件。

初次力阈值/阻抗/插入力必须按工件与工具测量。较大压入力需要相应的经审核刚度和允许跟踪偏差；`press_tcp` 是有界阻抗接触保持，不是经过真实零件辨识的专业插入搜索、力闭环压装或螺纹拧紧策略。Native 网络调用本身若阻塞，Python 的超时/停止请求受 SDK 和网络延迟限制，现场急停与机器人本体安全控制仍是必要条件。

验证命令：`python -m pytest tests/test_panda_real_adapter.py -q`。这些测试只使用假的机器人状态，不连接硬件。
