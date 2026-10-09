# SSH 901 最终项目与 Panda 实验流程

主项目集中在 `/home/jia/twingraph/`：`code/` 保存最后使用的物理运行时、技能库、价值模块和新增真机接口；`experiments/final/` 保存原始数据、扰动训练、测试和新实验；`experiments/videos/` 保存演示视频；`experiments/historical/` 保存旧实验及删除校验凭据；`runtime/twin/` 保存已验证的 MuJoCo 环境；`maintenance/` 保存整理清单。`code/results` 是到 `experiments/final` 的相对符号链接。

旧版本按明确目录清单逐个归档核验后删除。没有改动 `~/robot_panda` 原有控制代码、Miniconda 依赖或 V33 冻结 `simbench`。旧 Git 检出的缓存历史和当时工作源码另行存档；部分克隆中没有下载过的 Git 历史对象仍需从原 GitHub 仓库获取。

新入口安装为 `/home/jia/robot_panda/src/robot/scripts/twingraph_task.py`，通过 `real_robot.panda_adapter` 使用本机已安装的 `panda_py`。末端使用 Panda 原生基座坐标；机器人安装平面为桌面 `z=0`。零件坐标和偏航见 [坐标文档](real_robot_coordinates_v34.md)，机读示例见 `real_robot/config/nominal_layout_4000.json`。这些是可摆放的名义 CAD body 坐标，不能直接当成机械臂末端目标。

## 一次完整实验

1. 按坐标摆放并固定夹具。原布局导轨底座需要 **6 mm 垫块**。如果导轨底座直接放桌面，把其 body Z 改为 6 mm，重新生成方案与精验。原 V33 从已清洁条件开始；当前装配验证不包含真机擦拭。
2. 建立实测布局 JSON：核实 Panda 原生坐标轴、实际初始关节及夹爪开口、九个物体位姿、质量和接触参数。保留原始测量来源、时间及不确定度。不要仅把名义文件的 `measured` 改成 true。
3. 在同一实测布局中生成候选、用冻结价值模块取 Top3，并逐个新建孪生精验，直到首个有效成功。保留失败候选；若 Top3 全失败，停止并调整布局/方案。默认 LLM 模式需要现有认证或外部响应中继；901 上可直接运行 `--planner grounded` 使用几何约束候选池，其输出明确标为 grounded。
4. 从成功方案编译分阶段硬件程序，校准 FCI EE → TCP 变换、抓持宽度/力和接触参数，审核整段 Cartesian 路线及机器人连杆/夹持物避碰。仿真关节路径转为可审阅的 Cartesian 路线提案；不在真机上重放仿真关节角。
5. 先预演。确认实测起始状态、配置、路线和各阶段观测后，使用预演给出的精确 arming token 执行。阶段验收由新鲜实测 JSON 提供，最终接受装配必须通过独立实测条件。真机试验结果单独记录，不能用孪生通过替代。

当前真机接口执行经过审核的有限 Cartesian 程序，并在阶段边界等待实际测量；尚未接入在线视觉位姿识别或像仿真一样自动修正接收件路线。若实测零件或接收件超过配置容差，检查失败并停止，需使用新布局重规划/精验。运动代码本身不生成装配通过标记。

## 可直接运行的名义孪生示例

以下命令不连接机器人，输出目录必须是新的空目录：

```bash
cd /home/jia/twingraph/code
/home/jia/twingraph/runtime/twin/bin/python -m scripts.prepare_real_experiment_v34 \
  --layout real_robot/config/nominal_layout_4000.json \
  --training results/v33_robust_top3_20260927/training \
  --planner grounded --allow-nominal-layout \
  --out results/real_v34_next_demo
```

输出 `request.json`（布局与候选冻结输入）、`selection.json`（Top3 顺序与分数）、`validation/*/result.json`（新鲜精验）、`selected_feedback_program.json`（原子技能与接收件记录）、`reference_schedule.json`（完整末端参考轨迹）和文件哈希。实测实验移除 `--allow-nominal-layout`，改用实际测量文件；输出仍需通过硬件编译及预演。

```bash
python3 ~/robot_panda/src/robot/scripts/twingraph_task.py compile \
  --reference results/real_v34_release_20261009/reference_schedule.json \
  --calibration real_robot/config/panda_calibration_template.json \
  --output results/real_v34_release_20261009/hardware_schedule_review.json
python3 ~/robot_panda/src/robot/scripts/twingraph_task.py run \
  --schedule results/real_v34_release_20261009/hardware_schedule_review.json \
  --calibration real_robot/config/panda_calibration_template.json \
  --evidence-root results/real_v34_release_20261009 \
  --report results/real_v34_release_20261009/hardware_dry_run.json
```

模板未标定时，预演输出 `blocked_reference` 并列出待填写项。`run` 默认预演，不创建 Panda 连接。`capture-state` 是现场 FCI 状态读取命令，会连接机器人但不启动运动；仅在操作者准备采集时使用：

```bash
python3 ~/robot_panda/src/robot/scripts/twingraph_task.py capture-state \
  --hostname 192.168.1.2 --output /home/jia/twingraph/experiments/panda_state.json
```

物理执行需要同一 schedule/config 的 `--execute --arm <预演令牌>`。令牌绑定文件内容；修改程序或标定后必须重新预演。接口提供有界 Cartesian 运动、夹爪抓取/释放、轴向接触推进、接触检测后有界保持、力/力矩、工作空间、反馈陈旧、跟踪误差和超时检查。它不会自动回零、解锁或恢复错误。软件检查不能替代现场确认、机器人本身的保护和整机避碰评审。

阶段检查开始后，可由现场观测服务写入配置中的 `observations/*.json`，也可用 `scripts.record_panda_observation` 登记操作者刚取得的实际测量。输入格式为 `objects`（每个对象的 `pose_world` 4×4 矩阵）、`scalars`（如实际插深，单位米）及 `flags`（操作者实际确认的布尔结果），按当前 checkpoint 需要填写。程序不复制预期值当作实测值。`--measured-now` 仅用于输入确实是刚完成的现场测量时：

```bash
python3 -m scripts.record_panda_observation \
  --schedule results/actual_trial/hardware_schedule.json \
  --command-id carriage_01_observed --input /tmp/current_measured_values.json \
  --output results/actual_trial/observations/carriage_01.json --measured-now
```

`command-id` 与输出路径取当前 schedule 对应检查命令；示例名称需要替换。旧时间戳、缺失项、不符合验收条件或超过等待时间的测量不会通过。自动视觉测量生产者尚需连接现场相机与标定。

整理后可用以下只读命令核查冻结实验和归档哈希：

```bash
cd /home/jia/twingraph/code
/home/jia/twingraph/runtime/twin/bin/python -m scripts.verify_901_project --full-archives
```
