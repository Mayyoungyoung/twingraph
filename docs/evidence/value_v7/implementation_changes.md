# v7 实现改动记录

当前提交基线：`55707a4`；新分支：`research/value-v7-fulltask-view-ablation`。

## 已实现

- 新增 `simbench/value/observation.py`：`SensorConfig`、冻结观测和显式 `simulated_pose_sensor_proxy`。一次决策边界生成同一份位置+偏航带噪观测，写入 seed、噪声参数、版本和哈希。
- `Session` 快照/恢复保存冻结观测、污迹状态、阶段结果和行程记录；v5/v6 未安装观测时保持原行为。
- 候选几何 scratch session 复制冻结观测，初始抓取路径不再必须读 live target 的隐藏姿态；修正了连续 yaw 下的抓取宽度判断。
- 新增 `simbench/value/cleaning.py` 和 v7 场景中的 32 个可见污迹站点。污迹只在 pad 工作面与导轨表面有接触力、且产生切向移动时衰减；工具离开后独立清洁验收，静止按压和悬空经过不会通过。
- `wiping.surface_force` 只统计 `wipe_pad` 与指定表面的接触；清洁指标包含 initial/remaining amount、cleaning ratio、coverage、contact fraction、force error、peak force。
- `verify_stroke` 现在要求双向行程、横向偏移和峰值力约束，不再只使用 `max(x)-min(x)`。
- 新增 `stage_v7.py`、`full_task_v7.py`、`collect_v7.py`：候选是带真实执行参数的完整清洁—装配—把手后双向行程任务，支持 L0/L1/L2、12 候选、独立 target 扰动和断点文件。
- v7 pin 分支改用已有接触插入原子；旧 stage_v5/v6 默认分支保持原 guarded descent。
- 灰白棋盘格地板和双视角 task/top RGB-D 采集保留在 v7 场景，图像输入仍是 80×80，不把字幕或 rollout 结果喂给模型。
- 装配后的把手操作新增 `observe_execution_pose`：决策用冻结观测，执行期用单独的模拟跟踪反馈；开发重放确认原实现会去抓供应区旧把手位置。
- 销插入后增加持有状态下的 `align_axis` 闭环。严格验收仍拒绝放爪后约 5 mm/7° 的倾倒；尝试抬高环形座会阻塞真实插入（约 3.1 mm 残差），已回退并保留该失败。

## 尚未完成或仍需验证

- 目前开发 L0/L1 运行已证明清洁接触和阶段失败会被真实记录；24 次开发重放中清洁阶段通过 23 次、装配阶段通过 2 次、完整任务 0 次。pin 在脱离夹爪后的倾倒仍未解决，正式协议前必须完成足够的开发成功/失败覆盖，不能放宽验收掩盖它。
- 远程正式 128/32/64 场景和 9600 次完整任务尚未声称完成；未完成数量会在 `data_audit.json` 中如实列出。
- 本轮使用“真值经过统一传感器模型”的模拟感知代理，不声称已经训练真实 RGB-D 检测网络。

## 兼容性证明

`tests/test_system_v5.py` 在改动后仍为 10/10 通过。旧 v5/v6 数据、模型和视频未覆盖写入；只有新 v7 分支新增协议、技能注册和可选 v7 控制分支。
