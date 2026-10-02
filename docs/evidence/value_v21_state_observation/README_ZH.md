# V21：状态观测下的自然候选覆盖试验

## 任务边界

本阶段把候选生成和执行期重观测改为从 MuJoCo 读取零件、夹具位姿，再施加预先声明的位置与偏航噪声。零噪声先运行；RGB-D 位姿恢复暂不参与本实验。真实物理接触与功能验收仍由独立仿真状态计算，不能用带噪观测替代标签。

原 V20 六项完整任务的一条零噪声试跑在清洁验收处停止：剩余污渍比例 0.2197，高于 0.05 的原任务阈值，装配尚未开始。这一条只定位前置截断，不作为装配正负覆盖统计，也不据此放宽清洁验收。

为观察装配本身的自然覆盖，另立完整的短任务版本 `printed_end_stop_mount_v21`：固定底座上安装挡板。每条完整 PlanIR 从检测、抓取、搬运、受力落座，执行到释放、退爪与独立稳定支撑验收。成功要求挡板位于 CAD 安装区域、由非夹爪接触支撑、已释放、没有明显穿透，并在 0.5 秒检查窗内保持。理想位姿误差、定位柱捕获和后续插销孔对齐不作为此短任务的成功门槛。它们仍是更长任务的后续约束；本任务标签不能混入 V20 六项任务训练。

## 冻结及结果

两次运行使用 L0、seed 2050、相同物理扰动命名空间 `online`，各从标签盲的正常生成器产生 48 条提案，按生成顺序取前 12 个**不同的挡板可执行程序**。请求中先冻结决策时观测、完整 PlanIR 和哈希，再逐条从新建场景真实执行。运行时指纹均为 `e2fd2042b2e4950c8e2a731d455fdc1bf0be09348b21c27f3d79b3cc216b9e42`。本指纹对应移出隐藏噪声偏移后的重采版本；先前的探索性试跑留在本机旧 `results/v21_state_*` 目录，不进入下列归档。

| 决策和执行观测 | 有效候选 | 成功 | 失败 | 池类型 |
|---|---:|---:|---:|---|
| MuJoCo 位姿，零噪声 | 12/12 | 10 | 2 | 自然混合 |
| MuJoCo 位姿，位置标准差 1 mm、偏航标准差 1° | 12/12 | 8 | 4 | 自然混合 |

零噪声的两个失败均发生在接近阶段，检查到夹爪与挡板的候选路径碰撞。带噪池还出现一条笛卡尔轨迹跟踪失败和一条释放后落在安装区域外的失败。两池的候选名顺序相同，但**PlanIR 参数随观测变化**，不能把 10/12 与 8/12 的差直接解释成固定计划的因果噪声效应。这只是一个布局上的探索性覆盖证据，尚不能训练或评价跨布局价值排序器。

审计已逐条核对冻结 PlanIR、决策观测、图哈希、运行时、实际标签与最终谓词。可长期保留的自包含记录：

- [零噪声 12 条样本](zero_noise_samples.jsonl)
- [带噪 12 条样本](noisy_samples.jsonl)

每行包含布局/任务/运行时、决策观测、完整候选 PlanIR、执行标签、首发失败原子调用及实现、阶段到达/完成、最终功能谓词、观测后端/噪声/随机试次、耗时、超时和软件异常字段。真实注入偏移及 RNG 标识单列为 `observation_random_state` 审计元数据，**不在模型可见的 `decision_observation` 中**。完整原始轨迹、各原子指标及逐步日志位于本机忽略 Git 的 `results/v21_state_*_r2` 目录。若需要精确复跑，应同时保存相应运行时源码版本；上述指纹只覆盖 `simbench` 源码、CAD 和控制器权重。

## 复现入口

先设置 `PYTHONPATH=.`，然后运行：

```bash
python scripts/collect_state_end_stop_v21.py --out results/v21_zero --seed 2050 --pool-n 48 --candidate-n 12 --level L0 --domain online
python scripts/audit_state_end_stop_v21.py --root results/v21_zero --out results/v21_zero/audit.json
python scripts/export_state_end_stop_v21.py --root results/v21_zero --out docs/evidence/value_v21_state_observation/zero_noise_samples.jsonl
```

带噪版本在首条命令后增加 `--position-noise-std-m 0.001 --yaw-noise-std-rad 0.017453292519943295`，并使用新的空输出目录。原六项任务可通过 `freeze_v20_natural_matrix.py` 的 `--observation-backend mujoco_state_pose` 与相同噪声参数预冻结，再由 `collect_v20_frozen_matrix.py` 采集；它仍使用原六项最终验收，不能把短任务的正例搬过去。
