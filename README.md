# TwinGraph

基于 MuJoCo 与 Franka Panda 的桌面装配研究项目，探索**面向后续任务的终态价值**与**有限数字孪生预算下的候选计划筛选**。

## 当前可运行内容

- 手动直线滑台装配：滑块沿轨插入、端挡与双销安装、约 100 mm 往复检查、手柄安装。大方桌与被动零件，只有 Panda 的 9 个执行器。
- 11 个原子技能独立近景视频：1600 × 1000、25 fps、固定正面偏侧视角；加粗路径、坐标轴和测量引线，底部显示技能名称与实时结果。场景为方块、障碍物、单孔插销座和短导轨。
- 插销模仿学习：训练代码与可直接运行的行为克隆权重随仓库提供。
- 结构契约验证：检查对象绑定、夹爪资源与规划数据依赖；尚不等于整链几何可行性证明。

任务层统一为 **11 个原子技能**，既包含执行动作，也包含输出信息的计算：检测、物体位姿估计、抓取位姿估计、路径规划、移动、抓取、放置、插入、压靠、测量、检查。搬运、上移、下移、回位、退出都属于移动参数，不新增节点。参见 [原子技能清单](docs/atomic-skills.md)。旧控制器与演示入口保留兼容，默认图谱和导出的原子技能清单只显示这 11 项。

![Panda 桌面滑台](docs/demos/tabletop.png)

## 查看成品

- [完整滑台装配视频](docs/demos/assembly.mp4)
- [小方块连续抓取](docs/demos/cube_grasp.mp4)
- [检测目标球](docs/demos/atoms/detect.mp4) · [三条候选路径](docs/demos/atoms/plan_path.mp4) · [插入近景](docs/demos/atoms/insert.mp4)
- [11 个原子技能新录像](docs/demos/INDEX.md)

仓库只保存这一套精选成品。中间录像、运行日志、训练数据和缓存放在被忽略的 `results/` 中。

## 运行

推荐 Python 3.10。在仓库根目录运行：

```bash
python -m pip install -r requirements.txt
export MUJOCO_GL=egl
python -m pytest simbench/tests -q
python -m simbench.assembly.atomic_demos --out results/atomic_closeup
python -m simbench.assembly.candidate_demo --out results/candidate_demo
python -m simbench.assembly.task --record --out results/tabletop/assembly
```

默认使用仓库内的 `simbench/assembly/checkpoints/insert_bc.pt`，无需先生成旧场景或下载旧模型。Ubuntu 无头录制需要可用的 EGL/OpenGL 驱动及 Noto CJK 字体；详见 [运行指南](docs/setup.md)。

## 文档

| 文档 | 内容 |
|---|---|
| [原子技能清单](docs/atomic-skills.md) | 11 项技能、输入输出与调用示例 |
| [架构设计](docs/architecture.md) | 技能粒度、规划多解、条件图与当前边界 |
| [滑台任务](docs/tabletop.md) | 产品结构、流程、控制与评估 |
| [独立演示](docs/standalone-skills.md) | 专用场景、可视化与重录 |
| [价值模块设计](docs/value-module.md) | 价值定义、反事实采样、训练与 Top-k 评估 |
| [清理记录](docs/maintenance.md) | 移除的旧代码与保留范围 |

验证记录见 [docs/evidence](docs/evidence)。`docs/demos/atoms/` 是当前 11 项接口重新录制的独立近景视频；完整装配与兼容组件录像沿用此前版本。历史小样本结果为完整装配扰动测试 8/8、插销策略新起点测试 12/12，适用条件写在任务文档中；它们不是现实机器人或新价值模块的性能成绩。

上轮接口整理回归：31 项测试、30 个独立组件执行、种子 0 完整装配（108 次调用）、种子 1/19 扰动装配、同初态两种抓取前缀均通过；另重录抬升组件确认字幕和轨迹显示。详见 [本轮验证记录](docs/evidence/interface-refactor/README.md)。

当前 11 项接口版本：44 项回归测试通过；原 30 个兼容组件演示通过；完整装配通过支撑放置检查。种子 1/19 扰动装配也通过。最新定义以 [atomic-skills.md](docs/atomic-skills.md) 为准，[验证记录](docs/evidence/atomic-interface.json) 单独保存。

近景演示：11/11 执行成功；检查计算技能不推进仿真时间、装饰不改变物理状态，并核对视频首帧、执行中帧与末帧。详见 [录制验证](docs/evidence/atomic-closeup/README.md)。
