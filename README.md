# TwinGraph

基于 MuJoCo 与 Franka Panda 的桌面装配研究项目，探索**面向后续任务的终态价值**与**有限数字孪生预算下的候选计划筛选**。

## 当前演示

Panda 在大方桌上先抓取擦拭工具、清洁导轨底座、归还工具，再装配滑块、端挡、双销和手柄，并完成行程与终态验收。只有 Panda 的 9 个执行器，零件与擦拭工具都是被动物体。

- [完整成功视频](docs/demos/full_success.mp4)
- [完整失败视频：右销目标偏移 8 mm](docs/demos/full_failure.mp4)
- [擦拭近景视频](docs/demos/atoms/wipe.mp4)
- [10 个原子技能视频索引](docs/demos/INDEX.md)
- [技能图谱与条件解释](docs/skill-graph/README.md)

完整视频采用固定正面略向下镜头，1920 × 1200、25 fps，明确标注 2× 播放；底部显示当前技能。擦拭近景为 1×。没有网页前端。

## 技能与方法

当前 **10 个原子技能**：检测、物体位姿估计、抓取位姿估计、路径规划、移动、抓取、放置、插入、压靠、擦拭。搬运、上下移、回位和退出是移动参数；多抓法、多路线和控制策略是候选的不同参数实例。**测量与检查是辅助反馈和验收工具，不再是技能节点**，原有安全、接触与成功检查继续执行。

擦拭采用 24 条程序化专家轨迹训练的 RBF 轨迹模仿策略，配合接触力反馈，复用已有 IK 与伺服控制。当前模拟接触擦拭和几何覆盖，不模拟抛光材料去除。完整展示的插销使用已有接触反馈分支；旧插销行为克隆权重继续保留。

技能图谱由执行器共享的声明式契约生成，连线只表示部分数据或状态供给。完整绑定后的候选链仍需区分冲突、已通过必要检查与未知连续约束，再交给价值模块和数字孪生验证。

## 运行

```bash
python -m pip install -r requirements.txt
export MUJOCO_GL=egl
python -m pytest simbench/tests -q
python -m simbench.assembly.full_demos --record --out results/product_success
python -m simbench.assembly.full_demos --record --failure --out results/product_failure
python -m simbench.assembly.atomic_demos --out results/atomic_current
```

推荐 Python 3.10。无头录像需要 EGL/OpenGL 及 Noto CJK 字体；详见 [运行指南](docs/setup.md)。训练权重随仓库提供，中间文件放在 Git 忽略的 `results/`。

## 文档

| 文档 | 内容 |
|---|---|
| [原子技能](docs/atomic-skills.md) | 当前 10 项接口与输入输出 |
| [技能图谱](docs/skill-graph/README.md) | 共享契约、条件依赖与绑定 |
| [擦拭学习](docs/wiping.md) | 专家数据、轨迹拟合与接触执行 |
| [架构设计](docs/architecture.md) | 多解候选、图谱验证与边界 |
| [价值模块设计](docs/value-module.md) | 反事实数据、训练与 Top-k 评估 |
| [当前验证](docs/evidence/surface-assembly/README.md) | 回归、真实成功/失败与视频核验 |

历史验证保留于 `docs/evidence/`，包括旧 11 项接口与旧近景录像记录。它们不代表当前新增擦拭任务的鲁棒性统计。旧底层组件继续兼容；当前定义以上述 10 项为准。
