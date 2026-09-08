# 安装与复现

在仓库根目录运行，推荐 Python 3.10。已验证的环境为 MuJoCo 2.3.2、NumPy 1.26.4、SciPy 1.15.3、Torch 2.3.1；依赖见根目录 requirements.txt。

```bash
python -m pip install -r requirements.txt
export MUJOCO_GL=egl
python -m pytest simbench/tests -q
```

Ubuntu 无头渲染需要可用的 EGL/OpenGL 驱动。字幕使用 `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`，对应 `fonts-noto-cjk` 系统字体包。视频由 imageio-ffmpeg 编码。只有查看已录视频时不需要安装仿真依赖。

## 装配

```bash
python -m simbench.assembly.task --record --out results/tabletop/assembly
python -m simbench.assembly.evaluate --seeds 0,1,2,3,4,5,11,19 --out results/tabletop/evaluation
```

默认权重为仓库内的 `simbench/assembly/checkpoints/insert_bc.pt`。`--policy <路径>` 可指定另一个兼容的行为克隆模型。任务输出完整视频、步骤、验收和物理快照；不再从完整装配中自动切出旧的组件视频。

任务同时输出 `atomic_skills.json`（10 个原子技能）及 `candidates.json`：包含所有抓取候选，以及所选抓法在实际抓持状态下生成的多条搬运路线。`status=unknown` 不能解释为不可行。以下命令从同一初态复现两种方块抓取前缀，并将候选输入和仿真结果分开保存：

```bash
python -m simbench.assembly.candidate_demo --out results/candidate_demo
python -m simbench.assembly.graph --out results/skill_graph.json
```

## 10 个原子技能近景演示

```bash
python -m simbench.assembly.atomic_demos --out results/atomic_closeup
python -m simbench.assembly.atomic_demos --only plan_path move --out results/path_closeup
python -m simbench.assembly.atomic_demos --no-record --out results/atomic_checks
```

默认录制全部 10 项，也可以指定技能子集。每次输出到独立目录；输出包括 `skills/*.mp4`、`previews/*.png`、准备场景和 `verification.json`。原有九段近景采用 1600 × 1000，新增擦拭采用 1920 × 1200，均为 25 fps、固定近景，不切镜头。计算技能用 5 秒展示计算结果，不推进仿真时间。

准备阶段调用原有控制器；视频从指定技能的就绪状态开始。例如抓取从方块两侧闭爪开始，放置从已到达支撑面的释放开始，搬运和接近仍属于移动。插入使用已有行为克隆策略。

默认技能列表及输入输出见 [原子技能清单](atomic-skills.md)，直接播放见 [视频索引](demos/INDEX.md)。旧单独放置入口 `atomic_demo` 继续兼容。

## 兼容组件演示

```bash
python -m simbench.assembly.standalone_demos --record --out results/standalone_skills/all
python -m simbench.assembly.standalone_demos --cube-grasp --out results/standalone_skills/cube_example
python -m simbench.assembly.deliver_demos --sources results/standalone_skills/all
```

只重录一个组件：

```bash
python -m simbench.assembly.standalone_demos --groups pin --only guarded_descent --record --out results/standalone_skills/guard
```

场景组包括 `perception cube obstacle pin rail`。准备动作先在物理仿真中完成，录制时执行指定组件。组合抓取视频串联接近、闭爪与抬升，不新增能力类型。

## 重新训练插销策略

先运行装配命令，得到通过真实抓取与搬运产生的 `pin_left_checkpoint.pkl`：

```bash
python -m simbench.assembly.learning results/tabletop/assembly/pin_left_checkpoint.pkl --episodes 24 --out results/tabletop/learned_insertion
python -m simbench.assembly.learning results/tabletop/assembly/pin_left_checkpoint.pkl --evaluate-only --test-seed 991235 --out results/tabletop/learned_insertion
```

训练程序生成专家演示、模型和报告。这是模仿学习，不是强化学习。重新训练的结果需独立评估，不能自动沿用随仓库模型的历史成功率。快照包含场景路径和几何校验，不应直接跨不同路径/模型加载别处的旧快照；在当前检出目录重新生成。

## 目录

- `simbench/assembly/`：当前任务、控制、底层组件、学习和演示。
- `simbench/core/sim_context.py`：仿真上下文与完整状态保存/恢复。
- `simbench/assets/panda/`：实际被场景引用的 Panda 模型资源。
- `simbench/tests/`：当前实现的回归测试。
- `docs/demos/`：精选成品视频与索引。
- `docs/evidence/`：当前测试及历史评估证据。
- `results/`：本机重新生成的实验数据，Git 忽略。

## 含擦拭的完整成功与失败演示

```bash
python -m simbench.assembly.full_demos --record --out results/product_success
python -m simbench.assembly.full_demos --record --failure --out results/product_failure
python -m simbench.assembly.full_demos --record --wipe-only --out results/wipe_final
python -m simbench.assembly.graph_figure --out docs/skill-graph
```

完整视频 1920 × 1200、25 fps、明确标注 2× 播放，固定正面略向下镜头，无切镜头。擦拭独立演示为 1×。`--failure` 只将右侧插销的规划目标沿 X 偏移 8 mm，不改变成功阈值、不强制失败标志；若实际意外通过，程序反而报错。检测到失败时停止后续装配。

完整演示在原装配前加工具抓取、底座擦拭与工具归还。插销选用已有接触反馈分支；旧行为克隆插销策略仍保留供独立演示和研究。新增擦拭策略可通过 `python -m simbench.assembly.wiping` 从 24 条程序化专家轨迹重新训练，随仓库提供数据、权重及拟合报告。图形导出另需系统 Graphviz（`dot`）。
