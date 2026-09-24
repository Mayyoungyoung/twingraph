# TwinGraph

基于 MuJoCo 和 Franka Panda 的装配研究代码。当前版本用通用原子技能图表示完整候选计划，面向完整任务最终成功训练价值排序器，并用数字孪生执行验证。代码包含候选冻结、物理采集审计、价值训练和同池对照流程。

## 当前状态

按用户要求，2026-09-24 已停止 V20 独立布局采集。L0 首批 seed 2050–2069 的 960/960 次完整任务试验有效，只有 seed 2068 的 1 条计划通过全部六项最终验收，其余 19 个布局均为全负池。第二批 seed 2070–2089 已预冻结 960 条候选；停止时已取得 561 条有效结果、0 条完整成功，尚有 10 个布局不完整。验证与测试折缺少自然混合池，**未训练出 V20 完整任务价值模型，也没有价值 Top-4 对照结果**。详见 [V20 完整任务报告](docs/evidence/value_v20_full_flow/README_ZH.md)。

历史迭代结论见 [迭代记录](docs/ITERATION_HISTORY_ZH.md)。旧版原始数据、视频与价值模型权重已从当前仓库文件树清理；旧报告中的原始文件路径仅用于说明当时的实验，不代表文件仍可下载。装配控制器运行所需的少量权重保留在 `simbench/assembly/checkpoints/`。

## 运行

```bash
python -m pip install -r requirements.txt
export MUJOCO_GL=egl
python -m pytest simbench/tests -q
```

当前方法及接口见 [原子技能](docs/atomic-skills.md)、[技能图谱](docs/skill-graph/README.md)、[架构设计](docs/architecture.md) 和 [V20 完整任务报告](docs/evidence/value_v20_full_flow/README_ZH.md)。
