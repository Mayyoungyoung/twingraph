# v5 使用与实验复现

本轮使用 Python 3.10、MuJoCo 2.3.2、Torch 2.1.0+cu121 和 RTX 2080 Ti。
可在独立环境安装 `requirements-value-v5.txt`。实验实际依赖与硬件记录以
归档的 `runtime_environment.json` 为准。v5 不需要 torchvision 或相机图像。

## 业务接口

```python
from simbench.value.value_v5 import ValueScorer

value = ValueScorer(checkpoint="path/to/frozen/best.pt", device="cpu")
result = value.rank(state=current_state, plans=candidate_plans, k=4)
selected = result["top_k"]
```

`current_state` 遵守现有观测接口：`robot`、`objects`、`goals`。
`candidate_plans` 为 `PlanIR` 对象或 `PlanIR.to_dict()` 的 JSON。
返回原始候选标识、预测概率、完整可执行计划和绑定哈希；也返回字段分布
变化诊断与分阶段耗时。模型加载一次后可连续处理多个候选池。

输入仅在已知滑台、静止供料初态、固定技能接口范围内验证。不要把未见
任务的输入字段诊断忽略后直接宣称跨任务有效；新任务可重新训练。
已知初始关节路径包含全部路点，依赖实际抓持反馈的后续路径仍保留待求解
关系，交给数字孪生执行阶段解析。

## 完整实验命令

以下命令对应本轮已冻结的设计。`protocol.json` 中包含48个训练、12个验证、
12个测试配置以及4个系统配置。采集命令拒绝静默覆盖或重新采样失败配置。
正式复现先解压归档的源码与对应源码覆盖层，核对 `source.json`；Windows
Git 换行转换会改变原始字节哈希，不能用不匹配的源码绕过核对。

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl
python scripts/run_value_v5_collection.py \
  --protocol experiments/value_v5/protocol.json --splits train val --out results/v5/data
python scripts/run_value_v5_models.py \
  --protocol experiments/value_v5/protocol.json --data results/v5/data \
  --out results/v5/models --devices cuda:0 cuda:1
python scripts/export_value_v5_predictions.py validation \
  --data results/v5/data \
  --checkpoints results/v5/models/mlp_17/best.pt results/v5/models/mlp_29/best.pt \
    results/v5/models/mlp_43/best.pt results/v5/models/linear_17/best.pt \
  --expected-seeds $(seq 61200 61211) --test-seeds $(seq 61300 61311) \
  --n 12 --test-repeats 1 --out results/v5/evidence/validation_predictions.json \
  --selection-out results/v5/evidence/model_selection.json
python scripts/evaluate_value_v5.py freeze \
  --validation results/v5/evidence/validation_predictions.json \
  --out results/v5/evidence/thresholds.json --k 1 2 4 --primary-k 4
python scripts/run_value_v5_collection.py \
  --protocol experiments/value_v5/protocol.json --splits test --out results/v5/data \
  --selection results/v5/evidence/model_selection.json --thresholds results/v5/evidence/thresholds.json
python scripts/export_value_v5_predictions.py test \
  --data results/v5/data --selection results/v5/evidence/model_selection.json \
  --out results/v5/evidence/test_predictions.json
python scripts/evaluate_value_v5.py analyze \
  --predictions results/v5/evidence/test_predictions.json --freeze results/v5/evidence/thresholds.json \
  --out results/v5/evidence/test_metrics --bootstrap-samples 2000
python scripts/run_value_v5_system_batch.py \
  --protocol experiments/value_v5/protocol.json --selection results/v5/evidence/model_selection.json \
  --planner experiments/value_v5/planner_record.json --out results/v5/systems
python scripts/report_value_v5.py --run results/v5 --out results/v5/report
```

验证预测先完整保存到不含标签的评分文件，随后才连接执行结果。验证集
确定模型及分类阈值后固定；测试结果不能触发换模型。模型路径移动时，
导出器的 `--checkpoints name=path` 只允许保持原冻结文件哈希的重定位。

系统中的候选顺序来自当前助手作为 LLM 代理的明确记录。符号顺序由技能
编译器展开，模型选择 Top4，twin 真正执行这些候选，再在独立 target 中
执行选中程序。全量对照也真正执行全部12条候选，使用相同的选择规则。
`timing_rows.json` 与 `timing_rows_cold_start.json` 区分常驻模型和包含实际
模型加载时间的结果；并行批次墙钟时间另行保存。

## 回放与核查

```bash
python scripts/render_value_v5_execution.py \
  --run results/v5/systems/CASE/top_k --repeat 0 \
  --out results/v5/replays/CASE.mp4 --speed 4
```

`CASE` 使用实际系统输出目录名。回放读取真实记录的 `time_s/qpos/qvel/ctrl`
并核对哈希，只做渲染，不重新积分物理状态，也不产生新的成功标签。
完整执行参数、终验、初始状态和失败尝试均保留，可独立复核。

代码检查使用 `python -m pytest simbench/tests tests -q`。数据、权重、源文件
和执行轨迹由 `scripts/package_value_v5.py` 核对后归档。
