# Value v4：运行与复现

本轮使用原始滑台场景。机器人实际完成滑块和端挡后得到 checkpoint 2；再实际装好左销得到 checkpoint 3。偶数配置使用 checkpoint 2，奇数配置使用 checkpoint 3，每个配置只使用一个检查点。候选池固定为 `N=8`，每个候选执行两次；价值模块筛选 `K=4`。完整计划及其技能接口、物理参数和已物化的初始关节轨迹同时用于评分和执行，后续依赖抓持反馈的轨迹保留为显式延迟参数。

原始运行目录为 `/home/jia/twingraph-v4-20260916`，Python 为 `/home/jia/twingraph-v3-venv/bin/python`。原始冻结记录在测试前保存，相关提交为 `e94448d`。发布文件布局如下：

| 内容 | 发布位置 |
|---|---|
| 正式原始数据，包括未到达检查点的失败记录 | `datasets/value/stage_graph_v4.tar.gz` |
| 在线验证与独立执行记录 | `datasets/value/stage_graph_v4_execution.tar.gz` |
| 开发试验、失败和开发模型 | `datasets/value/stage_graph_v4_development.tar.gz` |
| 代码、必要资源、按哈希保存的源码版本 | `datasets/value/stage_graph_v4_source.tar.gz` |
| 四种表示 × 三个种子的 12 个模型 | `models/value/v4/<kind>_<seed>/` |
| 原冻结记录选定模型的副本 | `models/value/v4/best_graph_input.pt` |
| 原始冻结记录、指标、耗时、部署和图片证据 | `docs/evidence/value_v4/` |

本轮 `best_graph_input.pt` 是原冻结记录选中的 `sequence_17`（`sequence` 类型）的权重副本；模型类型以 `frozen.json` 的 `selected` 及模型元数据为准。

发布映射为 `release/models → models`、`release/datasets → datasets`、`release/evidence → docs/evidence/value_v4`。原始 `manifest.json` 和 `manifest.sha256` 放在 `docs/evidence/value_v4`，保持字节完全不变。核对清单时，将其中以 `evidence/` 开头的产物路径映射到 `docs/evidence/value_v4/`，其他路径仍相对仓库根目录解析。根层的 CSV、设备诊断和图表可以另有便捷副本，其清单所列原件仍保留。

从仓库根目录核对发布产物：

```bash
python3 - <<'PY'
import hashlib, json
from pathlib import Path, PurePosixPath
root = Path.cwd()
evidence = root / "docs/evidence/value_v4"
manifest = evidence / "manifest.json"
assert hashlib.sha256(manifest.read_bytes()).hexdigest() == (evidence / "manifest.sha256").read_text().split()[0]
records = json.loads(manifest.read_text())["artifacts"]
for row in records:
    rel = PurePosixPath(row["path"])
    path = evidence.joinpath(*rel.parts[1:]) if rel.parts[0] == "evidence" else root.joinpath(*rel.parts)
    assert path.stat().st_size == row["bytes"], path
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], path
print("发布清单及", len(records), "个产物的大小和 SHA256 已通过核对")
PY
```

## 环境和设备

以下命令使用 Linux Bash，均从相应源码副本根目录执行。已有模型的复现不需要运行训练命令。完整重采集、重训练应使用新的源码副本和空的 `results/v4`，保留原始运行目录及已发布证据。

```bash
export TG_PYTHON=/home/jia/twingraph-v3-venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl

"$TG_PYTHON" -c 'import sys,numpy,mujoco,torch; print(sys.version); print("numpy",numpy.__version__,"mujoco",mujoco.__version__,"torch",torch.__version__,"cuda",torch.version.cuda); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])'
"$TG_PYTHON" -c 'import matplotlib; print("matplotlib",matplotlib.__version__)'
nvidia-smi
```

核心依赖约束在 `requirements-value.txt` 和 `requirements.txt`；绘图另需 Matplotlib。已有远程环境直接复用；新环境需单独建立并记录上述版本。训练脚本将 `compact`、`graph` 放在物理 GPU 0，将 `port_mlp`、`sequence` 放在 GPU 1。单个命令设置 `CUDA_VISIBLE_DEVICES=0` 后，`--device cuda` 指向该进程可见的第一张 GPU。MuJoCo 采集没有 `--device` 参数；评分、评估和部署支持 `cpu` 或 `cuda`。改变设备、并发数或环境后，耗时属于新的测量条件。

## 使用已发布结果

只重画原始结果时，不加载模型文件，也不执行仿真：

```bash
"$TG_PYTHON" scripts/plot_value_v4.py \
  --evidence docs/evidence/value_v4 \
  --freeze docs/evidence/value_v4/frozen.json \
  --out results/v4_redraw
```

图中的三次模型种子训练共享同一批配置，不增加独立配置数。绘图程序使用保存的指标，不重新估计置信区间。

需要重新评分或执行时，先按发布的 `manifest.json`、SHA256 清单核对权重、归档及源码，再将正式数据解压到独立目录：

```bash
mkdir -p results/v4_replay
tar -xzf datasets/value/stage_graph_v4.tar.gz -C results/v4_replay
```

该归档包含 `data/`，因此解压后数据路径为 `results/v4_replay/data`。源码归档中的 `source_index.json` 说明每份 `source.json` 对应的源码覆盖目录；正式源码必须逐文件匹配其 SHA256。若需要恢复历史源码，在另一份源码副本中操作。开发记录中明确缺失的历史源码不能据此声称精确复现。

原始 `frozen.json` 保存了远程权重的**绝对路径**。迁移机器或数据、权重路径时，保留原记录，创建新的本地冻结文件来绑定同一批已有权重。下面的命令只读取保存的训练/验证结果和权重，不训练模型，不读取测试结果：

```bash
"$TG_PYTHON" scripts/evaluate_value_v4.py freeze \
  --models models/value/v4 \
  --freeze results/v4_replay/frozen.json \
  --test-seeds 51300 51301 51302 51303 51304 51305 51306 51307 \
  --deployment-seeds 51300 51301 51302 51303 \
  --checkpoints 2 3 --alternate-checkpoints --n 8 --k 4

"$TG_PYTHON" - <<'PY'
import json
from pathlib import Path
old = json.loads(Path("docs/evidence/value_v4/frozen.json").read_text())
new = json.loads(Path("results/v4_replay/frozen.json").read_text())
assert old["selected"] == new["selected"], "选定模型变化：停止复现"
assert old["test"] == new["test"] and old["deployment"] == new["deployment"]
before = {r["name"]: r for r in old["models"]}
after = {r["name"]: r for r in new["models"]}
assert before.keys() == after.keys()
for name in before:
    for key in ("sha256", "source_sha256", "split_sha256"):
        assert before[name][key] == after[name][key], (name, key)
print("同一组选定模型、权重、源码和划分哈希已核对；仅建立本地路径绑定")
PY
```

新冻结文件具有新的整体哈希。后续输出统一写入 `results/v4_replay/evidence`，并使用这一新冻结文件；不要把新冻结文件覆盖到原始证据中，也不要用它校验原始预测文件。下面各节的评估、计时和部署命令可将 `results/v4` 替换成 `results/v4_replay` 后使用。

仅检查价值模块的输入和 Top-K 输出，可执行：

```bash
"$TG_PYTHON" -m simbench.value.schema_rank \
  --inputs results/v4_replay/data/group_sliding_stage_assembly_51300_2/inputs.json \
  --checkpoint models/value/v4/best_graph_input.pt \
  --out results/v4_replay/top_k.json --k 4 --device cpu
```

该接口只读取候选输入和模型；输出含同一可执行图和计划的 `top_k`。运行它不会进行数字孪生验证或部署。

## 从空运行目录重新采集、训练和冻结

以下流程只在新源码副本、空的 `results/v4` 中运行。`train_value_v4.sh` 会写入模型目录，不能用作查看已有结果的命令。

```bash
bash scripts/collect_value_v4.sh
bash scripts/train_value_v4.sh
```

采集脚本先请求训练配置 `51100–51111`，再请求验证配置 `51200–51203`；均使用默认 `train` 扰动命名空间。它保存每个请求的成功或失败处置，并运行 `audit_value_v4_collection.py`。训练读取已完整完成的训练/验证组；四种模型类型为 `compact`、`port_mlp`、`sequence`、`graph`，种子为 `17、29、43`，每个模型运行 60 个 epoch。

12 个模型完成后，先冻结，再开始任何测试采集或测试指标计算：

```bash
"$TG_PYTHON" scripts/evaluate_value_v4.py freeze \
  --models results/v4/models --freeze results/v4/frozen.json \
  --test-seeds 51300 51301 51302 51303 51304 51305 51306 51307 \
  --deployment-seeds 51300 51301 51302 51303 \
  --checkpoints 2 3 --alternate-checkpoints --n 8 --k 4
```

冻结器依据已保存的验证集指标确定 `selected`，并记录权重、源码、划分和测试设计。已有冻结文件会被拒绝覆盖。不要因测试结果重新选择权重或重写原冻结记录。

## 测试采集与离线评估

```bash
if ! "$TG_PYTHON" -m simbench.value.schema_collect \
  --out results/v4/data --seed 51300 --groups 8 \
  --n 8 --repeats 2 --workers 8 \
  --checkpoints 2 3 --alternate-checkpoints \
  --split test --domain reference > collection_test.log 2>&1; then
  echo '采集包含失败请求；下面审计每个请求的实际处置。'
fi

"$TG_PYTHON" scripts/audit_value_v4_collection.py \
  --data results/v4/data --seed 51300 --groups 8 --split test

CUDA_VISIBLE_DEVICES=0 "$TG_PYTHON" scripts/evaluate_value_v4.py evaluate \
  --freeze results/v4/frozen.json --data results/v4/data \
  --out results/v4/evidence --device cuda
```

原运行请求训练/验证/测试配置数分别为 `12/4/8`，实际到达检查点为 `10/4/6`。训练配置 `51107、51109` 和测试配置 `51301、51303` 的准备失败记录均保留，没有补采替代配置。候选排名指标以实际到达检查点的配置为条件；端到端可行命中指标包含准备失败请求。墙钟超时单独保存为截尾记录，不能作为普通物理失败标签进入训练。审计未通过时应检查其具体错误，不跳过审计继续运行。

## 独立部署和模块计时

原运行的离线评分使用 CUDA；部署使用 CPU、4 个独立配置工作进程。部署只使用冻结的 `51300–51303`，进入 Top-K 的每个候选在线验证两次，每组最终选中的计划在独立 `deployment` 扰动命名空间执行两次：

```bash
"$TG_PYTHON" scripts/evaluate_value_v4.py deployment \
  --freeze results/v4/frozen.json --out results/v4/evidence \
  --device cpu --workers 4
```

学习策略、`source_order` 和 `shortest_initial` 每组验证预算为 `4×2=8` 次；`exhaustive` 为 `8×2=16` 次。准备失败或未接受候选的请求仍保留在独立部署的请求分母中。`*_decision.json` 保存实际排名、Top-K、在线验证、选择结果及独立执行轨迹；`deployment_summary.json` 保存请求数、实际调用数和实测耗时。

等待训练、采集和部署进程全部结束，再单独测量模块时间，避免并发负载改变计时条件：

```bash
CUDA_VISIBLE_DEVICES=0 "$TG_PYTHON" scripts/evaluate_value_v4.py benchmark \
  --freeze results/v4/frozen.json --data results/v4/data \
  --out results/v4/evidence --device cuda --repeats 5
```

计时模式只读取输入，包含图构建、完整性检查、编码、推理和导出；它不读取候选结果标签，也不把 `N/K` 当成实测加速比。部署的 `policy_wall_seconds` 包含排名、在线验证和独立执行，共享的检查点准备、候选生成及图构建另记。

## 绘图与打包

全部评价、部署及计时完成后运行：

```bash
"$TG_PYTHON" scripts/plot_value_v4.py \
  --evidence results/v4/evidence --freeze results/v4/frozen.json \
  --out results/v4/evidence/figures

# TG_CODE_COMMIT 应来自有 Git 的源码副本中的 git rev-parse HEAD，并核对源码快照。
: "${TG_CODE_COMMIT:?请先设为经过核对的源码提交ID}"
"$TG_PYTHON" scripts/package_value_v4.py \
  --run results/v4 --out results/v4/release --code-commit "$TG_CODE_COMMIT"
```

远程运行目录可以没有 `.git`；提交 ID 必须另行核对，不能据目录名推断。打包程序校验原始记录、冻结权重及正式源码哈希，将正式数据、执行和开发试验分开归档；所有模型的 `best.pt/history.json/split.json/source.json/summary.json` 一并保留。`selected_top_k.json` 从已有选定策略记录中提取，不重新排名。程序拒绝覆盖非空发布目录，并在任一产物超过 95 MiB 或正式源码无法按哈希恢复时中止；历史开发源码缺失会明确列入清单。

发布时按开头的路径映射放置产物，原样复制 `manifest.json` 和 `manifest.sha256` 到 `docs/evidence/value_v4/`；校验时转换路径，不修改清单内容。完整日志、准备失败、实际仿真次数与请求分母均以原始记录和清单为准。
