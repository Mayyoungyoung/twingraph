"""Package exact v3 artifacts and build a report from measured results."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import tarfile
import numpy as np
import torch
import mujoco
import scipy

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from simbench.value.collect import dump


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def pct(x):return f"{100*x:.1f}%" if x is not None else "—"


def archive(path,files,root):
    with tarfile.open(path,"w:gz") as tar:
        for p in sorted(files):
            if p.is_file() and "__pycache__" not in p.parts:
                tar.add(p,arcname=p.relative_to(root).as_posix())


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--run",default="runs/v3")
    ap.add_argument("--out",default="runs/v3/release");ap.add_argument("--code-commit",required=True);a=ap.parse_args()
    run=Path(a.run).resolve();out=Path(a.out).resolve();repo=Path(__file__).resolve().parents[1]
    evidence=out/"docs/evidence/value_v3";models=out/"models/value/v3";datasets=out/"datasets/value"
    for d in (evidence,models,datasets):d.mkdir(parents=True,exist_ok=True)
    frozen=read(run/"frozen.json");test=read(run/"evidence/test_metrics.json");timing=read(run/"evidence/module_timing.json")
    deploy=read(run/"evidence/deployment_summary.json");selected=frozen["selected"]
    for name in ("test_metrics.json","module_timing.json","deployment_summary.json"):
        shutil.copy2(run/"evidence"/name,evidence/name)
    shutil.copy2(run/"frozen.json",evidence/"frozen.json");shutil.copy2(run/"regression.log",evidence/"regression.txt")
    cases=sorted((run/"evidence").glob("deployment_*/decision.json"))
    archive(datasets/"skill_graph_v3_execution.tar.gz",[f for c in cases for f in c.parent.rglob("*")],run/"evidence")
    example=run/"evidence"/f"deployment_41200_{selected}"/"top_k.json"
    shutil.copy2(example,evidence/"selected_top_k.json")
    group_dirs=sorted((run/"fresh").glob("group_*/complete.json"))
    if len(group_dirs)!=32:raise ValueError("expected 32 fresh configurations")
    archive(datasets/"skill_graph_v3.tar.gz",[f for c in group_dirs for f in c.parent.rglob("*")],run/"fresh")
    inputs=[];trials=[]
    for complete in group_dirs:
        d=complete.parent;inp=read(d/"inputs.json");outcome=read(d/"outcomes.json");summary=read(complete)
        trials+=outcome["trials"]
        inputs.append(dict(seed=inp["task"]["seed"],split=inp["declared_split"],group_id=inp["group_id"],
          input_sha256=summary["input_sha256"],source_sha256=summary["source_sha256"],
          files={p.name:sha(p) for p in d.iterdir() if p.is_file()},summary=summary))
    model_rows=[]
    for row in frozen["models"]:
        source=Path(row["path"]).parent;target=models/row["name"];target.mkdir(exist_ok=True)
        for name in ("best.pt","history.json","split.json","summary.json"):shutil.copy2(source/name,target/name)
        saved=torch.load(target/"best.pt",map_location="cpu",weights_only=False)
        model_rows.append(dict(**{k:row[k] for k in ("name","sha256","kind","seed","epoch","source_sha256","split_sha256")},
               file=f"models/value/v3/{row['name']}/best.pt",config=saved["config"],dim=saved["dim"],summary=read(source/"summary.json")))
    shutil.copy2(models/selected/"best.pt",models/"best_graph_input.pt")
    dump(models/"MANIFEST.json",dict(selected=selected,best_sha256=sha(models/"best_graph_input.pt"),models=model_rows))
    pilot_files=[]
    for pattern in ("development_*","pilot_model","models"):
        for d in run.glob(pattern):
            if d.is_dir():pilot_files.extend(d.rglob("*"))
    archive(datasets/"skill_graph_v3_development.tar.gz",pilot_files,run)
    archive(datasets/"skill_graph_v3_source.tar.gz",list((repo/"simbench").rglob("*"))+list((repo/"scripts").glob("*skill_graph*")),repo)
    environment=dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,mujoco=mujoco.__version__,
       numpy=np.__version__,scipy=scipy.__version__,cuda=torch.version.cuda,
       gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    manifest=dict(schema="twingraph.value.graph.dataset.v1",runtime_code_commit=a.code_commit,environment=environment,
        new_groups=32,new_label_rollouts=len(trials),new_full_successes=sum(r["success"] for r in trials),
        new_prefix_successes=sum(r["prefix_success"] for r in trials),timeouts=sum(r["timeout"] for r in trials),
        physics_steps=sum(r["physics_steps"] for r in trials),rollout_wall_sum_seconds=sum(r["wall_seconds"] for r in trials),
        collection_worker_wall_sum_seconds=sum(x["summary"]["wall_seconds"] for x in inputs),
        formal_training_wall_sum_seconds=sum(x["summary"]["wall_seconds"] for x in model_rows),
        reused_old_train_groups=12,reused_old_val_groups=4,
        note="Fresh: train16/val4/test12, 16 candidates x 2 physical repeats. Old pin train12/val4 reused; old test excluded. Worker/process sums are not parallel elapsed time.",
        groups=inputs,archives={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in datasets.glob("skill_graph_v3*.tar.gz")})
    dump(datasets/"manifest_skill_graph_v3.json",manifest);dump(evidence/"environment.json",environment)
    with (out/"EXPERIMENT_TABLE_VALUE_GRAPH.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["method","groups","hit1","hit2","hit4","quality4","regret4","brier","module_median_ms","module_p95_ms","scoring_median_ms"]);w.writeheader()
        for name,result in test["results"].items():
            row={k:result[k] for k in ("groups","hit1","hit2","hit4","quality4","regret4","brier")};row["method"]=name
            if name in timing["summary"]:
                tm=timing["summary"][name];row.update(module_median_ms=tm["total"]["median_seconds"]*1000,module_p95_ms=tm["total"]["p95_seconds"]*1000,scoring_median_ms=tm["scoring_after_shared_graph"]["median_seconds"]*1000)
            w.writerow(row)
    table=[]
    for name in ("geometry","mlp_17","field_17","sequence_17","graph_17",selected):
        r=test["results"][name];tm=timing["summary"][name]
        table.append(f"| {name} | {pct(r['hit1'])} | {pct(r['hit2'])} | {pct(r['hit4'])} | {pct(r['quality4'])} | {tm['total']['median_seconds']*1000:.1f} / {tm['total']['p95_seconds']*1000:.1f} | {tm['scoring_after_shared_graph']['median_seconds']*1000:.1f} |")
    r=test["results"][selected];geo=test["results"]["geometry"]
    drows=deploy["decisions"];dlines=[]
    for method in sorted({r["method"] for r in drows}):
        rows=[r for r in drows if r["method"]==method]
        dlines.append(f"- {method}: {sum(x['deployment_successes'] for x in rows)}/{sum(x['deployment_attempts'] for x in rows)} 独立部署试验成功；{len(rows)} 个问题组；在线验证 {sum(x['rollouts'] for x in rows)} 次。")
    chosen=next(x for x in model_rows if x["name"]==selected)
    tm=timing["summary"];saving=(tm["geometry"]["total"]["median_seconds"]-tm[selected]["total"]["median_seconds"])*1000
    seedlines=[]
    for kind in ("port_mlp","sequence","graph"):
        rs=[test["results"][f"{kind}_{seed}"] for seed in (17,29,43)]
        seedlines.append(f"- {kind}: Hit@1 均值 {pct(np.mean([x['hit1'] for x in rs]))}，Hit@4 均值 {pct(np.mean([x['hit4'] for x in rs]))}，Top-4 平均质量 {pct(np.mean([x['quality4'] for x in rs]))}。")
    report=f'''# 技能图输入价值模块：滑台销装配验证结果

## 结论

已打通“技能接口 → 可执行图 → 价值输入 → Top-K → 同一 PlanIR 物理执行”。
验证集预先选定的方案是 **{selected}**，使用图中自动生成的类型化端口输入，
不计算几何排序代理。新测试集 Hit@1/2/4 均为 {pct(r['hit4'])}（12 个配置）。
当前证据支持统一图输入和浅层 MLP 评分；**不支持关系注意力已经带来增益**。

## 实验与数据

- 新采集 32 个配置 × 16 个真实候选 × 2 次配对扰动 = {len(trials)} 次物理试验。
- 28 个训练配置（旧12＋新16，896次）、8 个验证配置（旧4＋新4，256次）、12 个全新测试配置（384次）。
- 新数据整段成功 {manifest['new_full_successes']}/{len(trials)}，前缀成功 {manifest['new_prefix_successes']}/{len(trials)}，超时 {manifest['timeouts']}。
- 任务为滑台销安装完整后缀，19 次调用；不是整台滑台装配或真实机器人实验。
- 训练/验证选完 checkpoint 和方案后才打开新测试；全部试验保留，未过滤全成功／全失败组。
- 候选差异包括抓取方向、高度、搬运净空、夹持力和保护下降速度，均来自实际执行参数。

## 模块结果

![价值模块质量与耗时](docs/evidence/value_v3/module_summary.png)

| 方法 | Hit@1 | Hit@2 | Hit@4 | Top-4平均经验成功率 | 模块中位/P95(ms) | 共享图准备后评分(ms) |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

Hit@K：Top-K 中至少一个方案达到池内经验最佳值减 0.1；不表示 K 个全部成功。
每候选只有两次参考执行，经验率粗。12/12 的组级 Wilson 95% 区间约为
{r['hit_wilson95']['hit4'][0]*100:.1f}%–100%，不能写成可靠保证。

{chr(10).join(seedlines)}

几何规则与图端口 MLP 的 Hit@4 都已饱和；改善主要体现在第一名和前四名整体质量。
原 MLP87 仍是强基线：其 Top-4 平均质量和概率校准更好。选定端口 MLP 的
Brier={r['brier']:.3f}，MLP87={test['results']['mlp_17']['brier']:.3f}，应优先把新模型分数用于排序，
不能当作已经校准的成功概率。

## 耗时口径

- 同一 RTX 2080 Ti 主机；模型常驻并预热；同步 CUDA；每个配置重复测量5次，先取组内中位，再跨12组报中位/P95。
- 模块总时间包含：图构建、一致性检查、可选几何评分、输入编码、推理、Top-K排序导出。
- 表最后一列扣除了所有方法共有的图构建和一致性检查，便于看到评分本身的代价。
- 不包含场景搭建、候选必要几何检查、物理验证、渲染；本轮模型不使用图像。原始分项全部保存。
- 选定端口 MLP 相比几何规则的模块中位耗时差为节省 {saving:.1f} ms。该小幅收益不是端到端机器人加速比。
- 只在本次候选池内复用相同端口值；无跨请求特征/成功标签缓存。数值输入不变性有回归测试。
- 冷加载记录仅是权重载入，未声称是完整进程冷启动。`scoring_top_k.json` 单独保存实际CLI首调用；首次推理的延迟不等于表中的预热耗时。

## 独立执行

固定测试种子41200–41203；Top-4、总在线预算8、每候选重复2次；比较规则完全相同。
选中方案在另一个 `deployment` 扰动命名空间执行，未使用参考标签决定最终选择。

{chr(10).join(dlines)}

这是仿真中的独立部署，统计单位仍只有4个问题组；不能把重复试验视作独立配置。

## 模型和输入

选定模型输入维数 {chosen['dim']}，由训练图的端口词表自动生成（每个键24个通用数值通道＋1个计数），
隐藏层64、32，单个成功倾向输出。参数量 {chosen['summary']['parameters']}。
图关系模型和无关系控制均为64维、2层、4头，调用内端口注意力汇聚；仅图模型加入有向状态依赖偏置。
字段序列控制也为64维/2层，不是旧128维视觉模型原封不动复现。
具体端口来源、状态版本和结构图见 [设计说明](docs/value-v3-design.md)。

## 已完成／初步证据／未完成

**已完成**：共享接口、图编译、执行一致性、完整Top-K、92项回归测试、新物理数据、11个最终模型、
3种子对照、冻结测试、真实模块计时、独立执行、checkpoint及源码/数据哈希。

**有初步证据**：图接口输入的端口 MLP 能在此任务保持高Top-K保留质量，免去可选几何代理，
并取得表中实测成本差。输入选择可以追溯到执行接口，无需手写产品特征表。

**尚未完成**：完整滑台多零件、真实剩余检查点训练、未见长度/顺序组合物理泛化、视觉收益、
跨家族固定权重迁移及真实机器人验证。当前19步拓扑基本固定，正确依赖边并没有表现出可靠额外价值。
移除/打乱边的诊断结果和所有种子均已公开，不能只挑有利的一行。

下一轮应在保持此测试作回归的前提下，加入真实多零件/重新抓持检查点，使同一接口下的状态来源发生
有意义变化；预留新配置测试，再检验关系模型。优先扩大任务组合与数据覆盖，而不是扩大视觉网络。

## 复现与工件

- [运行命令](docs/value-v3-running.md)；[预先协议与开发修订](docs/value-v3-protocol.md)。
- [完整实验表](EXPERIMENT_TABLE_VALUE_GRAPH.csv)；[完整Top-K](docs/evidence/value_v3/selected_top_k.json)。
- 最佳图输入checkpoint：`models/value/v3/best_graph_input.pt`（{selected}）。
- SHA256：`{sha(models/'best_graph_input.pt')}`。
- 训练期代码基线：`{a.code_commit}`；模型各自记录训练源码与split哈希。
- 新数据、执行记录、开发历史、源码包与全部哈希：`datasets/value/manifest_skill_graph_v3.json`。
- 环境：Torch {torch.__version__}，MuJoCo {mujoco.__version__}，NumPy {np.__version__}。

PIGINet已研究计划/图像/目标/初态可行性预测及求解时间，参见
[原论文](https://roboticsproceedings.org/rss19/p061.html)。本轮是共享执行接口输入的验证原型，
不把工业场景、Transformer或关系偏置本身当成已经成立的新方法贡献。
'''
    (out/"VALUE_GRAPH_REPORT.md").write_text(report,encoding="utf-8")
    print(json.dumps(dict(out=str(out),selected=selected,new_rollouts=len(trials),archives=manifest["archives"])))


if __name__=="__main__":main()
