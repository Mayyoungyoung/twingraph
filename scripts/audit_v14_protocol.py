"""Reconcile V13 candidate, encoding, split, reversal and cost scopes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from simbench.value.graph_value_v12 import encode_graph
from simbench.value.plan import digest
from scripts.train_value_v12 import dump


def encoded_sha(encoded):
    h=hashlib.sha256()
    for key in ("x","relations","active"):
        value=np.ascontiguousarray(encoded[key]);h.update(key.encode());h.update(str(value.shape).encode());h.update(value.tobytes())
    return h.hexdigest()


def semantic_sha(proposal):
    return digest({k:v for k,v in proposal.items() if k not in ("name","source","rationale")})


def audit(root, manifest_path, gate_path, reversal_path, metrics_path, out):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(Path(manifest_path).read_text());gate=json.loads(Path(gate_path).read_text())
    reversal=json.loads(Path(reversal_path).read_text());metrics=json.loads(Path(metrics_path).read_text())
    split={int(s):"train" for s in manifest["train_layouts"]}|{int(s):"validation" for s in manifest["validation_layouts"]}
    layouts=[];graph_index={}
    for request_path in sorted(root.glob("seed_*/end_stop/request.json")):
        request=json.loads(request_path.read_text());seed=int(request["seed"]);semantics=[];encodings=[];valid=success=0
        for proposal in request["pool"]:
            result_path=request_path.parent/"candidates"/proposal["name"]/"result.json"
            graph_path=result_path.with_name("input_graph.json")
            if not result_path.exists() or not graph_path.exists():continue
            result=json.loads(result_path.read_text());graph=json.loads(graph_path.read_text())
            sem=semantic_sha(proposal);enc=encoded_sha(encode_graph(graph));semantics.append(sem);encodings.append(enc)
            valid+=int(result.get("valid") is True);success+=int(result.get("success") is True)
            graph_index[(seed,proposal["name"])]=dict(semantic_sha256=sem,encoded_sha256=enc,
                graph_sha256=digest(graph),success=bool(result.get("success")))
        collisions={e:sorted({s for s,x in zip(semantics,encodings) if x==e}) for e in set(encodings)}
        collisions={e:s for e,s in collisions.items() if len(s)>1}
        layouts.append(dict(layout=seed,scope="assembly_prefix_through_end_stop",request_n=request["n"],
            raw_candidates=len(request["pool"]),semantic_unique=len(set(semantics)),encoded_unique=len(set(encodings)),
            valid_executions=valid,successes=success,split=split.get(seed,"development_not_selected"),
            inclusion_reason=("V13 selected development training/validation snapshot" if seed in split else "not selected"),
            semantic_to_encoding_collisions=len(collisions)))
    rows=[]
    for check in reversal["checks"]:
        a,b=check["candidate_a"],check["candidate_b"]
        for d in check["directions"]:
            seed=int(d["seed"]);ga=graph_index[(seed,a)];gb=graph_index[(seed,b)]
            rows.append(dict(layout=seed,split=split.get(seed,"development_not_selected"),candidate_a=a,candidate_b=b,
                result_a=int(ga["success"]),result_b=int(gb["success"]),expected_above=d["expected_above"],
                score_a=d["score_a"],score_b=d["score_b"],correct=d["correct"],paired_repeats=1,
                definition="single deterministic physical-prefix execution; strict score inequality; ties incorrect"))
    selection_dir=Path(metrics_path).parent
    selection_path=selection_dir/"training_selection.json"
    if not selection_path.exists():
        selection_path=selection_dir/"selection.json"
    selected=json.loads(selection_path.read_text()).get("selected","graph")
    metric=metrics[selected]["validation"]
    report=dict(schema="twingraph.v14_protocol_audit.v1",runtime_sha256=manifest["physical_runtime_sha256"],
        layouts=layouts,all_groups_fixed_n8=all(r["raw_candidates"]==8 for r in layouts),
        encoding_collision_groups=sum(r["semantic_to_encoding_collisions"] for r in layouts),
        reversal_scope=dict(rows=len(rows),train_rows=sum(r["split"]=="train" for r in rows),
            validation_rows=sum(r["split"]=="validation" for r in rows),paired_repeats=1,
            old_correct=sum(r["correct"] for r in rows),old_total=len(rows)),
        cost_scope=dict(unit="simulator advanced seconds from archived complete matrices",actual_online_wall_clock=False,
            full_pool_seconds=metric["all_twin_verification_seconds"],
            value_early_stop_seconds=metric["value_early_stop_verification_seconds"],
            random_top2_expected_seconds=metric["random_verification_seconds"],
            value_top2_success=metric["success"],random_top2_exact_success=metric["random_success"],
            full_pool_definition="execute all candidates even after a success; label-collection cost, not an early-stop planner"),
        success_scope="physical execution from fresh scene initialization through carriage and end_stop prefix only",
        full_task_claim=False)
    dump(out/"PROTOCOL_AUDIT.json",report)
    with (out/"V13_GROUP_COUNTS.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(layouts[0]));w.writeheader();w.writerows(layouts)
    with (out/"REVERSAL_COVERAGE.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    md=f"""# V14 协议口径审计\n\n## 结论\n\n- V13 的 32/16 个图不是编码碰撞或记录缺失，而是训练快照混用了 `N=4` 与 `N=8`：逐布局见 `V13_GROUP_COUNTS.csv`。新可比较实验必须统一 `N=8`。\n- 所扫布局中不同执行语义映射到同一编码的碰撞组数为 **{report['encoding_collision_groups']}**。\n- 旧严格反转共 {len(rows)} 个方向行，其中训练布局 {report['reversal_scope']['train_rows']} 行、验证布局 {report['reversal_scope']['validation_rows']} 行；普通 97.78% 只统计 3 个验证布局，二者不是同一分母。\n- 每个旧反转方向只有 1 次确定性执行，没有外生扰动重复；严格比较使用 `score(expected) > score(other)`，平局计错。\n- “成功”仅覆盖 fresh scene 到 carriage + end_stop 的物理执行前缀，不含插销、把手、功能行程或完整任务验收。\n- 425.31/91.46/124.34 的单位是归档结果中的**仿真推进秒**，是矩阵回放的顺序成本估计，不是真实在线墙钟。Full-pool 指标即使已找到成功仍执行全部候选，属于标注成本。\n- 随机 Top-2 的精确期望成功率为 {metric['random_success']:.4f}，价值 Top-2 为 {metric['success']:.4f}；成本必须与成功率共同解释。\n\n## 修复要求\n\nV14 使用独立、预声明的统一 N=8 布局划分；模型冻结前不读取测试结果；同时报告仿真秒、实际孪生墙钟、生成/编码/推理墙钟和完整在线墙钟。\n"""
    (out/"V14_PROTOCOL_AUDIT_ZH.md").write_text(md,encoding="utf-8")
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True);p.add_argument("--gate",type=Path,required=True)
    p.add_argument("--reversal",type=Path,required=True);p.add_argument("--metrics",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    print(json.dumps(audit(a.root,a.manifest,a.gate,a.reversal,a.metrics,a.out),ensure_ascii=False,indent=2))


if __name__=="__main__":main()
