#!/usr/bin/env python3
"""Summarize a frozen V16 natural-candidate matrix without changing labels."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def failure_category(result):
    if result.get("timeout") or result.get("resource_censored"):
        return "resource_timeout"
    if not result.get("valid"):
        return "software_or_record_invalid"
    if result.get("success"):
        return None
    return "physical_execution_failure"


def score_value(graph_paths, checkpoint):
    if checkpoint is None:
        return None
    from simbench.value.generic_graph_value_v15 import ValueRankerV15
    model=ValueRankerV15(checkpoint)
    graphs=[load(path) for path in graph_paths]
    values=[float(x) for x in model.score(graphs)]
    order=sorted(range(len(values)),key=lambda i:(-values[i],i))
    return dict(checkpoint_sha256=model.sha256,scores=values,order=order)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--matrix",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--k",type=int,default=4)
    args=parser.parse_args()
    records=[];layouts=[]
    for seed_root in sorted(args.matrix.glob("seed_*"),key=lambda p:int(p.name.split("_")[-1])):
        collect=seed_root/"collect"; request=load(collect/"request.json")
        observation=request["initial_observation"]
        proposal_by_name={p["name"]:p for p in request["pool"]}
        names=[p["name"] for p in request["pool"]]
        graph_paths=[]; results=[]
        for name in names:
            root=collect/"candidates"/name
            result=load(root/"result.json"); graph_path=root/"input_graph.json"
            graph=load(graph_path); proposal=proposal_by_name[name]
            error=result.get("error") or ""
            first=error.split(":",1)[0] if error else None
            semantic=(graph.get("assembly",{}).get("plan",{}).get("prefix",{})
                      .get("semantic_program_id") or graph.get("assembly",{}).get("plan_sha256"))
            row=dict(layout_id=seed_root.name,runtime_id=result.get("runtime_sha256"),
                observation_hash=(observation.get("sha256") or observation.get("perception",{}).get("observation_sha256")),
                candidate_id=name,physical_candidate_id=result.get("candidate_id"),semantic_hash=semantic,
                generator_version=request.get("source",{}).get("source"),generation_round=0,
                feedback_used=False,full_success=bool(result.get("success")),
                first_failure_stage=first,failure_category=failure_category(result),
                validation_wall_time=float(result.get("total_wall_seconds",0.)),
                simulation_seconds=float(result.get("sim_seconds",0.)),
                record_status="valid_physical_label" if result.get("valid") else "invalid",
                proposal_sha256=hashlib.sha256(json.dumps(proposal,sort_keys=True,separators=(",",":"))
                    .encode()).hexdigest(),input_graph_sha256=result.get("input_graph_sha256"),
                result_sha256=sha(root/"result.json"),result_path=str(root/"result.json"))
            records.append(row);results.append(result);graph_paths.append(graph_path)
        ranking=score_value(graph_paths,args.checkpoint)
        success_indices=[i for i,r in enumerate(results) if r.get("success")]
        value_top=[] if ranking is None else ranking["order"][:args.k]
        random_order=sorted(range(len(names)),key=lambda i:hashlib.sha256(
            f"v16-random-{seed_root.name}-{names[i]}".encode()).digest())
        layout=dict(layout_id=seed_root.name,generated=len(names),submitted=len(names),attempted=len(results),
            successes=len(success_indices),pool_type=("all_negative" if not success_indices else
                "all_positive" if len(success_indices)==len(results) else "mixed"),
            full_n_retains_success=bool(success_indices),
            original_top_k_retains_success=any(i in success_indices for i in range(min(args.k,len(names)))),
            random_top_k_retains_success=any(i in success_indices for i in random_order[:args.k]),
            value_top_k_retains_success=any(i in success_indices for i in value_top) if ranking else None,
            original_top_k=[names[i] for i in range(min(args.k,len(names)))],
            random_top_k=[names[i] for i in random_order[:args.k]],
            value_top_k=[names[i] for i in value_top],value=ranking,
            first_failure_counts=dict(Counter((r.get("error") or "success").split(":",1)[0] for r in results)),
            validation_wall_seconds=sum(float(r.get("total_wall_seconds",0.)) for r in results),
            simulation_seconds=sum(float(r.get("sim_seconds",0.)) for r in results))
        layouts.append(layout)
    pools=Counter(row["pool_type"] for row in layouts)
    summary=dict(schema="twingraph.v16.natural_candidate_coverage.v1",
        matrix_root=str(args.matrix),top_k=args.k,layouts=layouts,records=records,
        layout_count=len(layouts),generated=sum(r["generated"] for r in layouts),
        submitted=sum(r["submitted"] for r in layouts),attempted=sum(r["attempted"] for r in layouts),
        successes=sum(r["successes"] for r in layouts),
        first_round_natural_coverage=sum(r["successes"]>0 for r in layouts)/max(1,len(layouts)),
        cumulative_two_round_coverage=None,feedback_round_executed=False,
        pool_types=dict(pools),generation_shortfall_layouts=sum(r["generated"]<16 for r in layouts),
        first_failure_counts=dict(Counter(r["first_failure_stage"] for r in records)),
        valid_physical_labels=sum(r["record_status"]=="valid_physical_label" for r in records),
        invalid_or_resource_records=sum(r["record_status"]!="valid_physical_label" for r in records),
        validation_wall_seconds=sum(r["validation_wall_time"] for r in records),
        simulation_seconds=sum(r["simulation_seconds"] for r in records),
        interpretation="No unexecuted candidate is assigned a physical label; coverage denominator includes every declared layout.")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k not in ("records","layouts")},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
