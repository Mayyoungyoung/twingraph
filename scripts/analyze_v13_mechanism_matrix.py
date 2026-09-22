"""Audit scenario-dependent candidate coverage before any model is trained."""
import argparse
import csv
import itertools
import json
from pathlib import Path

from simbench.value.plan import digest
from simbench.value.system_v11 import save
from scripts.run_v13_mechanism_matrix import SCHEMA


def analyze(roots, out):
    cases=[]; runtime=None; stage=None; names=None
    for root in map(Path,roots):
        for summary_path in sorted(root.glob("seed_*/**/summary.json")):
            summary=json.loads(summary_path.read_text())
            if summary.get("schema")!=SCHEMA or not summary.get("complete"):
                raise ValueError(f"incomplete or incompatible matrix: {summary_path}")
            request_path=summary_path.parent/"request.json"
            request=json.loads(request_path.read_text())
            if request.get("schema")!=SCHEMA or request.get("seed")!=summary.get("seed"):
                raise ValueError("summary/request binding mismatch")
            if runtime is None: runtime=summary["runtime_sha256"]
            if stage is None: stage=summary["stop_after"]
            if runtime!=summary["runtime_sha256"] or stage!=summary["stop_after"]:
                raise ValueError("matrices must share one runtime and label scope")
            order=[p["name"] for p in request["pool"]]
            labels={t["name"]:bool(t["success"]) for t in summary["trials"]}
            if set(order)!=set(labels) or len(order)!=len(set(order)):
                raise ValueError("candidate membership is not complete and unique")
            canonical=sorted(order)
            if names is None: names=canonical
            if names!=canonical:
                raise ValueError("semantic candidate identifiers differ across scenes")
            cases.append(dict(seed=int(summary["seed"]),labels=labels,request_sha256=digest(request),
                              successes=sum(labels.values()),mixed=0<sum(labels.values())<len(labels)))
    if len(cases)<2:
        raise ValueError("at least two independent layouts are required")
    crossovers=[]
    for a,b in itertools.combinations(names,2):
        a_wins=[c["seed"] for c in cases if c["labels"][a] and not c["labels"][b]]
        b_wins=[c["seed"] for c in cases if c["labels"][b] and not c["labels"][a]]
        if a_wins and b_wins:
            crossovers.append(dict(candidate_a=a,candidate_b=b,a_only_success_seeds=a_wins,
                                   b_only_success_seeds=b_wins,evidence_count=len(a_wins)+len(b_wins)))
    crossovers.sort(key=lambda r:(-r["evidence_count"],r["candidate_a"],r["candidate_b"]))
    solution_layouts=sum(c["successes"]>0 for c in cases)
    mixed_layouts=sum(c["mixed"] for c in cases)
    report=dict(schema="twingraph.mechanism_scenario_crossover.v13.v1",stop_after=stage,
        runtime_sha256=runtime,layouts=len(cases),candidates_per_layout=len(names),
        layouts_with_at_least_one_success=solution_layouts,mixed_layouts=mixed_layouts,
        candidate_coverage_rate=solution_layouts/len(cases),
        mixed_layout_rate=mixed_layouts/len(cases),pairwise_crossover_count=len(crossovers),
        crossover_gate_pass=bool(solution_layouts==len(cases) and mixed_layouts>=2 and crossovers),
        gate_definition=("every audited development layout has a physical prefix success; at least two layouts are mixed; "
                         "one aligned candidate pair reverses binary preference across layouts"),
        crossovers=crossovers,cases=[{k:v for k,v in c.items() if k!="labels"} for c in cases],
        limitations=("development prefix labels only; no full-task, held-out generalization, value-ranking, "
                     "online LLM, or hardware-success claim"))
    out=Path(out);out.mkdir(parents=True,exist_ok=True);save(out/"summary.json",report)
    with (out/"matrix.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(["seed",*names])
        for c in cases:w.writerow([c["seed"],*(int(c["labels"][n]) for n in names)])
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument("--roots",nargs="+",required=True)
    p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    print(json.dumps(analyze(a.roots,a.out),ensure_ascii=False,indent=2))


if __name__=="__main__":main()
