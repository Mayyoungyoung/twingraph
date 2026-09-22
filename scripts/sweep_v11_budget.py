"""Offline K sweep of the full physical matrix; never labeled online timing."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scripts.analyze_v9_candidate_matrix import outcome,random_expectation,aggregate_replay
from simbench.value.system_v11 import FrozenValue,save


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();value=FrozenValue(a.checkpoint);rows=[]
    for root in sorted(a.root.glob("seed_*/all_twin")):
        request=json.loads((root / "request.json").read_text())
        case={};graphs=[];names=[]
        for proposal in request["pool"]:
            name=proposal["name"];names.append(name)
            detail=json.loads((root / "twins" / name / "result.json").read_text())
            case[name]=dict(summary=detail)
            graphs.append(json.loads((root / "twins" / name / "input_graph.json").read_text()))
        t=time.perf_counter();scores=value.score(graphs);ranking=time.perf_counter()-t
        order=[names[i] for i in np.argsort(-np.asarray(scores),kind="stable")]
        for k in (1,2,4,8,12):
            rows.append(dict(seed=request["seed"],method="frozen_value",k=k,order=order,
                **outcome(case,order,k,request["generation_seconds"]+ranking)))
            rows.append(dict(seed=request["seed"],method="exact_random",k=k,
                **random_expectation(case,names,k,request["generation_seconds"])))
    aggregate=aggregate_replay(rows)
    save(a.out,dict(note="Offline replay of measured full-matrix outcomes and costs. Not independent deployment or newly measured online time.",
                   model_sha256=value.sha256,rows=rows,aggregate=aggregate))
    print(json.dumps(aggregate,indent=2))


if __name__=="__main__":main()
