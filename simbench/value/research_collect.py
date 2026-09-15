"""Resumable v2 physical collection; all-success/all-failure groups survive."""
import argparse
import hashlib
import contextlib
import concurrent.futures
from dataclasses import asdict
import json
import multiprocessing
from pathlib import Path
import time
import traceback
import numpy as np
from .research_scenarios import *
from .physical import PhysicalRunner,perturbation
from .plan import execute_calls
from .collect import dump,source_hash


def collection_source_hash():
    root=Path(__file__).resolve().parents[1]
    files=[root/"value"/n for n in ("research_scenarios.py","research_collect.py","physical.py","plan.py","scenarios.py","collect.py")]
    files+=list((root/"assembly").glob("*.py"))+list((root/"core").glob("*.py"))
    h=hashlib.sha256()
    for f in sorted(files):h.update(f.relative_to(root).as_posix().encode());h.update(f.read_bytes())
    return h.hexdigest()


def collect_one(family,seed,out,n=16,repeats=2,domain="train",split="train",checkpoint=0,render=True):
    directory=Path(out)/f"group_{family}_{seed}_{checkpoint}"
    directory.mkdir(parents=True,exist_ok=True)
    request=dict(family=family,seed=seed,n=n,repeats=repeats,domain=domain,split=split,checkpoint=checkpoint,render=render)
    if (directory/"complete.json").exists():
        result=json.loads((directory/"complete.json").read_text())
        if result["request_sha256"]!=digest(request):raise ValueError("resume request mismatch")
        return result
    start=time.perf_counter()
    with (directory/"steps.log").open("w") as log,contextlib.redirect_stdout(log):
        spec,s,path,targets=make_family(family,seed,directory)
        completed=[];checkpoint_trace=[]
        if checkpoint:
            order=list(s.parts)
            choices={p:dict(yaw=0.,height=.003,clearance=1.035,force=3.,speed=.008) for p in order}
            warm=program(s,targets,order,choices)
            # A genuine trajectory prefix is executed, not reconstructed by teleport.
            s.active_candidate_id=warm.id
            execute_calls(s,warm,warm.calls[:18*checkpoint])
            completed=order[:checkpoint];checkpoint_trace=plain(s.results)
        plans,counts=build_pool(s,targets,seed,n,completed)
        if not plans:raise RuntimeError("no candidate survived necessary checks")
        obs=observed(s,targets)
        t=time.perf_counter(); images=render_observation(s,directory,obs["objects"]) if render else []
        rendering=time.perf_counter()-t
        runner=PhysicalRunner(s)
        inputs=dict(schema="twingraph.group.v2",group_id=spec.config_id+f"_cp{checkpoint}",split_group=spec.config_id,
                    declared_split=split,protocol=PROGRAM_PROTOCOL,task=asdict(spec),checkpoint=checkpoint,
                    checkpoint_trace=checkpoint_trace,observation=obs,images=images,candidates=[p.to_dict() for p in plans],
                    pool_counts=counts,render_seconds=rendering,source_sha256=collection_source_hash(),snapshot_sha256=runner.initial)
        ih=digest(inputs);dump(directory/"inputs.json",inputs)
        np.savez_compressed(directory/"initial_physics.npz",**runner.snapshot["physics"]["data"])
        # Rule labels are optional pre-rollout features, stored separately from model inputs.
        t=time.perf_counter(); features=[geometric_features(s,p,targets) for p in plans]
        dump(directory/"geometry.json",dict(input_sha256=ih,features=features,wall_seconds=time.perf_counter()-t))
        trials=[]
        for repeat in range(repeats):
            trial=perturbation(seed,repeat,domain)
            for plan in plans:
                row=runner.run(plan,trial,keep_trace=True);trials.append(row)
                dump(directory/"outcomes.json",dict(input_sha256=ih,trials=trials))
        summary=dict(request_sha256=digest(request),input_sha256=ih,source_sha256=inputs["source_sha256"],
                     family=family,seed=seed,checkpoint=checkpoint,split=split,candidates=len(plans),trials=len(trials),
                     full_successes=sum(x["success"] for x in trials),prefix_successes=sum(x["prefix_success"] for x in trials),
                     wall_seconds=time.perf_counter()-start,plan_lengths=sorted({len(p.calls) for p in plans}))
        dump(directory/"complete.json",summary)
    return summary


def worker(args):
    try:return collect_one(**args)
    except Exception:return dict(error=traceback.format_exc(),request=args)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",required=True);p.add_argument("--family",default="rigid_connector_module")
    p.add_argument("--seed",type=int,default=20000);p.add_argument("--groups",type=int,default=1)
    p.add_argument("--n",type=int,default=16);p.add_argument("--repeats",type=int,default=2)
    p.add_argument("--workers",type=int,default=4);p.add_argument("--domain",default="train")
    p.add_argument("--split",default="train");p.add_argument("--checkpoint",type=int,default=0)
    p.add_argument("--no-render",action="store_true");a=p.parse_args()
    requests=[dict(family=a.family,seed=a.seed+i,out=a.out,n=a.n,repeats=a.repeats,domain=a.domain,split=a.split,
                   checkpoint=a.checkpoint,render=not a.no_render) for i in range(a.groups)]
    Path(a.out).mkdir(parents=True,exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        results=[]
        for result in pool.map(worker,requests):
            results.append(result);print(json.dumps(result),flush=True)
    dump(Path(a.out)/f"batch_{a.family}_{a.seed}_{a.checkpoint}.json",results)
    if any("error" in r for r in results):raise SystemExit(1)


if __name__=="__main__":main()
