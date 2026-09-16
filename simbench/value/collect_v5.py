"""Collect distinct full sliding-stage programs, starting unassembled.

Repeat 0 is an actual nominal execution of each candidate. Optional repeats add
small paired physics uncertainty; they never create extra candidate identities.
"""
import argparse
import concurrent.futures
import contextlib
from dataclasses import asdict
import json
import multiprocessing
from pathlib import Path
import time
import traceback

import numpy as np

from .collect import dump
from .physical import PhysicalRunner
from .plan import digest
from .skill_graph import compile_graph
from .program_input_v5 import source_manifest


def trial_spec(seed, repeat, domain, friction_span=.03, gain_span=.005):
    if repeat==0:
        return dict(domain="nominal",repeat=0,friction_scale=1.,actuator_gain_scale=1.)
    namespaces={"train":611,"reference":613,"development":610}
    if domain not in namespaces or not 0<=friction_span<1 or not 0<=gain_span<1:
        raise ValueError("unknown disturbance domain or invalid envelope")
    rng=np.random.default_rng(np.random.SeedSequence([seed,repeat,namespaces[domain]]))
    return dict(domain=domain,repeat=repeat,friction_scale=float(rng.uniform(1-friction_span,1+friction_span)),
                actuator_gain_scale=float(rng.uniform(1-gain_span,1+gain_span)))


def collect_one(seed,out,split="train",domain="train",n=12,repeats=1,timeout=240.):
    from . import stage_v5 as stage
    directory=Path(out)/f"group_{seed}"
    existed=directory.exists() and any(directory.iterdir())
    directory.mkdir(parents=True,exist_ok=True)
    source=source_manifest()
    request=dict(seed=seed,split=split,domain=domain,n=n,repeats=repeats,timeout_seconds=timeout,
                 source_sha256=digest(source),friction_span=.03,gain_span=.005)
    request_path=directory/"request.json"
    if request_path.exists() and json.loads(request_path.read_text())!=request:
        raise ValueError("existing collection request/source differs; use a new run directory")
    if (directory/"complete.json").exists():
        complete=json.loads((directory/"complete.json").read_text())
        if complete["request_sha256"]!=digest(request):raise ValueError("resume request/source mismatch")
        return complete
    if existed:
        raise FileExistsError("preserve incomplete/failed attempt; select a new output directory")
    dump(request_path,request)
    dump(directory/"source.json",source)
    started=time.perf_counter()
    with (directory/"steps.log").open("w") as log,contextlib.redirect_stdout(log):
        spec,session,xml,targets=stage.make_scene(seed,directory,role="twin")
        scene_seconds=time.perf_counter()-started
        begin=time.perf_counter()
        plans,counts=stage.build_pool(session,targets,seed,n=n)
        candidate_seconds=time.perf_counter()-begin
        observation=stage.observed(session,targets)
        runner=PhysicalRunner(session,timeout=timeout)
        inputs=dict(schema="twingraph.group.v5",group_id=spec.config_id,split_group=spec.config_id,
            declared_split=split,task=asdict(spec),checkpoint=0,nominal_index=0,
            observation=observation,candidates=[p.to_dict() for p in plans],pool_counts=counts,
            source_sha256=digest(source),snapshot_sha256=runner.initial,
            scene_seconds=scene_seconds,candidate_generation_seconds=candidate_seconds)
        ih=digest(inputs);graphs=[compile_graph(observation,p) for p in plans]
        dump(directory/"inputs.json",inputs)
        dump(directory/"skill_graphs.json",dict(input_sha256=ih,graphs=graphs))
        dump(directory/"source.json",source)
        np.savez_compressed(directory/"initial_physics.npz",**runner.snapshot["physics"]["data"])
        trials=[]
        for repeat in range(repeats):
            perturbation=trial_spec(seed,repeat,domain)
            for graph in graphs:
                trials.append(runner.run(graph,perturbation,keep_trace=True))
                dump(directory/"outcomes.json",dict(input_sha256=ih,trials=trials))
        complete=dict(request_sha256=digest(request),input_sha256=ih,source_sha256=digest(source),
            seed=seed,split=split,candidates=len(plans),trials=len(trials),
            nominal_successes=sum(t["success"] for t in trials if t["trial"]["repeat"]==0),
            all_successes=sum(t["success"] for t in trials),timeouts=sum(t["timeout"] for t in trials),
            plan_lengths=sorted({len(p.calls) for p in plans}),wall_seconds=time.perf_counter()-started,
            physical_wall_seconds=sum(t["wall_seconds"]+t["restore_seconds"] for t in trials),
            physics_steps=sum(t["physics_steps"] for t in trials))
        if source_manifest()!=source:
            raise RuntimeError("source changed during physical collection; preserve outputs for audit")
        if complete["timeouts"]:
            dump(directory/"censored.json",complete)
            raise RuntimeError("censored group retained; increase compute budget before reusing labels")
        dump(directory/"complete.json",complete)
        return complete


def worker(request):
    directory=Path(request["out"])/f"group_{request['seed']}"
    existed=directory.exists() and any(directory.iterdir())
    try:return collect_one(**request)
    except Exception as exc:
        row=dict(request=request,error=traceback.format_exc(),exception_type=type(exc).__name__,message=str(exc))
        directory.mkdir(parents=True,exist_ok=True)
        row["phase"]="after_inputs" if (directory/"inputs.json").exists() else "before_inputs"
        if not existed and not (directory/"complete.json").exists() and not (directory/"failure.json").exists():
            dump(directory/"failure.json",row)
        return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",required=True);p.add_argument("--seed",type=int,required=True)
    p.add_argument("--groups",type=int,default=1);p.add_argument("--n",type=int,default=12)
    p.add_argument("--repeats",type=int,default=1);p.add_argument("--workers",type=int,default=8)
    p.add_argument("--split",choices=("train","val","test","development"),default="train")
    p.add_argument("--domain",choices=("train","reference","development"),default="train")
    p.add_argument("--timeout",type=float,default=240.)
    a=p.parse_args()
    if a.n<1 or a.repeats<1:raise ValueError("positive candidates/repeats required")
    requests=[dict(seed=a.seed+i,out=a.out,split=a.split,domain=a.domain,n=a.n,repeats=a.repeats,timeout=a.timeout)
              for i in range(a.groups)]
    results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        pending={pool.submit(worker,r):r for r in requests}
        for future in concurrent.futures.as_completed(pending):
            row=future.result();results.append(row);print(json.dumps(row),flush=True)
    dump(Path(a.out)/f"batch_{a.split}_{a.seed}.json",results)
    if any("error" in r for r in results):raise SystemExit(1)


if __name__=="__main__":main()
