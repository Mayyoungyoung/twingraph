"""Collect development-only physical prefix labels for candidate coverage.

The experiment asks a narrow question before value-model fitting: does a
scene-conditioned pool contain executable alternatives, and do their outcomes
vary within a scene?  Prefix success is explicitly not full-task success.
"""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path

from simbench.value.plan import digest
from simbench.value.planner_v12 import normalized_graph, propose
from simbench.value.provenance_v12 import fingerprint as source_fingerprint
from simbench.value.system_v11 import save
from simbench.value.system_v12 import make_scene, rollout


SCHEMA="twingraph.mechanism_candidate_matrix.v13.v1"


def file_sha256(path):
    """Hash exact result bytes, including explicit non-finite diagnostics."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(seed, directory, *, n=12, stop_after="carriage", level="L1"):
    directory=Path(directory); directory.mkdir(parents=True, exist_ok=True)
    runtime=source_fingerprint()
    request_path=directory/"request.json"
    if request_path.exists():
        request=json.loads(request_path.read_text())
        if (request.get("schema")!=SCHEMA or request.get("seed")!=seed
                or request.get("n")!=n or request.get("stop_after")!=stop_after
                or request.get("runtime_sha256")!=runtime["sha256"]):
            raise ValueError("incompatible mechanism-matrix resume request")
        pool=request["pool"]
    else:
        with (directory/"generation.log").open("w",encoding="utf-8") as log, contextlib.redirect_stdout(log):
            _,session,_,_=make_scene(seed,directory/"decision",domain="development",level=level)
            pool,source=propose(session.decision_observation,cad=session.planning_cad,n=n,seed=seed)
            graphs=[normalized_graph(session,p) for p in pool]
        request=dict(schema=SCHEMA,seed=seed,n=n,stop_after=stop_after,level=level,
            domain="development",pool=pool,source=source,
            graph_sha256=[digest(g) for g in graphs],
            initial_observation_sha256=session.decision_observation["sha256"],
            runtime_sha256=runtime["sha256"],runtime_sources=runtime,
            label_semantics=("physical success of the fresh-scene assembly prefix through the named stage; "
                             "not complete-task success and not suitable for the frozen V12 full-task checkpoint"))
        save(request_path,request)
    trials=[]
    for proposal in pool:
        root=directory/"candidates"/proposal["name"]
        result_path=root/"result.json"
        if result_path.exists():
            result=json.loads(result_path.read_text())
        else:
            result=rollout(seed,proposal,root,domain="development",level=level,
                           stop_after=stop_after)
        if (not result.get("valid") or result.get("evaluation_scope")!=f"assembly_prefix_through_{stop_after}"
                or result.get("runtime_sha256")!=runtime["sha256"]):
            raise RuntimeError(f"invalid mechanism trial: {proposal['name']}")
        trials.append(dict(name=proposal["name"],success=bool(result["success"]),
            error=result.get("error",""),wall_seconds=float(result["total_wall_seconds"]),
            result_sha256=file_sha256(result_path),input_graph_sha256=result["input_graph_sha256"]))
        save(directory/"progress.json",dict(schema=SCHEMA,seed=seed,stop_after=stop_after,
            completed=len(trials),pool_size=n,trials=trials,runtime_sha256=runtime["sha256"]))
    summary=dict(schema=SCHEMA,seed=seed,stop_after=stop_after,pool_size=n,complete=len(trials)==n,
        successes=sum(t["success"] for t in trials),failures=sum(not t["success"] for t in trials),
        mixed=0<sum(t["success"] for t in trials)<n,trials=trials,
        runtime_sha256=runtime["sha256"],full_task_claim=False)
    save(directory/"summary.json",summary)
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--seeds",type=int,nargs="+",required=True)
    p.add_argument("--n",type=int,default=12)
    p.add_argument("--stop-after",choices=("carriage","end_stop"),default="carriage")
    p.add_argument("--level",choices=("L0","L1","L2"),default="L1")
    a=p.parse_args()
    for seed in a.seeds:
        result=collect(seed,a.out/f"seed_{seed}"/a.stop_after,n=a.n,
                       stop_after=a.stop_after,level=a.level)
        print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__": main()
