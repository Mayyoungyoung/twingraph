"""Development-only nested candidate expansion, reusing hash-equal N=8 trials."""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import shutil

from scripts.run_v13_mechanism_matrix import SCHEMA
from simbench.value.plan import digest
from simbench.value.planner_v12 import normalized_graph, propose
from simbench.value.provenance_v12 import fingerprint
from simbench.value.system_v11 import save
from simbench.value.system_v12 import make_scene, rollout


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-n8", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--level", default="L2", choices=("L0", "L1", "L2"))
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    source_request = json.loads((args.source_n8 / "request.json").read_text())
    runtime = fingerprint()
    with (args.out / "generation.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        _, session, _, _ = make_scene(args.seed, args.out / "decision", domain="development", level=args.level)
        pool, source = propose(session.decision_observation, cad=session.planning_cad, n=args.n, seed=args.seed)
        graphs = {proposal["name"]: normalized_graph(session, proposal) for proposal in pool}
    source_pool = {p["name"]: p for p in source_request["pool"]}
    new_pool = {p["name"]: p for p in pool}
    for name, proposal in source_pool.items():
        if name not in new_pool or digest(proposal) != digest(new_pool[name]):
            raise ValueError(f"N=8 is not a semantic prefix of N={args.n}: {name}")
    request = dict(schema=SCHEMA, seed=args.seed, n=args.n, stop_after="end_stop", level=args.level,
                   domain="development", pool=pool, source=source,
                   graph_sha256=[digest(graphs[p["name"]]) for p in pool],
                   initial_observation_sha256=session.decision_observation["sha256"],
                   runtime_sha256=runtime["sha256"], runtime_sources=runtime,
                   label_semantics="development physical prefix through end_stop; not full task",
                   nested_reuse=dict(source_request=str(args.source_n8 / "request.json"),
                                     source_request_sha256=hashlib.sha256((args.source_n8 / "request.json").read_bytes()).hexdigest(),
                                     reused_candidates=sorted(source_pool), newly_executed_candidates=sorted(set(new_pool)-set(source_pool))))
    save(args.out / "request.json", request)
    trials = []
    for proposal in pool:
        target = args.out / "candidates" / proposal["name"]
        target.mkdir(parents=True, exist_ok=True)
        if proposal["name"] in source_pool:
            source_dir = args.source_n8 / "candidates" / proposal["name"]
            for filename in ("result.json", "input_graph.json"):
                shutil.copy2(source_dir / filename, target / filename)
            result = json.loads((target / "result.json").read_text()); reused = True
        else:
            result = rollout(args.seed, proposal, target, domain="development", level=args.level,
                             stop_after="end_stop"); reused = False
        if (result.get("valid") is not True or result.get("proposal") != proposal
                or result.get("runtime_sha256") != runtime["sha256"]):
            raise ValueError(f"invalid expansion binding: {proposal['name']}")
        trials.append(dict(name=proposal["name"], success=bool(result["success"]), reused_n8=reused,
                           error=result.get("error", ""), wall_seconds=float(result["total_wall_seconds"])))
        save(args.out / "progress.json", dict(completed=len(trials), pool_size=args.n, trials=trials))
    successes = sum(t["success"] for t in trials)
    summary = dict(schema=SCHEMA, seed=args.seed, stop_after="end_stop", pool_size=args.n,
                   complete=True, successes=successes, failures=args.n-successes,
                   mixed=0 < successes < args.n, trials=trials,
                   runtime_sha256=runtime["sha256"], full_task_claim=False,
                   nested_reuse=request["nested_reuse"])
    save(args.out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
