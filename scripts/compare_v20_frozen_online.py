#!/usr/bin/env python3
"""Serial online comparison on the same pre-outcome frozen complete-plan pool.

Each method replays fresh physical twins in its own directory. The saved pool
and input graph hashes are checked before any method sees a rollout outcome.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from simbench.value.plan import digest
from simbench.value.planner_v12 import normalized_graph
from simbench.value.system_v12 import (
    make_scene, require_frozen_source, rollout, save,
    verification_schedule,
)

METHODS = ("all_twin", "random_top_k", "value_top_k")


def bind_frozen(request_path, seed, level, source, decision_dir):
    request_path = Path(request_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if (request.get("seed") != seed or request.get("level") != level
            or request.get("domain") != "online"
            or request.get("runtime_sha256") != source["sha256"]):
        raise ValueError("frozen request differs from seed, level, domain or physical runtime")
    pool = request.get("pool")
    if not isinstance(pool, list) or not pool or len({p["name"] for p in pool}) != len(pool):
        raise ValueError("frozen complete-plan pool is empty or has repeated names")
    hashes = request.get("graph_sha256")
    if not isinstance(hashes, dict) or set(hashes) != {p["name"] for p in pool}:
        raise ValueError("every complete plan must have a pre-outcome graph hash")
    _, session, _, _ = make_scene(seed, decision_dir, domain="online", level=level)
    if session.decision_observation["sha256"] != request["initial_observation"]["sha256"]:
        raise ValueError("fresh random layout differs from frozen decision observation")
    graphs = [normalized_graph(session, proposal) for proposal in pool]
    if any(digest(graph) != hashes[proposal["name"]]
           for proposal, graph in zip(pool, graphs)):
        raise ValueError("executed graph differs from the pre-outcome frozen input")
    return request, pool, graphs


def online_method(seed, method, *, request_path, out, value, k, level, preflight):
    if method not in METHODS:
        raise ValueError(f"unsupported comparison method: {method}")
    if method == "value_top_k" and value is None:
        raise ValueError("value_top_k requires a complete-task value checkpoint")
    out = Path(out)
    if (out / "summary.json").exists():
        raise ValueError(f"comparison method already completed: {out}")
    if any((out / "twins").glob("*/result.json")):
        raise ValueError(f"partial online run requires a new output directory: {out}")
    started = time.perf_counter()
    source = require_frozen_source()
    request, pool, graphs = bind_frozen(request_path, seed, level, source,
                                        out / "decision")
    rehydration_seconds = time.perf_counter() - started
    ranked = time.perf_counter()
    scores, order, budget, _ = verification_schedule(method, graphs, value, seed, k)
    ranking_seconds = time.perf_counter() - ranked
    request_bytes = Path(request_path).read_bytes()
    frozen_sha = hashlib.sha256(request_bytes).hexdigest()
    setup = dict(seed=seed, method=method, pool_size=len(pool), k=budget,
                 frozen_request_sha256=frozen_sha,
                 frozen_graph_sha256=request["graph_sha256"],
                 source_generation_seconds=request.get("generation_seconds"),
                 source_construction=request.get("source", {}).get("construction"),
                 physical_runtime_sha256=source["sha256"],
                 model_sha256=getattr(value, "sha256", None),
                 order=[pool[i]["name"] for i in order],
                 scores=[float(x) for x in scores] if scores is not None else None,
                 request_rehydration_seconds=rehydration_seconds,
                 ranking_seconds=ranking_seconds)
    out.mkdir(parents=True, exist_ok=True)
    save(out / "preflight.json", setup)
    if preflight:
        return dict(**{key: setup[key] for key in
                       ("seed", "method", "pool_size", "k", "frozen_request_sha256",
                        "physical_runtime_sha256")}, preflight_only=True)
    verified = []
    selected = None
    for index in order[:budget]:
        proposal = pool[index]
        result = rollout(seed, proposal, out / "twins" / proposal["name"],
                         domain="online", level=level)
        if (not result.get("valid") or result.get("resource_censored")
                or result.get("runtime_sha256") != source["sha256"]
                or result.get("proposal") != proposal
                or result.get("input_graph_sha256") != request["graph_sha256"][proposal["name"]]
                or result.get("evaluation_scope") != "complete_functional_task"
                or result.get("full_task_label") is not True
                or result.get("full_success") != result.get("success")):
            raise RuntimeError(f"invalid complete-task digital twin: {proposal['name']}")
        save(out / "twins" / proposal["name"] / "result.json", result)
        verified.append(dict(name=proposal["name"], full_success=result["full_success"],
                             wall_seconds=result["total_wall_seconds"],
                             first_error=str(result.get("error") or "").split(" {", 1)[0]))
        if result["full_success"] and selected is None:
            selected = index
        save(out / "progress.json", dict(verified=verified,
                                        selected=pool[selected]["name"] if selected is not None else None))
        if selected is not None and method != "all_twin":
            break
    decision_seconds = time.perf_counter() - started
    deployment = None
    if selected is not None:
        deployment = rollout(seed, pool[selected], out / "deployment",
                             domain="deployment", level=level)
        if not deployment.get("valid") or deployment.get("resource_censored"):
            raise RuntimeError("invalid deployment after verified full-task plan")
        save(out / "deployment" / "result.json", deployment)
    report = dict(schema="twingraph.frozen_full_task_online_comparison.v20.r1",
                  label_scope="complete_functional_task", **setup,
                  deployment_policy="execute_selected_complete_plan_without_replanning",
                  twin_verified=verified,
                  twin_success=selected is not None,
                  selected=pool[selected]["name"] if selected is not None else None,
                  deployment_full_success=bool(deployment and deployment["full_success"]),
                  deployment_error=(str(deployment.get("error") or "").split(" {", 1)[0]
                                    if deployment else "no_verified_plan"),
                  decision_seconds=decision_seconds,
                  deployment_seconds=deployment["total_wall_seconds"] if deployment else 0.,
                  total_wall_seconds=time.perf_counter()-started)
    save(out / "summary.json", report)
    require_frozen_source()
    return {key: report[key] for key in ("seed", "method", "pool_size", "k",
                                         "twin_success", "selected",
                                         "deployment_full_success", "decision_seconds",
                                         "total_wall_seconds")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--level", default="L1", choices=("L0", "L1", "L2"))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    source = require_frozen_source()
    value = None
    if "value_top_k" in args.methods:
        if args.checkpoint is None:
            parser.error("value_top_k requires --checkpoint")
        import torch
        from simbench.value.full_flow_graph_value_v20 import SCHEMA, ValueRankerV20
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if (checkpoint.get("schema") != SCHEMA
                or checkpoint.get("label_scope") != "complete_functional_task"
                or checkpoint.get("physical_runtime_sha256") != source["sha256"]):
            raise ValueError("value checkpoint is not trained for this complete-task physical runtime")
        value = ValueRankerV20(args.checkpoint)
    for seed in args.seeds:
        request = args.pool_root / f"seed_{seed}" / "collect" / "request.json"
        if not request.is_file():
            raise FileNotFoundError(request)
        for method in args.methods:
            report = online_method(seed, method, request_path=request,
                                   out=args.out / f"seed_{seed}" / method,
                                   value=value, k=args.k, level=args.level,
                                   preflight=args.preflight_only)
            print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
