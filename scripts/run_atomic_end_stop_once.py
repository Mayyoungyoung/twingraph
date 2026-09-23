#!/usr/bin/env python3
"""Execute one end-stop atomic PlanIR directly from a legal twin checkpoint."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_v17a_end_stop_place import obtain_carriage_checkpoint
from simbench.assembly.library import SkillFailure
from simbench.value.plan import PlanIR, digest, execute_calls, plain
from simbench.value.skill_graph import compile_graph
from simbench.value.system_v11 import restore_twin_checkpoint, save
from simbench.value.system_v12 import make_scene, require_frozen_source, _prepare_receiver_targets
from simbench.value.planner_v11 import assembly_program, normalized_graph
from simbench.value.planner_v12 import propose


def direct_plan(session, proposal):
    _prepare_receiver_targets(session, "end_stop", proposal)
    full = assembly_program(session, proposal)
    calls = [c for c in full.calls if c.roles.get("manipulated") == "end_stop"
             and not c.id.startswith("accept_")]
    prefix = dict(id="pending", execution="program",
                  steps=[dict(skill=c.skill,
                              params={k: plain(v.value) for k, v in c.arguments.items()})
                         for c in calls])
    plan_id = digest(dict(calls=plain([asdict(c) for c in calls]),
                          observation=session.decision_observation.get("sha256")))[:20]
    prefix["id"] = plan_id
    plan = PlanIR(plan_id, calls, len(calls), prefix,
                  "unknown", protocol="atomic.program.v1")
    return plan.validate(session.parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plan", type=Path, help="Execute this validated atomic PlanIR instead of the solver template")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    runtime = require_frozen_source()
    _, planning, _, _ = make_scene(args.seed, args.out / "planning", level="L1")
    pool, source = propose(planning.decision_observation, cad=planning.planning_cad,
                           n=48, seed=args.seed)
    pool.sort(key=lambda p: int(p["name"].rsplit("_", 1)[-1]))
    selected = pool[args.index]
    checkpoint, attempts = obtain_carriage_checkpoint(args.seed, pool,
        args.out / "legal_predecessor_scan", level="L1", record=False)
    if checkpoint is None:
        raise RuntimeError("no legal carriage predecessor checkpoint")
    save(args.out / "checkpoint.json", checkpoint)
    physics = checkpoint["physics"]
    physical_checkpoint_sha256 = digest(dict(
        geometry=checkpoint["clone_geometry_sha256"], data=physics["data"],
        model=physics["model"], time=physics["time"], clock=physics["clock"],
        clients=physics["clients"], held=checkpoint["held"],
        rotation=checkpoint["rotation"], grasp_epoch=checkpoint["grasp_epoch"]))
    predecessor = next(p for p in pool if p["name"] == attempts[-1]["name"])
    selected = json.loads(json.dumps(selected))
    selected["choices"]["carriage"] = predecessor["choices"]["carriage"]
    selected["necessary_geometry"]["carriage"] = predecessor["necessary_geometry"]["carriage"]
    _, session, _, _ = make_scene(args.seed, args.out / "scene", level="L1")
    restore_twin_checkpoint(session, checkpoint)
    session.end_stop_place_acceptance_v12 = "stable_supported"
    if args.plan is None:
        plan = direct_plan(session, selected)
    else:
        _prepare_receiver_targets(session, "end_stop", selected)
        plan = PlanIR.from_dict(json.loads(args.plan.read_text(encoding="utf-8")))
    # Reuse the V12 normalized observation while compiling only the direct
    # suffix calls. No outcome or simulator pose oracle enters the graph.
    observation = normalized_graph(session, selected, completed=("carriage",))["assembly"]["observation"]
    if args.plan is not None and plan.prefix.get("observation_sha256") != digest(observation):
        raise ValueError("model plan was composed for a different initial observation")
    save(args.out / "plan.json", plan.to_dict())
    graph = dict(schema="twingraph.atomic_flow_graph.v1",
                 assembly=compile_graph(observation, plan))
    save(args.out / "input_graph.json", graph)
    started = time.perf_counter()
    error = None
    try:
        execute_calls(session, plan, plan.calls)
    except SkillFailure as exc:
        error = f"{type(exc).__name__}: {exc}"
    result = dict(valid=True, success=error is None, error=error,
                  evaluation_scope="end_stop_stable_placement_only",
                  input_graph_sha256=digest(graph), plan_sha256=digest(plan.to_dict()),
                  runtime_sha256=runtime["sha256"], checkpoint_sha256=digest(checkpoint),
                  physical_checkpoint_sha256=physical_checkpoint_sha256,
                  candidate=selected["name"],
                  seed=args.seed, wall_seconds=time.perf_counter()-started,
                  source="direct atomic PlanIR execution from exact physical predecessor checkpoint",
                  full_task_success=None)
    save(args.out / "result.json", result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
