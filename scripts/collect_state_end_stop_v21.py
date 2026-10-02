#!/usr/bin/env python3
"""Freeze and execute a complete, short printed-kit mounting task.

The guide base is fixed. Each candidate uses the normal observation-grounded
V12 end-stop choice and a complete PlanIR: detect, grasp, transfer, seat,
release, retreat, and independent stable-support acceptance. No outcome is
read until the unique candidate pool and all plan hashes have been frozen.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

from simbench.assembly.library import HANDLERS
from simbench.assembly.skills_v12 import evaluate_end_stop_stable_support
from simbench.value.physical import PhysicalRunner, perturbation
from simbench.value.plan import PlanIR, digest
from simbench.value.planner_v12 import assembly_program, propose
from simbench.value.provenance_v12 import fingerprint as runtime_fingerprint
from simbench.value.system_v12 import make_scene, save

TASK_VERSION = "printed_end_stop_mount_v21"
PART = "end_stop"


def short_plan(session, proposal):
    session.end_stop_place_acceptance_v12 = "stable_supported"
    full = assembly_program(session, proposal)
    calls = [deepcopy(call) for call in full.calls
             if call.roles.get("manipulated") == PART and not call.id.startswith("accept_")]
    prefix = deepcopy(full.prefix)
    prefix.update(part=PART, order=[PART], choices={PART: deepcopy(proposal["choices"][PART])},
                  targets={PART: deepcopy(session.stage_targets[PART])},
                  task_scope=TASK_VERSION, completed_parts=[],
                  steps=[dict(skill=call.skill,
                              params={key: argument.value for key, argument in call.arguments.items()})
                         for call in calls])
    prefix["semantic_program_id"] = digest(dict(task=TASK_VERSION, calls=prefix["steps"]))
    prefix["id"] = digest(dict(start=prefix["start_state"], task=TASK_VERSION,
                                calls=prefix["steps"]))[:20]
    return PlanIR(prefix["id"], calls, len(calls), prefix, "unknown",
                  protocol="assembly.program.feedback.v2").validate(session.parts)


def stage_audit(steps):
    success = [row for row in steps if row["ok"]]
    attempted = [row["skill"] for row in steps]
    completed = [row["skill"] for row in success]
    return dict(
        grasp=dict(reached=bool(attempted), completed="verify_grasp" in completed),
        seat=dict(reached="verify_grasp" in completed, completed="press_seat" in completed),
        release_and_verify=dict(reached="press_seat" in completed,
                                completed="inspect_stable_support" in completed))


def collect(seed, out, *, pool_n, candidate_n, level, domain, position_noise, yaw_noise):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("output directory must be empty so the request is frozen before labels")
    runtime = runtime_fingerprint()
    backend = "mujoco_state_pose"
    _, planning, _, _ = make_scene(seed, out / "planning", domain=domain, level=level,
        observation_backend=backend, position_noise_std_m=position_noise,
        yaw_noise_std_rad=yaw_noise)
    pool, source = propose(planning.decision_observation,
        cad=planning.planning_cad, n=pool_n, seed=seed)
    unique = []
    seen = set()
    for proposal in pool:
        plan = short_plan(planning, proposal)
        signature = digest(plan.to_dict()["calls"])
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(dict(name=proposal["name"], proposal=proposal,
                           plan=plan.to_dict(), plan_sha256=digest(plan.to_dict())))
        if len(unique) == candidate_n:
            break
    if len(unique) < candidate_n:
        raise ValueError(f"only {len(unique)} distinct executable mounting programs")
    observation = deepcopy(planning.decision_observation)
    observation_random_state = deepcopy(planning.state_observation_random_state)
    request = dict(schema="twingraph.short_mount_freeze.v21.r1", seed=seed,
        task_version=TASK_VERSION, layout_level=level, domain=domain,
        observation_backend=backend, position_noise_std_m=position_noise,
        yaw_noise_std_rad=yaw_noise, runtime_sha256=runtime["sha256"],
        decision_observation=observation, source=source,
        observation_random_state=observation_random_state,
        pool_n=pool_n, candidate_n=candidate_n, candidates=unique,
        selection="first distinct end-stop executable programs in pre-outcome proposal order",
        outcome_labels_read=False)
    save(out / "request.json", request)
    results = []
    for entry in unique:
        name = entry["name"]
        directory = out / "candidates" / name
        started = time.perf_counter()
        _, session, _, _ = make_scene(seed, directory / "scene", domain=domain, level=level,
            observation_backend=backend, position_noise_std_m=position_noise,
            yaw_noise_std_rad=yaw_noise)
        session.end_stop_place_acceptance_v12 = "stable_supported"
        if session.decision_observation["sha256"] != observation["sha256"]:
            raise RuntimeError("fresh scene decision observation differs from frozen request")
        if session.state_observation_random_state != observation_random_state:
            raise RuntimeError("fresh scene observation random state differs from frozen request")
        plan = PlanIR.from_dict(entry["plan"]).validate(session.parts)
        if digest(plan.to_dict()) != entry["plan_sha256"]:
            raise RuntimeError("candidate PlanIR differs from frozen request")
        graph = dict(schema="twingraph.short_mount_input.v21.r1",
            decision_observation=observation, plan_ir=entry["plan"],
            plan_sha256=entry["plan_sha256"], task_version=TASK_VERSION,
            planning_cad=session.planning_cad)
        graph_sha = digest(graph)
        save(directory / "input_graph.json", graph)
        with (directory / "console.log").open("w", encoding="utf-8") as log:
            import contextlib
            with contextlib.redirect_stdout(log):
                trial = perturbation(seed, 0, domain)
                result = PhysicalRunner(session, timeout=600.).run(plan, trial, keep_trace=True)
        # The last PlanIR checker already evaluates after release and retreat.
        # These independent measurements are persisted even on a failed run.
        final_ok, final_metrics = evaluate_end_stop_stable_support(
            session, PART, session.stage_targets[PART], after_retreat=False)
        predicate = dict(stable_supported=bool(final_ok),
            region_ok=bool(final_metrics["region_ok"]),
            support_contact=bool(final_metrics["supported"]),
            released=bool(session.held is None),
            no_gross_penetration=not bool(final_metrics["gross_penetration"]))
        result["success"] = bool(result["success"] and all(predicate.values()))
        result["full_success"] = result["success"]
        result.update(evaluation_scope=TASK_VERSION, full_task_label=True,
            task_version=TASK_VERSION, layout_level=level, seed=seed,
            proposal=entry["proposal"], decision_observation_sha256=observation["sha256"],
            input_graph_sha256=graph_sha, plan_sha256=entry["plan_sha256"],
            runtime_sha256=runtime["sha256"], observation_backend=backend,
            observation_config=observation["config"], final_functional_predicates=predicate,
            observation_random_state=observation_random_state,
            final_acceptance_metrics=final_metrics,
            stage_audit=stage_audit(session.results),
            total_wall_seconds=time.perf_counter()-started,
            resource_censored=bool(result["timeout"]))
        if runtime_fingerprint()["sha256"] != runtime["sha256"]:
            result.update(valid=False, invalid_reason="runtime source changed during execution")
        first = next((row for row in session.results if not row["ok"]), None)
        result["first_failed_atom"] = (dict(skill=first["skill"], interface=first["interface"],
            implementation=HANDLERS[first["skill"]].implementation,
            reason=first["reason"], metrics=first["metrics"]) if first else None)
        result["failure_type"] = ("timeout" if result["timeout"] else
            "atomic_call" if first else "final_functional_predicate" if not result["success"] else None)
        if result["timeout"]:
            result.update(valid=False, invalid_reason="resource-censored; no physical feasibility label")
        save(directory / "result.json", result)
        save(directory / "training_sample.json", dict(
            schema="twingraph.candidate_execution_sample.v21.r1",
            layout=dict(seed=seed, level=level, task_version=TASK_VERSION,
                        runtime_sha256=runtime["sha256"]),
            decision_observation=observation, complete_candidate_plan_ir=entry["plan"],
            input_graph_path="input_graph.json", input_graph_sha256=graph_sha,
            actual_execution_label=result["success"] if result["valid"] else None,
            first_failed_atom=result["first_failed_atom"], failure_type=result["failure_type"],
            stage_audit=result["stage_audit"], final_functional_predicates=predicate,
            observation_backend=backend, observation_config=observation["config"],
            observation_random_state=observation_random_state,
            trial=trial, total_wall_seconds=result["total_wall_seconds"],
            timeout=result["timeout"], software_exception=None))
        results.append(dict(name=name, valid=result["valid"], success=result["success"],
                            error=result["error"], wall_seconds=result["total_wall_seconds"],
                            first_failed_atom=result["first_failed_atom"]))
        save(out / "progress.json", results)
        print(json.dumps(dict(name=name, valid=result["valid"], success=result["success"],
                              error=result["error"]), ensure_ascii=False), flush=True)
    summary = dict(task_version=TASK_VERSION, seed=seed, attempted=len(results),
        valid=sum(row["valid"] for row in results),
        successes=sum(row["valid"] and row["success"] for row in results),
        failures=sum(row["valid"] and not row["success"] for row in results),
        pool_type=("mixed" if any(row["success"] for row in results) and
                   any(not row["success"] for row in results) else
                   "all_positive" if all(row["success"] for row in results) else "all_negative"),
        candidates=results)
    save(out / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pool-n", type=int, default=48)
    parser.add_argument("--candidate-n", type=int, default=12)
    parser.add_argument("--level", choices=("L0", "L1", "L2"), default="L0")
    parser.add_argument("--domain", choices=("online", "development", "regression"), default="online")
    parser.add_argument("--position-noise-std-m", type=float, default=0.)
    parser.add_argument("--yaw-noise-std-rad", type=float, default=0.)
    args = parser.parse_args()
    if not 1 <= args.candidate_n <= args.pool_n <= 512:
        parser.error("candidate count must be in [1,pool-n] and pool-n <= 512")
    if min(args.position_noise_std_m, args.yaw_noise_std_rad) < 0:
        parser.error("noise standard deviations must be nonnegative")
    print(json.dumps(collect(args.seed, args.out, pool_n=args.pool_n,
        candidate_n=args.candidate_n, level=args.level, domain=args.domain,
        position_noise=args.position_noise_std_m, yaw_noise=args.yaw_noise_std_rad),
        ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
