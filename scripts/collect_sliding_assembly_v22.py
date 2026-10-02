#!/usr/bin/env python3
"""Freeze complete no-wipe sliding-table programs, then execute fresh rollouts."""
from __future__ import annotations

import argparse
import contextlib
from copy import deepcopy
import json
from pathlib import Path
import time
import traceback

from simbench.assembly.candidates import fingerprint
from simbench.assembly.library import HANDLERS
from simbench.value.physical import PhysicalRunner, perturbation
from simbench.value.plan import Call, PlanIR, argument, digest
from simbench.value.planner_v12 import assembly_program, propose
from simbench.value.provenance_v12 import fingerprint as runtime_fingerprint
from simbench.value.system_v12 import make_scene, save

TASK_VERSION = "sliding_assembly_no_wipe_v22"
REQUIRED = ("assembly_pass", "functional_test_pass", "fixture_capture_pass",
            "final_seat_pass", "final_release_and_retraction_pass")


def controller_plan(session, proposal):
    params = dict(order=list(proposal["order"]), choices=deepcopy(proposal["choices"]),
                  stroke_minimum=float(proposal["stroke_minimum"]))
    start = fingerprint(session)
    cid = digest(dict(task=TASK_VERSION, start=start, params=params))[:20]
    call = Call("complete_assembly", "run_sliding_assembly_v22",
                {key: argument(value) for key, value in params.items()}, {}, "executable")
    prefix = dict(id=cid, execution="program", start_state=start,
                  task_scope=TASK_VERSION, steps=[dict(skill=call.skill, params=params)],
                  order=params["order"], choices=params["choices"],
                  stroke_minimum=params["stroke_minimum"],
                  controller_contract="stagewise feedback-bound atomic assembly; final bidirectional stroke")
    return PlanIR(cid, [call], 1, prefix, "unknown", protocol="atomic.program.v1").validate(session.parts)


def collect(seed, out, *, pool_n=48, candidate_n=4, level="L0", domain="online",
            position_noise=0., yaw_noise=0.):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("output directory must be empty before freezing candidates")
    runtime = runtime_fingerprint()
    kwargs = dict(domain=domain, level=level, observation_backend="mujoco_state_pose",
                  position_noise_std_m=position_noise, yaw_noise_std_rad=yaw_noise)
    _, planning, _, _ = make_scene(seed, out / "planning", **kwargs)
    planning.sliding_assembly_v22 = True
    planning.end_stop_place_acceptance_v12 = "stable_supported"
    observation = deepcopy(planning.decision_observation)
    random_state = deepcopy(planning.state_observation_random_state)
    pool, source = propose(observation, cad=planning.planning_cad, n=pool_n, seed=seed,
                           completed=("cleaning",))
    unique = []
    seen = set()
    for proposal in pool:
        proposal = {key: deepcopy(value) for key, value in proposal.items()
                    if key not in {"wipe_variant", "wipe_force", "wipe_duration"}}
        plan = controller_plan(planning, proposal)
        signature = digest(plan.to_dict()["calls"])
        if signature in seen:
            continue
        seen.add(signature)
        assembly = assembly_program(planning, proposal)
        unique.append(dict(name=proposal["name"], proposal=proposal,
                           complete_candidate_plan_ir=plan.to_dict(),
                           plan_sha256=digest(plan.to_dict()),
                           assembly_plan_ir=assembly.to_dict(),
                           assembly_plan_sha256=digest(assembly.to_dict())))
        if len(unique) == candidate_n:
            break
    if len(unique) < candidate_n:
        raise ValueError(f"only {len(unique)} distinct complete programs")
    request = dict(schema="twingraph.sliding_assembly_freeze.v22.r1",
                   task_version=TASK_VERSION, seed=seed, level=level, domain=domain,
                   runtime_sha256=runtime["sha256"], decision_observation=observation,
                   observation_random_state=random_state,
                   observation_backend="mujoco_state_pose",
                   position_noise_std_m=position_noise, yaw_noise_std_rad=yaw_noise,
                   source=source, pool_n=pool_n, candidate_n=candidate_n,
                   selection="first distinct complete assembly programs in pre-outcome proposal order",
                   outcome_labels_read=False, candidates=unique)
    save(out / "request.json", request)
    results = []
    for entry in unique:
        directory = out / "candidates" / entry["name"]
        started = time.perf_counter()
        _, session, _, _ = make_scene(seed, directory / "scene", **kwargs)
        session.sliding_assembly_v22 = True
        session.end_stop_place_acceptance_v12 = "stable_supported"
        if session.decision_observation["sha256"] != observation["sha256"]:
            raise RuntimeError("fresh decision observation differs from frozen request")
        if session.state_observation_random_state != random_state:
            raise RuntimeError("fresh observation random state differs from frozen request")
        plan = PlanIR.from_dict(entry["complete_candidate_plan_ir"]).validate(session.parts)
        if digest(plan.to_dict()) != entry["plan_sha256"]:
            raise RuntimeError("candidate PlanIR changed after freezing")
        session.required_stage_passes = REQUIRED
        graph = dict(schema="twingraph.sliding_assembly_input.v22.r1",
                     task_version=TASK_VERSION, decision_observation=observation,
                     complete_candidate_plan_ir=entry["complete_candidate_plan_ir"],
                     assembly_plan_ir=entry["assembly_plan_ir"],
                     assembly_plan_sha256=entry["assembly_plan_sha256"],
                     planning_cad=session.planning_cad,
                     required_stage_passes=REQUIRED)
        graph_sha = digest(graph)
        save(directory / "input_graph.json", graph)
        trial = perturbation(seed, 0, domain)
        software_exception = None
        with (directory / "console.log").open("w", encoding="utf-8") as log:
            with contextlib.redirect_stdout(log):
                try:
                    result = PhysicalRunner(session, timeout=600.).run(plan, trial, keep_trace=True)
                except Exception as exc:
                    software_exception = dict(type=type(exc).__name__, message=str(exc),
                                              traceback=traceback.format_exc())
                    result = dict(valid=False, success=False, full_success=False,
                                  error=str(exc), timeout=False,
                                  stage_passes=deepcopy(getattr(session, "stage_passes", {})),
                                  wall_seconds=time.perf_counter()-started,
                                  executed_steps=len(session.results))
        stages = list(entry["proposal"]["order"]) + ["stroke", "retention"]
        # A binding marks a reached stage. Completed parts have a successful
        # subsequent binding or aggregate assembly acceptance.
        bindings = [row["stage"] for row in getattr(session, "receiver_binding_history", [])]
        stage_audit = {}
        for index, part in enumerate(entry["proposal"]["order"]):
            reached = part in bindings
            complete = (entry["proposal"]["order"][index + 1] in bindings
                        if index + 1 < len(entry["proposal"]["order"])
                        else bool(result["stage_passes"].get("assembly_pass")))
            stage_audit[part] = dict(reached=reached, completed=bool(complete or result["stage_passes"].get("assembly_pass")))
        stage_audit["stroke"] = dict(reached=bool(result["stage_passes"].get("assembly_pass")),
                                      completed=bool(result["stage_passes"].get("functional_test_pass")))
        stage_audit["retention"] = dict(reached=bool(result["stage_passes"].get("functional_test_pass")),
                                         completed=bool(result["success"]))
        first = next((row for row in session.results if not row["ok"]), None)
        failed_atom = (dict(index=first["index"], skill=first["skill"],
                            interface=first.get("interface"),
                            implementation=HANDLERS[first["skill"]].implementation,
                            reason=first.get("reason"), metrics=first.get("metrics")) if first else None)
        result.update(task_version=TASK_VERSION, evaluation_scope="complete_no_wipe_sliding_assembly",
                      full_task_label=True, seed=seed, level=level, proposal=entry["proposal"],
                      runtime_sha256=runtime["sha256"], input_graph_sha256=graph_sha,
                      plan_sha256=entry["plan_sha256"], decision_observation_sha256=observation["sha256"],
                      observation_backend="mujoco_state_pose", observation_config=observation["config"],
                      observation_random_state=random_state, first_failed_atom=failed_atom,
                      failure_type=("software_exception" if software_exception else
                                    "timeout" if result["timeout"] else "atomic_call" if first else
                                    "final_functional_predicate" if not result["success"] else None),
                      stage_audit=stage_audit,
                      final_functional_predicates=deepcopy(result["stage_passes"]),
                      total_wall_seconds=time.perf_counter()-started,
                      resource_censored=bool(result["timeout"]))
        if result["timeout"]:
            result.update(valid=False, invalid_reason="resource-censored execution")
        if runtime_fingerprint()["sha256"] != runtime["sha256"]:
            result.update(valid=False, invalid_reason="runtime changed during execution")
        save(directory / "result.json", result)
        save(directory / "training_sample.json", dict(
            schema="twingraph.candidate_execution_sample.v22.r1",
            layout=dict(seed=seed, level=level, task_version=TASK_VERSION,
                        runtime_sha256=runtime["sha256"]),
            decision_observation=observation,
            complete_candidate_plan_ir=entry["complete_candidate_plan_ir"],
            assembly_plan_ir=entry["assembly_plan_ir"],
            input_graph_path="input_graph.json", input_graph_sha256=graph_sha,
            actual_execution_label=result["success"] if result["valid"] else None,
            first_failed_atom=failed_atom, failure_type=result["failure_type"],
            stage_audit=stage_audit, final_functional_predicates=result["final_functional_predicates"],
            observation_backend=result["observation_backend"],
            observation_config=result["observation_config"],
            observation_random_state=random_state, trial=trial,
            total_wall_seconds=result["total_wall_seconds"], timeout=result["timeout"],
            software_exception=software_exception))
        results.append(dict(name=entry["name"], valid=result["valid"], success=result["success"],
                            error=result["error"], wall_seconds=result["total_wall_seconds"],
                            first_failed_atom=failed_atom, stage_audit=stage_audit))
        save(out / "progress.json", results)
        print(json.dumps(dict(seed=seed, name=entry["name"], valid=result["valid"],
                              success=result["success"], error=result["error"]),
                         ensure_ascii=False), flush=True)
    summary = dict(task_version=TASK_VERSION, seed=seed, level=level, attempted=len(results),
                   valid=sum(row["valid"] for row in results),
                   successes=sum(row["valid"] and row["success"] for row in results),
                   failures=sum(row["valid"] and not row["success"] for row in results),
                   candidates=results)
    save(out / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pool-n", type=int, default=48)
    parser.add_argument("--candidate-n", type=int, default=4)
    parser.add_argument("--level", choices=("L0", "L1", "L2"), default="L0")
    parser.add_argument("--domain", choices=("online", "development", "regression"), default="online")
    parser.add_argument("--position-noise-std-m", type=float, default=0.)
    parser.add_argument("--yaw-noise-std-rad", type=float, default=0.)
    args = parser.parse_args()
    if not 1 <= args.candidate_n <= args.pool_n <= 512:
        parser.error("candidate count must be in [1, pool-n] and pool-n <= 512")
    if min(args.position_noise_std_m, args.yaw_noise_std_rad) < 0:
        parser.error("noise standard deviations must be nonnegative")
    print(json.dumps(collect(args.seed, args.out, pool_n=args.pool_n,
        candidate_n=args.candidate_n, level=args.level, domain=args.domain,
        position_noise=args.position_noise_std_m, yaw_noise=args.yaw_noise_std_rad),
        ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
