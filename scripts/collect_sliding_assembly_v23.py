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
from simbench.assembly.sensor_learning_v12 import (
    PIN_PRESS_BUDGET_POLICIES,
    PIN_PRESS_REMAINING_V28,
)
from simbench.assembly.constrained_stroke_v29 import (
    STROKE_MOTION_POLICIES,
    STROKE_PROGRESS_V29,
)
from simbench.value.physical import PhysicalRunner, perturbation
from simbench.value.plan import Call, PlanIR, argument, digest
from simbench.value.planner_v12 import assembly_program, propose
from simbench.value.provenance_v12 import fingerprint as runtime_fingerprint
from simbench.value.system_v12 import make_scene, save

TASK_VERSION = "sliding_assembly_generic_transport_handle_terminal_v32"
REQUIRED = ("assembly_pass", "base_hole_engagement_pass", "fixture_capture_pass",
            "final_seat_pass", "final_release_and_retraction_pass")


def controller_plan(session, proposal, *, pin_press_budget_policy="remaining_distance_v28",
                    stroke_motion_policy=STROKE_PROGRESS_V29):
    params = dict(order=list(proposal["order"]), choices=deepcopy(proposal["choices"]),
                  pin_press_budget_policy=str(pin_press_budget_policy))
    start = fingerprint(session)
    cid = digest(dict(task=TASK_VERSION, start=start, params=params))[:20]
    call = Call("complete_assembly", "run_sliding_assembly_v23",
                {key: argument(value) for key, value in params.items()}, {}, "executable")
    prefix = dict(id=cid, execution="program", start_state=start,
                  task_scope=TASK_VERSION, steps=[dict(skill=call.skill, params=params)],
                  order=params["order"], choices=params["choices"],
                  controller_contract=("stagewise feedback-bound atomic assembly; "
                                       "planner chooses held insertion or released rear push; "
                                       "fresh carriage pose before handle planning; "
                                       "handle installation is terminal; "
                                       f"pin press budget {pin_press_budget_policy}"))
    return PlanIR(cid, [call], 1, prefix, "unknown", protocol="atomic.program.v1").validate(session.parts)


def failure_category(first, error, timeout, software_exception):
    if software_exception:
        return "software_exception"
    if timeout:
        return "resource_timeout"
    if first:
        skill, reason = first["skill"], str(first.get("reason") or "").lower()
        if skill == "select_grasp" and "no materialized grasp" in reason:
            return "grasp_solver_budget_exhausted"
        if "collision" in reason or skill == "plan_transfer":
            return "path_check_rejected"
        if skill.startswith("inspect"):
            return "functional_acceptance_failed"
        return "control_or_contact_failed"
    if "shared shaft route" in error or "two-layer receiver" in error:
        return "geometric_precondition_rejected"
    return "final_functional_acceptance_failed" if error else None


def collect(seed, out, *, pool_n=48, candidate_n=8, level="L0", domain="online",
            position_noise=0., yaw_noise=0., layout_family="broad",
            candidate_start=0, parent_request=None, neighborhood_parent_seed=2050,
            split_group_override=None, pin_press_budget_policy=PIN_PRESS_REMAINING_V28,
            stroke_motion_policy=STROKE_PROGRESS_V29, execute_start=0, execute_stop=None,
            timeout_seconds=600., planner="grounded", frozen_request=None):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("output directory must be empty before freezing candidates")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    runtime = runtime_fingerprint()
    if pin_press_budget_policy not in PIN_PRESS_BUDGET_POLICIES:
        raise ValueError("unknown pin press budget policy")
    if stroke_motion_policy not in STROKE_MOTION_POLICIES:
        raise ValueError("unknown constrained stroke motion policy")
    if layout_family not in ("broad", "neighborhood"):
        raise ValueError("unknown declared layout family")
    if layout_family == "neighborhood" and level != "L0":
        raise ValueError("the declared neighborhood is defined only for L0")
    if layout_family == "neighborhood":
        from functools import partial
        from simbench.value.supply_layout_v23 import apply
        layout_hook = partial(apply, parent_seed=neighborhood_parent_seed)
    else:
        layout_hook = None
    kwargs = dict(domain=domain, level=level, observation_backend="mujoco_state_pose",
                  scene_layout_hook=layout_hook,
                  position_noise_std_m=position_noise, yaw_noise_std_rad=yaw_noise)
    _, planning, _, _ = make_scene(seed, out / "planning", **kwargs)
    layout_provenance = json.loads((out / "planning" / "geometry_manifest.json").read_text(
        encoding="utf-8"))["scene_layout"]
    split_group = layout_provenance.get("split_group", f"broad_seed_{seed}")
    if layout_family == "broad" and seed == 2050:
        split_group = "neighborhood_parent_2050"
    if split_group_override is not None:
        if not split_group_override or not split_group_override.startswith("neighborhood_parent_"):
            raise ValueError("invalid declared root split group")
        split_group = split_group_override
    planning.sliding_assembly_v23 = True
    planning.end_stop_place_acceptance_v12 = "stable_supported"
    observation = deepcopy(planning.decision_observation)
    random_state = deepcopy(planning.state_observation_random_state)
    frozen = None
    if frozen_request is not None:
        frozen=json.loads(Path(frozen_request).read_text(encoding="utf-8"))
        if (frozen["task_version"] != TASK_VERSION or frozen["runtime_sha256"] != runtime["sha256"]
                or frozen["seed"] != seed or frozen["domain"] != domain
                or frozen["level"] != level or frozen["candidate_n"] != candidate_n
                or frozen["decision_observation"] != observation
                or frozen["observation_random_state"] != random_state):
            raise ValueError("frozen request runtime, layout or observation mismatch")
        pool=[deepcopy(row["proposal"]) for row in frozen["candidates"]]
        source=deepcopy(frozen["source"])
        planner=frozen["planner"]
    else:
        pool, source = propose(observation, cad=planning.planning_cad, n=pool_n, seed=seed,
                               completed=("cleaning",))
    if frozen is None and planner == "llm":
        if candidate_start:
            raise ValueError("LLM collection requires a fresh frozen pool")
        from scripts.llm_transport_v32 import choose
        pool, model_source = choose(pool, observation, planning.planning_cad, out/"planner", candidate_n)
        source = dict(grounding=source, **model_source)
    elif planner not in ("grounded", "llm"):
        raise ValueError("unknown planner")
    unique = []
    seen = set()
    for proposal in pool:
        proposal = {key: deepcopy(value) for key, value in proposal.items()
                    if key not in {"wipe_variant", "wipe_force", "wipe_duration"}}
        if any(row.get("status") == "rejected"
               for row in proposal.get("necessary_geometry", {}).values()):
            raise ValueError("generator returned a known rejected geometry")
        plan = controller_plan(planning, proposal,
                               pin_press_budget_policy=pin_press_budget_policy,
                               stroke_motion_policy=stroke_motion_policy)
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
        if len(unique) == candidate_start + candidate_n:
            break
    if len(unique) < candidate_start + candidate_n:
        raise ValueError(f"only {len(unique)} distinct complete programs")
    if frozen is not None and [(x["name"], x["plan_sha256"], x["assembly_plan_sha256"]) for x in unique] != [
            (x["name"], x["plan_sha256"], x["assembly_plan_sha256"]) for x in frozen["candidates"]]:
        raise ValueError("recompiled frozen candidate differs from its declared program")
    parent_sha = None
    if candidate_start:
        if parent_request is None:
            raise ValueError("append collection requires the original frozen request")
        parent = json.loads(Path(parent_request).read_text(encoding="utf-8"))
        if (parent.get("task_version") != TASK_VERSION or parent.get("seed") != seed
                or parent.get("level") != level or parent.get("domain") != domain
                or parent.get("layout_family") != layout_family
                or parent.get("split_group") != split_group
                or parent.get("runtime_sha256") != runtime["sha256"]
                or parent.get("pool_n") != pool_n
                or parent.get("position_noise_std_m") != position_noise
                or parent.get("yaw_noise_std_rad") != yaw_noise
                or parent.get("pin_press_budget_policy") != pin_press_budget_policy
                or parent.get("decision_observation") != observation
                or parent.get("observation_random_state") != random_state
                or parent.get("candidate_n") != candidate_start):
            raise ValueError("append request differs from the original frozen pool")
        prefix = parent["candidates"]
        if [(row["name"], row["plan_sha256"], row["assembly_plan_sha256"])
                for row in prefix] != [
                (row["name"], row["plan_sha256"], row["assembly_plan_sha256"])
                for row in unique[:candidate_start]]:
            raise ValueError("regenerated proposal prefix differs from original freeze")
        parent_sha = digest(parent)
    elif parent_request is not None:
        raise ValueError("parent request is only used for an append batch")
    unique = unique[candidate_start:candidate_start + candidate_n]
    if execute_stop is None:
        execute_stop = len(unique)
    if not 0 <= execute_start < execute_stop <= len(unique):
        raise ValueError("execution window must fit within the frozen candidates")
    request = dict(schema="twingraph.sliding_assembly_freeze.v23.r1",
                   task_version=TASK_VERSION, seed=seed, level=level, domain=domain,
                   runtime_sha256=runtime["sha256"], decision_observation=observation,
                   observation_random_state=random_state,
                   observation_backend="mujoco_state_pose",
                   layout_family=layout_family, split_group=split_group,
                   pin_press_budget_policy=pin_press_budget_policy,
                   pin_press_controller_version=PIN_PRESS_BUDGET_POLICIES[
                       pin_press_budget_policy]["controller_version"],
                   legacy_stroke_motion_policy_ignored=stroke_motion_policy,
                   layout_provenance=layout_provenance,
                   position_noise_std_m=position_noise, yaw_noise_std_rad=yaw_noise,
                   source=source, pool_n=pool_n, candidate_n=candidate_n,
                   planner=planner,
                   frozen_request_sha256=digest(frozen) if frozen else None,
                   selection=("first distinct complete assembly programs in pre-outcome proposal order"
                              if not candidate_start else
                              "next distinct complete assembly programs after verified frozen prefix"),
                   candidate_start=candidate_start, parent_request_sha256=parent_sha,
                   development_followup=bool(candidate_start),
                   layout_selected_from_prior_outcomes=bool(candidate_start),
                   outcome_labels_read=False, candidates=unique,
                   execution_window=[execute_start, execute_stop],
                   execution_timeout_seconds=float(timeout_seconds))
    save(out / "request.json", request)
    results = []
    for entry in unique[execute_start:execute_stop]:
        directory = out / "candidates" / entry["name"]
        started = time.perf_counter()
        _, session, _, _ = make_scene(seed, directory / "scene", **kwargs)
        session.sliding_assembly_v23 = True
        session.end_stop_place_acceptance_v12 = "stable_supported"
        if session.decision_observation["sha256"] != observation["sha256"]:
            raise RuntimeError("fresh decision observation differs from frozen request")
        if session.state_observation_random_state != random_state:
            raise RuntimeError("fresh observation random state differs from frozen request")
        plan = PlanIR.from_dict(entry["complete_candidate_plan_ir"]).validate(session.parts)
        if digest(plan.to_dict()) != entry["plan_sha256"]:
            raise RuntimeError("candidate PlanIR changed after freezing")
        session.required_stage_passes = REQUIRED
        graph = dict(schema="twingraph.sliding_assembly_input.v23.r1",
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
                    result = PhysicalRunner(session, timeout=float(timeout_seconds)).run(plan, trial, keep_trace=True)
                except Exception as exc:
                    software_exception = dict(type=type(exc).__name__, message=str(exc),
                                              traceback=traceback.format_exc())
                    result = dict(valid=False, success=False, full_success=False,
                                  error=str(exc), timeout=False,
                                  stage_passes=deepcopy(getattr(session, "stage_passes", {})),
                                  wall_seconds=time.perf_counter()-started,
                                  executed_steps=len(session.results))
        stages = list(entry["proposal"]["order"])
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
        stage_audit["final_acceptance"] = dict(
            reached=bool(result["stage_passes"].get("assembly_pass")),
            completed=bool(result["success"]))
        first = next((row for row in session.results if not row["ok"]), None)
        failed_atom = (dict(index=first["index"], skill=first["skill"],
                            interface=first.get("interface"),
                            implementation=HANDLERS[first["skill"]].implementation,
                            reason=first.get("reason"), metrics=first.get("metrics")) if first else None)
        result.update(task_version=TASK_VERSION, evaluation_scope="complete_no_wipe_assembly_through_handle_installation",
                      full_task_label=True, seed=seed, level=level, layout_family=layout_family,
                      split_group=split_group, proposal=entry["proposal"],
                      runtime_sha256=runtime["sha256"], input_graph_sha256=graph_sha,
                      plan_sha256=entry["plan_sha256"], decision_observation_sha256=observation["sha256"],
                      observation_backend="mujoco_state_pose", observation_config=observation["config"],
                      observation_random_state=random_state, first_failed_atom=failed_atom,
                      failure_type=failure_category(failed_atom, result["error"],
                                                    result["timeout"], software_exception),
                      stage_audit=stage_audit,
                      final_functional_predicates=deepcopy(result["stage_passes"]),
                      base_hole_acceptance=deepcopy(session.artifacts.get("pin_joint_engagement")),
                      fixture_acceptance=deepcopy(session.artifacts.get("final_fixture_acceptance")),
                      final_seat_acceptance=deepcopy(session.artifacts.get("final_seat_acceptance")),
                      total_wall_seconds=time.perf_counter()-started,
                      resource_censored=bool(result["timeout"]))
        if result["timeout"]:
            result.update(valid=False, invalid_reason="resource-censored execution")
        if runtime_fingerprint()["sha256"] != runtime["sha256"]:
            result.update(valid=False, invalid_reason="runtime changed during execution")
        save(directory / "result.json", result)
        save(directory / "training_sample.json", dict(
            schema="twingraph.candidate_execution_sample.v23.r1",
            layout=dict(seed=seed, level=level, family=layout_family, split_group=split_group,
                        source=layout_provenance, task_version=TASK_VERSION,
                        runtime_sha256=runtime["sha256"]),
            decision_observation=observation,
            complete_candidate_plan_ir=entry["complete_candidate_plan_ir"],
            assembly_plan_ir=entry["assembly_plan_ir"],
            input_graph_path="input_graph.json", input_graph_sha256=graph_sha,
            actual_execution_label=result["success"] if result["valid"] else None,
            first_failed_atom=failed_atom, failure_type=result["failure_type"],
            stage_audit=stage_audit, final_functional_predicates=result["final_functional_predicates"],
            base_hole_acceptance=result["base_hole_acceptance"],
            fixture_acceptance=result["fixture_acceptance"],
            final_seat_acceptance=result["final_seat_acceptance"],
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
    parser.add_argument("--planner", choices=("grounded", "llm"), default="grounded")
    parser.add_argument("--frozen-request", type=Path)
    parser.add_argument("--candidate-n", type=int, default=8)
    parser.add_argument("--candidate-start", type=int, default=0)
    parser.add_argument("--execute-start", type=int, default=0)
    parser.add_argument("--execute-stop", type=int)
    parser.add_argument("--timeout-seconds", type=float, default=600.)
    parser.add_argument("--parent-request", type=Path)
    parser.add_argument("--level", choices=("L0", "L1", "L2"), default="L0")
    parser.add_argument("--layout-family", choices=("broad", "neighborhood"), default="broad")
    parser.add_argument("--neighborhood-parent-seed", type=int, default=2050)
    parser.add_argument("--pin-press-budget-policy", choices=tuple(PIN_PRESS_BUDGET_POLICIES),
                        default=PIN_PRESS_REMAINING_V28)
    parser.add_argument("--stroke-motion-policy", choices=tuple(STROKE_MOTION_POLICIES),
                        default=STROKE_PROGRESS_V29)
    parser.add_argument("--split-group")
    parser.add_argument("--domain", choices=("online", "development", "regression"), default="online")
    parser.add_argument("--position-noise-std-m", type=float, default=0.)
    parser.add_argument("--yaw-noise-std-rad", type=float, default=0.)
    args = parser.parse_args()
    if not (1 <= args.candidate_n <= args.pool_n <= 512
            and 0 <= args.candidate_start
            and args.candidate_start + args.candidate_n <= args.pool_n):
        parser.error("candidate window must fit within pool-n <= 512")
    if bool(args.candidate_start) != bool(args.parent_request):
        parser.error("a nonzero candidate-start requires parent-request, and vice versa")
    if min(args.position_noise_std_m, args.yaw_noise_std_rad) < 0:
        parser.error("noise standard deviations must be nonnegative")
    print(json.dumps(collect(args.seed, args.out, pool_n=args.pool_n,
        candidate_n=args.candidate_n, level=args.level, domain=args.domain,
        position_noise=args.position_noise_std_m, yaw_noise=args.yaw_noise_std_rad,
        layout_family=args.layout_family, candidate_start=args.candidate_start,
        parent_request=args.parent_request, neighborhood_parent_seed=args.neighborhood_parent_seed,
        split_group_override=args.split_group,
        pin_press_budget_policy=args.pin_press_budget_policy,
        stroke_motion_policy=args.stroke_motion_policy,
        execute_start=args.execute_start, execute_stop=args.execute_stop,
        timeout_seconds=args.timeout_seconds, planner=args.planner, frozen_request=args.frozen_request),
        ensure_ascii=False, default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value)),
        flush=True)


if __name__ == "__main__":
    main()
