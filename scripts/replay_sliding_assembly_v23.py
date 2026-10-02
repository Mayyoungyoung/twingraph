#!/usr/bin/env python3
"""Paired physical replays of predeclared V23 mixed-pool examples."""
from __future__ import annotations

import argparse
import contextlib
from copy import deepcopy
import json
from pathlib import Path
import time
import traceback

from simbench.assembly.library import HANDLERS
from simbench.value.physical import PhysicalRunner, perturbation
from simbench.value.plan import PlanIR, digest
from simbench.value.provenance_v12 import fingerprint as runtime_fingerprint
from simbench.value.system_v12 import make_scene, save
from scripts.collect_sliding_assembly_v23 import REQUIRED, TASK_VERSION, failure_category


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def select(request, root, *, positive_only=False):
    """Select the first positive, optionally with two distinct failures."""
    successes, failures = [], []
    seen = set()
    for entry in request["candidates"]:
        result = read(Path(root) / "candidates" / entry["name"] / "result.json")
        if not result["valid"]:
            continue
        if result["success"]:
            successes.append(entry)
        elif result["failure_type"] not in seen:
            failures.append(entry)
            seen.add(result["failure_type"])
    if not successes or (not positive_only and not failures):
        raise ValueError("paired replay requires a fully observed mixed pool")
    return [successes[0]] if positive_only else [successes[0], *failures[:2]]


def replay(root, out, *, positive_only=False):
    root, out = Path(root), Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("replay output must be empty")
    request = read(root / "request.json")
    if len(request["candidates"]) != request["candidate_n"]:
        raise ValueError("incomplete frozen source pool")
    if request["task_version"] != TASK_VERSION:
        raise ValueError("different task version")
    if runtime_fingerprint()["sha256"] != request["runtime_sha256"]:
        raise ValueError("runtime differs from frozen source")
    selected = select(request, root, positive_only=positive_only)
    selection = dict(schema="twingraph.v23.paired_replay_selection.r1",
        source_pool=str(root), source_request_sha256=digest(request),
        rule=("first valid success in frozen order"
              if positive_only else
              "first valid success and first two valid failures with distinct failure categories in frozen order"),
        selected=[entry["name"] for entry in selected],
        trials=[dict(kind="same_initial_same_trial", repeat=0),
                dict(kind="same_initial_changed_physics", repeat=1)],
        selection_uses_failure_labels=not positive_only,
        selection_uses_outcome_labels=True, first_pass_statistic=False)
    save(out / "selection.json", selection)
    hook = None
    if request["layout_family"] == "neighborhood":
        from simbench.value.supply_layout_v23 import apply as hook
    kwargs = dict(domain=request["domain"], level=request["level"],
        observation_backend="mujoco_state_pose",
        position_noise_std_m=request["position_noise_std_m"],
        yaw_noise_std_rad=request["yaw_noise_std_rad"], scene_layout_hook=hook)
    rows = []
    for entry in selected:
        for trial_spec in selection["trials"]:
            directory = out / entry["name"] / trial_spec["kind"]
            began = time.perf_counter()
            _, session, _, _ = make_scene(request["seed"], directory / "scene", **kwargs)
            session.sliding_assembly_v23 = True
            session.end_stop_place_acceptance_v12 = "stable_supported"
            session.required_stage_passes = REQUIRED
            if session.decision_observation != request["decision_observation"]:
                raise ValueError("replay initial observation changed")
            if session.state_observation_random_state != request["observation_random_state"]:
                raise ValueError("replay noise state changed")
            plan = PlanIR.from_dict(entry["complete_candidate_plan_ir"]).validate(session.parts)
            if digest(plan.to_dict()) != entry["plan_sha256"]:
                raise ValueError("replay PlanIR changed")
            trial = perturbation(request["seed"], trial_spec["repeat"], request["domain"])
            software_exception = None
            with (directory / "console.log").open("w", encoding="utf-8") as stream:
                with contextlib.redirect_stdout(stream):
                    try:
                        result = PhysicalRunner(session, timeout=600.).run(plan, trial, keep_trace=True)
                    except Exception as exc:
                        software_exception = dict(type=type(exc).__name__, message=str(exc),
                                                  traceback=traceback.format_exc())
                        result = dict(valid=False, success=False, full_success=False,
                                      error=str(exc), timeout=False,
                                      stage_passes=deepcopy(getattr(session, "stage_passes", {})),
                                      wall_seconds=time.perf_counter()-began)
            first = next((row for row in session.results if not row["ok"]), None)
            failed = (dict(index=first["index"], skill=first["skill"],
                           interface=first.get("interface"),
                           implementation=HANDLERS[first["skill"]].implementation,
                           reason=first.get("reason"), metrics=first.get("metrics")) if first else None)
            bindings = [row["stage"] for row in getattr(session, "receiver_binding_history", [])]
            order = entry["proposal"]["order"]
            stage_audit = {}
            for index, part in enumerate(order):
                completed = (order[index + 1] in bindings if index + 1 < len(order)
                             else bool(result["stage_passes"].get("assembly_pass")))
                stage_audit[part] = dict(reached=part in bindings,
                                         completed=bool(completed or result["stage_passes"].get("assembly_pass")))
            stage_audit["stroke"] = dict(reached=bool(result["stage_passes"].get("assembly_pass")),
                                          completed=bool(result["stage_passes"].get("functional_test_pass")))
            stage_audit["retention"] = dict(reached=bool(result["stage_passes"].get("functional_test_pass")),
                                             completed=bool(result["success"]))
            result.update(task_version=TASK_VERSION, candidate_name=entry["name"],
                source_plan_sha256=entry["plan_sha256"], source_request_sha256=digest(request),
                replay_kind=trial_spec["kind"], first_failed_atom=failed,
                failure_type=failure_category(failed, result["error"], result["timeout"], software_exception),
                stage_audit=stage_audit,
                base_hole_acceptance=deepcopy(session.artifacts.get("pin_joint_engagement")),
                observation_random_state=deepcopy(session.state_observation_random_state),
                total_wall_seconds=time.perf_counter()-began,
                software_exception=software_exception)
            if result["timeout"]:
                result.update(valid=False, invalid_reason="resource-censored execution")
            if runtime_fingerprint()["sha256"] != request["runtime_sha256"]:
                result.update(valid=False, invalid_reason="runtime changed during replay")
            save(directory / "result.json", result)
            save(directory / "training_sample.json", dict(
                schema="twingraph.candidate_replay_sample.v23.r1",
                layout=dict(seed=request["seed"], level=request["level"],
                            family=request["layout_family"], split_group=request["split_group"],
                            task_version=TASK_VERSION, runtime_sha256=request["runtime_sha256"]),
                decision_observation=request["decision_observation"],
                complete_candidate_plan_ir=entry["complete_candidate_plan_ir"],
                assembly_plan_ir=entry["assembly_plan_ir"],
                actual_execution_label=result["success"] if result["valid"] else None,
                replay_kind=trial_spec["kind"], first_failed_atom=failed,
                failure_type=result["failure_type"],
                stage_audit=stage_audit,
                final_functional_predicates=deepcopy(result["stage_passes"]),
                base_hole_acceptance=result["base_hole_acceptance"], trial=trial,
                observation_backend="mujoco_state_pose",
                observation_config=request["decision_observation"]["config"],
                observation_random_state=result["observation_random_state"],
                total_wall_seconds=result["total_wall_seconds"], timeout=result["timeout"],
                software_exception=software_exception))
            rows.append(dict(name=entry["name"], kind=trial_spec["kind"],
                             valid=result["valid"], success=result["success"],
                             failure_type=result["failure_type"]))
            save(out / "progress.json", rows)
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    save(out / "summary.json", dict(selection=selection, rows=rows))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--positive-only", action="store_true")
    args = parser.parse_args()
    replay(args.source, args.out, positive_only=args.positive_only)


if __name__ == "__main__":
    main()
