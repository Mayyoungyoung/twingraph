#!/usr/bin/env python3
"""Paired V28/V29 stroke checks from normally executed pre-stroke states."""
from __future__ import annotations

import contextlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from functools import partial
import json
from pathlib import Path
import time

from simbench.assembly.constrained_stroke_v29 import (
    STROKE_FIXED_V28,
    STROKE_MOTION_POLICIES,
    STROKE_PROGRESS_V29,
)
from simbench.assembly.library import Result, SkillFailure
from simbench.assembly.skills_v12 import evaluate_end_stop_stable_support, evaluate_functional_seat
from simbench.assembly.sensor_learning_v12 import PIN_PRESS_REMAINING_V28
from simbench.value import full_task_v7, system_v12
from simbench.value.physical import PhysicalRunner, perturbation
from simbench.value.plan import digest
from simbench.value.provenance_v12 import fingerprint as runtime_fingerprint
from simbench.value.system_v12 import make_scene, save
from scripts.collect_sliding_assembly_v23 import REQUIRED, controller_plan


# Frozen before V29 local execution.  The first six are the complete V28 cases
# that actually reached stroke; the last two add distinct V27 stroke/release
# failures without changing their candidate programs or layouts.
CASES = (
    dict(layout="neighborhood_14100", seed=14100, family="neighborhood", parent=4100,
         candidate="grounded_001", source="v28", category="original_complete_success"),
    dict(layout="neighborhood_14100", seed=14100, family="neighborhood", parent=4100,
         candidate="grounded_033", source="v28", category="original_complete_success"),
    dict(layout="neighborhood_14100", seed=14100, family="neighborhood", parent=4100,
         candidate="grounded_028", source="v28", category="first_leg_contact_overload"),
    dict(layout="neighborhood_14100", seed=14100, family="neighborhood", parent=4100,
         candidate="grounded_004", source="v28", category="first_leg_contact_overload"),
    dict(layout="broad_4102", seed=4102, family="broad", parent=4102,
         candidate="grounded_038", source="v28", category="first_leg_contact_overload"),
    dict(layout="broad_4102", seed=4102, family="broad", parent=4102,
         candidate="grounded_028", source="v28", category="second_leg_contact_overload"),
    dict(layout="broad_4107", seed=4107, family="broad", parent=4107,
         candidate="grounded_046", source="v27", category="stroke_verification_failure"),
    dict(layout="broad_4100", seed=4100, family="broad", parent=4100,
         candidate="grounded_025", source="v27", category="post_stroke_release_failure"),
)


def source_directory(case, root):
    if case["source"] == "v28":
        return root / "results/v28_pin_budget/full_pairs" / case["layout"] / PIN_PRESS_REMAINING_V28
    return root / "results/v27_bulk" / case["layout"]


def load_case(raw, root):
    case = deepcopy(raw); source = source_directory(case, root)
    request = json.loads((source / "request.json").read_text(encoding="utf-8"))
    entry = next(row for row in request["candidates"] if row["name"] == case["candidate"])
    old = json.loads((source / "candidates" / case["candidate"] / "result.json").read_text(
        encoding="utf-8"))
    case.update(proposal=entry["proposal"], proposal_sha256=digest(entry["proposal"]),
                source_result=str((source / "candidates" / case["candidate"] / "result.json").relative_to(root)),
                source_success=bool(old["success"]),
                source_failure=(old.get("first_failed_atom") or {}).get("reason"),
                decision_observation_sha256=request["decision_observation"]["sha256"])
    return case


def scene_kwargs(case):
    hook = None
    if case["family"] == "neighborhood":
        from simbench.value.supply_layout_v23 import apply
        hook = partial(apply, parent_seed=case["parent"])
    return dict(domain="online", level="L0", observation_backend="mujoco_state_pose",
                scene_layout_hook=hook, position_noise_std_m=0., yaw_noise_std_rad=0.)


def acceptance_after_stroke(session):
    pin_rows = {}; error = None
    try:
        for pin in ("pin_left", "pin_right"):
            session.call("inspect", what="pin_joint", part=pin, phase="retained_after_stroke")
            pin_rows[pin] = deepcopy(session.artifacts.get("pin_joint_engagement", {}).get(
                pin, {}).get("retained_after_stroke"))
    except SkillFailure as exc:
        error = str(exc)
    handle_ok, handle = evaluate_functional_seat(session, "handle", session.stage_targets["handle"])
    carriage_ok, carriage = evaluate_functional_seat(session, "carriage", session.stage_targets["carriage"])
    fixture_ok, fixture = evaluate_end_stop_stable_support(
        session, "end_stop", session.stage_targets["end_stop"], after_retreat=False)
    return dict(success=bool(error is None and handle_ok and carriage_ok and fixture_ok
                             and session.held is None),
                error=error, pins=pin_rows,
                handle=dict(success=bool(handle_ok), **handle),
                carriage=dict(success=bool(carriage_ok), **carriage),
                fixture=dict(fixture, success=bool(fixture_ok)),
                released=session.held is None)


def run_branch(session, checkpoint, baseline_rows, minimum, grasp_force, policy):
    session.restore(checkpoint); session.results = deepcopy(baseline_rows)
    session.stroke_motion_policy = policy
    started = time.perf_counter(); error = None; acceptance = None; software_exception = None
    try:
        full_task_v7._stroke(session, minimum, grasp_force=grasp_force)
        acceptance = acceptance_after_stroke(session)
    except SkillFailure as exc:
        error = str(exc)
    except Exception as exc:
        software_exception = dict(type=type(exc).__name__, message=str(exc))
        error = f"software exception: {type(exc).__name__}: {exc}"
    rows = deepcopy(session.results[len(baseline_rows):])
    return dict(
        policy=policy,
        controller_version=STROKE_MOTION_POLICIES[policy]["controller_version"],
        stroke_call_success=error is None,
        local_acceptance_success=bool(acceptance and acceptance["success"]),
        error=error,
        software_exception=software_exception,
        functional_stroke_plan=deepcopy(session.artifacts.get("functional_stroke_plan")),
        stroke_runs=deepcopy(session.stroke_runs),
        stroke_peak_forces=list(session.stroke_peak_forces),
        acceptance=acceptance,
        rows=rows,
        wall_seconds=time.perf_counter() - started,
    )


def run_case(case, out):
    out.mkdir(parents=True, exist_ok=False)
    _, session, _, _ = make_scene(case["seed"], out / "scene", **scene_kwargs(case))
    session.sliding_assembly_v23 = True
    session.end_stop_place_acceptance_v12 = "stable_supported"
    session.required_stage_passes = REQUIRED
    session.pin_press_budget_policy = PIN_PRESS_REMAINING_V28
    session.stroke_motion_policy = STROKE_FIXED_V28
    plan = controller_plan(session, case["proposal"],
        pin_press_budget_policy=PIN_PRESS_REMAINING_V28,
        stroke_motion_policy=STROKE_FIXED_V28)
    captured = {}; original = system_v12._stroke

    def intercept(current, minimum, *, grasp_force=3.0):
        checkpoint = current.snapshot(); baseline = deepcopy(current.results)
        branches = {
            policy: run_branch(current, checkpoint, baseline, minimum, grasp_force, policy)
            for policy in (STROKE_FIXED_V28, STROKE_PROGRESS_V29)
        }
        current.restore(checkpoint); current.results = baseline
        captured.update(checkpoint_created_from_normal_prefix=True,
            direct_object_state_edit=False,
            pre_stroke_positions={part: current.ctx.obj_pos(part).tolist()
                                  for part in ("carriage", "handle", "end_stop",
                                               "pin_left", "pin_right")},
            branches=branches)
        return Result(False, reason="paired local stroke checkpoint captured; not a full-task label")

    system_v12._stroke = intercept
    try:
        trial = perturbation(case["seed"], 0, "online")
        with (out / "prefix_console.log").open("w", encoding="utf-8") as log:
            with contextlib.redirect_stdout(log):
                prefix = PhysicalRunner(session, timeout=600.).run(plan, trial, keep_trace=True)
    finally:
        system_v12._stroke = original
    row = {k: v for k, v in case.items() if k != "proposal"}
    row.update(plan_sha256=digest(plan.to_dict()), plan_ir=plan.to_dict(), trial=trial,
               prefix_terminal_error=prefix.get("error"), captured=bool(captured), **captured)
    save(out / "result.json", row)
    return row


def run_selected(raw, root, out):
    case = load_case(raw, root)
    directory = out / f"{case['layout']}__{case['candidate']}"
    if (directory / "result.json").exists():
        return json.loads((directory / "result.json").read_text(encoding="utf-8"))
    return run_case(case, directory)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("results/v29_stroke/local_pairs"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(); root = args.root.resolve(); args.out.mkdir(parents=True, exist_ok=True)
    save(args.out / "manifest.json", dict(
        schema="twingraph.constrained_stroke_v29.local_pair.r1",
        runtime_sha256=runtime_fingerprint()["sha256"],
        selection_frozen_before_v29_execution=True,
        pin_press_budget_policy=PIN_PRESS_REMAINING_V28,
        policies=[STROKE_FIXED_V28, STROKE_PROGRESS_V29],
        same_physical_checkpoint_required=True,
        direct_object_state_edit=False,
        local_success_is_not_full_task_success=True,
        cases=list(CASES)))
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_selected, raw, root, args.out): raw for raw in CASES}
        for future in as_completed(futures):
            row = future.result(); rows.append(row); save(args.out / "progress.json", rows)
            print(json.dumps(dict(layout=row["layout"], candidate=row["candidate"],
                captured=row.get("captured"), branches={key: value.get("local_acceptance_success")
                    for key, value in row.get("branches", {}).items()})), flush=True)
    save(args.out / "summary.json", dict(cases=rows))


if __name__ == "__main__":
    main()
