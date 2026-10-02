#!/usr/bin/env python3
"""Paired V27/V28 pin-press suffix checks from normally reached checkpoints.

Each checkpoint is created by executing the unchanged full candidate prefix.
No object pose or simulator state is edited.  A local positive means only that
the selected pin press, release, retreat, and two-receiver retention check pass.
"""
from __future__ import annotations

import contextlib
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
import json
from pathlib import Path
import time

from simbench.assembly.library import Result, SkillFailure
from simbench.assembly import sensor_learning_v12 as pin_control
from simbench.value.physical import PhysicalRunner, perturbation
from simbench.value.plan import digest
from simbench.value.provenance_v12 import fingerprint as runtime_fingerprint
from simbench.value.system_v12 import make_scene, save
from scripts.collect_sliding_assembly_v23 import REQUIRED, controller_plan


CASES = (
    dict(layout="neighborhood_14100", seed=14100, family="neighborhood", parent=4100,
         candidate="grounded_001", category="original_full_success"),
    dict(layout="neighborhood_14100", seed=14100, family="neighborhood", parent=4100,
         candidate="grounded_033", category="original_full_success"),
    dict(layout="broad_4102", seed=4102, family="broad", parent=4102,
         candidate="grounded_006", category="v27_remaining_exceeded_capacity"),
    dict(layout="broad_4100", seed=4100, family="broad", parent=4100,
         candidate="grounded_013", category="v27_remaining_exceeded_capacity"),
    dict(layout="neighborhood_14103", seed=14103, family="neighborhood", parent=4103,
         candidate="grounded_013", category="v27_remaining_exceeded_capacity"),
    dict(layout="broad_4104", seed=4104, family="broad", parent=4104,
         candidate="grounded_020", category="v27_nominal_capacity_was_enough"),
    dict(layout="neighborhood_14103", seed=14103, family="neighborhood", parent=4103,
         candidate="grounded_020", category="v27_nominal_capacity_was_enough"),
    dict(layout="broad_4126", seed=4126, family="broad", parent=4126,
         candidate="grounded_020", category="v27_nominal_capacity_was_enough"),
)


def load_case(case, source_root):
    request=json.loads((source_root/case["layout"]/"request.json").read_text(encoding="utf-8"))
    entry=next(row for row in request["candidates"] if row["name"]==case["candidate"])
    old=json.loads((source_root/case["layout"]/"candidates"/case["candidate"]/
                    "result.json").read_text(encoding="utf-8"))
    item=deepcopy(case);item.update(proposal=entry["proposal"],old_full_success=bool(old["success"]),
        old_full_failure=(old.get("first_failed_atom") or {}).get("reason"),
        proposal_sha256=digest(entry["proposal"]),decision_observation_sha256=request[
            "decision_observation"]["sha256"])
    if item["category"]=="original_full_success":
        item["target_part"]=entry["proposal"]["order"][2]
    else:
        atom=old["first_failed_atom"]
        item["target_part"]=old["executed_parameters"][atom["index"]]["params"]["part"]
    return item


def scene_kwargs(case):
    hook=None
    if case["family"]=="neighborhood":
        from functools import partial
        from simbench.value.supply_layout_v23 import apply
        hook=partial(apply,parent_seed=case["parent"])
    return dict(domain="online",level="L0",observation_backend="mujoco_state_pose",
        scene_layout_hook=hook,position_noise_std_m=0.,yaw_noise_std_rad=0.)


def run_suffix(session, checkpoint, baseline_rows, part, target_z, force_stop, policy, original):
    session.restore(checkpoint);session.results=deepcopy(baseline_rows)
    session.pin_press_budget_policy=policy
    started=time.perf_counter();press=original(session,part,target_z,force_stop)
    rows=[];error=None;retention=None
    if press.ok:
        try:
            target=list(session.stage_targets[part])
            session.call("place",part=part,target=target,tol=.0025,settle=.35,
                         acceptance="pin_inserted")
            session.call("move",delta=[0.,0.,.10])
            session.call("inspect",what="pin_joint",part=part,phase="inserted_after_release")
            session.call("move",target="home")
            retention=deepcopy(session.artifacts.get("pin_joint_engagement",{}).get(part,{}).get(
                "inserted_after_release"))
        except SkillFailure as exc:
            error=str(exc)
    rows=deepcopy(session.results[len(baseline_rows):])
    return dict(policy=policy,controller_version=pin_control.PIN_PRESS_BUDGET_POLICIES[
        policy]["controller_version"],press_success=bool(press.ok),press_reason=press.reason,
        press_metrics=deepcopy(press.metrics),suffix_success=bool(press.ok and error is None),
        suffix_error=error,released=session.held is None,two_receiver_retention=retention,
        suffix_rows=rows,wall_seconds=time.perf_counter()-started)


def run_case(case, out):
    out.mkdir(parents=True,exist_ok=False)
    _,session,_,_=make_scene(case["seed"],out/"scene",**scene_kwargs(case))
    session.sliding_assembly_v23=True;session.end_stop_place_acceptance_v12="stable_supported"
    session.required_stage_passes=REQUIRED
    session.pin_press_budget_policy=pin_control.PIN_PRESS_FIXED_V27
    plan=controller_plan(session,case["proposal"],
        pin_press_budget_policy=pin_control.PIN_PRESS_FIXED_V27)
    captured={};original=pin_control.bounded_pin_press

    def intercept(current,part,target_z,force_stop):
        if part!=case["target_part"]:
            return original(current,part,target_z,force_stop)
        checkpoint=current.snapshot();baseline=deepcopy(current.results)
        branches={}
        for policy in (pin_control.PIN_PRESS_FIXED_V27,pin_control.PIN_PRESS_REMAINING_V28):
            branches[policy]=run_suffix(current,checkpoint,baseline,part,target_z,force_stop,
                                        policy,original)
        current.restore(checkpoint);current.results=baseline
        captured.update(part=part,checkpoint_created_from_normal_prefix=True,
            direct_object_state_edit=False,branches=branches)
        return Result(False,reason="paired local checkpoint captured; local result is not a full-task label")

    pin_control.bounded_pin_press=intercept
    try:
        trial=perturbation(case["seed"],0,"online")
        with (out/"prefix_console.log").open("w",encoding="utf-8") as log:
            with contextlib.redirect_stdout(log):
                prefix_result=PhysicalRunner(session,timeout=600.).run(plan,trial,keep_trace=True)
    finally:
        pin_control.bounded_pin_press=original
    row={k:v for k,v in case.items() if k!="proposal"}
    row.update(plan_sha256=digest(plan.to_dict()),plan_ir=plan.to_dict(),trial=trial,
        prefix_terminal_error=prefix_result.get("error"),captured=bool(captured),**captured)
    save(out/"result.json",row)
    return row


def run_selected(raw,source_root,out):
    case=load_case(raw,source_root)
    directory=out/f"{case['layout']}__{case['candidate']}"
    if (directory/"result.json").exists():
        return json.loads((directory/"result.json").read_text(encoding="utf-8"))
    return run_case(case,directory)


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root",type=Path,default=Path("results/v27_bulk"))
    parser.add_argument("--out",type=Path,default=Path("results/v28_pin_budget/local_pairs"))
    parser.add_argument("--workers",type=int,default=3)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    manifest=dict(schema="twingraph.pin_press_budget_v28.local_pair.r1",
        runtime_sha256=runtime_fingerprint()["sha256"],selection_declared_before_v28_execution=True,
        local_success_is_not_full_task_success=True,cases=list(CASES),
        policies=[pin_control.PIN_PRESS_FIXED_V27,pin_control.PIN_PRESS_REMAINING_V28])
    save(args.out/"manifest.json",manifest)
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run_selected,raw,args.source_root,args.out):raw for raw in CASES}
        for future in as_completed(futures):
            row=future.result();rows.append(row);save(args.out/"progress.json",rows)
            print(json.dumps(dict(layout=row["layout"],candidate=row["candidate"],
                captured=row.get("captured"),branches={k:v.get("suffix_success") for k,v in
                row.get("branches",{}).items()})),flush=True)
    save(args.out/"summary.json",dict(cases=rows))


if __name__=="__main__":main()
