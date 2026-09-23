"""V17-A end-stop stable-supported local placement runs.

One executable scope only: grasp the end_stop from supply, carry it to the
declared base mounting region, descend with contact protection, stop pushing
once force feedback shows support, release, retreat, then verify stability
over a declared observation window.  Pin insertion, the complete task and
value screening are explicitly NOT tested here.

Candidate selection rule (fixed before any outcome is read): generate the
normal V12 candidate pool with the existing planner, then take the first
pool entry whose end_stop necessary-geometry row passes ("necessary_pass").
Every layout uses the same rule.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


def _source_grasp_screen(session, parts, candidates, observation):
    """Declared pre-execution feasibility screen; reads no outcome label.

    For each candidate in the planner's construction order, recompute the
    source grasp EEF pose from the candidate's own catalogue row and CAD,
    then apply the same existing necessary checks the joint_checked_v10
    approach uses: exact IK with restarts, endpoint collision validity, and
    a 0.2 s position-servo settle whose residual must stay inside 1.5 mm
    (the catalogue itself never checks source IK or servo sag).
    """
    from simbench.assembly.control import down
    arm = session.arm
    ctx = session.ctx
    objects = observation["objects"]
    cad = session.planning_cad
    audit = []
    selected = None
    for proposal in candidates:
        row_check = dict(construction_key=None, name=proposal["name"], parts={})
        try:
            construction = int(str(proposal["name"]).rsplit("_", 1)[-1])
        except ValueError:
            construction = 10 ** 6
        row_check["construction_key"] = construction
        ok = True
        try:
            for part in parts:
                # Each part is screened from the same neutral initial arm
                # state; check_joint_path interpolates from the current
                # configuration and must not see the previous part's hold.
                saved_qpos = ctx.data.qpos.copy()
                saved_qvel = ctx.data.qvel.copy()
                saved_ctrl = ctx.data.ctrl.copy()
                try:
                    row = proposal.get("necessary_geometry", {}).get(part, {})
                    detail = dict(status=row.get("status"))
                    row_check["parts"][part] = detail
                    if row.get("status") not in ("necessary_pass", "unknown") or "yaw" not in row:
                        detail["verdict"] = "not_admitted"
                        ok = False
                        break
                    objects_local = observation["objects"]
                    source = objects_local.get(part, {})
                    position = np.asarray(source.get("position_m"), float)
                    quat = np.asarray(source.get("quat_wxyz"), float)
                    R = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
                    source_yaw = float(np.arctan2(R[1, 0], R[0, 0]))
                    spec = cad["parts"][part]
                    reference = spec["grasp_reference"]
                    center = np.asarray(row.get("grasp_center_offset_body_m") or [0., 0., 0.], float)
                    offset = center + np.array([0., 0., float(reference["height_offset_m"]) + float(row["height"])])
                    eef = position + R @ offset
                    rotation = down(source_yaw + float(row["source_axis_offset_rad"]))
                    usable_q = None
                    branch_failures = []
                    for seed in [None, *arm.restart_seeds(12)]:
                        try:
                            q = arm.ik(eef, rotation, seed=seed)
                        except ValueError:
                            branch_failures.append("ik")
                            continue
                        if arm.check_joint_path([q])["valid"]:
                            usable_q = q
                            break
                        branch_failures.append("collision")
                    if usable_q is None:
                        detail["verdict"] = f"no_clear_branch({len(branch_failures)})"
                        ok = False
                        break
                    # Static-hold sag check: place the arm kinematically at the
                    # collision-cleared endpoint and let the position servo hold
                    # it briefly; en-route motion belongs to the real executor.
                    import mujoco
                    ctx.data.qpos[ctx.arm_qadr] = usable_q
                    ctx.data.qvel[:] = 0
                    ctx.set_arm_ctrl(usable_q)
                    mujoco.mj_forward(ctx.model, ctx.data)
                    for _ in range(15):
                        ctx.step()
                    residual = float(np.linalg.norm(np.asarray(ctx.eef_pos()) - eef))
                    detail["static_hold_residual_m"] = round(residual, 6)
                    if residual > .0015:
                        detail["verdict"] = "servo_sag"
                        ok = False
                        break
                    detail["verdict"] = "usable"
                finally:
                    ctx.data.qpos[:] = saved_qpos
                    ctx.data.qvel[:] = saved_qvel
                    ctx.data.ctrl[:] = saved_ctrl
                    import mujoco
                    mujoco.mj_forward(ctx.model, ctx.data)
            if ok and selected is None:
                selected = proposal
        except Exception as exc:
            row_check["error"] = f"{type(exc).__name__}:{str(exc)[:120]}"
        audit.append(row_check)
        if selected is not None:
            break
    return selected, audit


def select_proposal(seed, out, n=48, level="L1", domain="online"):
    from simbench.value.system_v12 import make_scene, require_frozen_source
    from simbench.value.planner_v12 import propose
    from simbench.value.plan import digest
    provenance = require_frozen_source()
    scene_dir = out / "selection"
    _, session, _, _ = make_scene(seed, scene_dir, domain=domain, level=level)
    pool, source = propose(session.decision_observation, cad=session.planning_cad, n=n, seed=seed)
    rule = ("candidates in the planner's construction order (pre-outcome "
            "conservative reference family first); the screen covers only the "
            "end-stop actions that have NOT run yet at the legal carriage "
            "checkpoint (the carriage itself is supplied by that checkpoint, "
            "so carriage geometry is not used to reject an end-stop plan); a "
            "candidate is usable when its end-stop source grasp passes the "
            "existing necessary checks: catalogue admission, exact IK with "
            "restarts, endpoint collision validity and a static-hold settle "
            "inside 1.5 mm; the first usable candidate is selected, and if no "
            "candidate is usable the seed is reported as such instead of "
            "falling back to an end-stop-unverified candidate")
    ordered = sorted(pool, key=lambda p: int(str(p["name"]).rsplit("_", 1)[-1])
                     if str(p["name"]).rsplit("_", 1)[-1].isdigit() else 10 ** 6)
    selected, audit = _source_grasp_screen(session, ("end_stop",), ordered,
                                           session.decision_observation)
    selection_status = ("selected_first_usable_end_stop_candidate" if selected is not None
                        else "no_candidate_passed_end_stop_screen")
    record = dict(seed=seed, n=n, level=level, selection_rule=rule,
        screened_actions="end_stop_only, after the legal carriage checkpoint",
        selection_status=selection_status,
        runtime_sha256=provenance["sha256"], cad_sha256=digest(session.planning_cad),
        selected=selected["name"] if selected else None,
        selection_construction_index=(int(str(selected["name"]).rsplit("_", 1)[-1])
                                      if selected else None),
        screen_audit=audit)
    (out / "selection.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return selected, ordered, record


def summarize(result, seed, repeat, proposal, run_dir, recorded):
    steps = result.get("executed_parameters") or []
    boundary = None
    last_failed_index = None
    for i, step in enumerate(steps):
        if step["skill"] == "estimate_pose" and step["params"].get("part") == "end_stop":
            boundary = i
        if not step["ok"]:
            last_failed_index = i
    last_failed = steps[last_failed_index] if last_failed_index is not None else None
    stable = next((step for step in reversed(steps)
                   if step["skill"] == "inspect_stable_support"), None)
    stable_metrics = (stable or {}).get("metrics") or {}
    if not result.get("valid"):
        failure_phase = "invalid_rollout"
    elif last_failed is None and result.get("success"):
        failure_phase = None
    elif boundary is not None and last_failed_index is not None:
        failure_phase = "end_stop" if last_failed_index >= boundary else "setup_carriage"
    else:
        failure_phase = "setup_carriage"
    placed = bool(result.get("success") and stable_metrics.get("success"))
    return dict(
        scope="end_stop_stable_placement_only",
        end_stop_placed=placed,
        full_task_success=None,
        pin_insertion_tested=False,
        seed=seed, repeat=repeat, proposal=proposal["name"],
        acceptance="stable_supported",
        evaluation_scope=result.get("evaluation_scope"),
        setup_failure=bool(failure_phase == "setup_carriage" and not result.get("success")),
        failure_phase=failure_phase,
        failure_reason=result.get("error") or (last_failed or {}).get("reason") or None,
        failed_skill=(last_failed or {}).get("skill"),
        released=bool(stable_metrics.get("gripper_released")),
        gripper_still_supporting=bool(stable_metrics.get("gripper_still_supporting")),
        retained_entire_window=bool(stable_metrics.get("retained_entire_window")),
        support_force_n=stable_metrics.get("support_force_n"),
        supporting_bodies=stable_metrics.get("supporting_bodies"),
        stop_offset_m=stable_metrics.get("stop_offset_m") if isinstance(stable_metrics.get("stop_offset_m"), list)
            else (stable_metrics.get("window_rows") or [{}])[0].get("stop_offset_m"),
        deepest_penetration_m=stable_metrics.get("deepest_penetration_m"),
        wall_seconds=result.get("total_wall_seconds"),
        execution_wall_seconds=result.get("wall_seconds"),
        video=str((run_dir / "execution.mp4")) if recorded else None,
        rollout_valid=result.get("valid"), rollout_success=result.get("success"),
    )


def obtain_carriage_checkpoint(seed, pool, scan_dir, *, level, record):
    """Bounded legal-predecessor scan: normal physical execution only.

    Tries candidates in the planner's construction order with a
    stop_after='carriage' mechanism prefix until one installs the carriage.
    The captured exact twin checkpoint is a legal predecessor state; the
    end-stop itself is never placed or pre-arranged by this scan.
    """
    from simbench.value import system_v12
    from simbench.value.system_v11 import twin_checkpoint
    scan_dir = Path(scan_dir)
    scan_dir.mkdir(parents=True, exist_ok=True)
    captured = {}
    original = system_v12.run_mechanism_prefix

    def capturing(session, *, stop_after, **params):
        result = original(session, stop_after=stop_after, **params)
        if result.ok and stop_after == "carriage":
            captured["state"] = twin_checkpoint(session)
        return result

    attempts = []
    system_v12.run_mechanism_prefix = capturing
    try:
        for proposal in pool[:8]:
            run_dir = scan_dir / proposal["name"]
            result = system_v12.rollout(seed, proposal, run_dir, level=level,
                                        record=record, stop_after="carriage",
                                        end_stop_acceptance="stable_supported")
            attempts.append(dict(name=proposal["name"], success=result.get("success"),
                                 error=result.get("error")))
            if result.get("success") and "state" in captured:
                break
    finally:
        system_v12.run_mechanism_prefix = original
    (scan_dir / "attempts.json").write_text(json.dumps(attempts, indent=2), encoding="utf-8")
    return captured.get("state"), attempts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1950, 1951, 1952])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--n", type=int, default=48)
    parser.add_argument("--level", default="L1")
    parser.add_argument("--out", default=str(ROOT / "docs/evidence/value_v17a_end_stop_place/runs"))
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    from simbench.value.system_v12 import rollout
    runs = []
    for seed in args.seeds:
        seed_dir = out / f"seed_{seed}"
        selected, ordered, selection = select_proposal(seed, seed_dir, n=args.n, level=args.level)
        if selected is None:
            for repeat in range(1, args.repeats + 1):
                runs.append(dict(scope="end_stop_stable_placement_only", end_stop_placed=False,
                    full_task_success=None, pin_insertion_tested=False, seed=seed, repeat=repeat,
                    setup_failure=True, failure_phase="selection",
                    selection_status=selection.get("selection_status"),
                    failure_reason=("no candidate passed the end-stop source-grasp screen; "
                                    "reported instead of falling back to an end-stop-unverified candidate")))
            continue
        checkpoint_cache = {}
        for repeat in range(1, args.repeats + 1):
            run_dir = seed_dir / f"run_{repeat}"
            started = time.perf_counter()
            try:
                result = rollout(seed, selected, run_dir, level=args.level,
                                 record=args.record, stop_after="end_stop",
                                 end_stop_acceptance="stable_supported")
                row = summarize(result, seed, repeat, selected, run_dir, args.record)
            except Exception as exc:
                row = dict(scope="end_stop_stable_placement_only", end_stop_placed=False,
                    full_task_success=None, pin_insertion_tested=False, seed=seed, repeat=repeat,
                    proposal=selected["name"], acceptance="stable_supported",
                    setup_failure=False, failure_phase="runtime_exception",
                    failure_reason=f"{type(exc).__name__}: {str(exc)[:200]}",
                    wall_seconds=time.perf_counter() - started)
                runs.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
                continue
            row["started_from"] = "fresh_scene"
            row["selection_wall_seconds"] = time.perf_counter() - started
            if row.get("setup_failure") and row.get("failure_phase") == "setup_carriage":
                # Setup only: reuse a legal predecessor checkpoint obtained by
                # normal physical execution.  The end_stop still starts at its
                # supply pose and is grasped normally in the resumed run.
                if "state" not in checkpoint_cache:
                    checkpoint_cache["state"], checkpoint_cache["attempts"] = (
                        obtain_carriage_checkpoint(seed, ordered, seed_dir / "setup_scan",
                            level=args.level, record=False))
                state = checkpoint_cache.get("state")
                if state is not None:
                    resumed_dir = run_dir / "resumed_from_checkpoint"
                    try:
                        result2 = rollout(seed, selected, resumed_dir, level=args.level,
                                          record=args.record, stop_after="end_stop",
                                          end_stop_acceptance="stable_supported",
                                          checkpoint=state, completed=("carriage",))
                        row2 = summarize(result2, seed, repeat, selected, resumed_dir, args.record)
                    except Exception as exc:
                        row2 = dict(row, failure_phase="runtime_exception",
                            failure_reason=f"{type(exc).__name__}: {str(exc)[:200]}",
                            end_stop_placed=False)
                    row2.update(started_from="legal_predecessor_checkpoint_after_carriage",
                        fresh_scene_setup_attempt=dict(
                            failure_phase=row["failure_phase"], failed_skill=row["failed_skill"],
                            failure_reason=row["failure_reason"]),
                        setup_scan_attempts=checkpoint_cache["attempts"],
                        selection_wall_seconds=time.perf_counter() - started)
                    (resumed_dir / "v17a_summary.json").write_text(
                        json.dumps(row2, indent=2), encoding="utf-8")
                    row = row2
            (run_dir / "v17a_summary.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
            runs.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    summary = dict(scope="end_stop_stable_placement_only",
        placed=sum(1 for r in runs if r["end_stop_placed"]),
        effective_attempts=sum(1 for r in runs if not r.get("setup_failure")),
        setup_failures=sum(1 for r in runs if r.get("setup_failure")),
        runs=runs)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, ensure_ascii=False))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    main()
