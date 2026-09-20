"""Isolated physical pin insertion in the V9 fixture, with release check."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from simbench.value import stage_v5, stage_v9
from simbench.value.plan import execute_calls


def run(seed, out, part="pin_left", pose_mode="rgbd", grasp_height=.001, grasp_yaw=0.):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    spec, session, scene, targets = stage_v9.make_scene(seed, out / "scene", preinstalled_end_stop=True)
    fixture_initial = session.ctx.obj_pos("end_stop").tolist()
    session.fixture_pose_mode = "ideal_diagnostic" if pose_mode == "ideal" else "rgbd"
    # The fixture is installed before the test starts; all pin motion is through
    # executable skills and live MuJoCo contacts.
    target = np.asarray(targets[part], float)
    if pose_mode == "rgbd":
        from simbench.value.stage_v7 import refresh_visual_observation
        obs = refresh_visual_observation(session)
        row = obs["objects"]["end_stop"]
        if not row["valid"]:
            raise RuntimeError("RGB-D did not detect installed end_stop")
        end = np.asarray(row["position_m"], float) + [0., -.0045, 0.]
    else:
        end = session.ctx.obj_pos("end_stop")
    target[:2] = end[:2] + [0., -.032 if part == "pin_left" else .032]
    choice = dict(yaw=float(grasp_yaw), height=float(grasp_height), clearance=.98, force=8., speed=.006)
    calls = stage_v5.stage_calls(part, target, choice, 0, v7=True, functional_clearance=True)
    plan = SimpleNamespace(prefix={"part": part})
    error = None
    states = []
    for index, call in enumerate(calls):
        try:
            execute_calls(session, plan, [call])
        except Exception as exc:
            error = dict(step=index, skill=call.skill, reason=str(exc))
            states.append(dict(step=index, skill=call.skill, failed=True,
                               pin_position_m=session.ctx.obj_pos(part).tolist(),
                               pin_axis=session.ctx.obj_axis(part).tolist(),
                               eef_position_m=session.ctx.eef_pos().tolist()))
            break
        states.append(dict(step=index, skill=call.skill,
                           pin_position_m=session.ctx.obj_pos(part).tolist(),
                           pin_axis=session.ctx.obj_axis(part).tolist(),
                           eef_position_m=session.ctx.eef_pos().tolist()))
    from simbench.value.pin_geometry import evaluate_pin_context
    bid = session.ctx.body_id(part)
    touching = False
    for contact in session.ctx.data.contact:
        b1, b2 = map(int, session.ctx.model.geom_bodyid[[contact.geom1, contact.geom2]])
        other = b2 if b1 == bid else b1 if b2 == bid else None
        if other is not None and "finger" in session.ctx.model.body(other).name and contact.dist <= 0:
            touching = True
    metrics = evaluate_pin_context(session.ctx, part, hole_offset_m=(0., -.032 if part == "pin_left" else .032, 0.),
                                   released=session.held != part, touching_finger=touching,
                                   config=session.pin_insertion_config)
    row = dict(task_version=stage_v9.TASK_VERSION, seed=seed, part=part,
               pose_mode=pose_mode, grasp_height_m=grasp_height, grasp_yaw_rad=grasp_yaw,
               spec=spec.__dict__, scene=str(scene),
               fixture_initial_position_m=fixture_initial,
               commanded_pin_target_m=target.tolist(),
               passed=error is None and metrics["success"], error=error,
               executed_steps=len(calls) if error is None else error["step"],
               insertion=metrics, states=states, results=session.results)
    (out / "result.json").write_text(json.dumps(row, indent=2, default=lambda x: np.asarray(x).tolist()))
    return row


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--seed", type=int, default=1200)
    p.add_argument("--part", choices=("pin_left", "pin_right"), default="pin_left")
    p.add_argument("--pose-mode", choices=("rgbd", "ideal"), default="rgbd")
    p.add_argument("--grasp-height", type=float, default=.001)
    p.add_argument("--grasp-yaw", type=float, default=0.)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    print(json.dumps(run(a.seed, a.out, a.part, a.pose_mode, a.grasp_height, a.grasp_yaw), default=lambda x: np.asarray(x).tolist()))
