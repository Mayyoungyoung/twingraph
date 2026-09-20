"""Isolated physical pin insertion in the V9 fixture, with release check."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from simbench.value import stage_v5, stage_v9
from simbench.value.plan import execute_calls


def run(seed, out, part="pin_left", pose_mode="rgbd"):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    spec, session, scene, targets = stage_v9.make_scene(seed, out / "scene")
    session.ctx.set_obj_pose("end_stop", targets["end_stop"][:2], targets["end_stop"][2])
    for _ in range(80): session.ctx.step()
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
    choice = dict(yaw=0., height=.001, clearance=.98, force=8., speed=.006)
    calls = stage_v5.stage_calls(part, target, choice, 0, v7=True, functional_clearance=True)
    plan = SimpleNamespace(prefix={"part": part})
    error = None
    for index, call in enumerate(calls):
        try:
            execute_calls(session, plan, [call])
        except Exception as exc:
            error = dict(step=index, skill=call.skill, reason=str(exc))
            break
    from simbench.value.pin_geometry import evaluate_pin_context
    metrics = evaluate_pin_context(session.ctx, part, hole_offset_m=(0., -.032 if part == "pin_left" else .032, 0.),
                                   released=session.held != part, touching_finger=False,
                                   config=session.pin_insertion_config)
    row = dict(task_version=stage_v9.TASK_VERSION, seed=seed, part=part,
               pose_mode=pose_mode, spec=spec.__dict__, scene=str(scene),
               passed=error is None and metrics["success"], error=error,
               executed_steps=len(calls) if error is None else error["step"],
               insertion=metrics, results=session.results)
    (out / "result.json").write_text(json.dumps(row, indent=2, default=lambda x: np.asarray(x).tolist()))
    return row


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--seed", type=int, default=1200)
    p.add_argument("--part", choices=("pin_left", "pin_right"), default="pin_left")
    p.add_argument("--pose-mode", choices=("rgbd", "ideal"), default="rgbd")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    print(json.dumps(run(a.seed, a.out, a.part, a.pose_mode), default=lambda x: np.asarray(x).tolist()))
