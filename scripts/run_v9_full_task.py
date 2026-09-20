"""Independent-session V9 full task with the V7 reference controller."""
import argparse
import json
from pathlib import Path
import numpy as np

from simbench.value import stage_v5, stage_v7, stage_v9
from simbench.value.physical import PhysicalRunner


def run(seed, out, pose_mode="rgbd"):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    spec, session, scene, targets = stage_v9.make_scene(seed, out / "scene")
    session.fixture_pose_mode = "ideal_diagnostic" if pose_mode == "ideal" else "rgbd"
    order = stage_v5.legal_orders()[0]
    choices = {part: dict(yaw=0., height=.001, clearance=.98,
                          force=8. if part.startswith("pin_") else 3.,
                          **({} if part == "carriage" else {"speed": .006}))
               for part in stage_v5.PARTS}
    plan = stage_v7._full_plan(session, targets, order, choices, 0, 1.5, 14., .08)
    result = PhysicalRunner(session, timeout=600.).run(
        plan, dict(domain="regression", repeat=0, friction_scale=1., actuator_gain_scale=1.),
        keep_trace=True)
    row = dict(task_version=stage_v9.TASK_VERSION, seed=seed, pose_mode=pose_mode,
               spec=spec.__dict__, candidate_id=plan.id, scene=str(scene), result=result)
    (out / "result.json").write_text(json.dumps(row, indent=2, default=lambda x: np.asarray(x).tolist()))
    return row


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--seed",type=int,default=1200)
    p.add_argument("--pose-mode",choices=("rgbd","ideal"),default="rgbd")
    p.add_argument("--out",required=True)
    a=p.parse_args(); row=run(a.seed,a.out,a.pose_mode)
    print(json.dumps(dict(seed=a.seed,mode=a.pose_mode,result=row["result"]), default=lambda x: np.asarray(x).tolist()))
