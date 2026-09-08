"""Complete robot-only sliding-stage assembly; every action uses library atoms."""

import argparse
import json
import pickle
from pathlib import Path
import numpy as np
from PIL import Image
from .control import make_context, DEFAULT_POLICY
from .library import Session, SkillFailure, export_catalog
from .scene import CENTER
from .recording import Recorder

CAR_Z = 0.824
STOP = np.r_[CENTER + [-0.092, 0], 0.836]
PIN_L = np.r_[CENTER + [-0.092, -0.032], 0.855]
PIN_R = np.r_[CENTER + [-0.092, 0.032], 0.855]


def pick(s, part, lift=True, candidate_id=None, terminal_targets=None):
    from .candidates import (
        build_pick_candidates,
        choose_candidate,
        execute_pick_candidate,
        save_batch,
    )

    s.call("observe")
    s.call("estimate_pose", part=part)
    s.call("propose_grasps", part=part)
    # Open before construction: candidates all start at this actual shared state.
    s.call("gripper", mode="open")
    candidates = build_pick_candidates(s, part, lift, terminal_targets=terminal_targets)
    chosen = choose_candidate(candidates, candidate_id)
    save_batch(s, candidates, chosen)
    execute_pick_candidate(s, chosen)


def transfer_part(s, part, xyz):
    s.call(
        "plan_transfer",
        target=s.arm.part_target(part, xyz),
        yaw=s.artifacts["grasp"]["yaw"],
    )
    s.call("execute_joint_path")


def release(s):
    from .candidates import release_template

    steps = release_template()
    s.call(steps[0]["skill"], **steps[0]["params"])
    s.hold(0.3)
    s.call(steps[1]["skill"], **steps[1]["params"])


def assemble(s, policy=None):
    pick(
        s,
        "carriage",
        terminal_targets=[
            dict(id="seated", part="carriage", xyz=np.r_[CENTER + [0.030, 0], CAR_Z])
        ],
    )
    entry = np.r_[CENTER + [-0.155, 0], CAR_Z + 0.030]
    transfer_part(s, "carriage", entry)
    s.call("lower", part="carriage", height=0.030)
    s.call("align_axis", part="carriage", target=np.r_[entry[:2], CAR_Z + 0.0015])
    carriage_target = np.r_[CENTER + [0.030, 0], CAR_Z + 0.0015]
    s.call(
        "plan_insertion",
        part="carriage",
        target=carriage_target,
        axis=(1, 0, 0),
        speed=0.025,
        force_limit=18.0,
    )
    s.call("slide_insert", part="carriage")
    release(s)
    s.call("inspect_seat", part="carriage", target=np.r_[carriage_target[:2], CAR_Z])
    s.call("measure_clearance")

    pick(s, "end_stop", terminal_targets=[dict(id="seated", part="end_stop", xyz=STOP)])
    transfer_part(s, "end_stop", STOP + [0, 0, 0.045])
    s.call("plan_linear", target=s.arm.part_target("end_stop", STOP + [0, 0, 0.010]))
    s.call("execute_cartesian_path")
    s.call("align_axis", part="end_stop", target=STOP + [0, 0, 0.010])
    s.call("guarded_descent", part="end_stop", target_z=STOP[2])
    s.call("press_seat", part="end_stop", target_z=STOP[2])
    release(s)
    s.call("inspect_seat", part="end_stop", target=STOP)

    for part, target in [("pin_left", PIN_L), ("pin_right", PIN_R)]:
        pick(s, part, terminal_targets=[dict(id="seated", part=part, xyz=target)])
        transfer_part(s, part, target + [0, 0, 0.069])
        s.call("align_axis", part=part, target=target + [0, 0, 0.069])
        s.call("plan_insertion", part=part, target=target, speed=0.006)
        s.checkpoints[part] = s.snapshot()
        if s.out:
            with (s.out / f"{part}_checkpoint.pkl").open("wb") as f:
                pickle.dump(s.checkpoints[part], f)
        if policy and part == "pin_right":
            s.call("learned_insert", part=part, policy=str(policy))
        else:
            s.call("guarded_descent", part=part, target_z=target[2], force_stop=3.0)
        s.call("press_seat", part=part, target_z=target[2])
        release(s)
        s.call("inspect_seat", part=part, target=target)

    pick(
        s,
        "carriage",
        lift=False,
        terminal_targets=[
            dict(
                id="stroke_end", part="carriage", xyz=np.r_[CENTER + [0.060, 0], CAR_Z]
            )
        ],
    )
    s.call("move_constrained", part="carriage", target_x=float(CENTER[0] - 0.04))
    s.call("move_constrained", part="carriage", target_x=float(CENTER[0] + 0.060))
    s.call("verify_stroke", minimum=0.09)
    release(s)

    handle_target = s.ctx.obj_pos("carriage") + [0, 0, 0.048]
    pick(
        s,
        "handle",
        terminal_targets=[dict(id="seated", part="handle", xyz=handle_target)],
    )
    transfer_part(s, "handle", handle_target + [0, 0, 0.045])
    s.call("align_axis", part="handle", target=handle_target + [0, 0, 0.023])
    s.call(
        "guarded_descent",
        part="handle",
        target_z=float(handle_target[2]),
        force_stop=3.0,
    )
    s.call(
        "press_seat", part="handle", target_z=float(handle_target[2]), force_stop=2.0
    )
    release(s)
    s.call("inspect_seat", part="handle", target=handle_target)
    for part, target in [("end_stop", STOP), ("pin_left", PIN_L), ("pin_right", PIN_R)]:
        s.call("inspect_seat", part=part, target=target)
    s.call("home")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/tabletop/assembly")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--policy", default=str(DEFAULT_POLICY))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ctx = make_context(args.seed)
    rec = Recorder(ctx, out / "assembly.mp4") if args.record else None
    if rec:
        ctx.on_control_step = rec
    s = Session(ctx, rec, out, args.seed, args.noise)
    ok = False
    error = ""
    try:
        assemble(s, args.policy)
        ok = True
    except (SkillFailure, ValueError) as exc:
        error = str(exc)
        print("ASSEMBLY FAILED:", error, flush=True)
    finally:
        payload = dict(
            success=ok,
            error=error,
            steps=len(s.results),
            unique_skills=sorted({r["skill"] for r in s.results}),
            robot_actuators=ctx.model.nu,
            objects={
                p: ctx.obj_pos(p).tolist()
                for p in ("carriage", "end_stop", "pin_left", "pin_right", "handle")
            },
            sim_seconds=float(ctx.data.time),
            seed=args.seed,
            noise_std_m=args.noise,
            policy=args.policy,
        )
        (out / "verification.json").write_text(json.dumps(payload, indent=2))
        if ok:
            with (out / "final_checkpoint.pkl").open("wb") as f:
                pickle.dump(s.snapshot(), f)
        export_catalog(out / "component_catalog.json")
        if rec:
            rec.label = "装配与行程验收完成" if ok else "验收失败：" + error[:60]
            Image.fromarray(rec.frame()).save(out / "final.png")
            rec.pause(2.0)
            rec.close()
        print(json.dumps(payload), flush=True)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
