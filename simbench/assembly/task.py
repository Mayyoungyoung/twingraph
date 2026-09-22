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


def pick(s, part, lift=True, candidate_id=None, terminal_targets=None,
         execution_feedback=False, grasp_force=3.0, required_parts=None, grasp_options=None):
    from .candidates import (
        build_pick_candidates,
        choose_candidate,
        execute_pick_candidate,
        save_batch,
    )

    previous_required = getattr(s, "_required_observation_parts", None)
    s._required_observation_parts = tuple(required_parts) if required_parts is not None else None
    try:
        s.call("detect")
    finally:
        s._required_observation_parts = previous_required
    if execution_feedback:
        s.call("observe_execution_pose", part=part)
    else:
        s.call("estimate_pose", part=part)
    s.call("estimate_grasp", part=part, **(grasp_options or {}))
    # Open before construction: candidates all start at this actual shared state.
    s.call("gripper", mode="open")
    controls = [dict(strategy="feedback", force=float(grasp_force))]
    candidates = build_pick_candidates(s, part, lift, terminal_targets=terminal_targets,
                                       control_options=controls)
    chosen = choose_candidate(candidates, candidate_id)
    save_batch(s, candidates, chosen)
    execute_pick_candidate(s, chosen)


def transfer_part(s, part, xyz):
    s.call(
        "plan_path",
        target=s.arm.part_target(part, xyz),
        yaw=s.artifacts["grasp"]["yaw"],
    )
    s.call("move", path="transfer")


def release(s):
    part = s.held
    if part is None:
        raise SkillFailure("release requires a held object")
    s.call("place", part=part, target=s.ctx.obj_pos(part).copy(), tol=0.003)
    s.call("move", delta=[0, 0, 0.10])


def assemble(s, policy=None, pin_offset=0.0):
    pick(
        s,
        "carriage",
        terminal_targets=[
            dict(id="seated", part="carriage", xyz=np.r_[CENTER + [0.030, 0], CAR_Z])
        ],
    )
    entry = np.r_[CENTER + [-0.155, 0], CAR_Z + 0.030]
    transfer_part(s, "carriage", entry)
    s.call("move", part="carriage", delta=[0, 0, -0.030])
    s.call(
        "move",
        reference="object",
        part="carriage",
        target=np.r_[entry[:2], CAR_Z + 0.0015],
    )
    carriage_target = np.r_[CENTER + [0.030, 0], CAR_Z + 0.0015]
    s.call(
        "plan_path",
        method="contact",
        part="carriage",
        target=carriage_target,
        axis=(1, 0, 0),
        speed=0.025,
        force_limit=18.0,
    )
    s.call("insert", part="carriage")
    s.call("press", part="carriage", target_z=CAR_Z)
    release(s)
    s.call("inspect", part="carriage", target=np.r_[carriage_target[:2], CAR_Z])
    s.call("measure", quantity="clearance", part="carriage")
    s.call("inspect", what="measurement", minimum=1.0e-12)

    pick(s, "end_stop", terminal_targets=[dict(id="seated", part="end_stop", xyz=STOP)])
    transfer_part(s, "end_stop", STOP + [0, 0, 0.045])
    s.call(
        "plan_path",
        method="cartesian",
        target=s.arm.part_target("end_stop", STOP + [0, 0, 0.010]),
    )
    s.call("move", path="linear", space="cartesian")
    s.call("move", reference="object", part="end_stop", target=STOP + [0, 0, 0.010])
    s.call("move", mode="guarded", part="end_stop", target_z=STOP[2])
    s.call("press", part="end_stop", target_z=STOP[2])
    release(s)
    s.call("inspect", part="end_stop", target=STOP)

    for part, intended in [("pin_left", PIN_L), ("pin_right", PIN_R)]:
        target = intended + ([pin_offset, 0, 0] if part == "pin_right" else [0, 0, 0])
        pick(s, part, terminal_targets=[dict(id="seated", part=part, xyz=target)])
        transfer_part(s, part, target + [0, 0, 0.069])
        s.call("move", reference="object", part=part, target=target + [0, 0, 0.069])
        s.call("plan_path", method="contact", part=part, target=target, speed=0.006)
        s.checkpoints[part] = s.snapshot()
        if s.out:
            with (s.out / f"{part}_checkpoint.pkl").open("wb") as f:
                pickle.dump(s.checkpoints[part], f)
        if policy and part == "pin_right":
            s.call("insert", strategy="learned", part=part, policy=str(policy))
        else:
            s.call(
                "move", mode="guarded", part=part, target_z=target[2], force_stop=3.0
            )
        s.call("press", part=part, target_z=target[2])
        release(s)
        s.call("inspect", part=part, target=intended)

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
    s.call(
        "move", mode="constrained", part="carriage", target_x=float(CENTER[0] - 0.04)
    )
    s.call(
        "move", mode="constrained", part="carriage", target_x=float(CENTER[0] + 0.060)
    )
    s.call("inspect", what="stroke", minimum=0.09)
    release(s)

    handle_target = s.ctx.obj_pos("carriage") + [0, 0, 0.048]
    pick(
        s,
        "handle",
        terminal_targets=[dict(id="seated", part="handle", xyz=handle_target)],
    )
    transfer_part(s, "handle", handle_target + [0, 0, 0.045])
    s.call(
        "move", reference="object", part="handle", target=handle_target + [0, 0, 0.023]
    )
    s.call(
        "move",
        mode="guarded",
        part="handle",
        target_z=float(handle_target[2]),
        force_stop=3.0,
    )
    s.call("press", part="handle", target_z=float(handle_target[2]), force_stop=2.0)
    release(s)
    s.call("inspect", part="handle", target=handle_target)
    for part, target in [("end_stop", STOP), ("pin_left", PIN_L), ("pin_right", PIN_R)]:
        s.call("inspect", part=part, target=target)
    s.call("move", target="home")


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
            unique_atoms=sorted({r["atom"] for r in s.results if r.get("atom")}),
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
        export_catalog(out / "atomic_skills.json")
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
