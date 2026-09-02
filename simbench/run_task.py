#!/usr/bin/env python3
"""simbench task runner.

    python simbench/run_task.py --task A [--scene 1]
        [--viewer] [--realtime] [--record out.mp4] [--seed N]
        [--bringup]

--task A|B|C     which task's chain to run (scene 1 by default)
--plan           use the new planner+executor architecture (default)
--legacy         use the old task-module run_chain instead
--bringup        M1 robot bring-up check instead of the task chain:
                 home -> square trajectory -> gripper open/close
--viewer         live passive viewer window (needs DISPLAY)
--realtime       throttle the sim to wall-clock speed (with --viewer)
--record PATH    record an annotated upright mp4 while running
"""
import argparse
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402

from simbench.core.camera import LiveViewer, VideoRecorder  # noqa: E402
from simbench.core.controller import CartesianController, Gripper  # noqa: E402
from simbench.core.sim_context import MjContext  # noqa: E402
from simbench.collector import DataCollector  # noqa: E402
from simbench.executor import Executor  # noqa: E402
from simbench.faults import FailureModel  # noqa: E402
from simbench.planner import nominal_plan  # noqa: E402
from simbench.skills.settle import settle  # noqa: E402

SCENES = {
    ("A", 1): "taskA_gearbox.xml",
    ("B", 1): "taskB_rack.xml",
    ("C", 1): "taskC_fixture.xml",
}
SCENE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "scenes")


def build_context(task, scene, seed=0):
    fname = SCENES.get((task, scene))
    if fname is None:
        raise SystemExit(f"no scene registered for task {task} scene "
                         f"{scene}; known: {sorted(SCENES)}")
    path = os.path.join(SCENE_DIR, fname)
    if not os.path.exists(path):
        raise SystemExit(f"scene file missing: {path}")
    np.random.seed(seed)
    ctx = MjContext(path)
    ctx.reset()
    return ctx


# --------------------------------------------------------------------- M1
def bringup(ctx, verbose=True):
    """Robot bring-up: settle at home, trace a small square, cycle the
    gripper.  Returns True when every leg converges."""
    arm = CartesianController(ctx)
    gripper = Gripper(ctx)

    # let the home servo settle first
    for _ in range(50):
        ctx.step()

    ok = True
    p0 = ctx.eef_pos()
    square = [
        p0 + np.array([0.10, 0.00, 0.00]),
        p0 + np.array([0.10, 0.10, 0.00]),
        p0 + np.array([0.00, 0.10, 0.00]),
        p0 + np.array([0.00, 0.00, 0.00]),
    ]
    for i, tgt in enumerate(square):
        r = arm.move_eef(tgt, tol=0.004)
        if verbose:
            print(f"  leg {i}: {'ok' if r else 'FAIL'} "
                  f"-> {np.round(ctx.eef_pos(), 4)}")
        ok &= r

    gripper.open()
    span_open = gripper.span()
    gripper.close_to_span(0.030)
    span_closed = gripper.span()
    gripper.open()
    if verbose:
        print(f"  gripper: open {span_open*1000:.1f} mm -> "
              f"closed {span_closed*1000:.1f} mm")
    ok &= span_open > 0.07 and abs(span_closed - 0.030) < 0.004
    return ok


# ----------------------------------------------------------- executor path
def run_task_plan(ctx, task, rec=None, verbose=True, noise_std=0.0,
                  data_col=None, faults=None):
    """Run the planner+executor architecture.

    Returns a dict {k: (ok, metrics)} compatible with the legacy
    run_task_chain output by evaluating the task module's stage
    predicates on the scene state after execution.

    noise_std (m): Gaussian perception noise on every grasp detection
    (position std; yaw std x10 -- see skills/perception.detect).  The
    noisy grasp pose propagates through the held-part offset into every
    subsequent place/insert, i.e. it is the manual execution-error knob:
    0.0 = nominal baseline, larger = deliberately degraded execution.

    faults (FailureModel): the seeded stochastic failure model of the
    two-class action runs (Task A): grasp-pose estimation error,
    detection miss/false, path-planning unreachable/collision,
    move-to-pose residual, grasp slip, incoming-part scatter.  Every
    sampled event is attributed in faults.log.

    data_col: optional DataCollector; the plan, per-step outcomes and
    stage verdicts are forwarded to it for the run record.

    rec: optional VideoRecorder; the current skill is written to the
    overlay banner (on_stage) and each finished skill appends a
    pass/fail ticker entry (add_result), so the mp4 shows the
    sub-task progress in real time.
    """
    plan = nominal_plan(task)
    if data_col is not None:
        data_col.set_plan(plan)
        if faults is not None:
            data_col.set_faults(faults)
    # robot bring-up before the plan: let the home servo settle from
    # the initial pose BEFORE the Executor is built.  The controller
    # captures rot_target = eef_mat() at construction; 50 steps is not
    # enough for the home servo to quiet (measured qvel_max=0.11, eef
    # tilt drifts 7.29 -> 6.5 deg afterwards), and the orientation
    # servo then fights every skill toward the stale captured pose.
    # 600 steps (30s): the initial free-part transients (0.5mm seat
    # drops, the pin in its sleeve) damp within ~500 steps; the pin's
    # residual sleeve wobble (~0.3deg, bounded by the 0.25mm/edge
    # clearance) never fully quits and is below grasp tolerance.
    settle(ctx, max_steps=600, verbose=verbose)
    ex = Executor(ctx, noise_std=noise_std, faults=faults)
    # wire the per-step callbacks: the data collector records every
    # skill outcome; the video recorder annotates the current sub-task
    # banner and the running pass/fail ticker.
    def _step_done(i, skill, params, ok):
        if data_col is not None:
            data_col.note_step(i, skill, params, ok)
        if rec is not None and hasattr(rec, "add_result"):
            rec.add_result(skill, ok, "")
    ex.on_step = _step_done
    if rec is not None and hasattr(rec, "set_stage"):
        def _annotate(i, skill, total):
            rec.set_stage(f"{i+1}/{total} {skill}")
        ex.on_stage = _annotate
        rec.set_stage("bring-up")
    ex.run(plan, verbose=verbose)
    if rec is not None and hasattr(rec, "set_stage"):
        rec.set_stage("evaluating stage predicates")

    # evaluate stage predicates from the scene state
    if task == "A":
        from simbench.tasks import taskA_gearbox as mod
    elif task == "B":
        from simbench.tasks import taskB_rack as mod
    else:
        from simbench.tasks import taskC_fixture as mod

    results = {}
    state = ex.state
    for k in range(1, len(mod.STAGES) + 1):
        if task == "C":
            # Task C predicates need state dict entries
            st = {"grasp_off": state.get("workpiece_offset",
                                          np.zeros(3)),
                  "slip_xy": state.get("slip_xy", np.zeros(2)),
                  "side_force": state.get("side_act_force", 0.0),
                  "clamp_force": state.get("clamp_act_force", 0.0),
                  "datum_detected": state.get("datum_detected",
                                              False),
                  "measured_offset": state.get("measured_offset",
                                               np.zeros(2)),
                  "tool_resid": state.get("tool_resid", 9.9)}
            m = mod.stage_metrics(ctx, k, st)
            if k == 1:
                # S1 is a transient predicate (plate aloft right after
                # the grasp); evaluated on the terminal state it always
                # reads "still on the floor" -- use the grasp skill's
                # own result instead
                ok = bool(ex.results[0]["ok"])
            else:
                ok = mod.stage_success(ctx, k, st)
        elif task == "B":
            # Task B S8 reads the continuity-test state; S1-S7 are
            # terminal-state geometric predicates
            st = {"contact_quality": state.get("contact_quality", 0.0),
                  "test_force": state.get("test_force", 0.0),
                  "test_resid": state.get("test_resid", 9.9)}
            m = mod.stage_metrics(ctx, k, st)
            ok = mod.stage_success(ctx, k, st)
        else:
            # Task A: S1-S8 are terminal-state geometric predicates;
            # S9 reads the executor's inspect verdict from the state
            st = state
            m = mod.stage_metrics(ctx, k, st)
            ok = mod.stage_success(ctx, k, st)
        results[k] = (bool(ok), m)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="A", choices=list("ABC"))
    ap.add_argument("--scene", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.0,
                   help="perception noise std (m) on grasp detection -- "
                        "manual execution-error knob (0 = nominal)")
    ap.add_argument("--fault-profile", default="default",
                    choices=["none", "mild", "default", "strong"],
                    help="stochastic fault-injection profile for the "
                         "two-class Task A run (none = deterministic "
                         "baseline; see simbench/faults.py)")
    ap.add_argument("--plan", action="store_true", default=True,
                   help="use planner+executor (default)")
    ap.add_argument("--bringup", action="store_true")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--record", default=None)
    ap.add_argument("--every", type=int, default=0,
                   help="video frame sampling interval in control steps "
                        "(0 = auto ~1x sim speed at 30fps)")
    ap.add_argument("--repeat", type=int, default=1,
                   help="slow motion: write each sampled frame N times "
                        "(stretches the video Nx at the same fps; good "
                        "for fast tasks like C)")
    ap.add_argument("--data-dir", default="results/simbench/data",
                   help="run data output dir (plan/steps/stages/trace; "
                        "success and failure runs are both saved)")
    ap.add_argument("--no-data", action="store_true",
                   help="disable run data collection")
    args = ap.parse_args()

    ctx = build_context(args.task, args.scene, seed=args.seed)
    hooks = []
    recorder = None

    if args.viewer:
        hooks.append(LiveViewer(ctx, realtime=args.realtime))
    if args.record:
        # sampling interval: auto-pick so the 30fps mp4 plays back at
        # ~1x the (much faster than wall-clock) sim speed -- every
        # task runs at a different control-step rate (measured).
        SIM_RATE = {"A": 66, "B": 128, "C": 325}     # control steps/s
        rec_every = args.every if args.every > 0 else \
            max(1, round(SIM_RATE.get(args.task, 60) / 30))
        recorder = VideoRecorder(
            ctx, args.record, title=f"task{args.task} s{args.scene}",
            every=rec_every, repeat=args.repeat)
        hooks.append(recorder)

    # run data logger (plan + step outcomes + stage verdicts + sampled
    # qpos/eef/finger/contact trace); attached for both task paths
    data_col = None
    if not args.no_data:
        data_col = DataCollector(
            ctx, os.path.join(REPO_ROOT, args.data_dir), args.task,
            scene=args.scene, seed=args.seed, noise=args.noise,
            every=5)
        hooks.append(data_col)

    # fan-in: run all hooks after every control step
    def dispatch(c):
        for h in hooks:
            h(c)
    ctx.on_control_step = dispatch

    t0 = time.time()
    n0 = ctx.n_control_steps
    ret = 1
    exec_end = None
    faults = None
    try:
        if args.bringup:
            print(f"== bring-up check (task {args.task}, scene "
                  f"{args.scene}) ==")
            ok = bringup(ctx)
            print(f"BRING-UP: {'PASS' if ok else 'FAIL'}")
            ret = 0 if ok else 1
        else:
            # the seeded failure model: same (profile, seed) =>
            # bit-identical run; --noise overrides the estimation
            # noise source for backward compatibility
            faults = FailureModel(
                args.fault_profile, seed=args.seed,
                noise_override=(args.noise if args.noise > 0.0
                                else None))
            results = run_task_plan(ctx, args.task, rec=recorder,
                                  noise_std=args.noise,
                                  data_col=data_col,
                                  faults=faults)
            ok_all = all(ok for ok, _ in results.values())
            if faults is not None:
                print(f"\nfault profile: {args.fault_profile} "
                      f"(seed {args.seed}) -> "
                      f"{faults.summary()}")
            if data_col is not None:
                data_col.set_stages(results)
            if recorder is not None:
                for k in sorted(results):
                    ok, _ = results[k]
                    recorder.add_result(f"S{k}", ok, "")
                recorder.set_stage(
                    f"ALL STAGES: {'SUCCESS' if ok_all else 'FAILURE'}")
            print("\n== per-stage symbolic success ==")
            for k in sorted(results):
                ok, m = results[k]
                print(f"  S{k}: {'PASS' if ok else 'FAIL'}  "
                      f"{ {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in m.items()} }")
            print(f"\nALL STAGES: {'SUCCESS' if ok_all else 'FAILURE'}")
            ret = 0 if ok_all else 1
            exec_end = ctx.n_control_steps
            if recorder is not None:
                # outro: hold the terminal state on screen for ~3s so
                # the result ticker + ALL-STAGES banner are captured
                # (repeat stretches each frame, so divide it back out)
                outro = int(90 * recorder.every / recorder.repeat)
                for _ in range(outro):
                    ctx.step()
    finally:
        wall = time.time() - t0
        steps = (exec_end if exec_end is not None
                 else ctx.n_control_steps) - n0
        print(f"steps={steps} wall={wall:.1f}s "
              f"({steps / max(wall, 1e-9):.0f} control steps/s)")
        if data_col is not None:
            d = data_col.close(all_ok=all(ok for ok, _ in
                                          results.items())
                               if 'results' in locals() else None)
            print(f"run data saved: {d}")
        hard = False
        for h in hooks:
            if h is data_col:
                continue       # closed explicitly above (needs all_ok)
            h.close() if hasattr(h, "close") else None
            hard |= bool(getattr(h, "hard_exit", False))
        if hard:
            # a 2.3.2 daemon-thread viewer GUI is still open: normal
            # interpreter shutdown aborts (glfw atexit race), so exit hard
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(ret)
    sys.exit(ret)


if __name__ == "__main__":
    main()
