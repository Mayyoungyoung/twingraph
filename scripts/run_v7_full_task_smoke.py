"""Run one recorded RGB-D full-task smoke trial.

This is intentionally a small development experiment, not the frozen formal
dataset runner.  The scene, detector, candidate and trajectory are all the
same code paths used by the v7 full-task interface; the output is kept as an
auditable real MuJoCo trajectory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from simbench.assembly import task as assembly
from simbench.value import stage_v7
from simbench.value import stage_v5
from simbench.value.physical import PhysicalRunner, perturbation
from simbench.value.video_v7 import TwinRecorder


def run(seed: int, out: Path, level: str = "L1", pin_force: float = 8.0,
        pin_speed: float = 0.0045, timeout: float = 1800.0, record: bool = True,
        video_width: int = 480, video_height: int = 360,
        video_stride: int = 3, nominal_trial: bool = False,
        pin_grasp_height: float = .001):
    out.mkdir(parents=True, exist_ok=True)
    spec, session, _xml, targets = stage_v7.make_scene(seed, out / "scene", level=level)
    order = stage_v5.legal_orders()[0]
    choices = {
        part: dict(
            yaw=0.0,
            height=float(pin_grasp_height if part.startswith("pin_") else .001),
            clearance=0.98,
            force=float(pin_force if part.startswith("pin_") else 3.0),
            **({} if part == "carriage" else {"speed": float(pin_speed)}),
        )
        for part in stage_v5.PARTS
    }
    plan = stage_v7._full_plan(
        session, targets, order, choices, wipe_variant=0,
        wipe_force=1.5, wipe_duration=14.0, stroke_minimum=0.08,
    )
    recorder = (TwinRecorder(session, out / "full_task.mp4", score=None,
                             width=video_width, height=video_height,
                             frame_stride=video_stride)
                if record else None)
    if recorder is not None:
        session.rec = recorder
        session.ctx.on_control_step = recorder
    try:
        result = PhysicalRunner(session, timeout=float(timeout)).run(
            plan, (dict(domain="regression", repeat=0, friction_scale=1., actuator_gain_scale=1.)
                   if nominal_trial else perturbation(seed, 0, "development")), keep_trace=True
        )
    finally:
        if recorder is not None:
            recorder.pause(1.5)
            recorder.close()
    row = {
        "seed": seed,
        "level": level,
        "pin_force_N": pin_force,
        "pin_speed_m_s": pin_speed,
        "pin_grasp_height_m": pin_grasp_height,
        "candidate_id": plan.id,
        "video": str((out / "full_task.mp4").resolve()) if recorder is not None else None,
        "video_config": (dict(width=video_width, height=video_height,
                               frame_stride=video_stride,
                               fps=20.0 / max(1, int(video_stride)))
                         if recorder is not None else None),
        "result": result,
        "source": "real MuJoCo state trajectory with rgbd_geometry execution observations",
        "spec": spec.__dict__,
    }
    (out / "result.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=2, default=lambda x: np.asarray(x).tolist()),
        encoding="utf-8",
    )
    print(json.dumps(row, ensure_ascii=False, default=lambda x: np.asarray(x).tolist()))
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1200)
    parser.add_argument("--out", required=True)
    parser.add_argument("--level", choices=("L0", "L1", "L2"), default="L1")
    parser.add_argument("--pin-force", type=float, default=8.0)
    parser.add_argument("--pin-speed", type=float, default=0.0045)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--video-width", type=int, default=480)
    parser.add_argument("--video-height", type=int, default=360)
    parser.add_argument("--video-stride", type=int, default=3,
                        help="record every Nth control step; playback fps is divided by N")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--nominal-trial", action="store_true")
    parser.add_argument("--pin-grasp-height", type=float, default=.001)
    args = parser.parse_args()
    run(args.seed, Path(args.out), args.level, args.pin_force, args.pin_speed,
        args.timeout, not args.no_video, args.video_width, args.video_height,
        args.video_stride, args.nominal_trial, args.pin_grasp_height)
