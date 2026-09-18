#!/usr/bin/env python3
"""Re-execute one formal v6 candidate and render frames from live MuJoCo physics.

Unlike the shared-world state visualizer, this script never assigns a recorded
object pose.  The saved candidate program is executed by RobustPhysicalRunner;
all frames are captured after MjContext.step has advanced mujoco.mj_step.
Run one process per candidate, then compose the equal-length tiles with ffmpeg.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from simbench.assembly.candidates import fingerprint
from simbench.assembly.control import HOME
from simbench.assembly.library import Session
from simbench.core.sim_context import MjContext
from simbench.value import stage_v5, stage_v6
from simbench.value.physical_v6 import RobustPhysicalRunner
from simbench.value.plan import PlanIR, digest
from simbench.value.skill_graph import compile_graph


CANDIDATES = [
    "57ed526f426891e1124a",
    "4ea815b900509b86a58b",
    "ec4d26554fd5cb502a8e",
    "4be3f53ed43b56c51e63",
    "c875010444c67684f1b6",
    "e03e5bc8b264dc7897e8",
    "b2df063756dbea03a712",
    "2ef21e72b5df67cda08a",
    "3536869bd0cbcd74a6bc",
    "e83f0990783267c02a3b",
    "7cf5222311fde636efd8",
    "e29adb7ac28b5567e9ec",
]

SCORES = {
    "57ed526f426891e1124a": 0.30357704653003303,
    "4ea815b900509b86a58b": 0.09612468826423355,
    "ec4d26554fd5cb502a8e": 0.20475383864896562,
    "4be3f53ed43b56c51e63": 0.1464099272416701,
    "c875010444c67684f1b6": 0.014535456363900419,
    "e03e5bc8b264dc7897e8": 0.017085886344380755,
    "b2df063756dbea03a712": 0.57224029197987,
    "2ef21e72b5df67cda08a": 0.2152903190326282,
    "3536869bd0cbcd74a6bc": 0.1808761741668723,
    "e83f0990783267c02a3b": 0.8813199868491216,
    "7cf5222311fde636efd8": 0.12257401274254273,
    "e29adb7ac28b5567e9ec": 0.9756215828910014,
}


class FfmpegWriter:
    """Stream RGB frames to the system encoder without touching simulation state."""

    def __init__(self, path: Path, width: int, height: int, fps: int) -> None:
        self.path = path
        self.process = subprocess.Popen(
            [
                "ffmpeg", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pixel_format", "rgb24",
                "-video_size", f"{width}x{height}", "-framerate", str(fps),
                "-i", "-", "-an", "-c:v", "libx264", "-crf", "18",
                "-preset", "medium", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(path),
            ],
            stdin=subprocess.PIPE,
        )

    def append_data(self, frame: np.ndarray) -> None:
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg input pipe is unavailable")
        self.process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        return_code = self.process.wait()
        if return_code:
            raise RuntimeError(f"ffmpeg failed with exit status {return_code}")


def style_scene(path: Path) -> None:
    """Change appearance and cameras only; leave every physics field intact."""
    tree = ET.parse(path)
    root = tree.getroot()
    asset = root.find("asset")
    if asset is None:
        raise ValueError("scene has no asset section")
    ET.SubElement(
        asset,
        "texture",
        type="2d",
        name="showcase_gray_white_checker",
        builtin="checker",
        rgb1="0.43 0.45 0.48",
        rgb2="0.82 0.83 0.84",
        width="1024",
        height="1024",
    )
    ET.SubElement(
        asset,
        "material",
        name="showcase_gray_white_checker_mat",
        texture="showcase_gray_white_checker",
        texuniform="true",
        texrepeat="5 5",
        reflectance="0.025",
    )
    floor = root.find("./worldbody/geom[@name='floor']")
    if floor is None:
        raise ValueError("scene has no floor geom")
    floor.attrib.pop("rgba", None)
    floor.set("material", "showcase_gray_white_checker_mat")

    sky = root.find("./asset/texture[@type='skybox']")
    if sky is not None:
        sky.set("rgb1", "0.33 0.37 0.42")
        sky.set("rgb2", "0.62 0.66 0.70")
    headlight = root.find("./visual/headlight")
    if headlight is not None:
        headlight.set("ambient", "0.30 0.30 0.30")
        headlight.set("diffuse", "0.48 0.48 0.48")
        headlight.set("specular", "0.055 0.055 0.055")
    light = root.find("./worldbody/light")
    if light is not None:
        light.set("diffuse", "0.62 0.62 0.62")
    tree.write(path, encoding="unicode")


def make_session(seed: int, directory: Path, role: str = "twin") -> tuple[Session, Path]:
    spec = stage_v6.StageV6Spec.sample(seed)
    path = stage_v6.write_scene(spec, directory)
    style_scene(path)
    ctx = MjContext(path, control_freq=50)
    ctx.reset()
    ctx.data.qpos[ctx.arm_qadr] = HOME
    mujoco.mj_forward(ctx.model, ctx.data)
    ctx.hold_arm()
    ctx.set_finger_ctrl(0.04)
    for _ in range(80):
        ctx.step()
    session = Session(ctx, seed=seed, noise=0.0)
    targets = stage_v5.nominal_targets()
    session.stage_targets = targets
    session.scene_role = role
    session.stage_completed = ()
    session.value_checkpoint_trace = []
    session.value_checkpoint = {
        "kind": "initial_unassembled_scene",
        "completed_parts": [],
        "assembly_part_count": 0,
        "simulation_time_s": float(ctx.data.time),
        "source": "domain_randomized_original_supply_layout",
        "role": role,
    }
    session.supplier_snapshot = {
        part: {
            "position": ctx.obj_pos(part).tolist(),
            "quaternion": ctx.obj_pose(part)[1].tolist(),
        }
        for part in stage_v6.PARTS
    }
    return session, path


def load_font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size=size)


def render_tile(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    camera: mujoco.MjvCamera,
    candidate_index: int,
    elapsed_s: float,
    font_path: Path,
    status: str | None = None,
    scope_label: str = "",
) -> np.ndarray:
    renderer.update_scene(data, camera=camera)
    image = Image.fromarray(renderer.render())
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    cid = CANDIDATES[candidate_index]
    draw.rectangle((0, 0, image.width, 45), fill=(13, 17, 23, 220))
    draw.text(
        (10, 5),
        f"{scope_label}C{candidate_index + 1:02d}  value={SCORES[cid]:.4f}  LIVE mj_step",
        font=load_font(font_path, 19),
        fill=(244, 247, 250, 255),
    )
    draw.text(
        (image.width - 112, 26),
        f"t={elapsed_s:05.1f}s",
        font=load_font(font_path, 14),
        fill=(206, 216, 226, 255),
    )
    if status is not None:
        success = status == "SUCCESS"
        color = (28, 151, 76, 235) if success else (190, 55, 48, 235)
        draw.rounded_rectangle((image.width - 132, image.height - 43, image.width - 8, image.height - 8), 8, fill=color)
        draw.text((image.width - 120, image.height - 39), status, font=load_font(font_path, 18), fill=(255, 255, 255, 255))
    draw.rectangle((0, 0, image.width - 1, image.height - 1), outline=(55, 61, 69, 255), width=3)
    return np.asarray(Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB"))


def run_candidate(args: argparse.Namespace) -> dict:
    case_dir = args.case_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    execution_scope = "twin_candidate_validation"
    scene_role = "twin"
    scope_label = ""
    if args.target_repeat is None:
        candidate_index = args.candidate_index - 1
        if not 0 <= candidate_index < len(CANDIDATES):
            raise ValueError("--candidate-index must be from 1 to 12")
        candidate_id = CANDIDATES[candidate_index]
        inputs = json.loads((case_dir / "full" / "inputs.json").read_text(encoding="utf-8"))
        plan_rows = {row["id"]: row for row in inputs["candidates"]}
        recorded_plan = PlanIR.from_dict(plan_rows[candidate_id])
        formal_path = case_dir / "full" / "validation" / f"{candidate_id}_0.json"
        formal = json.loads(formal_path.read_text(encoding="utf-8"))
    else:
        execution_scope = "independent_target_execution"
        scene_role = "target"
        scope_label = "ACTUAL "
        target_dir = case_dir / args.target_method
        target_input_path = target_dir / f"target_input_{args.target_repeat}.json"
        target_execution_path = target_dir / f"target_execution_{args.target_repeat}.json"
        target_input = json.loads(target_input_path.read_text(encoding="utf-8"))
        recorded_plan = PlanIR.from_dict(target_input["plan"])
        candidate_id = recorded_plan.id
        candidate_index = CANDIDATES.index(candidate_id)
        formal_path = target_execution_path
        formal = json.loads(target_execution_path.read_text(encoding="utf-8"))["execution"]
    trial = copy.deepcopy(formal["trial"])

    scene_name = (f"candidate_{candidate_index + 1:02d}" if args.target_repeat is None
                  else f"actual_target_{args.target_repeat}")
    scene_dir = output_dir / "scenes" / scene_name
    session, scene_path = make_session(args.seed, scene_dir, role=scene_role)
    bound_fingerprint = fingerprint(session)
    if bound_fingerprint != recorded_plan.prefix["start_state"]:
        raise ValueError("styled scene changed the candidate's bound physics state")

    # Reproduce the formal pre-execution call chain.  Candidate construction
    # and target rebinding do not change the physical-state fingerprint, but
    # they intentionally initialize motion-planning state used by execution.
    targets = session.stage_targets
    if args.target_repeat is None:
        observation = stage_v6.observed(session, targets)
        stage_v6.capture_vision(session)
        planner = json.loads((case_dir / "full" / "planner.json").read_text(encoding="utf-8"))
        regenerated, _ = stage_v6.build_pool(
            session,
            targets,
            args.seed,
            n=len(CANDIDATES),
            orders=planner["proposed_orders"],
            precheck=True,
        )
        regenerated_by_id = {item.id: item for item in regenerated}
        if candidate_id not in regenerated_by_id:
            raise ValueError("formal candidate was not recovered by the frozen generator")
        plan = regenerated_by_id[candidate_id]
    else:
        full_inputs = json.loads((case_dir / "full" / "inputs.json").read_text(encoding="utf-8"))
        chosen_rows = {row["id"]: row for row in full_inputs["candidates"]}
        chosen = PlanIR.from_dict(chosen_rows[target_input["selected_candidate_id"]])
        plan, _ = stage_v6.rebind_plan(session, targets, chosen)
        observation = stage_v6.observed(session, targets)
    if digest(plan.to_dict()) != digest(recorded_plan.to_dict()):
        raise ValueError("regenerated executable plan disagrees with the formal record")
    executable = compile_graph(observation, plan)
    if digest(executable) != formal.get("input_graph_sha256"):
        raise ValueError("regenerated executable graph disagrees with the formal record")

    runner = RobustPhysicalRunner(session, timeout=args.timeout)
    renderer = mujoco.Renderer(session.ctx.model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.array([-0.20, 0.0, 0.84])
    camera.distance = args.camera_distance
    camera.azimuth = args.camera_azimuth
    camera.elevation = args.camera_elevation

    if args.preview:
        frame = render_tile(renderer, session.ctx.data, camera, candidate_index, 0.0, args.font,
                            scope_label=scope_label)
        preview = output_dir / f"{scene_name}_preview.png"
        imageio.imwrite(preview, frame)
        renderer.close()
        return {"preview": str(preview)}

    output = (output_dir / f"candidate_{candidate_index + 1:02d}_{candidate_id}_live.mp4"
              if args.target_repeat is None else
              output_dir / f"actual_target_{args.target_repeat}_{candidate_id}_live.mp4")
    partial = output.with_suffix(".partial.mp4")
    for path in (output, partial):
        if path.exists():
            path.unlink()
    writer = FfmpegWriter(partial, args.width, args.height, args.fps)

    start_sim = float(session.ctx.data.time)
    physical_period = args.speed / args.fps
    target_frames = math.ceil(args.display_physical_seconds / args.speed * args.fps)
    next_capture_s = 0.0
    frame_count = 0
    last_frame = None
    previous_callback = session.ctx.on_control_step

    def capture(current: MjContext) -> None:
        nonlocal next_capture_s, frame_count, last_frame
        if previous_callback is not None:
            previous_callback(current)
        elapsed_s = max(0.0, float(current.data.time) - start_sim)
        if elapsed_s + 1e-9 < next_capture_s or frame_count >= target_frames:
            return
        last_frame = render_tile(renderer, current.data, camera, candidate_index, elapsed_s, args.font,
                                 scope_label=scope_label)
        writer.append_data(last_frame)
        frame_count += 1
        next_capture_s += physical_period

    started = time.perf_counter()
    session.ctx.on_control_step = capture
    try:
        capture(session.ctx)
        result = runner.run(executable, trial, keep_trace=True)
    finally:
        session.ctx.on_control_step = previous_callback
    status = "SUCCESS" if result["success"] else "FAILURE"
    terminal_elapsed = max(0.0, float(session.ctx.data.time) - start_sim)
    terminal_frame = render_tile(
        renderer,
        session.ctx.data,
        camera,
        candidate_index,
        terminal_elapsed,
        args.font,
        status=status,
        scope_label=scope_label,
    )
    while frame_count < target_frames:
        writer.append_data(terminal_frame)
        frame_count += 1
    writer.close()
    renderer.close()
    partial.replace(output)

    position_errors = {
        name: float(np.linalg.norm(np.asarray(result["final_positions"][name]) - np.asarray(formal["final_positions"][name])))
        for name in stage_v6.PARTS
    }
    success_matches = bool(formal["success"]) == bool(result["success"])
    terminal_comparison_applicable = bool(formal["success"]) and bool(result["success"])
    maximum_terminal_error = max(position_errors.values())
    formal_reproduction_pass = success_matches and (
        not terminal_comparison_applicable or maximum_terminal_error <= args.terminal_tolerance
    )
    audit = {
        "schema": "twingraph.showcase.live_physics_candidate.v1",
        "candidate_index": candidate_index + 1,
        "candidate_id": candidate_id,
        "score": SCORES[candidate_id],
        "execution_scope": execution_scope,
        "formal_source": str(formal_path),
        "execution": "RobustPhysicalRunner -> MjContext.step -> mujoco.mj_step",
        "renderer_object_qpos_writes": 0,
        "mujoco_version": mujoco.__version__,
        "initial_state_fingerprint": bound_fingerprint,
        "trial": trial,
        "formal_success": bool(formal["success"]),
        "rerun_success": bool(result["success"]),
        "success_matches_formal": success_matches,
        "formal_sim_seconds": float(formal["sim_seconds"]),
        "rerun_sim_seconds": float(result["sim_seconds"]),
        "physics_steps": int(result["physics_steps"]),
        "executed_steps": int(result["executed_steps"]),
        "terminal_position_errors_m": position_errors,
        "maximum_terminal_position_error_m": maximum_terminal_error,
        "terminal_comparison_applicable": terminal_comparison_applicable,
        "formal_comparison_tolerance_m": args.terminal_tolerance,
        "formal_reproduction_pass": formal_reproduction_pass,
        "scene": str(scene_path),
        "video": output.name,
        "fps": args.fps,
        "speed_multiplier": args.speed,
        "frames": frame_count,
        "duration_seconds": frame_count / args.fps,
        "wall_seconds": time.perf_counter() - started,
        "error": result.get("error", ""),
    }
    audit_path = output.with_suffix(".json")
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not formal_reproduction_pass:
        raise RuntimeError(f"live rerun disagrees with formal execution: {json.dumps(audit, ensure_ascii=False)}")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-index", type=int, required=True)
    parser.add_argument("--target-repeat", type=int)
    parser.add_argument("--target-method", default="top_k")
    parser.add_argument("--seed", type=int, default=71400)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--speed", type=float, default=4.0)
    parser.add_argument("--display-physical-seconds", type=float, default=208.0)
    parser.add_argument("--camera-distance", type=float, default=2.25)
    parser.add_argument("--camera-azimuth", type=float, default=125.0)
    parser.add_argument("--camera-elevation", type=float, default=-27.0)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--terminal-tolerance", type=float, default=2.5e-4)
    parser.add_argument("--font", type=Path, default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    parser.add_argument("--preview", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    result = run_candidate(parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False))
