#!/usr/bin/env python3
"""Render recorded independent-target states without re-running any physics.

Example: python scripts/render_value_v5_execution.py --run results/v5/system/run/top_k \
    --repeat 0 --out results/v5/system/run/top_k/target_replay.mp4 --speed 4
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import time


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recorded_path(run, declared, override, expected_sha, label):
    if not declared or not expected_sha:
        raise ValueError(f"recorded {label} path and SHA256 are required")
    if override is not None:
        candidates = [Path(override)]
    else:
        # Original paths are retained in evidence. A moved run may resolve its
        # own basename/subdirectory, but it still must have identical bytes.
        declared_path = Path(declared)
        candidates = [declared_path, run / declared_path.name]
        if label == "scene":
            candidates.append(run / declared_path.parent.name / declared_path.name)
    existing = [p for p in candidates if p.is_file()]
    for path in existing:
        if file_sha(path) == expected_sha:
            return path.resolve()
    if existing:
        raise ValueError(f"{label} SHA256 does not match the recorded execution")
    raise FileNotFoundError(f"recorded {label} unavailable; supply --{label} with the same file bytes")


def sample_indices(times, fps, speed, max_frames):
    """Previous recorded sample at each video time; never interpolate states."""
    import numpy as np
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError("time_s must be a nonempty finite, nondecreasing vector")
    duration = float(times[-1] - times[0])
    count = max(1, math.ceil(duration * fps / speed) + 1)
    if count > max_frames:
        raise ValueError(f"requested {count} frames exceeds --max-frames={max_frames}; increase speed or explicit bound")
    requested = np.minimum(times[0] + np.arange(count) * speed / fps, times[-1])
    indices = np.clip(np.searchsorted(times, requested, side="right") - 1, 0, len(times) - 1)
    indices[-1] = len(times) - 1
    return indices, requested


def render(args):
    import imageio
    import imageio.v2 as imageio_v2
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    started = time.perf_counter()
    run = Path(args.run).resolve()
    execution_path = run / f"target_execution_{args.repeat}.json"
    row = json.loads(execution_path.read_text(encoding="utf-8"))
    if row.get("status") != "executed" or row.get("repeat") != args.repeat:
        raise ValueError("requested target execution was not actually reached")
    execution = row["execution"]
    record = execution["state_trace"]
    trace_path = recorded_path(run, record.get("path"), args.trace, record.get("sha256"), "trace")
    scene_path = recorded_path(run, record.get("scene_path"), args.scene, record.get("scene_sha256"), "scene")
    output = Path(args.out).resolve()
    if output.suffix.lower() != ".mp4":
        raise ValueError("--out must end in .mp4")
    poster_path, metadata_path = output.with_suffix(".png"), output.with_suffix(".json")
    partial = output.with_name(output.stem + ".partial.mp4")
    for path in (output, poster_path, metadata_path, partial):
        if path.exists():
            raise FileExistsError(f"refuse to overwrite existing replay artifact: {path}")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
    if camera < 0:
        raise ValueError(f"scene has no camera named {args.camera}")
    with np.load(trace_path, allow_pickle=False) as saved:
        if set(saved.files) != {"time_s", "qpos", "qvel", "ctrl"}:
            raise ValueError("trace must contain exactly time_s/qpos/qvel/ctrl")
        arrays = {key: saved[key].copy() for key in saved.files}
    samples = len(arrays["time_s"])
    if record.get("samples") != samples:
        raise ValueError("trace sample count differs from recorded metadata")
    for key, width in (("qpos", model.nq), ("qvel", model.nv), ("ctrl", model.nu)):
        value = arrays[key]
        if value.shape != (samples, width) or value.dtype.kind not in "fi" or not np.isfinite(value).all():
            raise ValueError(f"{key} shape/values do not match the saved scene")
    indices, requested_times = sample_indices(arrays["time_s"], args.fps, args.speed, args.max_frames)
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), args.width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), args.height)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    writer = None
    last_frame = None
    try:
        writer = imageio_v2.get_writer(str(partial), fps=args.fps, codec="libx264",
                                   quality=8, macro_block_size=2, pixelformat="yuv420p")
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", max(9, min(22, args.width // 70)))
        except OSError:
            font = ImageFont.load_default()
        for index in indices:
            index = int(index)
            for key in ("qpos", "qvel", "ctrl"):
                getattr(data, key)[:] = arrays[key][index]
            data.time = float(arrays["time_s"][index])
            # Forward kinematics/contact visualization ONLY. mj_step and robot
            # controllers are intentionally absent from this replay script.
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=args.camera)
            picture = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(picture)
            draw.rectangle((0, 0, args.width, 68), fill=(18, 24, 30))
            draw.text((12, 9), f"RECORDED TARGET-STATE TRACE REPLAY | {args.speed:g}x | no physics re-run",
                      font=font, fill=(244, 248, 252))
            elapsed = data.time - float(arrays["time_s"][0])
            outcome = "SUCCESS" if row.get("success") else "FAILURE"
            draw.text((12, 39), f"Recorded t={data.time:.2f}s (+{elapsed:.2f}s) | saved terminal result: {outcome}",
                      font=font, fill=(200, 221, 232))
            last_frame = np.asarray(picture)
            writer.append_data(last_frame)
        writer.close()
        writer = None
    finally:
        if writer is not None:
            writer.close()
        if hasattr(renderer, "close"):
            renderer.close()
        elif hasattr(renderer, "_mjr_context"):
            renderer._mjr_context.free()
    partial.replace(output)
    Image.fromarray(last_frame).save(poster_path)
    selected_times = arrays["time_s"][indices]
    metadata = dict(schema="twingraph.recorded_target_replay.v5",
        interpretation="visualization of previously recorded independent-target states; no new physical execution, feasibility label or rollout",
        execution_path=str(execution_path), execution_sha256=file_sha(execution_path),
        trace_path=str(trace_path), trace_sha256=file_sha(trace_path),
        scene_path=str(scene_path), scene_sha256=file_sha(scene_path),
        selected_candidate_id=row.get("selected_candidate_id"),
        rebound_candidate_id=row.get("rebound_candidate_id"),
        recorded_success=bool(row.get("success")), recorded_trial=execution.get("trial"),
        simulation_integration_steps=0, recorded_samples=samples,
        first_recorded_time_s=float(arrays["time_s"][0]), last_recorded_time_s=float(arrays["time_s"][-1]),
        recorded_duration_s=float(arrays["time_s"][-1] - arrays["time_s"][0]),
        sampling=record.get("sampling"), replay_sampling="previous recorded sample at requested video time; terminal sample included; no state interpolation",
        fps=args.fps, nominal_speed_multiplier=args.speed, frames=len(indices),
        video_duration_s=len(indices) / args.fps, width=args.width, height=args.height,
        camera=args.camera, sampled_indices=indices.tolist(),
        maximum_sample_lag_s=float(np.max(requested_times - selected_times)),
        unique_displayed_states=int(len(np.unique(indices))),
        video_sha256=file_sha(output), terminal_png_sha256=file_sha(poster_path),
        renderer_versions=dict(python=platform.python_version(), mujoco=mujoco.__version__,
                               numpy=np.__version__, imageio=imageio.__version__),
        render_script_sha256=file_sha(__file__), wall_seconds=time.perf_counter() - started)
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    return dict(video=str(output), terminal_png=str(poster_path), metadata=str(metadata_path),
                frames=len(indices), recorded_success=metadata["recorded_success"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="system policy directory containing target_execution_<repeat>.json")
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--out", required=True, help="new MP4 output; sibling PNG and JSON are also written")
    parser.add_argument("--trace", help="relocated recorded NPZ; SHA256 must match")
    parser.add_argument("--scene", help="relocated recorded scene XML; SHA256 must match")
    parser.add_argument("--camera", default="task_view")
    parser.add_argument("--fps", type=float, default=25.)
    parser.add_argument("--speed", type=float, default=4.)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--max-frames", type=int, default=20000)
    args = parser.parse_args()
    if (not math.isfinite(args.fps) or not math.isfinite(args.speed) or not 1 <= args.fps <= 120
            or not 0 < args.speed <= 100 or args.repeat < 0 or args.max_frames < 1
            or not 160 <= args.width <= 3840 or not 120 <= args.height <= 2160
            or args.width % 2 or args.height % 2):
        parser.error("invalid playback settings; use finite positive FPS/speed, nonnegative repeat and bounded even resolution")
    Path(args.out).resolve().parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(render(args), indent=2))


if __name__ == "__main__":
    main()
