#!/usr/bin/env python3
"""Render the twelve formal v6 candidate traces in one styled MuJoCo world.

This is a presentation-only replay.  It never steps physics, changes outcomes, or
re-runs the value model.  Each candidate uses repeat-0 qpos recorded by the
formal full-policy digital-twin validation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


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

TOP4 = [
    "e29adb7ac28b5567e9ec",
    "e83f0990783267c02a3b",
    "b2df063756dbea03a712",
    "57ed526f426891e1124a",
]
SELECTED = "57ed526f426891e1124a"
OFFSETS = [
    (-2.985, 1.85),
    (-1.135, 1.85),
    (0.715, 1.85),
    (2.565, 1.85),
    (-2.985, 0.0),
    (-1.135, 0.0),
    (0.715, 0.0),
    (2.565, 0.0),
    (-2.985, -1.85),
    (-1.135, -1.85),
    (0.715, -1.85),
    (2.565, -1.85),
]
FREE_QPOS_STARTS = (9, 16, 23, 30, 37)


def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size=size)


def style_xml(source: Path, destination: Path) -> None:
    """Make a darker presentation-only copy without changing model topology."""
    tree = ET.parse(source)
    root = tree.getroot()
    headlight = root.find("./visual/headlight")
    if headlight is not None:
        headlight.set("ambient", "0.16 0.17 0.19")
        headlight.set("diffuse", "0.34 0.36 0.40")
        headlight.set("specular", "0.025 0.025 0.03")

    checker = root.find("./asset/texture[@name='checker']")
    if checker is not None:
        checker.set("rgb1", "0.10 0.115 0.14")
        checker.set("rgb2", "0.31 0.33 0.37")
    sky = root.find("./asset/texture[@type='skybox']")
    if sky is not None:
        sky.set("rgb1", "0.075 0.095 0.13")
        sky.set("rgb2", "0.25 0.29 0.35")
    material = root.find("./asset/material[@name='checker_mat']")
    if material is not None:
        material.set("texrepeat", "2.2 1.8")
        material.set("reflectance", "0.015")

    lights = root.findall("./worldbody/light")
    if lights:
        lights[0].set("diffuse", "0.48 0.50 0.54")
        lights[0].set("specular", "0.035 0.035 0.04")
    if len(lights) > 1:
        lights[1].set("diffuse", "0.16 0.18 0.22")
        lights[1].set("specular", "0.015 0.015 0.02")

    for geom in root.findall("./worldbody/geom"):
        name = geom.get("name", "")
        if not name.startswith("pad"):
            continue
        index = int(name[3:])
        candidate = CANDIDATES[index]
        if candidate == SELECTED:
            geom.set("rgba", "0.70 0.40 0.035 1")
        elif candidate in TOP4:
            geom.set("rgba", "0.035 0.40 0.14 1")
        else:
            geom.set("rgba", "0.12 0.135 0.16 1")

    for geom in root.findall(".//geom"):
        if geom.get("name", "").endswith("_table"):
            geom.set("rgba", "0.50 0.405 0.285 1")

    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="unicode")


def load_traces(case_dir: Path) -> list[np.ndarray]:
    traces = []
    for candidate in CANDIDATES:
        trace_path = case_dir / "full" / "validation" / f"{candidate}_0.npz"
        with np.load(trace_path) as trace:
            qpos = np.asarray(trace["qpos"], dtype=np.float64)
        if qpos.ndim != 2 or qpos.shape[1] != 44:
            raise ValueError(f"Unexpected qpos shape in {trace_path}: {qpos.shape}")
        traces.append(qpos)
    return traces


def apply_frame(data: mujoco.MjData, traces: list[np.ndarray], progress: float) -> None:
    for index, (trace, (dx, dy)) in enumerate(zip(traces, OFFSETS)):
        frame = min(round(progress * (len(trace) - 1)), len(trace) - 1)
        qpos = trace[frame].copy()
        for start in FREE_QPOS_STARTS:
            qpos[start] += dx
            qpos[start + 1] += dy
        data.qpos[index * 44 : (index + 1) * 44] = qpos


def overlay(frame: np.ndarray, progress: float, font_path: Path) -> np.ndarray:
    image = Image.fromarray(frame)
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    draw.rounded_rectangle((28, 24, 1115, 112), radius=18, fill=(12, 16, 24, 210))
    draw.text((52, 34), "TwinGraph · 12 个候选计划共享环境同步回放", font=_font(font_path, 34), fill=(242, 245, 250, 255))
    draw.text((52, 79), "绿色：Top-4   金色：最终执行方案   ·   正式实验 seed 71400", font=_font(font_path, 22), fill=(196, 207, 220, 255))

    bar_x0, bar_y0, bar_x1, bar_y1 = 1475, 48, 1880, 72
    draw.rounded_rectangle((bar_x0, bar_y0, bar_x1, bar_y1), radius=12, fill=(25, 31, 42, 215))
    filled = bar_x0 + int((bar_x1 - bar_x0) * progress)
    if filled > bar_x0:
        draw.rounded_rectangle((bar_x0, bar_y0, filled, bar_y1), radius=12, fill=(70, 157, 225, 255))
    draw.text((1475, 80), f"同步进度 {progress * 100:05.1f}%", font=_font(font_path, 18), fill=(220, 227, 237, 255))

    return np.asarray(Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB"))


def render(args: argparse.Namespace) -> None:
    case_dir = args.case_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    styled_xml = output_dir / "multi_candidate_shared_world_styled.xml"
    style_xml(args.source_xml.resolve(), styled_xml)

    model = mujoco.MjModel.from_xml_path(str(styled_xml))
    data = mujoco.MjData(model)
    traces = load_traces(case_dir)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.array(args.lookat, dtype=np.float64)
    camera.distance = args.distance
    camera.azimuth = args.azimuth
    camera.elevation = args.elevation

    frame_indices = [args.preview_frame] if args.preview else range(args.frames)
    writer = None
    output_path = output_dir / args.output_name
    if not args.preview:
        writer = imageio.get_writer(
            output_path,
            fps=args.fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
            ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )
    try:
        for frame_index in frame_indices:
            progress = frame_index / max(args.frames - 1, 1)
            apply_frame(data, traces, progress)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            frame = renderer.render()
            frame = overlay(frame, progress, args.font)
            if args.preview:
                imageio.imwrite(output_dir / args.preview_name, frame)
            else:
                writer.append_data(frame)
    finally:
        if writer is not None:
            writer.close()
        renderer.close()

    manifest = {
        "schema": "twingraph.showcase.shared_world.v2",
        "seed": 71400,
        "source": "formal full-policy validation repeat-0 qpos traces; presentation-only replay",
        "candidate_ids": CANDIDATES,
        "scores": SCORES,
        "top4": TOP4,
        "selected_candidate_id": SELECTED,
        "camera": {
            "distance": args.distance,
            "azimuth": args.azimuth,
            "elevation": args.elevation,
            "lookat": args.lookat,
        },
        "lighting": "reduced headlight and key/fill intensity; darker sky and checkerboard",
        "video": args.output_name,
        "fps": args.fps,
        "frames": args.frames,
        "duration_seconds": args.frames / args.fps,
        "model": styled_xml.name,
    }
    (output_dir / "shared_world_manifest_v2.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--source-xml", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-name", default="case_71400_12_candidates_shared_world_v2.mp4")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--distance", type=float, default=7.65)
    parser.add_argument("--azimuth", type=float, default=90.0)
    parser.add_argument("--elevation", type=float, default=-58.0)
    parser.add_argument("--lookat", type=float, nargs=3, default=(-0.2, 0.0, 0.68))
    parser.add_argument("--font", type=Path, default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-frame", type=int, default=90)
    parser.add_argument("--preview-name", default="preview_v2.png")
    return parser.parse_args()


if __name__ == "__main__":
    render(parse_args())
