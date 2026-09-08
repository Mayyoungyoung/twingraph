"""Minimal, independent physical scenes for demonstrations of the same atoms."""

from pathlib import Path
import os
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from .scene import SCENE, geom, holed_plate, fmt
from .control import HOME
from ..core.sim_context import MjContext
from .library import Session, GRASP

PIN_TARGET = np.array([0.04, 0.05, 0.855])


def build_demo_scene(kind, directory, view="overview"):
    root = ET.parse(SCENE).getroot()
    root.set("model", "isolated_skill_" + kind)
    assets = SCENE.parent.parent / "assets" / "panda"
    root.find("compiler").set(
        "meshdir", os.path.relpath(assets, Path(directory).resolve())
    )
    root.find("include").set(
        "file", os.path.relpath(assets / "panda.xml", Path(directory).resolve())
    )
    world = root.find("worldbody")
    keep = {
        "pin": {"pin_left", "pin_left_holder"},
        "rail": {"guide_base", "carriage"},
    }.get(kind, set())
    for body in list(world.findall("body")):
        if body.get("name") not in keep:
            world.remove(body)
    if kind in ("cube", "perception", "obstacle"):
        cubes = [("cube", [-0.18, -0.18, 0.82], ".15 .48 .78 1")]
        if kind == "perception":
            cubes += [
                ("cube_green", [0.02, 0.01, 0.82], ".23 .63 .38 1"),
                ("cube_orange", [0.12, -0.23, 0.82], ".9 .45 .13 1"),
            ]
        for name, pos, color in cubes:
            b = ET.SubElement(world, "body", name=name, pos=fmt(pos))
            ET.SubElement(b, "freejoint", name=name + "_free")
            geom(
                b,
                name + "_shape",
                [0, 0, 0],
                [0.02, 0.02, 0.02],
                color,
                friction="1 .02 .001",
            )
        parts = tuple(c[0] for c in cubes)
        specs = {p: (0.0, 0.04) for p in parts}
        if kind == "obstacle":
            geom(
                world,
                "obstacle",
                [-0.035, -0.07, 0.87],
                [0.045, 0.055, 0.07],
                ".55 .57 .60 1",
            )
    elif kind == "pin":
        # Keep the single supply pin in the central dexterous workspace.
        world.find("body[@name='pin_left']").set("pos", "-.22 -.22 .850")
        world.find("body[@name='pin_left_holder']").set("pos", "-.22 -.22 .819")
        receiver = ET.SubElement(
            world, "body", name="receiver", pos=fmt([*PIN_TARGET[:2], 0.827])
        )
        holed_plate(
            receiver,
            "receiver",
            (0.035, 0.035),
            0,
            0.027,
            [(0, 0)],
            ".35 .48 .6 1",
            holehalf=0.004,
        )
        parts = ("pin_left",)
        specs = {p: GRASP[p] for p in parts}
    elif kind == "rail":
        base = world.find("body[@name='guide_base']")
        for g in list(base.findall("geom")):
            if g.get("name", "").startswith("stop_locator"):
                base.remove(g)
        parts = ("carriage",)
        specs = {p: GRASP[p] for p in parts}
    else:
        raise ValueError(kind)
    # Fixed throughout each clip. A closer view makes the teaching objects clear.
    camera = world.find("camera")
    eye = np.array([0.64, -1.02, 1.70])
    target = np.array([-0.16, -0.07, 0.98])
    if kind == "pin":
        eye = np.array([0.38, -0.48, 1.35])
        target = np.array([0.01, 0.015, 0.915])
    z = (eye - target) / np.linalg.norm(eye - target)
    x = np.cross([0, 0, 1], z)
    x /= np.linalg.norm(x)
    camera.set("pos", fmt(eye))
    camera.set("xyaxes", fmt(np.r_[x, np.cross(z, x)]))
    camera.set("fovy", "42")
    if view == "front":
        # Fixed front-oblique close-ups: both jaws and the target remain visible.
        profiles = {
            "cube": ([0.40, -0.45, 1.16], [-0.18, -0.18, 0.90], 36),
            "perception": ([0.63, -0.62, 1.20], [-0.04, -0.10, 0.86], 40),
            "obstacle": ([0.66, -0.55, 1.30], [-0.02, -0.07, 1.01], 42),
            "pin": ([0.46, -0.22, 1.09], [0.04, 0.05, 0.895], 34),
            "rail": ([0.43, -0.28, 1.08], [0.065, 0.085, 0.86], 38),
        }
        eye, target, fovy = profiles[kind]
        eye, target = np.asarray(eye), np.asarray(target)
        z = (eye - target) / np.linalg.norm(eye - target)
        x = np.cross([0, 0, 1], z)
        x /= np.linalg.norm(x)
        camera.set("pos", fmt(eye))
        camera.set("xyaxes", fmt(np.r_[x, np.cross(z, x)]))
        camera.set("fovy", str(fovy))
        quality = root.find("visual/quality")
        quality.set("shadowsize", "4096")
        quality.set("offsamples", "8")
        headlight = root.find("visual/headlight")
        headlight.set("ambient", ".4 .4 .4")
        headlight.set("diffuse", ".55 .55 .55")
    elif view != "overview":
        raise ValueError(f"unknown camera profile {view}")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (kind + ".xml")
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="unicode")
    return path, parts, specs


def make_demo_session(kind, directory, view="overview"):
    path, parts, specs = build_demo_scene(kind, directory, view=view)
    ctx = MjContext(path, control_freq=50)
    ctx.reset()
    ctx.data.qpos[ctx.arm_qadr] = HOME
    mujoco.mj_forward(ctx.model, ctx.data)
    ctx.hold_arm()
    ctx.set_finger_ctrl(0.04)
    for _ in range(80):
        ctx.step()
    return Session(ctx, parts=parts, grasp_specs=specs)
