"""Independent wide-clearance functional assembly task (V9 development)."""

from dataclasses import asdict, dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from simbench.assembly.control import HOME
from simbench.assembly.library import Session, GRASP, DEFAULT_CAPABILITIES
from simbench.assembly.scene import fmt, geom, ring, holed_plate
from simbench.core.sim_context import MjContext
from . import stage_v7
from .pin_geometry import PinInsertionConfig

TASK_VERSION = "functional_assembly_v9_wide_bore_1"
BORE_RADIUS_M = .0065
HOLDER_INNER_RADIUS_M = .0060
HOLDER_OUTER_RADIUS_M = .010
HOLDER_HALF_HEIGHT_M = .012
PIN_CONFIG = PinInsertionConfig(
    guide_inner_radius_m=BORE_RADIUS_M,
    plate_hole_half_width_m=BORE_RADIUS_M,
    guide_length_m=.008,
    required_depth_m=.006,
    source="stage_v9.write_scene physical bore and plate CAD",
)


@dataclass
class StageV9Spec:
    source_spec: dict
    task_version: str = TASK_VERSION
    collision_bore_radius_m: float = BORE_RADIUS_M
    holder_inner_radius_m: float = HOLDER_INNER_RADIUS_M
    holder_half_height_m: float = HOLDER_HALF_HEIGHT_M


def visual_templates(installed=False):
    templates = stage_v7.visual_templates(installed=installed)
    templates["end_stop"]["through_hole_half_width_m"] = BORE_RADIUS_M
    templates["end_stop"]["guide_inner_radius_m"] = BORE_RADIUS_M
    for part in ("pin_left", "pin_right"):
        templates[part]["shaft_radius_m"] = PIN_CONFIG.shaft_radius_m
        templates[part]["supply_holder_inner_radius_m"] = HOLDER_INNER_RADIUS_M
    return templates


def write_scene(source_spec, directory):
    directory = Path(directory)
    source_path = stage_v7.write_scene(source_spec, directory)
    root = ET.parse(source_path).getroot()
    root.set("model", TASK_VERSION)
    stop = root.find(".//body[@name='end_stop']")
    if stop is None:
        raise ValueError("end_stop body missing")
    for old in list(stop.findall("geom")):
        if old.get("name", "").startswith(("stop_", "v7_bore_")) and old.get("name") != "stop_boss":
            stop.remove(old)
    holed_plate(stop, "v9_stop", (.012, .043), 0., .018,
                 [(0., -.032), (0., .032)], ".74 .45 .22 1", holehalf=BORE_RADIUS_M)
    for side, y in (("left", -.032), ("right", .032)):
        ring(stop, f"v9_bore_{side}", [0., y, .014], BORE_RADIUS_M,
             .011, .004, ".43 .45 .47 1", n=16, friction=".3 .02 .001")
    for part in ("pin_left", "pin_right"):
        holder = root.find(f".//body[@name='{part}_holder']")
        if holder is None:
            raise ValueError(f"{part} holder missing")
        for old in list(holder.findall("geom")):
            holder.remove(old)
        ring(holder, f"v9_{part}_holder", [0., 0., 0.],
             HOLDER_INNER_RADIUS_M, HOLDER_OUTER_RADIUS_M,
             HOLDER_HALF_HEIGHT_M, ".44 .44 .43 1", n=12)
    path = directory / "stage_v9_scene.xml"
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="unicode")
    return path


def make_scene(seed, directory, role="development", level="L1"):
    directory = Path(directory)
    source_spec = stage_v7.StageV7Spec.sample(seed, level)
    spec = StageV9Spec(asdict(source_spec))
    path = write_scene(source_spec, directory)
    ctx = MjContext(path, control_freq=50)
    ctx.reset()
    ctx.data.qpos[ctx.arm_qadr] = HOME
    mujoco.mj_forward(ctx.model, ctx.data)
    ctx.hold_arm()
    ctx.set_finger_ctrl(.04)
    for _ in range(80):
        ctx.step()
    session = Session(ctx, seed=seed, out=directory, parts=stage_v7.ALL_PARTS,
                      grasp_specs={**GRASP, stage_v7.WIPE_TOOL: (.047, .028)},
                      capabilities={**DEFAULT_CAPABILITIES, stage_v7.WIPE_TOOL: ("wipe",)})
    targets = stage_v7.base.nominal_targets()
    session.stage_targets = targets
    session.scene_role = role
    session.stage_completed = ()
    session.level = level
    session.v7_spec = asdict(source_spec)
    session.task_version = TASK_VERSION
    session.pin_insertion_config = PIN_CONFIG
    session.visual_templates_fn = visual_templates
    from . import cleaning
    points = []
    names = [f"dirty_{i}" for i in range(32)]
    for name in names:
        sid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_SITE, name)
        points.append(ctx.model.site_pos[sid].copy())
    cleaning.initialize(session, points, names, seed=seed)
    stage_v7.install_visual(session, stage_v7.ALL_PARTS)
    return spec, session, path, targets
