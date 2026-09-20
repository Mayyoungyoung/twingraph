"""Independent wide-clearance functional assembly task (V9 development)."""

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from simbench.assembly.control import HOME
from simbench.assembly.library import Session, GRASP, DEFAULT_CAPABILITIES
from simbench.assembly.scene import fmt, geom, ring, holed_plate
from simbench.core.sim_context import MjContext
from . import stage_v7
from .pin_geometry import PinInsertionConfig

TASK_VERSION = "functional_assembly_v9_funnel_r1"
BORE_RADIUS_M = .0075
BORE_PROFILE = ((.002, .0084), (.004, .0081), (.006, .0078), (.008, BORE_RADIUS_M))
HOLDER_INNER_RADIUS_M = .0060
HOLDER_OUTER_RADIUS_M = .010
HOLDER_HALF_HEIGHT_M = .012
PIN_SUPPLY_MIN_X_M = -.357
PIN_CONFIG = PinInsertionConfig(
    guide_inner_radius_m=BORE_RADIUS_M,
    plate_hole_half_width_m=BORE_RADIUS_M,
    guide_length_m=.008,
    required_depth_m=.006,
    radial_clearance_m=0.0,
    bore_profile=BORE_PROFILE,
    source="stage_v9.write_scene physical bore and plate CAD",
)


@dataclass
class StageV9Spec:
    source_spec: dict
    task_version: str = TASK_VERSION
    collision_bore_radius_m: float = BORE_RADIUS_M
    holder_inner_radius_m: float = HOLDER_INNER_RADIUS_M
    holder_half_height_m: float = HOLDER_HALF_HEIGHT_M
    pin_supply_min_x_m: float = PIN_SUPPLY_MIN_X_M
    setup: str = "complete_task"


def visual_templates(installed=False):
    templates = stage_v7.visual_templates(installed=installed)
    templates["end_stop"]["through_hole_half_width_m"] = BORE_RADIUS_M
    templates["end_stop"]["guide_inner_radius_m"] = BORE_RADIUS_M
    templates["end_stop"]["entrance_bore_profile_depth_radius_m"] = [list(row) for row in BORE_PROFILE]
    for part in ("pin_left", "pin_right"):
        templates[part]["shaft_radius_m"] = PIN_CONFIG.shaft_radius_m
        templates[part]["supply_holder_inner_radius_m"] = HOLDER_INNER_RADIUS_M
    return templates


def write_scene(source_spec, directory, preinstalled_end_stop=False):
    directory = Path(directory)
    source_path = stage_v7.write_scene(source_spec, directory)
    root = ET.parse(source_path).getroot()
    root.set("model", TASK_VERSION)
    stop = root.find(".//body[@name='end_stop']")
    if stop is None:
        raise ValueError("end_stop body missing")
    if preinstalled_end_stop:
        stop.set("pos", fmt(stage_v7.base.nominal_targets()["end_stop"]))
    for old in list(stop.findall("geom")):
        if old.get("name", "").startswith(("stop_", "v7_bore_")) and old.get("name") != "stop_boss":
            stop.remove(old)
    holes = [(0., -.032), (0., .032)]
    holed_plate(stop, "v9_stop_lower", (.012, .043), -.004, .014,
                 holes, ".74 .45 .22 1", holehalf=BORE_RADIUS_M)
    for band, (_end_depth, radius) in enumerate(BORE_PROFILE):
        z = .017 - .002 * band
        holed_plate(stop, f"v9_stop_band{band}", (.012, .043), z, .001,
                     holes, ".74 .45 .22 1", holehalf=radius)
        for side, y in (("left", -.032), ("right", .032)):
            ring(stop, f"v9_bore_{side}_band{band}", [0., y, z], radius,
                 .011, .001, ".43 .45 .47 1", n=16, friction=".3 .02 .001")
    effective_pin_supply = {}
    for part in ("pin_left", "pin_right"):
        for body_name in (part, f"{part}_holder"):
            supply_body = root.find(f".//body[@name='{body_name}']")
            xyz = np.fromstring(supply_body.get("pos"), sep=" ")
            xyz[0] = max(float(xyz[0]), PIN_SUPPLY_MIN_X_M)
            supply_body.set("pos", fmt(xyz))
            if body_name == part:
                effective_pin_supply[part] = xyz.tolist()
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
    (directory / "geometry_manifest.json").write_text(json.dumps({
        "task_version": TASK_VERSION,
        "setup": "isolated_pin_preinstalled_fixture" if preinstalled_end_stop else "complete_task",
        "cad_source": "stage_v9.write_scene parametric MJCF collision geometry",
        "plate_square_through_hole_half_width_m": BORE_RADIUS_M,
        "entrance_bore_profile_depth_radius_m": [list(row) for row in BORE_PROFILE],
        "annular_guide_inner_radius_m": BORE_RADIUS_M,
        "annular_guide_outer_radius_m": .011,
        "annular_guide_length_m": .008,
        "holder_inner_radius_m": HOLDER_INNER_RADIUS_M,
        "holder_outer_radius_m": HOLDER_OUTER_RADIUS_M,
        "holder_height_m": 2 * HOLDER_HALF_HEIGHT_M,
        "holder_segments": 12,
        "pin_supply_min_x_m": PIN_SUPPLY_MIN_X_M,
        "effective_pin_supply_positions_m": effective_pin_supply,
        "pin_acceptance": PIN_CONFIG.manifest(),
        "perception_templates": visual_templates(),
    }, indent=2), encoding="utf-8")
    return path


def make_scene(seed, directory, role="development", level="L1", preinstalled_end_stop=False):
    directory = Path(directory)
    source_spec = stage_v7.StageV7Spec.sample(seed, level)
    spec = StageV9Spec(asdict(source_spec), setup="isolated_pin_preinstalled_fixture" if preinstalled_end_stop else "complete_task")
    path = write_scene(source_spec, directory, preinstalled_end_stop=preinstalled_end_stop)
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
