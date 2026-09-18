"""Full clean–assemble–functional-test value-v7 scenario family."""

from dataclasses import asdict, dataclass
from pathlib import Path
import copy
import hashlib
import itertools
import math
import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from simbench.assembly.control import HOME
from simbench.assembly.scene import CENTER, SCENE, fmt, ring
from simbench.assembly.library import Session, GRASP, DEFAULT_CAPABILITIES
from simbench.core.sim_context import MjContext
from . import stage_v5 as base
from .observation import SensorConfig, install
from .plan import Call, Argument, PlanIR, argument, digest, plain
from simbench.assembly.candidates import fingerprint

PARTS = base.PARTS
WIPE_TOOL = "wipe_tool"
ALL_PARTS = (*PARTS, WIPE_TOOL)
TASK_SCOPE = "clean_assemble_five_parts_and_post_handle_bidirectional_stroke"
FAMILY = "sliding_stage_full_v7"
VIEW_NAMES = ("task_view", "top_view")
IMAGE_SIZE = 80


@dataclass
class StageV7Spec:
    seed: int
    level: str
    supply_shifts: dict
    supply_yaws_rad: dict
    camera_xy_jitter_m: list
    friction_scale: float
    mass_scale: float
    damping_scale: float
    family: str = FAMILY
    scope: str = TASK_SCOPE

    @classmethod
    def sample(cls, seed, level="L1"):
        limits = {"L0": (.012, 7), "L1": (.025, 15), "L2": (.040, 25)}
        if level not in limits:
            raise ValueError(level)
        span, yaw = limits[level]
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 907]))
        return cls(
            int(seed), level,
            {p: rng.uniform(-span, span, 2).tolist() for p in PARTS},
            {p: float(rng.uniform(-math.radians(yaw), math.radians(yaw))) for p in PARTS},
            rng.uniform(-.010 if level != "L2" else -.015, .010 if level != "L2" else .015, 2).tolist(),
            float(rng.uniform(.85, 1.15)), float(rng.uniform(.90, 1.10)),
            float(rng.uniform(.85, 1.15)),
        )

    @property
    def config_id(self):
        row = asdict(self); row.pop("seed")
        return digest(row)[:20]


def _add_checkerboard(root):
    asset = root.find("asset")
    tex = ET.SubElement(asset, "texture", name="v7_checker", type="2d",
                        builtin="checker", rgb1=".66 .68 .70", rgb2=".88 .89 .90",
                        width="512", height="512")
    ET.SubElement(asset, "material", name="v7_checker_mat", texture="v7_checker",
                  texrepeat="8 8", reflectance=".05")
    floor = root.find(".//geom[@name='floor']")
    if floor is not None:
        floor.set("material", "v7_checker_mat")


def _add_pin_guides(root):
    """Add physical, open bore guides to the two end-stop holes.

    The source fixture's rectangular cut-out leaves the narrow pin head close
    to an edge under perception error.  A flush annular seat is a real
    fixture geometry change (not a weld or state assignment): it supports the
    head at the plate top without constraining the shaft during insertion.
    """
    stop = root.find(".//body[@name='end_stop']")
    if stop is None:
        raise ValueError("end_stop body missing")
    for name, y in (("left", -.032), ("right", .032)):
        # Flush with the plate top (local z=.018; end-stop target z=.836),
        # leaving a generous central clearance for the 3.3 mm shaft.  A
        # raised collar blocks the physical insertion path, so support is
        # intentionally kept flush while the remaining tilt failure is
        # reported rather than hidden by a permissive fixture.
        ring(stop, f"v7_bore_{name}", [0, y, .014], .0055, .011, .004,
             ".43 .45 .47 1", n=16, friction=".3 .02 .001")


def _add_tool_and_stains(root, seed):
    world = root.find("worldbody")
    tool = ET.SubElement(world, "body", name=WIPE_TOOL, pos="-.31 .20 .804")
    ET.SubElement(tool, "freejoint", name="wipe_tool_free")
    ET.SubElement(tool, "geom", name="wipe_pad", type="box", pos="0 0 0",
                  size=".018 .012 .004", rgba=".16 .66 .68 1",
                  friction=".25 .01 .001", mass=".025", solref=".03 1", solimp=".8 .95 .002")
    ET.SubElement(tool, "geom", name="wipe_handle", type="box", pos="0 0 .047",
                  size=".012 .014 .022", rgba=".85 .65 .25 1", friction="1 .02 .001", mass=".045")
    ET.SubElement(tool, "geom", name="wipe_stem", type="box", pos="0 0 .014",
                  size=".006 .007 .011", rgba=".85 .65 .25 1", contype="0", conaffinity="0", mass="0")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 9201]))
    site_names = []
    # World-space stain sites lie above the guide-base top; they are visual
    # only and their alpha is changed by the contact-state model.
    for i, (x, y) in enumerate(itertools.product(np.linspace(-.050, .050, 8), np.linspace(-.004, .004, 4))):
        name = f"dirty_{i}"
        site_names.append(name)
        ET.SubElement(world, "site", name=name,
                      pos=fmt([CENTER[0] + x + rng.uniform(-.003, .003),
                               CENTER[1] + y + rng.uniform(-.002, .002), .821]),
                      type="box", size=".006 .003 .0005", rgba=".46 .22 .08 1")
    return site_names


def write_scene(spec, directory):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    root = ET.parse(SCENE).getroot(); root.set("model", FAMILY)
    world = root.find("worldbody")
    for part in PARTS:
        delta = np.asarray(spec.supply_shifts[part], float)
        yaw = float(spec.supply_yaws_rad[part])
        if delta.shape != (2,) or np.max(np.abs(delta)) > ({"L0": .012, "L1": .025, "L2": .040}[spec.level] + 1e-8):
            raise ValueError("supply shift outside level envelope")
        for name in ((part, part + "_holder") if part.startswith("pin_") else (part,)):
            body = world.find(f"body[@name='{name}']")
            xyz = np.fromstring(body.get("pos", "0 0 0"), sep=" "); xyz[:2] += delta
            body.set("pos", fmt(xyz)); body.set("quat", fmt([math.cos(yaw/2), 0, 0, math.sin(yaw/2)]))
    _add_pin_guides(root)
    _add_checkerboard(root)
    _add_tool_and_stains(root, spec.seed)
    jitter = np.asarray(spec.camera_xy_jitter_m, float)
    ET.SubElement(world, "camera", name="top_view", pos=fmt(np.r_[[-.15, -.08] + jitter, 1.62]),
                  xyaxes="1 0 0 0 1 0", fovy="46")
    assets = SCENE.parent.parent / "assets" / "panda"
    root.find("compiler").set("meshdir", os.path.relpath(assets, directory.resolve()).replace("\\", "/"))
    root.find("include").set("file", os.path.relpath(assets / "panda.xml", directory.resolve()).replace("\\", "/"))
    path = directory / "stage_v7_scene.xml"; ET.indent(root, space="  "); ET.ElementTree(root).write(path, encoding="unicode")
    return path


def make_scene(seed, directory, role="development", level="L1"):
    spec = StageV7Spec.sample(seed, level)
    path = write_scene(spec, directory)
    ctx = MjContext(path, control_freq=50); ctx.reset(); ctx.data.qpos[ctx.arm_qadr] = HOME
    mujoco.mj_forward(ctx.model, ctx.data); ctx.hold_arm(); ctx.set_finger_ctrl(.04)
    for _ in range(80): ctx.step()
    session = Session(ctx, seed=seed, parts=ALL_PARTS,
                      grasp_specs={**GRASP, WIPE_TOOL: (.047, .028)},
                      capabilities={**DEFAULT_CAPABILITIES, WIPE_TOOL: ("wipe",)})
    targets = base.nominal_targets()
    # Keep the source pin reference height, now stabilized by the explicit
    # physical bore guides added above.  The pin shaft remains clear of the
    # table while the guide prevents post-release lateral escape.
    session.stage_targets = targets; session.scene_role = role
    session.stage_completed = (); session.level = level; session.v7_spec = asdict(spec)
    from . import cleaning
    points = []
    names = [f"dirty_{i}" for i in range(32)]
    for i, name in enumerate(names):
        sid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_SITE, name)
        points.append(ctx.model.site_pos[sid].copy())
    cleaning.initialize(session, points, names, seed=seed)
    # One frozen structured observation is shared by every candidate and view.
    install(session, ALL_PARTS, SensorConfig(position_noise_std_m=.001 if level != "L2" else .002,
                                             yaw_noise_std_rad=math.radians(1.0 if level != "L2" else 2.0),
                                             seed=seed))
    return spec, session, path, targets


def observed(session, targets):
    obs = {"robot": {"joints": session.ctx.arm_qpos.tolist(), "fingers": session.ctx.finger_qpos.tolist(),
                      "eef": session.ctx.eef_pos().tolist()}, "objects": {}, "goals": [],
           "perception": {"backend": "simulated_sensor_proxy", "manifest": copy.deepcopy(session.sensor_manifest),
                          "observation_sha256": session.decision_observation.get("sha256") if session.decision_observation else None}}
    for part in ALL_PARTS:
        row = session.decision_observation["objects"][part]
        obs["objects"][part] = {"position": list(row["position_m"]), "quaternion": list(row["quat_wxyz"]),
                                 "category": "wipe_tool" if part == WIPE_TOOL else "product_part"}
    for part, target in targets.items():
        obs["goals"].append(dict(predicate="seated_released_retracted", manipulated=part,
                                  position=target, position_tolerance=.0025 if part.startswith("pin_") else .0015,
                                  tilt_tolerance_deg=3.,
                                  minimum_eef_clearance_m=.02))
    return obs


def capture_vision(session, size=IMAGE_SIZE):
    renderer = mujoco.Renderer(session.ctx.model, height=size, width=size); result = {}
    try:
        for name in VIEW_NAMES:
            renderer.disable_depth_rendering(); renderer.update_scene(session.ctx.data, camera=name)
            result[f"{name}_rgb"] = renderer.render().copy().astype(np.uint8)
            renderer.enable_depth_rendering(); renderer.update_scene(session.ctx.data, camera=name)
            result[f"{name}_depth_mm"] = np.clip(np.rint(renderer.render().copy()*1000), 0, 65535).astype(np.uint16)
    finally:
        close = getattr(renderer, "close", None) or getattr(renderer, "free", None)
        if close: close()
    return result


def save_vision(path, arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(path, **arrays)
    h = hashlib.sha256()
    for key in sorted(arrays):
        v = np.ascontiguousarray(arrays[key]); h.update(key.encode()); h.update(str(v.dtype).encode()); h.update(str(v.shape).encode()); h.update(v.tobytes())
    return dict(schema="twingraph.rgbd.v7", file=path.name, sha256=h.hexdigest(), cameras=list(VIEW_NAMES), image_size=IMAGE_SIZE,
                channels="uint8 RGB plus uint16 metric depth in millimetres", decision_boundary="pre-execution")


def _full_plan(session, targets, order, choices, wipe_variant, wipe_force, wipe_duration, stroke_minimum):
    params = dict(order=list(order), choices=plain(choices), wipe_variant=int(wipe_variant), wipe_force=float(wipe_force),
                  wipe_duration=float(wipe_duration), stroke_minimum=float(stroke_minimum))
    cid = digest(dict(scope=TASK_SCOPE, params=params, start=fingerprint(session)))[:20]
    call = Call("full_task", "run_full_task_v7", {k: argument(v) for k, v in params.items()}, {"manipulated": WIPE_TOOL}, "executable")
    payload = dict(id=cid, part=WIPE_TOOL, execution="full_task_v7", start_state=fingerprint(session), steps=[dict(skill=call.skill, params=params)],
                   task_scope=TASK_SCOPE, order=list(order), choices=plain(choices), wipe_variant=int(wipe_variant),
                   wipe_force=float(wipe_force), wipe_duration=float(wipe_duration), stroke_minimum=float(stroke_minimum), semantic_program_id=cid)
    return PlanIR(cid, [call], 1, payload, "unknown", protocol="full_task.v7").validate(session.parts)


def legal_orders(orders=None):
    return base.legal_orders(orders)


def build_pool(session, targets, seed, n=12, level=None):
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 991]))
    orders = base.legal_orders()
    plans = []; seen = set(); attempts = 0
    while len(plans) < n and attempts < 5000:
        attempts += 1; order = orders[int(rng.integers(len(orders)))]
        choices = {}
        for part in PARTS:
            choices[part] = dict(yaw=float(rng.choice([0., np.pi/2])), height=float(rng.choice([-.002, 0., .002])),
                                 clearance=float(rng.choice([.96, .98, 1.02])), force=float(rng.choice([2.5, 2.8, 3.2, 4.5])),
                                 # Slow contact insertions are deliberate
                                 # executable alternatives; the old pool only
                                 # sampled 6--8 mm/s and could lose a narrow
                                 # pin during vertical seating.
                                 speed=float(rng.choice([.003, .0045, .006, .007])))
        params = (tuple(order), tuple((p, tuple(sorted(choices[p].items()))) for p in PARTS),
                  int(rng.integers(4)), float(rng.choice([1.3, 1.5, 1.7])), float(rng.choice([12., 14., 16.])), float(rng.choice([.08, .09])))
        key = repr(params)
        if key in seen: continue
        seen.add(key)
        plans.append(_full_plan(session, targets, order, choices, params[2], params[3], params[4], params[5]))
    if len(plans) < n: raise RuntimeError("v7 candidate pool exhausted")
    return plans, dict(requested=n, raw_count=attempts, deduplicated=len(plans), executable_semantic_unique=len(seen),
                       structure_branches=len({tuple(p.prefix["order"]) for p in plans}),
                       grasp_branches=len({(p.prefix["choices"]["carriage"]["yaw"], p.prefix["choices"]["carriage"]["height"]) for p in plans}),
                       wipe_path_branches=len({(p.prefix["wipe_variant"], p.prefix["wipe_duration"]) for p in plans}),
                       task_scope=TASK_SCOPE, label_blind=True)


def rebind_plan(session, targets, selected_plan):
    old = selected_plan if isinstance(selected_plan, PlanIR) else PlanIR.from_dict(selected_plan)
    old.validate(session.parts)
    return _full_plan(session, targets, old.prefix["order"], copy.deepcopy(old.prefix["choices"]), old.prefix["wipe_variant"],
                      old.prefix["wipe_force"], old.prefix["wipe_duration"], old.prefix["stroke_minimum"])
