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
from .rgbd_perception import CameraCalibration, estimate_scene, to_sensor_observation
from .plan import Call, Argument, PlanIR, argument, digest, plain
from simbench.assembly.candidates import fingerprint

PARTS = base.PARTS
WIPE_TOOL = "wipe_tool"
ALL_PARTS = (*PARTS, WIPE_TOOL)
TASK_SCOPE = "clean_assemble_five_parts_and_post_handle_bidirectional_stroke"
TASK_STROKE_MINIMUM_M = .08
FAMILY = "sliding_stage_full_v7"
VIEW_NAMES = ("task_view", "top_view")
IMAGE_SIZE = 80
DETECTOR_SIZE = (480, 640)  # H, W; value inputs remain 80x80


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
    base = root.find(".//body[@name='guide_base']")
    if base is None:
        raise ValueError("guide_base body missing")
    # Passive cradle rails are part of the generated fixture geometry.  The
    # original scene only had low corner contacts, allowing a valid pin
    # insertion to push the free end-stop out of the pocket.
    for name, pos, size in (
        # Split the side rails around the carriage centreline so the loose
        # carriage can still travel through the pocket before the stop is
        # seated.
        ("stop_cradle_left_front", [-.111, -.042, .021], [.004, .008, .021]),
        ("stop_cradle_left_back", [-.111, .042, .021], [.004, .008, .021]),
        ("stop_cradle_right_front", [-.073, -.042, .021], [.004, .008, .021]),
        ("stop_cradle_right_back", [-.073, .042, .021], [.004, .008, .021]),
        ("stop_cradle_front", [-.092, -.052, .021], [.016, .004, .021]),
        ("stop_cradle_back", [-.092, .052, .021], [.016, .004, .021]),
    ):
        ET.SubElement(base, "geom", name=name, type="box", pos=fmt(pos), size=fmt(size),
                      rgba=".48 .51 .53 1", friction=".6 .01 .001",
                      condim="4", solref=".008 1", solimp=".95 .99 .001")


def _add_tool_and_stains(root, seed):
    world = root.find("worldbody")
    # Keep the tool in a visible, reachable supply area.  The previous point
    # was routinely occluded by the robot shoulder in both cameras, which
    # made a genuine RGB-D detector correctly return unknown before cleaning.
    tool = ET.SubElement(world, "body", name=WIPE_TOOL, pos="-.24 -.08 .820")
    ET.SubElement(tool, "freejoint", name="wipe_tool_free")
    ET.SubElement(tool, "geom", name="wipe_pad", type="box", pos="0 0 0",
                  size=".018 .012 .010", rgba=".16 .66 .68 1",
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
    session = Session(ctx, seed=seed, out=directory, parts=ALL_PARTS,
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
    # The default v7 scene now uses the declared RGB-D geometry backend.  The
    # old noisy pose proxy remains selectable by callers for diagnostic
    # comparisons, but is never an implicit fallback for missing detections.
    install_visual(session, ALL_PARTS)
    return spec, session, path, targets


def observed(session, targets):
    obs = {"robot": {"joints": session.ctx.arm_qpos.tolist(), "fingers": session.ctx.finger_qpos.tolist(),
                      "eef": session.ctx.eef_pos().tolist()}, "objects": {}, "goals": [],
           "perception": {"backend": session.decision_observation.get("backend") if session.decision_observation else None, "manifest": copy.deepcopy(session.sensor_manifest),
                          "observation_sha256": session.decision_observation.get("sha256") if session.decision_observation else None}}
    for part in ALL_PARTS:
        row = session.decision_observation["objects"][part]
        obs["objects"][part] = {"position": list(row["position_m"]), "quaternion": list(row["quat_wxyz"]),
                                 "category": "wipe_tool" if part == WIPE_TOOL else "product_part"}
    for part, target in targets.items():
        if part.startswith("pin_"):
            obs["goals"].append(dict(predicate="pin_inserted_in_hole", manipulated=part,
                                      hole_part="end_stop", hole_offset_m=[0., -.032 if part == "pin_left" else .032, 0.],
                                      minimum_insertion_depth_m=0.006,
                                      minimum_eef_clearance_m=.02))
        else:
            obs["goals"].append(dict(predicate="seated_released_retracted", manipulated=part,
                                      position=target, position_tolerance=.0025 if part.startswith("pin_") else .0015,
                                      tilt_tolerance_deg=3.,
                                      minimum_eef_clearance_m=.02))
    return obs


def capture_vision(session, size=IMAGE_SIZE):
    if isinstance(size, (tuple, list)):
        height, width = map(int, size)
    else:
        height = width = int(size)
    renderer = mujoco.Renderer(session.ctx.model, height=height, width=width); result = {}
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


def camera_calibrations(session, size=DETECTOR_SIZE):
    """Return calibrated pinhole metadata for the rendered detector frames."""
    if isinstance(size, (tuple, list)):
        height, width = map(int, size)
    else:
        height = width = int(size)
    out = {}
    for name in VIEW_NAMES:
        cid = mujoco.mj_name2id(session.ctx.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if cid < 0:
            raise ValueError(f"camera {name} missing")
        fovy = math.radians(float(session.ctx.model.cam_fovy[cid]))
        fy = 0.5 * height / math.tan(fovy / 2.0)
        fx = fy
        R = np.asarray(session.ctx.data.cam_xmat[cid], dtype=float).reshape(3, 3)
        t = np.asarray(session.ctx.data.cam_xpos[cid], dtype=float)
        out[name] = CameraCalibration(width, height, fx, fy, (width - 1) / 2.0,
                                      (height - 1) / 2.0,
                                      tuple(np.block([[R, t[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]).ravel()))
    return out


def visual_templates(installed=False):
    """Static CAD appearance/size hints; no per-scene pose is embedded."""
    return {
        "carriage": {"color_rgb": [119, 190, 247], "size_m": [.054, .046, .032], "pixel_area_hint": 260, "instances": 1,
                      "reference_offset_m": [0., 0., -.040], "world_bounds": [[-.40, -.50], [.20, .25]]},
        "end_stop": {"color_rgb": [255, 189, 109], "size_m": [.024, .086, .036], "pixel_area_hint": 450, "instances": 1,
                     "reference_offset_m": [0., 0., -.031],
                     "yaw_offset_rad": -math.pi / 2,
                     "world_bounds": [[-.35, -.50], [.20, .25]]},
        "handle": {"color_rgb": [54, 60, 68], "size_m": [.042, .042, .016], "pixel_area_hint": 300, "instances": 1, "symmetric_axis": True,
                   "world_bounds": [[-.50, -.50], [.30, .30]],
                   **({"anchor_part": "carriage", "anchor_offset_m": [0., 0., .048],
                       "anchor_max_distance_m": .035, "world_z_bounds": [.85, .93],
                       "rgb_tolerance": 110.0} if installed else {})},
        "pin_left": {"color_rgb": [255, 244, 104], "size_m": [.018, .018, .060], "pixel_area_hint": 150, "instances": 1, "symmetric_axis": True, "reference_offset_m": [0., 0., -.013],
                     "world_bounds": [[-.43, -.29], [-.28, -.14]]},
        "pin_right": {"color_rgb": [255, 244, 104], "size_m": [.018, .018, .060], "pixel_area_hint": 150, "instances": 1, "symmetric_axis": True, "reference_offset_m": [0., 0., -.013],
                      "world_bounds": [[-.43, -.42], [-.28, -.27]]},
        "wipe_tool": {"color_rgb": [36, 149, 153], "size_m": [.036, .024, .055], "pixel_area_hint": 150, "instances": 1,
                       "reference_offset_m": [-.002, .010, 0.0],
                       "world_bounds": [[-.50, -.50], [.20, .50]]},
    }


def capture_detector(session, size=DETECTOR_SIZE):
    arrays = capture_vision(session, size=size)
    calibrations = camera_calibrations(session, size=size)
    frames = {name: {"rgb": arrays[f"{name}_rgb"], "depth_mm": arrays[f"{name}_depth_mm"]} for name in VIEW_NAMES}
    return frames, calibrations


def install_visual(session, parts=ALL_PARTS, size=DETECTOR_SIZE):
    frames, calibrations = capture_detector(session, size=size)
    installed = bool(getattr(session, "stage_passes", {}).get("assembly_pass"))
    templates_fn = getattr(session, "visual_templates_fn", visual_templates)
    result = estimate_scene(frames, calibrations, templates_fn(installed=installed))
    observation = to_sensor_observation(result, parts)
    observation["calibration"] = {k: v.manifest() for k, v in calibrations.items()}
    observation["detector_resolution"] = [int(size[1]), int(size[0])] if isinstance(size, (tuple, list)) else [int(size), int(size)]
    observation["config"] = {"backend": "rgbd_geometry", "detector_resolution": observation["detector_resolution"],
                               "value_input_resolution": [IMAGE_SIZE, IMAGE_SIZE], "calibration_version": "mujoco-pinhole-v1"}
    unsigned = {k: v for k, v in observation.items() if k != "sha256"}
    observation["sha256"] = hashlib.sha256(__import__("json").dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest()
    session.set_decision_observation(observation)
    session.perception_backend = "rgbd_geometry"
    session.visual_calibration = observation["calibration"]
    return observation


def refresh_visual_observation(session, parts=ALL_PARTS, size=DETECTOR_SIZE):
    """Re-render and re-detect at an execution boundary.

    This function owns rendering only; the detector itself remains a pure
    image/CAD function.  Missing or invalid detections are preserved as such.
    """
    return install_visual(session, parts=parts, size=size)


def save_vision(path, arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(path, **arrays)
    h = hashlib.sha256()
    for key in sorted(arrays):
        v = np.ascontiguousarray(arrays[key]); h.update(key.encode()); h.update(str(v.dtype).encode()); h.update(str(v.shape).encode()); h.update(v.tobytes())
    return dict(schema="twingraph.rgbd.v7", file=path.name, sha256=h.hexdigest(), cameras=list(VIEW_NAMES), image_size=IMAGE_SIZE,
                channels="uint8 RGB plus uint16 metric depth in millimetres", decision_boundary="pre-execution")


def _full_plan(session, targets, order, choices, wipe_variant, wipe_force, wipe_duration, stroke_minimum):
    if float(stroke_minimum) != TASK_STROKE_MINIMUM_M:
        raise ValueError("candidate cannot change task stroke requirement")
    params = dict(order=list(order), choices=plain(choices), wipe_variant=int(wipe_variant), wipe_force=float(wipe_force),
                  wipe_duration=float(wipe_duration), stroke_minimum=float(stroke_minimum))
    task_scope = getattr(session, "task_version", TASK_SCOPE)
    cid = digest(dict(scope=task_scope, params=params, start=fingerprint(session)))[:20]
    call = Call("full_task", "run_full_task_v7", {k: argument(v) for k, v in params.items()}, {"manipulated": WIPE_TOOL}, "executable")
    payload = dict(id=cid, part=WIPE_TOOL, execution="full_task_v7", start_state=fingerprint(session), steps=[dict(skill=call.skill, params=params)],
                   task_scope=task_scope, order=list(order), choices=plain(choices), wipe_variant=int(wipe_variant),
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
                  int(rng.integers(4)), float(rng.choice([1.3, 1.5, 1.7])), float(rng.choice([12., 14., 16.])), TASK_STROKE_MINIMUM_M)
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
