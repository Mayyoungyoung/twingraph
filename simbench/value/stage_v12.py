"""Scene built from the user's nominal 3D-print kit, without V9 widenings."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

import mujoco

from simbench.assembly.control import HOME
from simbench.assembly.library import Session, GRASP, DEFAULT_CAPABILITIES
from simbench.assembly.printed_kit import apply_printed_kit, planning_metadata, source_manifest
from simbench.core.sim_context import MjContext
from . import stage_v7
from .pin_geometry import PinInsertionConfig

TASK_VERSION = "printed_functional_assembly_v12"
INITIAL_DROP_GAP_M = .001
# Local stop-bore acceptance used by isolated legacy skills. The complete
# task additionally requires the observed two-layer CAD bridge and base check.
PIN_CONFIG = PinInsertionConfig(guide_inner_radius_m=.004,
    acceptance_mode="functional_contiguous",
    contact_robustness=True,
    plate_hole_half_width_m=.004, guide_length_m=.036, required_depth_m=.006,
    radial_clearance_m=0., aperture_shape="square",
    source="user printed end-stop STL: 8 mm square through-hole, 36 mm depth")


@dataclass
class StageV12Spec:
    source_spec: dict
    task_version: str = TASK_VERSION
    geometry_source: str = "printed_kit_v0_1/htzp"
    setup: str = "complete_task"


def visual_templates(installed=False):
    from .cad_rgbd_v12 import templates
    return templates(installed=installed)


def write_scene(source_spec, directory, preinstalled_end_stop=False, scene_layout_hook=None):
    directory = Path(directory)
    source = stage_v7.write_scene(source_spec, directory)
    root = ET.parse(source).getroot()
    root.set("model", TASK_VERSION)
    apply_printed_kit(root)
    layout_provenance = scene_layout_hook(root, source_spec) if scene_layout_hook is not None else None
    from .wrist_rgbd_v12 import install_camera
    install_camera(root, directory)
    if preinstalled_end_stop:
        from simbench.assembly.scene import fmt
        base = root.find(".//body[@name='guide_base']")
        base_position = np.fromstring(base.get("pos", "0 0 0"), sep=" ")
        base_quat = np.fromstring(base.get("quat", "1 0 0 0"), sep=" ")
        rotation = np.empty(9); mujoco.mju_quat2Mat(rotation, base_quat)
        mate = planning_metadata()["end_stop_mating_pose_in_base"]
        target = base_position + rotation.reshape(3,3) @ np.asarray(mate["position_m"])
        stop = root.find(".//body[@name='end_stop']")
        stop.set("pos", fmt(target)); stop.set("quat", fmt(base_quat))
    # Real gravity settling from a finite gap avoids an exactly tangent
    # initial contact manifold. This is uniform scene initialization, never
    # a task pose correction, hole change, success tolerance or seed filter.
    from simbench.assembly.scene import fmt
    dropped = []
    for body in root.findall(".//worldbody/body"):
        if body.find("freejoint") is None and body.find("joint[@type='free']") is None:
            continue
        position = np.fromstring(body.get("pos", "0 0 0"), sep=" ")
        position[2] += INITIAL_DROP_GAP_M
        body.set("pos", fmt(position)); dropped.append(body.get("name"))
    path = directory / "stage_v12_scene.xml"
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="unicode")
    manifest = source_manifest()
    manifest.update(task_version=TASK_VERSION, setup="preinstalled_end_stop" if preinstalled_end_stop else "complete_task",
                    pin_acceptance=PIN_CONFIG.manifest(),
                    initialization=dict(method="uniform finite free-body gap followed by gravity settling",
                        gap_m=INITIAL_DROP_GAP_M, free_bodies=dropped, control_steps=80,
                        outcome_filtering=False, xy_yaw_layout_unchanged=True))
    if layout_provenance is not None:
        manifest["scene_layout"] = layout_provenance
    (directory / "geometry_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def bind_visual_receiver_targets(session):
    """Bind desired CAD mates to a measured receiver, never nominal world XYZ."""
    observation=session.decision_observation or {}
    base=observation.get("fixtures",{}).get("guide_base",{})
    targets=observation.get("assembly_targets",{})
    if not base.get("valid") or set(targets) != set(stage_v7.PARTS):
        raise ValueError("receiver RGB-D/CAD observation unknown; reobserve before candidate generation")
    session.stage_targets={part:list(row["position_m"]) for part,row in targets.items()}
    session.receiver_geometry=observation["receiver_geometry"]
    session.assembly_target_bindings=targets
    session.fixture_relations=observation["fixture_relations"]
    return session.stage_targets


def make_scene(seed, directory, role="development", level="L1", preinstalled_end_stop=False, scene_layout_hook=None):
    source_spec = stage_v7.StageV7Spec.sample(seed, level)
    spec = StageV12Spec(asdict(source_spec), setup="preinstalled_end_stop" if preinstalled_end_stop else "complete_task")
    path = write_scene(source_spec, directory, preinstalled_end_stop, scene_layout_hook)
    ctx = MjContext(path, control_freq=50)
    ctx.reset(); ctx.data.qpos[ctx.arm_qadr] = HOME
    mujoco.mj_forward(ctx.model, ctx.data); ctx.hold_arm(); ctx.set_finger_ctrl(.04)
    for _ in range(80): ctx.step()
    session = Session(ctx, seed=seed, out=directory, parts=stage_v7.ALL_PARTS,
        grasp_specs={**GRASP, stage_v7.WIPE_TOOL: (.047, .028)},
        capabilities={**DEFAULT_CAPABILITIES, stage_v7.WIPE_TOOL: ("wipe",)})
    cad = planning_metadata()
    # No command target exists before the actual receiver is observed.
    # Final bridge goals come from CAD; selected controller depth must be an
    # explicit PlanIR port, not the old isolated-stop shallow command.
    session.stage_targets = {}; session.scene_role = role; session.stage_completed = ()
    session.level = level; session.v7_spec = asdict(source_spec)
    session.task_version = TASK_VERSION; session.pin_insertion_config = PIN_CONFIG
    from .cad_rgbd_v12 import templates, estimate_scene
    session.visual_templates_fn = templates; session.perception_estimator_fn = estimate_scene
    from .wrist_rgbd_v12 import capture_detector, DETECTOR_SIZE
    session.capture_detector_fn = capture_detector
    session.detector_size = DETECTOR_SIZE
    session.planning_cad = cad
    from . import cleaning
    names = [f"dirty_{i}" for i in range(32)]
    points = [ctx.model.site_pos[mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_SITE, name)].copy() for name in names]
    cleaning.initialize(session, points, names, seed=seed)
    stage_v7.install_visual(session, stage_v7.ALL_PARTS)
    targets=bind_visual_receiver_targets(session)
    return spec, session, path, targets
