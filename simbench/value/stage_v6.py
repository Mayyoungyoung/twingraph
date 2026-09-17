"""Domain-randomized full sliding-stage scenes and pre-execution RGB-D views.

The value decision is made before physical execution.  Consequently the visual
input contains only two cameras that are available at that decision boundary:
the existing global oblique camera and a fixed overhead camera.  A wrist close
up would require an active sensing motion and is deliberately not fabricated.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
import os
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from simbench.assembly.control import HOME
from simbench.assembly.library import Session
from simbench.assembly.scene import SCENE, fmt
from simbench.core.sim_context import MjContext
from .plan import digest
from . import stage_v5 as base


FAMILY = "sliding_stage_full_v6"
PARTS = base.PARTS
TASK_SCOPE = base.TASK_SCOPE
IMAGE_SIZE = 80
VIEW_NAMES = ("task_view", "top_view")


@dataclass
class StageV6Spec:
    seed: int
    supply_shifts: dict
    supply_yaws_rad: dict
    light_scale: float
    color_scale: float
    camera_xy_jitter_m: list
    family: str = FAMILY
    scope: str = TASK_SCOPE
    checkpoint: int = 0

    @classmethod
    def sample(cls, seed):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 806]))
        return cls(
            int(seed),
            {p: rng.uniform(-.012, .012, 2).tolist() for p in PARTS},
            {p: float(rng.uniform(-math.radians(7), math.radians(7))) for p in PARTS},
            float(rng.uniform(.78, 1.22)),
            float(rng.uniform(.86, 1.14)),
            rng.uniform(-.006, .006, 2).tolist(),
        )

    @property
    def config_id(self):
        row = asdict(self)
        row.pop("seed")
        return digest(row)[:20]


def _scale_scene_appearance(root, spec):
    for light in root.findall(".//light"):
        if "diffuse" in light.attrib:
            value = np.fromstring(light.attrib["diffuse"], sep=" ")
            light.set("diffuse", fmt(np.clip(value * spec.light_scale, 0, 1)))
    for geom in root.findall(".//geom"):
        if "rgba" in geom.attrib:
            value = np.fromstring(geom.attrib["rgba"], sep=" ")
            value[:3] = np.clip(value[:3] * spec.color_scale, 0, 1)
            geom.set("rgba", fmt(value))


def write_scene(spec, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    root = ET.parse(SCENE).getroot()
    root.set("model", FAMILY)
    world = root.find("worldbody")
    if set(spec.supply_shifts) != set(PARTS) or set(spec.supply_yaws_rad) != set(PARTS):
        raise ValueError("v6 layout must contain all five parts")
    for part in PARTS:
        delta = np.asarray(spec.supply_shifts[part], float)
        yaw = float(spec.supply_yaws_rad[part])
        if (delta.shape != (2,) or not np.isfinite(delta).all()
                or np.max(np.abs(delta)) > .012000001 or abs(yaw) > math.radians(7.0001)):
            raise ValueError("v6 supply randomization exceeds its declared envelope")
        names = (part, part + "_holder") if part.startswith("pin_") else (part,)
        for name in names:
            body = world.find(f"body[@name='{name}']")
            if body is None:
                raise ValueError(f"original scene lacks {name}")
            xyz = np.fromstring(body.get("pos", "0 0 0"), sep=" ")
            xyz[:2] += delta
            body.set("pos", fmt(xyz))
            body.set("quat", fmt([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]))
    _scale_scene_appearance(root, spec)

    # The overhead view covers both the supply area and the assembly nest.  It
    # resolves lateral clearances that are foreshortened in the global view.
    jitter = np.asarray(spec.camera_xy_jitter_m, float)
    ET.SubElement(world, "camera", name="top_view",
                  pos=fmt(np.r_[[-.15, -.08] + jitter, 1.72]),
                  xyaxes="1 0 0 0 1 0", fovy="42")

    assets = SCENE.parent.parent / "assets" / "panda"
    try:
        os.path.relpath(assets, directory.resolve())
    except ValueError:
        shutil.copytree(assets, directory / "panda_assets", dirs_exist_ok=True)
        assets = directory.resolve() / "panda_assets"
    asset_path = lambda path: Path(os.path.relpath(path, directory.resolve())).as_posix()
    root.find("compiler").set("meshdir", asset_path(assets))
    root.find("include").set("file", asset_path(assets / "panda.xml"))
    path = directory / "stage_scene.xml"
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="unicode")
    return path


def make_scene(seed, directory, role="collection"):
    if role not in {"collection", "twin", "target", "development"}:
        raise ValueError("unknown scene role")
    spec = StageV6Spec.sample(seed)
    path = write_scene(spec, directory)
    ctx = MjContext(path, control_freq=50)
    ctx.reset()
    ctx.data.qpos[ctx.arm_qadr] = HOME
    mujoco.mj_forward(ctx.model, ctx.data)
    ctx.hold_arm()
    ctx.set_finger_ctrl(.04)
    for _ in range(80):
        ctx.step()
    session = Session(ctx, seed=seed, noise=0.)
    targets = base.nominal_targets()
    session.stage_targets = targets
    session.scene_role = role
    session.stage_completed = ()
    session.value_checkpoint_trace = []
    session.value_checkpoint = dict(kind="initial_unassembled_scene", completed_parts=[],
        assembly_part_count=0, simulation_time_s=float(ctx.data.time),
        source="domain_randomized_original_supply_layout", role=role)
    session.supplier_snapshot = {p: dict(position=ctx.obj_pos(p).tolist(),
        quaternion=ctx.obj_pose(p)[1].tolist()) for p in PARTS}
    return spec, session, path, targets


def observed(session, targets):
    value = base.observed(session, targets)
    value["sensor_model"] = dict(
        decision_boundary="before execution and before active wrist inspection",
        available_views=list(VIEW_NAMES),
        structured_pose_source="detector estimate contract, not target rollout state",
        training_position_noise_std_m=[.00025, .00125],
    )
    return value


def capture_vision(session, size=IMAGE_SIZE):
    """Capture paired RGB and metric depth without changing simulation state."""
    renderer = mujoco.Renderer(session.ctx.model, height=size, width=size)
    result = {}
    try:
        for name in VIEW_NAMES:
            renderer.disable_depth_rendering()
            renderer.update_scene(session.ctx.data, camera=name)
            rgb = renderer.render().copy().astype(np.uint8)
            renderer.enable_depth_rendering()
            renderer.update_scene(session.ctx.data, camera=name)
            depth = renderer.render().copy()
            result[f"{name}_rgb"] = rgb
            result[f"{name}_depth_mm"] = np.clip(np.rint(depth * 1000), 0, 65535).astype(np.uint16)
    finally:
        close = getattr(renderer, "close", None) or getattr(renderer, "free", None)
        if close:
            close()
    return result


def save_vision(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    import hashlib
    return dict(schema="twingraph.rgbd.v6", file=path.name,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                cameras=list(VIEW_NAMES), image_size=IMAGE_SIZE,
                channels="uint8 RGB plus uint16 metric depth in millimetres",
                decision_boundary="pre-execution")


# Candidate generation remains exactly the audited executable-program generator;
# v6 changes the scene distribution and supervision, not what gets executed.
build_pool = base.build_pool
rebind_plan = base.rebind_plan
legal_orders = base.legal_orders
CandidateGenerationError = base.CandidateGenerationError


def trial_for(config, repeat, role):
    from .collect_v6 import trial_spec
    return trial_spec(config.seed, repeat, role,
                      friction_span=config.friction_span,
                      gain_span=config.gain_span)
