"""Parameterized, physical pin assembly decisions using the existing Panda skills.

This is a sliding-stage pin subtask family, not a claim of three industrial
task families. Fixture channels, supply layouts, insertion clearances and
allowed grasp regions vary. No body is repositioned during a rollout.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import copy
import math
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from simbench.assembly.demo_scenes import build_demo_scene
from simbench.assembly.scene import geom, fmt
from simbench.assembly.control import HOME, down
from simbench.assembly.library import Session
from simbench.assembly.candidates import build_pick_candidates
from simbench.core.sim_context import MjContext
from .plan import Argument, Call, argument, from_pick, plain, digest


@dataclass
class TaskSpec:
    seed: int
    source_xy: list
    target_xy: list
    channel_yaw: float
    channel_half_gap: float
    wall_top: float
    hole_half: float
    settle: float = 0.5
    retreat: float = 0.10
    family: str = "sliding_stage_pin"

    @classmethod
    def sample(cls, seed):
        rng = np.random.default_rng(seed)
        return cls(
            seed,
            rng.uniform([-0.25, -0.26], [-0.16, -0.15]).tolist(),
            rng.uniform([-0.015, 0.005], [0.075, 0.105]).tolist(),
            float(rng.choice([0.0, math.pi / 2])),
            float(rng.uniform(0.025, 0.055)),
            float(rng.uniform(0.875, 0.920)),
            float(rng.uniform(0.0038, 0.0048)),
            retreat=float(rng.choice([0.08, 0.10, 0.12])),
        )

    @property
    def target(self):
        return np.array([*self.target_xy, 0.855])

    @property
    def config_id(self):
        values = asdict(self)
        values.pop("seed")
        return digest(values)[:20]


def make_task(spec, directory):
    path, parts, grasps = build_demo_scene("pin", directory)
    root = ET.parse(path).getroot()
    world = root.find("worldbody")
    world.find("body[@name='pin_left']").set("pos", fmt([*spec.source_xy, 0.850]))
    world.find("body[@name='pin_left_holder']").set(
        "pos", fmt([*spec.source_xy, 0.819])
    )
    receiver = world.find("body[@name='receiver']")
    receiver.set("pos", fmt([*spec.target_xy, 0.827]))
    from simbench.assembly.scene import holed_plate

    for child in list(receiver):
        receiver.remove(child)
    holed_plate(
        receiver,
        "receiver",
        (0.035, 0.035),
        0,
        0.027,
        [(0, 0)],
        ".35 .48 .6 1",
        holehalf=spec.hole_half,
    )
    # A physical access channel around the mating site. Both closed grasp
    # orientations fit; opening and retreat can require different free space.
    for sign in (-1, 1):
        offset = (
            np.array([-math.sin(spec.channel_yaw), math.cos(spec.channel_yaw)])
            * sign
            * (spec.channel_half_gap + 0.006)
        )
        geom(
            receiver,
            f"channel_wall_{sign}",
            [*offset, (spec.wall_top + 0.854) / 2 - 0.827],
            [0.055, 0.006, (spec.wall_top - 0.854) / 2],
            ".45 .48 .52 1",
            quat=fmt(
                [math.cos(spec.channel_yaw / 2), 0, 0, math.sin(spec.channel_yaw / 2)]
            ),
        )
    ET.ElementTree(root).write(path, encoding="unicode")
    ctx = MjContext(path, control_freq=50)
    ctx.reset()
    ctx.data.qpos[ctx.arm_qadr] = HOME
    mujoco.mj_forward(ctx.model, ctx.data)
    ctx.hold_arm()
    ctx.set_finger_ctrl(0.04)
    for _ in range(80):
        ctx.step()
    return Session(ctx, seed=spec.seed, parts=parts, grasp_specs=grasps), path


def build_plans(session, spec):
    s, part = session, "pin_left"
    s.call("detect")
    s.call("estimate_pose", part=part)
    s.call("estimate_grasp", part=part)
    s.call("gripper", mode="open")
    # Two physically meaningful vertical regions of the same head, both
    # computed before any trial. No candidate ID or cost becomes a feature.
    originals = s.artifacts["grasps"]["candidates"]
    grasps = []
    for original in originals:
        for offset in (-0.002, 0.003):
            g = copy.deepcopy(original)
            g["xyz"] = g["xyz"] + [0, 0, offset]
            g["id"] += f":height:{offset}"
            g["q_hover"] = s.arm.ik(g["xyz"] + [0, 0, 0.1], down(g["yaw"]))
            grasps.append(g)
    s.artifacts["grasps"]["candidates"] = grasps
    candidates = build_pick_candidates(
        s, part, terminal_targets=[dict(id="seated", part=part, xyz=spec.target)]
    )
    # Retain two distinct approach heights per grasp: 8 candidates, without
    # replicating identical plans just to inflate the screening budget.
    counts, chosen = {}, []
    for c in candidates:
        key = c.grasp["id"]
        if c.status == "conflict" or counts.get(key, 0) >= 2:
            continue
        counts[key] = counts.get(key, 0) + 1
        chosen.append(c)
    plans = []
    for c in chosen:
        producer = f"prefix_{len(c.steps)-1}"

        def call(cid, skill, **params):
            return Call(
                cid,
                skill,
                {
                    k: v if isinstance(v, Argument) else argument(v)
                    for k, v in params.items()
                },
                {"manipulated": part},
            )

        def position(value):
            return Argument(plain(value), "position", frame="world", unit="m")

        suffix = [
            call(
                "transport_plan",
                "plan_path",
                target=Argument(
                    plain(spec.target + [0, 0, 0.069]),
                    "position",
                    "deferred",
                    "world",
                    "m",
                    producer,
                    "object_to_eef",
                ),
                yaw=Argument(
                    None,
                    "scalar",
                    "deferred",
                    unit="rad",
                    source_call=producer,
                    source_output="grasp_yaw",
                ),
            ),
            call("transport", "move", path="transfer"),
            call(
                "align",
                "move",
                reference="object",
                part=part,
                target=position(spec.target + [0, 0, 0.069]),
            ),
            call(
                "insert",
                "move",
                mode="guarded",
                part=part,
                target_z=argument(float(spec.target[2]), unit="m"),
                force_stop=argument(3.0, unit="N"),
            ),
            call(
                "seat",
                "press",
                part=part,
                target_z=argument(float(spec.target[2]), unit="m"),
            ),
            call(
                "release",
                "place",
                part=part,
                target=position(spec.target),
                tol=argument(0.0015, unit="m"),
                settle=argument(spec.settle, unit="s"),
            ),
            call(
                "retreat",
                "move",
                delta=Argument(
                    [0, 0, spec.retreat], "position", frame="world", unit="m"
                ),
            ),
            call(
                "accept",
                "inspect",
                part=part,
                target=position(spec.target),
                tol=argument(0.0015, unit="m"),
            ),
        ]
        plans.append(from_pick(c, suffix))
    return plans


def observation(session, spec):
    ctx = session.ctx
    objects = {}
    for name in ("pin_left", "receiver"):
        bid = ctx.body_id(name)
        indices = np.where(ctx.model.geom_bodyid == bid)[0]
        objects[name] = dict(
            position=ctx.obj_pos(name).tolist(),
            quaternion=ctx.data.xquat[bid].tolist(),
            geoms=[
                dict(
                    type=int(ctx.model.geom_type[i]),
                    size=ctx.model.geom_size[i].tolist(),
                    position=ctx.model.geom_pos[i].tolist(),
                    quaternion=ctx.model.geom_quat[i].tolist(),
                )
                for i in indices
            ],
        )
    return dict(
        robot=dict(
            joints=ctx.arm_qpos.tolist(),
            fingers=ctx.finger_qpos.tolist(),
            eef=ctx.eef_pos().tolist(),
        ),
        objects=objects,
        goals=[
            dict(
                predicate="seated_released_retracted",
                manipulated="pin_left",
                receiver="receiver",
                position=spec.target.tolist(),
                position_tolerance=0.0015,
                tilt_tolerance_deg=3.0,
                retreat=spec.retreat,
            )
        ],
        perception="simulator_privileged_pose_and_segmentation",
    )


def render_observation(session, directory, names=None):
    from PIL import Image

    out = Path(directory)
    renderer = mujoco.Renderer(session.ctx.model, height=256, width=320)
    try:
        renderer.update_scene(session.ctx.data, camera=0)
        rgb = renderer.render().copy()
        Image.fromarray(rgb).save(out / "scene.png")
        renderer.enable_segmentation_rendering()
        renderer.update_scene(session.ctx.data, camera=0)
        seg = renderer.render().copy()
    finally:
        if hasattr(renderer, "close"):
            renderer.close()
        else:
            # MuJoCo 2.3.2 predates Renderer.close/context-manager support.
            renderer._mjr_context.free()
            renderer._gl_context.free()
    images = ["scene.png"]
    for name in (("pin_left", "receiver") if names is None else names):
        ids = np.where(session.ctx.model.geom_bodyid == session.ctx.body_id(name))[0]
        mask = np.isin(seg[:, :, 0], ids) & (
            seg[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
        )
        ys, xs = np.where(mask)
        if len(xs):
            x0, x1 = max(0, int(xs.min()) - 14), min(rgb.shape[1], int(xs.max()) + 15)
            y0, y1 = max(0, int(ys.min()) - 14), min(rgb.shape[0], int(ys.max()) + 15)
            Image.fromarray(rgb[y0:y1, x0:x1]).save(out / (name + ".png"))
            images.append(name + ".png")
    return images
