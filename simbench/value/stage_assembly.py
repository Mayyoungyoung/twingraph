"""Original sliding-stage continuation from a robot-created assembly checkpoint.

The benchmark begins after the existing robot recipe has installed the carriage
and end stop. A candidate installs both locking pins and the handle. The two
pins may exchange order; the handle remains last, as in the original recipe.
This is a three-part assembly continuation, not a full assembly/stroke test.

Only the *initial* supply layout is varied. No installed body is positioned by
state assignment, and no candidate is generated using rollout outcomes.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import itertools
import math
import os
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from simbench.assembly.candidates import transfer_routes
from simbench.assembly.control import HOME
from simbench.assembly.library import Session, SkillFailure
from simbench.assembly.scene import CENTER, SCENE, fmt
from simbench.assembly.task import CAR_Z, PIN_L, PIN_R, STOP, pick, release, transfer_part
from simbench.core.sim_context import MjContext
from .plan import JOINT_PATH_FIELDS, digest, materialize_initial_path, plain
from .research_scenarios import observed as observed_program
from .research_scenarios import program as generic_program
from .research_scenarios import semantic_key as generic_semantic_key

FAMILY = "sliding_stage_assembly"
CHECKPOINT_PARTS = ("carriage", "end_stop")
CONTINUATION_PARTS = ("pin_left", "pin_right", "handle")
# These are process constraints, not scoring weights. The stop closes the
# carriage entry corridor; the pins engage holes in that already seated stop.
PRECEDENCE = (("carriage", "end_stop"), ("end_stop", "pin_left"),
              ("end_stop", "pin_right"), ("pin_left", "handle"),
              ("pin_right", "handle"))


@dataclass
class StageAssemblySpec:
    seed: int
    supply_shifts: dict
    family: str = FAMILY
    checkpoint_parts: tuple = CHECKPOINT_PARTS
    checkpoint: int = 2
    scope: str = "dual_pin_and_handle_continuation"

    @classmethod
    def sample(cls, seed):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 421]))
        # Passive pin holders move with their supplied pin; the product,
        # assembly nest, robot and camera retain the original scene geometry.
        shifts = {part: rng.uniform(-.012, .012, 2).tolist()
                  for part in CONTINUATION_PARTS}
        return cls(int(seed), shifts)

    @property
    def config_id(self):
        row = asdict(self)
        row.pop("seed")
        return digest(row)[:20]


def write_scene(spec, directory):
    """Copy the original scene and change supply-body initial poses only."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    root = ET.parse(SCENE).getroot()
    root.set("model", FAMILY)
    world = root.find("worldbody")
    if set(spec.supply_shifts) != set(CONTINUATION_PARTS):
        raise ValueError("supply layout must describe both pins and the handle")
    for part, shift in spec.supply_shifts.items():
        delta = np.asarray(shift, dtype=float)
        if delta.shape != (2,) or not np.isfinite(delta).all() or np.max(np.abs(delta)) > .025:
            raise ValueError("supply displacement is outside the declared layout range")
        names = (part, part + "_holder") if part.startswith("pin_") else (part,)
        for name in names:
            body = world.find(f"body[@name='{name}']")
            if body is None:
                raise ValueError(f"original sliding-stage scene lacks {name}")
            position = np.fromstring(body.get("pos", "0 0 0"), sep=" ")
            position[:2] += delta
            body.set("pos", fmt(position))
    assets = SCENE.parent.parent / "assets" / "panda"
    def asset_path(path):
        try:
            return Path(os.path.relpath(path, directory.resolve())).as_posix()
        except ValueError:  # Windows temporary directory may be on another drive.
            return path.resolve().as_posix()
    root.find("compiler").set("meshdir", asset_path(assets))
    root.find("include").set("file", asset_path(assets / "panda.xml"))
    path = directory / "stage_scene.xml"
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="unicode")
    return path


def prepare_checkpoint(session):
    """Execute the first two parts of task.assemble through robot atoms."""
    s = session
    if s.held is not None:
        raise ValueError("checkpoint preparation requires an empty gripper")
    start = len(s.results)
    carriage_target = np.r_[CENTER + [.030, 0], CAR_Z]
    pick(s, "carriage", terminal_targets=[dict(id="seated", part="carriage", xyz=carriage_target)])
    entry = np.r_[CENTER + [-.155, 0], CAR_Z + .030]
    transfer_part(s, "carriage", entry)
    s.call("move", part="carriage", delta=[0, 0, -.030])
    s.call("move", reference="object", part="carriage", target=np.r_[entry[:2], CAR_Z + .0015])
    s.call("plan_path", method="contact", part="carriage", target=carriage_target + [0, 0, .0015],
           axis=(1, 0, 0), speed=.025, force_limit=18.)
    s.call("insert", part="carriage")
    s.call("press", part="carriage", target_z=CAR_Z)
    release(s)
    s.call("inspect", part="carriage", target=carriage_target)
    s.call("measure", quantity="clearance", part="carriage")
    s.call("inspect", what="measurement", minimum=1.e-12)

    pick(s, "end_stop", terminal_targets=[dict(id="seated", part="end_stop", xyz=STOP)])
    transfer_part(s, "end_stop", STOP + [0, 0, .045])
    s.call("plan_path", method="cartesian", target=s.arm.part_target("end_stop", STOP + [0, 0, .010]))
    s.call("move", path="linear", space="cartesian")
    s.call("move", reference="object", part="end_stop", target=STOP + [0, 0, .010])
    s.call("move", mode="guarded", part="end_stop", target_z=float(STOP[2]))
    s.call("press", part="end_stop", target_z=float(STOP[2]))
    release(s)
    s.call("inspect", part="end_stop", target=STOP)
    s.stage_completed = CHECKPOINT_PARTS
    s.stage_checkpoint_trace = plain(s.results[start:])
    s.value_checkpoint_trace = s.stage_checkpoint_trace
    s.value_checkpoint = dict(kind="robot_executed_prefix", completed_parts=list(CHECKPOINT_PARTS),
                              assembly_part_count=2, simulation_time_s=float(s.ctx.data.time),
                              source="simbench.assembly.task.assemble:first_two_parts")
    return s.stage_checkpoint_trace


def make_family(family, seed, directory):
    """Return ``spec, session, scene_path, targets`` after real warmup.

    Warmup failures raise and must be recorded by the collector; they are never
    replaced by a hand-positioned checkpoint or silently resampled layout.
    """
    if family != FAMILY:
        raise ValueError(family)
    spec = StageAssemblySpec.sample(seed)
    path = write_scene(spec, directory)
    ctx = MjContext(path, control_freq=50)
    ctx.reset()
    # Initial robot configuration, before any task execution; free-body state
    # is left at the scene initial condition and subsequently evolves in MuJoCo.
    ctx.data.qpos[ctx.arm_qadr] = HOME
    mujoco.mj_forward(ctx.model, ctx.data)
    ctx.hold_arm()
    ctx.set_finger_ctrl(.04)
    for _ in range(80):
        ctx.step()
    session = Session(ctx, seed=seed)
    prepare_checkpoint(session)
    targets = dict(carriage=np.r_[CENTER + [.030, 0], CAR_Z].tolist(),
                   end_stop=STOP.tolist(), pin_left=PIN_L.tolist(), pin_right=PIN_R.tolist(),
                   handle=(ctx.obj_pos("carriage") + [0, 0, .048]).tolist())
    session.stage_targets = targets
    check_preconditions(session, targets, CHECKPOINT_PARTS)
    return spec, session, path, targets


def check_preconditions(session, targets, completed):
    """Read-only checks of a real checkpoint; these are not future labels."""
    completed = tuple(completed)
    if not set(CHECKPOINT_PARTS).issubset(completed):
        raise ValueError("continuation requires the robot-installed carriage and end stop")
    if len(completed) != len(set(completed)) or set(completed) - set(targets):
        raise ValueError("invalid completed-part set")
    for before, after in PRECEDENCE:
        if after in completed and before not in completed:
            raise ValueError(f"assembly precondition violated: {before} before {after}")
    if session.held is not None:
        raise ValueError("candidate pool requires a released checkpoint")
    for part in completed:
        result = session.inspect_seat(part, targets[part], tol=.0015)
        if not result.ok:
            raise SkillFailure(f"checkpoint part {part} is not assembled: {result.metrics}")


def legal_orders(remaining, completed=CHECKPOINT_PARTS):
    remaining, completed = tuple(remaining), set(completed)
    if len(set(remaining)) != len(remaining) or set(remaining) & completed:
        raise ValueError("remaining parts must be unique and unassembled")
    if set(remaining) - set(CONTINUATION_PARTS):
        raise ValueError("this family supports pin/handle continuation only")
    result = []
    for order in itertools.permutations(remaining):
        done = set(completed)
        for part in order:
            if any(after == part and before not in done for before, after in PRECEDENCE):
                break
            done.add(part)
        else:
            result.append(order)
    return result


def program(session, targets, order, choices, completed=CHECKPOINT_PARTS):
    """Same executable PlanIR protocol as the graph compiler and runner."""
    if not order:
        raise ValueError("candidate requires an unassembled component")
    if tuple(order) not in legal_orders(order, completed):
        raise ValueError("candidate violates sliding-stage assembly precedence")
    if set(order) != set(CONTINUATION_PARTS) - set(completed):
        raise ValueError("candidate must assemble every remaining component")
    plan = generic_program(session, targets, order, choices)
    plan.prefix["task_precedence"] = plain(PRECEDENCE)
    plan.prefix["completed_parts"] = list(completed)
    plan.prefix["task_scope"] = "dual_pin_and_handle_continuation"
    return plan.validate(session.parts)


def observed(session, targets):
    """Include the original guide/rail and passive holders in the observation."""
    result = observed_program(session, targets)
    ctx, model = session.ctx, session.ctx.model
    for name in ("guide_base", "pin_left_holder", "pin_right_holder"):
        bid = ctx.body_id(name)
        indices = np.where(model.geom_bodyid == bid)[0]
        result["objects"][name] = dict(
            position=ctx.obj_pos(name).tolist(), quaternion=ctx.data.xquat[bid].tolist(),
            geoms=[dict(type=int(model.geom_type[i]), size=model.geom_size[i].tolist(),
                        position=model.geom_pos[i].tolist(), quaternion=model.geom_quat[i].tolist())
                   for i in indices])
    # Generic physical metadata; no product-specific clearance summary or score.
    for name, obj in result["objects"].items():
        bid = ctx.body_id(name)
        obj["mass_kg"] = float(model.body_mass[bid])
        obj["inertia_kg_m2"] = model.body_inertia[bid].tolist()
        indices = np.where(model.geom_bodyid == bid)[0]
        for geom, index in zip(obj["geoms"], indices):
            geom["friction"] = model.geom_friction[index].tolist()
    return result


def semantic_key(plan):
    """ID-independent program identity, including every executable waypoint."""
    paths = {name: {key: item[key] for key in JOINT_PATH_FIELDS}
             for name, item in plan.prefix.get("initial_artifacts", {}).items()}
    return digest(dict(program=generic_semantic_key(plan), paths=paths))


def build_pool(session, targets, seed, n=16, completed=(), precheck=True):
    """Deterministic label-blind pool, with nested N and distinct programs.

    Every variation changes executable geometry or controller parameters.
    Necessary IK/path checking touches only the current approach. Unknown
    future contact outcomes remain candidates and are learned from physics.
    """
    if n < 1:
        raise ValueError("candidate count must be positive")
    started = time.perf_counter()
    completed = tuple(completed or getattr(session, "stage_completed", CHECKPOINT_PARTS))
    check_preconditions(session, targets, completed)
    remaining = [p for p in CONTINUATION_PARTS if p not in completed]
    orders = legal_orders(remaining, completed)
    if not remaining or not orders:
        raise ValueError("no unassembled continuation remains")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 593]))
    # Each part receives its own grasp choice; joint route/force/speed branches
    # avoid the meaningless large grids produced by sub-tolerance jitter.
    grid = list(itertools.product(range(len(orders)), *([range(8)] * len(remaining)),
                                  range(2), range(2), range(2), range(3) if precheck else range(1)))
    rng.shuffle(grid)
    result, seen, cache, witnesses = [], set(), {}, []
    raw, conflicts, materialization_unknown, check_seconds = 0, 0, 0, 0.
    for row in grid:
        raw += 1
        oi, *options = row
        grasps, route, force, speed, route_index = options[:-4], *options[-4:]
        order = orders[oi]
        choices = {}
        for part, grasp in zip(remaining, grasps):
            choices[part] = dict(yaw=float((grasp // 4) * math.pi / 2),
                                 height=[-.002, 0., .002, .004][grasp % 4],
                                 clearance=.98 + .055 * route,
                                 force=2.5 + force, speed=.006 + .002 * speed)
        plan = program(session, targets, order, choices, completed)
        first = order[0]
        choice = choices[first]
        check_key = (first, choice["yaw"], choice["height"], choice["clearance"])
        if precheck and check_key not in cache:
            t = time.perf_counter()
            grasp = session.ctx.obj_pos(first) + [0, 0, session.grasp_specs[first][0] + choice["height"]]
            routes, details = transfer_routes(session, grasp + [0, 0, .10],
                                               clearance=choice["clearance"], yaw=choice["yaw"])
            cache[check_key] = routes
            witnesses.extend(details)
            check_seconds += time.perf_counter() - t
        if precheck:
            selected_route = cache[check_key][route_index]
            if selected_route["status"] != "necessary_pass":
                conflicts += int(selected_route["status"] == "conflict")
                materialization_unknown += int(selected_route["status"] != "conflict")
                continue
            plan = materialize_initial_path(session, plan, selected_route["path"])
        key = semantic_key(plan)
        if key in seen:
            continue
        seen.add(key)
        result.append(plan)
        if len(result) == n:
            break
    counts = dict(requested=n, raw_count=raw, known_conflict=conflicts, unresolved=len(result),
                  materialization_unknown=materialization_unknown,
                  deduplicated=len(seen), materializable=len(result), pool_exhausted=len(result) < n,
                  grasp_branches=len({(p.prefix["part"], p.prefix["choices"][p.prefix["part"]]["yaw"],
                                       p.prefix["choices"][p.prefix["part"]]["height"]) for p in result}),
                  structure_branches=len({tuple(p.prefix["order"]) for p in result}),
                  initial_trajectory_branches=len({digest({name: {key: item[key] for key in JOINT_PATH_FIELDS}
                                                           for name, item in p.prefix.get("initial_artifacts", {}).items()})
                                                    for p in result}) if precheck else 0,
                  necessary_geometry_seconds=check_seconds,
                  candidate_generation_seconds=time.perf_counter() - started - check_seconds,
                  optional_geometry_seconds=0., known_conflict_witnesses=witnesses,
                  completed_parts=list(completed), continuation_parts=remaining,
                  scope="dual_pin_and_handle_continuation", label_blind=True)
    return result, counts
