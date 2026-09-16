"""Complete original sliding-stage programs, generated before any execution.

Only initial supply positions vary. Geometry tests use independent scratch
MjData and nominal predecessor poses; these are proposal constraints, never
rollout labels. All five physical parts start unassembled in every live scene.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import copy
import itertools
import math
import os
import shutil
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from simbench.assembly.candidates import fingerprint, release_space_proxy, transfer_routes
from simbench.assembly.control import HOME, down
from simbench.assembly.library import Session, SkillFailure
from simbench.assembly.scene import CENTER, SCENE, fmt
from simbench.assembly.task import CAR_Z, PIN_L, PIN_R, STOP
from simbench.core.sim_context import MjContext
from .plan import Argument, Call, JOINT_PATH_FIELDS, PlanIR, argument, digest, materialize_initial_path, plain
from .stage_assembly import observed as _observed

FAMILY = "sliding_stage_full_v5"
PARTS = ("carriage", "end_stop", "pin_left", "pin_right", "handle")
TASK_SCOPE = "assemble_five_parts_from_supplies_and_release_no_stroke_test"
PRECEDENCE = (("carriage", "end_stop"), ("end_stop", "pin_left"),
              ("end_stop", "pin_right"), ("pin_left", "handle"), ("pin_right", "handle"))
HEIGHT_OFFSETS = (-.001, 0., .001)


class CandidateGenerationError(RuntimeError):
    """Declared proposal geometry or current-route set cannot fill the pool."""


@dataclass
class StageV5Spec:
    seed: int
    supply_shifts: dict
    family: str = FAMILY
    scope: str = TASK_SCOPE
    checkpoint: int = 0

    @classmethod
    def sample(cls, seed):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 705]))
        return cls(int(seed), {p: rng.uniform(-.006, .006, 2).tolist() for p in PARTS})

    @property
    def config_id(self):
        row = asdict(self)
        row.pop("seed")
        return digest(row)[:20]


def nominal_targets():
    carriage = np.r_[CENTER + [.030, 0], CAR_Z]
    return dict(carriage=carriage.tolist(), end_stop=STOP.tolist(),
                pin_left=PIN_L.tolist(), pin_right=PIN_R.tolist(),
                handle=(carriage + [0, 0, .048]).tolist())


def write_scene(spec, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    root = ET.parse(SCENE).getroot()
    root.set("model", FAMILY)
    world = root.find("worldbody")
    if set(spec.supply_shifts) != set(PARTS):
        raise ValueError("supply layout must include exactly the five assembly parts")
    for part, shift in spec.supply_shifts.items():
        delta = np.asarray(shift, float)
        if delta.shape != (2,) or not np.isfinite(delta).all() or np.max(np.abs(delta)) > .006000001:
            raise ValueError("supply displacement exceeds the declared 6 mm layout range")
        for name in ((part, part + "_holder") if part.startswith("pin_") else (part,)):
            body = world.find(f"body[@name='{name}']")
            if body is None:
                raise ValueError(f"original scene lacks {name}")
            xyz = np.fromstring(body.get("pos", "0 0 0"), sep=" ")
            xyz[:2] += delta
            body.set("pos", fmt(xyz))
    assets = SCENE.parent.parent / "assets" / "panda"
    try:
        os.path.relpath(assets, directory.resolve())
    except ValueError:
        # MuJoCo 2.3 XML includes do not understand a Windows drive-qualified
        # include on another drive. Keep a byte-identical local asset copy.
        shutil.copytree(assets, directory / "panda_assets", dirs_exist_ok=True)
        assets = directory.resolve() / "panda_assets"
    def asset_path(path):
        return Path(os.path.relpath(path, directory.resolve())).as_posix()
    root.find("compiler").set("meshdir", asset_path(assets))
    root.find("include").set("file", asset_path(assets / "panda.xml"))
    path = directory / "stage_scene.xml"
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="unicode")
    return path


def make_scene(seed, directory, role="collection"):
    """Return spec, independent Session, XML path and fixed product targets."""
    if role not in {"collection", "twin", "target", "development"}:
        raise ValueError("unknown scene role")
    spec = StageV5Spec.sample(seed)
    path = write_scene(spec, directory)
    ctx = MjContext(path, control_freq=50)
    ctx.reset()
    # Initial robot condition only. Product bodies subsequently move by physics.
    ctx.data.qpos[ctx.arm_qadr] = HOME
    mujoco.mj_forward(ctx.model, ctx.data)
    ctx.hold_arm()
    ctx.set_finger_ctrl(.04)
    for _ in range(80):
        ctx.step()
    session = Session(ctx, seed=seed)
    targets = nominal_targets()
    session.stage_targets = targets
    session.scene_role = role
    session.stage_completed = ()
    session.value_checkpoint_trace = []
    session.value_checkpoint = dict(kind="initial_unassembled_scene", completed_parts=[],
        assembly_part_count=0, simulation_time_s=float(ctx.data.time),
        source="simbench.assembly.scene:original_supply_layout", role=role)
    session.supplier_snapshot = {p: dict(position=ctx.obj_pos(p).tolist(),
        quaternion=ctx.obj_pose(p)[1].tolist()) for p in PARTS}
    return spec, session, path, targets


def make_family(family, seed, directory):
    if family != FAMILY:
        raise ValueError(family)
    return make_scene(seed, directory)


def observed(session, targets):
    result = _observed(session, targets)
    # All final checks are against measured physical pose and released fingers.
    # Stroke/contact quality beyond this terminal assembly task is not claimed.
    for goal in result["goals"]:
        goal["minimum_eef_clearance_m"] = .02
    return result


def legal_orders(orders=None):
    allowed = []
    for order in itertools.permutations(PARTS):
        rank = {p: i for i, p in enumerate(order)}
        if all(rank[a] < rank[b] for a, b in PRECEDENCE):
            allowed.append(order)
    if orders is None:
        return allowed
    result = []
    for order in orders:
        if tuple(order) not in allowed:
            raise ValueError("proposed order must contain all five parts and obey assembly precedence")
        if tuple(order) not in result:
            result.append(tuple(order))
    if not result:
        raise ValueError("planner supplied no valid complete order")
    return result


def stage_calls(part, target, choice, stage):
    """Original task.py operations, with feedback transforms kept deferred."""
    calls = []
    target = np.asarray(target, float)
    def add(skill, **params):
        cid = f"stage{stage}_{len(calls)}"
        calls.append(Call(cid, skill, {k: v if isinstance(v, Argument) else argument(v)
                     for k, v in params.items()}, {"manipulated": part},
                     "checker" if skill == "inspect" else "executable"))
        return cid
    def xyz(value):
        return Argument(plain(value), "position", frame="world", unit="m")
    def feedback(value, producer, output="object_to_eef"):
        return Argument(plain(value), "position", "deferred", "world", "m", producer, output)
    add("detect")
    add("estimate_pose", part=part)
    add("estimate_grasp", part=part, yaws=argument([choice["yaw"]], unit="rad"),
        height_offset=argument(choice["height"], unit="m"))
    grasp = add("select_grasp", part=part)
    add("plan_path", target=feedback([0, 0, .10], grasp, "grasp_hover"),
        yaw=argument(choice["yaw"], unit="rad"), clearance=argument(choice["clearance"], unit="m"))
    add("move", path="transfer")
    add("move", grasp="grasp", part=part)
    add("grasp", part=part, force=argument(choice["force"], unit="N"))
    add("inspect", what="grasp", part=part)
    held = add("move", part=part, delta=xyz([0, 0, .10]))
    if part == "carriage":
        approach = np.r_[CENTER + [-.155, 0], CAR_Z + .030]
    else:
        approach = target + [0, 0, .069 if part.startswith("pin_") else .045]
    add("plan_path", target=feedback(approach, held),
        yaw=Argument(None, "scalar", "deferred", unit="rad", source_call=held, source_output="grasp_yaw"),
        clearance=argument(choice["clearance"], unit="m"))
    add("move", path="transfer")
    if part == "carriage":
        add("move", part=part, delta=xyz([0, 0, -.030]))
        add("move", reference="object", part=part, target=xyz(np.r_[approach[:2], CAR_Z + .0015]))
        add("plan_path", method="contact", part=part, target=xyz(target + [0, 0, .0015]),
            axis=argument([1., 0., 0.], unit="1"), speed=argument(.025, unit="m/s"),
            force_limit=argument(18., unit="N"))
        add("insert", part=part)
    else:
        align = target + [0, 0, .010 if part == "end_stop" else .023 if part == "handle" else .069]
        if part == "end_stop":
            add("plan_path", method="cartesian", target=feedback(align, held))
            add("move", path="linear", space="cartesian")
        add("move", reference="object", part=part, target=xyz(align))
        add("move", mode="guarded", part=part, target_z=argument(float(target[2]), unit="m"),
            force_stop=argument(2. if part == "end_stop" else 3., unit="N"),
            speed=argument(choice["speed"], unit="m/s"))
    press = dict(part=part, target_z=argument(float(target[2]), unit="m"))
    if part == "handle":
        press["force_stop"] = argument(2., unit="N")
    add("press", **press)
    add("place", part=part, target=xyz(target), tol=argument(.003, unit="m"), settle=argument(.35, unit="s"))
    add("move", delta=xyz([0, 0, .10]))
    add("inspect", part=part, target=xyz(target), tol=argument(.0015, unit="m"))
    if part == "carriage":
        add("measure", quantity="clearance", part=part)
        add("inspect", what="measurement", minimum=argument(1.e-12, unit="m"))
    return calls


def program(session, targets, order, choices, initial_route_index=0):
    if tuple(order) not in legal_orders([order]):
        raise ValueError("invalid full-stage order")
    if set(targets) != set(PARTS) or set(choices) != set(PARTS):
        raise ValueError("complete five-part goals and choices are required")
    if initial_route_index not in (0, 1, 2):
        raise ValueError("invalid initial route index")
    calls = [c for i, p in enumerate(order) for c in stage_calls(p, targets[p], choices[p], i)]
    for part in PARTS:
        calls.append(Call(f"accept_{part}", "inspect", dict(part=argument(part),
            target=Argument(plain(targets[part]), "position", frame="world", unit="m"),
            tol=argument(.0015, unit="m")), {"manipulated": part}, "checker"))
    calls.append(Call("final_home", "move", dict(target=argument("home"))))
    semantics = dict(order=list(order), choices=plain(choices), task_scope=TASK_SCOPE,
                     initial_route_index=int(initial_route_index), targets=plain(targets))
    semantic_id = digest(semantics)
    cid = digest(dict(semantic_program_id=semantic_id, start_state=fingerprint(session)))[:20]
    boundary = 10
    payload = dict(id=cid, part=order[0], execution="program", start_state=fingerprint(session),
        steps=[dict(skill=c.skill, params={k: a.value for k, a in c.arguments.items()}) for c in calls[:boundary]],
        **semantics, semantic_program_id=semantic_id, task_precedence=plain(PRECEDENCE),
        completed_parts=[], cost=0., skill_version="feedback.v2")
    return PlanIR(cid, calls, boundary, payload, "unknown", protocol="assembly.program.feedback.v2").validate(session.parts)


def _scratch(session):
    ctx = copy.copy(session.ctx)
    ctx.data = mujoco.MjData(ctx.model)
    for name in ("qpos", "qvel", "ctrl", "act", "mocap_pos", "mocap_quat"):
        getattr(ctx.data, name)[:] = getattr(session.ctx.data, name)
    ctx.data.time = session.ctx.data.time
    ctx.on_control_step = None
    ctx._state_clients = {}
    mujoco.mj_forward(ctx.model, ctx.data)
    return Session(ctx, seed=0, noise=0., parts=session.parts,
                   grasp_specs=session.grasp_specs, capabilities=session.capabilities)


def _predecessors(part):
    done = set()
    while True:
        expanded = done | {a for a, b in PRECEDENCE if b == part or b in done}
        if expanded == done:
            return done
        done = expanded


def grasp_catalogue(session, targets):
    """Existing estimator plus nominal open-jaw geometry; no outcome access.

    Release rejection means an overlap in the prescribed nominal assembly,
    not a guarantee about future compliant motion. IK unknowns are recorded
    separately. No penalty or engineered clearance vector enters the scorer.
    """
    catalogue, evidence = {}, []
    for part in PARTS:
        catalogue[part] = []
        for height in HEIGHT_OFFSETS:
            preview = _scratch(session)
            preview.observe_parts()
            preview.estimate_pose(part)
            preview.propose_grasps(part, height_offset=height)
            for item in preview.artifacts["grasps"]["unresolved"]:
                evidence.append(dict(part=part, height=height, yaw=float(item["yaw"]),
                                     status="solver_unknown", reason=item["reason"]))
            for grasp in preview.artifacts["grasps"]["candidates"]:
                row = dict(part=part, height=height, yaw=float(grasp["yaw"]),
                           method="existing_grasp_estimator_and_nominal_jaw_geometry")
                approach = _scratch(session)
                approach.ctx.data.qpos[approach.ctx.arm_qadr] = grasp["q_hover"]
                mujoco.mj_forward(approach.ctx.model, approach.ctx.data)
                try:
                    goal_q = approach.arm.ik(grasp["xyz"], down(grasp["yaw"]), seed=grasp["q_hover"])
                except ValueError as exc:
                    if "IK" not in str(exc):
                        raise
                    evidence.append(dict(**row, status="solver_unknown", reason=str(exc)))
                    continue
                check = approach.arm.check_joint_path([goal_q])
                if not check["valid"]:
                    evidence.append(dict(**row, status="supply_approach_collision", check=plain(check)))
                    continue
                release = _scratch(session)
                for predecessor in _predecessors(part):
                    bid = release.ctx.body_id(predecessor)
                    qa = release.ctx.model.jnt_qposadr[release.ctx.model.body_jntadr[bid]]
                    release.ctx.data.qpos[qa:qa + 3] = targets[predecessor]
                    release.ctx.data.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]
                mujoco.mj_forward(release.ctx.model, release.ctx.data)
                proxy = release_space_proxy(release, part, grasp, dict(xyz=targets[part]))
                row["nominal_release"] = plain(proxy)
                if proxy["status"] == "proxy" and proxy["pairs"]:
                    evidence.append(dict(**row, status="nominal_release_collision"))
                    continue
                evidence.append(dict(**row, status="retained", release_uncertain=proxy["status"] != "proxy"))
                catalogue[part].append(dict(yaw=float(grasp["yaw"]), height=float(height)))
        if not catalogue[part]:
            raise CandidateGenerationError(f"no geometrically admissible grasp proposals for {part}; evidence={plain(evidence)}")
    return catalogue, evidence


def semantic_key(plan):
    # Call identifiers, route labels and provenance never manufacture diversity.
    return digest(dict(calls=[dict(skill=c.skill, roles=c.roles,
        arguments={k: dict(value=a.value, status=a.status, kind=a.kind, unit=a.unit,
                          frame=a.frame, source_output=a.source_output) for k, a in c.arguments.items()})
        for c in plan.calls], paths={name: {k: item[k] for k in JOINT_PATH_FIELDS}
        for name, item in plan.prefix.get("initial_artifacts", {}).items()}))


def _initial_routes(session, choice):
    target = session.ctx.obj_pos("carriage") + [0, 0, session.grasp_specs["carriage"][0] + choice["height"] + .10]
    return transfer_routes(session, target, clearance=choice["clearance"], yaw=choice["yaw"], grasp=None)


def build_pool(session, targets, seed, n=12, orders=None, precheck=True):
    if n < 1:
        raise ValueError("candidate count must be positive")
    if session.held is not None or getattr(session, "stage_completed", ()):
        raise ValueError("full assembly requires the initial unassembled, released scene")
    if set(targets) != set(PARTS):
        raise ValueError("full assembly requires exactly five product goals")
    started = time.perf_counter()
    orders = legal_orders(orders)
    catalogue, grasp_evidence = grasp_catalogue(session, targets)
    geometry_seconds = time.perf_counter() - started
    # Uniform shuffled Cartesian product: every branch changes executed inputs.
    # No preferred successful anchor, label-driven deletion, or duplicate IDs.
    sizes = [len(orders), *[len(catalogue[p]) for p in PARTS], 3, 2, 3 if precheck else 1]
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 759]))
    grid = rng.permutation(math.prod(sizes))
    plans, seen, cache, witnesses = [], set(), {}, []
    raw = conflicts = unknown = duplicates = 0
    for index in grid:
        row = np.unravel_index(int(index), sizes)
        raw += 1
        order = orders[row[0]]
        force, speed, route_index = (2.8, 3., 3.2)[row[-3]], (.006, .007)[row[-2]], row[-1]
        choices = {part: dict(**catalogue[part][row[i + 1]], clearance=.98, force=force,
                   **({} if part == "carriage" else dict(speed=speed))) for i, part in enumerate(PARTS)}
        plan = program(session, targets, order, choices, int(route_index))
        choice = choices["carriage"]
        key = (choice["yaw"], choice["height"], choice["clearance"])
        if precheck and key not in cache:
            t = time.perf_counter()
            cache[key], details = _initial_routes(session, choice)
            witnesses.extend(plain(details))
            geometry_seconds += time.perf_counter() - t
        if precheck:
            selected = cache[key][route_index]
            if selected["status"] != "necessary_pass":
                conflicts += int(selected["status"] == "conflict")
                unknown += int(selected["status"] != "conflict")
                continue
            plan = materialize_initial_path(session, plan, selected["path"])
        physical_key = semantic_key(plan)
        if physical_key in seen:
            duplicates += 1
            continue
        plans.append(plan)
        seen.add(physical_key)
        if len(plans) == n:
            break
    if len(plans) < n:
        raise CandidateGenerationError(f"geometric candidate pool exhausted: requested={n}, reached={len(plans)}, raw={raw}, conflicts={conflicts}, solver_unknown={unknown}")
    return plans, dict(requested=n, raw_count=raw, known_conflict=conflicts,
        unresolved=len(plans), materialization_unknown=unknown, deduplicated=len(plans), duplicates=duplicates,
        materializable=len(plans), pool_exhausted=False, label_blind=True, scope=TASK_SCOPE,
        completed_parts=[], continuation_parts=list(PARTS), grasp_catalogue=plain(catalogue),
        grasp_proposal_evidence=grasp_evidence, known_conflict_witnesses=witnesses,
        structure_branches=len({tuple(p.prefix["order"]) for p in plans}),
        grasp_branches=len({(p.prefix["choices"]["carriage"]["yaw"], p.prefix["choices"]["carriage"]["height"]) for p in plans}),
        initial_trajectory_branches=len({digest({k: a[k] for k in JOINT_PATH_FIELDS})
            for p in plans for a in p.prefix.get("initial_artifacts", {}).values()}),
        necessary_geometry_seconds=geometry_seconds, optional_geometry_seconds=0.,
        candidate_generation_seconds=time.perf_counter() - started - geometry_seconds)


def rebind_plan(session, targets, selected_plan):
    """Recompute only the selected initial approach in an independent scene."""
    old = selected_plan if isinstance(selected_plan, PlanIR) else PlanIR.from_dict(selected_plan)
    old.validate(session.parts)
    if (old.prefix.get("task_scope") != TASK_SCOPE or not old.prefix.get("initial_artifacts")
            or plain(targets) != old.prefix.get("targets")):
        raise ValueError("rebinding requires a materialized complete-stage plan with identical goals")
    if session.held is not None or getattr(session, "stage_completed", ()):
        raise ValueError("target rebinding requires a new unassembled scene")
    new = program(session, targets, old.prefix["order"], copy.deepcopy(old.prefix["choices"]),
                  old.prefix["initial_route_index"])
    if new.prefix["semantic_program_id"] != old.prefix.get("semantic_program_id"):
        raise ValueError("semantic program identity changed during target rebinding")
    routes, details = _initial_routes(session, new.prefix["choices"]["carriage"])
    route = routes[new.prefix["initial_route_index"]]
    if route["status"] != "necessary_pass":
        raise SkillFailure(f"target initial approach unavailable: {route['status']}: {plain(route.get('check'))}")
    new = materialize_initial_path(session, new, route["path"])
    return new, dict(old_candidate_id=old.id, new_candidate_id=new.id,
        semantic_program_id=new.prefix["semantic_program_id"], initial_route_index=new.prefix["initial_route_index"],
        old_start_state=old.prefix["start_state"], new_start_state=new.prefix["start_state"],
        method="selected_strategy_initial_approach_replanned_from_target_snapshot", geometry_checks=plain(details))
