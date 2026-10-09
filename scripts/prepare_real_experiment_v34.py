#!/usr/bin/env python3
"""Generate, value-rank and freshly validate a declared real-experiment layout.

The frozen V33 implementation is imported, never edited. The layout is an
explicit input; the seed controls proposal/random namespaces, not placement.
This script never connects to the physical Panda. Its TCP trace is a reference
for the calibrated hardware compiler, not a robot trajectory certificate.
Run as ``python -m scripts.prepare_real_experiment_v34 --help``.
"""
from __future__ import annotations

import argparse
import contextlib
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import time
import traceback
import xml.etree.ElementTree as ET

import numpy as np

# Capture at module load, before any later on-disk edit can relabel a running
# experiment. A full copy of these bytes is saved outside the frozen runtime.
_LOADED_SOURCE_BYTES = Path(__file__).read_bytes()
_LOADED_SOURCE_SHA256 = hashlib.sha256(_LOADED_SOURCE_BYTES).hexdigest()

SCHEMA = "twingraph.real_scene.v34"
POSE_NAMES = ("guide_base", "carriage", "end_stop", "handle", "pin_left",
              "pin_right", "pin_left_holder", "pin_right_holder", "wipe_tool")
PARTS = ("carriage", "end_stop", "pin_left", "pin_right", "handle")
FREE_BODIES = (*PARTS, "wipe_tool")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               allow_nan=False, default=plain), encoding="utf-8")


def plain(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON-serializable: {type(value).__name__}")


def digest(value):
    # Match simbench.value.plan.digest exactly; frozen plans were hashed with
    # Python's default JSON whitespace, not a different canonical format.
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    allow_nan=False, default=plain).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot_bridge_source(out):
    out = Path(out)
    relative = Path("source_snapshot") / "prepare_real_experiment_v34.py"
    path = out / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_LOADED_SOURCE_BYTES)
    provenance = dict(schema="twingraph.real_bridge_source.v34",
        script_sha256=_LOADED_SOURCE_SHA256, script_file=relative.as_posix(),
        import_source_path=str(Path(__file__).resolve()), capture_scope="bridge script loaded by this process",
        frozen_runtime_modified=False, captured_at_unix_s=time.time())
    repository = Path(__file__).resolve().parents[1]
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository,
            capture_output=True, text=True, timeout=3, check=False)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=repository,
            capture_output=True, text=True, timeout=3, check=False)
        if head.returncode == 0:
            provenance["git"] = dict(head=head.stdout.strip(),
                working_tree_dirty=bool(status.stdout.strip()) if status.returncode == 0 else None,
                status=status.stdout.splitlines() if status.returncode == 0 else None,
                network_accessed=False)
    except (OSError, subprocess.TimeoutExpired):
        provenance["git"] = dict(available=False, network_accessed=False)
    write(out / "bridge_source_provenance.json", provenance)
    return dict(script_sha256=_LOADED_SOURCE_SHA256, script_file=relative.as_posix(),
        provenance_file="bridge_source_provenance.json",
        provenance_sha256=file_sha(out / "bridge_source_provenance.json"))


def finite(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


def transform(value, name):
    result = finite(value, (4, 4), name)
    rotation = result[:3, :3]
    if (not np.allclose(result[3], [0., 0., 0., 1.], atol=1e-9)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-7)):
        raise ValueError(f"{name} must be a rigid right-handed transform")
    return result


def quat_matrix(value, name="quat_wxyz"):
    q = finite(value, (4,), name)
    if not np.isclose(np.linalg.norm(q), 1., atol=1e-5):
        raise ValueError(f"{name} must be a unit quaternion in w,x,y,z order")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def matrix_quat(rotation):
    # SciPy is only needed once a simulation is actually requested.
    from scipy.spatial.transform import Rotation
    xyzw = Rotation.from_matrix(rotation).as_quat()
    return xyzw[[3, 0, 1, 2]].tolist()


def pose_transform(row, name):
    result = np.eye(4)
    result[:3, 3] = finite(row.get("position_m"), (3,), f"{name}.position_m")
    result[:3, :3] = quat_matrix(row.get("quat_wxyz"), f"{name}.quat_wxyz")
    return result


def normalize_layout(layout, *, allow_nominal=False):
    """Validate input and return explicit world poses without seed sampling."""
    if layout.get("schema") != SCHEMA:
        raise ValueError(f"layout schema must be {SCHEMA}")
    if layout.get("frame") not in ("panda_base", "twin_world"):
        raise ValueError("layout frame must be panda_base or twin_world")
    if type(layout.get("measured")) is not bool:
        raise ValueError("layout.measured must explicitly be true or false")
    if not layout["measured"] and not allow_nominal:
        raise ValueError("nominal placement requires --allow-nominal-layout; it is not measured hardware")
    robot = layout.get("robot", {})
    world_base = transform(robot.get("T_world_base"), "robot.T_world_base")
    if not np.allclose(world_base[:3, 2], [0., 0., 1.], atol=1e-7):
        raise ValueError("V33 tabletop controllers require an upright robot/world gravity alignment")
    joints = finite(robot.get("joints_rad"), (7,), "robot.joints_rad")
    width = float(robot.get("finger_width_m", .08))
    if not np.isfinite(width) or not 0 <= width <= .08:
        raise ValueError("robot.finger_width_m must be in [0, .08] metres")
    poses = layout.get("poses", {})
    if set(poses) != set(POSE_NAMES):
        raise ValueError(f"layout poses must contain exactly {POSE_NAMES}")
    world_frame = world_base if layout["frame"] == "panda_base" else np.eye(4)
    world_poses = {}
    for name in POSE_NAMES:
        p = world_frame @ pose_transform(poses[name], name)
        if not np.allclose(p[:3, 2], [0., 0., 1.], atol=.02):
            raise ValueError(f"{name}: this controller supports upright CAD body axes only")
        world_poses[name] = dict(position_m=p[:3, 3].tolist(), quat_wxyz=matrix_quat(p[:3, :3]))
    table = layout.get("table", {})
    center = finite(table.get("center_xy_m"), (2,), "table.center_xy_m")
    half = finite(table.get("half_size_xy_m"), (2,), "table.half_size_xy_m")
    top = float(table.get("top_z_m"))
    if not np.isfinite(top) or np.any(half <= 0.):
        raise ValueError("table requires a finite top and positive half sizes")
    if not np.allclose(world_frame[:3, :3], np.eye(3), atol=1e-7):
        # The V33 table has fixed world-aligned collision edges. A rotated
        # surface must be modelled explicitly, not widened silently here.
        raise ValueError("table XY alignment must match twin world axes")
    center_world = (world_frame @ np.r_[center, top, 1.])[:3]
    uncertainty = layout.get("uncertainty", {})
    pos_tol = float(uncertainty.get("position_m", .002))
    rot_tol = float(uncertainty.get("orientation_rad", .035))
    if (not np.isfinite([pos_tol, rot_tol]).all() or not 0 < pos_tol <= .010
            or not 0 < rot_tol <= .20):
        raise ValueError("uncertainty must be positive and <=10 mm / 0.20 rad")
    physics = deepcopy(layout.get("physics", {"calibrated": False}))
    if layout["measured"]:
        if layout.get("placement_verified") is not True or layout.get("cad_match_verified") is not True:
            raise ValueError("measured layout needs placement_verified and cad_match_verified")
        if not layout.get("measurement_source") or not layout.get("measured_at_utc"):
            raise ValueError("measured layout needs measurement_source and measured_at_utc")
        if robot.get("start_state_verified") is not True or robot.get("base_frame_verified") is not True:
            raise ValueError("measured layout needs verified base frame and robot start state")
        if robot.get("state_measured") is not True:
            raise ValueError("measured layout needs robot.state_measured from the actual Panda")
        if physics.get("calibrated") is not True:
            raise ValueError("measured twin needs calibrated mass/friction; nominal assumptions cannot certify it")
    if physics.get("calibrated"):
        masses, friction = physics.get("mass_kg", {}), physics.get("friction", {})
        if not set(FREE_BODIES).issubset(masses) or not set(POSE_NAMES).issubset(friction):
            raise ValueError("calibrated physics requires all free-body masses and all body friction triples")
        for name in FREE_BODIES:
            if not np.isfinite(masses[name]) or not 0 < masses[name] <= 5:
                raise ValueError(f"invalid mass_kg for {name}")
        for name in POSE_NAMES:
            coefficients = finite(friction[name], (3,), f"{name}.friction")
            if np.any(coefficients < 0) or coefficients[0] <= 0:
                raise ValueError(f"invalid friction for {name}")
    result = dict(schema=SCHEMA, frame="twin_world", measured=layout["measured"],
                layout_input_sha256=digest(layout), source_frame=layout["frame"],
                robot=dict(T_world_base=world_base.tolist(), joints_rad=joints.tolist(), finger_width_m=width),
                table=dict(top_z_m=float(center_world[2]), center_xy_m=center_world[:2].tolist(),
                           half_size_xy_m=half.tolist()), poses=world_poses,
                uncertainty=dict(position_m=pos_tol, orientation_rad=rot_tol), physics=physics,
                seed=int(layout.get("seed", 4000)),
                source=deepcopy(layout.get("source", "explicit operator layout")))
    if layout["measured"]:
        # Hardware preflight consumes this normalized file, so retain the
        # actual measurement provenance instead of losing it at conversion.
        # Nominal normalization stays byte-compatible with existing demo
        # receipts; it never provides a hardware measurement certificate.
        result["robot"] = {**deepcopy(robot), **result["robot"]}
        for key in ("placement_verified", "cad_match_verified", "measurement_source", "measured_at_utc"):
            result[key] = deepcopy(layout[key])
    return result


def scene_hook(layout, directory):
    """Replace placement/robot/physical inputs in a fresh per-run scene XML."""
    directory = Path(directory)
    def apply(root, source_spec):
        from simbench.assembly.scene import fmt
        for name, row in layout["poses"].items():
            body = root.find(f".//worldbody/body[@name='{name}']")
            if body is None:
                raise ValueError(f"scene body missing: {name}")
            body.set("pos", fmt(row["position_m"]))
            body.set("quat", fmt(row["quat_wxyz"]))
            physics = layout["physics"]
            if physics.get("calibrated"):
                for geom in body.findall("geom"):
                    if geom.get("contype", "1") != "0":
                        geom.set("friction", fmt(physics["friction"][name]))
                if name in FREE_BODIES:
                    inertial = body.find("inertial")
                    scale = physics["mass_kg"][name] / float(inertial.get("mass"))
                    inertial.set("mass", str(physics["mass_kg"][name]))
                    key = "fullinertia" if inertial.get("fullinertia") is not None else "diaginertia"
                    inertial.set(key, fmt(np.fromstring(inertial.get(key), sep=" ") * scale))
        table = root.find(".//geom[@name='table']")
        thickness = float(np.fromstring(table.get("size"), sep=" ")[2])
        surface = layout["table"]
        table.set("pos", fmt([*surface["center_xy_m"], surface["top_z_m"] - thickness]))
        table.set("size", fmt([*surface["half_size_xy_m"], thickness]))
        # Change a copy of the included robot; original runtime assets retain
        # exactly their frozen V33 bytes and source fingerprint.
        include = root.find("include")
        panda = ET.parse((directory / include.get("file")).resolve()).getroot()
        base = panda.find(".//body[@name='panda_base']")
        world_base = np.asarray(layout["robot"]["T_world_base"])
        base.set("pos", fmt(world_base[:3, 3]))
        base.set("quat", fmt(matrix_quat(world_base[:3, :3])))
        path = directory / "panda_explicit_layout.xml"
        ET.ElementTree(panda).write(path, encoding="unicode")
        include.set("file", str(path.resolve()))
        return dict(schema=SCHEMA, explicit_layout_sha256=digest(layout),
                    layout_seed_sampled=False, measured=layout["measured"],
                    source_pose_frame="twin_world", source_poses=layout["poses"],
                    physical_parameters=layout["physics"])
    return apply


def make_explicit_scene(layout, directory):
    import mujoco
    from simbench.value.system_v12 import make_scene
    from simbench.value.stage_v7 import refresh_visual_observation
    from simbench.value.stage_v12 import bind_visual_receiver_targets
    from scripts.collect_sliding_assembly_v23 import REQUIRED
    directory = Path(directory)
    _, session, _, _ = make_scene(layout["seed"], directory, domain="online", level="L0",
        observation_backend="mujoco_state_pose", position_noise_std_m=0., yaw_noise_std_rad=0.,
        scene_layout_hook=scene_hook(layout, directory))
    # Existing scene initialization uses HOME. Rebind the actual start encoder
    # state before generating a plan; this is simulator initialization only.
    ctx = session.ctx
    joints = np.asarray(layout["robot"]["joints_rad"])
    if np.any(joints < ctx.model.jnt_range[ctx.arm_joint_ids, 0]) or np.any(joints > ctx.model.jnt_range[ctx.arm_joint_ids, 1]):
        raise ValueError("declared robot start joints violate Panda limits")
    ctx.data.qpos[ctx.arm_qadr] = joints
    ctx.data.qvel[ctx.model.jnt_dofadr[ctx.arm_joint_ids]] = 0.
    half_width = layout["robot"]["finger_width_m"] / 2.
    ctx.data.qpos[ctx.finger_qadr] = [half_width, -half_width]
    ctx.hold_arm(); ctx.set_finger_ctrl(half_width)
    mujoco.mj_forward(ctx.model, ctx.data)
    refresh_visual_observation(session, parts=session.parts)
    bind_visual_receiver_targets(session)
    session.sliding_assembly_v23 = True
    session.end_stop_place_acceptance_v12 = "stable_supported"
    session.required_stage_passes = REQUIRED
    discrepancies = []
    for name, target in layout["poses"].items():
        position, quat = ctx.obj_pose(name)
        distance = float(np.linalg.norm(np.asarray(position) - target["position_m"]))
        angle = rotation_distance(quat_matrix(quat), quat_matrix(target["quat_wxyz"]))
        discrepancies.append(dict(name=name, position_error_m=distance, orientation_error_rad=angle))
        if distance > layout["uncertainty"]["position_m"] or angle > layout["uncertainty"]["orientation_rad"]:
            write(directory / "layout_mismatch.json", discrepancies)
            raise ValueError(f"{name}: gravity-settled twin differs from declared actual layout: {distance:.5f} m/{angle:.5f} rad")
    write(directory / "layout_acceptance.json", dict(passed=True, rows=discrepancies,
          measured_hardware=layout["measured"], observation_source="reconstructed twin state, not live vision"))
    return session


def rotation_distance(first, second):
    return float(np.arccos(np.clip((np.trace(first.T @ second)-1.) / 2., -1., 1.)))


def tcp_pose(ctx):
    result = np.eye(4)
    result[:3, :3] = ctx.eef_mat()
    result[:3, 3] = ctx.eef_pos()
    return result.tolist()


class TwinTrace:
    """Record actual simulator primitives with their feedback/skill contexts."""
    def __init__(self, session):
        self.session = session
        self.events, self.skill_events, self.stack = [], [], []
        self.control_samples, self.active_primitives = [], []
        self.originals = []
        self.start_pose = tcp_pose(session.ctx)

    def stage(self):
        for row in reversed(self.stack):
            if row.get("part") in PARTS:
                return row["part"]
        held = getattr(self.session, "held", None)
        if held in PARTS:
            return held
        completed = getattr(self.session, "stage_completed", ())
        for row in reversed(self.stack):
            if row.get("order"):
                pending = [p for p in row["order"] if p not in completed]
                if pending:
                    return pending[0]
        return "acceptance"

    def __enter__(self):
        session = self.session
        original_call = session.call
        def call(name, **params):
            part = params.get("part")
            artifact = params.get("artifact")
            if part is None and isinstance(artifact, str):
                part = getattr(session, "artifacts", {}).get(artifact, {}).get("part")
            required = params.get("required_parts")
            if part is None and isinstance(required, (list, tuple)) and len(required) == 1:
                part = required[0]
            context = dict(skill=name, part=part, order=params.get("order"))
            self.stack.append(context)
            row = dict(skill=name, stage=self.stage(), parameters=deepcopy(params),
                       primitive_start=len(self.events), pose_before_world_eef=tcp_pose(session.ctx))
            try:
                result = original_call(name, **params)
                row.update(ok=bool(result.ok), metrics=deepcopy(result.metrics), reason=result.reason)
                return result
            except Exception as exc:
                row.update(ok=False, error=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                row.update(primitive_stop=len(self.events), pose_after_world_eef=tcp_pose(session.ctx))
                self.skill_events.append(row)
                self.stack.pop()
        self.originals.append((session, "call", original_call))
        session.call = call
        previous_hook = getattr(session.ctx, "on_control_step", None)
        def control_step(ctx):
            if previous_hook:
                previous_hook(ctx)
            self.control_samples.append(dict(sim_time_s=float(ctx.data.time), stage=self.stage(),
                primitive_id=self.active_primitives[-1] if self.active_primitives else None,
                pose_world_eef=tcp_pose(ctx), pad_span_m=float(ctx.pad_span())))
        self.originals.append((session.ctx, "on_control_step", previous_hook))
        session.ctx.on_control_step = control_step
        for name in ("move", "servo", "execute_joint", "execute_joint_waypoints", "open", "close"):
            original = getattr(session.arm, name)
            wrapper = self.wrap(name, original)
            self.originals.append((session.arm, name, original))
            setattr(session.arm, name, wrapper)
        return self

    def wrap(self, name, original):
        signature = inspect.signature(original)
        def wrapper(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            parameters = dict(bound.arguments)
            index = len(self.events)
            event = dict(id=f"primitive_{index:06d}", stage=self.stage(),
                         parent_primitive_id=self.active_primitives[-1] if self.active_primitives else None,
                         skill=self.stack[-1]["skill"] if self.stack else None,
                         primitive=name, parameters=deepcopy(parameters),
                         call_stack=deepcopy(self.stack), sim_time_s=float(self.session.ctx.data.time),
                         pose_before_world_eef=tcp_pose(self.session.ctx))
            if name in ("move", "servo"):
                goal = np.eye(4)
                goal[:3, 3] = parameters["xyz"]
                rotation = parameters.get("rotation")
                goal[:3, :3] = self.session.arm.rotation if rotation is None else rotation
                event["destination_world_eef"] = goal.tolist()
            self.events.append(event)
            self.active_primitives.append(event["id"])
            try:
                result = original(*args, **kwargs)
                event["returned"] = result if result is None or isinstance(result, (bool, int, float)) else str(result)
                return result
            except Exception as exc:
                event["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                event["pose_after_world_eef"] = tcp_pose(self.session.ctx)
                event["end_sim_time_s"] = float(self.session.ctx.data.time)
                self.active_primitives.pop()
        return wrapper

    def __exit__(self, exc_type, exc, tb):
        for obj, name, original in reversed(self.originals):
            setattr(obj, name, original)


def freeze_candidates(layout, out, *, planner="llm", candidate_n=8, pool_n=48):
    from scripts.collect_sliding_assembly_v23 import controller_plan, TASK_VERSION
    from simbench.value.planner_v12 import propose, assembly_program
    from simbench.value.provenance_v12 import fingerprint
    session = make_explicit_scene(layout, out / "planning")
    observation = deepcopy(session.decision_observation)
    pool, grounding = propose(observation, cad=session.planning_cad, n=pool_n,
                              seed=layout["seed"], completed=("cleaning",))
    if planner == "llm":
        from scripts.llm_transport_v32 import choose
        selected, source = choose(pool, observation, session.planning_cad, out / "planner", candidate_n)
    elif planner == "grounded":
        selected = pool[:candidate_n]
        source = dict(source="declared_grounded_pool_prefix", online_llm_call=False,
                      count=candidate_n, outcome_labels_read=False)
    else:
        raise ValueError("planner must be llm or grounded")
    entries, seen = [], set()
    for proposal in selected:
        proposal = {k:deepcopy(v) for k,v in proposal.items()
                    if k not in ("wipe_variant", "wipe_force", "wipe_duration")}
        if any(g.get("status") == "rejected" for g in proposal.get("necessary_geometry", {}).values()):
            raise ValueError("planner selected geometry-rejected proposal")
        complete, assembly = controller_plan(session, proposal), assembly_program(session, proposal)
        signature = digest(complete.to_dict()["calls"])
        if signature in seen:
            raise ValueError("duplicate complete executable candidates")
        seen.add(signature)
        entries.append(dict(name=proposal["name"], proposal=proposal,
            complete_candidate_plan_ir=complete.to_dict(), plan_sha256=digest(complete.to_dict()),
            assembly_plan_ir=assembly.to_dict(), assembly_plan_sha256=digest(assembly.to_dict())))
    request = dict(schema="twingraph.real_experiment_request.v34", task_version=TASK_VERSION,
        seed=layout["seed"], level="L0", domain="online", runtime_sha256=fingerprint()["sha256"],
        explicit_layout_sha256=digest(layout), decision_observation=observation,
        observation_random_state=session.state_observation_random_state,
        position_noise_std_m=0., yaw_noise_std_rad=0., candidates=entries,
        candidate_n=candidate_n, pool_n=pool_n, source=dict(grounding=grounding, **source),
        outcome_labels_read=False, measured_hardware=layout["measured"],
        twin_observation_source="reconstructed explicit layout; fresh hardware perception still required")
    write(out / "request.json", request)
    return request


def rank(request, training, out):
    from scripts.train_value_v33 import load_ranker, score_candidates
    from scripts.value_features_v33 import encode_candidates, SCHEMA as FEATURE_SCHEMA
    frozen, models = load_ranker(training)
    if frozen.get("feature_schema") != FEATURE_SCHEMA or frozen.get("budget") != 3:
        raise ValueError("need unchanged V33 typed-geometry Top3 model freeze")
    started = time.perf_counter()
    scores, predictions, uncertainty = score_candidates(encode_candidates(request), frozen, models)
    if scores.shape != (len(request["candidates"]),) or not np.isfinite(scores).all():
        raise ValueError("value outputs must be finite and match candidate pool")
    selection = dict(schema="twingraph.real_value_selection.v34", order=np.argsort(-scores, kind="stable").tolist(),
        budget=3, scores=scores.tolist(), predictions=predictions.tolist(), uncertainty=uncertainty.tolist(),
        request_sha256=digest(request), model_freeze_sha256=file_sha(training / "selection_frozen.json"),
        checkpoint_sha256=frozen["checkpoint_sha256"], model_paths=list(frozen["checkpoint_sha256"]),
        feature_schema=FEATURE_SCHEMA, outcome_labels_read=False, ranking_seconds=time.perf_counter()-started)
    write(out / "selection.json", selection)
    return selection


def fresh_rollout(request, entry, layout, out, *, repeat=20, timeout=1800.):
    from simbench.value.provenance_v12 import fingerprint
    from simbench.value.plan import PlanIR
    from simbench.value.physical import PhysicalRunner, perturbation
    from scripts.collect_sliding_assembly_v23 import REQUIRED
    out.mkdir(parents=True, exist_ok=False)
    if fingerprint()["sha256"] != request["runtime_sha256"] or digest(layout) != request["explicit_layout_sha256"]:
        raise ValueError("source runtime or explicit layout changed after planning")
    session = make_explicit_scene(layout, out / "scene")
    if session.decision_observation != request["decision_observation"]:
        raise ValueError("fresh same-layout observation differs from frozen planning request")
    if session.state_observation_random_state != request["observation_random_state"]:
        raise ValueError("fresh observation random namespace differs")
    plan = PlanIR.from_dict(entry["complete_candidate_plan_ir"]).validate(session.parts)
    if digest(plan.to_dict()) != entry["plan_sha256"]:
        raise ValueError("candidate plan hash changed")
    started = time.perf_counter()
    trial = perturbation(request["seed"], repeat, request["domain"])
    with TwinTrace(session) as trace, (out / "console.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        try:
            result = PhysicalRunner(session, timeout=timeout).run(plan, trial, keep_trace=True)
        except Exception as exc:
            result = dict(valid=False, success=False, error=f"{type(exc).__name__}: {exc}",
                          software_exception=traceback.format_exc(), timeout=False,
                          stage_passes=deepcopy(getattr(session, "stage_passes", {})))
    accepted = all(result.get("stage_passes", {}).get(k) is True for k in REQUIRED)
    if result.get("timeout") or bool(result["success"]) != accepted:
        result.update(valid=False, invalid_reason="timeout or functional acceptance mismatch")
    if fingerprint()["sha256"] != request["runtime_sha256"]:
        result.update(valid=False, invalid_reason="runtime changed during twin validation")
    result.update(schema="twingraph.real_twin_result.v34", candidate_name=entry["name"],
        request_sha256=digest(request), explicit_layout_sha256=digest(layout), plan_sha256=entry["plan_sha256"],
        runtime_sha256=request["runtime_sha256"], trial=trial,
        fresh_total_wall_seconds=time.perf_counter()-started,
        receiver_binding_history=deepcopy(getattr(session, "receiver_binding_history", [])),
        pin_joint_engagement=deepcopy(session.artifacts.get("pin_joint_engagement")))
    write(out / "result.json", result)
    write(out / "tcp_reference_trace.json", dict(schema="twingraph.twin_tcp_reference.v34", frame="twin_world",
        eef_site_name="grip_site", start_pose_world_eef=trace.start_pose, events=trace.events,
        skill_events=trace.skill_events, control_samples=trace.control_samples, hardware_executable=False,
        warning="simulation Cartesian primitives are not hardware force control or a replay certificate"))
    return result


def staged_program(request, entry, result, folder):
    stage_programs = []
    for path in sorted((folder / "scene" / "stage_programs").glob("*.json")):
        row = read(path)
        stage_programs.append(dict(stage=path.stem.split("_", 1)[1], **row))
    stages = []
    for part in entry["proposal"]["order"]:
        bindings = [p for p in stage_programs if p["stage"] == part]
        stages.append(dict(part=part, choices=entry["proposal"]["choices"][part],
            atomic_programs=bindings,
            hardware_preconditions=["fresh source-body pose in calibrated Panda frame",
                                    "receiver pose measured after every predecessor release",
                                    "measured object-to-TCP transform after grasp",
                                    "collision-checked free motion and guarded contact limits"],
            hardware_postconditions=(
                ["carriage retained between rails and released"] if part == "carriage" else
                ["stop captured; both bores share a straight route into base"] if part == "end_stop" else
                ["pin passes stop and >=6 mm into base; released pin remains engaged"] if part.startswith("pin_") else
                ["handle seated on post; both pins remain engaged; empty gripper retracted"])))
    return dict(schema="twingraph.real_feedback_program.v34", candidate_id=entry["name"],
        plan_sha256=entry["plan_sha256"], stages=stages, receiver_binding_history=result.get("receiver_binding_history"),
        complete_candidate_plan_ir=entry["complete_candidate_plan_ir"],
        task_acceptance_required=["assembly_pass", "base_hole_engagement_pass", "fixture_capture_pass",
                                  "final_seat_pass", "final_release_and_retraction_pass"],
        hardware_executable=False, twin_validated=bool(result.get("valid") and result.get("success")))


def prepare(layout_path, training, out, *, allow_nominal=False, planner="llm", candidate_n=8,
            pool_n=48, repeat=20, timeout=1800., plan_only=False):
    layout_path, training, out = Path(layout_path), Path(training), Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("output must be a new empty folder; preserve existing trial evidence")
    raw = read(layout_path)
    layout = normalize_layout(raw, allow_nominal=allow_nominal)
    out.mkdir(parents=True, exist_ok=True)
    bridge_source = snapshot_bridge_source(out)
    write(out / "layout_input.json", raw)
    write(out / "layout.json", layout)
    request = freeze_candidates(layout, out, planner=planner, candidate_n=candidate_n, pool_n=pool_n)
    selection = rank(request, training, out)
    write(out / "coordinate_targets.json", dict(schema="twingraph.real_coordinates.v34",
        frame="twin_world", T_world_base=layout["robot"]["T_world_base"],
        initial_poses=layout["poses"], initial_poses_measured=layout["measured"],
        cad_goals=request["decision_observation"]["assembly_targets"],
        pin_goal_note="planning target is minimum bridge-depth release, not a guaranteed full head seat",
        hardware_tcp_note="CAD body positions are not robot EE/TCP commands"))
    report = dict(schema="twingraph.real_preparation.v34", candidate_count=len(request["candidates"]),
        planner=planner, selected=None, validations=[], request_sha256=digest(request),
        layout_sha256=digest(layout), model_freeze_sha256=selection["model_freeze_sha256"],
        bridge_source=bridge_source,
        measured_layout=layout["measured"], hardware_executed=False, hardware_executable=False,
        phase="ranked" if plan_only else "validating")
    write(out / "preparation.json", report)
    if plan_only:
        return report
    for index in selection["order"][:selection["budget"]]:
        entry = request["candidates"][index]
        folder = out / "validation" / entry["name"]
        result = fresh_rollout(request, entry, layout, folder, repeat=repeat, timeout=timeout)
        report["validations"].append(dict(candidate=entry["name"], valid=bool(result["valid"]),
            success=bool(result["success"]), seconds=result["fresh_total_wall_seconds"],
            result=str(folder.relative_to(out) / "result.json"), result_sha256=file_sha(folder / "result.json")))
        write(out / "preparation.json", report)
        if not result["valid"]:
            raise ValueError(f"invalid twin validation: {entry['name']}; inspect preserved result")
        if result["success"]:
            report["selected"] = entry["name"]
            write(out / "selected_candidate.json", entry)
            write(out / "selected_feedback_program.json", staged_program(request, entry, result, folder))
            reference = read(folder / "tcp_reference_trace.json")
            reference.update(schema="twingraph.real_reference_schedule.v34", task_id=out.name,
                candidate_id=entry["name"], T_world_base=layout["robot"]["T_world_base"],
                twin_validated=True, measured_layout=layout["measured"],
                evidence=dict(request_sha256=file_sha(out / "request.json"), layout_sha256=file_sha(out / "layout.json"),
                    candidate_sha256=file_sha(out / "selected_candidate.json"), twin_result_sha256=file_sha(folder / "result.json")),
                artifact_files=dict(request="request.json", layout="layout.json", candidate="selected_candidate.json",
                                    twin_result=str(folder.relative_to(out) / "result.json")),
                model_freeze_sha256=selection["model_freeze_sha256"], runtime_sha256=request["runtime_sha256"],
                stages=staged_program(request, entry, result, folder)["stages"])
            # This is a separate optional source binding: old frozen
            # request/layout hashes and the four hardware evidence fields
            # retain their original meaning and serialized contents.
            reference["bridge_source"] = bridge_source
            write(out / "reference_schedule.json", reference)
            break
    report["phase"] = "twin_validated" if report["selected"] else "no_top3_passed"
    write(out / "preparation.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True, help="unchanged frozen V33 training folder")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--planner", choices=("llm", "grounded"), default="llm")
    parser.add_argument("--allow-nominal-layout", action="store_true")
    parser.add_argument("--candidate-n", type=int, default=8)
    parser.add_argument("--pool-n", type=int, default=48)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=1800.)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.candidate_n < 3 or args.candidate_n > args.pool_n or args.pool_n > 512:
        parser.error("need 3 <= candidate-n <= pool-n <= 512")
    if not np.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be finite and positive")
    report = prepare(args.layout, args.training, args.out, allow_nominal=args.allow_nominal_layout,
        planner=args.planner, candidate_n=args.candidate_n, pool_n=args.pool_n,
        repeat=args.repeat, timeout=args.timeout, plan_only=args.plan_only)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if args.plan_only or report["selected"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
