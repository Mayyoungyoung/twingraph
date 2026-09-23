"""Typed atomic skills; composite assembly recipes live in task.py.

Contracts are executable, not just descriptions. Perception currently has an
explicit simulator-pose backend. Planning and checking do not advance physics.
"""

from dataclasses import dataclass, field, asdict
import copy
import json
import inspect
import hashlib
from pathlib import Path
import time
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from .control import PoseController, down, HOME
from .contracts import check, apply_effects, session_state
from .interfaces import resolve, family, PUBLIC_SKILLS
from .ports import declare_ports, state_interface, validate_ports

PARTS = ("carriage", "end_stop", "handle", "pin_left", "pin_right")
DEFAULT_CAPABILITIES = {"pin_left": ("pin",), "pin_right": ("pin",)}
GRASP = {
    "carriage": (0.027, 0.028),
    "end_stop": (0.023, 0.026),
    "handle": (0.0, 0.042),
    "pin_left": (0.006, 0.018),
    "pin_right": (0.006, 0.018),
}


@dataclass
class Spec:
    name: str
    label: str
    category: str
    requires: tuple = ()
    produces: str = ""
    implementation: str = "feedback"
    description: str = ""
    kind: str = "executable"
    family: str = ""
    effects: tuple = ()
    obligations: tuple = ()
    output_bindings: tuple = (("part", "part"),)
    ports: tuple = ()
    state_reads: tuple = ()
    state_writes: tuple = ()


CATALOG = {}
HANDLERS = {}


def skill(
    name,
    label,
    category,
    requires=(),
    produces="",
    implementation="feedback",
    description="",
    effects=(),
    obligations=None,
    output_bindings=(("part", "part"),),
    legacy=True,
):
    def deco(fn):
        HANDLERS[name] = Spec(
            name,
            label,
            category,
            requires,
            produces,
            implementation,
            description or label,
            (
                "generator"
                if produces and name != "observe_parts"
                else ("checker" if category == "verification" else "executable")
            ),
            family(name),
            effects,
            tuple(
                obligations
                if obligations is not None
                else (
                    ("closed-loop tracking/contact and success checks",)
                    if not produces and category != "verification"
                    else ("measurement or solver outcome",)
                )
            ),
            output_bindings,
            declare_ports(fn),
            *state_interface(name, category),
        )
        if legacy:
            CATALOG[name] = HANDLERS[name]
        fn.spec = HANDLERS[name]
        return fn

    return deco


@dataclass
class Result:
    ok: bool = True
    metrics: dict = field(default_factory=dict)
    reason: str = ""


class SkillFailure(RuntimeError):
    pass


class Session:
    def __init__(
        self,
        ctx,
        recorder=None,
        out=None,
        seed=0,
        noise=0.0,
        parts=None,
        grasp_specs=None,
        capabilities=None,
    ):
        self.ctx = ctx
        self.arm = PoseController(ctx)
        self.rec = recorder
        self.artifacts = {}
        self.held = None
        self.results = []
        self.observations = {}
        self.rng = np.random.default_rng(seed)
        self.noise = noise
        self.out = Path(out) if out else None
        self.checkpoints = {}
        self.stroke = []
        self.stroke_runs = []
        self.stroke_peak_forces = []
        # Scene-specific inventories let the same atoms operate on isolated
        # teaching objects without pretending a cube is an assembly carriage.
        self.parts = tuple(PARTS if parts is None else parts)
        self.grasp_specs = dict(GRASP if grasp_specs is None else grasp_specs)
        self.capabilities = (
            capabilities
            if capabilities is not None
            else {
                p: DEFAULT_CAPABILITIES[p]
                for p in self.parts
                if p in DEFAULT_CAPABILITIES
            }
        )
        self.candidate_batches = []
        self.grasp_epoch = 0
        self.active_candidate_id = None
        # v7 installs one frozen simulated-sensor observation at the decision
        # boundary.  Leaving this unset preserves the legacy v5/v6 contract.
        self.decision_observation = None
        self.sensor_manifest = None
        self.dirty_state = None
        self.stage_passes = {}
        self.failure_reasons = []
        self.perception_backend = "legacy"
        self.visual_calibration = None

    def set_decision_observation(self, observation):
        if not isinstance(observation, dict) or "objects" not in observation:
            raise ValueError("invalid frozen observation")
        self.decision_observation = copy.deepcopy(observation)
        self.sensor_manifest = copy.deepcopy(observation.get("config", {}))
        self.observations = {}
        for part, row in observation["objects"].items():
            if row.get("valid", True) and row.get("position_m") is not None:
                self.observations[part] = [np.asarray(row["position_m"], dtype=float).copy() for _ in range(5)]
        self.perception_backend = observation.get("backend", self.perception_backend)
        self.visual_calibration = copy.deepcopy(observation.get("calibration"))

    def call(self, name, **params):
        requested = name
        try:
            name, params = resolve(name, params)
            if name not in HANDLERS:
                raise ValueError(f"unknown skill {name}")
            bound = inspect.signature(getattr(self, name)).bind(**params)
            bound.apply_defaults()
        except (TypeError, ValueError) as exc:
            raise SkillFailure(str(exc)) from exc
        params = dict(bound.arguments)
        spec = HANDLERS[name]
        validate_ports(spec, params)
        part = params.get("part")
        previous_atom = getattr(self, "_active_atom", None)
        atom = spec.family if spec.family in PUBLIC_SKILLS else previous_atom
        state = session_state(self)
        verdict = check(spec, params, state, contact=self.ctx.grasp_contacts)
        if verdict["conflicts"]:
            raise SkillFailure(f"{name}: " + "; ".join(verdict["conflicts"]))
        if self.rec:
            display = PUBLIC_SKILLS.get(atom)
            label = (
                f"{display['label']}  |  {atom}"
                if display
                else (
                    f"反馈 / 验收：{spec.label}"
                    if spec.family == "auxiliary"
                    else f"内部准备：{spec.label}"
                )
            )
            self.rec.set_skill(atom, label + (f"  ·  {part}" if part else ""))
            self.rec.pause(0.4)
        start = self.ctx.data.time
        wall = time.perf_counter()
        self._active_atom = atom
        try:
            outcome = getattr(self, name)(**params)
        finally:
            self._active_atom = previous_atom
        if not isinstance(outcome, Result):
            raise TypeError(f"{name} must return Result")
        if outcome.ok:
            outcome.reason = ""
            state.artifacts = self.artifacts
            apply_effects(spec, params, state, symbolic=False)
            self.held = state.held
            self.grasp_epoch = state.grasp_epoch
        row = dict(
            index=len(self.results),
            skill=name,
            interface=requested,
            atom=atom,
            params=params,
            ok=bool(outcome.ok),
            metrics=outcome.metrics,
            reason=outcome.reason,
            sim_seconds=float(self.ctx.data.time - start),
            wall_seconds=time.perf_counter() - wall,
        )
        self.results.append(row)
        print(json.dumps(row, default=lambda x: np.asarray(x).tolist()), flush=True)
        if self.out:
            self.out.mkdir(parents=True, exist_ok=True)
            (self.out / "steps.json").write_text(
                json.dumps(
                    self.results,
                    indent=2,
                    ensure_ascii=False,
                    default=lambda x: np.asarray(x).tolist(),
                )
            )
        if self.rec:
            self.rec.pause(0.4 if row["sim_seconds"] > 0.5 else 1.3)
        if not outcome.ok:
            raise SkillFailure(f"{name}: {outcome.reason} {outcome.metrics}")
        return outcome

    def artifact(self, name, kind, part=None, **data):
        if (
            kind
            in ("joint_path", "cartesian_path", "insertion", "recovery", "wipe_path")
            and "binding" not in data
        ):
            grasp = self.artifacts.get("grasp")
            data["binding"] = dict(
                held=self.held,
                grasp_epoch=self.grasp_epoch,
                grasp_artifact="grasp" if grasp else None,
                grasp_id=grasp.get("id") if grasp else None,
                prefix_id=self.active_candidate_id,
            )
        self.artifacts[name] = dict(type=kind, part=part, **data)

    def hold(self, seconds=0.25):
        for _ in range(max(1, int(seconds / self.ctx.control_dt))):
            self.ctx.step()

    def snapshot(self):
        geometry = hashlib.sha256(
            self.ctx.model.geom_size.tobytes() + self.ctx.model.geom_pos.tobytes()
        ).hexdigest()
        return dict(
            geometry_sha256=geometry,
            physics=self.ctx.snapshot(),
            held=self.held,
            artifacts=copy.deepcopy(self.artifacts),
            observations=copy.deepcopy(self.observations),
            rotation=self.arm.rotation.copy(),
            stroke=list(self.stroke),
            stroke_runs=copy.deepcopy(self.stroke_runs),
            stroke_peak_forces=list(self.stroke_peak_forces),
            rng=copy.deepcopy(self.rng.bit_generator.state),
            grasp_epoch=self.grasp_epoch,
            active_candidate_id=self.active_candidate_id,
            decision_observation=copy.deepcopy(self.decision_observation),
            sensor_manifest=copy.deepcopy(self.sensor_manifest),
            dirty_state=copy.deepcopy(self.dirty_state),
            stage_passes=copy.deepcopy(self.stage_passes),
            failure_reasons=list(self.failure_reasons),
            perception_backend=self.perception_backend,
            visual_calibration=copy.deepcopy(self.visual_calibration),
            stage_targets=copy.deepcopy(getattr(self, "stage_targets", None)),
            execution_relocalizations=copy.deepcopy(getattr(self, "execution_relocalizations", None)),
        )

    def restore(self, state):
        geometry = hashlib.sha256(
            self.ctx.model.geom_size.tobytes() + self.ctx.model.geom_pos.tobytes()
        ).hexdigest()
        if state.get("geometry_sha256", geometry) != geometry:
            raise ValueError("checkpoint geometry mismatch")
        self.ctx.restore(state["physics"])
        self.held = state["held"]
        self.artifacts = copy.deepcopy(state["artifacts"])
        self.observations = copy.deepcopy(state["observations"])
        self.arm.rotation = state["rotation"].copy()
        self.stroke = list(state["stroke"])
        self.stroke_runs = copy.deepcopy(state.get("stroke_runs", []))
        self.stroke_peak_forces = list(state.get("stroke_peak_forces", []))
        if "rng" in state:
            self.rng.bit_generator.state = copy.deepcopy(state["rng"])
        self.grasp_epoch = state.get("grasp_epoch", 0)
        self.active_candidate_id = state.get("active_candidate_id")
        self.decision_observation = copy.deepcopy(state.get("decision_observation"))
        self.sensor_manifest = copy.deepcopy(state.get("sensor_manifest"))
        self.dirty_state = copy.deepcopy(state.get("dirty_state"))
        self.stage_passes = copy.deepcopy(state.get("stage_passes", {}))
        self.failure_reasons = list(state.get("failure_reasons", []))
        self.perception_backend = state.get("perception_backend", self.perception_backend)
        self.visual_calibration = copy.deepcopy(state.get("visual_calibration"))
        for name in ("stage_targets", "execution_relocalizations"):
            if state.get(name) is None:
                if hasattr(self, name):
                    delattr(self, name)
            else:
                setattr(self, name, copy.deepcopy(state[name]))
        if self.dirty_state is not None:
            from simbench.value.cleaning import restore_visual
            restore_visual(self)

    def external_force(self, part):
        bid = self.ctx.body_id(part)
        total = 0.0
        for i, c in enumerate(self.ctx.data.contact):
            bodies = self.ctx.model.geom_bodyid[[c.geom1, c.geom2]]
            if bid not in bodies:
                continue
            other = int(bodies[0] if bodies[1] == bid else bodies[1])
            name = self.ctx.model.body(other).name
            if "finger" in name or name in ("eef", "right_gripper", "right_hand"):
                continue
            f = np.zeros(6)
            mujoco.mj_contactForce(self.ctx.model, self.ctx.data, i, f)
            total += max(0.0, float(f[0]))
        return total

    def stream_part(self, part, target, speed, force_limit):
        from .skills_v12 import control_position
        start = control_position(self, part).copy()
        evaluated_start = self.ctx.obj_pos(part).copy()
        target = np.asarray(target, float)
        offset = self.ctx.eef_pos() - start
        # In the completed product the handle is the physical user interface
        # for the carriage.  The handle may be the held body while the
        # carriage is the measured sliding body; do not require a second
        # grasp on the carriage boss (which is intentionally covered by the
        # installed handle).
        grasp_part = self.held if (self.held == "handle" and part == "carriage") else part
        peak = 0.0
        duration = max(1.0, 1.9 * np.linalg.norm(target - start) / speed)
        for t in np.linspace(0, 1, max(2, int(duration / self.ctx.control_dt))):
            s = t * t * t * (10 - 15 * t + 6 * t * t)
            self.arm.servo(start + s * (target - start) + offset)
            peak = max(peak, self.external_force(part))
            if peak > force_limit:
                return Result(False, {"peak_force_n": peak}, "contact overload")
            if not self.ctx.grasp_contacts(grasp_part)["held"]:
                return Result(
                    False,
                    {
                        "contact": self.ctx.grasp_contacts(grasp_part),
                        "held_part": grasp_part,
                        "object": self.ctx.obj_pos(part),
                        "axis": self.ctx.obj_axis(part),
                        "eef": self.ctx.eef_pos(),
                        "peak_force_n": peak,
                    },
                    "grasp lost",
                )
        self.hold(0.35)
        error = float(np.linalg.norm(self.ctx.obj_pos(part) - target))
        tracking_tolerance = .004 if str(part).startswith("pin_") else .0015
        if getattr(self, "functional_acceptance_v12", False):
            requested = target - start
            length = float(np.linalg.norm(requested))
            actual_delta = self.ctx.obj_pos(part) - evaluated_start
            progress = float(np.dot(actual_delta, requested / max(length, 1e-9)))
            # Motion completion is useful progress under the force/grasp
            # guards; the separate product engagement/stroke checks follow.
            return Result(progress >= .7 * length, dict(requested_travel_m=length,
                measured_progress_m=progress, peak_force_n=peak,
                measurement_source="independent_simulator_motion_evaluator"),
                "insufficient physical motion progress")
        return Result(
            error < tracking_tolerance,
            {
                "error_m": error,
                "travel_m": float(np.linalg.norm(self.ctx.obj_pos(part) - start)),
                "peak_force_n": peak,
                "tracking_tolerance_m": tracking_tolerance,
            },
            "contact motion tracking residual",
        )

    @skill(
        "observe_parts",
        "检测零件",
        "perception",
        produces="observations",
        implementation="sim_pose_sensor",
        effects=("seen:all",),
    )
    def observe_parts(self, required_parts=None):
        if getattr(self, "strict_rgbd_v12", False):
            if not self.decision_observation or self.decision_observation.get("backend") != "rgbd_geometry":
                return Result(False, reason="V12 requires RGB-D; simulator-pose detection is prohibited")
            # A detect atom at an execution boundary must acquire new pixels.
            # Reusing the proposal's frozen supply observation after placing
            # the stop would report its old supply holes as the installed ones.
            from simbench.value.stage_v7 import refresh_visual_observation
            refresh_visual_observation(self, parts=self.parts)
        if self.decision_observation is not None:
            requested = required_parts if required_parts is not None else getattr(self, "_required_observation_parts", None)
            required = tuple(requested) if requested is not None else tuple(self.parts)
            missing = [p for p in required if p not in self.observations]
            if missing:
                return Result(False, {"missing": missing, "backend": self.decision_observation.get("backend")},
                              "RGB-D detection invalid or occluded")
            backend = self.decision_observation.get("backend", "simulated_sensor_proxy")
            noise_std = self.decision_observation.get("config", {}).get("position_noise_std_m")
        else:
            self.observations = {
                p: [
                    self.ctx.obj_pos(p) + self.rng.normal(0, self.noise, 3)
                    for _ in range(5)
                ]
                for p in self.parts
            }
            backend = "legacy simulator pose observations"
            noise_std = self.noise
        return Result(
            metrics={
                "detected": list(self.observations),
                "backend": backend,
                "noise_std_m": noise_std,
                "frozen": self.decision_observation is not None and not getattr(self, "strict_rgbd_v12", False),
                "fresh_rgbd_capture": bool(getattr(self, "strict_rgbd_v12", False)),
                "observation_sha256": (self.decision_observation or {}).get("sha256"),
            }
        )

    @skill(
        "estimate_pose",
        "估计零件位姿",
        "perception",
        ("seen",),
        produces="pose",
        implementation="robust_estimator",
    )
    def estimate_pose(self, part, as_="pose"):
        if getattr(self, "strict_rgbd_v12", False):
            observation = self.decision_observation or {}
            row = observation.get("objects", {}).get(part, {})
            quaternion = np.asarray(row.get("quat_wxyz") if row.get("quat_wxyz") is not None else [], float)
            if (observation.get("backend") != "rgbd_geometry" or not row.get("valid")
                    or row.get("position_m") is None or quaternion.shape != (4,)
                    or not np.isfinite(quaternion).all()):
                return Result(False, {"part": part}, "V12 requires a valid RGB-D pose; oracle fallback is prohibited")
        if part not in self.observations:
            return Result(False, {"part": part}, "no valid visual estimate")
        xyz = np.median(self.observations[part], axis=0)
        if self.decision_observation is not None:
            quat = np.asarray(self.decision_observation["objects"][part]["quat_wxyz"], dtype=float)
            uncertainty = {
                "position_noise_std_m": self.decision_observation.get("config", {}).get("position_noise_std_m"),
                "yaw_noise_std_rad": self.decision_observation.get("config", {}).get("yaw_noise_std_rad"),
            }
        else:
            quat = self.ctx.obj_pose(part)[1]
            uncertainty = {"position_noise_std_m": self.noise}
        self.artifact(as_, "pose", part, xyz=xyz, quat=quat, uncertainty=self.noise)
        self.artifacts[as_]["uncertainty"] = uncertainty
        return Result(metrics={"position_m": xyz, "quaternion_wxyz": quat, "uncertainty": uncertainty})

    @skill(
        "observe_execution_pose",
        "执行期物体跟踪观测",
        "perception",
        ("seen",),
        produces="pose",
        implementation="simulated_execution_feedback",
    )
    def observe_execution_pose(self, part, as_="pose"):
        """Refresh one pose through the execution-feedback sensor boundary.

        The decision observation is intentionally frozen for candidate scoring.
        After an object has been placed, however, the controller may use a
        separate feedback observation to reacquire the installed object.  This
        method is the explicit simulated sensor-proxy boundary for that use;
        callers do not read a hidden pose directly into the value model.
        """
        if part not in self.parts:
            return Result(False, reason=f"unknown execution-feedback part: {part}")
        if getattr(self, "strict_rgbd_v12", False) and self.perception_backend != "rgbd_geometry":
            return Result(False, reason="V12 execution localization requires RGB-D; oracle fallback is prohibited")
        if self.perception_backend == "rgbd_geometry":
            from simbench.value.stage_v7 import refresh_visual_observation
            observation = refresh_visual_observation(self, parts=self.parts)
            row = observation.get("objects", {}).get(part, {})
            if not row.get("valid") or row.get("position_m") is None:
                return Result(False, {"part": part, "backend": "rgbd_geometry", "row": row},
                              "execution RGB-D re-observation invalid")
            xyz = np.asarray(row["position_m"], dtype=float)
            quat = np.asarray(row["quat_wxyz"], dtype=float)
            self.observations[part] = [xyz.copy() for _ in range(5)]
            uncertainty = {"backend": "rgbd_geometry", "quality": row.get("quality"),
                           "fit_residual_m": row.get("fit_residual_m"), "observation_age_s": 0.0}
            self.artifact(as_, "pose", part, xyz=xyz, quat=quat, uncertainty=uncertainty,
                          source="execution_rgbd_geometry")
            return Result(metrics={"position_m": xyz, "quaternion_wxyz": quat,
                                   "uncertainty": uncertainty,
                                   "frozen_decision_observation": False})
        xyz = self.ctx.obj_pos(part).copy()
        quat = self.ctx.obj_pose(part)[1].copy()
        uncertainty = {
            "backend": "simulated_execution_feedback",
            "position_noise_std_m": 0.0005,
            "yaw_noise_std_rad": float(np.deg2rad(0.5)),
        }
        self.artifact(as_, "pose", part, xyz=xyz, quat=quat,
                      uncertainty=uncertainty, source="execution_feedback")
        return Result(metrics={"position_m": xyz, "quaternion_wxyz": quat,
                               "uncertainty": uncertainty,
                               "frozen_decision_observation": False})

    @skill(
        "propose_grasps",
        "生成抓取候选",
        "planning",
        ("artifact:pose",),
        produces="grasps",
        implementation="geometry_and_ik",
    )
    def propose_grasps(self, part, artifact="pose", as_="grasps", yaws=None, height_offset=0.0, width=None,
                       yaw_frame="world", center_offset=None):
        pose = self.artifacts[artifact]
        dz, reference_width = self.grasp_specs[part]
        explicit_width = width
        if explicit_width is not None and (not np.isfinite(explicit_width) or not 0 < explicit_width <= .079):
            raise ValueError("grasp width outside actual finger actuator envelope")
        width = reference_width
        candidates = []
        unresolved = []
        pose_id = hashlib.sha256(
            np.asarray(pose["xyz"]).tobytes() + np.asarray(pose["quat"]).tobytes()
        ).hexdigest()[:12]
        if not np.isfinite(height_offset) or abs(height_offset) > 0.025:
            raise ValueError("invalid grasp height offset")
        if yaw_frame not in ("world", "object"):
            raise ValueError("grasp yaw_frame must be world or object")
        center_offset = np.zeros(3) if center_offset is None else np.asarray(center_offset, dtype=float)
        if (center_offset.shape != (3,) or not np.isfinite(center_offset).all()
                or np.linalg.norm(center_offset) > .04 or abs(center_offset[2]) > 1.e-12):
            raise ValueError("grasp center_offset must be a finite in-plane vector inside the 40-mm envelope")
        reference_yaw = 0.
        offset_world = center_offset.copy()
        if yaw_frame == "object":
            quat=np.asarray(pose["quat"],float)
            if quat.shape!=(4,) or not np.isfinite(quat).all() or np.linalg.norm(quat)<1e-8:
                raise ValueError("object-relative grasp requires a valid visual orientation")
            object_rotation = Rotation.from_quat(quat[[1,2,3,0]])
            reference_yaw=float(object_rotation.as_euler("xyz")[2])
            offset_world = object_rotation.apply(center_offset)
        for relative_yaw in ((0.0, np.pi / 2) if yaws is None else yaws):
            yaw = float(relative_yaw) + reference_yaw
            xyz = pose["xyz"] + offset_world + np.array([0, 0, dz + height_offset])
            try:
                # Width depends on the box face; round parts keep their diameter.
                w = (width if part not in ("carriage", "end_stop")
                     or np.isclose(np.sin(float(yaw)), 0.0, atol=1e-6)
                     else {"carriage": 0.044, "end_stop": 0.022}[part])
                if getattr(self, "strict_rgbd_v12", False) and part in ("carriage", "end_stop"):
                    quat = np.asarray(pose["quat"], float)
                    visual_yaw = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_euler("xyz")[2]
                    relative = float(yaw) - float(visual_yaw)
                    across = {"carriage": .044, "end_stop": .022}[part]
                    w = float(abs(np.cos(relative)) * width + abs(np.sin(relative)) * across)
                if explicit_width is not None:
                    w = float(explicit_width)
                checked = {}
                if getattr(self, "strict_rgbd_v12", False) and part.startswith("pin_"):
                    from .grasp_v12 import plan_grasp
                    checked = plan_grasp(self, part, pose, xyz, yaw, w)
                    q = checked.pop("q_hover")
                else:
                    q = self.arm.ik_with_restarts(xyz + [0, 0, 0.10], down(yaw))
                candidates.append(
                    dict(
                        id=f"{part}:yaw:{yaw:.6f}:pose:{pose_id}",
                        xyz=xyz,
                        yaw=yaw,
                        center_offset=center_offset.tolist(),
                        center_offset_frame=yaw_frame,
                        width=w,
                        q_hover=q,
                        cost=float(np.linalg.norm(q - self.ctx.arm_qpos) + 2 * w),
                        **checked,
                    )
                )
            except ValueError as exc:
                unresolved.append(
                    dict(
                        id=f"{part}:yaw:{yaw:.6f}:pose:{pose_id}",
                        xyz=xyz,
                        yaw=yaw,
                        status="unknown",
                        reason=str(exc),
                    )
                )
        self.artifact(as_, "grasps", part, candidates=candidates, unresolved=unresolved)
        return Result(
            bool(candidates or unresolved),
            {
                "feasible_candidates": len(candidates),
                "unknown_candidates": len(unresolved),
            },
            "no reachable grasp" if not candidates else "",
        )

    @skill(
        "select_grasp",
        "选择可达抓取",
        "planning",
        ("artifact:grasps",),
        produces="grasp",
        implementation="candidate_scoring",
        output_bindings=(("part", "part"), ("id", "candidate_id")),
    )
    def select_grasp(
        self, part, artifact="grasps", as_="grasp", index=None, candidate_id=None
    ):
        candidates = self.artifacts[artifact]["candidates"]
        if not candidates:
            return Result(False, reason="no materialized grasp within solver budget")
        if candidate_id is not None:
            matches = [i for i, c in enumerate(candidates) if c["id"] == candidate_id]
            if not matches:
                return Result(False, reason="unknown grasp candidate id")
            index = matches[0]
        if index is None:
            index = min(range(len(candidates)), key=lambda i: candidates[i]["cost"])
        if index < 0 or index >= len(candidates):
            return Result(False, reason="grasp index out of range")
        candidate = candidates[index]
        self.artifact(as_, "grasp", part, **candidate)
        return Result(
            metrics={
                "chosen_index": index,
                "yaw_rad": candidate["yaw"],
                "jaw_width_m": candidate["width"],
            }
        )

    @skill(
        "plan_transfer",
        "规划避让搬运路径",
        "planning",
        produces="joint_path",
        implementation="waypoint_ik",
    )
    def plan_transfer(
        self,
        target,
        as_="transfer",
        clearance=0.98,
        yaw=0.0,
        candidate_id=None,
        grasp_artifact="grasp",
    ):
        from .candidates import transfer_routes, record_continuation

        grasp = self.artifacts.get(grasp_artifact)
        routes, checks = transfer_routes(
            self, target, clearance, yaw, grasp, grasp_artifact
        )
        # Even when the budget finds no solution, unresolved solver attempts survive.
        self.artifact(
            as_ + "_candidates",
            "joint_paths",
            self.held,
            candidates=routes,
            attempts=checks,
        )
        ready = [r for r in routes if r["status"] == "necessary_pass"]
        if not ready:
            return Result(
                False,
                {"checks": checks, "status": "unknown"},
                "no checked transfer route within budget",
            )
        chosen = (
            next((r for r in ready if r["id"] == candidate_id), None)
            if candidate_id
            else min(ready, key=lambda r: r["cost"])
        )
        if chosen is None:
            return Result(False, reason="requested route is not materialized")
        self.artifacts[as_] = copy.deepcopy(chosen["path"])
        record_continuation(self, routes, chosen)
        return Result(
            metrics={
                "waypoints": len(chosen["path"]["joints"]),
                "clearance_m": chosen["clearance"],
                "collision_checks": checks,
                "candidate_count": len(routes),
                "selected_id": chosen["id"],
            }
        )

    @skill(
        "plan_linear",
        "规划直线操作路径",
        "planning",
        produces="cartesian_path",
        implementation="cartesian_ik",
    )
    def plan_linear(self, target, as_="linear", speed=0.05):
        target = np.asarray(target, float)
        q = self.ctx.arm_qpos.copy()
        for xyz in np.linspace(self.ctx.eef_pos(), target, 12):
            q = self.arm.ik(xyz, seed=q)
        self.artifact(
            as_,
            "cartesian_path",
            start=self.ctx.eef_pos(),
            rotation=self.arm.rotation.copy(),
            target=target,
            speed=speed,
        )
        return Result(
            metrics={
                "length_m": float(np.linalg.norm(target - self.ctx.eef_pos())),
                "ik_samples": 12,
            }
        )

    @skill("execute_joint_path", "执行关节搬运路径", "motion", ("artifact:joint_path",))
    def execute_joint_path(self, artifact="transfer"):
        plan = self.artifacts[artifact]
        if np.max(np.abs(plan["start_q"] - self.ctx.arm_qpos)) > 0.04:
            return Result(False, reason="stale joint path")
        for q in plan["joints"]:
            if not self.arm.execute_joint(q):
                return Result(False, reason="joint servo tracking error")
        self.arm.rotation = plan["rotation"].copy()
        err = float(np.linalg.norm(plan["target"] - self.ctx.eef_pos()))
        return Result(
            err < 0.002, {"target_error_m": err}, "transfer did not reach target"
        )

    @skill(
        "execute_cartesian_path",
        "执行直线操作路径",
        "motion",
        ("artifact:cartesian_path",),
    )
    def execute_cartesian_path(self, artifact="linear"):
        plan = self.artifacts[artifact]
        if np.linalg.norm(plan["start"] - self.ctx.eef_pos()) > 0.003:
            return Result(False, reason="stale Cartesian path")
        if np.linalg.norm(plan["rotation"] - self.ctx.eef_mat()) > 0.04:
            return Result(False, reason="stale Cartesian orientation")
        ok = self.arm.move(
            plan["target"], rotation=plan["rotation"], speed=plan["speed"]
        )
        return Result(
            ok,
            {
                "target_error_m": float(
                    np.linalg.norm(plan["target"] - self.ctx.eef_pos())
                )
            },
            "linear servo tracking error" if not ok else "",
        )

    @skill("move_to", "移动", "motion", ("ownership",), legacy=False)
    def move_to(self, target=None, delta=None, part=None, speed=0.06, tol=0.001):
        if (target is None) == (delta is None) or speed <= 0 or tol <= 0:
            return Result(
                False, reason="provide one target or delta and positive speed/tolerance"
            )
        goal = np.asarray(target if target is not None else delta, dtype=float)
        if goal.shape != (3,) or not np.all(np.isfinite(goal)):
            return Result(False, reason="move requires three finite coordinates")
        if delta is not None:
            goal = self.ctx.eef_pos() + goal
        # Use the existing planner/checker before the existing Cartesian controller.
        q = self.ctx.arm_qpos.copy()
        joints = []
        for point in np.linspace(self.ctx.eef_pos(), goal, 12):
            q = self.arm.ik(point, seed=q)
            joints.append(q.copy())
        collision = self.arm.check_joint_path(joints, part)
        if not collision["valid"]:
            return Result(False, {"collision": collision}, "move path collision")
        ok = self.arm.move(goal, speed=speed, tol=tol)
        retained = part is None or self.ctx.grasp_contacts(part)["held"]
        return Result(
            bool(ok and retained),
            dict(
                target_error_m=float(np.linalg.norm(self.ctx.eef_pos() - goal)),
                held=retained,
                collision=collision,
            ),
            "move tracking or retention failed",
        )

    @skill(
        "place_object",
        "放置",
        "gripper",
        ("held",),
        effects=("held:empty",),
        obligations=("support contact before release", "settled pose after release"),
        legacy=False,
    )
    def place_object(self, part, target, tol=0.0015, settle=0.30, minimum_support=0.01,
                     acceptance="pose"):
        """Release an already supported part; moving there is the Move atom's job."""
        if (tol <= 0 or settle < 0 or minimum_support <= 0
                or acceptance not in ("pose", "pin_inserted", "support_only", "stable_supported")):
            return Result(False, reason="invalid placement tolerances")
        stable = acceptance == "stable_supported"
        if stable and str(part) != "end_stop":
            return Result(False, reason="stable_supported acceptance is only defined for end_stop")
        if stable:
            if getattr(self, "end_stop_place_acceptance_v12", None) != "stable_supported":
                return Result(False, reason="stable_supported requires the explicit V17-A local-mode flag")
            from .skills_v12 import control_position, functional_geometry
            control = control_position(self, part)
            region_ok, region_row = functional_geometry(str(part), control, np.asarray(target, float))
            pose = Result(bool(region_ok), {"acceptance": acceptance, **region_row,
                "control_position_m": np.asarray(control, float).tolist()},
                "" if region_ok else "end_stop outside declared mounting region before release")
        else:
            pose = self.inspect_seat(part, target, tol) if acceptance == "pose" else Result(True, {"acceptance": acceptance})
        if not pose.ok:
            return Result(
                False, pose.metrics, "move to the placement target before releasing"
            )
        handle_geometry_ready = bool(getattr(self, "functional_acceptance_v12", False)
                                     and str(part) == "handle" and acceptance == "pose" and pose.ok)
        pin_geometry_ready = None
        if acceptance == "pin_inserted":
            from simbench.value.pin_geometry import PinInsertionConfig, evaluate_pin_context
            hole_offset = (0., -.032 if part == "pin_left" else .032, 0.)
            pin_geometry_ready = evaluate_pin_context(
                self.ctx, part, fixture_part="end_stop", hole_offset_m=hole_offset,
                phase="inserted_while_held", released=False,
                touching_finger=True, config=getattr(self, "pin_insertion_config", PinInsertionConfig(required_depth_m=.006)),
            )
            if getattr(self, "functional_acceptance_v12", False) and not pin_geometry_ready.get("success"):
                return Result(False, {"pin_geometry_while_held": pin_geometry_ready},
                              "pin is supported outside the bore; refusing incorrect release")
        support = 0.0
        bid = self.ctx.body_id(part)
        for i, c in enumerate(self.ctx.data.contact):
            b1, b2 = map(int, self.ctx.model.geom_bodyid[[c.geom1, c.geom2]])
            if bid not in (b1, b2):
                continue
            other = b2 if b1 == bid else b1
            if "finger" in self.ctx.model.body(other).name:
                continue
            # Only an approximately vertical contact below the object can support it.
            if abs(c.frame[2]) < 0.5 or c.pos[2] >= self.ctx.obj_pos(part)[2]:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(self.ctx.model, self.ctx.data, i, force)
            support += max(0.0, float(force[0])) * abs(float(c.frame[2]))
        # A seated handle can press on the carriage boss through a ring
        # contact that has no downward-facing MuJoCo contact frame.  Preserve
        # the physical release guard by accepting the independent force
        # feedback as supporting evidence, but only for this known ring/post
        # interface and only after the caller's explicit press/seat action.
        support_force_feedback = 0.0
        if str(part) == "handle" and acceptance == "pose":
            support_force_feedback = float(self.external_force(part))
            if support < minimum_support and support_force_feedback >= 0.05:
                support = support_force_feedback
        # A pin already inside the finite guide is supported by the hole
        # geometry, even when a downward contact sample is absent at this
        # instant.  It is safe to attempt release only in that case; the
        # post-release geometry predicate remains mandatory and can fail if
        # gravity lets the pin fall out.
        if (support < minimum_support and not handle_geometry_ready
                and not (pin_geometry_ready and pin_geometry_ready.get("success"))):
            return Result(
                False,
                {"support_force_n": support,
                 "support_force_feedback_n": support_force_feedback},
                "no supporting contact; refusing air release",
            )
        # This call commits the same declared release effect even if the later
        # settled-pose check fails. Never leave a released object marked held.
        if self.stage_passes and str(part).startswith("pin_"):
            # Pins need a physical seating dwell before release. V9 already
            # pressed and verified support immediately before Place; another
            # higher-force press can wedge the jaws against the wide guide.
            if not (str(getattr(self, "task_version", "")).startswith("functional_assembly_v9")
                    or getattr(self, "functional_acceptance_v12", False)):
                self.call("press", part=part, target_z=float(target[2]), force_stop=4.0)
            self.hold(.20)
        elif acceptance == "pose":
            # Let a supported placement stop drifting before the jaws open.
            # The post-release pose check remains the only acceptance test.
            self.hold(.20)
        try:
            self.call("open_gripper")
        except SkillFailure:
            # v7 permits one declared physical release recovery: lift the
            # fingers a few millimetres and re-open.  This is not a teleport
            # or a label override; the subsequent support/pose check remains
            # mandatory. Legacy v5/v6 sessions retain the strict behavior.
            if not self.stage_passes:
                raise
            if (str(part).startswith("pin_")
                    and (str(getattr(self, "task_version", "")).startswith("functional_assembly_v9")
                         or getattr(self, "functional_acceptance_v12", False))):
                # Keep a seated pin stationary if a release needs one retry.
                # The older 18 mm lifting recovery could extract it.
                self.hold(.20)
                self.call("open_gripper")
            else:
                # Legacy recovery clears a fixture-limited jaw physically
                # before retrying the release.
                self.arm.move(self.ctx.eef_pos() + [0, 0, .018], speed=.025)
                self.call("open_gripper")
        # The physical open command is the ownership boundary.  Keeping the
        # bookkeeping field populated after a successful release lets later
        # motion/planning treat a released pin as still carried and obscures
        # real post-release drift.
        self.held = None
        self.hold(settle)
        if stable:
            from .skills_v12 import evaluate_end_stop_stable_support
            stable_ok, stable_row = evaluate_end_stop_stable_support(self, str(part), target, after_retreat=False)
            settled = Result(bool(stable_ok), stable_row,
                             "" if stable_ok else "released end_stop is not stably supported")
        else:
            settled = self.inspect_seat(part, target, tol) if acceptance == "pose" else Result(True, {"acceptance": acceptance})
        if (settled.ok and self.stage_passes and str(part).startswith("pin_")
                and not getattr(self, "functional_acceptance_v12", False)):
            # Rotate the opened fingers in place before the normal vertical
            # retraction. This clears residual side contact without sweeping
            # the already released pin across the fixture.
            self.arm.move(self.ctx.eef_pos(), down(np.pi / 2), speed=.03)
            self.hold(.10)
        return Result(
            settled.ok,
            {
                **settled.metrics,
                "support_force_n": support,
                "support_force_feedback_n": support_force_feedback,
                "pin_geometry_while_held": pin_geometry_ready,
                "handle_geometry_while_held": pose.metrics if handle_geometry_ready else None,
                "jaw_span_m": self.ctx.pad_span(),
            },
            "placement did not remain within tolerance" if acceptance == "pose"
            else "released end_stop is not stably supported" if stable
            else "release failed",
        )

    @skill("measure_value", "测量", "perception", produces="measurement", legacy=False)
    def measure_value(self, quantity, part=None, as_="measurement"):
        if quantity in ("position", "force", "clearance") and part is None:
            return Result(False, reason="this measurement requires an object")
        if quantity == "position":
            value, unit = self.ctx.obj_pos(part).copy(), "m"
        elif quantity == "force":
            value, unit = self.external_force(part), "N"
        elif quantity == "stroke":
            value, unit = (
                max(self.stroke) - min(self.stroke) if self.stroke else 0.0
            ), "m"
        elif quantity == "clearance":
            reading = self.measure_clearance(part)
            if "lateral_margin_m" not in reading.metrics:
                return reading
            value, unit = reading.metrics["lateral_margin_m"], "m"
        else:
            return Result(
                False, reason="quantity must be position, force, stroke or clearance"
            )
        self.artifact(
            as_,
            "measurement",
            part,
            quantity=quantity,
            value=value,
            unit=unit,
            observed_at=float(self.ctx.data.time),
        )
        return Result(metrics={"quantity": quantity, "value": value, "unit": unit})

    @skill(
        "inspect_measurement",
        "检查",
        "verification",
        ("artifact:measurement",),
        legacy=False,
    )
    def inspect_measurement(self, artifact="measurement", minimum=None, maximum=None):
        if minimum is None and maximum is None:
            return Result(False, reason="inspection requires a threshold")
        if minimum is not None and maximum is not None and minimum > maximum:
            return Result(False, reason="minimum exceeds maximum")
        reading = self.artifacts[artifact]
        value = np.asarray(reading["value"])
        ok = (minimum is None or np.all(value >= minimum)) and (
            maximum is None or np.all(value <= maximum)
        )
        return Result(
            bool(ok),
            dict(
                value=reading["value"],
                unit=reading["unit"],
                minimum=minimum,
                maximum=maximum,
            ),
            "measurement outside required bounds",
        )

    @skill("open_gripper", "张开夹爪 / 释放", "gripper", effects=("held:empty",))
    def open_gripper(self):
        command_ok = self.arm.open()
        span = self.ctx.pad_span()
        # Small pins can be released while the neighbouring fixture limits
        # nominal pad span.  The actual contract is contact loss plus a
        # meaningful opening command, not a magic span threshold alone.
        contact_free = True
        if self.held is not None and self.held in self.parts:
            contact_free = not bool(self.ctx.grasp_contacts(self.held)["held"])
        # A large jaw span alone is not a release.  Require the bilateral pad
        # contacts to be gone whenever a part is held; place_object may then
        # perform its explicit small recovery lift.  This prevents the next
        # retreat motion from dragging an end-stop or pin away while the
        # command has merely reached its actuator limit.
        ok = bool(command_ok and contact_free) if self.held is not None else bool(command_ok)
        if self.held is not None and contact_free and span > .020:
            ok = True
        return Result(ok, {"jaw_span_m": span, "contact_free": contact_free})

    @skill("approach", "接近抓取位姿", "transition", ("empty", "artifact:grasp"))
    def approach(self, part, artifact="grasp", strategy="cartesian"):
        goal = self.artifacts[artifact]
        if getattr(self, "strict_rgbd_v12", False) and "approach_joints_v12" in goal:
            if np.max(np.abs(self.ctx.arm_qpos-goal["q_hover"])) > .05:
                return Result(False, reason="continuous grasp hover branch is stale")
            ok=self.arm.execute_joint_waypoints(goal["approach_joints_v12"])
            self.arm.rotation=down(goal["yaw"])
            return Result(ok,dict(position_error_m=float(np.linalg.norm(goal["xyz"]-self.ctx.eef_pos())),
                                  strategy="checked continuous grasp branch"),"approach tracking error" if not ok else "")
        if strategy == "joint_checked_v10":
            try:
                q = self.arm.ik_with_restarts(goal["xyz"], down(goal["yaw"]))
                verdict = self.arm.check_joint_path([q])
            except ValueError as exc:
                return Result(False, {"strategy": strategy}, str(exc))
            if not verdict["valid"]:
                return Result(False, {"strategy": strategy, "collision": verdict},
                              "checked joint approach collision")
            ok = self.arm.execute_joint(q)
            self.arm.rotation = down(goal["yaw"])
            return Result(ok, {"strategy": strategy,
                               "position_error_m": float(np.linalg.norm(goal["xyz"] - self.ctx.eef_pos()))},
                          "checked joint approach error" if not ok else "")
        if strategy != "cartesian":
            return Result(False, reason="unknown approach strategy")
        try:
            ok = self.arm.move(goal["xyz"], down(goal["yaw"]), speed=0.045)
            if not ok:
                # The first servo can finish just outside the 1-mm contract
                # after a long hover descent.  Re-solve from the attained
                # state and perform a slower bounded correction before failing.
                ok = self.arm.move(goal["xyz"], down(goal["yaw"]), speed=0.02)
            if not ok:
                # A second Cartesian tracking miss is still a reachable
                # endpoint candidate. Use the same whole-robot collision
                # check required by the explicit joint approach strategy.
                q = self.arm.ik_with_restarts(goal["xyz"], down(goal["yaw"]))
                verdict = self.arm.check_joint_path([q])
                if verdict["valid"]:
                    ok = self.arm.execute_joint(q)
                    self.arm.rotation = down(goal["yaw"])
        except ValueError as exc:
            # A Cartesian IK continuation can stall near a redundant-arm
            # singularity despite a reachable checked endpoint. Try one
            # independently checked joint continuation before declaring the
            # grasp approach unreachable.
            if not str(exc).startswith("IK unreachable:"):
                raise
            q = self.arm.ik_with_restarts(goal["xyz"], down(goal["yaw"]))
            verdict = self.arm.check_joint_path([q])
            if not verdict["valid"]:
                raise
            ok = self.arm.execute_joint(q)
            self.arm.rotation = down(goal["yaw"])
        return Result(
            ok,
            {
                "position_error_m": float(
                    np.linalg.norm(goal["xyz"] - self.ctx.eef_pos())
                )
            },
            "approach error" if not ok else "",
        )

    @skill(
        "close_gripper", "接触反馈夹持", "gripper", ("empty",), effects=("held:part",)
    )
    def close_gripper(self, part, force=3.0):
        contact = self.arm.close(part, force=force)
        if contact["held"] and getattr(self, "strict_rgbd_v12", False):
            from .skills_v12 import bind_grasp_observation
            bind_grasp_observation(self, part)
        return Result(
            contact["held"],
            contact,
            "bilateral part contact missing" if not contact["held"] else "",
        )

    @skill("verify_grasp", "验证夹持接触", "verification", ("held",))
    def verify_grasp(self, part):
        contact = self.ctx.grasp_contacts(part)
        return Result(contact["held"], contact)

    @skill("lift", "保持夹持并抬升", "transition", ("held",))
    def lift(self, part, height=0.10, speed=0.06):
        if not np.isfinite(speed) or not .005 <= speed <= .10:
            raise ValueError("lift speed outside declared controller envelope")
        start = self.ctx.obj_pos(part).copy()
        goal = self.ctx.eef_pos() + [0, 0, height]
        grasp=self.artifacts.get("grasp",{})
        if (getattr(self, "strict_rgbd_v12", False) and part.startswith("pin_")
                and "lift_joints_v12" in grasp
                and float(grasp.get("withdrawal_progress_m",0.)) < .10-1e-8):
            # Recheck the stored branch with the now observed grasp relation;
            # the holder remains a physical obstacle during withdrawal.
            from .grasp_v12 import withdrawal_slice
            path,progress=withdrawal_slice(grasp,height)
            registration=self.held_visual_transforms_v12[part]
            transform=dict(position=registration["local_position"], rotation=registration["local_rotation"])
            verdict=self.arm.check_joint_path(path,held=part,step=.02,held_transform=transform)
            if not verdict["valid"]:
                return Result(False,dict(collision=verdict),"bound shaft withdrawal collision")
            ok=self.arm.execute_joint_waypoints(path)
            delta=float(self.ctx.obj_pos(part)[2]-start[2])
            held=self.ctx.grasp_contacts(part)["held"]
            if ok and held and delta>.75*height:
                grasp["withdrawal_progress_m"]=progress
            return Result(bool(ok and held and delta>.75*height),
                dict(object_lift_m=delta,held=held,continuous_branch=True,collision=verdict,
                     cumulative_withdrawal_m=progress,attachment_source="RGB-D grasp registration and encoder FK",
                     collision_geometry="digital twin environment; holder not exempted"),
                "checked shaft withdrawal or retention failed")
        try:
            ok = self.arm.move(goal, speed=float(speed))
            if not ok and self.ctx.grasp_contacts(part)["held"]:
                # Contact breakaway and loaded compliance can leave the first
                # lift short while the bilateral grasp remains intact.  Retry
                # the same absolute checked target slowly from the attained
                # state; the command does not alter or reattach the object.
                ok = self.arm.move(goal, speed=max(.005, min(.02, float(speed))))
        except ValueError as exc:
            if not str(exc).startswith("IK unreachable:"):
                raise
            grasp = self.artifacts.get("grasp", {})
            seeds = []
            if grasp.get("part") == part and "q_hover" in grasp:
                seeds.append(np.asarray(grasp["q_hover"], dtype=float))
            rng = np.random.default_rng(17)
            for _ in range(24):
                seeds.append(np.clip(HOME + rng.normal(0, .3, 7),
                                     self.arm.limits[:, 0] + .02,
                                     self.arm.limits[:, 1] - .02))
            q = None
            collision = None
            for seed in seeds:
                try:
                    candidate = self.arm.ik(goal, self.arm.rotation, seed=seed)
                except ValueError:
                    continue
                collision = self.arm.check_joint_path([candidate], held=part)
                if collision["valid"]:
                    q = candidate
                    break
            if q is None:
                return Result(False, {"collision": collision, "ik_seeds_checked": len(seeds)},
                              "no checked lift fallback path")
            ok = self.arm.execute_joint(q)
        delta = float(self.ctx.obj_pos(part)[2] - start[2])
        held = self.ctx.grasp_contacts(part)["held"]
        return Result(
            bool(ok and held and delta > (.75 * height if getattr(self, "functional_acceptance_v12", False) else height - .003)),
            {"object_lift_m": delta, "held": held},
            "lift or retention failed",
        )

    @skill("lower", "受控下移至预接触高度", "transition", ("held",))
    def lower(self, part, height=0.025):
        ok = self.arm.move(self.ctx.eef_pos() - [0, 0, height], speed=0.035)
        return Result(
            bool(ok and self.ctx.grasp_contacts(part)["held"]),
            {"distance_m": height},
            "lowering failed",
        )

    @skill("retreat", "空夹爪退出操作区", "transition", ("empty",))
    def retreat(self, height=0.10):
        ok = self.arm.move(self.ctx.eef_pos() + [0, 0, height], speed=0.1)
        return Result(ok, {"height_m": self.ctx.eef_pos()[2]})

    @skill("home", "机械臂回位", "transition", ("empty",))
    def home(self):
        ok = self.arm.execute_joint(HOME)
        self.arm.rotation = down()
        return Result(
            ok, {"joint_error_rad": float(np.max(np.abs(self.ctx.arm_qpos - HOME)))}
        )

    @skill("orient_wrist", "调整末端朝向", "motion")
    def orient_wrist(self, yaw=0.0):
        if self.ctx.eef_pos()[2] < 0.94:
            return Result(False, reason="wrist rotation requires clearance height")
        ok = self.arm.move(self.ctx.eef_pos(), down(yaw), speed=0.06)
        return Result(ok, {"yaw_rad": yaw})

    @skill("align_axis", "持物轴线精对准", "contact", ("held",))
    def align_axis(self, part, target, tolerance=0.0007):
        from .skills_v12 import control_position
        target = np.asarray(target, float)
        for _ in range(6):
            error = target - control_position(self, part)
            if np.linalg.norm(error) < tolerance:
                break
            self.arm.move(self.ctx.eef_pos() + error, speed=0.035, tol=0.0004)
        error = float(np.linalg.norm(target - control_position(self, part)))
        return Result(
            error < tolerance and self.ctx.grasp_contacts(part)["held"],
            {"object_error_m": error},
            "alignment residual",
        )

    @skill(
        "plan_insertion",
        "规划接触插入参数",
        "planning",
        ("held",),
        produces="insertion",
        implementation="geometry_constraints",
    )
    def plan_insertion(
        self, part, target, axis=(0, 0, -1), as_="insert", speed=0.008, force_limit=12.0,
        pin_command_depth_m=None, pin_press_extra_m=None, hole_entry_m=None,
    ):
        target = np.asarray(target, float)
        axis = np.asarray(axis, float)
        if (target.shape != (3,) or axis.shape != (3,) or not np.isfinite(target).all()
                or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-9):
            return Result(False, reason="finite contact target and nonzero insertion axis required")
        axis /= np.linalg.norm(axis)
        self.arm.ik(self.arm.part_target(part, target))
        recovery = {}
        if getattr(self, "functional_acceptance_v12", False) and str(part).startswith("pin_"):
            if any(v is not None for v in (pin_command_depth_m, pin_press_extra_m, hole_entry_m)):
                if any(v is None for v in (pin_command_depth_m, pin_press_extra_m, hole_entry_m)):
                    return Result(False, reason="pin depth, press allowance and observed entry must be declared together")
                depth, extra = float(pin_command_depth_m), float(pin_press_extra_m)
                entry = np.asarray(hole_entry_m, float)
                upper = float(self.planning_cad["pin_head_seated_total_depth_m"])
                if (not np.isfinite([depth, extra, upper]).all() or not 0 < depth <= upper
                        or extra < 0 or depth + extra > upper + 1e-9
                        or entry.shape != (3,) or not np.isfinite(entry).all()):
                    return Result(False, reason="pin total feed exceeds declared CAD depth or lacks observed entry")
                recovery = dict(command_depth_m=depth, press_extra_m=extra,
                    maximum_total_depth_m=depth + extra, absolute_depth_limit_m=upper,
                    hole_entry_m=entry.tolist(), recovery_feed_depth_m=depth,
                    recovery_feed_upper_bound_m=depth + extra,
                    recovery_feed_source="explicit executable plan ports",
                    recovery_feed_reference="observed receiver entry and encoder/FK pin estimate")
            else:
                # Archived shallow-insertion callers retain their old contract;
                # the new two-receiver planner always supplies explicit ports.
                depth = float(self.planning_cad.get("pin_command_insertion_depth_m", .008))
                if not 0 < depth <= self.pin_insertion_config.guide_length_m * .5:
                    return Result(False, reason="CAD recovery feed exceeds half the real receiver length")
                recovery = dict(recovery_feed_depth_m=depth,
                    recovery_feed_source="planning_cad.pin_command_insertion_depth_m",
                    recovery_feed_reference="encoder position after sustained force-guided entry detection",
                    recovery_feed_upper_bound_m=self.pin_insertion_config.guide_length_m * .5)
        self.artifact(
            as_,
            "insertion",
            part,
            target=target,
            axis=axis,
            speed=speed,
            force_limit=force_limit,
            **recovery,
        )
        return Result(
            metrics={
                "target_m": target,
                "axis": axis,
                "speed_m_s": speed,
                "force_limit_n": force_limit,
                **recovery,
            }
        )

    @skill(
        "plan_wipe",
        "生成表面擦拭路径",
        "planning",
        ("held", "capability:wipe"),
        produces="wipe_path",
        legacy=False,
        obligations=(
            "surface geometry, tool clearance and force tracking require simulation",
        ),
    )
    def plan_wipe(
        self,
        part,
        center,
        halfspan=(0.055, 0.006),
        surface="guide_base",
        height=0.822,
        duration=14.0,
        as_="wipe",
    ):
        from .wiping import plan

        return plan(self, part, center, halfspan, surface, height, duration, as_)

    @skill(
        "wipe_surface",
        "学习轨迹接触擦拭",
        "contact",
        ("held", "capability:wipe", "artifact:wipe_path"),
        implementation="trajectory_imitation_with_force_feedback",
        legacy=False,
        obligations=(
            "actual surface contact, force limit and coverage require simulation",
        ),
    )
    def wipe_surface(
        self,
        part,
        artifact="wipe",
        target_force=1.5,
        force_limit=12.0,
        minimum_coverage=0.9,
    ):
        from .wiping import execute

        return execute(
            self, part, artifact, target_force, force_limit, minimum_coverage
        )

    @skill("slide_insert", "沿导轨约束插入", "contact", ("held", "artifact:insertion"))
    def slide_insert(self, part, artifact="insert"):
        plan = self.artifacts[artifact]
        result = self.stream_part(
            part, plan["target"], plan["speed"], plan["force_limit"]
        )
        # A pin can be effectively seated while its body origin remains away
        # from the nominal target because the shaft is already inside the
        # guide.  Do not turn that physically valid state into a failure based
        # on world-coordinate residual; require the independent hole geometry
        # predicate instead.  This branch is never used for other parts and
        # never changes a failed grasp/contact into success.
        if str(part).startswith("pin_") and not result.ok and self.held == part:
            from simbench.value.pin_geometry import PinInsertionConfig, evaluate_pin_context
            hole_offset = (0., -.032 if part == "pin_left" else .032, 0.)
            metrics = evaluate_pin_context(
                self.ctx, part, fixture_part="end_stop", hole_offset_m=hole_offset,
                phase="inserted_while_held", released=False,
                touching_finger=True, config=getattr(self, "pin_insertion_config", PinInsertionConfig(required_depth_m=.006)),
            )
            if metrics.get("success"):
                return Result(True, {**result.metrics, "insertion_geometry": metrics,
                                     "nominal_tracking_residual_ignored": True})
        return result

    @skill("guarded_descent", "接触保护下降", "contact", ("held",))
    def guarded_descent(self, part, target_z, force_stop=2.0, speed=0.006):
        from .skills_v12 import control_position
        start = control_position(self, part).copy()
        command = self.ctx.eef_pos().copy()
        peak = 0.0
        for _ in range(600):
            force = self.external_force(part)
            peak = max(peak, force)
            if force >= force_stop:
                break
            if control_position(self, part)[2] <= target_z + 0.0003:
                break
            command[2] -= speed * self.ctx.control_dt
            self.arm.servo(command)
            if not self.ctx.grasp_contacts(part)["held"]:
                return Result(False, reason="grasp lost during descent")
        self.hold(0.2)
        z = float(control_position(self, part)[2])
        # A guarded approach can finish against a stiff contact with a
        # residual load above the next press skill's stopping threshold.
        # Retract in the already checked vertical direction until the load
        # relaxes; the press then starts from an unloaded, held state. This
        # uses the same force/encoder feedback for every held object.
        unloaded = True
        unload_travel = 0.0
        if peak >= force_stop and self.external_force(part) >= .15 * force_stop:
            unloaded = False
            release_z = float(control_position(self, part)[2])
            for _ in range(100):
                if self.external_force(part) < .15 * force_stop:
                    unloaded = True
                    break
                command[2] += .00002
                self.arm.servo(command)
                unload_travel = float(control_position(self, part)[2] - release_z)
                if not self.ctx.grasp_contacts(part)["held"]:
                    return Result(False, reason="grasp lost during contact unload")
            if not unloaded and self.external_force(part) < .15 * force_stop:
                unloaded = True
        return Result(
            (z <= target_z + 0.001 or peak >= force_stop) and unloaded,
            {
                "z_m": z,
                "peak_contact_n": peak,
                "travel_m": float(start[2] - z),
                "contact_unloaded": unloaded,
                "unload_travel_m": unload_travel,
                "stopped_by": (
                    "contact"
                    if peak >= force_stop
                    else "height" if z <= target_z + 0.001 else "budget"
                ),
            },
            "guarded contact could not unload" if not unloaded else "descent exhausted",
        )

    def _press_functional_v12(self, part, target_z, force_stop):
        """Bounded contact establishment; final placement/stroke owns acceptance."""
        from .skills_v12 import control_position, evaluate_functional_seat
        if (part == "end_stop"
                and getattr(self, "end_stop_place_acceptance_v12", None) == "stable_supported"):
            from .skills_v12 import end_stop_stable_placement_config
            config = end_stop_stable_placement_config(self)
            command = self.ctx.eef_pos().copy()
            initial = command.copy()
            peak = float(self.external_force(part))
            for _ in range(int(.006 / .000025)):
                if peak >= float(force_stop):
                    break
                command[2] -= .000025
                self.arm.servo(command)
                peak = max(peak, float(self.external_force(part)))
                if not self.ctx.grasp_contacts(part)["held"]:
                    return Result(False, {"peak_force_n": peak}, "grasp lost during contact establishment")
            self.hold(.2)
            dwell = float(self.external_force(part))
            ok = peak >= config.press_contact_force_n
            return Result(ok, dict(peak_force_n=peak, dwell_force_n=dwell,
                extra_descent_m=float(initial[2] - command[2]),
                control_source="RGB-D grasp relation, encoder FK, force feedback",
                desired_force_n=float(force_stop),
                criterion="stable_supported bounded contact settle; no seating-band search"),
                "" if ok else "no measurable support contact before release")
        insertion = self.artifacts.get("insert", {})
        if (str(part).startswith("pin_") and insertion.get("part") == part
                and "command_depth_m" in insertion):
            from .skills_v12 import bounded_pin_press
            return bounded_pin_press(self, part, target_z, force_stop)
        if part == "end_stop":
            from .skills_v12 import bounded_end_stop_seat
            return bounded_end_stop_seat(self, part, target_z, force_stop)
        command = self.ctx.eef_pos().copy()
        initial = command.copy()
        peak = float(self.external_force(part))
        max_extra = .006
        for _ in range(int(max_extra / .000025)):
            if peak >= float(force_stop):
                break
            command[2] -= .000025
            self.arm.servo(command)
            peak = max(peak, float(self.external_force(part)))
            if not self.ctx.grasp_contacts(part)["held"]:
                return Result(False, {"peak_force_n": peak}, "grasp lost during contact establishment")
        target = control_position(self, part).copy()
        target[2] = float(target_z)
        if str(part).startswith("pin_"):
            # The next explicit pin/release predicate checks the real bore.
            # A force spike alone never creates an insertion success label.
            ok = True
            metrics = {"criterion": "bounded contact action completed; insertion acceptance follows"}
        else:
            ok, metrics = evaluate_functional_seat(self, part, target)
        metrics.update(peak_force_n=peak, extra_descent_m=float(initial[2] - command[2]),
                       control_source="RGB-D grasp relation, encoder FK, force feedback",
                       desired_force_n=float(force_stop))
        return Result(ok, metrics, "functional engagement region not reached")

    @skill("press_seat", "肩面压靠到位", "contact", ("held",))
    def press_seat(self, part, target_z, force_stop=2.5):
        if getattr(self, "functional_acceptance_v12", False):
            # The first assembly press owns ring insertion. A later stroke
            # release uses its current explicit target and bounded seating;
            # it must never search around the pre-stroke XY location again.
            if (part == "handle" and not self.stage_passes.get("assembly_pass")
                    and not self.artifacts.get("ring_insertion_v12", {}).get("success")):
                from .ring_insertion_v12 import execute
                metrics=execute(self,part,target_z,force_stop)
                return Result(bool(metrics["success"]),metrics,
                              "functional ring/post engagement not reached" if not metrics["success"] else "")
            return self._press_functional_v12(part, target_z, force_stop)
        command = self.ctx.eef_pos().copy()
        peak = self.external_force(part)
        max_steps = 1100 if str(part) == "handle" else 140
        extra_descent = 0.0
        def passive_support():
            bid = self.ctx.body_id(part)
            total = 0.0
            for index, contact in enumerate(self.ctx.data.contact):
                b1, b2 = map(int, self.ctx.model.geom_bodyid[[contact.geom1, contact.geom2]])
                if bid not in (b1, b2):
                    continue
                other = b2 if b1 == bid else b1
                if "finger" in self.ctx.model.body(other).name or abs(contact.frame[2]) < .5:
                    continue
                force = np.zeros(6)
                mujoco.mj_contactForce(self.ctx.model, self.ctx.data, index, force)
                total += max(0., float(force[0])) * abs(float(contact.frame[2]))
            return total
        for _ in range(max_steps):
            # The handle ring seats around the carriage boss.  Once its CAD
            # body origin is already within the seating band, continuing to
            # push can move it through the ring instead of increasing the
            # coarse contact proxy.  Release/settle and the independent
            # pose/support check remain the acceptance boundary.
            if str(part) == "handle" and abs(float(self.ctx.obj_pos(part)[2]) - float(target_z)) < .001:
                break
            if str(part) == "wipe_tool":
                # Guarded descent may stop on the stand just before its
                # position band. Continue a small bounded contact servo; the
                # original early break made press_seat a no-op in this case.
                if (abs(float(self.ctx.obj_pos(part)[2]) - float(target_z)) < .0015
                        and passive_support() >= .01):
                    break
                if peak >= 8.0 or extra_descent >= .001:
                    break
            elif str(part) != "handle" and peak >= force_stop:
                break
            command[2] -= 0.000025
            extra_descent += 0.000025
            self.arm.servo(command)
            peak = max(peak, self.external_force(part))
            if not self.ctx.grasp_contacts(part)["held"]:
                return Result(False, reason="grasp lost during seating")
        z = float(self.ctx.obj_pos(part)[2])
        err = abs(z - target_z)
        if str(part).startswith("pin_"):
            ok = peak > 0.35
        elif str(part) == "wipe_tool":
            ok = err < .0015 and passive_support() >= .01
        elif str(part) == "handle" and err < .0015:
            ok = True
        else:
            ok = err < 0.0015 and peak > 0.35
        contacts = []
        if str(part) == "wipe_tool":
            bid = self.ctx.body_id(part)
            for contact in self.ctx.data.contact:
                b1, b2 = map(int, self.ctx.model.geom_bodyid[[contact.geom1, contact.geom2]])
                if bid in (b1, b2):
                    contacts.append(dict(pair=[self.ctx.model.geom(contact.geom1).name,
                                               self.ctx.model.geom(contact.geom2).name],
                                         normal_z=float(contact.frame[2]),
                                         distance_m=float(contact.dist),
                                         position_m=np.asarray(contact.pos).tolist(),
                                         frame=np.asarray(contact.frame).tolist()))
        return Result(
            ok,
            {"height_error_m": err, "contact_force_n": peak,
             "extra_descent_m": extra_descent,
             "passive_support_n": passive_support() if str(part) == "wipe_tool" else None,
             "contacts": contacts},
            "pin seating contact not verified" if str(part).startswith("pin_") else "shoulder contact/height not verified",
        )

    @skill("inspect_seat", "检查零件装配位置", "verification")
    def inspect_seat(self, part, target, tol=0.0015):
        if getattr(self, "functional_acceptance_v12", False):
            from .skills_v12 import evaluate_functional_seat
            ok, metrics = evaluate_functional_seat(self, part, target)
            return Result(ok, metrics, "part is outside its functional engagement region")
        target = np.asarray(target, float)
        xyz = self.ctx.obj_pos(part)
        err = float(np.linalg.norm(xyz - target))
        tilt = float(np.degrees(np.arccos(np.clip(self.ctx.obj_axis(part)[2], -1, 1))))
        return Result(
            err < tol and tilt < 3.0,
            {"position_error_m": err, "tilt_deg": tilt},
            "assembly pose outside tolerance",
        )

    @skill("inspect_pin_inserted", "验收插销有效插入", "verification")
    def inspect_pin_inserted(self, part, hole_part="end_stop", hole_offset_m=(0., 0., 0.),
                             minimum_insertion_depth_m=0.006, phase="inserted_after_release"):
        from simbench.value.pin_geometry import PinInsertionConfig, evaluate_pin_context
        if not str(part).startswith("pin_"):
            return Result(False, reason="pin insertion predicate requires a pin")
        touching = False
        bid = self.ctx.body_id(part)
        for contact in self.ctx.data.contact:
            b1, b2 = map(int, self.ctx.model.geom_bodyid[[contact.geom1, contact.geom2]])
            if bid not in (b1, b2):
                continue
            other = b2 if b1 == bid else b1
            if "finger" in self.ctx.model.body(other).name and contact.dist <= 0:
                touching = True
        config=getattr(self, "pin_insertion_config", PinInsertionConfig(required_depth_m=float(minimum_insertion_depth_m)))
        def sample():
            return evaluate_pin_context(self.ctx,part,fixture_part=hole_part,hole_offset_m=hole_offset_m,
                phase=phase,released=self.held!=part,touching_finger=touching,config=config)
        metrics=sample()
        if config.contact_robustness and phase in ("inserted_after_release","retained_after_stroke"):
            from simbench.value.pin_contact_v12 import functional_retention_window
            def passive_wait(seconds):
                # Session.hold floors fractional control ticks. Acceptance
                # must observe at least its declared duration, not less.
                for _ in range(max(1,int(np.ceil(seconds/self.ctx.control_dt-1e-12)))):
                    self.ctx.step()
            metrics=functional_retention_window(sample,passive_wait,metrics,
                duration=config.retention_window_s,samples=config.retention_samples)
        self.artifact(f"{part}_{phase}", "pin_insertion", part, **metrics)
        return Result(bool(metrics["success"]), metrics, "pin not effectively inserted in hole")

    @skill("move_constrained", "沿已装导轨运动", "contact", ("held",))
    def move_constrained(self, part, target_x, max_force=14.0):
        from .skills_v12 import control_position
        measured_part = "carriage" if getattr(self, "functional_acceptance_v12", False) and part == "handle" else part
        start = self.ctx.obj_pos(measured_part).copy()
        target = control_position(self, part).copy()
        target[0] = target_x
        result = self.stream_part(part, target, 0.04, max_force)
        actual = self.ctx.obj_pos(measured_part)
        self.stroke.extend([float(start[0]), float(actual[0])])
        self.stroke_runs.append(dict(start_x=float(start[0]), end_x=float(actual[0]),
                                     requested_x=float(target_x), ok=bool(result.ok),
                                     cross_axis_m=float(np.linalg.norm((actual - start)[1:])),
                                     measured_part=measured_part))
        self.stroke_peak_forces.append(float(result.metrics.get("peak_force_n", 0.0)))
        result.metrics["cross_axis_m"] = float(np.linalg.norm((actual - start)[1:]))
        return result

    @skill("verify_stroke", "验证产品往复行程", "verification")
    def verify_stroke(self, minimum=0.08):
        forward = [r for r in self.stroke_runs if r["end_x"] - r["start_x"] > 0]
        reverse = [r for r in self.stroke_runs if r["end_x"] - r["start_x"] < 0]
        travel = max(self.stroke) - min(self.stroke) if self.stroke else 0.0
        forward_range = max((r["end_x"] - r["start_x"] for r in forward), default=0.0)
        reverse_range = max((r["start_x"] - r["end_x"] for r in reverse), default=0.0)
        cross_axis = max((float(r.get("cross_axis_m", 0.0)) for r in self.stroke_runs), default=0.0)
        peak = max(self.stroke_peak_forces, default=0.0)
        bidirectional = bool(forward and reverse)
        if getattr(self, "functional_acceptance_v12", False):
            bidirectional = bool(forward_range >= minimum and reverse_range >= minimum)
        return Result(
            travel >= minimum and bidirectional and cross_axis < (.010 if getattr(self, "functional_acceptance_v12", False) else .006) and peak <= 14.0,
            {"measured_range_m": travel, "forward_range_m": forward_range,
             "reverse_range_m": reverse_range, "required_range_m": minimum,
             "bidirectional": bidirectional, "cross_axis_m": cross_axis,
             "peak_force_n": peak},
            "insufficient bidirectional travel, lateral stability or force limit",
        )

    @skill("inspect_receiver_relation", "检查前序放置是否支持后续插销", "verification",
           ("empty", "seen"), legacy=False,
           obligations=("shared observed two-layer shaft corridor; physical engagement remains unverified",))
    def inspect_receiver_relation(self, part="end_stop"):
        if part != "end_stop":
            return Result(False, reason="receiver relation only defined for the printed stop/base pair")
        relation = copy.deepcopy((self.decision_observation or {}).get(
            "fixture_relations", {}).get("end_stop_to_base", {}))
        ok = bool(relation.get("observable") and relation.get("geometric_route_exists"))
        self.artifacts["receiver_relation_acceptance"] = relation
        return Result(ok, relation, "released end_stop has no observed shared shaft route into base" if not ok else "")

    @skill("inspect_stable_support", "验收端挡稳定支撑放置", "verification",
           ("empty",), legacy=False,
           obligations=("released stop supported in declared region after gripper retreat",))
    def inspect_stable_support(self, part="end_stop", target=None):
        if part != "end_stop":
            return Result(False, reason="stable-support acceptance is only defined for end_stop")
        if target is None:
            target = ((getattr(self, "receiver_execution_targets", {}).get("end_stop") or {}).get("position_m")
                      or getattr(self, "stage_targets", {}).get("end_stop"))
        if target is None:
            return Result(False, reason="stable-support acceptance requires the declared assembly target")
        from .skills_v12 import evaluate_end_stop_stable_support
        ok, metrics = evaluate_end_stop_stable_support(self, part, np.asarray(target, float),
                                                       after_retreat=True)
        self.artifacts["end_stop_stable_support"] = metrics
        return Result(bool(ok), metrics,
                      "" if ok else "end_stop is not stably supported after release and retreat")

    @skill("inspect_pin_joint", "验收插销连接两层真实孔", "verification",
           ("empty", "capability:pin"), legacy=False,
           obligations=("released shaft must engage both receivers and remain after functional stroke",))
    def inspect_pin_joint(self, part, phase="inserted_after_release"):
        from .skills_v12 import evaluate_pin_joint_engagement
        metrics = evaluate_pin_joint_engagement(self, part, phase=phase)
        self.artifacts.setdefault("pin_joint_engagement", {}).setdefault(part, {})[phase] = metrics
        return Result(bool(metrics.get("success")), metrics,
                      "pin does not connect both real receiver bores" if not metrics.get("success") else "")

    @skill("verify_clean", "验收清洁状态", "verification")
    def verify_clean(self, threshold=0.05):
        from simbench.value.cleaning import verify
        ok, metrics, reason = verify(self, threshold=float(threshold))
        return Result(ok, metrics, reason)

    @skill(
        "run_full_task_v7",
        "执行清洁—装配—功能测试",
        "execution",
        ("empty",),
        implementation="full_task_feedback_controller",
        obligations=("cleaning, assembly, bidirectional stroke and final release require simulation",),
    )
    def run_full_task_v7(self, order, choices, wipe_variant=0, wipe_force=1.5,
                         wipe_duration=14.0, stroke_minimum=0.08):
        from simbench.value.full_task_v7 import execute_full_task
        return execute_full_task(
            self, order=order, choices=choices, wipe_variant=int(wipe_variant),
            wipe_force=float(wipe_force), wipe_duration=float(wipe_duration),
            stroke_minimum=float(stroke_minimum),
        )

    @skill("measure_clearance", "测量导轨剩余间隙", "verification")
    def measure_clearance(self, part="carriage"):
        from .scene import CENTER

        if part != "carriage":
            return Result(
                False, reason="clearance model is specific to the carriage/guide pair"
            )
        error = abs(self.ctx.obj_pos(part)[1] - CENTER[1])
        clearance = 0.027 - 0.023 - error
        return Result(clearance > 0.0, {"lateral_margin_m": float(clearance)})

    @skill(
        "plan_recovery",
        "规划有预算的接触恢复",
        "planning",
        ("held",),
        produces="recovery",
        implementation="contact_state_rule",
    )
    def plan_recovery(self, part, as_="recovery", height=0.012):
        target = self.ctx.eef_pos() + [0, 0, height]
        self.arm.ik(target)
        self.artifact(as_, "recovery", part, target=target, retry_budget=1)
        return Result(metrics={"retreat_m": height, "retry_budget": 1})

    @skill("retract_contact", "从接触中撤回", "recovery", ("held", "artifact:recovery"))
    def retract_contact(self, part, artifact="recovery"):
        plan = self.artifacts[artifact]
        ok = self.arm.move(plan["target"], speed=0.025)
        return Result(
            ok and self.ctx.grasp_contacts(part)["held"],
            {"eef_height_m": self.ctx.eef_pos()[2]},
        )

    @skill(
        "spiral_search",
        "接触引导螺旋寻孔",
        "contact",
        ("held", "pin", "artifact:insertion"),
        implementation="force_guided_search",
    )
    def spiral_search(self, part, artifact="insert", radius=0.0025):
        plan = self.artifacts[artifact]
        target = plan["target"]
        start = self.ctx.obj_pos(part).copy()
        off = self.ctx.eef_pos() - start
        top = self.ctx.eef_pos()[2]
        command = self.ctx.eef_pos().copy()
        peak = 0.0
        for k in range(650):
            force = self.external_force(part)
            peak = max(peak, force)
            angle = k * 0.12
            r = radius * ((k % 200) / 199)
            command[:2] = (
                target[:2] + off[:2] + r * np.array([np.cos(angle), np.sin(angle)])
            )
            if force < 3.0:
                command[2] -= 0.00005
            else:
                command[2] = min(command[2] + 0.00002, top)
            self.arm.servo(command)
            # Pins have 48 mm between their body origin and insertion tip.
            # Entry must cross the receiver mouth, not merely descend in air.
            entry_z = target[2] + 0.048
            if self.ctx.obj_pos(part)[2] < entry_z - 0.004:
                break
            if not self.ctx.grasp_contacts(part)["held"]:
                return Result(False, reason="grasp lost while searching")
        depth = entry_z - self.ctx.obj_pos(part)[2]
        return Result(
            depth > 0.003,
            {"entry_depth_m": float(depth), "peak_force_n": peak},
            "search budget exhausted",
        )

    @skill(
        "learned_insert",
        "学习策略插销",
        "contact",
        ("held", "pin", "artifact:insertion"),
        implementation="behavior_cloning",
    )
    def learned_insert(self, part, artifact="insert", policy=None, max_steps=1000):
        if getattr(self, "functional_acceptance_v12", False):
            from .sensor_learning_v12 import execute
            return execute(self, part, artifact, policy or self.insertion_policy_v12, max_steps)
        from .learning import load_actor, insert_observation

        if policy is None:
            raise SkillFailure("learned_insert requires a trained checkpoint")
        actor = load_actor(policy)
        target = self.artifacts[artifact]["target"]
        peak = 0.0
        for _ in range(max_steps):
            obs = insert_observation(self, part, target)
            delta = actor(obs)
            self.arm.servo(self.ctx.eef_pos() + delta)
            peak = max(peak, self.external_force(part))
            if not self.ctx.grasp_contacts(part)["held"]:
                return Result(False, reason="learned policy lost part")
            if (
                abs(self.ctx.obj_pos(part)[2] - target[2]) < 0.0007
                and np.linalg.norm(self.ctx.obj_pos(part)[:2] - target[:2]) < 0.0008
            ):
                break
            if peak > 15.0:
                return Result(False, {"peak_force_n": peak}, "policy force limit")
        error = float(np.linalg.norm(self.ctx.obj_pos(part) - target))
        return Result(
            error < 0.0012,
            {
                "position_error_m": error,
                "peak_force_n": peak,
                "checkpoint": str(policy),
            },
            "learned policy did not seat",
        )


def export_catalog(out, components=False):
    from .graph import catalog_graph

    nodes = (
        [asdict(x) for x in HANDLERS.values()]
        if components
        else catalog_graph()["nodes"]
    )
    Path(out).write_text(json.dumps(nodes, ensure_ascii=False, indent=2))
