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
                else f"内部准备：{spec.label}"
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
            kind in ("joint_path", "cartesian_path", "insertion", "recovery")
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
            rng=copy.deepcopy(self.rng.bit_generator.state),
            grasp_epoch=self.grasp_epoch,
            active_candidate_id=self.active_candidate_id,
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
        if "rng" in state:
            self.rng.bit_generator.state = copy.deepcopy(state["rng"])
        self.grasp_epoch = state.get("grasp_epoch", 0)
        self.active_candidate_id = state.get("active_candidate_id")

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
        start = self.ctx.obj_pos(part).copy()
        target = np.asarray(target, float)
        offset = self.ctx.eef_pos() - start
        peak = 0.0
        duration = max(1.0, 1.9 * np.linalg.norm(target - start) / speed)
        for t in np.linspace(0, 1, max(2, int(duration / self.ctx.control_dt))):
            s = t * t * t * (10 - 15 * t + 6 * t * t)
            self.arm.servo(start + s * (target - start) + offset)
            peak = max(peak, self.external_force(part))
            if peak > force_limit:
                return Result(False, {"peak_force_n": peak}, "contact overload")
            if not self.ctx.grasp_contacts(part)["held"]:
                return Result(
                    False,
                    {
                        "contact": self.ctx.grasp_contacts(part),
                        "object": self.ctx.obj_pos(part),
                        "axis": self.ctx.obj_axis(part),
                        "eef": self.ctx.eef_pos(),
                        "peak_force_n": peak,
                    },
                    "grasp lost",
                )
        self.hold(0.35)
        error = float(np.linalg.norm(self.ctx.obj_pos(part) - target))
        return Result(
            error < 0.0015,
            {
                "error_m": error,
                "travel_m": float(np.linalg.norm(self.ctx.obj_pos(part) - start)),
                "peak_force_n": peak,
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
    def observe_parts(self):
        self.observations = {
            p: [
                self.ctx.obj_pos(p) + self.rng.normal(0, self.noise, 3)
                for _ in range(5)
            ]
            for p in self.parts
        }
        return Result(
            metrics={
                "detected": list(self.observations),
                "backend": "simulator pose observations",
                "noise_std_m": self.noise,
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
        xyz = np.median(self.observations[part], axis=0)
        quat = self.ctx.obj_pose(part)[1]
        self.artifact(as_, "pose", part, xyz=xyz, quat=quat, uncertainty=self.noise)
        return Result(metrics={"position_m": xyz, "quaternion_wxyz": quat})

    @skill(
        "propose_grasps",
        "生成抓取候选",
        "planning",
        ("artifact:pose",),
        produces="grasps",
        implementation="geometry_and_ik",
    )
    def propose_grasps(self, part, artifact="pose", as_="grasps"):
        pose = self.artifacts[artifact]
        dz, width = self.grasp_specs[part]
        candidates = []
        unresolved = []
        pose_id = hashlib.sha256(
            np.asarray(pose["xyz"]).tobytes() + np.asarray(pose["quat"]).tobytes()
        ).hexdigest()[:12]
        for yaw in (0.0, np.pi / 2):
            xyz = pose["xyz"] + np.array([0, 0, dz])
            try:
                q = self.arm.ik(xyz + [0, 0, 0.10], down(yaw))
                # Width depends on the box face; round parts keep their diameter.
                w = (
                    width
                    if yaw == 0 or part not in ("carriage", "end_stop")
                    else {"carriage": 0.044, "end_stop": 0.022}[part]
                )
                candidates.append(
                    dict(
                        id=f"{part}:yaw:{yaw:.6f}:pose:{pose_id}",
                        xyz=xyz,
                        yaw=yaw,
                        width=w,
                        q_hover=q,
                        cost=float(np.linalg.norm(q - self.ctx.arm_qpos) + 2 * w),
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
    def place_object(self, part, target, tol=0.0015, settle=0.30, minimum_support=0.01):
        """Release an already supported part; moving there is the Move atom's job."""
        if tol <= 0 or settle < 0 or minimum_support <= 0:
            return Result(False, reason="invalid placement tolerances")
        pose = self.inspect_seat(part, target, tol)
        if not pose.ok:
            return Result(
                False, pose.metrics, "move to the placement target before releasing"
            )
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
        if support < minimum_support:
            return Result(
                False,
                {"support_force_n": support},
                "no supporting contact; refusing air release",
            )
        # This call commits the same declared release effect even if the later
        # settled-pose check fails. Never leave a released object marked held.
        self.call("open_gripper")
        self.hold(settle)
        settled = self.inspect_seat(part, target, tol)
        return Result(
            settled.ok,
            {
                **settled.metrics,
                "support_force_n": support,
                "jaw_span_m": self.ctx.pad_span(),
            },
            "placement did not remain within tolerance",
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
        ok = self.arm.open()
        return Result(ok, {"jaw_span_m": self.ctx.pad_span()})

    @skill("approach", "接近抓取位姿", "transition", ("empty", "artifact:grasp"))
    def approach(self, part, artifact="grasp"):
        goal = self.artifacts[artifact]
        ok = self.arm.move(goal["xyz"], down(goal["yaw"]), speed=0.045)
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
    def lift(self, part, height=0.10):
        start = self.ctx.obj_pos(part).copy()
        ok = self.arm.move(self.ctx.eef_pos() + [0, 0, height], speed=0.06)
        delta = float(self.ctx.obj_pos(part)[2] - start[2])
        held = self.ctx.grasp_contacts(part)["held"]
        return Result(
            bool(ok and held and delta > height - 0.003),
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
        target = np.asarray(target, float)
        for _ in range(3):
            error = target - self.ctx.obj_pos(part)
            if np.linalg.norm(error) < tolerance:
                break
            self.arm.move(self.ctx.eef_pos() + error, speed=0.035, tol=0.0004)
        error = float(np.linalg.norm(target - self.ctx.obj_pos(part)))
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
        self, part, target, axis=(0, 0, -1), as_="insert", speed=0.008, force_limit=12.0
    ):
        target = np.asarray(target, float)
        axis = np.asarray(axis, float)
        axis /= np.linalg.norm(axis)
        self.arm.ik(self.arm.part_target(part, target))
        self.artifact(
            as_,
            "insertion",
            part,
            target=target,
            axis=axis,
            speed=speed,
            force_limit=force_limit,
        )
        return Result(
            metrics={
                "target_m": target,
                "axis": axis,
                "speed_m_s": speed,
                "force_limit_n": force_limit,
            }
        )

    @skill("slide_insert", "沿导轨约束插入", "contact", ("held", "artifact:insertion"))
    def slide_insert(self, part, artifact="insert"):
        plan = self.artifacts[artifact]
        return self.stream_part(
            part, plan["target"], plan["speed"], plan["force_limit"]
        )

    @skill("guarded_descent", "接触保护下降", "contact", ("held",))
    def guarded_descent(self, part, target_z, force_stop=2.0, speed=0.006):
        start = self.ctx.obj_pos(part).copy()
        command = self.ctx.eef_pos().copy()
        peak = 0.0
        for _ in range(600):
            force = self.external_force(part)
            peak = max(peak, force)
            if force >= force_stop:
                break
            if self.ctx.obj_pos(part)[2] <= target_z + 0.0003:
                break
            command[2] -= speed * self.ctx.control_dt
            self.arm.servo(command)
            if not self.ctx.grasp_contacts(part)["held"]:
                return Result(False, reason="grasp lost during descent")
        self.hold(0.2)
        z = float(self.ctx.obj_pos(part)[2])
        return Result(
            z <= target_z + 0.001 or peak >= force_stop,
            {
                "z_m": z,
                "peak_contact_n": peak,
                "travel_m": float(start[2] - z),
                "stopped_by": (
                    "contact"
                    if peak >= force_stop
                    else "height" if z <= target_z + 0.001 else "budget"
                ),
            },
            "descent exhausted",
        )

    @skill("press_seat", "肩面压靠到位", "contact", ("held",))
    def press_seat(self, part, target_z, force_stop=2.5):
        command = self.ctx.eef_pos().copy()
        peak = self.external_force(part)
        for _ in range(140):
            if peak >= force_stop:
                break
            command[2] -= 0.000025
            self.arm.servo(command)
            peak = max(peak, self.external_force(part))
            if not self.ctx.grasp_contacts(part)["held"]:
                return Result(False, reason="grasp lost during seating")
        z = float(self.ctx.obj_pos(part)[2])
        err = abs(z - target_z)
        return Result(
            err < 0.0015 and peak > 0.35,
            {"height_error_m": err, "contact_force_n": peak},
            "shoulder contact/height not verified",
        )

    @skill("inspect_seat", "检查零件装配位置", "verification")
    def inspect_seat(self, part, target, tol=0.0015):
        target = np.asarray(target, float)
        xyz = self.ctx.obj_pos(part)
        err = float(np.linalg.norm(xyz - target))
        tilt = float(np.degrees(np.arccos(np.clip(self.ctx.obj_axis(part)[2], -1, 1))))
        return Result(
            err < tol and tilt < 3.0,
            {"position_error_m": err, "tilt_deg": tilt},
            "assembly pose outside tolerance",
        )

    @skill("move_constrained", "沿已装导轨运动", "contact", ("held",))
    def move_constrained(self, part, target_x, max_force=14.0):
        start = self.ctx.obj_pos(part).copy()
        target = start.copy()
        target[0] = target_x
        result = self.stream_part(part, target, 0.04, max_force)
        actual = self.ctx.obj_pos(part)
        self.stroke.extend([float(start[0]), float(actual[0])])
        result.metrics["cross_axis_m"] = float(np.linalg.norm((actual - start)[1:]))
        return result

    @skill("verify_stroke", "验证产品往复行程", "verification")
    def verify_stroke(self, minimum=0.08):
        travel = max(self.stroke) - min(self.stroke) if self.stroke else 0.0
        return Result(
            travel >= minimum,
            {"measured_range_m": travel, "required_range_m": minimum},
            "insufficient tested travel",
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
