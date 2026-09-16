"""One serialized plan is consumed by both the scorer and the executor.

Trace IDs are metadata. Parameters have explicit known/deferred semantics;
deferred object-to-hand targets are resolved only after the measured grasp.
"""
from dataclasses import asdict, dataclass, field
import copy
import hashlib
import json
import math
from typing import Any

SCHEMA = "twingraph.plan.v1"
PROTOCOL = "pin_suffix.feedback.v1"

# These are the actual execute_joint_path payload fields, with controller units.
# Identifiers and provenance remain integrity metadata, never learned features.
JOINT_PATH_FIELDS = {
    "type": ("category", "", ""),
    "part": ("object_ref", "", ""),
    "start_q": ("vector", "rad", "robot_joint"),
    "joints": ("joint_path", "rad", "robot_joint"),
    "target": ("position", "m", "world"),
    "rotation": ("rotation", "1", "world"),
}


def initial_artifacts(plan):
    """Validate materialized initial free-motion paths without accepting logs.

    Carrying paths depend on the subsequently measured grasp and stay deferred.
    Supporting an initial held-object checkpoint requires an explicit initial
    holding state; it must not be inferred from a nominal future grasp.
    """
    import numpy as np

    artifacts = plain(copy.deepcopy(plan.prefix.get("initial_artifacts", {})))
    if not isinstance(artifacts, dict):
        raise ValueError("initial_artifacts must be a named artifact mapping")
    if artifacts and plan.prefix.get("execution") != "program":
        raise ValueError("initial artifacts require an explicit program")
    epoch = plan.prefix.get("initial_grasp_epoch", 0)
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
        raise ValueError("initial grasp acquisition epoch must be a nonnegative integer")
    for name, item in artifacts.items():
        if not isinstance(name, str) or not name or not isinstance(item, dict):
            raise ValueError("invalid initial artifact")
        allowed = set(JOINT_PATH_FIELDS) | {"id", "binding"}
        if set(item) - allowed or not set(JOINT_PATH_FIELDS).issubset(item):
            raise ValueError("initial joint path has missing or unsupported fields")
        if item["type"] != "joint_path" or item["part"] is not None:
            raise ValueError("only materialized initial free joint paths are supported")
        binding = item.get("binding")
        if not isinstance(binding, dict) or set(binding) != {
            "held", "grasp_artifact", "grasp_id", "grasp_epoch", "prefix_id"
        }:
            raise ValueError("materialized path requires an exact execution binding")
        if (binding["held"] is not None or binding["grasp_artifact"] is not None
                or binding["grasp_id"] is not None or binding["grasp_epoch"] != epoch
                or binding["prefix_id"] != plan.id):
            raise ValueError("materialized path is not bound to this initial free-motion program")
        arrays = {key: np.asarray(item[key])
                  for key in ("start_q", "joints", "target", "rotation")}
        if any(x.dtype.kind not in "fi" for x in arrays.values()):
            raise ValueError("materialized path geometry must contain numeric values")
        q, joints = arrays["start_q"], arrays["joints"]
        if (q.ndim != 1 or len(q) == 0 or joints.ndim != 2
                or joints.shape[0] == 0 or joints.shape[1] != len(q)
                or arrays["target"].shape != (3,) or arrays["rotation"].shape != (3, 3)
                or any(not np.isfinite(x).all() for x in arrays.values())):
            raise ValueError("materialized path requires finite, dimensionally consistent geometry")
        rotation = arrays["rotation"]
        if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6)):
            raise ValueError("materialized path rotation must be a proper rotation matrix")
    return artifacts


def materialize_initial_path(session, plan, path):
    """Replace the first deferred free transfer with the exact preplanned path.

    The caller supplies a solver-produced path from this initial snapshot.
    Necessary collision checks remain the candidate generator's responsibility.
    This helper does not turn a path into a physical feasibility certificate.
    """
    import numpy as np
    from simbench.assembly.candidates import fingerprint
    from simbench.assembly.interfaces import resolve

    plan = copy.deepcopy(plan if isinstance(plan, PlanIR) else PlanIR.from_dict(plan))
    plan.validate(session.parts)
    if (plan.prefix.get("execution") != "program" or session.held is not None
            or fingerprint(session) != plan.prefix["start_state"]):
        raise ValueError("materialization requires the bound initial empty-gripper snapshot")
    planner = None
    for i, call in enumerate(plan.calls):
        name, params = resolve(call.skill, {k: a.value for k, a in call.arguments.items()})
        if name == "plan_transfer":
            planner = (i, call, params)
            break
        if name not in {"observe_parts", "estimate_pose", "propose_grasps", "select_grasp"}:
            raise ValueError("initial path must be consumed before any robot motion")
    if planner is None:
        raise ValueError("program has no initial joint-path planner")
    i, call, params = planner
    name = params.get("as_", "transfer")
    if any(a.source_call == call.id for c in plan.calls for a in c.arguments.values()):
        raise ValueError("cannot remove a planner with explicit deferred consumers")
    if i + 1 >= len(plan.calls):
        raise ValueError("initial planner must be followed by its path execution")
    following = plan.calls[i + 1]
    implementation, consumer = resolve(following.skill, {k: a.value for k, a in following.arguments.items()})
    if implementation != "execute_joint_path" or consumer.get("artifact", "transfer") != name:
        raise ValueError("initial planner must be followed by its path execution")
    item = plain(copy.deepcopy(path))
    binding = item.get("binding", {})
    if (binding.get("held") is not None or binding.get("grasp_artifact") is not None
            or binding.get("grasp_id") is not None or binding.get("grasp_epoch") != session.grasp_epoch):
        raise ValueError("cannot reuse a path bound to another grasp acquisition")
    if not np.array_equal(np.asarray(item.get("start_q")), session.ctx.arm_qpos):
        raise ValueError("materialized path has a stale joint-space start")
    if not set(JOINT_PATH_FIELDS).issubset(item):
        raise ValueError("initial joint path has missing or unsupported fields")
    plan.id = digest(dict(program=plan.id, path={k: item[k] for k in JOINT_PATH_FIELDS}))[:20]
    plan.prefix["id"] = plan.id
    plan.prefix["initial_grasp_epoch"] = int(session.grasp_epoch)
    item["binding"] = {**binding, "prefix_id": plan.id}
    plan.prefix.setdefault("initial_artifacts", {})[name] = item
    del plan.calls[i]
    if i < plan.boundary:
        plan.boundary -= 1
    plan.prefix["steps"] = [dict(skill=c.skill, params={k: a.value for k, a in c.arguments.items()})
                            for c in plan.calls[:plan.boundary]]
    return plan.validate(session.parts)


def plain(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def digest(value):
    return hashlib.sha256(
        json.dumps(plain(value), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


@dataclass
class Argument:
    value: Any = None
    kind: str = "scalar"
    status: str = "known"
    frame: str = ""
    unit: str = ""
    source_call: str | None = None
    source_output: str | None = None

    def validate(self):
        if self.status not in {"known", "deferred", "unknown"}:
            raise ValueError("invalid parameter status")
        if self.kind in {"pose", "position", "path", "joint_path"} and not self.frame:
            raise ValueError("geometric parameters require a coordinate frame")
        if self.status == "deferred" and (
            not self.source_call or not self.source_output
        ):
            raise ValueError("deferred parameters require a producer and output")
        json.dumps(plain(self.value), allow_nan=False)


@dataclass
class Call:
    id: str
    skill: str
    arguments: dict[str, Argument] = field(default_factory=dict)
    roles: dict[str, str] = field(default_factory=dict)
    kind: str = "executable"


@dataclass
class PlanIR:
    id: str
    calls: list[Call]
    boundary: int
    prefix: dict
    status: str = "necessary_pass"
    schema: str = SCHEMA
    protocol: str = PROTOCOL

    @property
    def edges(self):
        # Distinct calls of the same skill still have an execution edge.
        return [(a.id, b.id) for a, b in zip(self.calls, self.calls[1:])]

    def validate(self, objects=None):
        if self.schema != SCHEMA or not self.protocol:
            raise ValueError("unsupported plan schema/protocol")
        if not self.calls or not 0 < self.boundary <= len(self.calls):
            raise ValueError("invalid execution boundary")
        if self.status not in {"necessary_pass", "unknown", "conflict"}:
            raise ValueError("invalid plan status")
        if self.id != self.prefix.get("id"):
            raise ValueError("plan identity disagrees with executable prefix")
        initial_artifacts(self)
        ids = [c.id for c in self.calls]
        if len(ids) != len(set(ids)):
            raise ValueError("call identifiers must be unique")
        for i, call in enumerate(self.calls):
            part_argument = call.arguments.get("part")
            if (
                part_argument
                and part_argument.status == "known"
                and (
                    call.roles.get("manipulated", part_argument.value)
                    != part_argument.value
                )
            ):
                raise ValueError("object role disagrees with execution parameter")
            for ref in call.roles.values():
                if objects is not None and ref not in objects:
                    raise ValueError("unbound object role")
            for arg in call.arguments.values():
                arg.validate()
                if arg.source_call is not None and arg.source_call not in ids[:i]:
                    raise ValueError("parameter source must be an earlier call")
        # Prefix is executable legacy payload. Verify the scoring view is exact.
        if len(self.prefix.get("steps", [])) != self.boundary:
            raise ValueError("prefix execution boundary disagrees with calls")
        for call, step in zip(self.calls, self.prefix["steps"]):
            if call.roles.get("manipulated") != self.prefix["part"]:
                raise ValueError("prefix object role disagrees with execution")
            params = {
                k: a.value
                for k, a in call.arguments.items()
                if not k.startswith("bound_")
            }
            if call.skill != step["skill"] or plain(params) != plain(
                step.get("params", {})
            ):
                raise ValueError("scoring calls disagree with executable prefix")
        if self.prefix.get("execution") == "program":
            if self.protocol != "assembly.program.feedback.v2":
                raise ValueError("program payload requires v2 execution protocol")
            order=[]; choices={}; current=None
            for call in self.calls:
                args={k:a.value for k,a in call.arguments.items()}
                if call.skill=="estimate_grasp":
                    current=call.roles["manipulated"]
                    order.append(current)
                    choices[current]=dict(yaw=args["yaws"][0],height=args["height_offset"])
                elif current is not None:
                    if call.skill=="grasp": choices[current]["force"]=args["force"]
                    if call.skill=="plan_path" and "clearance" in args: choices[current]["clearance"]=args["clearance"]
                    if call.skill=="move" and args.get("mode")=="guarded": choices[current]["speed"]=args["speed"]
            if plain(order)!=self.prefix["order"] or plain(choices)!=self.prefix["choices"]:
                raise ValueError("program scoring/execution choices disagree")
            return self
        bound = self.calls[0].arguments
        for key in ("grasp", "path", "control", "terminal"):
            if plain(bound["bound_" + key].value) != plain(self.prefix[key]):
                raise ValueError("scoring parameters disagree with executable prefix")
        return self

    def to_dict(self):
        row = plain(asdict(self))
        if not row["prefix"].get("initial_artifacts"):
            row["prefix"].pop("initial_artifacts", None)
        return row

    @classmethod
    def from_dict(cls, row):
        row = copy.deepcopy(row)
        row["calls"] = [
            Call(
                **{
                    **c,
                    "arguments": {k: Argument(**a) for k, a in c["arguments"].items()},
                }
            )
            for c in row["calls"]
        ]
        return cls(**row).validate()


def argument(value, **kw):
    kind = (
        "category"
        if isinstance(value, str)
        else "vector"
        if isinstance(value, (list, tuple))
        else "scalar"
    )
    return Argument(value=plain(value), kind=kw.pop("kind", kind), **kw)


def from_pick(candidate, suffix):
    payload = plain(candidate.to_dict())
    calls = [
        Call(
            f"prefix_{i}",
            step["skill"],
            {k: argument(v) for k, v in step.get("params", {}).items()},
            {"manipulated": candidate.part},
            "checker" if step["skill"] == "inspect" else "executable",
        )
        for i, step in enumerate(payload["steps"])
    ]
    for key in ("grasp", "path", "control", "terminal"):
        calls[0].arguments["bound_" + key] = Argument(
            payload[key], kind="record", frame="world"
        )
    return PlanIR(
        candidate.id, calls + suffix, len(calls), payload, candidate.status
    ).validate()


def resolve_argument(arg, session, plan, part=None):
    if arg.kind in {"position", "pose", "path"} and arg.frame != "world":
        raise ValueError("executor requires world-frame geometric targets")
    if arg.status == "known":
        return copy.deepcopy(arg.value)
    if arg.status == "unknown":
        raise ValueError("cannot execute an unresolved parameter")
    part = part or plan.prefix["part"]
    if arg.source_output == "object_to_eef":
        return session.arm.part_target(part, arg.value)
    if arg.source_output == "grasp_yaw":
        return session.artifacts["grasp"]["yaw"]
    if arg.source_output == "grasp_hover":
        import numpy as np
        grasp = session.artifacts["grasp"]
        if grasp["part"] != part:
            raise ValueError("deferred grasp producer object mismatch")
        return np.asarray(grasp["xyz"]) + np.asarray(arg.value)
    raise ValueError(f"unsupported deferred output {arg.source_output}")


def execute_prefix(session, plan):
    import numpy as np
    from simbench.assembly.candidates import Candidate, execute_pick_candidate

    plan.validate(session.parts)
    if plan.prefix.get("execution") == "program":
        from simbench.assembly.candidates import fingerprint
        if fingerprint(session) != plan.prefix["start_state"]:
            raise ValueError("stale program initial snapshot")
        materialized = initial_artifacts(plan)
        if materialized and (session.held is not None
                or session.grasp_epoch != plan.prefix.get("initial_grasp_epoch", 0)):
            raise ValueError("materialized path initial grasp acquisition changed")
        for name, item in materialized.items():
            for key in ("start_q", "joints", "target", "rotation"):
                item[key] = np.asarray(item[key], dtype=float)
            session.artifacts[name] = item
        session.active_candidate_id = plan.id
        execute_calls(session, plan, plan.calls[:plan.boundary])
        return
    row = copy.deepcopy(plan.prefix)
    for key in ("xyz", "q_hover"):
        if key in row["grasp"]:
            row["grasp"][key] = np.asarray(row["grasp"][key], float)
    if row["path"]:
        for key in ("target", "rotation", "joints", "start_q"):
            row["path"][key] = np.asarray(row["path"][key], float)
    execute_pick_candidate(session, Candidate(**row))


def execute_suffix(session, plan):
    execute_calls(session, plan, plan.calls[plan.boundary:])


def execute_calls(session, plan, calls):
    for call in calls:
        params = {
            key: resolve_argument(arg, session, plan, call.roles.get("manipulated"))
            for key, arg in call.arguments.items()
        }
        session.call(call.skill, **params)
