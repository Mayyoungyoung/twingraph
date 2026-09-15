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
        return plain(asdict(self))

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
