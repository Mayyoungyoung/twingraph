"""Task-independent LLM plan admission, value screening, and twin arbitration.

An LLM supplies complete atomic calls.  This module never repairs a rejected
call or reads a rollout label while constructing the value input.  The caller
provides the physical-twin verifier for its declared TaskSpec.
"""
from __future__ import annotations

from dataclasses import asdict
import copy
import inspect
from typing import Callable

from simbench.assembly.interfaces import AUXILIARY_INTERFACES, INTERFACES, PUBLIC_SKILLS
from simbench.assembly.library import HANDLERS, Session
from simbench.assembly.ports import validate_ports

from .plan import Argument, Call, PlanIR, argument, digest, plain
from .skill_graph import compile_graph


RESPONSE_SCHEMA = "twingraph.atomic_candidate_response.v1"
SUPPORTING_CALLS = ("select_grasp", "inspect", "measure")


def skill_catalog():
    """Publish the registered implementation choices and typed ports to the LLM."""
    catalog = {}
    for name, description in PUBLIC_SKILLS.items():
        implementations = {}
        for mode, implementation in INTERFACES.get(name, {"default": name}).items():
            if implementation not in HANDLERS:
                continue
            spec = HANDLERS[implementation]
            implementations[mode] = dict(implementation=implementation,
                parameters=str(inspect.signature(getattr(Session, implementation))),
                ports=[asdict(port) for port in spec.ports],
                requires=list(spec.requires), effects=list(spec.effects))
        catalog[name] = dict(**description, implementations=implementations)
    return catalog


def supporting_catalog():
    catalog = {}
    for name in SUPPORTING_CALLS:
        implementations = AUXILIARY_INTERFACES.get(name, {"default": name})
        catalog[name] = {}
        for mode, implementation in implementations.items():
            spec = HANDLERS[implementation]
            catalog[name][mode] = dict(parameters=str(inspect.signature(getattr(Session, implementation))),
                ports=[asdict(port) for port in spec.ports],
                requires=list(spec.requires), effects=list(spec.effects))
    return catalog


def planner_request(task_spec, observation, *, candidate_count):
    if not isinstance(candidate_count, int) or not 1 <= candidate_count <= 512:
        raise ValueError("candidate_count must be in [1,512]")
    return dict(schema="twingraph.atomic_candidate_request.v1",
        task_spec=plain(task_spec), observation=plain(observation),
        observation_sha256=digest(observation), atomic_skills=skill_catalog(),
        supporting_calls=supporting_catalog(),
        candidate_count=candidate_count,
        response_schema=RESPONSE_SCHEMA,
        rules=("Return complete ordered atomic calls with roles and registered parameters."
               " Vary skill/solver choices only within registered interfaces."
               " Do not include success labels, simulator state, or outcome-derived scores."))


def _call(row, index):
    if not isinstance(row, dict) or set(row) - {"id", "skill", "params", "arguments", "roles", "kind"}:
        raise ValueError(f"call {index}: unsupported fields")
    if ("params" in row) == ("arguments" in row):
        raise ValueError(f"call {index}: supply exactly one of params or arguments")
    skill = row.get("skill")
    if skill not in PUBLIC_SKILLS and skill not in SUPPORTING_CALLS:
        raise ValueError(f"call {index}: unregistered public or supporting skill {skill}")
    raw = row.get("arguments")
    if raw is None:
        raw = {key: argument(value) for key, value in row["params"].items()}
    else:
        raw = {key: (value if isinstance(value, Argument) else Argument(**value))
               for key, value in raw.items()}
    if not isinstance(row.get("roles", {}), dict):
        raise ValueError(f"call {index}: roles must be a mapping")
    return Call(str(row.get("id", f"call_{index:03d}")), skill, raw,
                dict(row.get("roles", {})), row.get("kind", "executable"))


def compile_candidates(response, observation, *, expected_count=None):
    """Validate each LLM-authored plan against the same graph used by value."""
    if not isinstance(response, dict) or set(response) != {"schema", "observation_sha256", "plans"}:
        raise ValueError("response must contain only schema, observation_sha256, plans")
    if response["schema"] != RESPONSE_SCHEMA or response["observation_sha256"] != digest(observation):
        raise ValueError("response schema or observation binding differs")
    rows = response["plans"]
    if not isinstance(rows, list) or not rows or (expected_count is not None and len(rows) != expected_count):
        raise ValueError("candidate count differs from request")
    accepted = []; names = set(); hashes = set()
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"name", "calls"}:
            raise ValueError(f"plan {i}: only name and calls are permitted")
        name = row["name"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f"plan {i}: invalid or repeated name")
        calls = [_call(call, j) for j, call in enumerate(row["calls"])]
        if not calls:
            raise ValueError(f"plan {i}: empty atomic program")
        steps = [dict(skill=c.skill, params={k: plain(v.value) for k, v in c.arguments.items()})
                 for c in calls]
        plan_id = digest(dict(calls=[asdict(c) for c in calls],
                              observation=digest(observation)))[:20]
        if plan_id in hashes:
            raise ValueError(f"plan {i}: duplicate executable program")
        plan = PlanIR(plan_id, calls, len(calls),
            dict(id=plan_id, execution="program", steps=steps,
                 observation_sha256=digest(observation)),
            status="unknown", protocol="atomic.program.v1")
        plan.validate(observation.get("objects", {}))
        graph = compile_graph(observation, plan)
        accepted.append(dict(name=name, plan=plan, graph=dict(
            schema="twingraph.atomic_flow_graph.v1", assembly=graph),
            plan_sha256=digest(plan.to_dict())))
        names.add(name); hashes.add(plan_id)
    return accepted


def compose_from_template(template, observation, proposals, allowed_values):
    """Apply model-chosen registered port alternatives to an executable template.

    ``allowed_values`` comes from pre-execution solvers/CAD, never physical
    outcomes.  The model chooses a combination; the graph audit and then the
    twin decide whether that combination is physically useful.
    """
    template = template if isinstance(template, PlanIR) else PlanIR.from_dict(template)
    if template.protocol != "atomic.program.v1":
        raise ValueError("port composition requires a generic atomic template")
    if not isinstance(proposals, list) or not proposals:
        raise ValueError("at least one composition is required")
    accepted = []; seen = set(); names = set()
    for row in proposals:
        if not isinstance(row, dict) or set(row) != {"name", "edits"}:
            raise ValueError("composition requires only name and edits")
        if not isinstance(row["name"], str) or not row["name"] or row["name"] in names:
            raise ValueError("duplicate or empty composition name")
        plan = copy.deepcopy(template)
        edited = set()
        calls = {c.id: c for c in plan.calls}
        for edit in row["edits"]:
            if not isinstance(edit, dict) or set(edit) != {"call_id", "argument", "value"}:
                raise ValueError("invalid port edit")
            key = f"{edit['call_id']}.{edit['argument']}"
            if key in edited or key not in allowed_values:
                raise ValueError(f"unregistered or repeated port alternative: {key}")
            if edit["call_id"] not in calls or edit["argument"] not in calls[edit["call_id"]].arguments:
                raise ValueError(f"port does not exist in executable plan: {key}")
            arg = calls[edit["call_id"]].arguments[edit["argument"]]
            if arg.status != "known" or digest(edit["value"]) not in {digest(v) for v in allowed_values[key]}:
                raise ValueError(f"value is not a grounded alternative: {key}")
            arg.value = plain(edit["value"])
            edited.add(key)
        plan.prefix["steps"] = [dict(skill=c.skill,
            params={k: plain(v.value) for k, v in c.arguments.items()}) for c in plan.calls]
        plan.prefix["observation_sha256"] = digest(observation)
        plan.id = digest(dict(calls=[asdict(c) for c in plan.calls],
                              observation=digest(observation)))[:20]
        plan.prefix["id"] = plan.id
        if plan.id in seen:
            raise ValueError("duplicate composed executable program")
        plan.validate(observation.get("objects", {}))
        graph = dict(schema="twingraph.atomic_flow_graph.v1",
                     assembly=compile_graph(observation, plan))
        accepted.append(dict(name=row["name"], plan=plan, graph=graph,
                             plan_sha256=digest(plan.to_dict())))
        seen.add(plan.id); names.add(row["name"])
    return accepted


def screen_and_verify(candidates, ranker, verifier: Callable, *, k, evaluation_scope):
    """Rank all admitted plans, physically verify at most K, retain failures."""
    if not isinstance(k, int) or not 1 <= k <= len(candidates):
        raise ValueError("k must be within the candidate pool")
    scores = [float(x) for x in ranker.score([row["graph"] for row in candidates])]
    if len(scores) != len(candidates):
        raise ValueError("value model returned a different number of scores")
    from math import isfinite
    if any(not isfinite(score) for score in scores):
        raise ValueError("value model returned a non-finite score")
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    trials = []; selected = None
    for rank, index in enumerate(order[:k], 1):
        candidate = candidates[index]
        result = verifier(candidate["plan"], candidate["graph"])
        if not isinstance(result, dict) or not result.get("valid") or result.get("resource_censored"):
            raise RuntimeError("twin result is invalid or resource-censored")
        if (result.get("evaluation_scope") != evaluation_scope
                or result.get("input_graph_sha256") != digest(candidate["graph"])):
            raise RuntimeError("twin result is not bound to the scored graph and declared task scope")
        trial = dict(rank=rank, name=candidate["name"], plan_sha256=candidate["plan_sha256"],
                     score=scores[index], success=bool(result.get("success")), result=result)
        trials.append(trial)
        if trial["success"]:
            selected = candidate["name"]
            break
    return dict(schema="twingraph.atomic_flow_result.v1",
                pool_size=len(candidates), k=k,
                ranking=[dict(name=candidates[i]["name"], score=scores[i]) for i in order],
                verified=trials, selected=selected,
                status="verified_success" if selected else "top_k_all_failed")
