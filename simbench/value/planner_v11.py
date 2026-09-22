"""Model-authored strategy families, grounded in the current observed graph.

The JSON is the assistant's proposal, not a claim of an online API call.
Metric targets are bound by the existing RGB-D/IK/contact skills at execution.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from . import stage_v5, stage_v7
from .skill_graph import compile_graph, validate_graph
from .plan import PlanIR, argument, plain, digest

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "planner_v11.json"
SOURCE = "codex_authored_strategies_observation_grounded_v11"


def propose(observation, *, completed=(), config=CONFIG):
    specification = json.loads(Path(config).read_text(encoding="utf-8"))
    objects = observation["objects"]
    for part in stage_v7.ALL_PARTS:
        if part in completed or (part == "wipe_tool" and "cleaning" in completed):
            continue
        row = objects.get(part, {})
        if not row.get("valid") or row.get("position_m") is None:
            raise ValueError(f"reobserve required: {part}")
    # Clearance is a world height above the highest currently observed object.
    # Constants here are calibrated safety/control limits, not supplied poses.
    highest = max(float(row["position_m"][2]) for row in objects.values()
                  if row.get("valid") and row.get("position_m") is not None)
    clearance = float(np.clip(highest + .14, .96, 1.02))
    uncertainty = max(float(objects[p].get("fit_residual_m") or 0.)
                      for p in ("pin_left", "pin_right", "end_stop"))
    speed = float(np.clip(.006 / (1 + uncertainty / .004), .003, .006))
    # The initial pin order follows observed reach distance. Both orders meet
    # the same precedence constraints; it does not use success labels.
    xy = {p: np.asarray(objects[p]["position_m"][:2]) for p in stage_v5.PARTS
          if objects[p].get("valid") and objects[p].get("position_m") is not None}
    pins = sorted(("pin_left", "pin_right"), key=lambda p:
                  float(np.linalg.norm(xy[p]-xy["end_stop"])) if p in xy and "end_stop" in xy else float("inf"))
    base_order = ["carriage", "end_stop", *pins, "handle"]
    pool = []
    for strategy in specification["strategies"]:
        choices = {p: dict(yaw=0., height=.001, clearance=clearance,
                           force=8. if p.startswith("pin_") else 3.,
                           **({} if p == "carriage" else {"speed": speed}))
                   for p in stage_v5.PARTS}
        order = list(base_order)
        action = strategy["action"]
        if action == "reverse_pins": order[2:4] = reversed(order[2:4])
        if action == "clearance_high":
            for c in choices.values(): c["clearance"] = min(1.04, clearance + .025)
        if action == "clearance_low":
            for c in choices.values(): c["clearance"] = max(.95, clearance - .015)
        if action in ("pin_slow", "pin_firm", "pin_gentle", "pin_high_grasp"):
            for p in pins:
                if action == "pin_slow": choices[p]["speed"] = speed * .75
                if action == "pin_firm": choices[p]["force"] = 9.
                if action == "pin_gentle": choices[p]["force"] = 7.
                if action == "pin_high_grasp": choices[p]["height"] = .002
        if action == "stop_slow": choices["end_stop"]["speed"] = speed * .75
        if action == "handle_cross_grasp": choices["handle"]["yaw"] = float(np.pi / 2)
        if action == "pin_cross_grasp": choices[pins[0]]["yaw"] = float(np.pi / 2)
        pool.append(dict(name=strategy["name"], source=SOURCE, rationale=strategy["rationale"],
                         order=order, choices=choices, wipe_variant=int(action == "wipe_reverse"),
                         wipe_force=1.5, wipe_duration=14., stroke_minimum=.08))
    # Label-independent shuffle; tie breaking cannot secretly prefer reference.
    rng = np.random.default_rng(211)
    pool = [pool[i] for i in rng.permutation(len(pool))]
    return pool, dict(source=SOURCE, online_llm_call=False,
        strategy_spec_sha256=hashlib.sha256(Path(config).read_bytes()).hexdigest(),
        observation_sha256=observation.get("sha256"), clearance_m=clearance,
        insertion_speed_m_s=speed, pin_order=pins,
        explanation="Assistant-authored symbolic alternatives; deterministic grounding from current RGB-D")


def assembly_program(session, proposal):
    plan=stage_v5.program(session, session.stage_targets, proposal["order"],
                         deepcopy(proposal["choices"]), v7=True)
    # Observation dependencies are object-specific. An installed, occluded
    # pin is not an input to detecting and picking the next supply object.
    for call in plan.calls:
        if call.skill == "detect":
            call.arguments["required_parts"]=argument([call.roles["manipulated"]])
    plan.prefix["steps"]=[dict(skill=c.skill,params={k:plain(v.value) for k,v in c.arguments.items()})
                          for c in plan.calls[:plan.boundary]]
    plan.id=digest(dict(calls=plain(plan.to_dict()["calls"]),start=plan.prefix["start_state"]))[:20]
    plan.prefix["id"]=plan.id
    return plan.validate(session.parts)


def normalized_graph(session, proposal, *, completed=()):
    raw = session.decision_observation
    observation = dict(robot=dict(joints=session.ctx.arm_qpos.tolist(),
                                  fingers=session.ctx.finger_qpos.tolist(), eef=session.ctx.eef_pos().tolist()),
        objects={p: dict(position=r.get("position_m"), quaternion=r.get("quat_wxyz"),
                         valid=bool(r.get("valid")), quality=r.get("quality"),
                         fit_residual_m=r.get("fit_residual_m"),
                         capabilities=list(session.capabilities.get(p, ()))) for p, r in raw["objects"].items()},
        goals=[dict(predicate="functional_task", stroke_minimum_m=.08, pin_minimum_depth_m=.006)],
        perception=dict(backend=raw.get("backend"), observation_sha256=raw.get("sha256")))
    # Receiver observations are part of the same immutable pre-execution
    # input. They must survive graph compilation rather than live only in a
    # parallel feature vector unavailable to the executor.
    for field in ("fixtures", "assembly_targets", "receiver_geometry", "fixture_relations"):
        if field in raw:
            observation[field] = deepcopy(raw[field])
    graph = compile_graph(observation, assembly_program(session, proposal))
    return dict(schema="twingraph.full_task_graph.v11", assembly=graph,
        full_plan=stage_v7._full_plan(session,session.stage_targets,proposal["order"],proposal["choices"],
            proposal["wipe_variant"],proposal["wipe_force"],proposal["wipe_duration"],proposal["stroke_minimum"]).to_dict(),
        cleaning=dict(wipe_variant=proposal["wipe_variant"], wipe_force=proposal["wipe_force"],
                      wipe_duration=proposal["wipe_duration"], minimum_coverage=.72),
        completion=dict(completed=list(completed), remaining=[p for p in proposal["order"] if p not in completed]),
        phase_edges=[["cleaning", "carriage"], *map(list, stage_v5.PRECEDENCE),
                     ["handle", "bidirectional_stroke"], ["bidirectional_stroke", "pin_retention"]],
        obligations=["cleaning and stroke retain feedback controllers", "continuous feasibility requires twin"],
        proposal=deepcopy(proposal))


def value_features(graph):
    """Port readout compatible with the frozen V10/V6 numerical checkpoint.

    Validation precedes readout. Features come from the executable graph, not
    a separately maintained proposal vector. Relation attention is not claimed.
    """
    plan = validate_graph(graph["assembly"])
    full = PlanIR.from_dict(graph["full_plan"])
    params = {k:a.value for k,a in full.calls[0].arguments.items()}
    for key in ("order", "choices"):
        if params[key] != plan.prefix[key]:
            raise ValueError("full-task envelope and atomic graph disagree")
    for key in ("wipe_variant", "wipe_force", "wipe_duration"):
        if params[key] != graph["cleaning"][key]:
            raise ValueError("cleaning graph and executable envelope disagree")
    if params["stroke_minimum"] != .08:
        raise ValueError("candidate changed the task requirement")
    output = []
    for part in stage_v7.ALL_PARTS:
        row = graph["assembly"]["observation"]["objects"].get(part, {})
        output.extend(float(v) for v in (row.get("position") or [0., 0., 0.]))
        output.extend((float(row.get("quality") or 0.), float(bool(row.get("valid")))))
    choices = {}
    order = []
    for call in plan.calls:
        args = {k: a.value for k, a in call.arguments.items()}
        p = call.roles.get("manipulated")
        if call.skill == "estimate_grasp":
            order.append(p); choices[p] = dict(yaw=args["yaws"][0], height=args["height_offset"])
        if p not in choices: continue
        if call.skill == "grasp": choices[p]["force"] = args["force"]
        if call.skill == "plan_path" and "clearance" in args: choices[p]["clearance"] = args["clearance"]
        if ((call.skill == "move" and args.get("mode") == "guarded") or
            (call.skill == "plan_path" and args.get("method") == "contact" and p.startswith("pin_"))):
            choices[p]["speed"] = args["speed"]
    for p in stage_v5.PARTS:
        output.extend(float(choices[p].get(k, 0.)) for k in ("yaw", "height", "clearance", "force", "speed"))
    clean = graph["cleaning"]
    output.extend((float(order.index("pin_left") > order.index("pin_right")),
                   float(clean["wipe_variant"]), float(clean["wipe_force"]), float(clean["wipe_duration"])))
    return np.asarray(output, np.float32)
