"""Whole-flow value input: atomic assembly graph plus both feedback phases.

Cleaning and functional stroke are currently feedback controllers in the
executor. Their nodes here are controller interfaces with graph-bound ports,
not a claim that every internal adaptive action has been compiled to PlanIR.
The physical label is always the complete functional task outcome.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .generic_graph_value_v15 import (
    AtomicValueNetV15, FEATURES, RELATIONS, ValueRankerV15,
    _hash_add, encode_graph as encode_assembly_graph,
)
from .plan import PlanIR


SCHEMA = "twingraph.full_flow_graph_value.v20.r1"


def _controller_row(*, skill, manipulated, ports, observation):
    row = np.zeros(len(FEATURES), np.float32)
    _hash_add(row, 0, skill)
    _hash_add(row, 24, skill)
    _hash_add(row, 48, "feedback_controller")
    _hash_add(row, 72, f"manipulated={manipulated}")
    known = 0
    for name, value, unit in ports:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"invalid controller port {skill}.{name}")
        scale = {"N": 10., "s": 10., "m": .1, "": 1.}.get(unit, 1.)
        _hash_add(row, 5 * 24, name, float(np.clip(value / scale, -10., 10.)))
        _hash_add(row, 6 * 24, name)
        known += 1
    obj = observation.get("objects", {}).get(manipulated, {})
    pos = obj.get("position", obj.get("position_m"))
    tail = {name:i for i,name in enumerate(FEATURES)}
    row[tail["node_execution"]] = 1.
    row[tail["port_count"]] = known / 32.
    row[tail["known_port_fraction"]] = 1.
    row[tail["object_valid"]] = float(bool(obj.get("valid")))
    if isinstance(pos, (list, tuple)) and len(pos) == 3:
        for name, value in zip(("object_position_x_m", "object_position_y_m", "object_position_z_m"), pos):
            row[tail[name]] = float(value)
        row[tail["object_position_known"]] = 1.
    return row


def encode_graph(graph, *, check=True):
    """Read all executable phase parameters from the same bound input graph."""
    if graph.get("schema") != "twingraph.full_task_graph.v12":
        raise ValueError("whole-flow value requires V12 full-task graph")
    assembly = graph["assembly"]
    plan = PlanIR.from_dict(graph["full_plan"])
    if len(plan.calls) != 1 or plan.calls[0].skill != "run_full_task_v7":
        raise ValueError("missing executable full-task controller")
    params = {key: arg.value for key,arg in plan.calls[0].arguments.items()}
    prefix = assembly["plan"]["prefix"]
    if params["order"] != prefix["order"] or params["choices"] != prefix["choices"]:
        raise ValueError("full-task and atomic assembly plans disagree")
    cleaning = graph["cleaning"]
    if any(params[key] != cleaning[key] for key in ("wipe_variant", "wipe_force", "wipe_duration")):
        raise ValueError("cleaning controller ports disagree with full-task plan")
    goal = assembly["observation"]["goals"][0]
    if float(params["stroke_minimum"]) != float(goal["stroke_minimum_m"]):
        raise ValueError("functional controller port disagrees with immutable task goal")
    encoded = encode_assembly_graph(graph, check=check)
    observation = assembly["observation"]
    clean = _controller_row(skill="cleaning_feedback", manipulated="wipe_tool",
        ports=(("wipe_variant", params["wipe_variant"], ""),
               ("target_force", params["wipe_force"], "N"),
               ("duration", params["wipe_duration"], "s"),
               ("minimum_coverage", cleaning["minimum_coverage"], "")),
        observation=observation)
    functional = _controller_row(skill="functional_stroke_feedback", manipulated="handle",
        ports=(("stroke_minimum", params["stroke_minimum"], "m"),),
        observation=observation)
    n = len(encoded["x"])
    x = np.concatenate((clean[None, :], encoded["x"], functional[None, :]), axis=0)
    x[0, FEATURES.index("call_position")] = 0.
    x[0, FEATURES.index("before_boundary")] = 1.
    x[-1, FEATURES.index("call_position")] = 1.
    relations = np.zeros((len(RELATIONS), n + 2, n + 2), np.float32)
    relations[:, 1:n+1, 1:n+1] = encoded["relations"]
    relation = RELATIONS.index("execution")
    relations[relation, 0, 1] = 1.
    relations[relation, n, n+1] = 1.
    x[0, FEATURES.index("edge_out_count")] = 1./16.
    x[-1, FEATURES.index("edge_in_count")] = 1./16.
    return dict(x=x, relations=relations, active=np.ones(n+2, np.float32))


class ValueRankerV20(ValueRankerV15):
    def __init__(self, checkpoint, device="cpu"):
        import hashlib
        import torch
        self.device = device
        self.sha256 = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved.get("schema") != SCHEMA:
            raise ValueError("checkpoint is not a V20 full-flow value model")
        self.model = AtomicValueNetV15.build(**saved["model_config"]).to(device)
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()

    def score(self, graphs):
        import torch
        from .generic_graph_value_v15 import collate
        encoded = [encode_graph(graph) for graph in graphs]
        with torch.inference_mode():
            return self.model(collate(encoded, self.device)).sigmoid().cpu().numpy()
