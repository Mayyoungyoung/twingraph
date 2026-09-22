"""Task-agnostic value input over the complete executable atomic skill graph.

The encoder has no assembly-stage or part-name schema.  New tasks reuse the
same representation as long as they compile to the registered PlanIR/skill
graph contract.  Categorical strings use stable feature hashing; numeric ports
use declared units plus known/deferred/present masks.  Variable node counts are
padded only at batch collation and are never truncated silently.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from simbench.assembly.interfaces import family
from .plan import PlanIR
from .skill_graph import RELATIONS, validate_graph
try:
    from .skill_graph import validate_geometry_conditions
except ImportError:  # audited frozen V12 graphs predate optional V13 geometry conditions
    def validate_geometry_conditions(graph, plan=None):
        if graph.get("geometry_conditions") is not None:
            raise ValueError("this frozen skill-graph runtime cannot validate geometry_conditions")
        return {}


SCHEMA = "twingraph.atomic_graph_value.v15.r2"
HASH = 24
MAX_NODES = 512
VALUE_RELATIONS = RELATIONS
UNIT_SCALE = {
    "": 1.0, "1": 1.0, "m": .1, "m/s": .05, "s": 10.0,
    "rad": math.pi, "N": 10.0, "kg": 1.0, "pixel": 1000.0,
}


def _bucket(value, size=HASH):
    return int.from_bytes(hashlib.sha256(str(value).encode()).digest()[:8], "big") % size


def _signed(value):
    return -1.0 if hashlib.sha256(("sign:" + str(value)).encode()).digest()[0] & 1 else 1.0


def _hash_add(vector, offset, value, weight=1.0):
    vector[offset + _bucket(value)] += _signed(value) * float(weight)


def _numbers(value):
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)] if math.isfinite(float(value)) else []
    if isinstance(value, (list, tuple)):
        answer = []
        for item in value:
            answer.extend(_numbers(item))
        return answer
    return []


def feature_names():
    names = []
    for group in ("skill", "implementation", "family", "roles", "requirements",
                  "numeric_port_value", "numeric_port_known", "numeric_port_deferred",
                  "categorical_port", "state", "goal"):
        names.extend(f"{group}_hash_{i}" for i in range(HASH))
    names.extend(("call_position", "before_boundary", "node_information", "node_execution",
                  "port_count", "known_port_fraction", "deferred_port_fraction",
                  "unknown_port_fraction", "read_count", "write_count", "edge_in_count",
                  "edge_out_count", "object_valid", "object_position_x_m", "object_position_y_m",
                  "object_position_z_m", "object_position_known", "object_fit_residual_m",
                  "object_fit_residual_known", "object_quality", "object_quality_known",
                  "object_yaw_sin", "object_yaw_cos", "object_yaw_known", "robot_joint_sin_mean",
                  "robot_joint_cos_mean", "robot_joint_known_fraction", "robot_eef_x_m",
                  "robot_eef_y_m", "robot_eef_z_m", "robot_eef_known", "geometry_present",
                  "geometry_pass", "geometry_unknown", "geometry_rejected", "geometry_margin_m",
                  "geometry_margin_known", "graph_node_count", "graph_edge_count"))
    return tuple(names)


FEATURES = feature_names()


def _yaw(quaternion):
    if not isinstance(quaternion, (list, tuple)) or len(quaternion) != 4:
        return None
    q = np.asarray(quaternion, float)
    if not np.isfinite(q).all() or np.linalg.norm(q) < 1.e-8:
        return None
    w, x, y, z = q / np.linalg.norm(q)
    return math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))


def encode_graph(graph, *, check=True):
    assembly = graph["assembly"]
    plan = validate_graph(assembly) if check else PlanIR.from_dict(assembly["plan"])
    nodes = assembly["nodes"]
    if not nodes or len(nodes) > MAX_NODES:
        raise ValueError(f"atomic graph node count must be in [1,{MAX_NODES}], got {len(nodes)}")
    conditions = validate_geometry_conditions(graph, plan) if check else (
        graph.get("geometry_conditions", {}).get("parts", {}))
    observation = assembly["observation"]
    objects = observation.get("objects", {})
    robot = observation.get("robot", {})
    joints = [float(x) for x in robot.get("joints", ()) if isinstance(x, (int, float))]
    if any(not math.isfinite(x) for x in joints):
        raise ValueError("non-finite robot joints")
    eef = robot.get("eef")
    eef = np.asarray(eef, float) if isinstance(eef, (list, tuple)) and len(eef) == 3 else None
    if eef is not None and not np.isfinite(eef).all():
        raise ValueError("non-finite robot eef")
    incoming = np.zeros(len(nodes), float); outgoing = np.zeros(len(nodes), float)
    relations = np.zeros((len(RELATIONS), len(nodes), len(nodes)), np.float32)
    for edge in assembly.get("edges", ()):
        a, b = int(edge["source"]), int(edge["target"])
        if not 0 <= a < len(nodes) or not 0 <= b < len(nodes):
            raise ValueError("atomic graph edge index out of range")
        relation = edge["relation"]
        if relation not in RELATIONS:
            raise ValueError(f"unregistered atomic graph relation: {relation}")
        relations[RELATIONS.index(relation), a, b] = 1.0
        outgoing[a] += 1.; incoming[b] += 1.
    rows = []
    categorical_groups = 11
    tail_offset = categorical_groups * HASH
    for i, node in enumerate(nodes):
        row = np.zeros(len(FEATURES), np.float32)
        _hash_add(row, 0 * HASH, node.get("skill", ""))
        _hash_add(row, 1 * HASH, node.get("implementation", ""))
        _hash_add(row, 2 * HASH, family(node.get("implementation", "")))
        for key, value in sorted(node.get("roles", {}).items()):
            _hash_add(row, 3 * HASH, f"{key}={value}")
        for value in (*node.get("requires", ()), *node.get("effects", ()), *node.get("obligations", ())):
            _hash_add(row, 4 * HASH, value)
        ports = node.get("ports", ())
        known = deferred = unknown = 0
        for port in ports:
            name = port.get("name", "")
            status = port.get("status", "unknown")
            known += status == "known"; deferred += status == "deferred"; unknown += status == "unknown"
            values = _numbers(port.get("value"))
            if values:
                scale = UNIT_SCALE.get(port.get("unit", ""), 1.0)
                normalized = np.clip(np.asarray(values) / max(scale, 1.e-8), -10., 10.)
                for component, value in enumerate(normalized):
                    # Preserve vector/path direction.  Averaging a target or
                    # axis would make permutations with the same mean
                    # indistinguishable, even though execution differs.
                    key = name if len(normalized) == 1 else f"{name}[{component}]"
                    _hash_add(row, 5 * HASH, key, float(value))
                if len(normalized) > 1:
                    _hash_add(row, 8 * HASH, f"{name}:numeric_length={len(normalized)}")
                _hash_add(row, (6 if status == "known" else 7) * HASH, name)
            elif port.get("value") is not None:
                _hash_add(row, 8 * HASH, f"{name}={port.get('value')}")
        for ref in (*node.get("reads", ()), *node.get("writes", ())):
            _hash_add(row, 9 * HASH, ref.get("state", ""))
        for goal in observation.get("goals", ()):
            for key, value in sorted(goal.items()):
                nums = _numbers(value)
                if nums:
                    for component, item in enumerate(nums):
                        _hash_add(row, 10 * HASH,
                                  key if len(nums) == 1 else f"{key}[{component}]", item)
                else:
                    _hash_add(row, 10 * HASH, f"{key}={value}")
        part = node.get("roles", {}).get("manipulated")
        obj = objects.get(part, {}) if part is not None else {}
        pos = obj.get("position_m", obj.get("position"))
        pos = np.asarray(pos, float) if isinstance(pos, (list, tuple)) and len(pos) == 3 else None
        if pos is not None and not np.isfinite(pos).all():
            raise ValueError("non-finite object position")
        residual = obj.get("fit_residual_m"); quality = obj.get("quality")
        yaw = _yaw(obj.get("quat_wxyz", obj.get("quaternion")))
        geometry = conditions.get(part, {}) if isinstance(conditions, dict) and part is not None else {}
        minimum, required = geometry.get("min_clearance_m"), geometry.get("required_clearance_m")
        margin = float(minimum) - float(required) if minimum is not None and required is not None else None
        kind = node.get("kind", "")
        values = (
            i / max(len(nodes) - 1, 1), float(i < int(plan.boundary)), float(kind == "information"),
            float(kind in ("execution", "contact")), len(ports) / 32., known / max(len(ports), 1),
            deferred / max(len(ports), 1), unknown / max(len(ports), 1), len(node.get("reads", ())) / 8.,
            len(node.get("writes", ())) / 8., incoming[i] / 16., outgoing[i] / 16.,
            float(bool(obj.get("valid", pos is not None))), *(pos.tolist() if pos is not None else (0., 0., 0.)),
            float(pos is not None), float(residual or 0.), float(residual is not None),
            float(quality or 0.), float(quality is not None), math.sin(yaw) if yaw is not None else 0.,
            math.cos(yaw) if yaw is not None else 0., float(yaw is not None),
            float(np.mean(np.sin(joints))) if joints else 0., float(np.mean(np.cos(joints))) if joints else 0.,
            min(len(joints), 7) / 7., *(eef.tolist() if eef is not None else (0., 0., 0.)), float(eef is not None),
            float(bool(geometry)), float(geometry.get("status") == "necessary_pass"),
            float(geometry.get("status") == "unknown"), float(geometry.get("status") == "rejected"),
            float(margin or 0.), float(margin is not None), len(nodes) / MAX_NODES,
            len(assembly.get("edges", ())) / (MAX_NODES * 4.),
        )
        row[tail_offset:] = np.asarray(values, np.float32)
        rows.append(row)
    x = np.stack(rows)
    if not np.isfinite(x).all():
        raise ValueError("non-finite generic graph encoding")
    return dict(x=x, relations=relations, active=np.ones(len(nodes), np.float32))


def collate(encoded, device="cpu"):
    import torch
    batch = len(encoded); nodes = max(len(row["x"]) for row in encoded)
    dim = encoded[0]["x"].shape[-1]; rel = len(RELATIONS)
    x = np.zeros((batch, nodes, dim), np.float32)
    relations = np.zeros((batch, rel, nodes, nodes), np.float32)
    active = np.zeros((batch, nodes), np.float32)
    for i, row in enumerate(encoded):
        n = len(row["x"]); x[i, :n] = row["x"]; relations[i, :, :n, :n] = row["relations"]
        active[i, :n] = row["active"]
    return {"x": torch.as_tensor(x, device=device),
            "relations": torch.as_tensor(relations, device=device),
            "active": torch.as_tensor(active, device=device)}


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AtomicValueNetV15:
    """Namespace wrapper keeps torch an optional import for graph-only tooling."""
    @staticmethod
    def build(input_dim=len(FEATURES), width=64, message_layers=2):
        import torch
        from torch import nn

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = dict(input_dim=input_dim, width=width, message_layers=message_layers)
                self.register_buffer("mean", torch.zeros(input_dim))
                self.register_buffer("scale", torch.ones(input_dim))
                self.embed = nn.Sequential(nn.Linear(input_dim, width), nn.GELU(), nn.LayerNorm(width))
                self.messages = nn.ModuleList([nn.Linear(width, width, bias=False)
                                               for _ in range(message_layers * len(RELATIONS) * 2)])
                self.updates = nn.ModuleList([nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.LayerNorm(width))
                                              for _ in range(message_layers)])
                self.attention = nn.Sequential(nn.Linear(width, width // 2), nn.GELU(), nn.Linear(width // 2, 1))
                self.plan = nn.Sequential(nn.Linear(width * 3, width), nn.GELU(), nn.Dropout(.1), nn.Linear(width, 1))

            def forward(self, batch):
                x = ((batch["x"] - self.mean) / self.scale).clamp(-8., 8.)
                mask = batch["active"] > 0
                h = self.embed(x) * mask.unsqueeze(-1)
                index = 0
                for layer in range(message_layers):
                    total = torch.zeros_like(h)
                    for r in range(len(RELATIONS)):
                        a = batch["relations"][:, r]
                        for direction in range(2):
                            matrix = a.transpose(-1, -2) if direction == 0 else a
                            matrix = matrix / matrix.sum(-1, keepdim=True).clamp_min(1.)
                            total = total + matrix @ self.messages[index](h); index += 1
                    h = (h + self.updates[layer](torch.cat((h, total / (2 * len(RELATIONS))), -1))) * mask.unsqueeze(-1)
                logits = self.attention(h).squeeze(-1).masked_fill(~mask, -1.e9)
                weights = torch.softmax(logits, -1)
                attended = (weights.unsqueeze(-1) * h).sum(1)
                mean = h.sum(1) / mask.sum(1, keepdim=True).clamp_min(1.)
                maximum = h.masked_fill(~mask.unsqueeze(-1), -1.e9).max(1).values
                return self.plan(torch.cat((attended, mean, maximum), -1)).squeeze(-1)

        return Model()


class ValueRankerV15:
    def __init__(self, checkpoint, device="cpu"):
        import torch
        self.device = device; self.sha256 = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved.get("schema") != SCHEMA:
            raise ValueError("checkpoint is not a V15 atomic graph value model")
        self.model = AtomicValueNetV15.build(**saved["model_config"]).to(device)
        self.model.load_state_dict(saved["state_dict"]); self.model.eval()

    def score(self, graphs):
        import torch
        encoded = [encode_graph(graph) for graph in graphs]
        with torch.inference_mode():
            return self.model(collate(encoded, self.device)).sigmoid().cpu().numpy()

    rank = score
    __call__ = score
