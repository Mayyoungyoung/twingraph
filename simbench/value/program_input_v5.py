"""Schema-derived state/program input. No task-specific feature list or GNN.

Every declared execution port is serialized in call order. Numeric leaves are
raw values (normalization is fitted on training inputs); categorical leaves are
one-hot vocabulary entries. Constant/duplicate columns are learned, not chosen.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .plan import JOINT_PATH_FIELDS, digest

SCHEMA = "twingraph.state_program.v5"


def source_manifest():
    root = Path(__file__).parents[2]
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for d in ("simbench/value", "simbench/assembly", "simbench/core")
            for p in sorted((root/d).glob("*.py"))}


def encoding_hash():
    root = Path(__file__).parent
    return digest({n: (root/n).read_text(encoding="utf-8").replace("\r\n", "\n")
                   for n in ("program_input_v5.py", "skill_graph.py", "plan.py", "program_audit.py")})


def program_record(graph):
    """Canonical bindings remove trace IDs while retaining object identity.

    Manipulated objects receive slots in first-use order. Unused context objects
    follow in geometry order. Roles, goals, and state references use those slots;
    a consistent rename of object bindings therefore changes no numeric input.
    """
    obs = graph["observation"]
    objects = obs["objects"]
    order = []

    def bind(name):
        if isinstance(name, str) and name in objects and name not in order:
            order.append(name)

    for node in graph["nodes"]:
        for _, value in sorted(node["roles"].items()):
            bind(value)
        for port in node["ports"]:
            if port["kind"] == "object_ref":
                bind(port["value"])
    for name in sorted(objects, key=lambda k: json.dumps(objects[k], sort_keys=True)):
        bind(name)
    slots = {name: i for i, name in enumerate(order)}

    def value(v):
        if isinstance(v, dict):
            return {k: value(x) for k, x in sorted(v.items())}
        if isinstance(v, (list, tuple)):
            return [value(x) for x in v]
        if isinstance(v, str) and v in slots:
            return {"object_slot": slots[v]}
        return v

    def state_ref(ref):
        result = copy.deepcopy(ref)
        key = result["state"]
        if key.startswith("object:"):
            name = key[len("object:"):]
            result["state"] = "object"
            result["object_slot"] = slots[name]
        return result

    calls = []
    for node in graph["nodes"]:
        row = {k: value(node[k]) for k in
               ("skill", "implementation", "kind", "roles", "requires", "effects", "outputs", "obligations")}
        row["reads"] = [state_ref(x) for x in node["reads"]]
        row["writes"] = [state_ref(x) for x in node["writes"]]
        ports = {}
        for port in node["ports"]:
            p = {k: port[k] for k in ("kind", "unit", "frame", "status", "required")}
            # Artifact addresses/selected IDs are provenance, their producers
            # and materialized executable payload carry the actual semantics.
            p["value"] = None if port["kind"] in ("artifact_ref", "binding") else value(port["value"])
            p["source"] = value(port["source"])
            if "materialized" in port:
                p["materialized"] = {k: value(port["materialized"][k]) for k in JOINT_PATH_FIELDS}
            ports[port["name"]] = p
        row["ports"] = ports
        calls.append(row)
    return dict(schema=SCHEMA, state=dict(robot=value(obs["robot"]),
                objects=[value(objects[k]) for k in order], goals=value(obs["goals"])),
                plan=dict(boundary=graph["boundary"], calls=calls))


def leaves(record):
    """Uniform numeric/categorical/presence leaves, including every list item."""
    result = {}

    def walk(path, v):
        if isinstance(v, dict):
            for k, x in sorted(v.items()):
                walk(path + "/" + k, x)
        elif isinstance(v, (tuple, list)):
            result[path + "/@length|number"] = float(len(v))
            for i, x in enumerate(v):
                walk(path + "/" + str(i), x)
        elif isinstance(v, (bool, int, float, np.number)):
            f = float(v)
            if not np.isfinite(f) or abs(f)>np.finfo(np.float32).max:
                raise ValueError("non-finite execution input")
            result[path + "|number"] = f
        else:
            if v is not None and not isinstance(v, str):
                raise TypeError(f"unsupported execution input {type(v)}")
            result[path + "|category=" + json.dumps(v, ensure_ascii=False)] = 1.

    walk("", record)
    return result


class InputSchema:
    def __init__(self, saved):
        if saved["schema"] != SCHEMA:
            raise ValueError("unsupported input schema")
        self.saved = saved
        self.keys = saved["keys"]
        self.index = {k: i for i, k in enumerate(self.keys)}
        self.known = set(saved["all_keys"])

    @classmethod
    def fit(cls, records):
        """Only training records; no outcomes/validation accepted by this API."""
        flat = [leaves(x) for x in records]
        if not flat:
            raise ValueError("empty training inputs")
        keys = sorted({k for row in flat for k in row})
        index = {k: i for i, k in enumerate(keys)}
        x = np.zeros((len(flat), len(keys)), np.float32)
        for i, row in enumerate(flat):
            for k, v in row.items():
                x[i, index[k]] = v
        keep, seen, aliases = [], {}, {}
        varying = np.flatnonzero(np.ptp(x, axis=0) > 0)
        for j in varying:
            token = np.ascontiguousarray(x[:, j]).tobytes()
            if token not in seen:
                seen[token]=int(j)
                keep.append(int(j))
            else:
                aliases[keys[j]]=keys[seen[token]]
        constants={keys[j]:float(x[0,j]) for j in np.flatnonzero(np.ptp(x,axis=0)==0)}
        if not keep:
            raise ValueError("training programs have no varying input")
        return cls(dict(schema=SCHEMA, all_keys=keys, keys=[keys[i] for i in keep],
                        raw_dim=len(keys), dim=len(keep), training_records=len(records),
                        removed_constant_columns=len(keys)-len(varying),
                        removed_duplicate_columns=len(varying)-len(keep),
                        training_constants=constants,duplicate_aliases=aliases,
                        input_only_fit=True))

    def transform(self, records, return_diagnostics=False):
        # Write directly to retained columns; never construct a huge discarded
        # dense feature vector at deployment as the previous port MLP did.
        x = np.zeros((len(records), len(self.keys)), np.float32)
        unseen = [];diagnostics=[]
        for i, record in enumerate(records):
            row = leaves(record)
            for key, v in row.items():
                j = self.index.get(key)
                if j is not None:
                    x[i, j] = v
            unseen.append(sorted(set(row)-self.known))
            if return_diagnostics:
                constants=self.saved.get("training_constants",{})
                aliases=self.saved.get("duplicate_aliases",{})
                changed=[key for key,val in constants.items() if np.float32(row.get(key,0.))!=np.float32(val)]
                broken=[key for key,alias in aliases.items() if np.float32(row.get(key,0.))!=np.float32(row.get(alias,0.))]
                diagnostics.append(dict(unseen_fields=len(unseen[-1]),changed_training_constants=len(changed),
                                        broken_duplicate_relations=len(broken),examples=(changed+broken)[:10]))
        return (x,unseen,diagnostics) if return_diagnostics else (x, unseen)
