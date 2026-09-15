"""Deterministic typed tokenization. No rollout output, ID, or cost is read.

Object references expand to observed attributes; categorical vocabulary is a
fixed hash space, so training never learns a table of problem/candidate IDs.
"""
import hashlib
import math
import numpy as np
from .plan import PlanIR

VOCAB = 4096
NUMERIC = 24
EXCLUDE = {
    "id",
    "cost",
    "start_state",
    "bindings",
    "binding",
    "q_hover",
    "grasp_id",
    "prefix_id",
    "route_id",
    "terminal_id",
    "control_id",
    "grasp_epoch",
    "terminal_release_check",
    "observed_at",
}


def bucket(text):
    return 1 + int.from_bytes(
        hashlib.blake2b(str(text).encode(), digest_size=4).digest(), "little"
    ) % (VOCAB - 1)


def encode_plan(observation, plan):
    if not isinstance(plan, PlanIR):
        plan = PlanIR.from_dict(plan)
    plan.validate(observation["objects"])
    rows = []
    objects = observation["objects"]

    def add(key, value, stage, step=0, status="known", frame="", unit=""):
        numbers = np.zeros(NUMERIC, dtype=np.float32)
        category = 0
        if isinstance(value, (int, float, bool)):
            vals = np.array([value], dtype=np.float32)
        elif isinstance(value, (list, tuple)) and all(
            isinstance(v, (int, float, bool)) for v in value
        ):
            vals = np.array(value, dtype=np.float32)
        else:
            vals = np.array([], dtype=np.float32)
            if value is not None:
                category = bucket(value)
        if not np.isfinite(vals).all():
            raise ValueError("nonfinite input feature")
        # Split long vectors into chunks; never silently truncate a trajectory.
        if len(vals) > 8:
            for i in range(0, len(vals), 8):
                add(
                    f"{key}/chunk/{i//8}",
                    vals[i : i + 8].tolist(),
                    stage,
                    step,
                    status,
                    frame,
                    unit,
                )
            return
        for i, scale in enumerate((1.0, 10.0, 100.0)):
            numbers[i * 8 : i * 8 + len(vals)] = np.tanh(vals * scale)
        rows.append(
            (
                bucket(f"{key}|{frame}|{unit}"),
                category,
                {"known": 1, "deferred": 2, "unknown": 3}[status],
                stage,
                step,
                numbers,
            )
        )

    def visit(key, value, stage, step=0, status="known", frame="", unit=""):
        if isinstance(value, dict):
            for child, item in sorted(value.items()):
                if child not in EXCLUDE:
                    visit(key + "/" + child, item, stage, step, status, frame, unit)
        elif isinstance(value, (list, tuple)) and any(
            isinstance(v, (list, tuple, dict)) for v in value
        ):
            # Preserve endpoint and arc-length representatives for long paths.
            seq = value
            if len(seq) > 24 and all(isinstance(v, (list, tuple)) for v in seq):
                seq = sample_path(seq, 24)
            for i, item in enumerate(seq):
                visit(key + f"/{i}", item, stage, step, status, frame, unit)
        elif isinstance(value, str) and value in objects:
            obj = objects[value]
            visit(
                key + "/object_pose",
                [*obj["position"], *obj["quaternion"]],
                stage,
                step,
                status,
                "world",
                "m,wxyz",
            )
        else:
            add(key, value, stage, step, status, frame, unit)

    visit("robot", observation["robot"], 0)
    # No object-list position encoding. Canonicalize by attributes rather than
    # names so object renaming/reordering does not alter the scoring input.
    import json

    for obj in sorted(objects.values(), key=lambda o: json.dumps(o, sort_keys=True)):
        visit("object", obj, 0)
    for goal in observation["goals"]:
        visit("goal", goal, 3)
    for i, call in enumerate(plan.calls):
        stage = 1 if i < plan.boundary else 2
        add("skill", call.skill, stage, i + 1)
        add("call_kind", call.kind, stage, i + 1)
        for role, obj in sorted(call.roles.items()):
            visit("role/" + role, obj, stage, i + 1)
        for name, arg in sorted(call.arguments.items()):
            a_stage = 2 if name == "bound_terminal" else stage
            visit(
                "argument/" + name,
                arg.value,
                a_stage,
                i + 1,
                arg.status,
                arg.frame,
                arg.unit,
            )
            if arg.source_call:
                add(
                    "producer_relative_index",
                    [
                        i
                        - next(
                            j
                            for j, c in enumerate(plan.calls)
                            if c.id == arg.source_call
                        )
                    ],
                    a_stage,
                    i + 1,
                )
                add("producer_output", arg.source_output, a_stage, i + 1)
    if not rows:
        raise ValueError("empty plan input")
    return dict(
        keys=np.array([r[0] for r in rows]),
        categories=np.array([r[1] for r in rows]),
        statuses=np.array([r[2] for r in rows]),
        stages=np.array([r[3] for r in rows]),
        positions=np.array([r[4] for r in rows]),
        numbers=np.stack([r[5] for r in rows]),
    )


def sample_path(path, count):
    points = np.asarray(path, float)
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if distance[-1] == 0:
        return [points[0].tolist()]
    sampled = np.stack(
        [
            np.interp(np.linspace(0, distance[-1], count), distance, points[:, i])
            for i in range(points.shape[1])
        ],
        axis=1,
    )
    return sampled.tolist()


def collate(encoded, visuals=None, device="cpu"):
    import torch

    if not encoded:
        raise ValueError("empty batch")
    size = max(len(x["keys"]) for x in encoded)
    batch = {}
    for key in ("keys", "categories", "statuses", "stages", "positions", "numbers"):
        shape = (
            (len(encoded), size, NUMERIC) if key == "numbers" else (len(encoded), size)
        )
        array = np.zeros(shape, dtype=np.float32 if key == "numbers" else np.int64)
        for i, row in enumerate(encoded):
            array[i, : len(row[key])] = row[key]
        batch[key] = torch.as_tensor(array, device=device)
    batch["padding"] = (
        torch.arange(size, device=device)[None, :]
        >= torch.tensor([len(x["keys"]) for x in encoded], device=device)[:, None]
    )
    if visuals is not None:
        # One pooled feature per panorama/object crop, padded only for batching.
        n = max(len(v) for v in visuals)
        feats = np.zeros((len(visuals), n, 512), np.float32)
        mask = np.ones((len(visuals), n), bool)
        for i, v in enumerate(visuals):
            feats[i, : len(v)] = v
            mask[i, : len(v)] = False
        batch["visual"] = torch.as_tensor(feats, device=device)
        batch["visual_padding"] = torch.as_tensor(mask, device=device)
    return batch
