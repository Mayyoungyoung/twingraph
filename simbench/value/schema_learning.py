"""Graph-only training without product feature lists or geometric score targets.

Constant and exactly redundant columns are learned from training inputs alone.
The same fitted transform is persisted for inference. All physical configurations
remain together across splits; trajectories and labels have immutable bindings.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from .collect import dump
from .graph_encode import encode_graphs, collate_graph
from .graph_learning import metrics
from .graph_network import GraphConfig, GraphValueNet
from .plan import PlanIR, digest
from .port_pool import PortPoolNet, fit_vocabulary, vectorize
from .skill_graph import compile_graph, interface_hash


def representation_hash():
    directory = Path(__file__).parent
    return digest({name: (directory/name).read_text(encoding="utf-8").replace("\r\n", "\n")
                   for name in ("graph_encode.py", "port_pool.py", "skill_graph.py", "plan.py", "program_audit.py")})


def load_groups(roots, splits=("train", "val")):
    groups, seen, configurations = [], set(), {}
    for root in roots:
        for complete in sorted(Path(root).glob("group_*/complete.json")):
            directory = complete.parent
            inp = json.loads((directory / "inputs.json").read_text())
            split = inp["declared_split"]
            # Checking metadata is allowed; held-out outcome files stay unopened.
            config = (inp["task"]["family"], inp.get("split_group", inp["task"]["seed"]))
            if config in configurations and configurations[config] != split:
                raise ValueError("physical configuration crosses data splits")
            configurations[config] = split
            if split not in splits:
                continue
            gid = inp["group_id"]
            if gid in seen:
                raise ValueError("duplicate physical group")
            seen.add(gid)
            ih = digest(inp)
            summary = json.loads(complete.read_text())
            outcomes = json.loads((directory / "outcomes.json").read_text())
            if summary["input_sha256"] != ih or outcomes["input_sha256"] != ih:
                raise ValueError("input/label/complete binding mismatch")
            plans = [PlanIR.from_dict(p) for p in inp["candidates"]]
            if len({p.id for p in plans}) != len(plans):
                raise ValueError("duplicate plan identity")
            graphs = [compile_graph(inp["observation"], p) for p in plans]
            graph_file = directory / "skill_graphs.json"
            if graph_file.exists():
                saved = json.loads(graph_file.read_text())
                if saved["input_sha256"] != ih or digest(saved["graphs"]) != digest(graphs):
                    raise ValueError("pre-execution graph drift")
            records = {p.id: {} for p in plans}
            disturbances = {}
            for row in outcomes["trials"]:
                cid, repeat = row["candidate_id"], row["trial"]["repeat"]
                if cid not in records or repeat in records[cid]:
                    raise ValueError("unknown candidate or duplicate rollout")
                if row.get("valid") is not True or type(row["success"]) is not bool:
                    raise ValueError("invalid physical success label")
                if bool(row["prefix_success"] and row["suffix_success"]) != row["success"]:
                    raise ValueError("inconsistent complete-program label")
                if row.get("timeout"):
                    raise ValueError("wall-clock censored rollout cannot be a physical failure label")
                th = digest(row["trial"])
                if row.get("trial_sha256") != th:
                    raise ValueError("disturbance hash mismatch")
                if repeat in disturbances and disturbances[repeat] != th:
                    raise ValueError("unpaired candidate disturbances")
                disturbances[repeat] = th
                records[cid][repeat] = row
            if not disturbances or summary["trials"] != len(outcomes["trials"]):
                raise ValueError("empty/incomplete trial set")
            for p, g in zip(plans, graphs):
                rs = records[p.id]
                if set(rs) != set(disturbances):
                    raise ValueError("incomplete candidate repeats")
                if graph_file.exists() and any(r.get("input_graph_sha256") != digest(g) for r in rs.values()):
                    raise ValueError("executed graph differs from scored graph")
            groups.append(dict(id=gid, seed=inp["task"]["seed"], config=config,
                split=split, inputs=inp, plans=plans, graphs=graphs, path=str(directory),
                encoded=encode_graphs(graphs, check=False), repeats=len(disturbances),
                input_sha256=ih, y=np.asarray([np.mean([r["success"] for r in records[p.id].values()])
                                             for p in plans], dtype=np.float32)))
    if not groups:
        raise ValueError("empty graph dataset")
    return groups


def compact_columns(x):
    """No outcomes: keep nonconstant, bitwise-distinct training columns."""
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("finite training matrix required")
    varying = np.flatnonzero(np.ptp(x, axis=0) > 1.e-7)
    columns, seen = [], set()
    for i in varying:
        key = np.ascontiguousarray(x[:, i]).tobytes()
        if key not in seen:
            seen.add(key)
            columns.append(int(i))
    if not columns:
        raise ValueError("no varying pre-execution inputs")
    return columns


def prepare(groups, saved):
    if saved["kind"] in {"compact", "port_mlp"}:
        for g in groups:
            g["port_x"] = np.stack([vectorize(e, saved["vocabulary"])[saved["columns"]]
                                      for e in g["encoded"]])


def make_model(saved):
    if saved["kind"] in {"compact", "port_mlp"}:
        return PortPoolNet(len(saved["columns"]))
    return GraphValueNet(GraphConfig(**saved["config"]))


def forward(model, kind, group, indices, device):
    if kind in {"compact", "port_mlp"}:
        return model(torch.as_tensor(group["port_x"][indices], device=device))
    return model(collate_graph([group["encoded"][i] for i in indices], device))


@torch.inference_mode()
def logits(model, kind, group, device="cpu"):
    model.eval()
    output = []
    for start in range(0, len(group["plans"]), 8):
        ix = np.arange(start, min(start + 8, len(group["plans"])))
        output.extend(forward(model, kind, group, ix, device).cpu().tolist())
    return np.asarray(output)


def sigmoid(x):
    return 1. / (1. + np.exp(-np.clip(x, -50, 50)))


def score_metrics(groups, zs, calibration=None):
    """Rank logits; use probabilities only for proper scoring rules."""
    c = calibration or dict(scale=1., bias=0.)
    result = metrics(groups, zs)
    for g, z, row in zip(groups, zs, result["rows"]):
        p = sigmoid(c["scale"]*np.asarray(z)+c["bias"])
        row["ranking_scores"] = np.asarray(z).tolist()
        row["scores"] = p.tolist()
        row["brier"] = float(np.mean((p-g["y"])**2))
        q = np.clip(p, 1e-7, 1-1e-7)
        row["log_loss"] = float(-np.mean(g["y"]*np.log(q)+(1-g["y"])*np.log1p(-q)))
    # Select models at the same independent-configuration level as calibration.
    by_config = {}
    for g, row in zip(groups, result["rows"]):
        by_config.setdefault(g.get("config", g["id"]), []).append(row)
    for key in ("brier", "log_loss"):
        result[key] = float(np.mean([np.mean([r[key] for r in rows]) for rows in by_config.values()]))
    return result


def calibrate(zs, groups):
    """Positive temperature and bias, fit on validation binomial labels only."""
    z = torch.tensor(np.concatenate(zs), dtype=torch.float64)
    y = torch.tensor(np.concatenate([g["y"] for g in groups]), dtype=torch.float64)
    if not bool(y.sum() > 0 and (1-y).sum() > 0):
        return dict(scale=1., bias=0., fit="identity", groups=len(groups),
                    skip_reason="validation has only one observed outcome class")
    # Equal configuration contribution despite differing candidate/checkpoint counts.
    counts = {}
    for g in groups:
        counts[g["config"]] = counts.get(g["config"], 0) + 1
    w = torch.tensor(np.concatenate([np.full(len(g["y"]), 1. / (counts[g["config"]] * len(g["y"])))
                                     for g in groups]), dtype=torch.float64)
    scale = torch.zeros((), dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([scale, bias], max_iter=100, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        # Compact box only prevents numerical divergence on perfectly separable data.
        a = scale.clamp(-4, 4).exp()
        loss = (nn.functional.binary_cross_entropy_with_logits(a*z + bias.clamp(-12, 12), y, reduction="none")*w).sum()/w.sum()
        loss.backward()
        return loss
    optimizer.step(closure)
    return dict(scale=float(scale.detach().clamp(-4, 4).exp()), bias=float(bias.detach().clamp(-12, 12)),
                fit="validation_only_positive_affine_logit", groups=len(groups))


def train(args):
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    groups = load_groups(args.data)
    tr = [g for g in groups if g["split"] == "train"]
    va = [g for g in groups if g["split"] == "val"]
    if not tr or not va:
        raise ValueError("nonempty training and validation configurations required")
    cfg = GraphConfig(relations=args.kind == "graph", pooling="attention", normalize=True)
    saved = dict(schema="twingraph.value.schema.v1", kind=args.kind, seed=args.seed,
                 config=asdict(cfg), interface_sha256=interface_hash(), representation_sha256=representation_hash(), training=vars(args))
    if args.kind in {"compact", "port_mlp"}:
        vocab = fit_vocabulary([e for g in tr for e in g["encoded"]])
        x = np.stack([vectorize(e, vocab) for g in tr for e in g["encoded"]])
        cols = compact_columns(x) if args.kind == "compact" else list(range(x.shape[1]))
        saved.update(vocabulary=vocab, columns=cols, raw_dim=x.shape[1], dim=len(cols))
        prepare(groups, saved)
    model = make_model(saved).to(args.device)
    if args.kind in {"compact", "port_mlp"}:
        x = torch.as_tensor(np.concatenate([g["port_x"] for g in tr]), device=args.device)
        model.mean.copy_(x.mean(0)); model.std.copy_(x.std(0).clamp_min(.01))
    else:
        from .encode import VOCAB, NUMERIC
        keys = np.concatenate([e["keys"] for g in tr for e in g["encoded"]])
        values = np.concatenate([e["numbers"] for g in tr for e in g["encoded"]])
        count = np.bincount(keys, minlength=VOCAB).clip(1)[:, None]
        total = np.zeros((VOCAB, NUMERIC)); squares = np.zeros_like(total)
        np.add.at(total, keys, values); np.add.at(squares, keys, values.astype(float)**2)
        mean = total/count; std = np.sqrt(np.maximum(squares/count - mean**2, 0)).clip(.01)
        model.number_mean.copy_(torch.as_tensor(mean, device=args.device))
        model.number_std.copy_(torch.as_tensor(std, device=args.device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003 if args.kind in {"compact", "port_mlp"} else .0005, weight_decay=.01)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    split_manifest = {s: [dict(id=g["id"], config=g["config"], input_sha256=g["input_sha256"]) for g in groups if g["split"] == s] for s in ("train", "val")}
    dump(out/"split.json", split_manifest)
    root = Path(__file__).parents[2]
    source = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for d in ("simbench/value", "simbench/assembly", "simbench/core") for p in sorted((root/d).glob("*.py"))}
    saved.update(source_sha256=digest(source), split_sha256=digest(split_manifest), parameters=sum(p.numel() for p in model.parameters()))
    dump(out/"source.json", source)
    best, history = float("inf"), []
    config_groups = {}
    for g in tr:
        config_groups.setdefault(g["config"], []).append(g)
    configurations = list(config_groups.values())
    for epoch in range(args.epochs):
        model.train(); losses = []
        for ci in np.random.permutation(len(configurations)):
            checkpoint_groups = configurations[ci]
            optimizer.zero_grad()
            for g in checkpoint_groups:
                order = np.random.permutation(len(g["plans"]))
                for start in range(0, len(order), 8):
                    ix = order[start:start+8]
                    z = forward(model, args.kind, g, ix, args.device)
                    loss = nn.functional.binary_cross_entropy_with_logits(z, torch.as_tensor(g["y"][ix], device=args.device))
                    loss = loss * len(ix) / (len(order) * len(checkpoint_groups))
                    loss.backward()
                    losses.append(float(loss.detach()))
            nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
        validation = score_metrics(va, [logits(model, args.kind, g, args.device) for g in va])
        row = dict(epoch=epoch+1, loss=float(np.mean(losses)), validation={k:v for k,v in validation.items() if k != "rows"}, seconds=time.perf_counter()-started)
        history.append(row)
        if validation["brier"] < best:
            best = validation["brier"]
            saved.update(epoch=epoch+1, state_dict=model.state_dict(), validation=validation)
            torch.save(saved, out/"best.pt")
        dump(out/"history.json", history)
        if epoch % 10 == 0:
            print(json.dumps(row), flush=True)
    selected = torch.load(out/"best.pt", map_location=args.device, weights_only=False)
    model.load_state_dict(selected["state_dict"])
    zs = [logits(model, args.kind, g, args.device) for g in va]
    selected["calibration"] = calibrate(zs, va)
    c = selected["calibration"]
    selected["validation_calibrated"] = score_metrics(va, zs, c)
    torch.save(selected, out/"best.pt")
    dump(out/"summary.json", {k:v for k,v in selected.items() if k not in {"state_dict", "vocabulary", "columns"}} | dict(wall_seconds=time.perf_counter()-started, train_groups=len(tr), val_groups=len(va), training_rollouts=sum(len(g["plans"])*g["repeats"] for g in tr)))


def load_model(path, device="cpu"):
    saved = torch.load(path, map_location=device, weights_only=False)
    if saved.get("schema") != "twingraph.value.schema.v1" or saved["interface_sha256"] != interface_hash():
        raise ValueError("checkpoint/schema/execution interface mismatch")
    if saved.get("representation_sha256") != representation_hash():
        raise ValueError("checkpoint typed graph representation changed")
    model = make_model(saved).to(device)
    model.load_state_dict(saved["state_dict"]); model.eval()
    return model, saved


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", nargs="+", required=True); p.add_argument("--out", required=True)
    p.add_argument("--kind", choices=("compact", "port_mlp", "graph", "sequence"), default="compact")
    p.add_argument("--seed", type=int, default=17); p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--device", default="cuda")
    train(p.parse_args())


if __name__ == "__main__":
    main()
