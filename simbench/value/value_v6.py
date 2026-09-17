"""Robust state/program value model with an optional pre-execution RGB-D branch."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from .collect import dump
from .plan import PlanIR, digest
from .program_input_v5 import InputSchema, program_record
from .skill_graph import compile_graph, interface_hash


SCHEMA = "twingraph.value.v6"
VIEW_MODES = ("none", "task", "top", "both")


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_manifest():
    root = Path(__file__).parents[2]
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for d in ("simbench/value", "simbench/assembly", "simbench/core")
            for p in sorted((root / d).glob("*.py"))}


def encoding_hash():
    root = Path(__file__).parent
    names = ("value_v6.py", "program_input_v5.py", "skill_graph.py", "plan.py")
    return digest({name: (root / name).read_text(encoding="utf-8").replace("\r\n", "\n")
                   for name in names})


def _load_vision(directory, manifest):
    path = directory / manifest["file"]
    if not path.is_file():
        raise ValueError("RGB-D artifact is missing")
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    content = hashlib.sha256()
    for key in sorted(arrays):
        value = np.ascontiguousarray(arrays[key])
        content.update(key.encode()); content.update(str(value.dtype).encode())
        content.update(json.dumps(value.shape).encode()); content.update(value.tobytes())
    if content.hexdigest() != manifest["sha256"]:
        raise ValueError("RGB-D artifact does not match its value manifest")
    return arrays


def _vision_array(arrays, mode):
    if mode not in VIEW_MODES:
        raise ValueError(f"unsupported view mode {mode}")
    if mode == "none":
        return np.zeros((0, 1, 1), np.float32)
    names = {"task": ("task_view",), "top": ("top_view",),
             "both": ("task_view", "top_view")}[mode]
    channels = []
    for name in names:
        rgb = np.asarray(arrays[f"{name}_rgb"], np.float32) / 255.
        depth = np.asarray(arrays[f"{name}_depth_mm"], np.float32) / 2000.
        if rgb.ndim != 3 or rgb.shape[2] != 3 or depth.shape != rgb.shape[:2]:
            raise ValueError("invalid RGB-D observation shape")
        channels.extend([rgb[..., 0], rgb[..., 1], rgb[..., 2], np.clip(depth, 0, 1)])
    return np.stack(channels).astype(np.float32)


def load_groups(roots, splits=("train", "val", "test"), view_mode="none",
                require_source=True):
    groups = []
    seen_ids = set()
    current_source = source_manifest()
    for root in roots:
        for path in sorted(Path(root).glob("group_*/inputs.json")):
            directory = path.parent
            inputs = json.loads(path.read_text())
            split = inputs["declared_split"]
            if split not in splits:
                continue
            if not (directory / "complete.json").is_file():
                raise ValueError(f"incomplete v6 group {directory}")
            request = json.loads((directory / "request.json").read_text())
            source = json.loads((directory / "source.json").read_text())
            complete = json.loads((directory / "complete.json").read_text())
            outcomes = json.loads((directory / "outcomes.json").read_text())
            if require_source and source != current_source:
                raise ValueError("v6 collection source differs from the current frozen source")
            if (request["source_sha256"] != digest(source)
                    or inputs["source_sha256"] != digest(source)
                    or complete["source_sha256"] != digest(source)):
                raise ValueError("v6 source binding mismatch")
            ih = digest(inputs)
            if complete["input_sha256"] != ih or outcomes["input_sha256"] != ih:
                raise ValueError("v6 input/outcome binding mismatch")
            gid = inputs["group_id"]
            if gid in seen_ids:
                raise ValueError("duplicate v6 configuration")
            seen_ids.add(gid)
            plans = [PlanIR.from_dict(p) for p in inputs["candidates"]]
            graphs = [compile_graph(inputs["observation"], p) for p in plans]
            saved_graphs = json.loads((directory / "skill_graphs.json").read_text())
            if (saved_graphs["input_sha256"] != ih
                    or digest(saved_graphs["graphs"]) != digest(graphs)):
                raise ValueError("v6 executable graph drift")
            rows = {p.id: {} for p in plans}
            trial_hashes = {}
            for trial in outcomes["trials"]:
                cid = trial["candidate_id"]
                repeat = int(trial["trial"]["repeat"])
                if cid not in rows or repeat in rows[cid]:
                    raise ValueError("unexpected or duplicate v6 trial")
                if trial.get("valid") is not True or trial.get("timeout") or type(trial.get("success")) is not bool:
                    raise ValueError("censored/invalid trial cannot train v6")
                th = digest(trial["trial"])
                if trial["trial_sha256"] != th or (repeat in trial_hashes and trial_hashes[repeat] != th):
                    raise ValueError("v6 perturbations are not paired across candidates")
                trial_hashes[repeat] = th
                rows[cid][repeat] = trial
            expected = set(range(int(request["repeats"])))
            if set(trial_hashes) != expected or any(set(r) != expected for r in rows.values()):
                raise ValueError("v6 repeat coverage differs from the request")
            rates = np.asarray([np.mean([rows[p.id][r]["success"] for r in sorted(expected)])
                                for p in plans], np.float32)
            nominal = np.asarray([rows[p.id][0]["success"] for p in plans], np.float32)
            arrays = _load_vision(directory, inputs["vision"])
            groups.append(dict(id=gid, split=split, seed=inputs["task"]["seed"], path=str(directory),
                inputs=inputs, plans=plans, graphs=graphs,
                records=[program_record(g) for g in graphs], vision=_vision_array(arrays, view_mode),
                y=rates, nominal=nominal, outcomes=[[int(rows[p.id][r]["success"])
                    for r in sorted(expected)] for p in plans], input_sha256=ih))
    if not groups:
        raise ValueError("no complete v6 groups")
    return groups


class RobustProgramNet(nn.Module):
    def __init__(self, dim, view_mode="none"):
        super().__init__()
        if view_mode not in VIEW_MODES:
            raise ValueError(view_mode)
        self.view_mode = view_mode
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
        self.program = nn.Sequential(nn.Linear(dim, 128), nn.LayerNorm(128), nn.SiLU(),
                                     nn.Dropout(.12), nn.Linear(128, 96), nn.SiLU())
        if view_mode == "none":
            self.vision = None
            self.head = nn.Sequential(nn.Linear(96, 48), nn.SiLU(), nn.Dropout(.08), nn.Linear(48, 1))
        else:
            channels = 8 if view_mode == "both" else 4
            self.vision = nn.Sequential(
                nn.Conv2d(channels, 16, 5, stride=2, padding=2), nn.GroupNorm(4, 16), nn.SiLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GroupNorm(8, 32), nn.SiLU(),
                nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.GroupNorm(8, 48), nn.SiLU(),
                nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(), nn.Linear(192, 96), nn.SiLU())
            self.head = nn.Sequential(nn.Linear(96 * 3, 96), nn.SiLU(), nn.Dropout(.12),
                                      nn.Linear(96, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, x, image=None):
        p = self.program((x - self.mean) / self.std)
        if self.vision is None:
            return self.head(p).squeeze(-1)
        if image is None:
            raise ValueError("visual checkpoint requires RGB-D input")
        v = self.vision(image)
        return self.head(torch.cat([p, v, p * v], dim=-1)).squeeze(-1)


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -50, 50)))


@torch.inference_mode()
def infer(model, x, image, device):
    model.eval()
    xt = torch.as_tensor(x, device=device)
    it = None if image is None else torch.as_tensor(image, device=device)
    return model(xt, it).cpu().numpy().astype(float)


def _augment(image):
    if image is None:
        return None
    result = image.clone()
    # Per-sample brightness/contrast and mild sensor noise.  Geometric flips are
    # excluded because left/right and insertion order have physical meaning.
    batch, channels = result.shape[:2]
    for start in range(0, channels, 4):
        scale = .85 + .30 * torch.rand(batch, 1, 1, 1, device=result.device)
        offset = -.04 + .08 * torch.rand(batch, 1, 1, 1, device=result.device)
        result[:, start:start+3] = result[:, start:start+3] * scale + offset
        result[:, start+3:start+4] += torch.randn_like(result[:, start+3:start+4]) * .002
    result += torch.randn_like(result) * .006
    return result.clamp(0, 1)


def validation_summary(groups, logits):
    rows = []
    for g, z in zip(groups, logits):
        p = sigmoid(z); robust = (g["y"] >= .5).astype(float)
        order = np.argsort(-z, kind="stable")
        rows.append(dict(group_id=g["id"], seed=g["seed"], scores=p.tolist(),
            success_rates=g["y"].tolist(), outcomes=g["outcomes"],
            brier=float(np.mean((p-g["y"])**2)),
            accuracy_at_half=float(np.mean((p >= .5) == robust)),
            top4_has_robust_candidate=bool(robust[order[:4]].max()),
            top4_mean_success_rate=float(g["y"][order[:4]].mean())))
    return dict(brier=float(np.mean([r["brier"] for r in rows])),
                accuracy_at_half=float(np.mean([r["accuracy_at_half"] for r in rows])), rows=rows)


def train(args):
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    groups = load_groups(args.data, splits=("train", "val"), view_mode=args.view_mode)
    tr = [g for g in groups if g["split"] == "train"]
    va = [g for g in groups if g["split"] == "val"]
    if not tr or not va:
        raise ValueError("v6 training and validation configurations are required")
    encoder = InputSchema.fit([r for g in tr for r in g["records"]])
    for g in groups:
        g["x"], g["unseen"] = encoder.transform(g["records"])
    x = np.concatenate([g["x"] for g in tr])
    y = np.concatenate([g["y"] for g in tr])
    images = None if args.view_mode == "none" else np.concatenate([
        np.repeat(g["vision"][None], len(g["y"]), axis=0) for g in tr])
    model = RobustProgramNet(x.shape[1], args.view_mode).to(args.device)
    xt = torch.as_tensor(x, device=args.device)
    yt = torch.as_tensor(y, device=args.device)
    it = None if images is None else torch.as_tensor(images, device=args.device)
    model.mean.copy_(xt.mean(0)); model.std.copy_(xt.std(0, unbiased=False).clamp_min(1e-6))
    weights = torch.as_tensor(np.concatenate([np.full(len(g["y"]), 1/len(g["y"])) for g in tr]),
                              dtype=torch.float32, device=args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.015)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    split = {s: [dict(group_id=g["id"], seed=g["seed"], input_sha256=g["input_sha256"])
                 for g in groups if g["split"] == s] for s in ("train", "val")}
    saved = dict(schema=SCHEMA, seed=args.seed, view_mode=args.view_mode,
        input_schema=encoder.saved, encoding_sha256=encoding_hash(),
        interface_sha256=interface_hash(), source_sha256=digest(source_manifest()),
        split_sha256=digest(split), training=vars(args),
        target="empirical full-program success rate across paired sensing/dynamics draws")
    dump(out / "split.json", split); dump(out / "input_schema.json", encoder.saved)
    best = float("inf"); history = []
    for epoch in range(args.epochs):
        model.train(); order = np.random.permutation(len(x)); losses = []
        for begin in range(0, len(order), args.batch_size):
            ix = order[begin:begin+args.batch_size]
            optimizer.zero_grad()
            z = model(xt[ix], _augment(it[ix]) if it is not None else None)
            loss = (nn.functional.binary_cross_entropy_with_logits(z, yt[ix], reduction="none")
                    * weights[ix]).mean() * len(x) / len(tr)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
            losses.append(float(loss.detach()))
        vz = []
        for g in va:
            image = None if args.view_mode == "none" else np.repeat(g["vision"][None], len(g["y"]), axis=0)
            vz.append(infer(model, g["x"], image, args.device))
        summary = validation_summary(va, vz)
        row = dict(epoch=epoch+1, loss=float(np.mean(losses)),
                   validation_brier=summary["brier"],
                   validation_accuracy=summary["accuracy_at_half"])
        history.append(row)
        if summary["brier"] < best:
            best = summary["brier"]
            torch.save(saved | dict(state_dict=model.state_dict(), epoch=epoch+1,
                                    validation=summary), out / "best.pt")
        if epoch % 20 == 0:
            print(json.dumps(row), flush=True)
    dump(out / "history.json", history)
    chosen = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    summary = {k: v for k, v in chosen.items() if k not in ("state_dict", "input_schema")}
    summary.update(input_dim=len(encoder.keys), raw_dim=encoder.saved["raw_dim"],
        parameters=sum(p.numel() for p in model.parameters()), train_configurations=len(tr),
        validation_configurations=len(va), training_candidates=len(y),
        mean_training_success_rate=float(y.mean()), wall_seconds=time.perf_counter()-started,
        checkpoint_sha256=file_sha(out / "best.pt"))
    dump(out / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("validation", "training")}), flush=True)


class ValueScorer:
    def __init__(self, checkpoint, device="cpu"):
        start = time.perf_counter(); self.device = device
        self.saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if (self.saved.get("schema") != SCHEMA
                or self.saved["encoding_sha256"] != encoding_hash()
                or self.saved["interface_sha256"] != interface_hash()):
            raise ValueError("v6 checkpoint differs from the executable interface")
        self.encoder = InputSchema(self.saved["input_schema"])
        self.model = RobustProgramNet(len(self.encoder.keys), self.saved["view_mode"]).to(device)
        self.model.load_state_dict(self.saved["state_dict"]); self.model.eval()
        self.checkpoint_sha256 = file_sha(checkpoint)
        self.load_seconds = time.perf_counter() - start

    def rank(self, state, plans, k=4, vision=None):
        started = time.perf_counter()
        plans = [p if isinstance(p, PlanIR) else PlanIR.from_dict(p) for p in plans]
        if not plans or not 1 <= k <= len(plans) or len({p.id for p in plans}) != len(plans):
            raise ValueError("invalid v6 candidate pool")
        graphs = [compile_graph(state, p) for p in plans]
        construction = time.perf_counter() - started
        t = time.perf_counter()
        records = [program_record(g) for g in graphs]
        x, unknown, diagnostics = self.encoder.transform(records, return_diagnostics=True)
        mode = self.saved["view_mode"]
        image = None
        if mode != "none":
            if vision is None:
                raise ValueError("visual v6 checkpoint requires pre-execution RGB-D arrays")
            one = _vision_array(vision, mode)
            image = np.repeat(one[None], len(plans), axis=0)
        encoding = time.perf_counter() - t
        t = time.perf_counter(); z = infer(self.model, x, image, self.device)
        inference = time.perf_counter() - t
        p = sigmoid(z); order = np.argsort(-z, kind="stable")
        return dict(schema="twingraph.topk.v6", checkpoint_sha256=self.checkpoint_sha256,
            view_mode=mode, scores=p.tolist(), logits=z.tolist(),
            order=[plans[i].id for i in order],
            top_k=[dict(candidate_id=plans[i].id, score=float(p[i]),
                        graph_sha256=digest(graphs[i]), plan=plans[i].to_dict()) for i in order[:k]],
            unknown_input_fields=[len(u) for u in unknown], input_diagnostics=diagnostics,
            seconds=dict(graph_construction=construction, encoding=encoding,
                         inference=inference, total=time.perf_counter()-started))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", nargs="+", required=True); p.add_argument("--out", required=True)
    p.add_argument("--view-mode", choices=VIEW_MODES, default="none")
    p.add_argument("--seed", type=int, default=17); p.add_argument("--epochs", type=int, default=140)
    p.add_argument("--batch-size", type=int, default=64); p.add_argument("--lr", type=float, default=.0008)
    p.add_argument("--device", default="cuda")
    train(p.parse_args())


if __name__ == "__main__":
    main()
