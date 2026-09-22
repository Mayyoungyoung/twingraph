"""Train or evaluate complete, hash-bound V12 printed-kit candidate matrices.

The split JSON must be created before the experiment. Test layouts are only
opened by --evaluate-test, which never trains or selects a checkpoint.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from scripts.train_value_v12 import SELECTION_CRITERION, dump, evaluate, local_labels, predict, selection_key
from simbench.value.graph_value_v12 import SCHEMA, STAGES, ValueRankerV12, encode_graph
from simbench.value.plan import digest
from simbench.value.value_v12 import MODEL_KINDS, LOSS_CONFIG, StageValueNet, collate, supervised_loss

DEFAULT_SPLITS = dict(train=list(range(2000, 2032)), validation=list(range(2100, 2108)), test=list(range(2200, 2208)), k=4)
# r6 online was interrupted and reviewed. All older layouts are development
# material; source changes never make their outcomes blind again.
NONBLIND_LAYOUTS = frozenset((*range(1600, 1715), *range(1800, 1832)))


def validate_splits(splits):
    seen = set()
    for key in ("train", "validation", "test"):
        values = splits.get(key, [])
        if any(not isinstance(x, int) for x in values) or len(values) != len(set(values)):
            raise ValueError(f"invalid or duplicate layout IDs in {key}")
        if seen.intersection(values):
            raise ValueError("layout leakage across split groups")
        seen.update(values)
    if not isinstance(splits.get("k", 4), int) or splits.get("k", 4) < 1:
        raise ValueError("invalid predeclared K")
    for group, count in splits.get("candidate_counts", {}).items():
        if group not in ("train", "validation", "test") or not isinstance(count, int) or count < splits.get("k", 4):
            raise ValueError("invalid predeclared per-split candidate count")
    reused = NONBLIND_LAYOUTS.intersection(splits.get("test", ()))
    if reused:
        raise ValueError(f"development or retired layouts cannot be an unseen test set: {sorted(reused)}")


def validate_checkpoint_runtime(saved, manifest):
    """A native fit and its held-out labels must use the same physics source."""
    if saved.get("label_domain") == "printed_kit_v12":
        test_runtimes = {record["runtime_sha256"] for record in manifest}
        if test_runtimes != {saved.get("physical_runtime_sha256")}:
            raise ValueError("native V12 checkpoint and test labels come from different or unbound physical runtimes")


def validate_training_declaration(splits, epochs):
    expected = dict(model_families=list(MODEL_KINDS), training_seed=1212, epochs=120, width=48,
                    selection_criterion=SELECTION_CRITERION)
    for key, value in expected.items():
        if splits.get(key) != value:
            raise ValueError(f"predeclared {key} does not match fixed value training implementation")
    if epochs != splits["epochs"]:
        raise ValueError("requested epochs differs from predeclared training budget")


def load_matrices(roots, allowed_seeds, *, require_all=True):
    """Refuse missing/extra labels, drifted graph bindings and partial pools."""
    rows, manifest, seen, runtimes = [], [], set(), set()
    allowed = set(allowed_seeds)
    for root in roots:
        paths = list(Path(root).glob("**/collect/request.json"))
        if Path(root).name == "collect" and (Path(root)/"request.json").exists():
            paths.append(Path(root)/"request.json")
        for request_path in sorted(set(paths)):
            request = json.loads(request_path.read_text())
            seed = request["seed"]
            if seed not in allowed:
                continue  # Never open test outcomes during train/validation.
            if seed in seen:
                raise ValueError(f"duplicate layout matrix {seed}; choose one frozen run")
            seen.add(seed)
            runtime = request.get("runtime_sha256")
            if not isinstance(runtime, str) or len(runtime) != 64 or any(c not in "0123456789abcdef" for c in runtime):
                raise ValueError("matrix request lacks a valid physical runtime SHA256")
            runtimes.add(runtime)
            if len(runtimes) != 1:
                raise ValueError("mixed physical runtimes across allowed layout matrices; do not combine source revisions")
            pool = request["pool"]
            names = [p["name"] for p in pool]
            if len(names) != len(set(names)) or not names:
                raise ValueError("empty or duplicate candidate pool")
            result_paths = {p.parent.name: p for p in request_path.parent.glob("candidates/*/result.json")}
            if set(result_paths) != set(names):
                raise ValueError(f"incomplete matrix layout {seed}: {len(result_paths)} labels / {len(names)} requested; cannot evaluate as full pool")
            geometries, observations, interfaces = set(), set(), set()
            for proposal in pool:
                path = result_paths[proposal["name"]]
                graph_path = path.with_name("input_graph.json")
                result, graph = json.loads(path.read_text()), json.loads(graph_path.read_text())
                if not isinstance(result.get("success"), bool) or result.get("valid") is not True:
                    raise ValueError("missing or invalid physical outcome label")
                if result.get("runtime_sha256") != runtime:
                    raise ValueError("candidate physical runtime differs from its collection request")
                if result["seed"] != seed or result.get("proposal") != proposal or graph.get("proposal") != proposal:
                    raise ValueError("candidate/request/execution mismatch")
                if result.get("input_graph_sha256") != digest(graph):
                    raise ValueError("result is not bound to archived pre-execution graph")
                if graph.get("task_geometry_version") != result.get("geometry_version"):
                    raise ValueError("graph/execution geometry version mismatch")
                geometry = result["geometry_version"]
                if request.get("geometry_version") != geometry:
                    raise ValueError("request/execution geometry version mismatch")
                observed = result["initial_observation"]["sha256"]
                if observed != request["initial_observation"]["sha256"]:
                    raise ValueError("unpaired initial RGB-D observation within matrix")
                if graph["assembly"]["observation"]["perception"]["observation_sha256"] != observed:
                    raise ValueError("value observation differs from executed initial observation")
                geometries.add(geometry); observations.add(observed)
                interfaces.add(graph["assembly"]["interface_sha256"])
                encoded = encode_graph(graph)  # Hash/port consistency validation before inference.
                ly, lm = local_labels(result)
                for boundary in result.get("boundaries", []):
                    for completed in boundary.get("completed", []):
                        stage = {"stroke": "bidirectional_stroke", "retention": "pin_retention"}.get(completed, completed)
                        if stage in STAGES:
                            ly[STAGES.index(stage)], lm[STAGES.index(stage)] = 1., 1.
                condition = result.get("domain", "train")
                duration = float(result["total_wall_seconds"])
                if not np.isfinite(duration) or duration < 0:
                    raise ValueError("invalid measured rollout duration")
                rows.append(dict(seed=seed, source=geometry, name=proposal["name"], graph=graph,
                    encoded=encoded, y=float(result["success"]), local_y=ly, local_mask=lm,
                    outcomes=[dict(condition=condition, success=result["success"], seconds=duration)]))
                manifest.append(dict(layout=seed, candidate=proposal["name"], result=str(path), graph=str(graph_path),
                    graph_sha256=digest(graph), result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    request_sha256=hashlib.sha256(request_path.read_bytes()).hexdigest(), pool_size=len(pool),
                    runtime_sha256=runtime))
            if len(geometries) != 1 or len(observations) != 1 or len(interfaces) != 1:
                raise ValueError("matrix is not paired under one geometry, observation and interface")
    if require_all and seen != allowed:
        raise ValueError(f"missing complete layout matrices: {sorted(allowed-seen)}")
    if not rows:
        raise ValueError("no complete physical matrices")
    return rows, manifest


def train(rows, splits, manifest, args):
    validate_training_declaration(splits, args.epochs)
    runtimes = {record["runtime_sha256"] for record in manifest}
    if len(runtimes) != 1:
        raise ValueError("training must bind one physical runtime")
    physical_runtime = next(iter(runtimes))
    split_rows = {s: [r for r in rows if r["seed"] in splits[s]] for s in ("train", "validation")}
    tr, va = split_rows["train"], split_rows["validation"]
    pool_sizes = {s: {seed: sum(r["seed"] == seed for r in split_rows[s]) for seed in splits[s]}
                  for s in ("train", "validation")}
    for group in pool_sizes:
        count = splits.get("candidate_counts", {}).get(group)
        if count is not None and any(n != count for n in pool_sizes[group].values()):
            raise ValueError(f"matrix pool size differs from predeclared {group} candidate count")
    mixed = [seed for seed in splits["train"] if len({r["y"] for r in tr if r["seed"] == seed}) > 1]
    if not tr or not va or not mixed:
        raise ValueError("nonempty train/validation and at least one mixed-outcome training layout required; all-failure classification is not ranking evidence")
    if not any(r["y"] > 0 for r in va):
        raise ValueError("validation contains no feasible plan; Hit@K model selection is uninformative")
    dump(args.out/"data_manifest.json", dict(schema=SCHEMA, splits=splits, records=manifest,
        labels="actual V12 printed-kit full-task physical rollout outcomes", k=splits.get("k", 4),
        mixed_training_layouts=mixed, physical_runtime_sha256=physical_runtime, test_outcomes_opened=False,
        selection_criterion=SELECTION_CRITERION, loss_config=LOSS_CONFIG, model_kinds=MODEL_KINDS,
        pool_sizes=pool_sizes, layout_weighting="equal training layout per update; validation metrics average layout cases"))
    np.savez_compressed(args.out/"encoded_dataset.npz", x=np.stack([r["encoded"]["x"] for r in rows]),
        relations=np.stack([r["encoded"]["relations"] for r in rows]), active=np.stack([r["encoded"]["active"] for r in rows]),
        y=np.asarray([r["y"] for r in rows]), local_y=np.stack([r["local_y"] for r in rows]),
        local_mask=np.stack([r["local_mask"] for r in rows]), layout=np.asarray([r["seed"] for r in rows]))
    reports = {}
    for kind in MODEL_KINDS:
        torch.manual_seed(1212)
        rng = np.random.default_rng(1212)
        model = StageValueNet(tr[0]["encoded"]["x"].shape[-1], kind=kind).to(args.device)
        x = np.concatenate([r["encoded"]["x"] for r in tr])
        model.mean.copy_(torch.as_tensor(x.mean(0), device=args.device))
        model.scale.copy_(torch.as_tensor(np.maximum(x.std(0), .1), device=args.device))
        opt = torch.optim.AdamW(model.parameters(), lr=.003 if kind == "linear" else .001, weight_decay=.03)
        best, state, history = None, None, []
        started = time.perf_counter()
        for epoch in range(args.epochs):
            losses = []
            for seed in rng.permutation(splits["train"]):
                group = [r for r in tr if r["seed"] == seed]
                model.train(); opt.zero_grad()
                prediction = model(collate([r["encoded"] for r in group], args.device))
                loss = supervised_loss(prediction, torch.as_tensor([r["y"] for r in group], dtype=torch.float32, device=args.device),
                    torch.as_tensor(np.stack([r["local_y"] for r in group]), device=args.device),
                    torch.as_tensor(np.stack([r["local_mask"] for r in group]), device=args.device),
                    local_weight=0. if kind == "linear" else LOSS_CONFIG["local_weight"])
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.); opt.step()
                losses.append(float(loss))
            scores = predict(model, va, args.device)
            utility = evaluate(va, scores, splits.get("k", 4))
            key = selection_key(utility)
            history.append(dict(epoch=epoch+1, loss=float(np.mean(losses)), validation_brier=utility["brier"],
                validation_hit_at_k=utility["success"], validation_normalized_first_success_calls=utility["normalized_first_success_calls"]))
            if best is None or key < best:
                best, chosen = key, epoch+1
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(state)
        checkpoint = args.out/f"{kind}.pt"
        torch.save(dict(schema=SCHEMA, model_config=model.config, state_dict=state,
            train_layouts=splits["train"], validation_layouts=splits["validation"], kind=kind,
            label_domain="printed_kit_v12", selected_epoch=chosen, predeclared_k=splits.get("k", 4),
            physical_runtime_sha256=physical_runtime,
            selection_criterion=SELECTION_CRITERION, loss_config=LOSS_CONFIG,
            selected_validation_key=list(best),
            manifest_sha256=hashlib.sha256((args.out/"data_manifest.json").read_bytes()).hexdigest()), checkpoint)
        reports[kind] = dict(selected_epoch=chosen, selected_validation_key=list(best), training_seconds=time.perf_counter()-started,
            **{s: evaluate(rr, predict(model, rr, args.device), splits.get("k", 4)) for s, rr in split_rows.items()})
        dump(args.out/f"{kind}_history.json", history)
        dump(args.out/"metrics.json", reports)
    selected = min(reports, key=lambda kind: selection_key(reports[kind]["validation"]))
    dump(args.out/"selection.json", dict(selected=selected, criterion=SELECTION_CRITERION, k=splits.get("k", 4),
         selected_validation_key=list(selection_key(reports[selected]["validation"])), model_order=list(MODEL_KINDS),
         checkpoint_sha256=hashlib.sha256((args.out/f"{selected}.pt").read_bytes()).hexdigest(), test_outcomes_opened=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--roots", nargs="+", required=True)
    p.add_argument("--splits", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--device", default="cuda")
    p.add_argument("--evaluate-test", action="store_true")
    p.add_argument("--checkpoint", type=Path)
    args = p.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), root/"scripts/train_value_v12.py", root/"simbench/value/graph_value_v12.py",
               root/"simbench/value/value_v12.py", root/"simbench/value/skill_graph.py",
               root/"simbench/value/plan.py", root/"simbench/assembly/library.py", root/"simbench/assembly/ports.py"]
    dump(args.out/"source_provenance.json", {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sources})
    splits = json.loads(args.splits.read_text())
    validate_splits(splits)
    allowed = splits["test"] if args.evaluate_test else splits["train"]+splits["validation"]
    rows, manifest = load_matrices(args.roots, allowed)
    dump(args.out/"split_binding.json", dict(split_file=str(args.splits), sha256=hashlib.sha256(args.splits.read_bytes()).hexdigest(), splits=splits))
    if args.evaluate_test:
        if not args.checkpoint:
            p.error("--evaluate-test requires an already frozen --checkpoint")
        scorer = ValueRankerV12(args.checkpoint, args.device)
        used = set(scorer.saved.get("train_layouts", ())) | set(scorer.saved.get("validation_layouts", ()))
        if used & set(allowed):
            raise ValueError("checkpoint training/validation layouts overlap requested test")
        validate_checkpoint_runtime(scorer.saved, manifest)
        count = splits.get("candidate_counts", {}).get("test")
        if count is not None and any(sum(r["seed"] == seed for r in rows) != count for seed in allowed):
            raise ValueError("test matrix pool size differs from predeclared candidate count")
        started = time.perf_counter()
        scores = scorer.rank([r["graph"] for r in rows])
        elapsed = time.perf_counter()-started
        report = evaluate(rows, scores, splits.get("k", 4))
        report.update(status="complete physical matrix offline paired replay; no new deployment executions",
            ranking_seconds=elapsed, checkpoint_sha256=scorer.sha256, data_manifest=manifest)
        dump(args.out/"test_replay.json", report)
    else:
        if args.checkpoint:
            p.error("checkpoint initialization is not implemented; train is a fresh deterministic fit")
        train(rows, splits, manifest, args)


if __name__ == "__main__":
    main()
