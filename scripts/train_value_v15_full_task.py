#!/usr/bin/env python3
"""Fit the task-agnostic V15 graph value model on complete-task twin labels.

Each input root is explicitly classified as either an audited historical full
task archive or a current V15 collection.  Prefix-only and censored outcomes
are rejected instead of being silently converted to negative labels.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from simbench.value.generic_graph_value_v15 import (
    AtomicValueNetV15,
    FEATURES,
    RELATIONS,
    SCHEMA,
    collate,
    encode_graph,
    parameter_count,
)
from simbench.value.plan import digest


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _candidate_paths(root):
    root = Path(root)
    paths = sorted(root.glob("seed_*/collect/candidates/*/result.json"))
    if not paths:
        paths = sorted(root.glob("seed_*/**/candidates/*/result.json"))
    return paths


def load_rows(root, seeds, *, audited_legacy=False):
    rows, manifest = [], []
    seeds = set(map(int, seeds))
    for result_path in _candidate_paths(root):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        seed = int(result.get("seed", result_path.parts[-4].replace("seed_", "")))
        if seed not in seeds:
            continue
        graph_path = result_path.with_name("input_graph.json")
        if not graph_path.exists():
            raise ValueError(f"missing input graph: {graph_path}")
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        if not result.get("valid") or result.get("resource_censored"):
            raise ValueError(f"invalid/censored label is not training data: {result_path}")
        if result.get("input_graph_sha256") != digest(graph):
            raise ValueError(f"graph/result binding mismatch: {result_path}")
        passes = result.get("stage_passes", {})
        required = {"cleaning_pass", "assembly_pass", "functional_test_pass"}
        if not required.issubset(passes):
            raise ValueError(f"not a complete functional-task record: {result_path}")
        if audited_legacy:
            # The native V12 archive is accepted only via this explicit flag.
            # It predates the V15 evaluation_scope field but its request,
            # graphs, complete-task results, source hashes and audit report are
            # immutable and retained beside the archive.
            scope = "audited_v12_complete_task_archive"
            checked = False
        else:
            scope = result.get("evaluation_scope")
            if scope != "complete_functional_task":
                raise ValueError(f"current label lacks complete_functional_task scope: {result_path}")
            checked = True
        encoded = encode_graph(graph, check=checked)
        name = result.get("proposal", {}).get("name", result_path.parent.name)
        rows.append(dict(seed=seed, name=name, y=float(bool(result.get("full_success", result["success"]))),
                         seconds=float(result.get("total_wall_seconds", result.get("wall_seconds", 0.))),
                         encoded=encoded, graph_sha256=digest(graph), source_scope=scope,
                         result_path=str(result_path), runtime_sha256=result.get("runtime_sha256")))
        manifest.append(dict(seed=seed, name=name, graph=str(graph_path), result=str(result_path),
                             graph_file_sha256=file_sha(graph_path), result_file_sha256=file_sha(result_path),
                             graph_sha256=digest(graph), label=bool(result.get("full_success", result["success"])),
                             source_scope=scope))
    if not rows:
        raise ValueError(f"none of the requested layouts exist under {root}")
    counts = defaultdict(int)
    for row in rows:
        counts[row["seed"]] += 1
    if len(set(counts.values())) != 1:
        raise ValueError(f"incomplete unequal candidate matrices: {dict(counts)}")
    return rows, manifest


def set_normalization(model, rows):
    values = np.concatenate([row["encoded"]["x"] for row in rows], axis=0)
    mean = values.mean(0).astype(np.float32)
    scale = values.std(0).astype(np.float32)
    scale[scale < .1] = .1
    with torch.no_grad():
        model.mean.copy_(torch.as_tensor(mean, device=model.mean.device))
        model.scale.copy_(torch.as_tensor(scale, device=model.scale.device))
    return mean, scale


def groups(rows):
    out = defaultdict(list)
    for row in rows:
        out[row["seed"]].append(row)
    return {seed: sorted(group, key=lambda row: row["name"]) for seed, group in sorted(out.items())}


def layout_loss(logits, labels):
    positive = labels > .5
    if positive.any() and (~positive).any():
        classification = .5 * (F.binary_cross_entropy_with_logits(logits[positive], labels[positive]) +
                                F.binary_cross_entropy_with_logits(logits[~positive], labels[~positive]))
        differences = logits[positive][:, None] - logits[~positive][None, :]
        ranking = F.softplus(-differences).mean()
    else:
        classification = F.binary_cross_entropy_with_logits(logits, labels)
        ranking = logits.sum() * 0.
    return classification + .5 * ranking


def score(model, rows, device):
    model.eval(); output = []
    with torch.inference_mode():
        for start in range(0, len(rows), 48):
            batch_rows = rows[start:start + 48]
            batch = collate([row["encoded"] for row in batch_rows], device)
            output.extend(model(batch).sigmoid().cpu().tolist())
    return np.asarray(output, float)


def random_hit(n, positives, k):
    if positives <= 0:
        return 0.
    k = min(k, n)
    return 1. - math.comb(n - positives, k) / math.comb(n, k) if n - positives >= k else 1.


def random_calls(n, positives, k):
    # E[min(first positive position, K)] = sum_j P(no positive in first j).
    answer = 0.
    for j in range(min(k, n)):
        answer += math.comb(n - positives, j) / math.comb(n, j) if n - positives >= j else 0.
    return answer


def evaluate(rows, probabilities, k=4):
    by_seed = groups(rows); offset = 0; layouts = []
    all_y, all_p, pair_correct, pair_total = [], [], 0, 0
    for seed, group in by_seed.items():
        p = np.asarray(probabilities[offset:offset + len(group)]); offset += len(group)
        y = np.asarray([row["y"] for row in group], bool)
        order = np.argsort(-p, kind="stable")
        selected = order[:min(k, len(order))]
        hit = bool(y[selected].any())
        calls = next((i + 1 for i, index in enumerate(selected) if y[index]), len(selected))
        positives = int(y.sum())
        for a in np.flatnonzero(y):
            for b in np.flatnonzero(~y):
                pair_correct += float(p[a] > p[b]) + .5 * float(p[a] == p[b]); pair_total += 1
        original = np.arange(min(k, len(group)))
        original_hit = bool(y[original].any())
        original_calls = next((i + 1 for i, index in enumerate(original) if y[index]), len(original))
        layouts.append(dict(seed=seed, candidates=len(group), positives=positives,
            full_n_hit=bool(positives), value_hit=hit, value_calls=calls,
            value_normalized_calls=calls / len(group), random_hit=random_hit(len(group), positives, k),
            random_calls=random_calls(len(group), positives, k),
            random_normalized_calls=random_calls(len(group), positives, k) / len(group),
            original_hit=original_hit, original_calls=original_calls,
            top_k=[dict(name=group[i]["name"], score=float(p[i]), success=bool(y[i])) for i in selected]))
        all_y.extend(y.astype(float)); all_p.extend(p)
    mean = lambda key: float(np.mean([row[key] for row in layouts]))
    return dict(layout_count=len(layouts), k=k, full_n_hit=mean("full_n_hit"),
        value_hit=mean("value_hit"), value_calls=mean("value_calls"),
        value_normalized_calls=mean("value_normalized_calls"), random_hit=mean("random_hit"),
        random_calls=mean("random_calls"), random_normalized_calls=mean("random_normalized_calls"),
        original_hit=mean("original_hit"), original_calls=mean("original_calls"),
        brier=float(np.mean((np.asarray(all_p) - np.asarray(all_y)) ** 2)),
        pair_accuracy=float(pair_correct / pair_total) if pair_total else None, layouts=layouts)


def selection_key(report, epoch):
    pair = report["pair_accuracy"] if report["pair_accuracy"] is not None else -.1
    return (-report["value_hit"], report["value_normalized_calls"], -pair, report["brier"], epoch)


def dataset_audit(rows):
    by_seed = groups(rows)
    success = {seed: {row["name"] for row in group if row["y"] > .5}
               for seed, group in by_seed.items()}
    reversals = []
    for a, seed_a in enumerate(success):
        names_a = {row["name"] for row in by_seed[seed_a]}
        for seed_b in list(success)[a + 1:]:
            common = names_a & {row["name"] for row in by_seed[seed_b]}
            a_only = sorted((success[seed_a] - success[seed_b]) & common)
            b_only = sorted((success[seed_b] - success[seed_a]) & common)
            if a_only and b_only:
                reversals.append(dict(layout_a=seed_a, layout_b=seed_b,
                                      a_better_examples=a_only[:8], b_better_examples=b_only[:8],
                                      a_better_count=len(a_only), b_better_count=len(b_only)))
    return dict(layouts=len(by_seed), rows=len(rows), candidates_per_layout={str(k): len(v) for k, v in by_seed.items()},
        positives_per_layout={str(k): len(success[k]) for k in by_seed},
        feasible_layouts=sum(bool(value) for value in success.values()),
        mixed_layouts=sum(0 < len(success[k]) < len(by_seed[k]) for k in by_seed),
        strict_reversal_layout_pairs=len(reversals), reversal_examples=reversals)


def train_once(train_rows, validation_rows, *, initialization, epochs, lr, width, device, k):
    random.seed(initialization); np.random.seed(initialization); torch.manual_seed(initialization)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(initialization)
    model = AtomicValueNetV15.build(width=width).to(device)
    set_normalization(model, train_rows)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.03)
    best = None; trace = []
    grouped = groups(train_rows)
    for epoch in range(1, epochs + 1):
        model.train(); order = list(grouped); random.Random(initialization * 10000 + epoch).shuffle(order)
        losses = []
        for seed in order:
            rows = grouped[seed]
            labels = torch.as_tensor([row["y"] for row in rows], dtype=torch.float32, device=device)
            logits = model(collate([row["encoded"] for row in rows], device))
            loss = layout_loss(logits, labels)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step(); losses.append(float(loss.detach().cpu()))
        report = evaluate(validation_rows, score(model, validation_rows, device), k)
        key = selection_key(report, epoch)
        trace.append(dict(epoch=epoch, loss=float(np.mean(losses)), validation={k: v for k, v in report.items() if k != "layouts"}))
        if best is None or key < best["key"]:
            best = dict(key=key, epoch=epoch, state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        report=report)
    model.load_state_dict(best["state"])
    return model, best, trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-root", action="append", default=[])
    parser.add_argument("--current-root", action="append", default=[])
    parser.add_argument("--train-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--validation-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--initializations", nargs="+", type=int, default=[1501, 1502, 1503])
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    train_rows = []; validation_rows = []; manifest = []
    for root, legacy in [*((root, True) for root in args.legacy_root), *((root, False) for root in args.current_root)]:
        rows, rows_manifest = load_rows(root, set(args.train_seeds) | set(args.validation_seeds), audited_legacy=legacy)
        train_rows.extend(row for row in rows if row["seed"] in args.train_seeds)
        validation_rows.extend(row for row in rows if row["seed"] in args.validation_seeds)
        manifest.extend(rows_manifest)
    if not train_rows or not validation_rows:
        raise ValueError("both train and validation rows are required")
    train_rows.sort(key=lambda row: (row["seed"], row["name"]))
    validation_rows.sort(key=lambda row: (row["seed"], row["name"]))
    if set(args.train_seeds) & set(args.validation_seeds):
        raise ValueError("layout train/validation split overlaps")
    available_train = {row["seed"] for row in train_rows}
    available_validation = {row["seed"] for row in validation_rows}
    if available_train != set(args.train_seeds) or available_validation != set(args.validation_seeds):
        raise ValueError(f"requested split is incomplete: train={sorted(available_train)}, "
                         f"validation={sorted(available_validation)}")
    attempts = []; started = time.time()
    for initialization in args.initializations:
        model, best, trace = train_once(train_rows, validation_rows, initialization=initialization,
            epochs=args.epochs, lr=args.lr, width=args.width, device=args.device, k=args.k)
        attempt_dir = out / f"init_{initialization}"
        dump(attempt_dir / "training_trace.json", trace)
        checkpoint = dict(schema=SCHEMA, state_dict=model.state_dict(), model_config=model.config,
            feature_names=FEATURES, relations=RELATIONS, initialization=initialization,
            selected_epoch=best["epoch"], validation=best["report"], train_seeds=args.train_seeds,
            validation_seeds=args.validation_seeds, k=args.k)
        torch.save(checkpoint, attempt_dir / "checkpoint.pt")
        attempts.append(dict(initialization=initialization, selected_epoch=best["epoch"],
                             selection_key=list(best["key"]), validation=best["report"],
                             checkpoint=str(attempt_dir / "checkpoint.pt")))
    selected = min(attempts, key=lambda row: tuple(row["selection_key"]))
    selected_path = Path(selected["checkpoint"])
    final_path = out / "value_v15_selected.pt"
    final_path.write_bytes(selected_path.read_bytes())
    saved = torch.load(final_path, map_location=args.device, weights_only=False)
    final_model = AtomicValueNetV15.build(**saved["model_config"]).to(args.device)
    final_model.load_state_dict(saved["state_dict"]); final_model.eval()
    train_report = evaluate(train_rows, score(final_model, train_rows, args.device), args.k)
    validation_report = evaluate(validation_rows, score(final_model, validation_rows, args.device), args.k)
    report = dict(schema="twingraph.value_training.v15.r1", value_input_schema=SCHEMA,
        task_agnostic_input=True, full_task_labels_only=True, audited_legacy_roots=args.legacy_root,
        current_roots=args.current_root, train_seeds=args.train_seeds, validation_seeds=args.validation_seeds,
        candidate_count=sorted({len(v) for v in groups(train_rows).values()}), k=args.k,
        feature_count=len(FEATURES), relation_count=len(RELATIONS), parameter_count=parameter_count(final_model),
        device=args.device, elapsed_seconds=time.time() - started, attempts=attempts, selected=selected,
        checkpoint=str(final_path), checkpoint_sha256=file_sha(final_path),
        train_dataset=dataset_audit(train_rows), validation_dataset=dataset_audit(validation_rows),
        train=train_report, validation=validation_report)
    dump(out / "training_report.json", report)
    dump(out / "input_manifest.json", manifest)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
