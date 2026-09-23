#!/usr/bin/env python3
"""Train generic atomic-graph value on natural local-stage labels and compare screening.

No candidate is selected or relabelled to make a mixed pool.  Layout folds are
assigned by seed before reading outcomes.  Rejected layouts remain in the
coverage audit.  Verification time is the sum of measured serial twin trial
times, not the wall time of parallel data collection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from scripts.train_value_v15_full_task import dump, evaluate, random_calls, score, train_once
from simbench.value.generic_graph_value_v15 import (
    FEATURES, RELATIONS, SCHEMA, ValueRankerV15, encode_graph, parameter_count,
)
from simbench.value.plan import digest


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_pools(roots, *, stage="end_stop", expected_n=12):
    pools = {}
    audit = []
    for root in map(Path, roots):
        for path in sorted(root.glob("seed_*/summary.json")):
            summary = _load_json(path)
            seed = int(summary["seed"])
            if seed in pools:
                raise ValueError(f"duplicate layout seed {seed}")
            if "candidates" in summary:
                candidates = summary["candidates"]
                status = summary.get("status")
                valid_count = summary.get("valid")
                success_count = summary.get("successes")
                pool_type = summary.get("pool_type")
                stage_dir = path.parent
            else:
                stage_row = next((row for row in summary.get("stages", [])
                                  if row["stage"] == stage), None)
                candidates = stage_row.get("rows", []) if stage_row else []
                status = ("complete" if stage_row and len(candidates) == stage_row.get("valid")
                          else "incomplete")
                valid_count = stage_row.get("valid") if stage_row else 0
                success_count = stage_row.get("success") if stage_row else 0
                pool_type = stage_row.get("pool_type") if stage_row else "unreached"
                stage_dir = next((d for d in path.parent.glob(f"*_{stage}") if d.is_dir()),
                                 path.parent)
            record = dict(seed=seed, root=str(root), status=status,
                candidate_count=len(candidates), valid=valid_count,
                successes=success_count, pool_type=pool_type)
            audit.append(record)
            if len(candidates) != expected_n or status != "complete":
                continue
            rows = []
            for candidate in candidates:
                label = candidate.get("local_success", candidate.get("success"))
                if not candidate.get("valid") or label is None:
                    raise ValueError(f"invalid candidate inside complete pool: {path}")
                result_path = Path(candidate.get("result_path") or candidate.get("result") or "")
                if not result_path.is_file():
                    result_path = (stage_dir / "candidates" / candidate["name"] / "result.json"
                                   if "candidates" in summary else
                                   stage_dir / candidate["name"] / "result.json")
                graph_path = result_path.with_name("input_graph.json")
                result, graph = _load_json(result_path), _load_json(graph_path)
                if result.get("evaluation_scope") != f"assembly_prefix_through_{stage}":
                    raise ValueError(f"wrong stage scope: {result_path}")
                if not result.get("valid") or bool(result.get("success")) != bool(label):
                    raise ValueError(f"candidate/rollout label mismatch: {result_path}")
                graph_hash = digest(graph)
                if result.get("input_graph_sha256") != graph_hash:
                    raise ValueError(f"executed/scored graph mismatch: {result_path}")
                seconds = float(result.get("total_wall_seconds", 0.))
                if not math.isfinite(seconds) or seconds <= 0:
                    raise ValueError(f"invalid twin time: {result_path}")
                rows.append(dict(seed=seed, name=candidate["name"],
                    y=float(bool(result["success"])), seconds=seconds,
                    encoded=encode_graph(graph), graph=graph,
                    graph_sha256=graph_hash, result_path=str(result_path)))
            if len({row["graph_sha256"] for row in rows}) != len(rows):
                raise ValueError(f"duplicate normalized graph in pool {seed}")
            pools[seed] = rows
    if not audit:
        raise ValueError("no layout summaries found")
    return pools, audit


def random_expected_time(rows, k):
    """Exact serial validation time, stopping at success or after K trials."""
    n = len(rows); m = sum(bool(row["y"]) for row in rows)
    k = min(k, n)
    if m == 0:
        return k / n * sum(row["seconds"] for row in rows)
    hit = 1. - (math.comb(n-m, k) / math.comb(n, k) if n-m >= k else 0.)
    positive_weight = hit / m
    negative_weight = sum(
        math.comb(n-1-m, j) / math.comb(n-1, j)
        if n-1-m >= j else 0.
        for j in range(k)
    ) / n
    return sum(row["seconds"] * (positive_weight if row["y"] else negative_weight)
               for row in rows)


def ordered_result(rows, order, k):
    attempted = 0; seconds = 0.; hit = False
    for index in order[:min(k, len(rows))]:
        attempted += 1
        seconds += rows[index]["seconds"]
        if rows[index]["y"]:
            hit = True
            break
    return dict(success=hit, twin_calls=attempted, twin_seconds=seconds)


def compare(pools, ranker, *, k):
    layouts = []
    for seed, rows in sorted(pools.items()):
        started = time.perf_counter()
        probabilities = ranker.score([row["graph"] for row in rows])
        ranking_seconds = time.perf_counter() - started
        order = np.argsort(-probabilities, kind="stable").tolist()
        value = ordered_result(rows, order, k)
        full = dict(success=bool(any(row["y"] for row in rows)),
                    twin_calls=len(rows), twin_seconds=sum(row["seconds"] for row in rows))
        positives = sum(bool(row["y"]) for row in rows)
        random = dict(success_probability=1. -
            (math.comb(len(rows)-positives, min(k,len(rows))) /
             math.comb(len(rows), min(k,len(rows))) if len(rows)-positives >= min(k,len(rows)) else 0.),
            twin_calls=random_calls(len(rows), positives, k),
            twin_seconds=random_expected_time(rows, k))
        layouts.append(dict(seed=seed, n=len(rows), positive=positives,
            full_twin=full, random_top_k=random,
            value_top_k={**value, "ranking_seconds":ranking_seconds,
                "total_measured_seconds":value["twin_seconds"]+ranking_seconds,
                "selected": [rows[i]["name"] for i in order[:k]],
                "selected_scores": [float(probabilities[i]) for i in order[:k]]}))
    mean = lambda f: float(np.mean([f(row) for row in layouts]))
    rng = np.random.default_rng(20260923)
    def paired_interval(values):
        values = np.asarray(values, float)
        samples = rng.integers(0, len(values), size=(5000, len(values)))
        boot = values[samples].mean(axis=1)
        return dict(mean=float(values.mean()),
            layout_bootstrap_95_percent=[float(v) for v in np.quantile(boot, [.025, .975])])
    return dict(layout_count=len(layouts), candidate_count=sorted({r["n"] for r in layouts}),
        k=k, full_twin=dict(success_rate=mean(lambda r:r["full_twin"]["success"]),
            twin_calls=mean(lambda r:r["full_twin"]["twin_calls"]),
            twin_seconds=mean(lambda r:r["full_twin"]["twin_seconds"])),
        random_top_k=dict(expected_success_rate=mean(lambda r:r["random_top_k"]["success_probability"]),
            expected_twin_calls=mean(lambda r:r["random_top_k"]["twin_calls"]),
            expected_twin_seconds=mean(lambda r:r["random_top_k"]["twin_seconds"])),
        value_top_k=dict(success_rate=mean(lambda r:r["value_top_k"]["success"]),
            twin_calls=mean(lambda r:r["value_top_k"]["twin_calls"]),
            twin_seconds=mean(lambda r:r["value_top_k"]["twin_seconds"]),
            ranking_seconds=mean(lambda r:r["value_top_k"]["ranking_seconds"]),
            total_measured_seconds=mean(lambda r:r["value_top_k"]["total_measured_seconds"])),
        paired_value_minus_random_success=paired_interval([
            float(r["value_top_k"]["success"])-r["random_top_k"]["success_probability"]
            for r in layouts]),
        paired_value_minus_random_seconds=paired_interval([
            r["value_top_k"]["total_measured_seconds"]-r["random_top_k"]["twin_seconds"]
            for r in layouts]),
        time_definition="serial sum of recorded twin rollout wall seconds; value adds graph encoding and model scoring; common planning/checkpoint cost excluded",
        layouts=layouts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, action="append", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stage", default="end_stop")
    p.add_argument("--pool-n", type=int, default=12)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--width", type=int, default=48)
    p.add_argument("--lr", type=float, default=.001)
    p.add_argument("--initializations", type=int, nargs="+", default=[7, 17, 29])
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    pools, audit = load_pools(args.root, stage=args.stage, expected_n=args.pool_n)
    mixed = {seed: rows for seed, rows in pools.items()
             if 0 < sum(bool(row["y"]) for row in rows) < len(rows)}
    fold = lambda seed: "test" if seed % 5 == 0 else "validation" if seed % 5 == 1 else "train"
    splits = {part: sorted(seed for seed in mixed if fold(seed) == part)
              for part in ("train", "validation", "test")}
    dump(args.out / "coverage_audit.json", dict(attempted=audit,
        complete_pools=len(pools), mixed_pools=len(mixed),
        all_negative=sum(not any(r["y"] for r in rows) for rows in pools.values()),
        all_positive=sum(all(r["y"] for r in rows) for rows in pools.values()),
        fold_rule="seed modulo 5: 0 test, 1 validation, 2/3/4 train",
        splits=splits))
    if any(not seeds for seeds in splits.values()):
        raise ValueError(f"need naturally mixed layouts in all predeclared folds: {splits}")
    flatten = lambda seeds: [row for seed in seeds for row in sorted(mixed[seed], key=lambda r:r["name"])]
    train_rows, validation_rows = flatten(splits["train"]), flatten(splits["validation"])
    test_rows = flatten(splits["test"])
    attempts = []
    for initialization in args.initializations:
        model, best, trace = train_once(train_rows, validation_rows,
            initialization=initialization, epochs=args.epochs, lr=args.lr,
            width=args.width, device=args.device, k=args.k)
        attempts.append(dict(initialization=initialization, model=model,
            selected_epoch=best["epoch"], selection_key=list(best["key"]),
            validation=best["report"], trace=trace))
    selected = min(attempts, key=lambda a: tuple(a["selection_key"]))
    checkpoint_path = args.out / "stage_value.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(schema=SCHEMA, state_dict=selected["model"].state_dict(),
        model_config=selected["model"].config, feature_names=FEATURES,
        relations=RELATIONS, stage=args.stage, train_seeds=splits["train"],
        validation_seeds=splits["validation"], selected_epoch=selected["selected_epoch"]),
        checkpoint_path)
    ranker = ValueRankerV15(checkpoint_path, device=args.device)
    report = dict(schema="twingraph.natural_stage_value_comparison.v1",
        stage=args.stage, value_input_schema=SCHEMA,
        task_specific_model_features=False, model_parameters=parameter_count(ranker.model),
        checkpoint=str(checkpoint_path),
        checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        splits=splits, selected_initialization=selected["initialization"],
        selected_epoch=selected["selected_epoch"],
        validation=evaluate(validation_rows, score(selected["model"], validation_rows, args.device), args.k),
        test=compare({seed:mixed[seed] for seed in splits["test"]}, ranker, k=args.k),
        attempts=[{key:value for key,value in attempt.items() if key not in ("model","trace")}
                  for attempt in attempts])
    dump(args.out / "comparison.json", report)
    for attempt in attempts:
        dump(args.out / f"trace_{attempt['initialization']}.json", attempt["trace"])
    print(json.dumps({k:v for k,v in report.items() if k not in ("attempts","validation")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
