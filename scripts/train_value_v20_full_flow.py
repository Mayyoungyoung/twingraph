#!/usr/bin/env python3
"""Train and compare value screening using complete functional-task labels only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from scripts.audit_full_task_pools_v20 import audit_seed
from scripts.train_and_compare_stage_value import compare
from scripts.train_value_v15_full_task import dump, evaluate, score, train_once
from simbench.value.full_flow_graph_value_v20 import (
    FEATURES, RELATIONS, SCHEMA, ValueRankerV20, encode_graph,
)
from simbench.value.plan import digest


def load_pools(roots):
    pools, audit = {}, []
    for root in map(Path, roots):
        for directory in sorted(root.glob("seed_*/collect")):
            item = audit_seed(directory)
            audit.append(item)
            seed = item["seed"]
            if seed in pools:
                raise ValueError(f"duplicate layout seed {seed}")
            if item["pool_type"] == "incomplete":
                continue
            rows = []
            for candidate in item["rows"]:
                result_path = directory / "candidates" / candidate["name"] / "result.json"
                result = json.loads(result_path.read_text(encoding="utf-8"))
                graph = json.loads(result_path.with_name("input_graph.json").read_text(encoding="utf-8"))
                seconds = float(result["total_wall_seconds"])
                if not math.isfinite(seconds) or seconds <= 0:
                    raise ValueError(f"invalid measured twin time: {result_path}")
                rows.append(dict(seed=seed, name=candidate["name"],
                    y=float(result["full_success"]), seconds=seconds,
                    graph=graph, encoded=encode_graph(graph),
                    graph_sha256=digest(graph), result_path=str(result_path),
                    runtime_sha256=result["runtime_sha256"]))
            if len({row["graph_sha256"] for row in rows}) != len(rows):
                raise ValueError(f"duplicate full-flow graph in layout {seed}")
            pools[seed] = rows
    if not audit:
        raise ValueError("no complete-task candidate requests found")
    return pools, audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, action="append", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--lr", type=float, default=.001)
    p.add_argument("--initializations", type=int, nargs="+", default=[7, 17, 29])
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    pools, audit = load_pools(args.root)
    fold = lambda seed: "test" if seed % 5 == 0 else "validation" if seed % 5 == 1 else "train"
    splits = {key: sorted(seed for seed in pools if fold(seed) == key)
              for key in ("train", "validation", "test")}
    audit_report = dict(schema="twingraph.full_flow_training_gate.v20.r1",
        all_attempted_layouts=[dict(seed=r["seed"], pool_type=r["pool_type"],
                                    generated=r["generated"], valid=r["valid"],
                                    success=r["success"], failure=r["failure"])
                               for r in audit],
        fold_rule="seed modulo 5 fixed before labels: 0 test, 1 validation, 2/3/4 train",
        splits=splits,
        full_task_mixed_layouts=sum(r["pool_type"] == "mixed" for r in audit),
        all_negative_layouts=sum(r["pool_type"] == "all_negative" for r in audit),
        all_positive_layouts=sum(r["pool_type"] == "all_positive" for r in audit),
        incomplete_layouts=sum(r["pool_type"] == "incomplete" for r in audit))
    dump(args.out / "coverage_audit.json", audit_report)
    if any(r["pool_type"] != "mixed" for r in audit):
        raise ValueError("formal value training requires naturally mixed complete-task pools in every attempted layout; see coverage_audit.json")
    runtimes = {row["runtime_sha256"] for rows in pools.values() for row in rows}
    if len(runtimes) != 1:
        raise ValueError("all train/validation/test layouts must share one frozen physical runtime")
    sizes = {len(rows) for rows in pools.values()}
    if len(sizes) != 1:
        raise ValueError("candidate pool sizes differ")
    if any(not seeds for seeds in splits.values()):
        raise ValueError("need naturally mixed complete-task layouts in all predeclared folds")
    flatten = lambda seeds: [row for seed in seeds for row in sorted(pools[seed], key=lambda r:r["name"])]
    train, validation, test = map(flatten, (splits["train"], splits["validation"], splits["test"]))
    attempts = []
    for initialization in args.initializations:
        model, best, trace = train_once(train, validation, initialization=initialization,
            epochs=args.epochs, lr=args.lr, width=args.width, device=args.device, k=args.k)
        attempts.append(dict(initialization=initialization, model=model,
            selected_epoch=best["epoch"], selection_key=list(best["key"]),
            validation=best["report"], trace=trace))
    chosen = min(attempts, key=lambda r: tuple(r["selection_key"]))
    checkpoint = args.out / "full_flow_value.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(schema=SCHEMA, state_dict=chosen["model"].state_dict(),
        model_config=chosen["model"].config, feature_names=FEATURES,
        relations=RELATIONS, label_scope="complete_functional_task",
        physical_runtime_sha256=next(iter(runtimes)),
        train_seeds=splits["train"], validation_seeds=splits["validation"],
        selected_epoch=chosen["selected_epoch"]), checkpoint)
    ranker = ValueRankerV20(checkpoint, device=args.device)
    report = dict(schema="twingraph.full_flow_value_comparison.v20.r1",
        label_scope="complete_functional_task", graph_input_schema=SCHEMA,
        graph_limit="cleaning and functional stroke remain feedback-controller interface nodes",
        runtime_sha256=next(iter(runtimes)), splits=splits,
        checkpoint=str(checkpoint), checkpoint_sha256=ranker.sha256,
        selected_initialization=chosen["initialization"],
        selected_epoch=chosen["selected_epoch"],
        validation=evaluate(validation, score(chosen["model"], validation, args.device), args.k),
        test=compare({seed:pools[seed] for seed in splits["test"]}, ranker, k=args.k))
    dump(args.out / "comparison.json", report)
    for attempt in attempts:
        dump(args.out / f"trace_{attempt['initialization']}.json", attempt["trace"])
    print(json.dumps({k:v for k,v in report.items() if k != "validation"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
