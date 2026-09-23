#!/usr/bin/env python3
"""Evaluate pre-execution value ranking against local physical placement labels."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simbench.value.generic_graph_value_v15 import ValueRankerV15
from simbench.value.system_v11 import save


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrices", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    model = ValueRankerV15(args.checkpoint)
    layouts = []
    seed_roots = [root for matrix in args.matrices for root in matrix.glob("seed_*")]
    if len({root.name for root in seed_roots}) != len(seed_roots):
        raise ValueError("duplicate layout id across matrices")
    for seed_root in sorted(seed_roots, key=lambda root: int(root.name.split("_")[-1])):
        summary_path = seed_root / "summary.json"
        if not summary_path.exists():
            raise RuntimeError(f"unfinished layout: {seed_root}")
        summary = read(summary_path)
        if summary["status"] != "complete":
            layouts.append(dict(layout_id=seed_root.name, status=summary["status"],
                                valid=summary.get("valid", 0),
                                successes=summary.get("successes", 0)))
            continue
        rows = summary["candidates"]
        graphs = [read(seed_root / "candidates" / row["name"] / "input_graph.json")
                  for row in rows]
        scores = [float(x) for x in model.score(graphs)]
        order = sorted(range(len(rows)), key=lambda i: (-scores[i], i))
        wins = {i for i, row in enumerate(rows) if row["local_success"]}
        top = order[:args.k]
        n = len(rows); k = min(args.k, n); positives = len(wins)
        random_hit = (1. - math.comb(n-positives, k) / math.comb(n, k)
                      if n-positives >= k else 1.)
        layouts.append(dict(layout_id=seed_root.name, status="complete",
            scope="end_stop_stable_placement_only", candidate_n=len(rows),
            successes=len(wins), pool_type=summary["pool_type"],
            scores=[dict(name=row["name"], score=scores[i],
                         local_success=row["local_success"]) for i, row in enumerate(rows)],
            top_k=[rows[i]["name"] for i in top],
            value_top_k_retains_success=bool(wins.intersection(top)),
            original_top_k_retains_success=bool(wins.intersection(range(min(args.k, len(rows))))),
            first_verified_rank=next((rank for rank, i in enumerate(order, 1) if i in wins), None),
            exact_uniform_random_top_k_hit_probability=random_hit,
            exact_uniform_random_first_success_calls=(n+1)/(positives+1) if positives else None))
    report = dict(schema="twingraph.end_stop_natural_pool_analysis.v1",
        scope="end_stop_stable_placement_only", full_task_success=None,
        checkpoint_sha256=model.sha256, k=args.k, layouts=layouts,
        completed_layouts=sum(r["status"] == "complete" for r in layouts),
        mixed_layouts=sum(r.get("pool_type") == "mixed" for r in layouts),
        value_retained=sum(r.get("value_top_k_retains_success", False) for r in layouts),
        exact_random_expected_hits=sum(r.get("exact_uniform_random_top_k_hit_probability", 0.) for r in layouts),
        value_mean_first_success_calls=(sum(r["first_verified_rank"] for r in layouts if r.get("first_verified_rank") is not None)
                                        / max(1, sum(r.get("first_verified_rank") is not None for r in layouts))),
        exact_random_mean_first_success_calls=(sum(r["exact_uniform_random_first_success_calls"] for r in layouts
                                                   if r.get("exact_uniform_random_first_success_calls") is not None)
                                              / max(1, sum(r.get("exact_uniform_random_first_success_calls") is not None
                                                           for r in layouts))),
        interpretation="Value scores are pre-execution; physical labels are read only for analysis. The V15 checkpoint was trained for the full task, so local ranking quality is exploratory.")
    save(args.out, report)
    print(json.dumps({k:v for k,v in report.items() if k != "layouts"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
