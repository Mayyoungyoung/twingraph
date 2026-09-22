#!/usr/bin/env python3
"""Evaluate one frozen V15 checkpoint on a complete labelled matrix."""
import argparse
import json
from pathlib import Path

from scripts.train_value_v15_full_task import dataset_audit, dump, evaluate, load_rows, score
from simbench.value.generic_graph_value_v15 import ValueRankerV15


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--audited-legacy", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    rows, manifest = load_rows(args.root, args.seeds, audited_legacy=args.audited_legacy)
    rows.sort(key=lambda row: (row["seed"], row["name"]))
    ranker = ValueRankerV15(args.checkpoint, args.device)
    probabilities = ranker.score([json.loads(Path(row["result_path"]).with_name("input_graph.json").read_text())
                                  for row in rows])
    report = dict(schema="twingraph.value_matrix_evaluation.v15.r1", checkpoint_sha256=ranker.sha256,
                  seeds=args.seeds, dataset=dataset_audit(rows), metrics=evaluate(rows, probabilities, args.k))
    dump(args.out, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
