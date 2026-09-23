#!/usr/bin/env python3
"""Audit a frozen full-task matrix, train if supported, and report same-pool screening.

The fold rule, model settings and Top-K budget are fixed before reading labels.
The comparison reuses complete physical candidate results from each held-out
pool; recorded concurrent trial times are serial-sum proxies, not an online
wall-clock measurement.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

from scripts.audit_full_task_pools_v20 import audit_seed
from simbench.value.full_flow_graph_value_v20 import SCHEMA
from simbench.value.provenance_v12 import fingerprint


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifests = [json.loads((root / "freeze_manifest.json").read_text(encoding="utf-8"))
                 for root in args.root]
    runtime = fingerprint()["sha256"]
    if any(manifest["runtime_sha256"] != runtime for manifest in manifests):
        raise RuntimeError("finalization runtime differs from a frozen physical runtime")
    if len({(manifest["candidate_count"], manifest["level"], manifest["domain"])
            for manifest in manifests}) != 1:
        raise ValueError("frozen matrix protocol differs across roots")
    audits = []
    for root, manifest in zip(args.root, manifests):
        if [row["seed"] for row in manifest["layouts"]] != manifest["seeds"]:
            raise ValueError("freeze manifest layout order/seed mismatch")
        audits.extend(audit_seed(root / f"seed_{seed}" / "collect") for seed in manifest["seeds"])
    if len({row["seed"] for row in audits}) != len(audits):
        raise ValueError("duplicate layout seed across frozen roots")
    counts = Counter(row["pool_type"] for row in audits)
    complete = [row for row in audits if row["pool_type"] != "incomplete"]
    mixed = [row for row in complete if row["pool_type"] == "mixed"]
    fold = lambda seed: "test" if seed % 5 == 0 else "validation" if seed % 5 == 1 else "train"
    folds = {name: [row["seed"] for row in mixed if fold(row["seed"]) == name]
             for name in ("train", "validation", "test")}
    status = dict(schema="twingraph.natural_full_task_matrix_finalization.v20.r1",
                  task_scope="complete_functional_task", graph_input_schema=SCHEMA,
                  runtime_sha256=runtime, frozen_roots=[str(root) for root in args.root],
                  attempted_layouts=len(audits),
                  evaluable_layouts=len(complete), confirmed_mixed_layouts=len(mixed),
                  natural_mixed_fraction_of_evaluable=len(mixed)/len(complete) if complete else None,
                  confirmed_mixed_fraction_of_attempted=len(mixed)/len(audits),
                  pool_types=dict(counts), mixed_seeds_by_fold=folds,
                  layouts=[dict(seed=row["seed"], generated=row["generated"],
                                valid=row["valid"], full_success=row["success"],
                                pool_type=row["pool_type"])
                           for row in audits],
                  timing_note="parallel collection trial wall times support serial-sum counterfactual proxies only")
    save(args.out / "coverage_summary.json", status)
    if not all(folds.values()):
        status["state"] = "coverage_insufficient_for_training"
        save(args.out / "finalization_status.json", status)
        print(json.dumps({key: status[key] for key in
            ("state", "pool_types", "mixed_seeds_by_fold")}), flush=True)
        return
    command = [sys.executable, "scripts/train_value_v20_full_flow.py"]
    for root in args.root:
        command.extend(("--root", str(root)))
    command.extend(("--out", str(args.out / "value_training"),
               "--k", "4", "--epochs", "80", "--width", "64",
               "--lr", "0.001", "--initializations", "7", "17", "29", "--device", "cpu"))
    with (args.out / "training.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        status.update(state="training_failed", training_exit_code=result.returncode)
        save(args.out / "finalization_status.json", status)
        raise RuntimeError(f"full-task value training failed: {args.out / 'training.log'}")
    comparison = json.loads((args.out / "value_training" / "comparison.json").read_text(encoding="utf-8"))
    status.update(state="same_pool_comparison_complete",
                  checkpoint=comparison["checkpoint"],
                  checkpoint_sha256=comparison["checkpoint_sha256"],
                  conditional_test={key: comparison["conditional_test"][key]
                                    for key in ("layout_count", "full_twin", "random_top_k", "value_top_k")},
                  all_layout_test={key: comparison["all_layout_test"][key]
                                   for key in ("layout_count", "full_twin", "random_top_k", "value_top_k")})
    save(args.out / "finalization_status.json", status)
    print(json.dumps({key: status[key] for key in
        ("state", "pool_types", "mixed_seeds_by_fold", "conditional_test", "all_layout_test")},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
