"""Freeze old-matrix rankers before inspecting any V10 paired test outcomes."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from scripts.analyze_v9_candidate_matrix import load
from scripts.cv_v9_interaction_models import METHODS, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    cases, pool = load(args.matrix)
    names = [p["name"] for p in pool]
    seeds = sorted({seed for seed, _ in cases})
    assert seeds == list(range(1302, 1314))
    args.out.mkdir(parents=True, exist_ok=True)
    records = {}
    for method in METHODS:
        path = args.out / f"{method}.pt"
        _, info = train(cases, names, seeds, method, epochs=100, checkpoint=path)
        records[method] = dict(**info, checkpoint=path.name,
                               sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    manifest = dict(training_pool="script_curated_v9_r1", training_layouts=seeds,
                    conditions=["nominal", "light_low", "light_high"],
                    candidate_names=names, test_layouts=[1320, 1321, 1322],
                    target_candidate_pool="script_targeted_grasp_v10_r1",
                    limitation="new approach_strategy is unseen in old training data and absent from v1 features",
                    models=records)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
