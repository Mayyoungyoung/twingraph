"""Frozen offline screening replay on preregistered V10 paired test layouts."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from scripts.analyze_v9_candidate_matrix import features, outcome, random_expectation
from scripts.collect_v10_paired_test import NEW_NAMES, SEEDS
from scripts.cv_v9_interaction_models import InteractionMLP
from simbench.value.value_v6 import RobustProgramNet
from simbench.value.v9_candidates import proposals as old_proposals
from simbench.value.v10_candidates import proposals as new_proposals


KS = (2, 4, 12)
WALL_BUDGET_SECONDS = 180.


def score_model(observation, pool, checkpoint):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    x = torch.as_tensor(np.stack([features(observation, p) for p in pool]), dtype=torch.float32)
    method = saved["method"]
    if method == "global_logistic":
        model = nn.Linear(saved["input_dim"], 1)
        model.load_state_dict(saved["state_dict"])
        forward = lambda values: model((values - saved["mean"]) / saved["std"]).squeeze(-1)
    elif method == "interaction_mlp":
        model = InteractionMLP(saved["input_dim"])
        model.load_state_dict(saved["state_dict"])
        forward = lambda values: model((values - saved["mean"]) / saved["std"])
    else:
        model = RobustProgramNet(saved["input_dim"], "none")
        model.load_state_dict(saved["state_dict"])
        forward = model
    model.eval()
    with torch.no_grad():
        scores = forward(x).numpy()
    names = [p["name"] for p in pool]
    return [names[i] for i in np.argsort(-scores, kind="stable")]


def rule_order(observation, pool):
    def key(proposal):
        name = proposal["name"]
        if name == "reference":
            return 10.
        part = next((p for p in ("carriage", "pin_left", "handle") if name.startswith(p)), "carriage")
        row = observation["objects"][part]
        quality = float(row.get("quality") or 0.)
        residual = float(row.get("fit_residual_m") or 0.)
        # Entirely observable before execution. The minor term only resolves
        # semantic variants; it never accesses outcomes or simulator truth.
        preference = .02 if "joint" in name else .01 if "yaw" in name else 0.
        return 1. - quality + min(residual, .02) * 10 + preference
    return [p["name"] for p in sorted(pool, key=lambda p: -key(p))]


def replay_budget(case, order, budget, overhead=0.):
    used = overhead
    verification = 0.
    count = 0
    success = False
    for name in order:
        row = case[name]["summary"]
        used += row["total_wall_seconds"]
        verification += row["wall_seconds"]
        count += 1
        success = bool(row["success"])
        if success or used >= budget:
            break
    return dict(success=success, verification_count=count,
                verification_wall_seconds=verification, full_system_wall_seconds=used)


def random_budget(case, names, budget, seed, overhead=0., trials=4096):
    rng = np.random.default_rng(seed)
    rows = [replay_budget(case, rng.permutation(names), budget, overhead) for _ in range(trials)]
    return dict(success=float(np.mean([r["success"] for r in rows])),
                verification_count=float(np.mean([r["verification_count"] for r in rows])),
                verification_wall_seconds=float(np.mean([r["verification_wall_seconds"] for r in rows])),
                full_system_wall_seconds=float(np.mean([r["full_system_wall_seconds"] for r in rows])),
                monte_carlo_permutations=trials)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    methods = ("global_logistic", "interaction_mlp", "v6_bce", "v6_bce_pair")
    manifest = json.loads((args.checkpoints / "manifest.json").read_text())
    assert tuple(manifest["test_layouts"]) == SEEDS
    for method in methods:
        path = args.checkpoints / manifest["models"][method]["checkpoint"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["models"][method]["sha256"]
    rows = []
    for seed in SEEDS:
        root = args.matrix / f"seed_{seed}"
        summary = json.loads((root / "summary.json").read_text())
        assert len([r for r in summary["rows"] if "infrastructure_error" not in r]) == 24
        pools = json.loads((root / "pools.json").read_text())
        paired_geometry = set()
        paired_observation = set()
        paired_physics = set()
        for pool_name in ("old_v9", "new_v10"):
            generated_start = time.perf_counter()
            generated = old_proposals() if pool_name == "old_v9" else [p for p in new_proposals() if p["name"] in NEW_NAMES]
            generation_overhead = time.perf_counter() - generated_start
            pool = pools[pool_name]
            assert generated == pool
            names = [p["name"] for p in pool]
            if pool_name == "new_v10":
                assert tuple(names) == NEW_NAMES
            case = {}
            geometry = set()
            observation_hashes = set()
            for p in pool:
                path = root / f"seed_{seed}" / pool_name / p["name"] / "result.json"
                detail = json.loads(path.read_text())
                assert detail["task_version"] == "functional_assembly_v9_funnel_r1"
                assert detail["candidate"] == p
                assert detail["result"]["valid"]
                assert detail["result"]["full_success"] == next(r["success"] for r in summary["rows"]
                    if r.get("pool") == pool_name and r.get("candidate") == p["name"])
                applied = detail["result"]["applied_parameters"]
                assert applied["friction_scale"] == applied["actuator_gain_scale"] == 1.0
                paired_physics.add((applied["geom_friction_sha256"], applied["actuator_gain_sha256"]))
                geometry.add(detail["geometry_sha256"])
                observation_hashes.add(detail["pre_execution_observation"]["sha256"])
                case[p["name"]] = dict(summary=next(r for r in summary["rows"] if r.get("pool") == pool_name and r.get("candidate") == p["name"]),
                                       observation=detail["pre_execution_observation"])
            assert len(geometry) == len(observation_hashes) == 1
            paired_geometry.update(geometry)
            paired_observation.update(observation_hashes)
            observation = case[names[0]]["observation"]
            rule_start = time.perf_counter()
            ruled = rule_order(observation, pool)
            rule_overhead = time.perf_counter() - rule_start
            orders = {"reference_first": names, "simple_rule": ruled}
            overheads = {"reference_first": generation_overhead,
                         "simple_rule": generation_overhead + rule_overhead}
            fixed = outcome(case, ["reference"], 1)
            rows.append(dict(seed=seed, pool=pool_name, method="fixed_reference", k=1,
                             mode="top_k", order=["reference"], **fixed))
            rows.append(dict(seed=seed, pool=pool_name, method="fixed_reference", k=None,
                             mode="budget_180s", order=["reference"],
                             **replay_budget(case, ["reference"], WALL_BUDGET_SECONDS)))
            for method in methods:
                start = time.perf_counter()
                orders[method] = score_model(observation, pool,
                    args.checkpoints / manifest["models"][method]["checkpoint"])
                overheads[method] = generation_overhead + time.perf_counter() - start
            for method, order in orders.items():
                for k in KS:
                    result = outcome(case, order, k, overheads[method])
                    rows.append(dict(seed=seed, pool=pool_name, method=method, k=k,
                                     mode="top_k", order=order, **result))
                rows.append(dict(seed=seed, pool=pool_name, method=method, k=None,
                                 mode="budget_180s", order=order,
                                 **replay_budget(case, order, WALL_BUDGET_SECONDS, overheads[method])))
            for k in KS:
                rows.append(dict(seed=seed, pool=pool_name, method="exact_random", k=k,
                                 mode="top_k", order="uniform_all_permutations",
                                 **random_expectation(case, names, k, generation_overhead)))
            rows.append(dict(seed=seed, pool=pool_name, method="random_mc", k=None,
                             mode="budget_180s", order="uniform_sampled_permutations",
                             **random_budget(case, names, WALL_BUDGET_SECONDS,
                                 seed + (0 if pool_name == "old_v9" else 10000), generation_overhead)))
        assert len(paired_geometry) == len(paired_observation) == len(paired_physics) == 1
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["pool"], row["method"], row["mode"], row["k"])].append(row)
    aggregate = []
    for key, group in sorted(grouped.items(), key=lambda x: str(x[0])):
        aggregate.append(dict(pool=key[0], method=key[1], mode=key[2], k=key[3],
            cases=len(group), success_rate=float(np.mean([r["success"] for r in group])),
            mean_verifications=float(np.mean([r["verification_count"] for r in group])),
            mean_verification_wall_seconds=float(np.mean([r["verification_wall_seconds"] for r in group])),
            mean_full_system_wall_seconds=float(np.mean([r["full_system_wall_seconds"] for r in group]))))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps(dict(test_seeds=SEEDS,
        budget_seconds=WALL_BUDGET_SECONDS, random_budget_permutations=4096,
        note="offline replay of frozen rankings and actual candidate timings, not deployment",
        aggregate=aggregate), indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
