"""Grouped cross-validation on the immutable V9 matrix; no target data."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from scripts.analyze_v9_candidate_matrix import features, load, outcome, random_expectation, rule_score
from simbench.value.value_v6 import RobustProgramNet


FOLDS = ((1302, 1306, 1311), (1303, 1307, 1312),
         (1304, 1308, 1313), (1305, 1309, 1310))
METHODS = ("global_logistic", "interaction_mlp", "v6_bce", "v6_bce_pair")


class InteractionMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 16), nn.Tanh(), nn.Linear(16, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train(cases, names, seeds, method, epochs=100, checkpoint=None):
    torch.manual_seed(51)
    rows = []
    labels = []
    for seed in seeds:
        case = cases[(seed, "nominal")]
        rows.extend(features(case[name]["observation"], case[name]["proposal"]) for name in names)
        labels.extend(np.mean([cases[(seed, condition)][name]["summary"]["success"]
                               for condition in ("nominal", "light_low", "light_high")]) for name in names)
    x = torch.as_tensor(np.stack(rows), dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.float32)
    mean = x.mean(0)
    std = x.std(0, unbiased=False).clamp_min(1e-5)
    if method == "global_logistic":
        model = nn.Linear(x.shape[1], 1)
        forward = lambda values: model((values - mean) / std).squeeze(-1)
    elif method == "interaction_mlp":
        model = InteractionMLP(x.shape[1])
        forward = lambda values: model((values - mean) / std)
    else:
        model = RobustProgramNet(x.shape[1], "none")
        model.mean.copy_(mean)
        model.std.copy_(std)
        forward = model
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.03)
    pair_indices = []
    n = len(names)
    if method == "v6_bce_pair":
        for base in range(0, len(y), n):
            for i in range(n):
                for j in range(i + 1, n):
                    delta = float(y[base + i] - y[base + j])
                    # One of three perturbations is weak evidence. All-fail
                    # and all-success layouts still contribute to BCE.
                    weight = max(abs(delta) - 1 / 3, 0.)
                    if weight > 1e-6:
                        pair_indices.append((base + i, base + j, 1 if delta > 0 else -1, weight))
    started = time.perf_counter()
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        score = forward(x)
        loss = nn.functional.binary_cross_entropy_with_logits(score, y)
        if pair_indices:
            left, right, sign, weight = (torch.as_tensor(col) for col in zip(*pair_indices))
            ranking = nn.functional.softplus(-(score[left.long()] - score[right.long()]) * sign.float())
            loss = loss + .25 * (ranking * weight.float()).sum() / weight.float().sum()
        loss.backward()
        optimizer.step()
    model.eval()
    if checkpoint is not None:
        torch.save(dict(method=method, state_dict=model.state_dict(), mean=mean, std=std,
                        input_dim=x.shape[1], train_seeds=list(seeds), epochs=epochs,
                        feature_schema="v9_rgbd_xyz_quality_valid_plus_executable_parameters_v1"), checkpoint)
    return forward, dict(training_seconds=time.perf_counter() - started,
                         training_layouts=list(seeds), pair_count=len(pair_indices),
                         training_loss=float(loss.detach()))


def evaluate(cases, names, seeds, scorer):
    rows = []
    for seed in seeds:
        base = cases[(seed, "nominal")]
        x = torch.as_tensor(np.stack([features(base[name]["observation"], base[name]["proposal"])
                                      for name in names]), dtype=torch.float32)
        with torch.no_grad():
            logits = scorer(x).numpy()
        probability = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
        order = [names[i] for i in np.argsort(-logits, kind="stable")]
        for condition in ("nominal", "light_low", "light_high"):
            case = cases[(seed, condition)]
            actual = np.asarray([case[name]["summary"]["success"] for name in names], dtype=float)
            pairs = [(i, j) for i in range(len(names)) for j in range(len(names)) if actual[i] > actual[j]]
            record = dict(seed=seed, condition=condition, order=order,
                          mixed=bool(pairs), brier=float(np.mean((probability - actual) ** 2)),
                          log_loss=float(np.mean(-actual * np.log(np.clip(probability, 1e-8, 1))
                              - (1 - actual) * np.log(np.clip(1 - probability, 1e-8, 1)))),
                          pair_accuracy=float(np.mean([logits[i] > logits[j] for i, j in pairs])) if pairs else None)
            for k in (2, 4, 12):
                record[f"k{k}"] = outcome(case, order, k)
            rows.append(record)
    return rows


def summarize(rows):
    mixed = [r for r in rows if r["mixed"]]
    return dict(cases=len(rows), mixed_cases=len(mixed),
                brier=float(np.mean([r["brier"] for r in rows])),
                log_loss=float(np.mean([r["log_loss"] for r in rows])),
                mixed_pair_accuracy=float(np.mean([r["pair_accuracy"] for r in mixed])) if mixed else None,
                distinct_rankings=len({tuple(r["order"]) for r in rows}),
                distinct_top1=len({r["order"][0] for r in rows}),
                **{f"k{k}": dict(success_rate=float(np.mean([r[f"k{k}"]["success"] for r in rows])),
                     verification_count=float(np.mean([r[f"k{k}"]["verification_count"] for r in rows])),
                     full_system_wall_seconds=float(np.mean([r[f"k{k}"]["full_system_wall_seconds"] for r in rows])))
                   for k in (2, 4, 12)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    torch.set_num_threads(2)
    cases, pool = load(args.matrix)
    names = [p["name"] for p in pool]
    all_seeds = {seed for fold in FOLDS for seed in fold}
    assert all_seeds == {seed for seed, _ in cases}
    by_method = defaultdict(list)
    training = defaultdict(list)
    for fold_index, test in enumerate(FOLDS):
        train_seeds = sorted(all_seeds - set(test))
        for method in METHODS:
            scorer, info = train(cases, names, train_seeds, method, args.epochs)
            training[method].append(dict(fold=fold_index, **info))
            by_method[method].extend(evaluate(cases, names, test, scorer))
    # The same held-out layouts also define rule and exact-random comparisons.
    for seed in sorted(all_seeds):
        for condition in ("nominal", "light_low", "light_high"):
            case = cases[(seed, condition)]
            obs = case[names[0]]["observation"]
            ruled = sorted(names, key=lambda n: -rule_score(obs, case[n]["proposal"]))
            for method, order in (("reference_first", names), ("simple_rule", ruled)):
                row = dict(seed=seed, condition=condition, order=order)
                for k in (2, 4, 12):
                    row[f"k{k}"] = outcome(case, order, k)
                by_method[method].append(row)
            random = dict(seed=seed, condition=condition)
            for k in (2, 4, 12):
                random[f"k{k}"] = random_expectation(case, names, k)
            by_method["exact_random"].append(random)
    summary = {method: summarize(rows) for method, rows in by_method.items() if method in METHODS}
    for method in ("reference_first", "simple_rule", "exact_random"):
        rows = by_method[method]
        summary[method] = {f"k{k}": dict(success_rate=float(np.mean([r[f"k{k}"]["success"] for r in rows])),
            verification_count=float(np.mean([r[f"k{k}"]["verification_count"] for r in rows])),
            full_system_wall_seconds=float(np.mean([r[f"k{k}"]["full_system_wall_seconds"] for r in rows]))) for k in (2, 4, 12)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(dict(folds=FOLDS, epochs=args.epochs,
        note="offline grouped CV on old V9 pool; not an independent new-method test",
        model_training=training, methods=summary), indent=2), encoding="utf-8")
    (args.out / "rows.json").write_text(json.dumps(dict(by_method), indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
