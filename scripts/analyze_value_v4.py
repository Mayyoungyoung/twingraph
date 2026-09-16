"""Audit saved candidate rankings without reading labels during inference.

The reference is a finite-repetition empirical success fraction, not a known
success probability. Bootstrap intervals resample configurations, not repeated
rollouts or checkpoints. Tolerance diagnostics never change the saved ranking.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


INTERPRETATION = (
    "Reference values are finite-repetition empirical success fractions. Near-best "
    "Hit uses the empirical pool maximum, not the unknown true optimum. A bootstrap "
    "over configurations captures observed configuration variation, not uncertainty "
    "in each candidate's success probability. Degenerate intervals at 100% are not "
    "reliability guarantees. All-failure configurations remain in feasible-hit and "
    "quality metrics; only near-best Hit excludes all-failure rows."
)


def _reference(values):
    y = np.asarray(values, dtype=float)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all():
        raise ValueError("reference must be a nonempty finite vector")
    if np.any((y < 0) | (y > 1)):
        raise ValueError("reference must contain empirical fractions in [0,1]")
    return y


def _hit_probability(n, good, k):
    """At least one marked candidate in a uniform subset without replacement."""
    return 1.0 if n - good < k else 1.0 - math.comb(n - good, k) / math.comb(n, k)


def exact_uniform_random(reference, k, epsilon=.1):
    y = _reference(reference)
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer")
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    n = len(y)
    effective_k = min(int(k), n)
    best = float(y.max())
    good = int(np.count_nonzero(y >= best - epsilon))
    feasible = int(np.count_nonzero(y > 0))
    return dict(
        effective_k=effective_k,
        hit=_hit_probability(n, good, effective_k) if best > 0 else None,
        feasible=_hit_probability(n, feasible, effective_k),
        quality=float(y.mean()),
    )


def score_diagnostics(scores, k, tolerance=1e-6):
    s = np.asarray(scores, dtype=float)
    if s.ndim != 1 or not len(s) or not np.isfinite(s).all():
        raise ValueError("scores must be a nonempty finite vector")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tie tolerance must be finite and nonnegative")
    order = np.argsort(-s, kind="stable")
    sorted_scores = s[order]
    effective_k = min(k, len(s))
    gaps = sorted_scores[:-1] - sorted_scores[1:]
    boundary = float(gaps[effective_k - 1]) if effective_k < len(s) else None
    return dict(
        score_spread=float(np.ptp(s)),
        numerically_flat=bool(np.ptp(s) <= tolerance),
        adjacent_near_tie_fraction=float(np.mean(gaps <= tolerance)) if len(gaps) else 0.,
        exact_unique_scores=int(len(np.unique(s))),
        boundary_gap=boundary,
        boundary_near_tie=bool(boundary <= tolerance) if boundary is not None else None,
    )


def configuration_key(row):
    """A v4 config_id must identify all checkpoint/trajectory siblings together."""
    if row.get("config_id") is not None:
        return str(row["config_id"])
    if row.get("seed") is not None:
        return f"{row.get('family', 'unspecified')}:{row['seed']}"
    if row.get("split_group") is not None:
        return str(row["split_group"])
    return str(row["group_id"])


def cluster_summary(rows, key, *, bootstrap_samples=5000, seed=17):
    """Equal weight for configurations, averaging eligible siblings within each."""
    clusters = {}
    for row in rows:
        value = row.get(key)
        if value is not None:
            clusters.setdefault(row["config_id"], []).append(float(value))
    values = np.asarray([np.mean(v) for _, v in sorted(clusters.items())], dtype=float)
    if not len(values):
        return dict(mean=None, bootstrap95=None, eligible_configurations=0, eligible_groups=0)
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    rng = np.random.default_rng(seed)
    # Bound transient memory for larger archives.
    means = []
    for start in range(0, bootstrap_samples, 500):
        take = min(500, bootstrap_samples - start)
        means.extend(values[rng.integers(len(values), size=(take, len(values)))].mean(1))
    return dict(
        mean=float(values.mean()),
        bootstrap95=np.quantile(means, [.025, .975]).tolist(),
        eligible_configurations=len(values),
        eligible_groups=sum(len(v) for v in clusters.values()),
    )


def _methods(payload):
    methods = payload.get("methods", payload.get("results"))
    if not isinstance(methods, dict) or not methods:
        raise ValueError("expected nonempty methods (v4) or results (v3) mapping")
    return {name: value.get("rows") if isinstance(value, dict) else value
            for name, value in methods.items()}


def analyze(payload, *, ks=(1, 2, 4), epsilon=.1, tolerance=1e-6,
            bootstrap_samples=5000, seed=17):
    if not ks or any(not isinstance(k, int) or k < 1 for k in ks):
        raise ValueError("ks must contain positive integers")
    results = {}
    reference_pools = {}
    for name, inputs in _methods(payload).items():
        if not isinstance(inputs, list) or not inputs:
            raise ValueError(f"{name}: expected a nonempty row list")
        rows = []
        seen = set()
        for source in inputs:
            group_id = str(source["group_id"])
            if group_id in seen:
                raise ValueError(f"{name}: duplicate group_id {group_id}")
            seen.add(group_id)
            y = _reference(source["reference"])
            s = np.asarray(source["scores"], dtype=float)
            if s.ndim != 1 or len(s) != len(y) or not np.isfinite(s).all():
                raise ValueError(f"{name}/{group_id}: score/reference cardinality or finiteness mismatch")
            ranking_scores = np.asarray(source.get("ranking_scores", s), dtype=float)
            if ranking_scores.shape != s.shape or not np.isfinite(ranking_scores).all():
                raise ValueError(f"{name}/{group_id}: invalid ranking scores")
            binding = (configuration_key(source), y.tolist())
            if group_id in reference_pools and reference_pools[group_id] != binding:
                raise ValueError(f"{group_id}: methods use different reference pools")
            reference_pools[group_id] = binding
            order = np.argsort(-ranking_scores, kind="stable")
            best = float(y.max())
            row = dict(group_id=group_id, config_id=binding[0], seed=source.get("seed"),
                       n=len(y), pool_type="all_failure" if best == 0 else
                       "all_success" if y.min() == 1 else "mixed",
                       empirical_best=best, empirical_pool_quality=float(y.mean()),
                       intermediate_label_fraction=float(np.mean((y > 0) & (y < 1))))
            probabilities = source.get("probability_metrics", bool(np.all((s >= 0) & (s <= 1))))
            if probabilities and np.any((s < 0) | (s > 1)):
                raise ValueError("probability metrics require scores in [0,1]")
            clipped = np.clip(s, 1.e-12, 1-1.e-12)
            row.update(brier=float(np.mean((s-y)**2)) if probabilities else None,
                       bernoulli_brier=float(np.mean(y*(1-s)**2+(1-y)*s**2)) if probabilities else None,
                       log_loss=float(np.mean(-y*np.log(clipped)-(1-y)*np.log1p(-clipped))) if probabilities else None,
                       ranking_score_spread=float(np.ptp(ranking_scores)))
            for k in dict.fromkeys(ks):
                selected = y[order[:k]]
                random = exact_uniform_random(y, k, epsilon)
                hit = bool(selected.max() >= best - epsilon) if best > 0 else None
                feasible = bool(selected.max() > 0)
                quality = float(selected.mean())
                row.update({
                    f"effective_k{k}":random["effective_k"],
                    f"hit{k}":hit, f"feasible{k}":feasible, f"quality{k}":quality,
                    f"regret{k}":float(best - selected.max()),
                    f"random_hit{k}":random["hit"],
                    f"random_feasible{k}":random["feasible"],
                    f"random_quality{k}":random["quality"],
                    f"excess_hit{k}":float(hit) - random["hit"] if hit is not None else None,
                    f"excess_feasible{k}":float(feasible) - random["feasible"],
                    f"excess_quality{k}":quality - random["quality"],
                })
                diagnostics = score_diagnostics(s, k, tolerance)
                row.update({f"boundary_gap{k}":diagnostics.pop("boundary_gap"),
                            f"boundary_near_tie{k}":diagnostics.pop("boundary_near_tie")})
                row.update(diagnostics)
            rows.append(row)
        keys = [f"{metric}{k}" for k in dict.fromkeys(ks) for metric in
                ("hit", "feasible", "quality", "regret", "random_hit", "random_feasible",
                 "random_quality", "excess_hit", "excess_feasible", "excess_quality",
                 "boundary_near_tie")]
        keys += ["score_spread", "numerically_flat", "adjacent_near_tie_fraction",
                 "intermediate_label_fraction", "empirical_pool_quality", "brier", "bernoulli_brier", "log_loss"]
        summary = {key:cluster_summary(rows, key, bootstrap_samples=bootstrap_samples, seed=seed)
                   for key in keys}
        results[name] = dict(
            groups=len(rows), configurations=len({r["config_id"] for r in rows}),
            all_failure_groups=sum(r["pool_type"] == "all_failure" for r in rows),
            all_success_groups=sum(r["pool_type"] == "all_success" for r in rows),
            mixed_groups=sum(r["pool_type"] == "mixed" for r in rows),
            summary=summary, rows=rows,
        )
    return dict(schema="twingraph.value.ranking_audit.v1", methods=results,
                settings=dict(ks=list(ks), epsilon=epsilon, tie_tolerance=tolerance,
                              bootstrap_samples=bootstrap_samples, bootstrap_seed=seed),
                aggregation="equal configuration weight; eligible sibling groups averaged within configuration",
                ranking="descending ranking_scores if supplied, otherwise scores; stable input-order tie break; tolerance is diagnostic only",
                probability_metrics="brier is squared error to empirical fractions; bernoulli_brier and log_loss average individual binary outcomes implied by those fractions; unavailable for declared nonprobability baselines",
                random_baseline="exact expectation of uniform subsets without replacement, no sampled random seed",
                reference_interpretation=INTERPRETATION,
                configuration_key_policy="config_id, else family+seed, else split_group, else group_id; v4 should provide config_id",
                source_metadata={k:v for k,v in payload.items() if k not in {"methods", "results"}})


def save_outputs(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "ranking_audit.json"
    csv_path = output / "ranking_summary.csv"
    json_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        fields = ("method", "metric", "mean", "ci95_low", "ci95_high", "groups",
                  "configurations", "eligible_groups", "eligible_configurations", "all_failure_groups")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, method in result["methods"].items():
            for metric, value in method["summary"].items():
                interval = value["bootstrap95"] or (None, None)
                writer.writerow(dict(method=name, metric=metric, mean=value["mean"],
                    ci95_low=interval[0], ci95_high=interval[1], groups=method["groups"],
                    configurations=method["configurations"], eligible_groups=value["eligible_groups"],
                    eligible_configurations=value["eligible_configurations"],
                    all_failure_groups=method["all_failure_groups"]))
    return json_path, csv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--epsilon", type=float, default=.1)
    parser.add_argument("--tie-tolerance", type=float, default=1e-6)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    payload = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    result = analyze(payload, ks=args.k, epsilon=args.epsilon, tolerance=args.tie_tolerance,
                     bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    paths = save_outputs(result, args.out)
    print(json.dumps(dict(json=str(paths[0]), csv=str(paths[1]),
                          methods=list(result["methods"]))))


if __name__ == "__main__":
    main()
