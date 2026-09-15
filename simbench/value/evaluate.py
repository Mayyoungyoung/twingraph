"""Screening metrics against repeated-rollout reference values (not a new rollout)."""
import itertools
import math
import numpy as np
import torch
from .encode import collate


def subset_metrics(reference, indices, epsilon=0.1):
    best = float(max(reference))
    selected = float(max(reference[i] for i in indices)) if len(indices) else 0.0
    return dict(
        hit=float(selected >= best - epsilon) if best > 0 else None,
        feasible=float(selected > 0),
        regret=best - selected,
        selected_reference=selected,
    )


@torch.inference_mode()
def score_group(model, group, device="cpu", chunk=8):
    outputs = []
    for start in range(0, len(group.plans), chunk):
        enc = group.encoded[start : start + chunk]
        visual = [group.visual] * len(enc) if model.config.vision else None
        raw = model(collate(enc, visual, device))
        outputs.append({k: v.cpu().numpy() for k, v in raw.items()})
    return {k: np.concatenate([row[k] for row in outputs]) for k in outputs[0]}


def evaluate(model, groups, device="cpu", k=2, epsilon=0.1, objective="dual"):
    model.eval()
    methods = {
        key: []
        for key in (
            "learned",
            "prefix_head",
            "geometric",
            "fixed_first",
            "random",
            "exhaustive",
        )
    }
    scored = []
    brier = []
    for group in groups:
        out = score_group(model, group, device)
        score = (
            out["q"] if objective == "dual" else 1 / (1 + np.exp(-out["direct_logit"]))
        )
        n = len(group.plans)
        kk = min(k, n)
        rankings = dict(
            learned=np.argsort(-score, kind="stable")[:kk],
            prefix_head=np.argsort(-out["p_prefix"], kind="stable")[:kk],
            geometric=np.argsort(
                [p.prefix["cost"] for p in group.plans], kind="stable"
            )[:kk],
            fixed_first=np.arange(kk),
            exhaustive=np.arange(n),
        )
        for method, selected in rankings.items():
            methods[method].append(subset_metrics(group.reference, selected, epsilon))
        combinations = list(itertools.combinations(range(n), kk))
        if len(combinations) > 10000:
            rng = np.random.default_rng(0)
            combinations = [rng.choice(n, kk, replace=False) for _ in range(10000)]
        random_rows = [
            subset_metrics(group.reference, idx, epsilon) for idx in combinations
        ]
        methods["random"].append(
            {
                key: float(np.mean([r[key] for r in random_rows]))
                if random_rows[0][key] is not None
                else None
                for key in random_rows[0]
            }
        )
        brier.extend((score - group.reference) ** 2)
        scored.append(
            dict(
                group_id=group.id,
                candidate_ids=[p.id for p in group.plans],
                scores=score.tolist(),
                reference=group.reference.tolist(),
                selected=[group.plans[i].id for i in rankings["learned"]],
            )
        )
    summary = {}
    for name, rows in methods.items():
        summary[name] = {
            key: float(np.mean([r[key] for r in rows if r[key] is not None]))
            if any(r[key] is not None for r in rows)
            else None
            for key in ("hit", "regret", "feasible", "selected_reference")
        }
    return dict(
        groups=len(groups),
        viable_groups=sum(g.reference.max() > 0 for g in groups),
        informative_groups=sum(g.reference.max() > g.reference.min() for g in groups),
        candidates=sum(len(g.plans) for g in groups),
        k=k,
        epsilon=epsilon,
        methods=summary,
        brier_to_empirical_rate=float(np.mean(brier)) if brier else None,
        estimated_candidate_rollout_reduction=1
        - sum(min(k, len(g.plans)) for g in groups)
        / max(1, sum(len(g.plans) for g in groups)),
        interpretation="held-out configuration screening against finite repeated-rollout rates; selected_reference is an oracle within Top-K, not measured closed-loop success",
        predictions=scored,
    )


def main():
    import argparse
    from .collect import dump
    from .dataset import load_groups, split_name
    from .rank import PlanRanker

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="results/value/evaluation.json")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.k < 1:
        parser.error("k must be positive")
    ranker = PlanRanker(args.checkpoint, args.device)
    seed = ranker.saved["training"]["seed"]
    groups = [
        group
        for group in load_groups(args.data, ranker.model.config.vision)
        if split_name(group.id, seed) == args.split
    ]
    if not groups:
        raise ValueError("requested evaluation split is empty")
    metrics = evaluate(
        ranker.model,
        groups,
        args.device,
        args.k,
        ranker.saved["training"]["epsilon"],
        ranker.saved["objective"],
    )
    dump(args.out, metrics)
    print(f"{args.split}: {len(groups)} groups; {metrics['methods']['learned']}")


if __name__ == "__main__":
    main()
