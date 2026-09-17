"""Select v6 on validation data and report held-out robust test metrics."""
import argparse
import json
from pathlib import Path

import numpy as np

from simbench.value.collect import dump
from simbench.value.value_v6 import ValueScorer, load_groups


def balanced_accuracy(y, pred):
    values = []
    for label in (0, 1):
        mask = y == label
        if mask.any():
            values.append(float(np.mean(pred[mask] == label)))
    return float(np.mean(values))


def threshold_from_rows(rows):
    p = np.concatenate([np.asarray(r["scores"], float) for r in rows])
    rates = np.concatenate([np.asarray(r["success_rates"], float) for r in rows])
    candidates = np.unique(np.r_[0., p, 1.])
    # Brier is naturally defined on the empirical success-rate target. A
    # threshold turns probabilities into a hard 0/1 forecast, so choose the
    # threshold minimizing that same squared-error criterion. Balanced
    # accuracy is intentionally not used for threshold selection.
    choices = []
    for threshold in candidates:
        hard = (p >= threshold).astype(float)
        brier = float(np.mean((hard - rates) ** 2))
        choices.append((brier, abs(float(threshold)-.5), float(threshold)))
    return min(choices)[-1]


def auc(y, score):
    pos = score[y == 1]; neg = score[y == 0]
    if not len(pos) or not len(neg):
        return None
    return float(np.mean(pos[:, None] > neg[None, :]) + .5*np.mean(pos[:, None] == neg[None, :]))


def evaluate(checkpoint, data, device):
    saved = __import__("torch").load(checkpoint, map_location="cpu", weights_only=False)
    mode = saved["view_mode"]
    threshold = threshold_from_rows(saved["validation"]["rows"])
    groups = load_groups([data], splits=("test",), view_mode=mode)
    scorer = ValueScorer(checkpoint, device)
    rows = []
    all_y = []; all_p = []; all_rates = []
    for group in groups:
        arrays = None
        if mode != "none":
            # load_groups has already verified and normalized this observation;
            # scorer accepts the original named arrays.
            manifest = group["inputs"]["vision"]
            with np.load(Path(group["path"]) / manifest["file"], allow_pickle=False) as value:
                arrays = {key: value[key] for key in value.files}
        result = scorer.rank(group["inputs"]["observation"], group["plans"],
                             k=min(4, len(group["plans"])), vision=arrays)
        p = np.asarray(result["scores"], float)
        rates = group["y"]; y = (rates >= .5).astype(int)
        order = np.argsort(-p, kind="stable")
        successes = int(y.sum()); n = len(y); k = min(4, n)
        random_hit = 1. if n-successes < k else 1-float(np.prod([(n-successes-i)/(n-i) for i in range(k)]))
        rows.append(dict(group_id=group["id"], seed=group["seed"], candidates=n,
            robust_positives=successes, scores=p.tolist(), success_rates=rates.tolist(),
            outcomes=group["outcomes"], top4_indices=order[:k].tolist(),
            top4_hit=bool(y[order[:k]].max()),
            top4_mean_success_rate=float(rates[order[:k]].mean()),
            pool_mean_success_rate=float(rates.mean()), random_top4_hit_expectation=random_hit,
            scoring_seconds=result["seconds"]))
        all_y.append(y); all_p.append(p); all_rates.append(rates)
    y = np.concatenate(all_y); p = np.concatenate(all_p); rates = np.concatenate(all_rates)
    pred = p >= threshold
    return dict(checkpoint=str(checkpoint), view_mode=mode, threshold=threshold,
        threshold_selection="minimum validation hard-forecast Brier to empirical success rate; distance-to-0.5 then lower threshold tie break",
        test_configurations=len(rows), test_candidates=len(y),
        accuracy=float(np.mean(pred == y)), balanced_accuracy=balanced_accuracy(y, pred),
        roc_auc=auc(y, p), brier_to_empirical_rate=float(np.mean((p-rates)**2)),
        mean_empirical_success_rate=float(rates.mean()),
        top4_hit_rate=float(np.mean([r["top4_hit"] for r in rows])),
        top4_mean_success_rate=float(np.mean([r["top4_mean_success_rate"] for r in rows])),
        pool_mean_success_rate=float(np.mean([r["pool_mean_success_rate"] for r in rows])),
        random_top4_hit_expectation=float(np.mean([r["random_top4_hit_expectation"] for r in rows])),
        rows=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", required=True); p.add_argument("--data", required=True)
    p.add_argument("--out", required=True); p.add_argument("--device", default="cuda")
    a = p.parse_args()
    summaries = []
    for path in sorted(Path(a.models).glob("*/summary.json")):
        row = json.loads(path.read_text())
        summaries.append(dict(name=path.parent.name, checkpoint=str(path.parent / "best.pt"),
            view_mode=row["view_mode"], seed=row["seed"],
            validation_brier=row["validation"]["brier"],
            validation_accuracy=row["validation"]["accuracy_at_half"], epoch=row["epoch"]))
    if not summaries:
        raise ValueError("no v6 model summaries")
    # Seed selection is within each declared view ablation; final deployment is
    # the view ablation with the lowest validation Brier.  Test is never opened
    # by this rule.
    best_by_view = {}
    for row in summaries:
        previous = best_by_view.get(row["view_mode"])
        if previous is None or (row["validation_brier"], row["name"]) < (previous["validation_brier"], previous["name"]):
            best_by_view[row["view_mode"]] = row
    selected = min(best_by_view.values(), key=lambda row: (row["validation_brier"], row["name"]))
    tests = {mode: evaluate(row["checkpoint"], a.data, a.device)
             for mode, row in sorted(best_by_view.items())}
    result = dict(schema="twingraph.value_v6.evaluation", selection_rule=
        "minimum configuration-mean validation Brier; model-name tie break; threshold minimum validation hard-forecast Brier; no test retuning",
        selected=selected["name"], selected_view_mode=selected["view_mode"],
        all_models=summaries, best_by_view=best_by_view, held_out_test=tests,
        selected_test=tests[selected["view_mode"]])
    dump(a.out, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
