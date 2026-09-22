"""Train V14 contextual end-stop rankers without consulting the blind test.

Variant ``base`` reproduces the V13 layout-balanced objective.  Variant
``reversal_balanced`` adds equal-weight constraints for candidate pairs whose
preferred direction reverses across *training* layouts.  Candidate names are
used only to construct those training pairs and never enter the network.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from scripts.train_value_v12 import dump, evaluate, predict
from scripts.train_value_v13_mechanism import load_rows
from simbench.value.graph_value_v12 import SCHEMA
from simbench.value.value_v12 import StageValueNet, collate, supervised_loss


TRAINING_SCHEMA = "twingraph.contextual_value_training.v14.v1"
VARIANTS = {"base": 0.0, "reversal_balanced": 2.0}


def metrics(rows, predictions, k):
    report = evaluate(rows, predictions, k)
    scores = np.asarray(predictions, float)
    by_layout = {}
    for seed in sorted({r["seed"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["seed"] == seed]
        comparisons = [scores[p] > scores[n] for p in idx for n in idx
                       if rows[p]["y"] > rows[n]["y"]]
        if comparisons:
            by_layout[str(seed)] = float(np.mean(comparisons))
    report["pair_accuracy_by_layout"] = by_layout
    report["mixed_layouts"] = len(by_layout)
    report["mean_within_layout_pair_accuracy"] = (
        float(np.mean(list(by_layout.values()))) if by_layout else 0.0)
    return report


def selection_key(report):
    return (-float(report["success"]),
            float(report["normalized_first_success_calls"]),
            -float(report["mean_within_layout_pair_accuracy"]),
            float(report["brier"]))


def reversal_targets(rows):
    """Return within-layout directions for pairs that reverse elsewhere."""
    by_seed = {s: {r["name"]: int(r["y"]) for r in rows if r["seed"] == s}
               for s in sorted({r["seed"] for r in rows})}
    names = sorted(set.intersection(*(set(v) for v in by_seed.values())))
    targets = {s: [] for s in by_seed}
    pairs = []
    for a, b in itertools.combinations(names, 2):
        positive_a = [s for s, y in by_seed.items() if y[a] > y[b]]
        positive_b = [s for s, y in by_seed.items() if y[b] > y[a]]
        if not positive_a or not positive_b:
            continue
        pairs.append(dict(candidate_a=a, candidate_b=b,
                          a_above_layouts=positive_a, b_above_layouts=positive_b))
        for seed in positive_a:
            targets[seed].append((a, b))
        for seed in positive_b:
            targets[seed].append((b, a))
    return targets, pairs


def reversal_loss(logits, group, directions):
    if not directions:
        return logits.sum() * 0.0
    index = {r["name"]: i for i, r in enumerate(group)}
    losses = [nn.functional.softplus(-(logits[index[high]] - logits[index[low]]))
              for high, low in directions]
    return torch.stack(losses).mean()


def fit_one(train, validation, split, variant, training_seed, epochs, device, out):
    torch.manual_seed(training_seed)
    rng = np.random.default_rng(training_seed)
    model = StageValueNet(train[0]["encoded"]["x"].shape[-1], kind="graph").to(device)
    x = np.concatenate([r["encoded"]["x"] for r in train])
    model.mean.copy_(torch.as_tensor(x.mean(0), device=device))
    model.scale.copy_(torch.as_tensor(np.maximum(x.std(0), .1), device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.03)
    targets, pairs = reversal_targets(train)
    weight = VARIANTS[variant]
    best = state = chosen = None
    history = []
    started = time.perf_counter()
    for epoch in range(epochs):
        epoch_losses = []
        for seed in rng.permutation(split["training_layouts"]):
            group = [r for r in train if r["seed"] == seed]
            model.train(); optimizer.zero_grad()
            pred = model(collate([r["encoded"] for r in group], device))
            labels = torch.as_tensor([r["y"] for r in group], dtype=torch.float32, device=device)
            zeros = torch.zeros((len(group), 8), device=device)
            ordinary = supervised_loss(pred, labels, zeros, zeros, local_weight=0.)
            reverse = reversal_loss(pred["plan_logit"], group, targets[int(seed)])
            loss = ordinary + weight * reverse
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.); optimizer.step()
            epoch_losses.append((float(ordinary.detach()), float(reverse.detach()), float(loss.detach())))
        val = metrics(validation, predict(model, validation, device), split["k_values"][1])
        key = selection_key(val)
        history.append(dict(epoch=epoch + 1,
                            ordinary_loss=float(np.mean([x[0] for x in epoch_losses])),
                            reversal_loss=float(np.mean([x[1] for x in epoch_losses])),
                            total_loss=float(np.mean([x[2] for x in epoch_losses])),
                            validation_hit_at_2=val["success"],
                            validation_normalized_calls=val["normalized_first_success_calls"],
                            validation_pair_accuracy=val["mean_within_layout_pair_accuracy"],
                            validation_brier=val["brier"]))
        if best is None or key < best:
            best, chosen = key, epoch + 1
            state = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}
    model.load_state_dict(state)
    checkpoint = out / f"{variant}_seed{training_seed}.pt"
    torch.save(dict(schema=SCHEMA, model_config=model.config, state_dict=state, kind="graph",
                    label_domain="mechanism_prefix_end_stop_v14", selected_epoch=chosen,
                    predeclared_k=2, train_layouts=split["training_layouts"],
                    validation_layouts=split["validation_layouts"],
                    training_seed=training_seed, variant=variant,
                    reversal_weight=weight, full_task_claim=False), checkpoint)
    report = dict(variant=variant, training_seed=training_seed, selected_epoch=chosen,
                  selection_key=list(best), training_seconds=time.perf_counter() - started,
                  checkpoint=str(checkpoint),
                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  reversal_pairs=len(pairs), reversal_directions=sum(map(len, targets.values())),
                  train=metrics(train, predict(model, train, device), 2),
                  validation=metrics(validation, predict(model, validation, device), 2))
    dump(out / f"{variant}_seed{training_seed}_history.json", history)
    return report, pairs


def aggregate_variant(reports):
    fields = ["success", "normalized_first_success_calls",
              "mean_within_layout_pair_accuracy", "brier"]
    validation = {f: float(np.mean([r["validation"][f] for r in reports])) for f in fields}
    validation["training_seed_runs"] = len(reports)
    return dict(validation=validation,
                checkpoint_sha256=[r["checkpoint_sha256"] for r in reports],
                selection_key=list(selection_key(validation)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--roots", nargs="+", required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(); torch.set_num_threads(1)
    split = json.loads(args.split.read_text())
    allowed = split["training_layouts"] + split["validation_layouts"]
    rows, manifest, runtime = load_rows(args.roots, allowed)
    counts = {s: sum(r["seed"] == s for r in rows) for s in allowed}
    if any(n != split["candidate_count"] for n in counts.values()):
        raise ValueError(f"V14 requires fixed N={split['candidate_count']}: {counts}")
    train = [r for r in rows if r["seed"] in split["training_layouts"]]
    validation = [r for r in rows if r["seed"] in split["validation_layouts"]]
    args.out.mkdir(parents=True, exist_ok=True)
    declaration = dict(schema=TRAINING_SCHEMA, split=split, physical_runtime_sha256=runtime,
                       epochs=args.epochs, variants=VARIANTS,
                       prohibited_model_features=["layout_seed", "candidate_name", "rollout_result", "simulator_truth"],
                       selection="variant ensemble selected by mean validation Hit@2, normalized calls, pair accuracy, Brier",
                       test_outcomes_opened=False, records=manifest)
    dump(args.out / "data_manifest.json", declaration)
    all_reports = {}; all_pairs = {}
    for variant in VARIANTS:
        variant_reports = []
        for training_seed in split["training_seeds"]:
            report, pairs = fit_one(train, validation, split, variant, training_seed,
                                    args.epochs, args.device, args.out)
            variant_reports.append(report); all_pairs[variant] = pairs
            dump(args.out / "run_metrics.json", all_reports | {variant: variant_reports})
        all_reports[variant] = variant_reports
    aggregate = {variant: aggregate_variant(reports) for variant, reports in all_reports.items()}
    selected = min(aggregate, key=lambda v: tuple(aggregate[v]["selection_key"]))
    lock = dict(schema="twingraph.contextual_value_selection.v14.v1", selected_variant=selected,
                ensemble="arithmetic mean of three predeclared training-seed probabilities",
                checkpoints=[r["checkpoint"] for r in all_reports[selected]],
                checkpoint_sha256=aggregate[selected]["checkpoint_sha256"],
                validation=aggregate[selected]["validation"],
                physical_runtime_sha256=runtime, test_outcomes_opened=False,
                full_task_claim=False)
    dump(args.out / "run_metrics.json", all_reports)
    dump(args.out / "variant_aggregate.json", aggregate)
    dump(args.out / "reversal_pairs.json", all_pairs)
    dump(args.out / "locked_model.json", lock)
    print(json.dumps(lock, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
