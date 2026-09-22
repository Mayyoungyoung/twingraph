"""Evaluate frozen V14 methods on a uniform-N blind end-stop matrix."""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from scripts.train_value_v12 import dump, evaluate, predict
from scripts.train_value_v13_mechanism import load_rows
from scripts.train_value_v14_contextual import metrics
from simbench.value.value_v12 import StageValueNet
from simbench.value.system_v12 import geometry_scores


def load_ensemble(lock_path, device):
    lock = json.loads(Path(lock_path).read_text())
    models = []
    for path in lock["checkpoints"]:
        payload = torch.load(path, map_location=device, weights_only=False)
        model = StageValueNet(**payload["model_config"]).to(device)
        model.load_state_dict(payload["state_dict"]); model.eval(); models.append(model)
    return lock, models


def load_one(path, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = StageValueNet(**payload["model_config"]).to(device)
    model.load_state_dict(payload["state_dict"]); model.eval()
    return model


def ensemble_predict(models, rows, device):
    return np.mean([predict(model, rows, device) for model in models], axis=0)


def rank_scores(rows, order_by_seed):
    values = np.zeros(len(rows), float)
    for seed, order in order_by_seed.items():
        rank = {name: len(order) - i for i, name in enumerate(order)}
        for i, row in enumerate(rows):
            if row["seed"] == seed:
                values[i] = rank[row["name"]]
    return values


def fixed_order(rows, prefix):
    names = sorted({r["name"] for r in rows})
    order = list(prefix) + [n for n in names if n not in prefix]
    return {s: order for s in {r["seed"] for r in rows}}


def geometry_order(rows):
    answer = {}
    for seed in sorted({r["seed"] for r in rows}):
        group = [r for r in rows if r["seed"] == seed]
        scores = geometry_scores([r["graph"] for r in group])
        answer[seed] = [group[i]["name"] for i in np.argsort(-scores, kind="stable")]
    return answer


def compact(report):
    return {k: report[k] for k in ("success", "hit_at_k_given_feasible", "calls",
                                    "verification_seconds", "normalized_first_success_calls",
                                    "feasible_layout_cases", "layouts")}


def exact_random(rows, k):
    report = evaluate(rows, np.zeros(len(rows)), k)
    feasible = [case for case in report["cases"] if case["pool_success"] > 0]
    return dict(success=float(report["random_success"]),
                hit_at_k_given_feasible=(float(np.mean([case["random"]["success"] for case in feasible]))
                                         if feasible else None),
                calls=float(report["random_calls"]),
                verification_seconds=float(report["random_verification_seconds"]),
                normalized_first_success_calls=None,
                feasible_layout_cases=report["feasible_layout_cases"], layouts=report["layouts"])


def reversal_audit(rows, predictions):
    by_seed = {s: {r["name"]: int(r["y"]) for r in rows if r["seed"] == s}
               for s in sorted({r["seed"] for r in rows})}
    score = {method: {(r["seed"], r["name"]): float(values[i]) for i, r in enumerate(rows)}
             for method, values in predictions.items()}
    names = sorted(set.intersection(*(set(v) for v in by_seed.values())))
    directions = []
    for a, b in itertools.combinations(names, 2):
        a_up = [s for s, y in by_seed.items() if y[a] > y[b]]
        b_up = [s for s, y in by_seed.items() if y[b] > y[a]]
        if not a_up or not b_up:
            continue
        for seed, high, low in ([(s, a, b) for s in a_up] + [(s, b, a) for s in b_up]):
            row = dict(layout=seed, candidate_high=high, candidate_low=low)
            for method in predictions:
                row[method] = score[method][(seed, high)] > score[method][(seed, low)]
            directions.append(row)
    summary = {method: dict(correct=sum(d[method] for d in directions), total=len(directions),
                            accuracy=(float(np.mean([d[method] for d in directions])) if directions else None))
               for method in predictions}
    return directions, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--roots", nargs="+", required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--freeze", type=Path, required=True)
    p.add_argument("--baseline-lock", type=Path, required=True)
    p.add_argument("--model-lock", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(); torch.set_num_threads(1)
    split = json.loads(args.split.read_text())
    freeze = json.loads(args.freeze.read_text())
    baselines = json.loads(args.baseline_lock.read_text())
    if freeze.get("test_outcomes_opened") is not False:
        raise ValueError("invalid blind-test freeze declaration")
    rows, _, runtime = load_rows(args.roots, split["test_layouts"])
    counts = {s: sum(r["seed"] == s for r in rows) for s in split["test_layouts"]}
    if runtime != freeze["physical_runtime_sha256"] or any(n != split["candidate_count"] for n in counts.values()):
        raise ValueError(f"blind matrix violates frozen protocol: runtime={runtime}, counts={counts}")
    _, models = load_ensemble(args.model_lock, args.device)
    model_scores = ensemble_predict(models, rows, args.device)
    v13_scores = predict(load_one(freeze["v13_checkpoint"], args.device), rows, args.device)
    orders = {
        "reference_first": fixed_order(rows, baselines["reference_first_order"]),
        "global_fixed_order": fixed_order(rows, baselines["global_fixed_order"]),
        "train_best_fixed_top2": fixed_order(rows, baselines["best_fixed_top2_order"]),
        "geometry_rule": geometry_order(rows),
    }
    predictions = {"v13_graph_model": v13_scores,
                   "locked_contextual_model": model_scores}
    predictions.update({name: rank_scores(rows, order) for name, order in orders.items()})
    table = []
    full = {}
    for k in split["k_values"]:
        random = exact_random(rows, k)
        full[("uniform_random", k)] = random
        table.append(dict(method="uniform_random", k=k, **random))
        for method, scores in predictions.items():
            report = metrics(rows, scores if method == "locked_contextual_model" else scores, k)
            result = compact(report); full[(method, k)] = result
            table.append(dict(method=method, k=k, **result))
    directions, reversal = reversal_audit(rows, predictions)
    by_seed = {s: sum(r["y"] for r in rows if r["seed"] == s) for s in split["test_layouts"]}
    rankings = {}
    for method, scores in predictions.items():
        rankings[method] = {}
        for seed in split["test_layouts"]:
            indexes = [i for i, row in enumerate(rows) if row["seed"] == seed]
            order = sorted(indexes, key=lambda i: -float(scores[i]))
            rankings[method][str(seed)] = [dict(candidate=rows[i]["name"], score=float(scores[i]),
                                                        success=bool(rows[i]["y"])) for i in order]
    report = dict(schema="twingraph.contextual_blind_evaluation.v14.v1",
                  test_layout_success_counts=by_seed, fixed_candidate_count=split["candidate_count"],
                  physical_runtime_sha256=runtime, metrics=table,
                  rankings=rankings, strict_reversal=reversal, strict_reversal_directions=directions,
                  full_task_claim=False,
                  limitations="four-layout resource-bounded pilot; descriptive evidence, not a powered superiority claim")
    args.out.mkdir(parents=True, exist_ok=True); dump(args.out / "BLIND_EVALUATION.json", report)
    with (args.out / "METHOD_METRICS.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    md = ["# V14 冻结盲测", "", f"- 统一候选数：N={split['candidate_count']}。",
          f"- 测试布局成功候选数：`{by_seed}`。",
          f"- 严格跨场景反转方向数：{len(directions)}。",
          "- 成功定义：仅 fresh scene 到 end_stop 的物理执行前缀；不是完整滑台装配成功。", "",
          "| 方法 | K | Hit@K | 平均调用 | 仿真推进秒 |", "|---|---:|---:|---:|---:|"]
    for row in table:
        md.append(f"| {row['method']} | {row['k']} | {row['success']:.3f} | {row['calls']:.3f} | {row['verification_seconds']:.2f} |")
    md += ["", "样本只有 4 个独立布局，所有比较仅作受控先导实验描述，不能据此宣称统计显著或完整任务已解决。"]
    (args.out / "BLIND_EVALUATION_ZH.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
