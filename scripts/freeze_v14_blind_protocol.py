"""Freeze train-only baselines and the selected V14 ensemble before blind test."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from scripts.train_value_v12 import dump
from scripts.train_value_v13_mechanism import load_rows


SCHEMA = "twingraph.contextual_blind_freeze.v14.v1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ordered_metrics(rows, prefix):
    by_seed = {s: {r["name"]: r for r in rows if r["seed"] == s}
               for s in sorted({r["seed"] for r in rows})}
    hits, calls, seconds = [], [], []
    for candidates in by_seed.values():
        order = list(prefix) + [n for n in sorted(candidates) if n not in prefix]
        tried = []
        for name in order[:len(prefix)]:
            tried.append(candidates[name])
            if candidates[name]["y"]:
                break
        hits.append(bool(tried[-1]["y"]))
        calls.append(len(tried))
        seconds.append(sum(r["outcomes"][0]["seconds"] for r in tried))
    return dict(hit=float(np.mean(hits)), calls=float(np.mean(calls)),
                simulator_seconds=float(np.mean(seconds)))


def train_only_baselines(rows):
    names = sorted({r["name"] for r in rows})
    stats = {}
    for name in names:
        group = [r for r in rows if r["name"] == name]
        stats[name] = dict(success_rate=float(np.mean([r["y"] for r in group])),
                           mean_simulator_seconds=float(np.mean([r["outcomes"][0]["seconds"] for r in group])))
    global_order = sorted(names, key=lambda n: (-stats[n]["success_rate"],
                                                stats[n]["mean_simulator_seconds"], n))
    ordered_pairs = []
    for pair in itertools.permutations(names, 2):
        metric = ordered_metrics(rows, pair)
        ordered_pairs.append(((-metric["hit"], metric["calls"], metric["simulator_seconds"], pair), pair, metric))
    _, best_pair, best_pair_metric = min(ordered_pairs)
    reference_order = (["grounded_000"] if "grounded_000" in names else []) + [n for n in names if n != "grounded_000"]
    return dict(candidate_names=names, candidate_train_statistics=stats,
                global_fixed_order=global_order,
                best_fixed_top2_order=list(best_pair),
                best_fixed_top2_train_metrics=best_pair_metric,
                reference_first_order=reference_order,
                geometry_rule=dict(
                    implementation="simbench.value.system_v12.geometry_scores",
                    ordering="descending worst nominal clearance / required visual reserve; checked-stage count tie term",
                    outcome_free=True,
                    note="Uses proposal.necessary_geometry available before execution; no fitted coefficients."))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--roots", nargs="+", required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--model-lock", type=Path, required=True)
    p.add_argument("--v13-checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    split = json.loads(args.split.read_text())
    model = json.loads(args.model_lock.read_text())
    rows, _, runtime = load_rows(args.roots, split["training_layouts"])
    args.out.mkdir(parents=True, exist_ok=True)
    baselines = train_only_baselines(rows)
    dump(args.out / "BASELINE_LOCK.json", baselines)
    root = Path(__file__).resolve().parents[1]
    source_paths = [root / "scripts/train_value_v14_contextual.py",
                    root / "scripts/freeze_v14_blind_protocol.py",
                    root / "scripts/evaluate_v14_blind.py",
                    root / "scripts/run_v14_online_timing.py",
                    root / "simbench/value/graph_value_v12.py",
                    root / "simbench/value/value_v12.py",
                    root / "simbench/value/system_v12.py"]
    missing = [str(x) for x in source_paths if not x.exists()]
    if missing:
        raise FileNotFoundError(f"freeze sources missing: {missing}")
    manifest = dict(schema=SCHEMA, test_outcomes_opened=False,
                    split_sha256=sha256(args.split), split=split,
                    model_lock_sha256=sha256(args.model_lock), model=model,
                    v13_checkpoint=str(args.v13_checkpoint),
                    v13_checkpoint_sha256=sha256(args.v13_checkpoint),
                    baseline_lock_sha256=sha256(args.out / "BASELINE_LOCK.json"),
                    physical_runtime_sha256=runtime,
                    source_sha256={str(x.relative_to(root)): sha256(x) for x in source_paths},
                    claim_scope="physical execution prefix through end_stop only",
                    full_task_claim=False,
                    limitations="resource-bounded pilot; no minimum-power claim")
    dump(args.out / "FREEZE_MANIFEST.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
