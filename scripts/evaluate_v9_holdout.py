"""Evaluate frozen V9 screeners on the preregistered six-layout batch."""

import argparse
import json
from pathlib import Path

from scripts.analyze_v9_candidate_matrix import load, replay, aggregate_replay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--development-analysis", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cases, pool = load(args.matrix)
    seeds = sorted({key[0] for key in cases})
    if seeds != list(range(1314, 1320)):
        raise ValueError(f"selection validation seeds differ from preregistration: {seeds}")
    development_path = Path(args.development_analysis) / "summary.json"
    development = json.loads(development_path.read_text())
    split = json.loads((Path(args.development_analysis) / "split.json").read_text())
    if seeds != split["selection_validation"]:
        raise ValueError("selection validation differs from frozen development split")
    if set(seeds) & set(split["train"] + split["validation"]):
        raise ValueError("selection validation overlaps model training or early stopping layouts")
    training = development.get("model_training", {})
    checkpoints = {name: row["checkpoint"] for name, row in training.items()}
    names = [proposal["name"] for proposal in pool]
    rows = replay(cases, names, seeds, checkpoints)
    result = dict(role="external_selection_validation", seeds=seeds,
                  layouts=len(seeds), candidates=len(names), conditions=3,
                  task_version="functional_assembly_v9_funnel_r1",
                  development_analysis=str(development_path),
                  model_training=training,
                  raw_candidate_successes=sum(int(cases[(seed, condition)][name]["summary"]["success"])
                      for seed in seeds for condition in ("nominal", "light_low", "light_high") for name in names),
                  reference_successes=sum(int(cases[(seed, condition)]["reference"]["summary"]["success"])
                      for seed in seeds for condition in ("nominal", "light_low", "light_high")),
                  validation=aggregate_replay(rows))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "replay.json").write_text(json.dumps(rows, indent=2))
    (out / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(layouts=len(seeds), raw_candidate_successes=result["raw_candidate_successes"],
                          reference_successes=result["reference_successes"])), flush=True)


if __name__ == "__main__":
    main()
