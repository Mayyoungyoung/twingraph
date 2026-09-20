"""Freeze the validation-selected V9 method before target execution."""

import argparse
import hashlib
import json
from pathlib import Path


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    analysis = Path(args.analysis)
    summary_path = analysis / "summary.json"
    summary = json.loads(summary_path.read_text())
    choices = sorted(summary["validation"], key=lambda row: (
        -row["success_rate"], row["mean_full_system_wall_seconds"], row["k"], row["method"]))
    best = choices[0]
    model = summary.get("model_training", {}).get(best["method"])
    checkpoint = model["checkpoint"] if model else None
    selection = dict(method=best["method"], k=best["k"],
        validation_success_rate=best["success_rate"],
        validation_mean_full_system_wall_seconds=best["mean_full_system_wall_seconds"],
        selection_rule="maximize grouped validation success; tie: minimize full system wall; tie: smaller K",
        selection_validation_summary_sha256=file_hash(summary_path),
        checkpoint=checkpoint,
        checkpoint_sha256=file_hash(checkpoint) if checkpoint else None,
        target_seeds=[1400, 1401, 1402],
        target_conditions=["nominal", "light_low", "light_high"])
    Path(args.out).write_text(json.dumps(selection, indent=2))
    print(json.dumps(selection))


if __name__ == "__main__":
    main()
