"""Collect old/new script pools on preregistered unseen L1 layouts."""

import argparse
import contextlib
import json
from pathlib import Path
import time

from scripts.collect_v9_candidate_matrix import collect
from simbench.value.v9_candidates import proposals as old_proposals
from simbench.value.v10_candidates import proposals as new_proposals


SEEDS = (1320, 1321, 1322)
NEW_NAMES = (
    "reference", "carriage_yaw90", "carriage_grasp_low",
    "pin_left_yaw90", "pin_left_grasp_high", "pin_left_yaw45",
    "pin_left_grasp_low", "handle_yaw90", "handle_grasp_high",
    "handle_grasp_low", "handle_joint_approach", "handle_yaw90_joint",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = parser.parse_args()
    if not set(args.seeds) <= set(SEEDS):
        raise ValueError("only preregistered seeds are allowed")
    pools = {"old_v9": old_proposals(),
             "new_v10": [p for p in new_proposals() if p["name"] in NEW_NAMES]}
    assert tuple(p["name"] for p in pools["new_v10"]) == NEW_NAMES
    assert all(len(pool) == 12 for pool in pools.values())
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pools.json").write_text(json.dumps(pools, indent=2), encoding="utf-8")
    for seed in args.seeds:
        for pool_name, pool in pools.items():
            for proposal in pool:
                name = proposal["name"]
                directory = args.out / f"seed_{seed}" / pool_name / name
                if (directory / "result.json").exists():
                    continue
                directory.parent.mkdir(parents=True, exist_ok=True)
                started = time.perf_counter()
                with (directory.parent / f"{name}.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
                    try:
                        row = collect(seed, proposal, "nominal", directory)
                        row["source"] = proposal["source"]
                    except Exception as exc:
                        row = dict(seed=seed, pool=pool_name, candidate=name,
                                   infrastructure_error=repr(exc))
                row["pool"] = pool_name
                row["total_wall_seconds"] = time.perf_counter() - started
                summary_path = args.out / "summary.json"
                rows = json.loads(summary_path.read_text())["rows"] if summary_path.exists() else []
                rows.append(row)
                summary_path.write_text(json.dumps(dict(seeds=SEEDS, condition="nominal", rows=rows), indent=2), encoding="utf-8")
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
