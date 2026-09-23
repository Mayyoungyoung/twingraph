#!/usr/bin/env python3
"""Audit naturally varying cleaning plans from fresh randomized scenes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from simbench.value.plan import digest
from simbench.value.planner_v12 import propose
from simbench.value.system_v12 import make_scene, rollout, save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--pool-n", type=int, default=48)
    p.add_argument("--submit-n", type=int, default=12)
    p.add_argument("--level", choices=("L0", "L1", "L2"), default="L1")
    args = p.parse_args()
    if not 1 <= args.submit_n <= args.pool_n:
        p.error("submit-n must be in [1,pool-n]")
    summaries = []
    for seed in args.seeds:
        root = args.out / f"seed_{seed}"
        _, session, _, _ = make_scene(seed, root / "planning", level=args.level)
        pool, source = propose(session.decision_observation, cad=session.planning_cad,
                               n=args.pool_n, seed=seed)
        pool.sort(key=lambda row: int(row["name"].rsplit("_", 1)[-1]))
        submitted = pool[:args.submit_n]
        save(root / "request.json", dict(scope="cleaning_only", seed=seed,
            observation_sha256=digest(session.decision_observation), source=source,
            pool_n=len(pool), submitted_n=len(submitted), submitted=submitted,
            selection_rule="first N in pre-outcome construction order"))
        rows = []
        for proposal in submitted:
            directory = root / "candidates" / proposal["name"]
            try:
                result = rollout(seed, proposal, directory, level=args.level,
                                 stop_after="cleaning")
                valid = bool(result.get("valid") and not result.get("resource_censored")
                    and result.get("evaluation_scope") == "cleaning_only")
                row = dict(name=proposal["name"], valid=valid,
                    local_success=bool(result["success"]) if valid else None,
                    error=result.get("error"), result_path=str(directory / "result.json"),
                    wall_seconds=result.get("total_wall_seconds"))
            except Exception as exc:
                row = dict(name=proposal["name"], valid=False, local_success=None,
                           error=f"{type(exc).__name__}: {exc}")
            rows.append(row)
            save(root / "progress.json", rows)
            print(json.dumps(dict(seed=seed, **row)), flush=True)
        valid_rows = [row for row in rows if row["valid"]]
        positive = sum(bool(row["local_success"]) for row in valid_rows)
        summary = dict(seed=seed, scope="cleaning_only",
            status="complete" if len(valid_rows) == len(submitted) else "incomplete",
            candidates=rows, valid=len(valid_rows), successes=positive,
            pool_type=("mixed" if 0 < positive < len(valid_rows) else
                       "all_positive" if positive == len(valid_rows) and valid_rows else
                       "all_negative" if valid_rows else "unlabelled"))
        save(root / "summary.json", summary)
        summaries.append(summary)
    save(args.out / "summary.json", dict(scope="cleaning_only", layouts=summaries))


if __name__ == "__main__":
    main()
