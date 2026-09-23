#!/usr/bin/env python3
"""Audit a natural end-stop suffix pool from one legal carriage checkpoint.

The checkpoint is obtained by normal physical execution. Every submitted
candidate is evaluated from the same checkpoint; no outcome is consulted when
choosing the pool. Labels are local stable-support outcomes, never full-task
assembly outcomes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.run_v17a_end_stop_place import obtain_carriage_checkpoint
from simbench.value.plan import digest
from simbench.value.planner_v12 import propose
from simbench.value.system_v12 import make_scene, rollout, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--pool-n", type=int, default=48)
    parser.add_argument("--submit-n", type=int, default=12)
    parser.add_argument("--level", choices=("L0", "L1", "L2"), default="L1")
    args = parser.parse_args()
    if not 1 <= args.submit_n <= args.pool_n:
        parser.error("submit-n must be in [1,pool-n]")
    summaries = []
    for seed in args.seeds:
        root = args.out / f"seed_{seed}"
        root.mkdir(parents=True, exist_ok=True)
        _, session, _, _ = make_scene(seed, root / "planning", level=args.level)
        pool, source = propose(session.decision_observation, cad=session.planning_cad,
                               n=args.pool_n, seed=seed)
        # Construction order is fixed before any physical label is observed.
        pool.sort(key=lambda p: int(p["name"].rsplit("_", 1)[-1]))
        submitted = pool[:args.submit_n]
        save(root / "request.json", dict(scope="end_stop_stable_placement_only",
             full_task_success=None, seed=seed, source=source,
             observation=session.decision_observation,
             observation_sha256=digest(session.decision_observation),
             pool_n=len(pool), submitted_n=len(submitted),
             selection_rule="first N proposals in pre-outcome construction order",
             submitted=submitted))
        checkpoint, attempts = obtain_carriage_checkpoint(
            seed, pool, root / "legal_predecessor_scan", level=args.level, record=False)
        if checkpoint is None:
            summary = dict(seed=seed, status="no_legal_carriage_checkpoint",
                           setup_attempts=attempts, candidates=[])
            save(root / "summary.json", summary)
            summaries.append(summary)
            continue
        predecessor = next(p for p in pool if p["name"] == attempts[-1]["name"])
        rows = []
        for proposal in submitted:
            # Completed action metadata must match the checkpoint's actual
            # executed predecessor. The prospective end-stop choice is intact.
            trial = json.loads(json.dumps(proposal))
            trial["choices"]["carriage"] = predecessor["choices"]["carriage"]
            if "necessary_geometry" in trial:
                trial["necessary_geometry"]["carriage"] = predecessor["necessary_geometry"]["carriage"]
            directory = root / "candidates" / proposal["name"]
            try:
                result = rollout(seed, trial, directory, level=args.level,
                    checkpoint=checkpoint, completed=("carriage",),
                    stop_after="end_stop", end_stop_acceptance="stable_supported")
                valid = bool(result.get("valid") and not result.get("resource_censored")
                             and result.get("evaluation_scope") == "assembly_prefix_through_end_stop")
                row = dict(name=proposal["name"], valid=valid,
                           local_success=bool(result.get("success")) if valid else None,
                           error=result.get("error"), result_path=str(directory / "result.json"),
                           wall_seconds=result.get("total_wall_seconds"))
            except Exception as exc:
                row = dict(name=proposal["name"], valid=False, local_success=None,
                           error=f"{type(exc).__name__}: {exc}")
            rows.append(row)
            save(root / "progress.json", rows)
            print(json.dumps(dict(seed=seed, **row), ensure_ascii=False), flush=True)
        valid_rows = [r for r in rows if r["valid"]]
        positives = sum(r["local_success"] for r in valid_rows)
        summary = dict(seed=seed, scope="end_stop_stable_placement_only",
            full_task_success=None, status="complete" if len(valid_rows) == len(rows) else "incomplete",
            setup_attempts=attempts, candidates=rows,
            valid=len(valid_rows), successes=positives,
            pool_type=("mixed" if 0 < positives < len(valid_rows) else
                       "all_positive" if positives == len(valid_rows) and valid_rows else
                       "all_negative" if valid_rows else "unlabelled"))
        save(root / "summary.json", summary)
        summaries.append(summary)
    save(args.out / "summary.json", dict(scope="end_stop_stable_placement_only",
        full_task_success=None, layouts=summaries))


if __name__ == "__main__":
    main()
