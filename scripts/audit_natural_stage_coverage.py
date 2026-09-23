#!/usr/bin/env python3
"""Audit physical success/failure coverage for each assembly stage.

The candidate sequence is frozen before outcomes are observed.  A downstream
stage starts from one physically achieved predecessor checkpoint; a failed
predecessor is recorded as a coverage gap, never as a negative target label.
This audit does not assert that the complete functional task succeeded.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from simbench.value import system_v12
from simbench.value.plan import digest
from simbench.value.planner_v12 import propose
from simbench.value.system_v11 import twin_checkpoint, restore_twin_checkpoint


def capture_rollout(seed, proposal, path, *, completed=(), checkpoint=None,
                    target, end_stop_acceptance=None, level="L1"):
    """Run the real stage controller and capture its exact final simulator state."""
    captured = {}
    original = system_v12.run_mechanism_prefix

    def capture(session, *, stop_after, **kwargs):
        result = original(session, stop_after=stop_after, **kwargs)
        if result.ok:
            captured["checkpoint"] = twin_checkpoint(session)
        return result

    system_v12.run_mechanism_prefix = capture
    try:
        result = system_v12.rollout(seed, proposal, path, completed=completed,
            checkpoint=checkpoint, stop_after=target, level=level,
            end_stop_acceptance=end_stop_acceptance)
    finally:
        system_v12.run_mechanism_prefix = original
    return result, captured.get("checkpoint")


def adapt(proposal, completed, predecessor, order):
    trial = deepcopy(proposal)
    trial["order"] = list(order)
    for part in completed:
        trial["choices"][part] = deepcopy(predecessor["choices"][part])
        # The new planner marks already completed geometry as unknown.  Its
        # old supply-pose row would be stale at this physical checkpoint.
    return trial


def audit_seed(seed, root, *, n, submit_n, setup_scan, level, target_stage,
               vary_predecessor_stage=None, predecessor_success_rank=1):
    root.mkdir(parents=True, exist_ok=True)
    _, session, _, _ = system_v12.make_scene(seed, root / "planning", level=level)
    pool, source = propose(session.decision_observation, cad=session.planning_cad,
                           n=n, seed=seed)
    pool.sort(key=lambda p: int(p["name"].rsplit("_", 1)[-1]))
    order = list(pool[0]["order"])
    if target_stage not in order:
        raise ValueError(f"requested target {target_stage} is not in {order}")
    target_index = order.index(target_stage)
    if vary_predecessor_stage is not None and (vary_predecessor_stage not in order or
            order.index(vary_predecessor_stage) >= target_index):
        raise ValueError("varied predecessor must precede target stage")
    # The order is determined solely by the initial observation and planner.
    system_v12.save(root / "request.json", dict(seed=seed, source=source,
        observation_sha256=digest(session.decision_observation), order=order,
        candidate_rule="first N in pre-outcome construction order",
        setup_scan=setup_scan, submit_n=submit_n, target_stage=target_stage,
        vary_predecessor_stage=vary_predecessor_stage,
        predecessor_success_rank=predecessor_success_rank,
        submitted=pool[:submit_n]))
    completed = []
    checkpoint = None
    predecessor = None
    stages = []
    for index, target in enumerate(order[:target_index + 1]):
        rows = []
        stage_dir = root / f"{index:02d}_{target}"
        if index:
            from simbench.value.stage_v7 import refresh_visual_observation
            _, replanning, _, _ = system_v12.make_scene(seed, stage_dir / "replanning",
                                                        level=level)
            restore_twin_checkpoint(replanning, checkpoint)
            refresh_visual_observation(replanning, parts=replanning.parts)
            checkpoint = twin_checkpoint(replanning)
            pool, stage_source = propose(replanning.decision_observation,
                cad=replanning.planning_cad, n=n, seed=seed, completed=completed)
            pool.sort(key=lambda p: int(p["name"].rsplit("_", 1)[-1]))
            system_v12.save(stage_dir / "request.json", dict(seed=seed,
                completed=completed, observation_sha256=digest(replanning.decision_observation),
                source=stage_source, submitted=pool[:submit_n],
                candidate_rule="first N from current checkpoint observation"))
        candidates = pool[:submit_n] if index == target_index else pool[:setup_scan]
        successor = None
        successor_proposal = None
        eligible_successes = 0
        desired_success_rank = (predecessor_success_rank
            if target == vary_predecessor_stage else 1)
        for proposal in candidates:
            trial = adapt(proposal, completed, predecessor, order) if completed else deepcopy(proposal)
            directory = stage_dir / proposal["name"]
            try:
                result, state = capture_rollout(seed, trial, directory,
                    completed=completed, checkpoint=checkpoint, target=target,
                    # A downstream pin needs the genuine shared shaft route.
                    end_stop_acceptance=("stable_supported" if target == "end_stop" and
                        index == target_index else None), level=level)
                valid = bool(result.get("valid") and not result.get("resource_censored") and
                    result.get("evaluation_scope") == f"assembly_prefix_through_{target}")
                row = dict(name=proposal["name"], valid=valid,
                    success=bool(result["success"]) if valid else None,
                    wall_seconds=result.get("total_wall_seconds"), error=result.get("error"),
                    result=str(directory / "result.json"))
                if valid and result["success"] and state is not None:
                    eligible_successes += 1
                    if eligible_successes == desired_success_rank:
                        successor, successor_proposal = state, trial
            except Exception as exc:
                row = dict(name=proposal["name"], valid=False, success=None,
                           error=f"{type(exc).__name__}: {exc}")
            rows.append(row)
            system_v12.save(stage_dir / "progress.json", rows)
            print(json.dumps(dict(seed=seed, stage=target, **row)), flush=True)
            # Only the target stage needs the whole pool.  Predecessor scans
            # stop at their first natural success, with every trial retained.
            if index < target_index and successor is not None:
                break
        valid_rows = [row for row in rows if row["valid"]]
        positive = sum(bool(row["success"]) for row in valid_rows)
        stage = dict(stage=target, attempted=len(rows), valid=len(valid_rows),
            success=positive, negative=len(valid_rows)-positive,
            selected_success_rank=(desired_success_rank if index < target_index else None),
            pool_type=("mixed" if 0 < positive < len(valid_rows) else
                       "all_positive" if positive == len(valid_rows) and valid_rows else
                       "all_negative" if valid_rows else "unlabelled"),
            predecessor_checkpoint=bool(checkpoint), rows=rows)
        system_v12.save(stage_dir / "summary.json", stage)
        stages.append(stage)
        if successor is None and index < target_index:
            for skipped in order[index+1:target_index+1]:
                stages.append(dict(stage=skipped, status="blocked_no_legal_predecessor",
                                   attempted=0, valid=0, success=0, negative=0))
            break
        if index < target_index:
            checkpoint, predecessor = successor, successor_proposal
            completed.append(target)
    summary = dict(seed=seed, order=order, target_stage=target_stage, stages=stages,
                   reached_target=bool(stages[-1]["stage"] == target_stage and
                                       "pool_type" in stages[-1]))
    system_v12.save(root / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--pool-n", type=int, default=48)
    parser.add_argument("--submit-n", type=int, default=12)
    parser.add_argument("--setup-scan", type=int, default=48)
    parser.add_argument("--target-stage", choices=("carriage", "end_stop", "pin_left",
                        "pin_right", "handle"), required=True)
    parser.add_argument("--vary-predecessor-stage", choices=("carriage", "end_stop",
                        "pin_left", "pin_right"))
    parser.add_argument("--predecessor-success-rank", type=int, default=1)
    parser.add_argument("--level", choices=("L0", "L1", "L2"), default="L1")
    args = parser.parse_args()
    if not 1 <= args.submit_n <= args.pool_n or not 1 <= args.setup_scan <= args.pool_n:
        parser.error("submit-n and setup-scan must be within pool-n")
    if args.predecessor_success_rank < 1:
        parser.error("predecessor-success-rank must be positive")
    summaries = [audit_seed(seed, args.out / f"seed_{seed}", n=args.pool_n,
                 submit_n=args.submit_n, setup_scan=args.setup_scan, level=args.level,
                 target_stage=args.target_stage,
                 vary_predecessor_stage=args.vary_predecessor_stage,
                 predecessor_success_rank=args.predecessor_success_rank)
                 for seed in args.seeds]
    system_v12.save(args.out / "summary.json", dict(layouts=summaries))


if __name__ == "__main__":
    main()
