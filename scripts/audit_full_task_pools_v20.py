#!/usr/bin/env python3
"""Audit complete functional-task labels before any value training.

This is a coverage gate, not a dataset curator: every generated candidate and
every all-negative/all-positive layout remains in the report. No local prefix
result can become a positive or negative full-task label.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from simbench.value.plan import digest


FLAGS = (
    "cleaning_pass", "assembly_pass", "functional_test_pass",
    "fixture_capture_pass", "final_seat_pass",
    "final_release_and_retraction_pass",
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def audit_seed(root):
    if not (root / "request.json").is_file():
        try:
            seed = int(root.parent.name.removeprefix("seed_"))
        except ValueError as exc:
            raise ValueError(f"cannot identify incomplete layout directory: {root}") from exc
        return dict(seed=seed, root=str(root), generated=0, valid=0,
                    success=0, failure=0, pool_type="incomplete",
                    runtime_sha256=None, rows=[], reason="missing_frozen_request")
    request = read(root / "request.json")
    seed = int(request["seed"])
    pool = request["pool"]
    names = [item["name"] for item in pool]
    if len(names) != len(set(names)) or not names:
        raise ValueError(f"duplicate/empty candidate pool: {root}")
    rows = []
    for name in names:
        frozen_proposal = next(item for item in pool if item["name"] == name)
        result_path = root / "candidates" / name / "result.json"
        if not result_path.is_file():
            rows.append(dict(name=name, status="missing"))
            continue
        result = read(result_path)
        graph_path = result_path.with_name("input_graph.json")
        errors = []
        if not graph_path.is_file():
            errors.append("missing_graph")
        else:
            graph = read(graph_path)
            if digest(graph) != result.get("input_graph_sha256"):
                errors.append("graph_result_hash_mismatch")
            frozen_graphs = request.get("graph_sha256")
            if frozen_graphs is not None and digest(graph) != frozen_graphs.get(name):
                errors.append("pre_outcome_graph_hash_mismatch")
            if graph.get("proposal") != frozen_proposal:
                errors.append("graph_request_proposal_mismatch")
        if result.get("seed") != seed:
            errors.append("seed_mismatch")
        if result.get("proposal") != frozen_proposal:
            errors.append("result_request_proposal_mismatch")
        if result.get("evaluation_scope") != "complete_functional_task" or result.get("full_task_label") is not True:
            errors.append("not_complete_functional_task_scope")
        if not result.get("valid") or result.get("resource_censored"):
            errors.append("invalid_or_censored")
        passes = result.get("stage_passes", {})
        if not all(flag in passes and isinstance(passes[flag], bool) for flag in FLAGS):
            errors.append("missing_final_acceptance_flags")
        outcome = result.get("full_success")
        if not isinstance(outcome, bool) or outcome != result.get("success"):
            errors.append("full_success_result_disagree")
        if outcome is True and not all(passes.get(flag) is True for flag in FLAGS):
            errors.append("success_without_all_final_acceptance")
        if outcome is False and all(passes.get(flag) is True for flag in FLAGS):
            errors.append("failure_despite_all_final_acceptance")
        if request.get("runtime_sha256") != result.get("runtime_sha256"):
            errors.append("runtime_mismatch")
        rows.append(dict(name=name, status="invalid" if errors else "valid",
                         errors=errors, success=outcome if not errors else None,
                         seconds=result.get("total_wall_seconds"),
                         passed=[flag for flag in FLAGS if passes.get(flag) is True],
                         error=result.get("error")))
    valid = [r for r in rows if r["status"] == "valid"]
    successes = sum(r["success"] for r in valid)
    complete = len(valid) == len(pool)
    return dict(seed=seed, root=str(root), generated=len(pool), valid=len(valid),
                success=successes, failure=len(valid)-successes,
                pool_type=("incomplete" if not complete else "mixed" if 0 < successes < len(pool)
                           else "all_positive" if successes == len(pool) else "all_negative"),
                runtime_sha256=request.get("runtime_sha256"), rows=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, action="append", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    pools = []
    for directory in args.root:
        pools.extend(audit_seed(root) for root in sorted(directory.glob("seed_*/collect")))
    if not pools:
        p.error("no seed_*/collect/request.json pools found")
    seeds = [r["seed"] for r in pools]
    if len(seeds) != len(set(seeds)):
        raise ValueError("duplicate layout seed across roots")
    types = Counter(r["pool_type"] for r in pools)
    runtimes = {r["runtime_sha256"] for r in pools}
    complete = [r for r in pools if r["pool_type"] != "incomplete"]
    mixed = [r for r in complete if r["pool_type"] == "mixed"]
    fold = lambda seed: "test" if seed % 5 == 0 else "validation" if seed % 5 == 1 else "train"
    mixed_folds = {fold(r["seed"]) for r in mixed}
    report = dict(schema="twingraph.full_task_pool_audit.v20.r2",
                  label="all six final functional acceptance flags",
                  attempted_layouts=len(pools), generated_candidates=sum(r["generated"] for r in pools),
                  pool_types=dict(types), one_frozen_runtime=len(runtimes) == 1,
                  runtime_sha256=sorted(str(x) for x in runtimes),
                  natural_mixed_coverage=(len(mixed) / len(complete) if complete else None),
                  confirmed_mixed_fraction_of_attempted=(len(mixed) / len(pools)),
                  incomplete_layouts=len(pools)-len(complete),
                  mixed_layout_seeds_by_fold={name: sorted(r["seed"] for r in mixed if fold(r["seed"]) == name)
                                              for name in ("train", "validation", "test")},
                  ready_for_full_task_value_training=(len(complete) == len(pools)
                      and mixed_folds == {"train", "validation", "test"}
                      and len(runtimes) == 1
                      and len({r["generated"] for r in pools}) == 1),
                  layouts=pools)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "layouts"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
