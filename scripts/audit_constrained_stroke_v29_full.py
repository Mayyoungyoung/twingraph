#!/usr/bin/env python3
"""Three-layout, eight-candidate paired full-task V28/V29 stroke comparison."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time

from simbench.assembly.constrained_stroke_v29 import STROKE_FIXED_V28, STROKE_PROGRESS_V29
from simbench.assembly.sensor_learning_v12 import PIN_PRESS_REMAINING_V28
from simbench.value.plan import digest
from simbench.value.provenance_v12 import fingerprint as runtime_fingerprint
from simbench.value.system_v12 import save


LAYOUTS = (
    dict(name="neighborhood_14100", seed=14100, family="neighborhood", parent=4100),
    dict(name="broad_4102", seed=4102, family="broad", parent=4102),
    dict(name="broad_4104", seed=4104, family="broad", parent=4104),
)
POLICIES = (STROKE_FIXED_V28, STROKE_PROGRESS_V29)


def complete(directory):
    path = directory / "summary.json"
    if not path.exists():
        return None
    row = json.loads(path.read_text(encoding="utf-8"))
    return row if row.get("attempted") == 8 and row.get("valid") == 8 else None


def run_job(job, base):
    directory = base / job["name"] / job["policy"]
    if summary := complete(directory):
        return dict(**job, reused=True, summary=summary)
    directory.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "scripts.collect_sliding_assembly_v23",
        "--seed", str(job["seed"]), "--out", str(directory), "--pool-n", "48",
        "--candidate-n", "8", "--level", "L0", "--domain", "online",
        "--layout-family", job["family"], "--split-group",
        f"neighborhood_parent_{job['parent']}", "--pin-press-budget-policy",
        PIN_PRESS_REMAINING_V28, "--stroke-motion-policy", job["policy"]]
    if job["family"] == "neighborhood":
        command.extend(["--neighborhood-parent-seed", str(job["parent"])])
    log = base / "logs" / f"{job['name']}__{job['policy']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                   timeout=10800, check=False)
    summary = complete(directory)
    if completed.returncode or summary is None:
        raise RuntimeError(f"full paired run failed: {job}; see {log}")
    return dict(**job, reused=False, wall_seconds=time.perf_counter() - started,
                summary=summary)


def phase(result):
    audit = result["stage_audit"]
    if result["success"]:
        return "complete_success"
    if not audit["handle"]["completed"]:
        return "handle_failure"
    if not audit["stroke"]["completed"]:
        return "stroke_or_release_failure"
    return "final_acceptance_failure"


def candidate_rows(directory):
    request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    rows = []
    for entry in request["candidates"]:
        path = directory / "candidates" / entry["name"] / "result.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        audit = result["stage_audit"]
        first = result.get("first_failed_atom") or {}
        rows.append(dict(
            name=entry["name"], proposal_sha256=digest(entry["proposal"]),
            trial_sha256=result["trial_sha256"], success=bool(result["success"]),
            pins_completed=all(audit[p]["completed"] for p in ("pin_left", "pin_right")),
            handle_completed=bool(audit["handle"]["completed"]),
            stroke_completed=bool(audit["stroke"]["completed"]),
            outcome=phase(result), failed_skill=first.get("skill"),
            first_failure=first.get("reason") or result.get("error"),
            total_wall_seconds=result.get("total_wall_seconds"),
            result=str(path),
        ))
    return request, rows


def compare(base, jobs):
    layouts = []
    for layout in LAYOUTS:
        paths = {p: base / layout["name"] / p for p in POLICIES}
        old_request, old = candidate_rows(paths[STROKE_FIXED_V28])
        new_request, new = candidate_rows(paths[STROKE_PROGRESS_V29])
        pairs = []
        for a, b in zip(old, new):
            if a["name"] != b["name"] or a["proposal_sha256"] != b["proposal_sha256"]:
                raise ValueError("paired candidate semantics changed")
            if a["trial_sha256"] != b["trial_sha256"]:
                raise ValueError("paired physical trial changed")
            pairs.append(dict(candidate=a["name"], proposal_sha256=a["proposal_sha256"],
                              trial_sha256=a["trial_sha256"], old=a, new=b))
        if old_request["decision_observation"] != new_request["decision_observation"]:
            raise ValueError("paired decision observation changed")
        layouts.append(dict(**layout,
            decision_observation_sha256=old_request["decision_observation"]["sha256"],
            pairs=pairs))
    all_pairs = [pair for layout in layouts for pair in layout["pairs"]]

    def aggregate(side):
        rows = [pair[side] for pair in all_pairs]
        return dict(candidates=len(rows),
            post_dual_pin=sum(row["pins_completed"] for row in rows),
            handle_completed=sum(row["handle_completed"] for row in rows),
            stroke_completed=sum(row["stroke_completed"] for row in rows),
            full_successes=sum(row["success"] for row in rows),
            outcomes=dict(Counter(row["outcome"] for row in rows)),
            failure_reasons=dict(Counter(row["first_failure"] for row in rows if not row["success"])),
            total_wall_seconds=sum(float(row["total_wall_seconds"] or 0.) for row in rows))

    def names(predicate):
        return [f"{pair['candidate']}@{layout['name']}" for layout in layouts
                for pair in layout["pairs"] if predicate(pair)]

    return dict(schema="twingraph.constrained_stroke_v29.full_pair_summary.r1",
        layouts=layouts, old=aggregate("old"), new=aggregate("new"),
        rescued_stroke_failures=names(lambda p: p["old"]["handle_completed"]
            and not p["old"]["stroke_completed"] and p["new"]["stroke_completed"]),
        new_full_positives=names(lambda p: not p["old"]["success"] and p["new"]["success"]),
        broken_old_full_successes=names(lambda p: p["old"]["success"] and not p["new"]["success"]),
        jobs=jobs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/v29_stroke/full_pairs"))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    jobs = [dict(**layout, policy=policy) for layout in LAYOUTS for policy in POLICIES]
    save(args.out / "manifest.json", dict(
        schema="twingraph.constrained_stroke_v29.full_pair.r1",
        runtime_sha256=runtime_fingerprint()["sha256"], layouts=list(LAYOUTS),
        candidates_per_layout=8, policies=list(POLICIES),
        pin_press_budget_policy=PIN_PRESS_REMAINING_V28,
        layout_and_candidate_selection_frozen_before_v29_execution=True,
        same_initial_observation_candidate_semantics_and_physical_trial_required=True,
        no_training=True))
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_job, job, args.out): job for job in jobs}
        for future in as_completed(futures):
            row = future.result(); rows.append(row); save(args.out / "progress.json", rows)
            print(json.dumps(dict(name=row["name"], policy=row["policy"],
                                  successes=row["summary"]["successes"])), flush=True)
    save(args.out / "summary.json", compare(args.out, rows))


if __name__ == "__main__":
    main()
