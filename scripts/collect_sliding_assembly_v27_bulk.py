#!/usr/bin/env python3
"""V27 first-pass collection with an automatic all-negative batch stop.

The 400 root seeds and paired neighborhoods are declared before outcomes.
Each layout uses the restored 48-proposal generator and executes its first
eight distinct complete programs. Batches contain eight roots (16 layouts).
If a full batch has no positive, preserve its samples and stop for diagnosis.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

from scripts.collect_sliding_assembly_v25_bulk import declared_jobs, run_one, write_json
from simbench.value.provenance_v12 import fingerprint as runtime_fingerprint


ROOTS_PER_BATCH = 8
LAYOUTS_PER_BATCH = 2 * ROOTS_PER_BATCH


def summarize(results):
    complete = [row for row in results if "error" not in row]
    groups = {}
    for row in complete:
        group = groups.setdefault(row["split_group"], dict(layouts=0, positives=0, negatives=0))
        group["layouts"] += 1
        group["positives"] += row["positives"]
        group["negatives"] += row["negatives"]
    return dict(completed_layouts=len(complete), failed_layouts=len(results)-len(complete),
                positives=sum(row["positives"] for row in complete),
                negatives=sum(row["negatives"] for row in complete),
                mixed_roots=sum(g["layouts"] == 2 and g["positives"] > 0 and g["negatives"] > 0
                                for g in groups.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/v27_bulk"))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("workers must be between one and four")
    base = args.out.resolve()
    base.mkdir(parents=True, exist_ok=True)
    jobs = list(declared_jobs())
    runtime_sha = runtime_fingerprint()["sha256"]
    manifest = dict(schema="twingraph.v27_natural_collection.r1",
                    task="complete_no_wipe_sliding_assembly_base_holes",
                    task_version="sliding_assembly_base_holes_v23",
                    runtime_sha256=runtime_sha,
                    layout_roots=400, layouts=len(jobs), jobs=jobs,
                    proposal_budget_per_layout=48, first_pass_candidates_per_layout=8,
                    observation_backend="mujoco_state_pose",
                    position_noise_std_m=0., yaw_noise_std_rad=0.,
                    layout_selection_uses_outcome=False,
                    candidate_selection="first eight distinct complete plans before outcomes",
                    stop_rule="stop after any complete eight-root batch with zero positives, any invalid layout, or 7500 negatives",
                    no_training=True)
    manifest_path = base / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("collection manifest/runtime changed; use a new output directory")
    else:
        write_json(manifest_path, manifest)
    existing_path = base / "progress.json"
    existing = json.loads(existing_path.read_text(encoding="utf-8")) if existing_path.exists() else {}
    by_key = {(r["family"], r["seed"]): r for r in existing.get("results", [])}
    for start in range(0, len(jobs), LAYOUTS_PER_BATCH):
        batch = jobs[start:start + LAYOUTS_PER_BATCH]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_one, job, base): job for job in batch}
            for future in as_completed(futures):
                result = future.result()
                by_key[(result["family"], result["seed"])] = result
                ordered = [by_key[(job["family"], job["seed"])] for job in jobs
                           if (job["family"], job["seed"]) in by_key]
                summary = summarize(ordered)
                write_json(existing_path, dict(**summary, results=ordered))
                print(json.dumps(summary), flush=True)
        batch_rows = [by_key[(job["family"], job["seed"])] for job in batch]
        summary = summarize(list(by_key.values()))
        batch_positive = sum(row.get("positives", 0) for row in batch_rows)
        reason = ("invalid_layout" if any("error" in row for row in batch_rows) else
                  "all_negative_batch" if batch_positive == 0 else
                  "negative_budget" if summary["negatives"] >= 7500 else None)
        write_json(base / "batch_audit.json", dict(last_batch_index=start // LAYOUTS_PER_BATCH,
                   last_batch_jobs=batch, last_batch_positives=batch_positive,
                   **summary, halt_reason=reason))
        if reason:
            print(json.dumps(dict(halt_reason=reason, **summary)), flush=True)
            return 2
    final = summarize(list(by_key.values()))
    write_json(base / "first_pass_complete.json", final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
