#!/usr/bin/env python3
"""Resumeable, outcome-independent first-pass collection for 400 source roots.

Each root has one broad L0 supply layout and one predeclared L0 neighborhood
layout. Both execute the first eight distinct plans from the same 48-proposal
budget. This script only collects data; it does not train or rank a model.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT_START = 4100
ROOT_COUNT = 400
POOL_N = 48
CANDIDATE_N = 8
WORKER_LIMIT = 4


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def declared_jobs():
    for parent in range(ROOT_START, ROOT_START + ROOT_COUNT):
        group = f"neighborhood_parent_{parent}"
        yield dict(seed=parent, family="broad", parent=parent, split_group=group)
        yield dict(seed=parent + 10000, family="neighborhood", parent=parent, split_group=group)


def complete_summary(directory: Path):
    path = directory / "summary.json"
    if not path.exists():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("attempted") == CANDIDATE_N and summary.get("valid") == CANDIDATE_N:
        return summary
    return None


def run_one(job, base: Path):
    stem = f"{job['family']}_{job['seed']}"
    primary = base / stem
    if summary := complete_summary(primary):
        return dict(**job, output=str(primary), reused=True,
                    positives=summary["successes"], negatives=summary["failures"], valid=summary["valid"])
    output = primary
    retry = 0
    while output.exists() and any(output.iterdir()):
        retry += 1
        output = base / f"{stem}_retry{retry}"
        if summary := complete_summary(output):
            return dict(**job, output=str(output), reused=True,
                        positives=summary["successes"], negatives=summary["failures"], valid=summary["valid"])
    command = [sys.executable, "-m", "scripts.collect_sliding_assembly_v23",
               "--seed", str(job["seed"]), "--out", str(output),
               "--pool-n", str(POOL_N), "--candidate-n", str(CANDIDATE_N),
               "--level", "L0", "--domain", "online", "--layout-family", job["family"],
               "--split-group", job["split_group"]]
    if job["family"] == "neighborhood":
        command.extend(["--neighborhood-parent-seed", str(job["parent"])])
    started = time.time()
    log_path = base / "logs" / f"{output.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        try:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                       timeout=7200, check=False)
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            returncode = -1
            log.write("\ncollector exceeded 7200 wall seconds\n")
    summary = complete_summary(output)
    if returncode or summary is None:
        return dict(**job, output=str(output), log=str(log_path), error="collector_failed_or_invalid",
                    returncode=returncode, wall_seconds=time.time() - started)
    return dict(**job, output=str(output), log=str(log_path), reused=False,
                positives=summary["successes"], negatives=summary["failures"],
                valid=summary["valid"], wall_seconds=time.time() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/v25_bulk"))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= WORKER_LIMIT:
        parser.error(f"workers must be in [1,{WORKER_LIMIT}]")
    base = args.out.resolve()
    base.mkdir(parents=True, exist_ok=True)
    jobs = list(declared_jobs())
    manifest = dict(schema="twingraph.v25_bulk_arrangement.r1", task="complete_no_wipe_sliding_assembly_base_holes",
                    layout_roots=ROOT_COUNT, layouts=len(jobs), root_start=ROOT_START,
                    layout_families=["broad_L0", "declared_parent_neighborhood_L0"],
                    proposal_budget_per_layout=POOL_N, first_pass_candidates_per_layout=CANDIDATE_N,
                    observation_backend="mujoco_state_pose", position_noise_std_m=0., yaw_noise_std_rad=0.,
                    selection="first eight distinct complete plans before outcome", jobs=jobs,
                    no_training=True)
    manifest_path = base / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("declared collection manifest changed; use a new output directory")
    else:
        write_json(manifest_path, manifest)
    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, job, base): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            key = f"{result['family']}_{result['seed']}"
            results[key] = result
            rows = sorted(results.values(), key=lambda x: (x["parent"], x["family"]))
            completed = [row for row in rows if "error" not in row]
            groups = {}
            for row in completed:
                group = groups.setdefault(row["split_group"], dict(layouts=0, positives=0, negatives=0))
                group["layouts"] += 1
                group["positives"] += row["positives"]
                group["negatives"] += row["negatives"]
            progress = dict(completed_layouts=len(completed), failed_layouts=len(rows)-len(completed),
                            positives=sum(r["positives"] for r in completed),
                            negatives=sum(r["negatives"] for r in completed),
                            mixed_roots=sum(g["positives"] > 0 and g["negatives"] > 0 and g["layouts"] == 2
                                            for g in groups.values()),
                            results=rows)
            write_json(base / "progress.json", progress)
            print(json.dumps({k: progress[k] for k in
                              ("completed_layouts", "failed_layouts", "positives", "negatives", "mixed_roots")}),
                  flush=True)
    return 0 if not any("error" in row for row in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
