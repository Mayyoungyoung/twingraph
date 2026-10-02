#!/usr/bin/env python3
"""Follow the V27 first pass with separately marked, outcome-directed additions.

Waits for the 800 predeclared layouts. If the first pass is valid but below the
sample targets, appends proposals 9-16 to layouts from mixed root groups. The
first-pass coverage statistics are never overwritten or recomputed from these.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from scripts.collect_sliding_assembly_v25_bulk import (
    CANDIDATE_N, declared_jobs, complete_summary, write_json,
)


TARGET_POSITIVES = 600
TARGET_NEGATIVES = 6000
TARGET_MIXED_ROOTS = 200
MAX_NEGATIVES = 7500


def first_pass(base: Path):
    audit_path = base / "batch_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("halt_reason"):
            raise RuntimeError(f"first-pass collection halted: {audit['halt_reason']}; inspect before append")
    progress_path = base / "progress.json"
    if not progress_path.exists():
        return None
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if progress["completed_layouts"] + progress["failed_layouts"] < len(list(declared_jobs())):
        return None
    if progress["failed_layouts"]:
        raise RuntimeError(f"first pass has {progress['failed_layouts']} invalid layouts; inspect before append")
    by_key = {(row["family"], row["seed"]): row for row in progress["results"]}
    if len(by_key) != len(list(declared_jobs())):
        raise RuntimeError("first-pass progress has missing or duplicate layouts")
    return progress, by_key


def choose_jobs(by_key):
    """Prefer a successful layout within a naturally mixed, predeclared root."""
    for parent in range(4100, 4500):
        broad = by_key[("broad", parent)]
        near = by_key[("neighborhood", parent + 10000)]
        positive = broad["positives"] + near["positives"]
        negative = broad["negatives"] + near["negatives"]
        if not (positive and negative):
            continue
        selected = broad if broad["positives"] else near
        yield dict(seed=selected["seed"], family=selected["family"], parent=parent,
                   split_group=f"neighborhood_parent_{parent}", first_output=selected["output"])


def append_one(job, base: Path):
    stem = f"append_{job['family']}_{job['seed']}"
    primary = base / stem
    if summary := complete_summary(primary):
        return dict(**job, output=str(primary), reused=True,
                    positives=summary["successes"], negatives=summary["failures"])
    output = primary
    retry = 0
    while output.exists() and any(output.iterdir()):
        retry += 1
        output = base / f"{stem}_retry{retry}"
        if summary := complete_summary(output):
            return dict(**job, output=str(output), reused=True,
                        positives=summary["successes"], negatives=summary["failures"])
    command = [sys.executable, "-m", "scripts.collect_sliding_assembly_v23",
               "--seed", str(job["seed"]), "--out", str(output),
               "--pool-n", "48", "--candidate-n", str(CANDIDATE_N),
               "--candidate-start", "8", "--parent-request", str(Path(job["first_output"]) / "request.json"),
               "--level", "L0", "--domain", "online", "--layout-family", job["family"],
               "--split-group", job["split_group"]]
    if job["family"] == "neighborhood":
        command.extend(["--neighborhood-parent-seed", str(job["parent"])])
    log_path = base / "logs" / f"{output.name}.log"
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
        raise RuntimeError(f"append failed for {job['family']} {job['seed']}; see {log_path}")
    return dict(**job, output=str(output), reused=False,
                positives=summary["successes"], negatives=summary["failures"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/v25_bulk"))
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    base = args.out.resolve()
    while True:
        state = first_pass(base)
        if state is not None:
            break
        if not args.wait:
            raise RuntimeError("first pass has not completed")
        time.sleep(300)
    first, by_key = state
    chosen = list(choose_jobs(by_key))
    manifest = dict(schema="twingraph.v27_targeted_append.r1", source="v27 first-pass progress",
                    selection="successful layout in each naturally mixed root, ascending parent seed",
                    candidate_window=[8, 16], candidate_budget=48,
                    first_pass_statistics_unchanged=True, first_pass_results_are_not_independent_test=True,
                    stop_targets=dict(positives=TARGET_POSITIVES, negatives=TARGET_NEGATIVES,
                                      mixed_roots=TARGET_MIXED_ROOTS, max_negatives=MAX_NEGATIVES),
                    eligible_jobs=chosen, no_training=True)
    manifest_path = base / "append_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("append selection changed; preserve the original manifest")
    else:
        write_json(manifest_path, manifest)
    positive, negative = first["positives"], first["negatives"]
    results = []
    for job in chosen:
        if positive >= TARGET_POSITIVES and negative >= TARGET_NEGATIVES and first["mixed_roots"] >= TARGET_MIXED_ROOTS:
            break
        if negative >= MAX_NEGATIVES:
            break
        result = append_one(job, base)
        results.append(result)
        positive += result["positives"]
        negative += result["negatives"]
        write_json(base / "append_progress.json", dict(first_pass_positives=first["positives"],
                   first_pass_negatives=first["negatives"], first_pass_mixed_roots=first["mixed_roots"],
                   append_layouts=len(results), total_positives=positive, total_negatives=negative,
                   append_results=results))
        print(json.dumps(dict(append_layouts=len(results), total_positives=positive,
                              total_negatives=negative)), flush=True)
    write_json(base / "collection_status.json", dict(
        first_pass_layouts=first["completed_layouts"], first_pass_positives=first["positives"],
        first_pass_negatives=first["negatives"], first_pass_mixed_roots=first["mixed_roots"],
        append_layouts=len(results), total_positives=positive, total_negatives=negative,
        targets_met=positive >= TARGET_POSITIVES and negative >= TARGET_NEGATIVES
                    and first["mixed_roots"] >= TARGET_MIXED_ROOTS,
        limitations="append is development collection chosen after first-pass outcomes; not first-pass coverage"))


if __name__ == "__main__":
    main()
