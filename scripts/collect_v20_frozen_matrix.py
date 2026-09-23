#!/usr/bin/env python3
"""Collect every member of a pre-frozen natural full-task matrix.

Only resource-censored attempts may be archived and retried. Physical success
and failure results stay in place, and the same candidate graph is reused.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.archive_v20_censored_trials import archive
from scripts.audit_full_task_pools_v20 import audit_seed
from simbench.value.provenance_v12 import fingerprint


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-censored-retries", type=int, default=2)
    args = parser.parse_args()
    manifest = read(args.root / "freeze_manifest.json")
    frozen = {int(row["seed"]): row for row in manifest["layouts"]}
    if len(args.seeds) != len(set(args.seeds)) or not set(args.seeds) <= set(frozen):
        parser.error("seeds must be distinct members of the frozen matrix")
    if fingerprint()["sha256"] != manifest["runtime_sha256"]:
        raise RuntimeError("physical runtime differs from the pre-outcome freeze")
    for seed in manifest["seeds"]:
        root = args.root / f"seed_{seed}" / "collect"
        request = read(root / "request.json")
        if (request["runtime_sha256"] != manifest["runtime_sha256"]
                or request["level"] != manifest["level"]
                or request["domain"] != manifest["domain"]
                or len(request["pool"]) != manifest["candidate_count"]
                or request.get("graph_sha256") != frozen[seed]["graph_sha256"]
                or [row["name"] for row in request["pool"]] != frozen[seed]["candidate_names"]):
            raise RuntimeError(f"frozen request changed before execution: {seed}")
    summary = []
    for seed in args.seeds:
        root = args.root / f"seed_{seed}" / "collect"
        attempts = 0
        while True:
            workers = args.workers if attempts == 0 else 1
            command = [sys.executable, "scripts/collect_v12_parallel.py",
                       "--out", str(args.root), "--seeds", str(seed),
                       "--workers", str(workers), "--n", str(manifest["candidate_count"]),
                       "--level", manifest["level"], "--domain", manifest["domain"]]
            execution = subprocess.run(command, env=os.environ.copy(), check=False)
            audit = audit_seed(root)
            print(json.dumps(dict(seed=seed, attempt=attempts + 1,
                                  collector_exit=execution.returncode,
                                  generated=audit["generated"], valid=audit["valid"],
                                  full_task_success=audit["success"],
                                  pool_type=audit["pool_type"])), flush=True)
            if audit["pool_type"] != "incomplete":
                summary.append(dict(seed=seed, status=audit["pool_type"],
                                    success=audit["success"], failure=audit["failure"],
                                    retries=attempts))
                break
            invalid = [row for row in audit["rows"] if row["status"] == "invalid"]
            missing = [row for row in audit["rows"] if row["status"] == "missing"]
            if invalid and attempts < args.max_censored_retries:
                try:
                    archived = archive(root)
                except ValueError as exc:
                    summary.append(dict(seed=seed, status="incomplete_non_censored",
                                        invalid=[row["name"] for row in invalid], reason=str(exc)))
                    break
                print(json.dumps(dict(seed=seed, resource_retry=archived)), flush=True)
                attempts += 1
                continue
            if missing and not invalid and attempts < args.max_censored_retries:
                attempts += 1
                continue
            summary.append(dict(seed=seed, status="incomplete",
                                invalid=[row["name"] for row in invalid],
                                missing=[row["name"] for row in missing], retries=attempts))
            break
    output = args.root / f"collection_group_{args.seeds[0]}_{args.seeds[-1]}.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(group_summary=str(output), layouts=len(summary),
                          incomplete=sum(row["status"].startswith("incomplete") for row in summary))), flush=True)
    if any(row["status"].startswith("incomplete") for row in summary):
        raise RuntimeError("one or more frozen layouts remain incomplete; inspect group summary")


if __name__ == "__main__":
    main()
