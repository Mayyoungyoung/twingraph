"""Audit the independent V9 target runs from their immutable raw archives."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import tarfile
from collections import Counter
from pathlib import Path


def analyze(archive_dir: Path) -> dict:
    rows: list[dict] = []
    archive_hashes: dict[str, str] = {}
    failures: Counter[str] = Counter()
    failure_skills: Counter[str] = Counter()
    geometry_hashes: set[str] = set()
    observation_by_case: dict[tuple[int, str], str] = {}
    total_attempts = 0
    for seed in (1400, 1401, 1402):
        path = archive_dir / f"seed_{seed}.tar.gz"
        archive_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with tarfile.open(path) as tar:
            members = {member.name: member for member in tar.getmembers()}

            def read_json(name: str) -> dict:
                member = members[name]
                handle = tar.extractfile(member)
                assert handle is not None, name
                return json.load(handle)

            prefix = f"seed_{seed}"
            summary = read_json(f"{prefix}/summary.json")
            assert len(summary["rows"]) == 6
            rows.extend(summary["rows"])
            for condition in ("nominal", "light_low", "light_high"):
                for method in ("random", "fixed_reference"):
                    base = f"{prefix}/{prefix}/{condition}/{method}"
                    selected_path = f"{base}/selection_before_execution.json"
                    selected = read_json(selected_path)
                    attempts = sorted(
                        name for name in members
                        if name.startswith(f"{base}/attempt_") and name.endswith("/result.json")
                    )
                    row = next(
                        r for r in summary["rows"]
                        if r["condition"] == condition and r["method"] == method
                    )
                    assert len(attempts) == row["verification_count"]
                    assert len(attempts) <= selected["k"]
                    assert members[selected_path].mtime <= min(members[name].mtime for name in attempts)
                    attempt_data = [read_json(name) for name in attempts]
                    observation_hashes = {a["pre_execution_observation"]["sha256"] for a in attempt_data}
                    assert len(observation_hashes) == 1
                    observation_hash = observation_hashes.pop()
                    if selected["observation_sha256"] is not None:
                        assert observation_hash == selected["observation_sha256"]
                    case = (seed, condition)
                    assert case not in observation_by_case or observation_by_case[case] == observation_hash
                    observation_by_case[case] = observation_hash
                    assert all(a["task_version"] == "functional_assembly_v9_funnel_r1" for a in attempt_data)
                    assert all(
                        name.split("/attempt_", 1)[1].split("/", 1)[0].split("_", 1)[1]
                        == selected["order"][i]
                        for i, name in enumerate(attempts)
                    )
                    assert all(a["result"]["valid"] for a in attempt_data)
                    assert any(bool(a["result"]["full_success"]) for a in attempt_data) == bool(row["success"])
                    if row["success"]:
                        assert attempt_data[-1]["result"]["full_success"]
                    geometry_hashes.update(a["geometry_sha256"] for a in attempt_data)
                    total_attempts += len(attempts)
                    for attempt in attempt_data:
                        if not attempt["result"]["full_success"]:
                            stages = attempt["result"].get("stage_passes") or {}
                            first_failure = next((k for k, passed in stages.items() if not passed), "unknown")
                            failures[first_failure] += 1
                            error = attempt["result"].get("error", "")
                            failure_skills[error.split(":", 1)[0] if error else "unknown"] += 1
                    row["winning_candidate"] = selected["order"][len(attempts) - 1] if row["success"] else None
                    row["observation_sha256"] = observation_hash
    assert len(rows) == 18 and total_attempts == sum(row["verification_count"] for row in rows)
    metrics = {}
    for method in ("random", "fixed_reference"):
        subset = [row for row in rows if row["method"] == method]
        metrics[method] = {
            "success_count": sum(bool(row["success"]) for row in subset),
            "case_count": len(subset),
            "mean_verification_count": statistics.mean(row["verification_count"] for row in subset),
            "mean_verification_wall_seconds": statistics.mean(row["verification_wall_seconds"] for row in subset),
            "mean_full_system_wall_seconds": statistics.mean(row["full_system_wall_seconds"] for row in subset),
        }
    return {
        "task_version": "functional_assembly_v9_funnel_r1",
        "archive_sha256": archive_hashes,
        "target_geometry_sha256": sorted(geometry_hashes),
        "target_geometry_unique_count": len(geometry_hashes),
        "method_run_count": len(rows),
        "candidate_verification_count": total_attempts,
        "metrics": metrics,
        "failed_stage_count": dict(failures),
        "failed_skill_count": dict(failure_skills),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.archives)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.out / "rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["rows"][0]))
        writer.writeheader()
        writer.writerows(result["rows"])
    print(json.dumps({key: result[key] for key in ("method_run_count", "candidate_verification_count", "metrics", "target_geometry_unique_count", "failed_stage_count")}, indent=2))


if __name__ == "__main__":
    main()
