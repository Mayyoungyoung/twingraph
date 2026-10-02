#!/usr/bin/env python3
"""Check frozen short-task programs against every physical training sample."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from simbench.value.plan import digest


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def audit(root):
    root = Path(root)
    request = read(root / "request.json")
    observation = request["decision_observation"]
    rows = []
    for entry in request["candidates"]:
        directory = root / "candidates" / entry["name"]
        errors = []
        if not all((directory / filename).is_file() for filename in
                   ("input_graph.json", "result.json", "training_sample.json")):
            rows.append(dict(name=entry["name"], valid=False, errors=["missing_artifact"]))
            continue
        graph = read(directory / "input_graph.json")
        result = read(directory / "result.json")
        sample = read(directory / "training_sample.json")
        if (digest(entry["plan"]) != entry["plan_sha256"]
                or graph.get("plan_ir") != entry["plan"]
                or sample.get("complete_candidate_plan_ir") != entry["plan"]):
            errors.append("frozen_plan_mismatch")
        if (graph.get("decision_observation") != observation
                or sample.get("decision_observation") != observation
                or result.get("decision_observation_sha256") != observation["sha256"]):
            errors.append("decision_observation_mismatch")
        if ("noise_state" in observation or "seed" in observation.get("config", {})
                or sample.get("observation_random_state") != request.get("observation_random_state")
                or result.get("observation_random_state") != request.get("observation_random_state")):
            errors.append("observation_noise_leak_or_random_state_mismatch")
        if (digest(graph) != result.get("input_graph_sha256")
                or digest(graph) != sample.get("input_graph_sha256")):
            errors.append("graph_hash_mismatch")
        if (result.get("runtime_sha256") != request["runtime_sha256"]
                or sample.get("layout", {}).get("runtime_sha256") != request["runtime_sha256"]):
            errors.append("runtime_mismatch")
        if (result.get("task_version") != request["task_version"]
                or sample.get("layout", {}).get("task_version") != request["task_version"]):
            errors.append("task_version_mismatch")
        label = result.get("success") if result.get("valid") else None
        if (sample.get("actual_execution_label") != label
                or sample.get("final_functional_predicates") != result.get("final_functional_predicates")
                or sample.get("first_failed_atom") != result.get("first_failed_atom")):
            errors.append("execution_label_or_diagnostics_mismatch")
        if label and not all(result.get("final_functional_predicates", {}).values()):
            errors.append("success_without_functional_acceptance")
        rows.append(dict(name=entry["name"], valid=not errors, errors=errors,
                         physical_valid=result.get("valid"), success=label))
    valid = [row for row in rows if row["valid"] and row["physical_valid"]]
    positives = sum(row["success"] for row in valid)
    report = dict(schema="twingraph.short_mount_audit.v21.r1", root=str(root),
        task_version=request["task_version"], observation_backend=request["observation_backend"],
        noise=dict(position_noise_std_m=request["position_noise_std_m"],
                   yaw_noise_std_rad=request["yaw_noise_std_rad"]),
        frozen_candidates=len(request["candidates"]), audited_valid=len(valid),
        positives=positives, negatives=len(valid)-positives,
        pool_type=("incomplete" if len(valid) != len(rows) else
                   "mixed" if 0 < positives < len(valid) else
                   "all_positive" if positives == len(valid) else "all_negative"),
        rows=rows)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = audit(args.root)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: value for k, value in report.items() if k != "rows"},
                     ensure_ascii=False, indent=2))
    if report["pool_type"] == "incomplete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
