#!/usr/bin/env python3
"""Audit frozen no-wipe full-assembly programs and export complete samples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from simbench.value.plan import PlanIR, digest
from scripts.collect_sliding_assembly_v22 import REQUIRED, TASK_VERSION


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def audit(root):
    root = Path(root)
    request = read(root / "request.json")
    errors = []
    rows = []
    if request.get("task_version") != TASK_VERSION or request.get("outcome_labels_read") is not False:
        errors.append("request_task_or_freeze_invalid")
    observation = request["decision_observation"]
    if observation.get("backend") != "mujoco_state_pose" or "noise_state" in observation:
        errors.append("observation_backend_or_noise_leak")
    for entry in request["candidates"]:
        path = root / "candidates" / entry["name"]
        row_errors = []
        if not all((path / name).is_file() for name in
                   ("input_graph.json", "result.json", "training_sample.json")):
            rows.append(dict(name=entry["name"], errors=["missing_artifact"], valid=False))
            continue
        graph, result, sample = (read(path / name) for name in
                                 ("input_graph.json", "result.json", "training_sample.json"))
        full = entry["complete_candidate_plan_ir"]
        assembly = entry["assembly_plan_ir"]
        try:
            PlanIR.from_dict(full).validate(observation.get("objects", {}))
            PlanIR.from_dict(assembly).validate(observation.get("objects", {}))
        except ValueError as exc:
            row_errors.append(f"invalid_plan:{exc}")
        if (digest(full) != entry["plan_sha256"] or digest(assembly) != entry["assembly_plan_sha256"]
                or sample.get("complete_candidate_plan_ir") != full
                or sample.get("assembly_plan_ir") != assembly
                or graph.get("complete_candidate_plan_ir") != full
                or graph.get("assembly_plan_ir") != assembly):
            row_errors.append("frozen_plan_mismatch")
        if (full["calls"][0]["skill"] != "run_sliding_assembly_v22"
                or any("wipe" in str(call["skill"]) for call in assembly["calls"])
                or any(key.startswith("wipe") for key in entry["proposal"])):
            row_errors.append("task_contains_wipe")
        if (sample.get("decision_observation") != observation
                or graph.get("decision_observation") != observation
                or result.get("decision_observation_sha256") != observation["sha256"]):
            row_errors.append("observation_mismatch")
        if (sample.get("observation_random_state") != request.get("observation_random_state")
                or result.get("observation_random_state") != request.get("observation_random_state")):
            row_errors.append("observation_random_state_mismatch")
        if (digest(graph) != result.get("input_graph_sha256")
                or digest(graph) != sample.get("input_graph_sha256")):
            row_errors.append("graph_hash_mismatch")
        if (result.get("runtime_sha256") != request["runtime_sha256"]
                or sample.get("layout", {}).get("runtime_sha256") != request["runtime_sha256"]):
            row_errors.append("runtime_mismatch")
        label = result.get("success") if result.get("valid") else None
        if (sample.get("actual_execution_label") != label
                or sample.get("first_failed_atom") != result.get("first_failed_atom")
                or sample.get("stage_audit") != result.get("stage_audit")
                or sample.get("final_functional_predicates") != result.get("final_functional_predicates")):
            row_errors.append("label_or_diagnostics_mismatch")
        if label and not all(result.get("stage_passes", {}).get(key) for key in REQUIRED):
            row_errors.append("success_without_complete_acceptance")
        rows.append(dict(name=entry["name"], valid=not row_errors,
                         physical_valid=result.get("valid"), success=label,
                         errors=row_errors))
    audited = [row for row in rows if row["valid"] and row["physical_valid"]]
    positives = sum(row["success"] is True for row in audited)
    return dict(schema="twingraph.sliding_assembly_audit.v22.r1", root=str(root),
                task_version=TASK_VERSION, seed=request["seed"], level=request["level"],
                noise=dict(position_std_m=request["position_noise_std_m"],
                           yaw_std_rad=request["yaw_noise_std_rad"]),
                frozen_candidates=len(request["candidates"]), audited_valid=len(audited),
                positives=positives, negatives=len(audited)-positives,
                complete=not errors and len(audited) == len(rows), errors=errors, rows=rows)


def export(roots, output):
    reports = [audit(root) for root in roots]
    if not all(report["complete"] for report in reports):
        raise ValueError("at least one V22 freeze/label audit is incomplete")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for root in roots:
            request = read(Path(root) / "request.json")
            for entry in request["candidates"]:
                sample = read(Path(root) / "candidates" / entry["name"] / "training_sample.json")
                # The portable record carries both PlanIRs, so a relative
                # graph path into an ignored raw run would be misleading.
                sample.pop("input_graph_path", None)
                failure = sample.get("first_failed_atom")
                if failure:
                    failure["handler"] = f"Session.{failure['skill']}"
                    metrics = failure.get("metrics") or {}
                    failure["internal_implementation"] = (
                        metrics.get("controller") or metrics.get("strategy")
                        or failure.get("implementation"))
                sample["candidate_name"] = entry["name"]
                sample["plan_sha256"] = entry["plan_sha256"]
                sample["assembly_plan_sha256"] = entry["assembly_plan_sha256"]
                stream.write(json.dumps(sample, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.out:
        reports = export(args.roots, args.out)
    else:
        reports = [audit(root) for root in args.roots]
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    if not all(report["complete"] for report in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
