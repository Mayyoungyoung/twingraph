#!/usr/bin/env python3
"""Export small self-contained candidate records from audited V21 runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.audit_state_end_stop_v21 import audit, read


def export(root, output):
    report = audit(root)
    if report["pool_type"] == "incomplete":
        raise ValueError(f"candidate audit is incomplete: {root}")
    request = read(Path(root) / "request.json")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for entry in request["candidates"]:
            sample = read(Path(root) / "candidates" / entry["name"] / "training_sample.json")
            failed = sample.get("first_failed_atom")
            first_failure = ({key: failed.get(key) for key in
                              ("skill", "interface", "implementation", "reason")}
                             if failed else None)
            if first_failure:
                metrics = failed.get("metrics") or {}
                first_failure["handler"] = f"Session.{failed['skill']}"
                detail = metrics.get("strategy") or metrics.get("controller")
                if not detail and failed["skill"] == "inspect_stable_support":
                    detail = "evaluate_end_stop_stable_support"
                first_failure["internal_implementation"] = detail or failed["skill"]
            reason = (first_failure or {}).get("reason", "")
            failure_type = ("geometry_collision" if "collision" in reason else
                "control_tracking" if "tracking error" in reason else
                "functional_acceptance" if first_failure and
                    first_failure["skill"] == "inspect_stable_support" else
                sample["failure_type"])
            record = dict(schema="twingraph.candidate_execution_sample.v21.compact.r1",
                candidate_name=entry["name"], layout=sample["layout"],
                decision_observation=sample["decision_observation"],
                complete_candidate_plan_ir=sample["complete_candidate_plan_ir"],
                plan_sha256=entry["plan_sha256"],
                input_graph_sha256=sample["input_graph_sha256"],
                actual_execution_label=sample["actual_execution_label"],
                first_failed_atom=first_failure, failure_type=failure_type,
                failure_origin=sample["failure_type"],
                stage_audit=sample["stage_audit"],
                final_functional_predicates=sample["final_functional_predicates"],
                observation_backend=sample["observation_backend"],
                observation_config=sample["observation_config"],
                observation_random_state=sample["observation_random_state"],
                trial=sample["trial"], total_wall_seconds=sample["total_wall_seconds"],
                timeout=sample["timeout"], software_exception=sample["software_exception"])
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = export(args.root, args.out)
    print(json.dumps(dict(output=str(args.out), samples=report["audited_valid"],
                          positives=report["positives"], negatives=report["negatives"]),
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
