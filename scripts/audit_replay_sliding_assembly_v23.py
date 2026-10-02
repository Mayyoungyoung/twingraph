#!/usr/bin/env python3
"""Audit and export the separately labelled V23 paired replays."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from simbench.value.plan import digest
from scripts.collect_sliding_assembly_v23 import REQUIRED, TASK_VERSION


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def audit(source, replay, output=None):
    source, replay = Path(source), Path(replay)
    request = read(source / "request.json")
    selection = read(replay / "selection.json")
    assert request["task_version"] == TASK_VERSION
    assert selection["source_request_sha256"] == digest(request)
    entries = {row["name"]: row for row in request["candidates"]}
    assert selection["first_pass_statistic"] is False
    samples, rows = [], []
    for name in selection["selected"]:
        entry = entries[name]
        for trial in selection["trials"]:
            folder = replay / name / trial["kind"]
            result = read(folder / "result.json")
            sample = read(folder / "training_sample.json")
            errors = []
            if result["source_plan_sha256"] != entry["plan_sha256"]:
                errors.append("plan_hash")
            if sample["complete_candidate_plan_ir"] != entry["complete_candidate_plan_ir"]:
                errors.append("plan_content")
            if sample["decision_observation"] != request["decision_observation"]:
                errors.append("observation")
            if sample["actual_execution_label"] != (result["success"] if result["valid"] else None):
                errors.append("label")
            if sample["replay_kind"] != trial["kind"]:
                errors.append("trial_kind")
            if result["success"]:
                if not all(result["stage_passes"].get(key) for key in REQUIRED):
                    errors.append("incomplete_success")
                joints = result.get("base_hole_acceptance") or {}
                if not all(
                    (metric := joints.get(pin, {}).get(phase) or {}).get("success")
                    and metric.get("pin_base_bridge_is_required") is True
                    and metric.get("released") is True
                    and all((metric.get("receivers", {}).get(receiver) or {}).get("success")
                            for receiver in ("end_stop", "guide_base"))
                    for pin in ("pin_left", "pin_right")
                    for phase in ("inserted_after_release", "retained_after_stroke")
                ):
                    errors.append("base_hole_retention")
            rows.append(dict(name=name, kind=trial["kind"], valid=result["valid"],
                             success=result["success"], failure_type=result["failure_type"],
                             errors=errors))
            sample["candidate_name"] = name
            sample["plan_sha256"] = entry["plan_sha256"]
            samples.append(sample)
    if any(row["errors"] for row in rows) or len(rows) != len(selection["selected"]) * len(selection["trials"]):
        raise ValueError(rows)
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as stream:
            for sample in samples:
                stream.write(json.dumps(sample, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    return dict(source=str(source), replay=str(replay), rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.source, args.replay, args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
