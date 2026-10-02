#!/usr/bin/env python3
"""Audit and summarize the frozen V24 collection without changing labels."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from scripts.audit_sliding_assembly_v23 import audit, export
from simbench.value.plan import digest


EVIDENCE = Path("docs/evidence/value_v24_collection")
RESULTS = Path("results")
NEW = [
    ("broad", 2062), ("broad", 2063), ("broad", 2064), ("broad", 2065),
    ("near", 3005), ("near", 3006), ("near", 3007), ("near", 3008),
]
APPEND = [("near", 3000), ("near", 3002)]


def root(kind, seed, *, append=False):
    prefix = "v24_append" if append else "v24_first"
    return RESULTS / f"{prefix}_{kind}_{seed}"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def realized_choices(directory):
    result = read(directory / "result.json")
    steps = read(directory / "scene" / "steps.json")
    return dict(plan_sha256=result["plan_sha256"], order=result["proposal"]["order"],
        grasp_yaw_rad={row["params"]["part"]: row["metrics"]["yaw_rad"]
                       for row in steps if row["skill"] == "select_grasp" and row["ok"]},
        selected_transfer_ids=[row["metrics"].get("selected_id") for row in steps
                               if row["skill"] == "plan_transfer" and row["ok"]])


def main():
    manifest = read(EVIDENCE / "collection_manifest.json")
    assert manifest["task_version"] == "sliding_assembly_base_holes_v23"
    roots = [root(kind, seed) for kind, seed in NEW]
    append_roots = [root(kind, seed, append=True) for kind, seed in APPEND]
    for (kind, seed), current in zip(APPEND, append_roots):
        parent_path = (RESULTS / "v23_first_near_3000" / "request.json" if seed == 3000
                       else RESULTS / "v23_first_near_3002_retry" / "request.json")
        parent, request = read(parent_path), read(current / "request.json")
        assert request["candidate_start"] == 8
        assert request["parent_request_sha256"] == digest(parent)
        assert request["decision_observation"] == parent["decision_observation"]
        assert request["runtime_sha256"] == parent["runtime_sha256"]
        assert not {x["name"] for x in request["candidates"]} & {
            x["name"] for x in parent["candidates"]}
    first_reports = export(roots, EVIDENCE / "new_layout_first_pass_samples.jsonl")
    append_reports = export(append_roots, EVIDENCE / "append_samples.jsonl")
    all_reports = first_reports + append_reports
    assert all(report["complete"] and report["audited_valid"] == 8 for report in all_reports)

    pools = []
    groups = {}
    for (kind, seed), directory, report in zip(NEW, roots, first_reports):
        request = read(directory / "request.json")
        group = request["split_group"]
        groups.setdefault(group, dict(layouts=0, candidates=0, positives=0, negatives=0))
        groups[group]["layouts"] += 1
        groups[group]["candidates"] += report["audited_valid"]
        groups[group]["positives"] += report["positives"]
        groups[group]["negatives"] += report["negatives"]
        pools.append(dict(seed=seed, family=request["layout_family"], split_group=group,
                          positives=report["positives"], negatives=report["negatives"],
                          pool_type=("mixed" if report["positives"] and report["negatives"]
                                     else "all_positive" if report["positives"] else "all_negative")))
    for (kind, seed), directory, report in zip(APPEND, append_roots, append_reports):
        request = read(directory / "request.json")
        pools.append(dict(seed=seed, family=request["layout_family"],
                          split_group=request["split_group"], batch="append_9_to_16",
                          positives=report["positives"], negatives=report["negatives"]))

    old = realized_choices(RESULTS / "v23_first_near_3002_retry" / "candidates" / "grounded_033")
    new = realized_choices(RESULTS / "v24_append_near_3002" / "candidates" / "grounded_011")
    assert old["plan_sha256"] != new["plan_sha256"]
    assert len(old["selected_transfer_ids"]) == len(new["selected_transfer_ids"])
    diversity = dict(layout_seed=3002, original_candidate="grounded_033",
        appended_candidate="grounded_011", original=old, appended=new,
        changed_grasp_yaw_parts=[part for part in old["grasp_yaw_rad"]
                                 if abs(old["grasp_yaw_rad"][part] - new["grasp_yaw_rad"][part]) > 1e-4],
        changed_selected_transfer_ids=sum(a != b for a, b in zip(
            old["selected_transfer_ids"], new["selected_transfer_ids"])),
        trajectory_claim="different executed grasp choices, pin order and selected path identities; no full joint-trajectory equivalence test")

    samples = [read_line for filename in ("new_layout_first_pass_samples.jsonl", "append_samples.jsonl")
               for read_line in (json.loads(line) for line in (EVIDENCE / filename).open(encoding="utf-8"))]
    categories = {name: dict(Counter(x["failure_type"] for x in subset
                                     if x["actual_execution_label"] is False))
                  for name, subset in (
                      ("new_layout_first_pass", samples[:64]), ("append", samples[64:]))}
    summary = dict(schema="twingraph.v24_collection_analysis.r1",
        task_version=manifest["task_version"], runtime_sha256=manifest["runtime_sha256"],
        new_layout_first_pass=dict(layouts=8, candidates=64,
            positives=sum(x["positives"] for x in first_reports),
            negatives=sum(x["negatives"] for x in first_reports),
            mixed_pools=sum(x["pool_type"] == "mixed" for x in pools[:8])),
        append=dict(layouts=2, candidates=16,
            positives=sum(x["positives"] for x in append_reports),
            negatives=sum(x["negatives"] for x in append_reports),
            first_pass_statistic=False),
        pools=pools, split_groups=groups, failure_categories=categories,
        same_layout_success_diversity=diversity)
    (EVIDENCE / "analysis.json").write_text(json.dumps(summary, ensure_ascii=False,
        indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (EVIDENCE / "all_pools_audit.json").write_text(json.dumps(all_reports, ensure_ascii=False,
        indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(first_pass=summary["new_layout_first_pass"],
                          append=summary["append"], groups=groups,
                          changed_transfer_ids=diversity["changed_selected_transfer_ids"]),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
