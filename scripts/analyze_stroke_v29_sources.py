#!/usr/bin/env python3
"""Keep V28 and V27 post-pin outcome audits separate for the V29 stroke work."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from simbench.value.system_v12 import save


def outcome(result):
    if result.get("success"):
        return "complete_success"
    audit = result.get("stage_audit") or {}
    if not audit.get("handle", {}).get("completed"):
        return "handle_failure"
    if not audit.get("stroke", {}).get("completed"):
        return "stroke_or_release_failure"
    return "final_acceptance_failure"


def v28_rows(root):
    summary = json.loads((root / "results/v28_pin_budget/full_pairs/summary.json").read_text(
        encoding="utf-8"))
    rows = []
    for layout in summary["layouts"]:
        for pair in layout["pairs"]:
            item = pair["new"]
            if not item["pins_completed"]:
                continue
            result = json.loads((root / Path(item["result"])).read_text(encoding="utf-8"))
            atom = result.get("first_failed_atom") or {}
            metrics = atom.get("metrics") or {}
            rows.append(dict(
                layout=layout["name"], candidate=pair["candidate"],
                outcome=outcome(result), failed_skill=atom.get("skill"),
                failure_reason=atom.get("reason"),
                peak_force_n=metrics.get("peak_force_n"),
                cross_axis_m=metrics.get("cross_axis_m"),
                wall_seconds=result.get("total_wall_seconds"),
                result=str(item["result"]),
            ))
    return rows


def v27_rows(root):
    base = root / "results/v27_bulk"
    rows = []
    completed_layouts = []
    for layout in sorted(path for path in base.iterdir() if path.is_dir()):
        summary_path = layout / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("attempted") != 8 or summary.get("valid") != 8:
            continue
        completed_layouts.append(layout.name)
        for entry in summary["candidates"]:
            path = layout / "candidates" / entry["name"] / "result.json"
            result = json.loads(path.read_text(encoding="utf-8"))
            audit = result.get("stage_audit") or {}
            if not all(audit.get(pin, {}).get("completed") for pin in ("pin_left", "pin_right")):
                continue
            atom = result.get("first_failed_atom") or {}
            rows.append(dict(layout=layout.name, candidate=entry["name"],
                outcome=outcome(result), failed_skill=atom.get("skill"),
                failure_reason=atom.get("reason"), result=str(path.relative_to(root))))
    return completed_layouts, rows


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path,
                        default=Path("results/v29_stroke/source_audit.json"))
    args = parser.parse_args(); root = args.root.resolve()
    recent = v28_rows(root)
    completed, prior = v27_rows(root)
    payload = dict(
        schema="twingraph.constrained_stroke_v29.source_audit.r1",
        batches_must_not_be_combined=True,
        v28=dict(
            source="V28 fixed three-layout complete comparison, remaining-distance pin budget",
            post_dual_pin_count=len(recent),
            outcomes=dict(Counter(row["outcome"] for row in recent)),
            rows=recent,
        ),
        v27=dict(
            source="V27 audited first pass only; interrupted layouts excluded",
            completed_layouts=len(completed),
            post_dual_pin_count=len(prior),
            outcomes=dict(Counter(row["outcome"] for row in prior)),
            stroke_failure_atoms={
                f"{skill}: {reason}": count for (skill, reason), count in Counter(
                    (row["failed_skill"], row["failure_reason"]) for row in prior
                    if row["outcome"] == "stroke_or_release_failure").items()
            },
        ),
    )
    save(args.out, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
