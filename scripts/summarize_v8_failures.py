"""Extract terminal atom, part and command from the V8 development log."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    group = Path(args.group)
    outcomes = json.loads((group / "outcomes.json").read_text())["trials"]
    segments = []
    for line in (group / "steps.log").read_text(encoding="utf-8").splitlines():
        try:
            step = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(step, dict) or "skill" not in step or "index" not in step:
            continue
        if step["index"] == 0:
            segments.append([])
        if segments:
            segments[-1].append(step)
    if len(segments) != len(outcomes):
        raise ValueError(f"step log has {len(segments)} executions, outcomes have {len(outcomes)}")
    rows = []
    for outcome, steps in zip(outcomes, segments):
        failures = [s for s in steps if not s["ok"] and s.get("reason")]
        terminal = failures[-1] if failures else None
        part = terminal["params"].get("part", "") if terminal else ""
        if not part and terminal and terminal["skill"] == "plan_transfer":
            checks = terminal["metrics"].get("checks", [])
            if checks:
                part = checks[0].get("route_id", "").split(":", 1)[0]
        rows.append(dict(candidate_id=outcome["candidate_id"],
                         success=outcome["success"],
                         stage="cleaning" if not outcome["stage_passes"].get("cleaning_pass") else
                               "assembly" if not outcome["stage_passes"].get("assembly_pass") else
                               "functional_test",
                         atom=terminal["skill"] if terminal else "",
                         part=part,
                         command=json.dumps(terminal["params"], ensure_ascii=False) if terminal else "",
                         reason=terminal["reason"] if terminal else outcome["error"],
                         metrics=json.dumps(terminal["metrics"], ensure_ascii=False) if terminal else ""))
    with Path(args.out).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
