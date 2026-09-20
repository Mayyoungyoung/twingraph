"""Summarize targeted physical rollouts, retaining infrastructure exceptions."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    hashes = {}
    for seed in (1305, 1306, 1311):
        path = args.archives / f"seed_{seed}.tar.gz"
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with tarfile.open(path) as tar:
            summary = json.load(tar.extractfile("./summary.json"))
            for entry in summary["rows"]:
                name = entry["candidate"]
                if "infrastructure_error" in entry:
                    rows.append(dict(seed=seed, candidate=name, status="infrastructure_error",
                                     error=entry["infrastructure_error"]))
                    continue
                prefix = f"./seed_{seed}/nominal/{name}"
                detail = json.load(tar.extractfile(f"{prefix}/result.json"))
                steps = json.load(tar.extractfile(f"{prefix}/scene/steps.json"))
                failed = next((step for step in steps if not step["ok"]), None)
                wanted = ("inspect_seat", "carriage") if seed == 1305 else ("lift", "pin_left") if seed == 1306 else ("approach", "handle")
                matched = [step for step in steps if step["skill"] == wanted[0] and step["params"].get("part") == wanted[1]]
                original_stage_passed = any(step["ok"] for step in matched)
                if seed == 1306 and name.endswith("staged"):
                    # A 25 mm first segment is not the original 100 mm lift.
                    original_stage_passed = len(matched) >= 2 and all(step["ok"] for step in matched[:2])
                rows.append(dict(seed=seed, candidate=name, status="completed",
                                 full_success=bool(detail["result"]["full_success"]),
                                 original_failure_skill=f"{wanted[0]}:{wanted[1]}",
                                 original_failure_skill_passed=original_stage_passed,
                                 first_failure=dict(index=failed["index"], skill=failed["skill"],
                                                    part=failed["params"].get("part"), reason=failed["reason"],
                                                    metrics=failed["metrics"]) if failed else None,
                                 executed_steps=len(steps), wall_seconds=entry["total_wall_seconds"]))
    report = dict(archive_sha256=hashes,
                  completed=sum(r["status"] == "completed" for r in rows),
                  infrastructure_exceptions=sum(r["status"] == "infrastructure_error" for r in rows),
                  full_successes=sum(bool(r.get("full_success")) for r in rows),
                  first_failure_skill_counts=dict(Counter(r["first_failure"]["skill"] for r in rows
                      if r.get("first_failure"))), rows=rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("completed", "infrastructure_exceptions", "full_successes", "first_failure_skill_counts")}, indent=2))


if __name__ == "__main__":
    main()
