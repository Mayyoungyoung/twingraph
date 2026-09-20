"""Read archived V9 trajectories without changing their historical labels."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import tarfile


def audit(folder):
    records = []
    for archive in sorted(Path(folder).glob("seed_*.tar.gz")):
        seed = int(archive.stem.split("_")[1].split(".")[0])
        with tarfile.open(archive) as tar:
            names = set(tar.getnames())
            for condition in ("nominal", "light_low", "light_high"):
                for candidate in ("reference", "pin_order", "pin_slow", "pin_firm", "pin_gentle", "pin_high_grasp", "pin_low_grasp", "transfer_high", "transfer_low", "stop_slow", "wipe_reverse", "wipe_long"):
                    base = f"seed_{seed}/seed_{seed}/{condition}/{candidate}"
                    result_name = f"{base}/result.json"
                    if result_name not in names:
                        continue
                    result = json.load(tar.extractfile(result_name))
                    steps = json.load(tar.extractfile(f"{base}/scene/steps.json"))
                    failed = next((s for s in steps if not s["ok"]), None)
                    records.append(dict(seed=seed, condition=condition, candidate=candidate,
                                        success=result["result"]["full_success"],
                                        failed=failed, steps=steps,
                                        first_failure_index=failed["index"] if failed else None))
    cases = defaultdict(list)
    for r in records:
        cases[(r["seed"], r["condition"])].append(r)
    divergence = defaultdict(Counter)
    divergence_examples = defaultdict(list)
    for key, group in cases.items():
        reference = next(r for r in group if r["candidate"] == "reference")
        for row in group:
            if row is reference:
                continue
            first = next((i for i, (left, right) in enumerate(zip(reference["steps"], row["steps"]))
                          if (left["skill"], left["params"]) != (right["skill"], right["params"])),
                         min(len(reference["steps"]), len(row["steps"])))
            ref_failure = reference["first_failure_index"]
            label = "before_or_at_failure" if ref_failure is not None and first <= ref_failure else "after_failure_or_no_change"
            divergence[row["candidate"]][label] += 1
            divergence_examples[row["candidate"]].append(dict(seed=key[0], condition=key[1], first_different_step=first,
                reference_first_failure=ref_failure, label=label))
    layout = {}
    for seed in sorted({r["seed"] for r in records}):
        group = [r for r in records if r["seed"] == seed]
        successes = sum(r["success"] for r in group)
        layout[seed] = dict(successes=successes, total=len(group),
                            category="all_success" if successes == len(group) else "all_failure" if successes == 0 else "mixed")
    failures = Counter()
    examples = defaultdict(list)
    for r in records:
        f = r["failed"]
        if f:
            part = f["params"].get("part") or "unknown"
            key = (f["skill"], part, f["reason"])
            failures[(f["skill"], part)] += 1
            examples[key].append((r["seed"], r["condition"], r["candidate"], f["index"], f["metrics"]))
    return dict(layout=layout,
                case_categories=Counter("all_success" if all(r["success"] for r in group)
                                        else "all_failure" if not any(r["success"] for r in group) else "mixed"
                                        for group in cases.values()),
                reference_success=sum(next(r["success"] for r in group if r["candidate"] == "reference") for group in cases.values()),
                alternative_only=sum(not next(r["success"] for r in group if r["candidate"] == "reference")
                                     and any(r["success"] for r in group) for group in cases.values()),
                failure_part_skill={f"{skill}:{part}": count for (skill, part), count in failures.most_common()},
                failure_examples={f"{skill}:{part}:{reason}": examples[(skill, part, reason)][:3]
                                  for skill, part, reason in examples},
                control_divergence={name: dict(counts) for name, counts in divergence.items()},
                control_divergence_examples=dict(divergence_examples),
                records=records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archives", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = audit(args.archives)
    brief = {key: value for key, value in result.items() if key != "records"}
    Path(args.out).write_text(json.dumps(brief, indent=2, default=dict), encoding="utf-8")
    print(json.dumps({key: brief[key] for key in ("layout", "case_categories", "reference_success", "alternative_only", "failure_part_skill")}, indent=2, default=dict))


if __name__ == "__main__":
    main()
