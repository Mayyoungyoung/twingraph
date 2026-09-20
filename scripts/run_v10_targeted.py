"""Run targeted V10 script plans on three historical V9 development layouts."""

import argparse
import contextlib
import json
from pathlib import Path
import time

from scripts.collect_v9_candidate_matrix import collect
from simbench.value.v10_candidates import proposals, SOURCE


CASES = {
    1305: ("reference", "carriage_yaw90", "carriage_grasp_low", "pin_left_yaw90"),
    1306: ("reference", "pin_left_yaw90", "pin_left_grasp_high", "pin_left_yaw90_high", "carriage_yaw90"),
    1311: ("reference", "handle_yaw90", "handle_grasp_high", "handle_yaw90_high", "pin_left_yaw90"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(CASES))
    parser.add_argument("--condition", default="nominal")
    parser.add_argument("--candidates", nargs="+", default=None)
    args = parser.parse_args()
    pool = {row["name"]: row for row in proposals()}
    args.out.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        if seed not in CASES:
            raise ValueError(f"unregistered diagnosis seed {seed}")
        for name in (args.candidates or CASES[seed]):
            if name not in pool:
                raise ValueError(f"unknown candidate {name}")
            directory = args.out / f"seed_{seed}" / args.condition / name
            if (directory / "result.json").exists():
                continue
            started = time.perf_counter()
            directory.parent.mkdir(parents=True, exist_ok=True)
            with (directory.parent / f"{name}.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
                try:
                    row = collect(seed, pool[name], args.condition, directory)
                    row["source"] = SOURCE
                except Exception as exc:
                    row = dict(seed=seed, condition=args.condition, candidate=name, source=SOURCE,
                               infrastructure_error=repr(exc))
            row["total_wall_seconds"] = time.perf_counter() - started
            summary = args.out / "summary.json"
            rows = json.loads(summary.read_text())["rows"] if summary.exists() else []
            rows.append(row)
            summary.write_text(json.dumps(dict(source=SOURCE, task_version="functional_assembly_v9_funnel_r1", rows=rows), indent=2), encoding="utf-8")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
