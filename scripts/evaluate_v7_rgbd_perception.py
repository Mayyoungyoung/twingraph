"""Small held-out RGB-D geometry audit for the v7 development protocol.

The detector only receives saved rendered RGB-D and calibration.  MuJoCo
poses are read after inference solely by this independent evaluator to report
error; they never enter the detector or execution observation.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from simbench.value import stage_v7


def run(out: Path, seeds_per_level: int = 4):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for level, start in (("L0", 1300), ("L1", 1400), ("L2", 1500)):
        for seed in range(start, start + seeds_per_level):
            scene_dir = out / f"{level.lower()}_{seed}"
            _spec, session, _xml, _targets = stage_v7.make_scene(seed, scene_dir, level=level)
            observation = session.decision_observation
            for part in stage_v7.ALL_PARTS:
                estimated = observation["objects"].get(part, {})
                valid = bool(estimated.get("valid")) and estimated.get("position_m") is not None
                truth = np.asarray(session.ctx.obj_pos(part), dtype=float)
                if valid:
                    pos = np.asarray(estimated["position_m"], dtype=float)
                    error = float(np.linalg.norm(pos - truth))
                else:
                    error = float("nan")
                rows.append({
                    "level": level, "seed": seed, "part": part,
                    "valid": int(valid), "position_error_m": error,
                    "quality": estimated.get("quality"),
                    "source_view": estimated.get("source_view"),
                    "backend": observation.get("backend"),
                    "detector_resolution": "x".join(map(str, observation.get("detector_resolution", []))),
                })
    path = out / "perception_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(path.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds-per-level", type=int, default=4)
    args = parser.parse_args()
    run(Path(args.out), args.seeds_per_level)
