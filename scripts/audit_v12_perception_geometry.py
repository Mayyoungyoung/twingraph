#!/usr/bin/env python3
"""Read-only RGB-D pose accuracy audit; simulation truth is evaluator-only."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from simbench.value.stage_v12 import make_scene
from simbench.value.stage_v7 import refresh_visual_observation, capture_vision, capture_detector
from simbench.assembly.skills_v12 import configure_v12_skills


def run(seed, out, preinstalled_stop=False):
    _, session, _, _ = make_scene(seed, out, preinstalled_end_stop=preinstalled_stop)
    configure_v12_skills(session)
    observation = refresh_visual_observation(session)
    # The detector has already returned. Ground truth below is only a diagnostic
    # comparison and is not stored in Session or used for any control decision.
    rows = {}
    for part, row in observation["objects"].items():
        truth = session.ctx.obj_pos(part).copy()
        prediction = np.asarray(row["position_m"], float) if row.get("position_m") is not None else None
        rows[part] = dict(valid=bool(row["valid"]), predicted_position_m=prediction.tolist() if prediction is not None else None,
                         evaluator_only_truth_position_m=truth.tolist(),
                         error_xyz_mm=((prediction - truth) * 1000).tolist() if prediction is not None else None,
                         error_norm_mm=float(np.linalg.norm(prediction - truth) * 1000) if prediction is not None else None,
                         detector_diagnostics=row)
    out.mkdir(parents=True, exist_ok=True)
    frames, calibration = capture_detector(session)
    np.savez_compressed(out / "frames.npz", **{view + "_" + key: value for view, frame in frames.items() for key, value in frame.items()})
    (out / "calibrations.json").write_text(json.dumps({k:v.manifest() for k,v in calibration.items()}, indent=2), encoding="utf-8")
    from PIL import Image, ImageDraw
    views = capture_vision(session, (640, 800))
    canvas = Image.new("RGB", (1600, 675), "#16202b")
    for i, camera in enumerate(("task_view", "top_view")):
        canvas.paste(Image.fromarray(views[camera + "_rgb"]), (i * 800, 35))
    ImageDraw.Draw(canvas).text((12, 10), f"Printed V12 / RGB-D audit / seed {seed} / simulation truth is read only after detection", fill="white")
    canvas.save(out / "scene.png")
    return dict(seed=seed, preinstalled_stop=preinstalled_stop, provenance="rendered calibrated RGB-D -> detector, then read-only evaluator truth comparison", objects=rows)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seeds", default="1600,1603,1604")
    p.add_argument("--preinstalled-stop", action="store_true"); args = p.parse_args()
    rows = [run(int(seed), args.out / str(seed), args.preinstalled_stop) for seed in args.seeds.split(",")]
    summary = dict(schema="twingraph.perception-pose-audit.v12",
                   truth_usage="offline errors only; neither model input nor controller feedback", cases=rows)
    (args.out / "pose_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps([{r["seed"]: {p: {"valid": x["valid"], "mm": round(x["error_norm_mm"], 2) if x["error_norm_mm"] is not None else None} for p, x in r["objects"].items()}} for r in rows]))


if __name__ == "__main__": main()
