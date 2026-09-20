"""Collect paired V9 candidate outcomes, one fresh Session per rollout."""

import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np

from simbench.value import stage_v9
from simbench.value.physical import PhysicalRunner
from simbench.value.v9_candidates import bind, proposals, SOURCE


CONDITIONS = {
    "nominal": (1.0, 1.0),
    "light_low": (0.95, 0.995),
    "light_high": (1.05, 1.005),
}


def geometry_signature(scene):
    # Hash the emitted physical geometry, not the input seed or camera pose.
    root = ET.parse(scene).getroot()
    row = [(element.tag, sorted(element.attrib.items()))
           for element in root.iter() if element.tag in ("body", "geom", "joint")]
    return hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()


def collect(seed, proposal, condition, directory):
    directory.mkdir(parents=True, exist_ok=True)
    spec, session, scene, targets = stage_v9.make_scene(seed, directory / "scene")
    session.fixture_pose_mode = "rgbd"
    initial = session.decision_observation
    pre_execution_observation = dict(
        backend=initial["backend"], sha256=initial["sha256"],
        decision_boundary=initial["decision_boundary"],
        objects={part: {key: value for key, value in item.items()
                        if key in ("valid", "position_m", "quat_wxyz", "quality", "source_view", "fit_residual_m")}
                 for part, item in initial["objects"].items()})
    plan = bind(session, targets, proposal)
    friction, gain = CONDITIONS[condition]
    trial = dict(domain="development", repeat=0, friction_scale=friction,
                 actuator_gain_scale=gain)
    result = PhysicalRunner(session, timeout=600.).run(plan, trial, keep_trace=True)
    manifest = json.loads((directory / "scene" / "geometry_manifest.json").read_text())
    row = dict(task_version=stage_v9.TASK_VERSION, seed=seed,
               geometry_sha256=geometry_signature(scene),
               pre_execution_observation=pre_execution_observation,
               candidate=proposal, candidate_id=plan.id, condition=condition,
               trial=trial, result=result)
    (directory / "result.json").write_text(json.dumps(row, indent=2,
                  default=lambda x: np.asarray(x).tolist()), encoding="utf-8")
    return dict(seed=seed, geometry_sha256=row["geometry_sha256"],
                candidate=proposal["name"], source=SOURCE, condition=condition,
                success=result["success"], error=result["error"],
                wall_seconds=result["wall_seconds"], sim_seconds=result["sim_seconds"],
                steps=result["executed_steps"], path=str(directory))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--candidates", nargs="*", default=None)
    parser.add_argument("--conditions", nargs="+", choices=tuple(CONDITIONS), default=list(CONDITIONS))
    args = parser.parse_args()
    root = Path(args.out); root.mkdir(parents=True, exist_ok=True)
    pool = proposals()
    if args.candidates is not None:
        wanted = set(args.candidates)
        pool = [p for p in pool if p["name"] in wanted]
        if len(pool) != len(wanted):
            raise ValueError("unknown or duplicate candidate name")
    (root / "candidate_pool.json").write_text(json.dumps(pool, indent=2), encoding="utf-8")
    source_root = Path(__file__).resolve().parents[1]
    source_hashes = {str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in (source_root / "simbench").rglob("*.py")}
    summary_path = root / "summary.json"
    rows = json.loads(summary_path.read_text())["rows"] if summary_path.exists() else []
    completed = {(r["seed"], r["candidate"], r["condition"]) for r in rows}
    for seed in args.seeds:
        for condition in args.conditions:
            for proposal in pool:
                key = (seed, proposal["name"], condition)
                if key in completed:
                    continue
                directory = root / f"seed_{seed}" / condition / proposal["name"]
                directory.parent.mkdir(parents=True, exist_ok=True)
                t0 = time.perf_counter()
                with (directory.parent / f"{proposal['name']}.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
                    try:
                        row = collect(seed, proposal, condition, directory)
                    except Exception as exc:
                        row = dict(seed=seed, candidate=proposal["name"], condition=condition,
                                   infrastructure_error=repr(exc), path=str(directory))
                row["total_wall_seconds"] = time.perf_counter() - t0
                rows.append(row); completed.add(key)
                summary_path.write_text(json.dumps(dict(task_version=stage_v9.TASK_VERSION,
                    candidate_source=SOURCE, conditions=CONDITIONS,
                    source_sha256=source_hashes, rows=rows), indent=2), encoding="utf-8")
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
