"""Nine independent V9 functional assembly runs on three development layouts."""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path

from run_v9_full_task import run
from simbench.value.stage_v9 import TASK_VERSION


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[1200, 1202, 1203])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--pose-mode", choices=("rgbd", "ideal"), default="rgbd")
    a = p.parse_args()
    root = Path(a.out); root.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parents[1]
    hashes = {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in (source / "simbench").rglob("*.py")}
    rows = []
    for seed in a.seeds:
        for repeat in range(a.repeats):
            directory = root / f"seed_{seed}_repeat_{repeat}"
            directory.mkdir(exist_ok=True)
            with (directory / "console.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
                try:
                    outcome = run(seed, directory, a.pose_mode)["result"]
                    row = dict(seed=seed, repeat=repeat, success=outcome["success"],
                               error=outcome.get("error"), stages=outcome.get("stage_passes"),
                               steps=outcome.get("executed_steps"),
                               wall_seconds=outcome.get("wall_seconds"), run=directory.name)
                except Exception as exc:
                    row = dict(seed=seed, repeat=repeat, infrastructure_error=repr(exc), run=directory.name)
            rows.append(row)
            (root / "summary.json").write_text(json.dumps(dict(
                protocol="v9.functional_assembly.independent_session",
                task_version=TASK_VERSION, pose_mode=a.pose_mode,
                source_sha256=hashes, rows=rows), indent=2), encoding="utf-8")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
