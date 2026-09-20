"""Independent-session reference executions for three development layouts."""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path

from run_v7_full_task_smoke import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1200, 1202, 1203])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parents[1]
    version = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in (source / "simbench").rglob("*.py")}
    rows = []
    for seed in args.seeds:
        for repeat in range(args.repeats):
            directory = root / f"seed_{seed}_repeat_{repeat}"
            directory.mkdir(exist_ok=True)
            with (directory / "console.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
                try:
                    result = run(seed, directory, level="L1", pin_force=8.,
                                 pin_speed=.006, timeout=600., record=False,
                                 nominal_trial=True)
                    trial = result["result"]
                    row = dict(seed=seed, repeat=repeat, success=trial["success"],
                               error=trial["error"], stages=trial["stage_passes"],
                               steps=trial["executed_steps"], wall_seconds=trial["wall_seconds"],
                               run=str(directory.name))
                except Exception as exc:
                    row = dict(seed=seed, repeat=repeat, infrastructure_error=repr(exc),
                               run=str(directory.name))
            rows.append(row)
            (root / "summary.json").write_text(
                json.dumps(dict(protocol="v8.reference.nominal.independent_session",
                                source_sha256=version, rows=rows), indent=2), encoding="utf-8")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
