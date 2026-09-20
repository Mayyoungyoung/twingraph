"""Run preselected screening and fixed-reference control on unseen layouts."""

import argparse
import contextlib
import hashlib
import json
from pathlib import Path

from scripts.run_v9_target_screening import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()
    selection_path = Path(args.selection)
    selection = json.loads(selection_path.read_text())
    if not set(args.seeds).issubset(set(selection["target_seeds"])):
        raise ValueError("target seed not predeclared")
    checkpoint = selection["checkpoint"]
    if checkpoint and hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() != selection["checkpoint_sha256"]:
        raise ValueError("frozen checkpoint hash changed")
    root = Path(args.out); root.mkdir(parents=True, exist_ok=True)
    (root / "selection.json").write_text(json.dumps(selection, indent=2))
    rows = []
    for seed in args.seeds:
        for condition in selection["target_conditions"]:
            methods = [(selection["method"], selection["k"], checkpoint)]
            if selection["method"] != "fixed_reference":
                methods.append(("fixed_reference", 1, None))
            for method, k, model_path in methods:
                directory = root / f"seed_{seed}" / condition / method
                directory.mkdir(parents=True, exist_ok=True)
                with (directory / "console.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
                    try:
                        result = run(seed, condition, method, k, directory, model_path)
                        row = dict(seed=seed, condition=condition, method=method,
                            k=k, success=result["success"],
                            verification_count=result["verification_count"],
                            verification_wall_seconds=result["verification_wall_seconds"],
                            full_system_wall_seconds=result["full_system_wall_seconds"])
                    except Exception as exc:
                        row = dict(seed=seed, condition=condition, method=method,
                                   k=k, infrastructure_error=repr(exc))
                rows.append(row)
                (root / "summary.json").write_text(json.dumps(dict(
                    selection_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest(),
                    rows=rows), indent=2))
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
