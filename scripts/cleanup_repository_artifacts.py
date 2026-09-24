"""Remove archived experiment payloads while retaining source and written reports.

Run from the repository root. The default is a dry run; --apply removes only
tracked paths selected below, after checking every resolved path stays in root.
"""

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KEEP_EVIDENCE = {
    "docs/evidence/value_v15_full_system/protocol/llm_round0_response.json",
    "docs/evidence/value_v13_mechanism/splits_development.json",
}


def selected(path: str) -> bool:
    p = Path(path)
    if path.startswith(("datasets/", "models/")):
        return True
    if path.startswith("docs/demos/"):
        return True
    if path.startswith("docs/evidence/"):
        if p.suffix.lower() in {".md", ".png", ".jpg", ".svg"}:
            return False
        if path.startswith("docs/evidence/value_v20_full_flow/") and p.parent.name == "value_v20_full_flow" and p.suffix == ".json":
            return False
        return path not in KEEP_EVIDENCE
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    paths = sorted(p for p in paths if p and selected(p))
    total_bytes = 0
    for rel in paths:
        absolute = (ROOT / rel).resolve()
        if not absolute.is_relative_to(ROOT.resolve()):
            raise RuntimeError(f"unsafe path: {rel}")
        if absolute.is_file():
            total_bytes += absolute.stat().st_size
    print(json.dumps({"files": len(paths), "bytes": total_bytes, "apply": args.apply}, indent=2))
    if args.apply:
        for start in range(0, len(paths), 100):
            subprocess.run(["git", "rm", "--sparse", "--", *paths[start : start + 100]], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
