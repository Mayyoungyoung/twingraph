#!/usr/bin/env python3
"""Recompile archived model port choices into exact PlanIR and value graphs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simbench.value.atomic_flow import compose_from_template


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--compositions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    proposals = json.loads(args.compositions.read_text(encoding="utf-8"))
    candidates = compose_from_template(request["template_plan"], request["observation"],
                                       proposals, request["allowed_values"])
    args.out.mkdir(parents=True, exist_ok=True)
    for i, candidate in enumerate(candidates):
        stem = f"candidate_{i:02d}"
        (args.out / f"{stem}_plan.json").write_text(
            json.dumps(candidate["plan"].to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        (args.out / f"{stem}_graph.json").write_text(
            json.dumps(candidate["graph"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(count=len(candidates), names=[c["name"] for c in candidates])))


if __name__ == "__main__":
    main()
