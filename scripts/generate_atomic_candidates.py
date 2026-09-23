#!/usr/bin/env python3
"""Create an auditable LLM request and obtain validated atomic plans."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simbench.value.atomic_flow import planner_request
from simbench.value.codex_planner import generate_with_codex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True, help="TaskSpec JSON")
    parser.add_argument("--observation", type=Path, required=True, help="Pre-execution observation JSON")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--codex", default="codex")
    parser.add_argument("--request-only", action="store_true")
    args = parser.parse_args()
    task = json.loads(args.task.read_text(encoding="utf-8"))
    observation = json.loads(args.observation.read_text(encoding="utf-8"))
    request = planner_request(task, observation, candidate_count=args.n)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.request_only:
        print(json.dumps(dict(status="request_created", path=str(args.out / "request.json"))))
        return
    _, candidates = generate_with_codex(request, out=args.out, executable=args.codex)
    print(json.dumps(dict(status="compiled", candidates=len(candidates),
                          names=[row["name"] for row in candidates]), ensure_ascii=False))


if __name__ == "__main__":
    main()
