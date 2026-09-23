#!/usr/bin/env python3
"""Run Value Top-K model-composed plans through the local physical twin."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simbench.value.plan import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--template-index", type=int, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--k", type=int, required=True)
    args = parser.parse_args()
    ranking = json.loads((args.candidates / "ranking.json").read_text(encoding="utf-8"))
    order = ranking["order"][:args.k]
    if len(order) != args.k or len(set(order)) != len(order):
        raise ValueError("invalid top-k ranking")
    args.out.mkdir(parents=True, exist_ok=True)
    trials = []; selected = None; checkpoint_hash = None
    for rank, index in enumerate(order, 1):
        plan_path = args.candidates / f"candidate_{index:02d}_plan.json"
        graph_path = args.candidates / f"candidate_{index:02d}_graph.json"
        scored_graph = json.loads(graph_path.read_text(encoding="utf-8"))
        run_root = args.out / f"rank_{rank:02d}_candidate_{index:02d}"
        command = [sys.executable, str(Path(__file__).with_name("run_atomic_end_stop_once.py")),
            "--seed", str(args.seed), "--index", str(args.template_index),
            "--out", str(run_root), "--plan", str(plan_path)]
        proc = subprocess.run(command, text=True, capture_output=True,
                              env=os.environ.copy(), check=False)
        (run_root / "stdout.log").write_text(proc.stdout, encoding="utf-8")
        (run_root / "stderr.log").write_text(proc.stderr, encoding="utf-8")
        if proc.returncode:
            raise RuntimeError(f"twin process failed for candidate {index}; inspect {run_root}")
        result = json.loads((run_root / "result.json").read_text(encoding="utf-8"))
        if (not result.get("valid") or result.get("evaluation_scope") != "end_stop_stable_placement_only"
                or result.get("input_graph_sha256") != digest(scored_graph)
                or result.get("full_task_success") is not None):
            raise RuntimeError("twin result is not bound to the scored plan and local task scope")
        if checkpoint_hash is not None and checkpoint_hash != result["physical_checkpoint_sha256"]:
            raise RuntimeError("top-k candidates did not start from the same exact checkpoint")
        checkpoint_hash = result["physical_checkpoint_sha256"]
        row = dict(rank=rank, candidate_index=index, score=ranking["scores"][index],
                   success=result["success"], result_path=str(run_root / "result.json"),
                   input_graph_sha256=result["input_graph_sha256"])
        trials.append(row)
        if result["success"]:
            selected = index
            break
    summary = dict(schema="twingraph.atomic_end_stop_top_k.v1",
        scope="end_stop_stable_placement_only", full_task_success=None,
        seed=args.seed, k=args.k, ranking=ranking,
        trials=trials, selected_index=selected,
        status="verified_success" if selected is not None else "top_k_all_failed",
        checkpoint_sha256=checkpoint_hash)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(status=summary["status"], selected_index=selected,
                          attempts=len(trials)), ensure_ascii=False))


if __name__ == "__main__":
    main()
