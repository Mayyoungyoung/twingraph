"""Reproducible multi-grasp prefix trials; outcomes are separate from ranker inputs.

This is a small digital-twin example, not a trained value model or a claim that
an entire assembly suffix has been verified.
"""

import argparse
import contextlib
import json
from pathlib import Path
import numpy as np
from .demo_scenes import make_demo_session
from .candidates import build_pick_candidates, execute_pick_candidate
from .library import SkillFailure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/candidate_demo")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    s = make_demo_session("cube", out / "scene")
    with (out / "steps.log").open("w") as log, contextlib.redirect_stdout(log):
        s.call("observe")
        s.call("estimate_pose", part="cube")
        s.call("propose_grasps", part="cube")
        candidates = build_pick_candidates(s, "cube")
        (out / "candidates.json").write_text(
            json.dumps(
                [c.to_dict() for c in candidates],
                indent=2,
                default=lambda x: np.asarray(x).tolist(),
            )
        )
        # Deliberately test one route for each distinct grasp, for coverage.
        selected = {}
        for candidate in sorted(candidates, key=lambda c: c.cost):
            if candidate.status == "necessary_pass":
                selected.setdefault(candidate.grasp["id"], candidate)
        state = s.snapshot()
        outcomes = []
        try:
            for candidate in selected.values():
                s.restore(state)
                ok, error = False, ""
                try:
                    execute_pick_candidate(s, candidate)
                    ok = True
                except (ValueError, SkillFailure) as exc:
                    error = str(exc)
                outcomes.append(
                    dict(
                        candidate_id=candidate.id,
                        success=ok,
                        error=error,
                        held=s.held,
                        object_xyz=s.ctx.obj_pos("cube").tolist(),
                        scope="pick_prefix_only",
                    )
                )
        finally:
            s.restore(state)
    summary = dict(
        generated=len(candidates),
        distinct_grasps=len(selected),
        tested=len(outcomes),
        successful=sum(r["success"] for r in outcomes),
        outcomes=outcomes,
    )
    (out / "outcomes.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
    if not outcomes or not all(r["success"] for r in outcomes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
