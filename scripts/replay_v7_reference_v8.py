"""Replay the recorded V7 success configuration against current code.

The historical result did not save PlanIR itself. Reconstruct it using the
recorded script arguments, verify the saved scene spec and candidate ID, and
save the reconstructed plan before execution. Any mismatch aborts the replay.
"""
import argparse
import json
from pathlib import Path
import numpy as np

from simbench.value import stage_v7, stage_v5
from simbench.value.physical import PhysicalRunner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--fixture-pose-mode", choices=("rgbd", "ideal_diagnostic"), default="rgbd")
    parser.add_argument("--fixture-yaw-mode", choices=("rgbd", "nominal_constraint_diagnostic"), default="rgbd")
    args = parser.parse_args()
    record = json.loads(Path(args.record).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec, session, _, targets = stage_v7.make_scene(record["seed"], out, level=record["level"])
    if spec.__dict__ != record["spec"]:
        raise ValueError("historical scene specification does not match")
    choices = {
        part: dict(yaw=0., height=.001, clearance=.98,
                   force=float(record["pin_force_N"] if part.startswith("pin_") else 3.),
                   **({} if part == "carriage" else {"speed": float(record["pin_speed_m_s"]) }))
        for part in stage_v5.PARTS
    }
    plan = stage_v7._full_plan(session, targets, stage_v5.legal_orders()[0], choices,
                               wipe_variant=0, wipe_force=1.5, wipe_duration=14.,
                               stroke_minimum=stage_v7.TASK_STROKE_MINIMUM_M)
    if plan.id != record["candidate_id"]:
        raise ValueError(f"historical candidate ID mismatch: {plan.id}")
    session.fixture_pose_mode = args.fixture_pose_mode
    session.fixture_yaw_mode = args.fixture_yaw_mode
    (out / "reconstructed_plan.json").write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
    result = PhysicalRunner(session, timeout=args.timeout).run(
        plan, record["result"]["trial"], keep_trace=True)
    (out / "replay_result.json").write_text(
        json.dumps(result, indent=2, default=lambda x: np.asarray(x).tolist()), encoding="utf-8")
    print(json.dumps({"candidate_id": plan.id, "success": result["success"],
                      "error": result["error"], "executed_steps": result["executed_steps"],
                      "wall_seconds": result["wall_seconds"]}))


if __name__ == "__main__":
    main()
