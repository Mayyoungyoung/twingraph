#!/usr/bin/env python3
"""Development-only probe of fresh RGB-D registration after a physical grasp.

The probe reads simulator object poses only for its diagnostic error report.
Its modified execution is never eligible for value labels or pool coverage.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from simbench.assembly import skills_v12
from simbench.value.stage_v7 import refresh_visual_observation
from simbench.value.system_v12 import rollout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    proposal = next((row for row in request["pool"] if row["name"] == args.name), None)
    if proposal is None:
        parser.error("candidate name is absent from the frozen request")
    samples = []
    original = skills_v12.bind_grasp_observation

    def probe(session, part):
        original(session, part)
        saved_observation = copy.deepcopy(session.decision_observation)
        saved_acquisition = copy.deepcopy(getattr(session, "last_rgbd_acquisition", None))
        saved_registration = copy.deepcopy(session.held_visual_transforms_v12[part])
        true_position = np.asarray(session.ctx.obj_pos(part), float).copy()
        predicted = (np.asarray(session.ctx.eef_pos(), float)
                     + np.asarray(session.ctx.eef_mat(), float)
                     @ np.asarray(saved_registration["local_position"], float))
        sample = dict(part=part, preclose_fk_error_m=float(np.linalg.norm(predicted-true_position)),
                      oracle_used_for_diagnostic_only=True)
        try:
            observation = refresh_visual_observation(session, parts=session.parts)
            row = observation.get("objects", {}).get(part, {})
            sample["fresh_valid"] = bool(row.get("valid"))
            sample["fresh_scan_status"] = observation.get("acquisition", {}).get("scan_status")
            if row.get("valid") and row.get("position_m") is not None:
                fresh = np.asarray(row["position_m"], float)
                sample["fresh_rgbd_error_m"] = float(np.linalg.norm(fresh-true_position))
                sample["fresh_minus_preclose_m"] = float(np.linalg.norm(fresh-predicted))
                sample["fresh_fit_residual_m"] = row.get("fit_residual_m")
        except Exception as exc:
            sample["fresh_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            session.set_decision_observation(saved_observation)
            session.last_rgbd_acquisition = saved_acquisition
            session.held_visual_transforms_v12[part] = saved_registration
            samples.append(sample)

    skills_v12.bind_grasp_observation = probe
    args.out.mkdir(parents=True, exist_ok=True)
    result = rollout(int(request["seed"]), proposal, args.out / "rollout",
                     domain=request["domain"], level=request["level"])
    report = dict(schema="twingraph.held_rgbd_diagnostic.v20.r1",
                  development_only=True, training_label_eligible=False,
                  seed=request["seed"], candidate=args.name,
                  runtime_sha256=result.get("runtime_sha256"),
                  full_success=result.get("full_success"), error=result.get("error"),
                  samples=samples)
    (args.out / "probe.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "error"},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
