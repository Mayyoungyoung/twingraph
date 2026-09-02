#!/usr/bin/env python3
"""Fault model + two-class dispatch checks (headless, pytest).

Checks:
  1. FailureModel determinism: same (profile, seed) -> identical outcome
     sequences and identical event logs; different seeds diverge.
  2. FailureModel outcome rates: over many trials the sampled miss/
     false/unreachable/collision/slip rates approximate the profile
     probabilities (loose bounds).
  3. Two-class dispatch smoke: an Executor runs a plan-only sequence
     (scatter_parts + detect_part + plan_grasp_pose + plan_path) with
     zero physics cost and the results carry the right kind tags and
     plan artifacts.
  4. A failed detect (p_miss=1 profile) makes plan_grasp_pose fail and
     records the attributed fault event.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))  # repo root

import numpy as np
import pytest  # noqa: E402

from simbench.core.sim_context import MjContext  # noqa: E402
from simbench.executor import Executor  # noqa: E402
from simbench.faults import FailureModel  # noqa: E402
from simbench.planner import TaskPlan  # noqa: E402

SCENE = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scenes", "taskA_gearbox.xml")


# ---------------------------------------------------------------- model
def test_fault_model_deterministic():
    a = FailureModel("default", seed=7)
    b = FailureModel("default", seed=7)
    seq_a = [a.detect_outcome(), a.plan_path_outcome(),
             a.slip_event(), a.detect_noise(), a.move_error(),
             a.measure_noise(), a.init_jitter()]
    seq_b = [b.detect_outcome(), b.plan_path_outcome(),
             b.slip_event(), b.detect_noise(), b.move_error(),
             b.measure_noise(), b.init_jitter()]
    for x, y in zip(seq_a[:3], seq_b[:3]):
        assert x == y
    # detect_noise returns a (pos3, yaw) tuple; the rest are arrays
    for x, y in zip(seq_a[3:5], seq_b[3:5]):
        assert np.allclose(x[0], y[0]) and np.isclose(x[1], y[1])
    for x, y in zip(seq_a[5:], seq_b[5:]):
        assert np.allclose(x, y)
    assert [e["source"] for e in a.log] == [e["source"] for e in b.log]

    c = FailureModel("default", seed=8)
    outcomes_c = [c.detect_outcome() for _ in range(50)]
    outcomes_a = [a.detect_outcome() for _ in range(50)]
    assert outcomes_c != outcomes_a      # different seeds diverge


def test_fault_model_rates():
    fm = FailureModel("default", seed=1)
    n = 2000
    counts = {"miss": 0, "false": 0, "unreachable": 0, "collision": 0,
              "slip": 0}
    for _ in range(n):
        o = fm.detect_outcome()
        if o == "miss":
            counts["miss"] += 1
        elif o == "false":
            counts["false"] += 1
        p = fm.plan_path_outcome()
        if p == "unreachable":
            counts["unreachable"] += 1
        elif p == "collision":
            counts["collision"] += 1
        if fm.slip_event():
            counts["slip"] += 1
    pr = fm.params
    assert abs(counts["miss"] / n - pr["p_miss"]) < 0.02
    assert abs(counts["false"] / n - pr["p_false"]) < 0.02
    assert abs(counts["unreachable"] / n - pr["p_unreachable"]) < 0.02
    assert abs(counts["collision"] / n - pr["p_collision"]) < 0.02
    assert abs(counts["slip"] / n - pr["p_slip"]) < 0.02
    assert fm.summary()


def test_fault_model_none_is_quiet():
    fm = FailureModel("none", seed=0)
    assert fm.detect_outcome() == "ok"
    assert fm.plan_path_outcome() == "ok"
    assert not fm.slip_event()
    assert np.allclose(fm.detect_noise()[0], 0.0)
    assert not fm.log


# ------------------------------------------------------------ dispatch
def _ctx():
    ctx = MjContext(SCENE)
    ctx.reset()
    return ctx


def test_two_class_dispatch_smoke():
    """A plan-only sequence runs with zero arm motion; the results
    carry kind tags and the plan artifacts land in the state."""
    ctx = _ctx()
    ex = Executor(ctx, faults=FailureModel("none", seed=0))
    plan = TaskPlan([
        ("scatter_parts", {"parts": ["gear"]}),
        ("detect_part", {"part": "gear"}),
        ("plan_grasp_pose", {"part": "gear", "yaw": None}),
        ("plan_path", {"to": {"part": "gear", "lift": 0.15},
                       "style": "safe_z", "as": "path_gear_pick"}),
    ])
    res = ex.run(plan, verbose=False)
    kinds = [r["kind"] for r in res]
    assert kinds == ["exec", "exec", "plan", "plan"]
    assert all(r["ok"] for r in res)
    assert ex.state["percepts"]["gear"]["pos"] is not None
    assert ex.state["plans"]["grasp_part"] == "gear"
    assert ex.state["plans"]["grasp"]["outer_d"] == 0.020
    assert len(ex.state["plans"]["path_gear_pick"]["waypoints"]) == 3


def test_detect_miss_propagates():
    """p_miss=1 forces plan_grasp_pose to fail and the fault event is
    attributed in the log."""
    ctx = _ctx()
    fm = FailureModel("none", seed=0)
    fm.params["p_miss"] = 1.0
    ex = Executor(ctx, faults=fm)
    plan = TaskPlan([
        ("detect_part", {"part": "gear"}),
        ("plan_grasp_pose", {"part": "gear"}),
    ])
    res = ex.run(plan, verbose=False)
    assert res[0]["ok"] is False
    assert res[1]["ok"] is False
    assert any(e["source"] == "grasp_est_fail" for e in fm.log)


def test_plan_path_collision_gate():
    """A path whose travel leg crosses a tray wall is rejected by the
    real geometric check (with zero fault probability)."""
    ctx = _ctx()
    fm = FailureModel("none", seed=0)
    ex = Executor(ctx, faults=fm)
    # from above the tray centre to below the wall band, cutting
    # through the wall: a clearance-free direct hop collides
    plan = TaskPlan([
        ("plan_path", {"to": (0.12, 0.0, 0.805), "style": "direct",
                       "as": "bad_path"}),
    ])
    res = ex.run(plan, verbose=False)
    assert res[0]["ok"] is False
    assert "bad_path" not in ex.state.get("plans", {})


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
