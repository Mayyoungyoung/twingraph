#!/usr/bin/env python3
"""Skill-library acceptance: registry contracts, composition, and the
learned insertion layer (headless, pytest).

Checks:
  1. REGISTRY completeness: >=10 skills spanning all four categories,
     every inventory row carries the deliverable fields.
  2. Planning skills run through registry.run on the Task A scene and
     emit multi-candidate artifacts (grasp pose / path).
  3. Execution + transition skills dispatch through registry.run.
  4. run_chain composes skills and aggregates SkillResults.
  5. The executor's REGISTRY fallback runs a registry-only skill in a
     plan (additive; existing tables take precedence).
  6. Learned layer (guarded on torch + gymnasium): build_obs /
     action_to_delta contract, InsertEnv rollout, MLPGaussianPolicy.act,
     and peg_insert(mode='policy') seats the peg from the committed
     checkpoint.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))  # repo root

import numpy as np                                  # noqa: E402
import pytest                                       # noqa: E402

import simbench.skills as S                         # noqa: E402
from simbench.skills import base                    # noqa: E402
from simbench.core.sim_context import MjContext     # noqa: E402
from simbench.core.controller import CartesianController, Gripper  # noqa: E402

_SCENE_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scenes")
TASKA = os.path.join(_SCENE_DIR, "taskA_gearbox.xml")
PEG_SCENE = os.path.join(_SCENE_DIR, "peg_in_hole.xml")
CKPT = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "skills", "learned", "checkpoints",
    "peg_insert.pt")

CATS = ("exec", "plan", "trans", "ext")


def _taskA():
    ctx = MjContext(TASKA)
    ctx.reset()
    return ctx, CartesianController(ctx), Gripper(ctx)


# --------------------------------------------------------- 1. completeness
def test_registry_complete():
    reg = S.REGISTRY
    names = reg.names()
    assert len(names) >= 10, f"only {len(names)} skills registered"
    for c in CATS:
        assert len(reg.names(c)) >= 1, f"no skills in category {c}"
    # the skills the task brief calls out by name must exist
    for expected in ("detect", "inspect", "move", "grasp", "place",
                     "plan_grasp_pose", "plan_path", "approach",
                     "pre_align", "retreat_lift", "return_home",
                     "peg_insert", "wipe", "pull"):
        assert reg.has(expected), f"missing expected skill {expected}"


def test_inventory_rows_have_deliverable_fields():
    rows = S.REGISTRY.inventory()
    assert len(rows) >= 10
    req = ("name", "category", "category_cn", "description", "inputs",
           "outputs", "impl", "deps", "wrapped", "failure_policy",
           "preconditions", "postconditions")
    for r in rows:
        for k in req:
            assert k in r, f"row {r.get('name')} lacks field {k}"
        assert r["category"] in CATS
        assert r["wrapped"] is True
        assert isinstance(r["inputs"], dict)
        assert isinstance(r["outputs"], dict)


def test_inventory_markdown_renders():
    from simbench.skill_inventory import build_rows, to_markdown, to_json
    rows = build_rows()
    md = to_markdown(rows)
    js = to_json(rows)
    assert "技能名称" in md and "实现方式" in md
    for r in rows:
        assert r["name"] in md
    assert js.strip().startswith("[")


# ------------------------------------------------------- 2. planning skills
def test_plan_grasp_pose_multicandidate():
    ctx, arm, grip = _taskA()
    res = base.REGISTRY.run("plan_grasp_pose", ctx, arm, grip,
                            part="gear", n_yaw=6)
    assert res.ok, res.reason
    gp = res.metrics["grasp_pose"]
    assert "pos" in gp and "yaw" in gp
    assert res.metrics["n_candidates"] >= 2


def test_plan_path_collision_gate_and_selection():
    ctx, arm, grip = _taskA()
    tgt = ctx.eef_pos() + np.array([0.05, 0.0, 0.05])
    res = base.REGISTRY.run("plan_path", ctx, arm, grip, to=tgt,
                            style="safe_z")
    assert res.ok, res.reason
    assert len(res.metrics["waypoints"]) >= 1
    assert res.metrics["n_candidates"] >= 1
    # an obstacle straddling the goal makes every candidate infeasible
    ob = [dict(name="block", lo=tgt - 0.05, hi=tgt + 0.05)]
    bad = base.REGISTRY.run("plan_path", ctx, arm, grip, to=tgt,
                            style="direct", obstacles=ob)
    assert not bad.ok


# --------------------------------------------- 3. exec + transition skills
def test_exec_move_and_transitions():
    ctx, arm, grip = _taskA()
    eef0 = ctx.eef_pos().copy()
    mv = base.REGISTRY.run("move", ctx, arm, grip,
                           to=eef0 + np.array([0.0, 0.0, 0.05]),
                           style="direct", tol=0.01)
    assert mv.ok, mv.reason
    assert ctx.eef_pos()[2] > eef0[2] + 0.02
    rl = base.REGISTRY.run("retreat_lift", ctx, arm, grip, height=0.05)
    assert rl.ok and rl.metrics["lift_m"] > 0.02
    pa = base.REGISTRY.run("pre_align", ctx, arm, grip,
                           at=ctx.eef_pos()[:2])
    assert pa.ok and pa.metrics["resid"] < 0.01


def test_detect_exec_skill():
    ctx, arm, grip = _taskA()
    res = base.REGISTRY.run("detect", ctx, arm, grip, part="gear")
    assert res.ok and res.metrics["found"]
    assert res.metrics["pos"] is not None
    # a missing part fails the precondition gate cleanly (no exception)
    bad = base.REGISTRY.run("detect", ctx, arm, grip, part="no_such_part")
    assert not bad.ok and "not in scene" in bad.reason


# -------------------------------------------------------- 4. composition
def test_run_chain_composes():
    ctx, arm, grip = _taskA()
    chain = base.run_chain(base.REGISTRY, ctx, arm, grip, [
        ("detect", {"part": "gear"}),
        ("plan_grasp_pose", {"part": "gear", "n_yaw": 4}),
        ("retreat_lift", {"height": 0.03}),
    ])
    assert isinstance(chain, base.ChainResult)
    assert chain.ok, chain.metrics()
    assert chain.stopped_at == -1
    assert len(chain.results) == 3


def test_run_chain_stops_on_fail():
    ctx, arm, grip = _taskA()
    chain = base.run_chain(base.REGISTRY, ctx, arm, grip, [
        ("detect", {"part": "no_such_part"}),   # fails precondition
        ("retreat_lift", {"height": 0.03}),     # never reached
    ], stop_on_fail=True)
    assert not chain.ok
    assert chain.stopped_at == 0
    assert len(chain.results) == 1


# ------------------------------------------------- 5. executor fallback
def test_executor_registry_fallback():
    from simbench.executor import Executor
    from simbench.planner import TaskPlan
    from simbench.faults import FailureModel
    ctx = MjContext(TASKA)
    ctx.reset()
    ex = Executor(ctx, faults=FailureModel("none", seed=0))
    # retreat_lift is NOT in the executor tables -> registry fallback
    plan = TaskPlan([("retreat_lift", {"height": 0.05})])
    res = ex.run(plan, verbose=False)
    assert res[0]["ok"] is True
    assert res[0]["skill"] == "retreat_lift"


# ---------------------------------------------------- 6. learned layer
@pytest.fixture(scope="module")
def insert_env():
    pytest.importorskip("gymnasium")
    if not os.path.exists(PEG_SCENE):
        pytest.skip("peg_in_hole.xml missing (run gen_insert_scene.py)")
    from simbench.skills.learned.insert_env import InsertEnv
    return InsertEnv(randomize=True)


def test_obs_action_contract(insert_env):
    from simbench.skills.learned.insert_env import (
        build_obs, action_to_delta, OBS_DIM, ACT_DIM)
    obs, info = insert_env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    assert np.all(np.isfinite(obs))
    assert info["held"]              # peg captured in the grip
    d = action_to_delta(np.array([1.0, -1.0, 0.5]))
    assert d.shape == (ACT_DIM,)
    assert np.allclose(d, [0.001, -0.001, 0.0005])


def test_insert_env_rollout(insert_env):
    obs, _ = insert_env.reset(seed=1)
    total = 0.0
    for _ in range(10):
        obs, r, term, trunc, info = insert_env.step(
            insert_env.action_space.sample())
        assert np.isfinite(r)
        total += r
        if term or trunc:
            break
    assert obs.shape == insert_env.observation_space.shape


def test_torch_policy_act(insert_env):
    torch = pytest.importorskip("torch")  # noqa: F841
    from simbench.skills.learned.insert_env import OBS_DIM, ACT_DIM
    from simbench.skills.learned.policy_torch import MLPGaussianPolicy
    pol = MLPGaussianPolicy(OBS_DIM, ACT_DIM, hidden=(32, 32), seed=0)
    obs, _ = insert_env.reset(seed=2)
    a_det = pol.act(obs, deterministic=True)
    a_sto = pol.act(obs, deterministic=False)
    assert a_det.shape == (ACT_DIM,)
    assert np.all(a_det >= -1.0) and np.all(a_det <= 1.0)
    assert a_sto.shape == (ACT_DIM,)


def test_peg_insert_policy_seats(insert_env):
    pytest.importorskip("torch")
    if not os.path.exists(CKPT):
        pytest.skip("no trained checkpoint (run train_insert)")
    from simbench.skills import extension
    env = insert_env
    seeded = 0
    trials = 3
    for s in range(trials):
        env.reset(seed=200 + s)
        res = extension.peg_insert(env.ctx, env.arm, env.gripper, "peg",
                                   env.hole_xy, env.to_z, mode="policy",
                                   half=env.half, max_steps=200)
        seeded += int(res.ok)
    # the committed BC checkpoint is measured ~95%; require a majority
    assert seeded >= 2, f"policy seated only {seeded}/{trials}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
