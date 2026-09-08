"""Regression boundaries for shared contracts and coupled multi-solution planning."""

import copy
import numpy as np
import pytest
from simbench.assembly.control import make_context
from simbench.assembly.library import Session, SkillFailure, CATALOG
from simbench.assembly.graph import validate_skeleton, execute_skeleton, catalog_graph
from simbench.assembly.candidates import (
    build_pick_candidates,
    choose_candidate,
    execute_pick_candidate,
    transfer_routes,
    release_space_proxy,
)
from simbench.assembly.demo_scenes import make_demo_session


def prepare(s, part):
    s.call("observe")
    s.call("estimate_pose", part=part)
    s.call("propose_grasps", part=part)


@pytest.mark.parametrize(
    "name,params",
    [
        ("lift", {"part": "carriage"}),
        ("move", {"mode": "lift", "part": "carriage"}),
        ("move", {"mode": "lower", "part": "carriage"}),
    ],
)
def test_aliases_preserve_pre_motion_grasp_contract(name, params):
    s = Session(make_context())
    before = s.ctx.snapshot()
    assert validate_skeleton([dict(skill=name, params=params)])["status"] == "conflict"
    with pytest.raises(SkillFailure, match="held"):
        s.call(name, **params)
    np.testing.assert_array_equal(s.ctx.data.qpos, before["data"]["qpos"])
    assert s.ctx.data.time == before["time"]


def test_effects_commit_only_after_success_and_home_keeps_empty_guard(monkeypatch):
    s = Session(make_context())
    s.held = "carriage"
    monkeypatch.setattr(s.arm, "open", lambda: False)
    with pytest.raises(SkillFailure):
        s.call("gripper", mode="open")
    assert s.held == "carriage" and s.grasp_epoch == 0
    with pytest.raises(SkillFailure, match="empty"):
        s.call("move", mode="home")
    monkeypatch.setattr(s.arm, "open", lambda: True)
    s.call("gripper", mode="open")
    assert s.held is None and s.grasp_epoch == 1


def test_custom_scene_bindings_and_unknown_are_not_rejected(tmp_path):
    s = make_demo_session("cube", tmp_path)
    steps = [
        dict(skill="observe"),
        dict(skill="estimate_pose", params=dict(part="cube")),
        dict(skill="propose_grasps", params=dict(part="cube")),
        dict(skill="select_grasp", params=dict(part="cube")),
        dict(skill="move", params=dict(mode="approach", part="cube")),
        dict(skill="gripper", params=dict(mode="close", part="cube")),
        dict(skill="move", params=dict(mode="lift", part="cube")),
    ]
    verdict = validate_skeleton(steps, parts=s.parts)
    assert verdict["valid"] and verdict["status"] == "unknown"
    assert verdict["necessary_checks_passed"] and verdict["unknown"]
    assert verdict["final_state"]["held"] == "cube"
    assert CATALOG["propose_grasps"].kind == "generator"
    assert CATALOG["verify_grasp"].kind == "checker"
    assert CATALOG["lift"].family == "move"
    assert all("supplies" in edge for edge in catalog_graph()["edges"])


def test_multiple_routes_and_grasps_survive_without_live_physics_mutation(tmp_path):
    s = make_demo_session("cube", tmp_path)
    prepare(s, "cube")
    before = s.snapshot()
    candidates = build_pick_candidates(
        s,
        "cube",
        control_options=[
            dict(strategy="feedback", force=3.0),
            dict(strategy="feedback", force=4.0),
        ],
    )
    assert len(candidates) == 12  # 2 grasps x 3 routes x 2 force settings
    assert len({c.grasp["id"] for c in candidates}) == 2
    assert len({c.bindings["route_id"] for c in candidates}) == 6
    assert len({c.control["force"] for c in candidates}) == 2
    assert sum(c.status == "necessary_pass" for c in candidates) >= 4
    for candidate in candidates:
        if candidate.path:
            assert candidate.path["binding"]["prefix_id"] == candidate.id
            assert candidate.path["binding"]["grasp_id"] == candidate.grasp["id"]
            np.testing.assert_allclose(
                candidate.path["target"], candidate.grasp["xyz"] + [0, 0, 0.10]
            )
    for key, value in before["physics"]["data"].items():
        np.testing.assert_array_equal(s.ctx.snapshot()["data"][key], value)
    assert s.ctx.data.time == before["physics"]["time"]


def test_failed_solver_keeps_unknown_grasps_and_routes(monkeypatch):
    s = Session(make_context())

    def exhausted(*args, **kwargs):
        raise ValueError("finite IK iteration budget exhausted")

    monkeypatch.setattr(s.arm, "ik", exhausted)
    prepare(s, "carriage")
    candidates = build_pick_candidates(s, "carriage")
    assert len(candidates) == 2 and all(c.status == "unknown" for c in candidates)
    with pytest.raises(ValueError, match="budget"):
        choose_candidate(candidates)
    routes, _ = transfer_routes(s, [-0.2, -0.2, 0.98])
    assert len(routes) == 3 and all(r["status"] == "unknown" for r in routes)


def test_cross_grasp_path_reuse_rejected_by_graph_and_executor():
    s = Session(make_context())
    prepare(s, "carriage")
    s.call("select_grasp", part="carriage", index=0)
    s.call("plan_transfer", target=s.artifacts["grasp"]["xyz"] + [0, 0, 0.1])
    s.call("select_grasp", part="carriage", index=1)
    steps = [dict(skill="move", params=dict(mode="joint_path"))]
    assert (
        validate_skeleton(steps, initial_artifacts=s.artifacts)["status"] == "conflict"
    )
    before = s.ctx.data.time
    with pytest.raises(SkillFailure, match="grasp binding"):
        execute_skeleton(s, steps)
    with pytest.raises(SkillFailure, match="grasp binding"):
        s.call("move", mode="joint_path")
    assert s.ctx.data.time == before


def test_reacquisition_invalidates_path_even_with_same_object_and_grasp():
    s = Session(make_context())
    s.call("plan_transfer", target=[-0.25, -0.2, 0.95])
    s.grasp_epoch += 2
    with pytest.raises(SkillFailure, match="epoch"):
        s.call("execute_joint_path")


def test_generated_unknown_identity_is_not_false_conflict():
    artifacts = {
        "grasps": dict(type="grasps", part="carriage"),
        "transfer": dict(
            type="joint_path",
            binding=dict(
                held=None, grasp_artifact="grasp", grasp_id="not-yet-resolved"
            ),
        ),
    }
    rows = [
        dict(skill="select_grasp", params=dict(part="carriage")),
        dict(skill="execute_joint_path"),
    ]
    result = validate_skeleton(rows, initial_artifacts=artifacts)
    assert result["valid"] and result["unknown"]


def test_live_execution_never_accepts_deferred_artifact():
    s = Session(make_context())
    s.artifacts["transfer"] = dict(type="joint_path", deferred=True)
    with pytest.raises(SkillFailure, match="materialize"):
        s.call("execute_joint_path")


def test_candidate_swap_and_stale_start_are_rejected(tmp_path):
    s = make_demo_session("cube", tmp_path)
    prepare(s, "cube")
    candidates = build_pick_candidates(s, "cube")
    good = choose_candidate(candidates)
    bad = copy.deepcopy(good)
    bad.path["binding"]["grasp_id"] = "different-grasp"
    with pytest.raises(SkillFailure, match="binding"):
        execute_pick_candidate(s, bad)
    s.ctx.data.qpos[s.ctx.arm_qadr[0]] += 0.001
    with pytest.raises(SkillFailure, match="stale"):
        execute_pick_candidate(s, good)


def test_snapshot_replays_observation_rng():
    s = Session(make_context(), seed=41, noise=0.001)
    state = s.snapshot()
    # Restore before both branches, so MuJoCo refreshes derived positions equally.
    s.restore(state)
    s.call("observe")
    first = copy.deepcopy(s.observations)
    s.restore(state)
    s.call("observe")
    for part in s.parts:
        np.testing.assert_array_equal(first[part], s.observations[part])


def test_output_contract_checked_in_execution_and_explicit_identity_propagated(
    monkeypatch,
):
    from simbench.assembly.library import Result

    s = Session(make_context())
    s.observations["carriage"] = [s.ctx.obj_pos("carriage")]

    def wrong_pose(part, as_="pose"):
        s.artifact(as_, "pose", "handle")
        return Result()

    monkeypatch.setattr(s, "estimate_pose", wrong_pose)
    with pytest.raises(ValueError, match="binding"):
        s.call("estimate_pose", part="carriage")
    initial = {
        "grasps": dict(type="grasps", part="carriage"),
        "transfer": dict(
            type="joint_path",
            binding=dict(held=None, grasp_artifact="grasp", grasp_id="A"),
        ),
    }
    rows = [
        dict(skill="select_grasp", params=dict(part="carriage", candidate_id="B")),
        dict(skill="execute_joint_path"),
    ]
    assert validate_skeleton(rows, initial_artifacts=initial)["status"] == "conflict"


def test_terminal_and_force_alternatives_keep_their_own_bindings(tmp_path):
    s = make_demo_session("cube", tmp_path)
    prepare(s, "cube")
    before = s.ctx.snapshot()
    candidates = build_pick_candidates(
        s,
        "cube",
        terminal_targets=[
            dict(id="left", part="cube", xyz=[-0.1, -0.1, 0.82]),
            dict(id="right", part="cube", xyz=[0.1, -0.1, 0.82]),
        ],
    )
    assert len(candidates) == 12
    assert {c.terminal["id"] for c in candidates} == {"left", "right"}
    for candidate in candidates:
        assert candidate.bindings["transport"]["target"] == candidate.terminal
        assert "terminal_release_check" in candidate.bindings
    for key, value in before["data"].items():
        np.testing.assert_array_equal(s.ctx.snapshot()["data"][key], value)
