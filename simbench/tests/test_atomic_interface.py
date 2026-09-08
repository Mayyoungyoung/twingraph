"""Public atom semantics: no direction clones, measurement != acceptance, supported placement."""

import numpy as np
import pytest
from simbench.assembly.interfaces import PUBLIC_SKILLS, resolve
from simbench.assembly.graph import catalog_graph, validate_skeleton, execute_skeleton
from simbench.assembly.library import Session, SkillFailure, Result, CATALOG
from simbench.assembly.control import make_context
from simbench.assembly.demo_scenes import make_demo_session
from simbench.assembly.task import pick


def test_graph_has_eleven_distinct_atoms_and_no_direction_or_solver_nodes():
    graph = catalog_graph()
    expected = {
        "detect",
        "estimate_pose",
        "estimate_grasp",
        "plan_path",
        "move",
        "grasp",
        "place",
        "insert",
        "press",
        "measure",
        "inspect",
    }
    assert {n["name"] for n in graph["nodes"]} == expected == set(PUBLIC_SKILLS)
    assert graph["atom_count"] == 11
    assert (
        len(CATALOG) == 30
    )  # old demo entry points remain available, not public nodes
    moves = next(n for n in graph["nodes"] if n["name"] == "move")
    assert {"lift", "lower", "home", "retreat", "move_constrained"} <= {
        c["name"] for c in moves["contracts"]
    }
    assert all(
        e["source"] in expected and e["target"] in expected for e in graph["edges"]
    )


@pytest.mark.parametrize(
    "params,handler",
    [
        ({"part": "cube", "delta": [0, 0, 0.10]}, "lift"),
        ({"part": "cube", "delta": [0, 0, -0.03]}, "lower"),
        ({"delta": [0, 0, 0.10]}, "retreat"),
        ({"target": "home"}, "home"),
        ({"path": "transfer"}, "execute_joint_path"),
        ({"path": "linear", "space": "cartesian"}, "execute_cartesian_path"),
    ],
)
def test_move_parameters_dispatch_to_original_guarded_handlers(params, handler):
    assert resolve("move", params)[0] == handler


def test_move_checks_conflicting_targets_and_held_object_before_motion():
    s = Session(make_context())
    before = s.ctx.data.time
    with pytest.raises(SkillFailure, match="exactly one"):
        s.call("move", delta=[0, 0, 0.1], target=[0, 0, 1.0])
    s.held = "carriage"
    with pytest.raises(SkillFailure, match="ownership"):
        s.call("move", delta=[0.01, 0, 0])
    assert s.ctx.data.time == before


def test_general_move_uses_existing_controller_and_checks_path():
    s = Session(make_context())
    target = s.ctx.eef_pos() + [0.015, 0, 0]
    result = s.call("move", target=target)
    assert result.ok and result.metrics["collision"]["valid"]
    np.testing.assert_allclose(s.ctx.eef_pos(), target, atol=0.001)
    assert s.results[-1]["atom"] == "move"


def test_measurement_outputs_data_even_when_inspection_fails(monkeypatch):
    s = Session(make_context())
    monkeypatch.setattr(
        s, "measure_clearance", lambda part: Result(False, {"lateral_margin_m": -0.001})
    )
    before = s.ctx.snapshot()
    result = s.call("measure", quantity="clearance", part="carriage")
    assert result.ok and s.artifacts["measurement"]["value"] == -0.001
    assert s.artifacts["measurement"]["unit"] == "m"
    with pytest.raises(SkillFailure, match="bounds"):
        s.call("inspect", what="measurement", minimum=0.0)
    np.testing.assert_array_equal(s.ctx.data.qpos, before["data"]["qpos"])
    assert s.ctx.data.time == before["time"]


def test_supported_place_releases_cube_and_graph_uses_its_success_effect(tmp_path):
    s = make_demo_session("cube", tmp_path)
    pick(s, "cube", lift=False)
    target = s.ctx.obj_pos("cube").copy()
    steps = [
        dict(skill="place", params=dict(part="cube", target=target)),
        dict(skill="move", params=dict(delta=[0, 0, 0.05])),
    ]
    verdict = validate_skeleton(steps, initial_held="cube", parts=s.parts)
    assert verdict["valid"] and verdict["final_state"]["held"] is None
    epoch = s.grasp_epoch
    execute_skeleton(s, steps)
    assert s.held is None and s.grasp_epoch == epoch + 1
    np.testing.assert_allclose(s.ctx.obj_pos("cube"), target, atol=0.0015)
    assert any(row["atom"] == "place" and row["ok"] for row in s.results)


def test_place_never_opens_gripper_without_support(tmp_path):
    s = make_demo_session("cube", tmp_path)
    pick(s, "cube", lift=True)
    before = s.ctx.data.time
    with pytest.raises(SkillFailure, match="supporting contact"):
        s.call("place", part="cube", target=s.ctx.obj_pos("cube"))
    assert s.held == "cube" and s.ctx.data.time == before


def test_place_failure_after_release_does_not_restore_stale_held_state(
    tmp_path, monkeypatch
):
    s = make_demo_session("cube", tmp_path)
    pick(s, "cube", lift=False)
    actual = s.inspect_seat
    calls = []

    def inspect(part, target, tol):
        calls.append(1)
        return (
            actual(part, target, tol)
            if len(calls) == 1
            else Result(False, reason="test disturbed support")
        )

    monkeypatch.setattr(s, "inspect_seat", inspect)
    with pytest.raises(SkillFailure, match="tolerance"):
        s.call("place", part="cube", target=s.ctx.obj_pos("cube"))
    assert s.held is None
