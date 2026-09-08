"""Boundary tests for robot-only dynamics and artifact-gated atomic skills."""

import copy
import numpy as np
import pytest
from simbench.assembly.control import make_context, down
from simbench.assembly.library import Session, SkillFailure, PARTS
from simbench.assembly.graph import validate_skeleton


def test_only_panda_actuators_and_passive_assembly_parts():
    ctx = make_context()
    assert ctx.model.nu == 9
    assert {ctx.model.actuator(i).name for i in range(ctx.model.nu)} == {
        *[f"act_j{i}" for i in range(1, 8)],
        "act_finger1",
        "act_finger2",
    }
    actuated = set(ctx.model.actuator_trnid[:, 0])
    for part in PARTS:
        assert ctx.model.body_jntadr[ctx.body_id(part)] not in actuated


def test_perception_and_planning_leave_live_physics_unchanged():
    s = Session(make_context())
    before = s.ctx.snapshot()
    s.call("observe_parts")
    s.call("estimate_pose", part="carriage")
    s.call("propose_grasps", part="carriage")
    s.call("select_grasp", part="carriage")
    s.call("plan_transfer", target=s.artifacts["grasp"]["xyz"] + [0, 0, 0.1])
    after = s.ctx.snapshot()
    for key, value in before["data"].items():
        np.testing.assert_array_equal(after["data"][key], value)
    assert before["time"] == after["time"]
    grasps = s.artifacts["grasps"]["candidates"]
    assert len(grasps) == 2 and not np.allclose(
        grasps[0]["q_hover"], grasps[1]["q_hover"]
    )


def test_invalid_binding_is_rejected_before_any_motion():
    s = Session(make_context())
    before = s.ctx.get_state().copy()
    s.artifacts["grasp"] = dict(type="grasp", part="handle")
    with pytest.raises(SkillFailure, match="wrong part"):
        s.call("approach", part="carriage")
    with pytest.raises(SkillFailure, match="held"):
        s.call("lift", part="carriage")
    np.testing.assert_array_equal(before, s.ctx.get_state())


def test_skeleton_checks_artifacts_and_single_gripper_across_whole_chain():
    assert not validate_skeleton([])["valid"]
    assert not validate_skeleton(
        [dict(skill="slide_insert", params={"part": "carriage"})]
    )["valid"]
    chain = [
        dict(skill="observe_parts"),
        dict(skill="estimate_pose", params={"part": "carriage"}),
        dict(skill="propose_grasps", params={"part": "carriage"}),
        dict(skill="select_grasp", params={"part": "carriage"}),
        dict(skill="approach", params={"part": "carriage"}),
        dict(skill="close_gripper", params={"part": "carriage"}),
        dict(skill="lift", params={"part": "carriage"}),
    ]
    assert validate_skeleton(chain)["valid"]
    broken = copy.deepcopy(chain)
    broken[-1]["params"]["part"] = "handle"
    assert not validate_skeleton(broken)["valid"]
    assert not validate_skeleton(
        chain + [dict(skill="close_gripper", params={"part": "handle"})]
    )["valid"]


def test_discrete_planner_rejects_a_path_through_the_table_or_part():
    s = Session(make_context())
    before = s.ctx.snapshot()
    q = s.arm.ik([-0.25, -0.20, 0.797])
    verdict = s.arm.check_joint_path([q])
    assert not verdict["valid"] and verdict["penetration_m"] > 0.0008
    np.testing.assert_array_equal(before["data"]["qpos"], s.ctx.data.qpos)


def test_stale_joint_plan_is_not_executed():
    s = Session(make_context())
    s.call("plan_transfer", target=[-0.25, -0.2, 0.95])
    s.artifacts["transfer"]["start_q"][0] += 0.1
    before = s.ctx.snapshot()
    with pytest.raises(SkillFailure, match="stale"):
        s.call("execute_joint_path")
    np.testing.assert_array_equal(before["data"]["qpos"], s.ctx.data.qpos)
    assert s.ctx.data.time == before["time"]


def test_changed_geometry_is_not_silently_restored_as_same_world():
    s = Session(make_context())
    state = s.snapshot()
    s.ctx.model.geom_size[s.ctx.geom_id("rear_stop"), 0] += 0.001
    with pytest.raises(ValueError, match="geometry mismatch"):
        s.restore(state)


def test_cartesian_plan_rejects_an_intervening_wrist_rotation():
    s = Session(make_context())
    s.call("plan_linear", target=s.ctx.eef_pos() + [0, 0, -0.01])
    assert s.arm.move(s.ctx.eef_pos(), down(0.15))
    before = s.ctx.data.time
    with pytest.raises(SkillFailure, match="stale Cartesian orientation"):
        s.call("execute_cartesian_path")
    assert s.ctx.data.time == before
