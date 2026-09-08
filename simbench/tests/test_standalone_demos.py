"""Scene portability and read-only planning visualization regression checks."""

import numpy as np
import pytest
from simbench.assembly.demo_scenes import make_demo_session
from simbench.assembly.demo_visuals import joint_path_points


@pytest.mark.parametrize(
    "kind,count",
    [("perception", 3), ("cube", 1), ("obstacle", 1), ("pin", 1), ("rail", 1)],
)
def test_isolated_inventory_uses_same_skills_without_assembly_dependencies(
    tmp_path, kind, count
):
    s = make_demo_session(kind, tmp_path)
    assert s.ctx.model.nu == 9
    assert s.ctx.model.neq == 0
    assert len(s.parts) == count
    before = s.ctx.snapshot()
    result = s.call("observe_parts")
    assert set(result.metrics["detected"]) == set(s.parts)
    for part in s.parts:
        s.call("estimate_pose", part=part)
        s.call("propose_grasps", part=part)
        assert len(s.artifacts["grasps"]["candidates"]) == 2
    after = s.ctx.snapshot()
    for key in before["data"]:
        np.testing.assert_array_equal(before["data"][key], after["data"][key])


def test_plotted_joint_path_has_real_fk_endpoints_and_does_not_move_robot(tmp_path):
    s = make_demo_session("obstacle", tmp_path)
    s.arm.move([-0.18, -0.18, 0.94], linear=False)
    s.call("plan_transfer", target=[0.14, 0.09, 0.96], clearance=1.08)
    before = s.ctx.snapshot()
    plan = s.artifacts["transfer"]
    points = joint_path_points(s, plan)
    np.testing.assert_allclose(points[0], s.ctx.eef_pos(), atol=1e-9)
    np.testing.assert_allclose(points[-1], plan["target"], atol=0.0002)
    # The computed route must visibly clear the 0.94 m obstacle top.
    assert np.max(points[:, 2]) > 1.07
    after = s.ctx.snapshot()
    for key in before["data"]:
        np.testing.assert_array_equal(before["data"][key], after["data"][key])
