"""The new surface skill is bound to a real held tool and measured contact."""

import numpy as np
import pytest
from simbench.assembly.full_demos import make_session, prepare_wipe
from simbench.assembly.graph import validate_skeleton
from simbench.assembly.interfaces import PUBLIC_SKILLS
from simbench.assembly.library import SkillFailure
from simbench.assembly.scene import CENTER
from simbench.assembly.wiping import trajectory, teacher


def test_wipe_contract_rejects_non_tools_and_keeps_unresolved_contact_unknown():
    steps = [
        dict(
            skill="plan_path", params=dict(method="surface", part="cube", center=[0, 0])
        ),
        dict(skill="wipe", params=dict(part="cube")),
    ]
    rejected = validate_skeleton(steps, initial_held="cube", parts=("cube",))
    assert rejected["status"] == "conflict"
    bound = validate_skeleton(
        steps, initial_held="cube", parts=("cube",), capabilities={"cube": ("wipe",)}
    )
    assert bound["necessary_checks_passed"] and bound["status"] == "unknown"
    assert "measure" not in PUBLIC_SKILLS and "inspect" not in PUBLIC_SKILLS


def test_learned_path_generalizes_to_unsampled_phase_grid():
    phases = np.linspace(0, 1, 733)
    assert np.max(np.linalg.norm(trajectory(733) - teacher(phases), axis=1)) < 0.006


def test_wiping_requires_live_grasp_and_measures_actual_coverage(tmp_path):
    s = make_session(tmp_path, isolated=True)
    before = s.ctx.snapshot()
    with pytest.raises(SkillFailure, match="held"):
        s.call("wipe", part="wipe_tool")
    np.testing.assert_array_equal(before["data"]["qpos"], s.ctx.data.qpos)
    prepare_wipe(s)
    before = s.ctx.snapshot()
    s.call("plan_path", method="surface", part="wipe_tool", center=CENTER.tolist())
    np.testing.assert_array_equal(before["data"]["qpos"], s.ctx.data.qpos)
    assert s.ctx.data.time == before["time"]
    assert s.artifacts["wipe"]["binding"]["held"] == "wipe_tool"
    epoch = s.grasp_epoch
    s.grasp_epoch += 1
    with pytest.raises(SkillFailure, match="grasp acquisition epoch"):
        s.call("wipe", part="wipe_tool")
    assert s.ctx.data.time == before["time"]
    s.grasp_epoch = epoch
    result = s.call("wipe", part="wipe_tool")
    assert result.metrics["coverage"] >= 0.9
    assert result.metrics["contact_fraction"] > 0.8
    assert 0 < result.metrics["peak_force_n"] < 5
    assert s.held == "wipe_tool" and s.ctx.model.nu == 9
