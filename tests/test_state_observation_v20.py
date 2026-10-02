"""The state-observation ablation must be reproducible and label blind."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from scripts.collect_state_end_stop_v21 import short_plan, stage_audit
from simbench.value.planner_v12 import propose
from simbench.value.stage_v7 import refresh_visual_observation
from simbench.value.system_v12 import make_scene


ROOT = Path(__file__).resolve().parents[1]


def test_zero_noise_state_observation_and_frozen_short_plan():
    with TemporaryDirectory(prefix="state_observation_v20_", dir=ROOT) as directory:
        _, session, _, _ = make_scene(2050, Path(directory), level="L0",
            observation_backend="mujoco_state_pose")
        observation = session.decision_observation
        assert observation["backend"] == "mujoco_state_pose"
        assert observation["config"]["position_noise_std_m"] == 0
        for part, row in observation["objects"].items():
            np.testing.assert_allclose(row["position_m"], session.ctx.obj_pos(part), atol=1e-12)
        snapshot = session.snapshot()
        next_observation = refresh_visual_observation(session, parts=session.parts)
        session.restore(snapshot)
        replay = refresh_visual_observation(session, parts=session.parts)
        assert replay["config"] == next_observation["config"]
        assert "noise_state" not in replay
        assert "seed" not in replay["config"]
        assert session.state_observation_random_state["observation_index"] == replay["config"]["observation_index"]
        pool, _ = propose(observation, cad=session.planning_cad, n=2, seed=2050)
        plan = short_plan(session, pool[0])
        assert plan.prefix["task_scope"] == "printed_end_stop_mount_v21"
        assert {call.roles["manipulated"] for call in plan.calls} == {"end_stop"}
        assert plan.calls[-1].skill == "inspect"
        assert plan.calls[-1].arguments["what"].value == "stable_supported"


def test_stage_audit_uses_resolved_atomic_names():
    steps = [dict(skill=name, ok=True) for name in
             ("observe_parts", "close_gripper", "verify_grasp", "press_seat",
              "place_object", "retreat", "inspect_stable_support")]
    assert all(row["reached"] and row["completed"] for row in stage_audit(steps).values())
