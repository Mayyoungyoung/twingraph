"""Preserve counterfactual replay guarantees after retiring the old task stack."""

import copy
import numpy as np
from simbench.assembly.control import make_context
from simbench.assembly.library import Session


def test_complete_snapshot_replays_same_controls_and_restores_model():
    s = Session(make_context())
    s.artifacts["binding"] = {"type": "test", "part": "carriage", "values": [1]}
    state = s.snapshot()

    def rollout():
        s.restore(state)
        rows = []
        for value in np.linspace(0.04, 0.025, 20):
            s.ctx.set_finger_ctrl(float(value))
            s.ctx.step()
            rows.append(np.r_[s.ctx.data.qpos, s.ctx.data.qvel, s.ctx.data.ctrl])
        return np.asarray(rows)

    first = rollout()
    s.artifacts["binding"]["values"].append(99)
    s.ctx.model.geom_friction[:] *= 1.3
    second = rollout()
    np.testing.assert_allclose(first, second, atol=1e-10, rtol=0)
    assert s.artifacts["binding"]["values"] == [1]
    np.testing.assert_array_equal(
        s.ctx.model.geom_friction, state["physics"]["model"]["geom_friction"]
    )


def test_snapshot_restores_clock_and_registered_client_without_aliasing():
    ctx = make_context()
    client = {"retry_budget": [1]}

    def restore(value):
        client.clear()
        client.update(copy.deepcopy(value))

    ctx.register_state_client("recovery", lambda: client, restore)
    state = ctx.snapshot()
    ctx.step()
    client["retry_budget"].append(2)
    ctx.restore(state)
    assert client == {"retry_budget": [1]}
    assert (ctx.n_control_steps, ctx.n_physics_steps, ctx._hook_substeps) == state[
        "clock"
    ]
    client["retry_budget"].append(3)
    assert state["clients"]["recovery"] == {"retry_budget": [1]}
