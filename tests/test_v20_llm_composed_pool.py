"""Grounded model compositions remain complete and bound to one observation."""
from copy import deepcopy

import pytest

from scripts.prepare_v20_llm_composed_pool import compose, prompt_input, response_schema


def test_model_composition_binds_arbitrary_roles_and_rejects_stale_observation():
    roles = ("part_a", "part_b")
    base = dict(initial_observation=dict(sha256="observed", objects={}), pool=[])
    for i in range(3):
        base["pool"].append(dict(name=f"base_{i}", order=list(roles),
            choices={role:dict(yaw=float(i), force=float(i + 2)) for role in roles},
            necessary_geometry={role:dict(status="necessary_pass") for role in roles},
            wipe_variant=i, wipe_force=1. + i, wipe_duration=10. + i,
            stroke_minimum=.02))
    request = prompt_input(base, 2)
    assert request["baseline_donor_index"] == 0
    assert len(request["generic_atomic_skill_graph"]["nodes"]) == 10
    assert request["generic_atomic_skill_graph"]["edges"]
    schema = response_schema(roles)
    assert schema["properties"]["plans"]["items"]["properties"]["role_donors"]["required"] == list(roles)
    answer = dict(observation_sha256=request["observation_sha256"],
        base_pool_sha256=request["base_pool_sha256"], plans=[
            dict(name="llm_000", order_donor=0, cleaning_donor=1,
                 role_donors=dict(part_a=1, part_b=2)),
            dict(name="llm_001", order_donor=1, cleaning_donor=0,
                 role_donors=dict(part_a=2, part_b=1)),
        ])
    candidates = compose(base, answer, 2)
    assert candidates[0]["choices"]["part_a"]["yaw"] == 1.
    assert candidates[0]["choices"]["part_b"]["yaw"] == 2.
    assert candidates[0]["wipe_force"] == 2.
    stale = deepcopy(answer)
    stale["observation_sha256"] = "different_scene"
    with pytest.raises(ValueError, match="different observation"):
        compose(base, stale, 2)
