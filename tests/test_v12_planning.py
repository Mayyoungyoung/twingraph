from copy import deepcopy
import numpy as np
import pytest
from simbench.value.planner_v12 import propose, PARTS
from simbench.value import planner_v12


def observation():
    return dict(sha256="test", objects={p:dict(valid=True, position_m=[-.2+i*.04, -.15, .85],
        quat_wxyz=[1,0,0,0], fit_residual_m=.002, quality=.9)
        for i,p in enumerate((*PARTS,"wipe_tool"))},
        fixtures={"guide_base":dict(valid=True,position_m=[.0,.08,.812],quat_wxyz=[1,0,0,0])},
        assembly_targets={p:dict(position_m=[.1,.08,.84],quat_wxyz=[1,0,0,0]) for p in PARTS})


@pytest.fixture
def scripted_catalogs(monkeypatch):
    """Unit-test plan combinations; actual CAD queries have separate tests.

    All numbers here are disclosed software fixtures, not physical outcomes.
    The fixture responds to object yaw and retains independent pin goal yaw.
    """
    def catalogs(obs, cad, completed):
        out = {}
        for part in PARTS:
            if part in completed:
                continue
            q = obs["objects"][part]["quat_wxyz"]
            source_yaw = np.arctan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]**2+q[3]**2))
            rows = []
            for index in range(4):
                pin = part.startswith("pin_")
                angle = index*np.pi/2 if pin else (index % 2)*np.pi/2
                row = dict(part=part, yaw=float(source_yaw+angle), height=.004*(index//2),
                    source_axis_offset_rad=float(angle),
                    placement_yaw=float(np.pi/2 if pin else angle),
                    grasp_width_m=.018 if pin else .028, status="necessary_pass", known=True,
                    min_clearance_m=.010, required_clearance_m=.003,
                    source_grasp_ik_checked=False, software_fixture=True)
                if pin:
                    row.update(pin_command_depth_m=.043, pin_press_extra_m=.001)
                rows.append(row)
            out[part] = rows
        return out, deepcopy(out)
    monkeypatch.setattr(planner_v12, "_geometry_catalogs", catalogs)
    return catalogs


def test_pool_budget_unique_and_prefix_diversity(scripted_catalogs):
    small, _ = propose(observation(), n=12, seed=123)
    large, meta = propose(observation(), n=48, seed=123)
    assert {p["name"] for p in small} <= {p["name"] for p in large}
    assert len(large) == meta["unique_executable_parameters"] == 48
    assert len({(p["choices"]["carriage"]["yaw"], p["choices"]["carriage"].get("approach_strategy")) for p in large}) == 4
    assert len({tuple(p["order"]) for p in large}) == 2
    assert len({p["choices"]["pin_left"]["yaw"] for p in large}) == 4
    assert len({p["choices"]["pin_right"]["yaw"] for p in large}) == 4
    for proposal in large:
        for part, record in proposal["necessary_geometry"].items():
            for key in ("yaw", "height", "placement_yaw"):
                assert record[key] == proposal["choices"][part][key]
            assert proposal["choices"][part]["width"] == record["grasp_width_m"]
        for part in ("pin_left", "pin_right"):
            assert proposal["choices"][part]["pin_command_depth_m"] == .043
            assert proposal["choices"][part]["pin_press_extra_m"] == .001
    for p in large:
        assert p["order"][:2] == ["carriage", "end_stop"]
        assert p["order"][-1] == "handle"
    with pytest.raises(ValueError): propose(observation(), n=0)


def test_parameters_respond_to_observation_and_explicit_material_prior(scripted_catalogs):
    obs = observation()
    _, a = propose(obs)
    obs["objects"]["handle"]["position_m"][2] += .025
    obs["objects"]["pin_left"]["fit_residual_m"] = .006
    _, b = propose(obs)
    assert b["derivation"]["carriage"]["baseline"]["clearance"] > a["derivation"]["carriage"]["baseline"]["clearance"]
    assert b["derivation"]["pin_left"]["baseline"]["speed"] < a["derivation"]["pin_left"]["baseline"]["speed"]
    _, c = propose(observation(), priors=dict(friction_lower_bound=.2))
    assert c["derivation"]["carriage"]["force_balance_n"] > a["derivation"]["carriage"]["force_balance_n"]


def test_missing_pending_state_is_not_replaced_by_simulator_truth(scripted_catalogs):
    obs = observation()
    obs["objects"]["pin_left"].update(valid=False,position_m=None)
    with pytest.raises(ValueError,match="reobserve required"):
        propose(obs)
    plans,_=propose(obs,completed=["cleaning","carriage","end_stop","pin_left"])
    assert plans
    bad=deepcopy(obs)
    bad["objects"]["handle"]["position_m"][0]=np.nan
    with pytest.raises(ValueError,match="nonfinite"):
        propose(bad,completed=["pin_left"])


def test_assembly_orientation_uses_visual_object_yaw_and_independent_pin_goal_yaw(scripted_catalogs):
    obs=observation(); yaw=.2
    obs["objects"]["carriage"]["quat_wxyz"]=[np.cos(yaw/2),0,0,np.sin(yaw/2)]
    pool,_=propose(obs,n=8)
    for proposal in pool:
        choice=proposal["choices"]["carriage"]
        assert choice["grasp_yaw_frame"] == "object"
        np.testing.assert_allclose(np.sin(choice["placement_yaw"]-choice["yaw"]), 0., atol=1.e-12)
        record = proposal["necessary_geometry"]["carriage"]
        assert record["evaluated_world_yaw_rad"] == pytest.approx(choice["yaw"] + yaw)
        pin=proposal["choices"]["pin_left"]
        assert pin["placement_yaw"] == pytest.approx(np.pi/2)
    assert any(p["choices"]["pin_left"]["yaw"] != p["choices"]["pin_left"]["placement_yaw"] for p in pool)


def test_contact_proposals_cover_slow_and_firm_branches_without_expanding_limits(scripted_catalogs):
    pool, meta = propose(observation(), n=48)
    choices = [p["choices"]["carriage"] for p in pool]
    assert min(c["speed"] for c in choices) < .011
    assert max(c["force_limit"] for c in choices) > 14.
    assert max(c["press_force"] for c in choices) > 3.
    for p in pool:
        for part, c in p["choices"].items():
            assert 1. <= c["press_force"] <= 4.
            if "force_limit" in c:
                assert 4. <= c["force_limit"] <= 18.
                if p["name"] != "grounded_000":
                    minimum = min(9., 1.2*c["force_limit"]/(2*.8))
                    assert c["force"] >= minimum - 1e-12
    nominal = next(p for p in pool if p["name"] == "grounded_000")
    assert nominal["choices"]["carriage"]["force_limit"] == meta["derivation"]["carriage"]["baseline"]["force_limit"]


def test_real_catalog_constructor_requires_cad_and_observed_receiver():
    obs=observation()
    with pytest.raises(ValueError,match="declared printed-kit CAD"):
        propose(obs)
    cad=dict(parts={},collision_primitives={},pin_shaft_offsets_m=[-.047,.006])
    obs["fixtures"]["guide_base"]["valid"]=False
    with pytest.raises(ValueError,match="reobserve required: guide_base"):
        propose(obs,cad=cad)
    obs["fixtures"]["guide_base"]["valid"]=True
    del obs["assembly_targets"]["handle"]
    with pytest.raises(ValueError,match="assembly targets required"):
        propose(obs,cad=cad)
