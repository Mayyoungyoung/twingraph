from copy import deepcopy
import numpy as np
import pytest
from simbench.value import stage_v9
from simbench.value.planner_v11 import propose, normalized_graph, value_features
from simbench.value.system_v11 import discrepancy, twin_checkpoint, restore_twin_checkpoint
from scripts.analyze_v9_candidate_matrix import features


@pytest.fixture(scope="module")
def decision(tmp_path_factory):
    return stage_v9.make_scene(1501,tmp_path_factory.mktemp("v11_scene"))[1]


def test_model_receives_exact_executable_port_values(decision):
    pool,_=propose(decision.decision_observation)
    for p in pool:
        g=normalized_graph(decision,p)
        assert len(g["assembly"]["nodes"]) > 90
        assert len(g["assembly"]["edges"]) > 300
        np.testing.assert_array_equal(value_features(g),features(decision.decision_observation,p))


def test_graph_rejects_stale_port_or_cleaning_binding(decision):
    p=propose(decision.decision_observation)[0][0]
    g=normalized_graph(decision,p)
    modified=deepcopy(g)
    modified["assembly"]["nodes"][0]["skill"]="grasp"
    with pytest.raises(ValueError,match="mismatch"): value_features(modified)
    modified=deepcopy(g); modified["cleaning"]["wipe_duration"]+=2
    with pytest.raises(ValueError,match="disagree"): value_features(modified)


def test_observation_dependency_excludes_occluded_completed_parts(decision):
    obs=deepcopy(decision.decision_observation)
    obs["objects"]["pin_left"].update(valid=False,position_m=None)
    pool,_=propose(obs,completed=["cleaning","carriage","end_stop","pin_left"])
    graph=normalized_graph(decision,pool[0])
    calls=graph["assembly"]["plan"]["calls"]
    detections=[c for c in calls if c["skill"]=="detect"]
    assert all(c["arguments"]["required_parts"]["value"]==[c["roles"]["manipulated"]] for c in detections)
    from simbench.assembly.contracts import State,apply_effects,check
    from simbench.assembly.library import HANDLERS
    state=State(parts=decision.parts)
    apply_effects(HL:=HANDLERS["observe_parts"],{"required_parts":["pin_right"]},state)
    assert state.seen=={"pin_right"}
    assert check(HL,{"required_parts":["unknown_part"]},state)["conflicts"]


def test_state_grounding_and_precedence(decision):
    obs=deepcopy(decision.decision_observation)
    a,meta=propose(obs)
    obs["objects"]["handle"]["position_m"][2]=max(r["position_m"][2] for r in obs["objects"].values())+.02
    obs["objects"]["pin_left"]["fit_residual_m"]*=.25
    obs["objects"]["pin_right"]["fit_residual_m"]*=.25
    obs["objects"]["end_stop"]["fit_residual_m"]*=.25
    b,changed=propose(obs)
    assert meta["clearance_m"] != changed["clearance_m"]
    assert meta["insertion_speed_m_s"] != changed["insertion_speed_m_s"]
    assert a[0]["choices"] != b[0]["choices"]
    broken=deepcopy(a[0]); broken["order"][0:2]=reversed(broken["order"][0:2])
    with pytest.raises(ValueError): normalized_graph(decision,broken)
    obs["objects"]["handle"]["valid"]=False
    with pytest.raises(ValueError,match="reobserve"): propose(obs)


def test_monitor_residual_missing_data_and_completed_state():
    expected=dict(observation=dict(objects={"handle":dict(valid=True,position_m=[0,0,1])}),
                  robot_joints=[0]*7,held=None,completed=["cleaning"])
    actual=deepcopy(expected)
    assert not discrepancy(expected,actual)["replan"]
    actual["observation"]["objects"]["handle"]["position_m"][0]=.01
    assert discrepancy(expected,actual)["reason"]=="prediction_error"
    actual=deepcopy(expected); actual["observation"]["objects"]["handle"]["valid"]=False
    assert discrepancy(expected,actual)["reason"]=="pending_observation_unknown"
    actual=deepcopy(expected); actual["held"]="handle"
    assert discrepancy(expected,actual)["reason"]=="state_changed"


def test_independent_checkpoint_transfer_checks_geometry(decision,tmp_path):
    clone=stage_v9.make_scene(1501,tmp_path / "clone")[1]
    state=twin_checkpoint(decision)
    restore_twin_checkpoint(clone,state)
    np.testing.assert_array_equal(clone.ctx.data.qpos,decision.ctx.data.qpos)
    assert clone.ctx.model is not decision.ctx.model
    assert not np.shares_memory(clone.ctx.data.qpos,decision.ctx.data.qpos)
    clone.ctx.model.body_pos[1,0]+=.01
    with pytest.raises(ValueError,match="geometry"):
        restore_twin_checkpoint(clone,state)
