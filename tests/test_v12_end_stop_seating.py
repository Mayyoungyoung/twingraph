"""Pure sensor/command contract tests; no claimed physical assembly success."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from simbench.assembly.skills_v12 import EndStopSeatConfig,bounded_end_stop_seat
from simbench.assembly.end_stop_seating_v12 import seat_contract,search_offsets
from simbench.assembly.printed_kit import planning_metadata


def session(monkeypatch,contact_height=.033):
    from simbench.assembly import sensor_learning_v12
    monkeypatch.setattr(sensor_learning_v12,"gripper_fixture_contacts",
        lambda *a,**k:dict(normal_force_n=0.,contacts=[]))
    def forbidden(*args):raise AssertionError("object/evaluator truth must not drive seating")
    state=dict(position=np.array([-.092,0.,.033]),calls=[])
    config=replace(EndStopSeatConfig(),lift_speed_m_s=.03,descent_speed_m_s=.01)
    cad=planning_metadata();cad["end_stop_seat_search"]=config.manifest()
    row=dict(valid=True,position_m=[0,0,0],quat_wxyz=[1,0,0,0],fit_residual_m=.0005)
    ctx=SimpleNamespace(eef_pos=lambda:state["position"].copy(),eef_mat=lambda:np.eye(3),
        grasp_contacts=lambda p:dict(held=True,left_n=2.,right_n=2.),control_dt=.02,
        obj_pose=forbidden,obj_pos=forbidden,obj_axis=forbidden)
    def servo(command,**kwargs):
        state["position"]=np.asarray(command).copy();state["calls"].append(state["position"].copy())
    s=SimpleNamespace(held="end_stop",strict_rgbd_v12=True,ctx=ctx,planning_cad=cad,
        decision_observation=dict(backend="rgbd_geometry",fixtures={"guide_base":dict(row)},objects={"end_stop":dict(row)}),
        held_visual_transforms_v12={"end_stop":dict(local_position=np.zeros(3),local_rotation=np.eye(3),
            source="rgbd_at_grasp_plus_encoder_forward_kinematics")},
        arm=SimpleNamespace(rotation=np.eye(3),servo=servo),artifacts={},end_stop_seat_config_v12=config,
        external_force=lambda part:1.2 if state["position"][2]<=contact_height else 0.)
    return s,state


def test_contract_uses_original_cad_posts_observed_base_and_symmetric_uncertainty_search(monkeypatch):
    s,_=session(monkeypatch)
    q=Rotation.from_euler("z",.6).as_quat();base=s.decision_observation["fixtures"]["guide_base"]
    base.update(position_m=[.12,.34,.8],quat_wxyz=q[[3,0,1,2]].tolist())
    c=seat_contract(s,.824,s.end_stop_seat_config_v12)
    assert c["locator_top_height"]==pytest.approx(.017)
    assert c["support_surface_height"]==pytest.approx(.006)
    assert c["seating_band"]==pytest.approx(.00275)
    np.testing.assert_allclose(c["target"],[.12,.34,.8]+Rotation.from_euler("z",.6).as_matrix()@np.array([-.092,0,.024]))
    offsets=np.array(list(search_offsets(c["search_radius"])))
    assert len(offsets)==25 and np.linalg.norm(offsets,axis=1).max()<=.003+1.e-12
    np.testing.assert_allclose(np.unique(np.round(np.linalg.norm(offsets,axis=1),9)),
        [0,.001,.002,.003],atol=1.e-12)
    np.testing.assert_allclose(offsets.sum(axis=0),[0,0],atol=1.e-12)
    assert c["safe_center_height"]+c["bottom_body_z_m"]>c["locator_top_height"]
    assert c["safe_center_height"]<c["target_height"]+c["body_bounding_radius"]


def test_post_top_contact_requires_lift_and_exhausts_instead_of_becoming_capture(monkeypatch):
    s,state=session(monkeypatch)
    result=bounded_end_stop_seat(s,"end_stop",.024,1.)
    assert not result.ok and len(result.metrics["attempts"])==25
    assert all(r["result"]=="contact_above_seating_band" for r in result.metrics["attempts"])
    assert state["calls"][0][2]>.033
    # Each lateral command occurs only after the registered whole part has
    # cleared all locator tops; there is no scrape/search at blocked height.
    high=result.metrics["geometry"]["safe_center_height"]
    previous=np.array([-.092,0,.033])
    for command in state["calls"]:
        if np.linalg.norm(command[:2]-previous[:2])>1.e-9:
            assert previous[2]>=high-s.end_stop_seat_config_v12.control_tracking_tolerance_m
        previous=command
    assert not result.metrics["actual_capture_or_bridge_verified"]


def test_sensor_depth_and_contact_can_complete_action_but_never_claim_physical_bridge(monkeypatch):
    s,state=session(monkeypatch,contact_height=.025)
    result=bounded_end_stop_seat(s,"end_stop",.024,1.)
    assert result.ok and len(result.metrics["attempts"])==1
    assert not result.metrics["actual_capture_or_bridge_verified"]
    assert not result.metrics["axial_slip_observed"]
    assert state["position"][2]>=.024-1.e-10


@pytest.mark.parametrize("fault",["unknown_base","stale_target","graph_mismatch","lost_grasp","nan_force","budget"])
def test_invalid_or_unobservable_seating_state_fails_bounded(monkeypatch,fault):
    s,state=session(monkeypatch);target=.024
    if fault=="unknown_base":s.decision_observation["fixtures"]["guide_base"]["valid"]=False
    elif fault=="stale_target":target=.8
    elif fault=="graph_mismatch":s.planning_cad["end_stop_seat_search"]={}
    elif fault=="lost_grasp":s.ctx.grasp_contacts=lambda p:dict(held=False,left_n=0.,right_n=0.)
    elif fault=="nan_force":s.external_force=lambda p:np.nan
    elif fault=="budget":
        s.end_stop_seat_config_v12=replace(s.end_stop_seat_config_v12,maximum_control_steps=1)
        s.planning_cad["end_stop_seat_search"]=s.end_stop_seat_config_v12.manifest()
    result=bounded_end_stop_seat(s,"end_stop",target,1.)
    assert not result.ok
    assert len(state["calls"])==(1 if fault=="budget" else 0)
