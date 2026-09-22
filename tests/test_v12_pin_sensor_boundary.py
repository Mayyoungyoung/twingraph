"""No physics: fail-closed sensor contracts and observability counterexamples."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from simbench.assembly import sensor_learning_v12 as controller
from simbench.assembly.skills_v12 import control_position, bind_grasp_observation


def session():
    def forbidden(*args):
        raise AssertionError("controller read object truth")
    calls=[]
    ctx=SimpleNamespace(eef_pos=lambda:np.array([0.,0.,.05]),eef_mat=lambda:np.eye(3),
        control_dt=.02,obj_pos=forbidden,obj_pose=forbidden,obj_axis=forbidden,
        grasp_contacts=lambda part:dict(held=True,left_n=1.,right_n=1.))
    row=dict(valid=True,position_m=[0.,0.,.05],quat_wxyz=[1.,0.,0.,0.],fit_residual_m=.0005)
    s=SimpleNamespace(ctx=ctx,strict_rgbd_v12=True,held="pin_left",
        held_visual_transforms_v12={"pin_left":dict(local_position=np.zeros(3),local_rotation=np.eye(3),
            source="rgbd_at_grasp_plus_encoder_forward_kinematics")},
        decision_observation=dict(backend="rgbd_geometry",objects={"pin_left":row},sha256="source-image"),
        pin_insertion_config=SimpleNamespace(shaft_tip_offset_m=-.047,plate_hole_half_width_m=.004),
        planning_cad=dict(pin_head_seated_total_depth_m=.046),external_force=lambda part:0.,
        arm=SimpleNamespace(rotation=np.eye(3),servo=lambda *args,**kwargs:calls.append(args)),
        artifacts=dict(insert=dict(axis=[0,0,-1],hole_entry_m=[0,0,0],command_depth_m=.043,
            press_extra_m=0.,maximum_total_depth_m=.043,absolute_depth_limit_m=.046,
            speed=.004,force_limit=8.,target=[0,0,.004])))
    s.calls=calls
    return s


def setup_feedback(monkeypatch):
    monkeypatch.setattr(controller,"gripper_fixture_contacts",
        lambda s:dict(normal_force_n=0.,contacts=[],source="unit sensor stub"))
    monkeypatch.setattr(controller,"load_actor",lambda path:lambda obs:np.array([0.,0.,-1.]))


def test_yaw_change_transforms_registered_offset_and_axis_without_object_truth():
    s=session();source_R=Rotation.from_euler("xyz",[.1,.02,.4]).as_matrix()
    initial_R=Rotation.from_euler("z",.3).as_matrix();current_R=Rotation.from_euler("z",.3+np.pi/2).as_matrix()
    source=np.array([.003,-.002,.054]);s.decision_observation["objects"]["pin_left"].update(
        position_m=source.tolist(),quat_wxyz=Rotation.from_matrix(source_R).as_quat()[[3,0,1,2]].tolist())
    s.ctx.eef_mat=lambda:initial_R;bind_grasp_observation(s,"pin_left")
    s.ctx.eef_mat=lambda:current_R
    np.testing.assert_allclose(control_position(s,"pin_left"),s.ctx.eef_pos()+current_R@initial_R.T@(source-s.ctx.eef_pos()))
    np.testing.assert_allclose(controller.sensor_pin_axis(s,"pin_left"),current_R@initial_R.T@source_R[:,2])
    reg=s.held_visual_transforms_v12["pin_left"]
    assert reg["observation_sha256"]=="source-image"
    assert not reg["axial_slip_observable_from_bilateral_contact_and_fk"]


@pytest.mark.parametrize("fault",["missing","invalidated","wrong_source","nan_offset","nonrotation","nan_fk"])
def test_held_registration_unknown_never_reuses_old_free_pose(fault):
    s=session();reg=s.held_visual_transforms_v12["pin_left"]
    if fault=="missing":s.held_visual_transforms_v12.clear()
    elif fault=="invalidated":reg["valid"]=False
    elif fault=="wrong_source":reg["source"]="simulator_truth"
    elif fault=="nan_offset":reg["local_position"][0]=np.nan
    elif fault=="nonrotation":reg["local_rotation"][0,0]=2.
    elif fault=="nan_fk":s.ctx.eef_pos=lambda:np.array([0.,0.,np.nan])
    with pytest.raises(ValueError):control_position(s,"pin_left")


@pytest.mark.parametrize("fault",["nan_force","nan_touch","nan_actor","nan_target","infinite_depth","invalid_registration","unbounded_steps","encoder_depth_outside_envelope"])
def test_large_sensor_or_command_anomaly_stops_before_servo(monkeypatch,fault):
    setup_feedback(monkeypatch);s=session();budget=1000
    if fault=="nan_force":s.external_force=lambda part:np.nan
    elif fault=="nan_touch":s.ctx.grasp_contacts=lambda part:dict(held=True,left_n=np.nan,right_n=1.)
    elif fault=="nan_actor":monkeypatch.setattr(controller,"load_actor",lambda p:lambda obs:np.full(3,np.nan))
    elif fault=="nan_target":s.artifacts["insert"]["target"][0]=np.nan
    elif fault=="infinite_depth":s.artifacts["insert"]["absolute_depth_limit_m"]=np.inf
    elif fault=="invalid_registration":s.held_visual_transforms_v12["pin_left"]["valid"]=False
    elif fault=="unbounded_steps":budget=np.inf
    elif fault=="encoder_depth_outside_envelope":s.ctx.eef_pos=lambda:np.array([0.,0.,-.05])
    result=controller.execute(s,"pin_left","insert","not-read-in-stub",budget)
    assert not result.ok and result.metrics["sensor_boundary_status"]=="unknown"
    assert not s.calls


def test_press_budget_exhaustion_is_not_claimed_depth_reached(monkeypatch):
    setup_feedback(monkeypatch);s=session()
    # An actuator with no encoder movement is observable. Constant bilateral
    # contact must not turn repeated commands into measured progress.
    result=controller.bounded_pin_press(s,"pin_left",.004,2.)
    assert not result.ok and result.metrics["stop_reason"]=="press_step_budget_exhausted"
    assert len(s.calls)==241
    assert not result.metrics["sensor_contract"]["axial_slip_observed"]


def test_bilateral_contact_cannot_observe_constant_axial_slip():
    s=session();first=control_position(s,"pin_left").copy()
    # Different hidden slip states can produce identical encoders/contact.
    # Do not invent a slip measurement from the registered FK prediction.
    s.hidden_part_position=[0,0,.09]
    np.testing.assert_allclose(control_position(s,"pin_left"),first)
    assert s.ctx.grasp_contacts("pin_left")["held"]
    assert controller._pin_sensor_contract()["rigid_attachment_assumed"]
