from dataclasses import replace
import numpy as np
import pytest
from scripts.check_v12_pin_contact_tolerance import CONFIG
from simbench.value.pin_contact_v12 import evaluate_contact_state,functional_retention_window
from simbench.value.pin_geometry import evaluate_pin_state


def state(origin=(0,0,.012),*,phase='inserted_after_release',released=True,touching=False,contacts=()):
    return evaluate_contact_state(origin,[0,0,1],[0,0,0],np.eye(3),released=released,
        touching_finger=touching,phase=phase,config=replace(CONFIG,contact_robustness=True),contacts=contacts)


def test_soft_overlap_reports_strict_failure_and_requires_matching_contact():
    origin=[.00078,0,.012]
    contact=[dict(distance_m=-.00008,normal_force_n=.08,shaft_receiver=True,normal_world=[1,0,0])]
    row=state(origin,contacts=contact)
    assert row['success'] and not row['strict_zero_interpenetration']['success']
    assert row['maximum_receiver_contact_penetration_m']==.00008
    assert row['numerical_contact_guard_m']==pytest.approx(.000165)
    assert not state(origin)['success']
    assert not state(origin,contacts=[dict(contact[0],shaft_receiver=False)])['success']
    assert not state(origin,contacts=[dict(contact[0],normal_world=[0,0,1])])['success']
    assert not state(origin,contacts=[dict(contact[0],normal_world=[.1,0,.9])])['success']
    assert not state(origin,contacts=[dict(contact[0],normal_world=[])])['success']
    assert row['receiver_contacts'][0]['radially_dominant']
    assert row['receiver_contacts'][0]['hole_axis_world']==[0.,0.,1.]
    assert not state(origin,contacts=[dict(contact[0],distance_m=-.00017)])['success']
    assert not state([.00087,0,.012],contacts=contact)['success']


def test_held_geometry_does_not_require_release_but_retention_does():
    assert state(released=False,touching=True,phase='inserted_while_held')['success']
    assert not state(released=False)['success']
    assert not state(touching=True)['success']
    assert state(phase='retained_after_stroke')['retained_after_stroke']


def test_legacy_ideal_geometry_is_unchanged():
    row=evaluate_pin_state([.00078,0,.012],[0,0,1],[0,0,0],[0,0,1],True,False,
        config=CONFIG,hole_axes=np.eye(3)[:2])
    assert not row['success'] and 'strict_zero_interpenetration' not in row


def test_functional_window_allows_motion_but_rejects_any_loss_of_occupancy():
    held=[state(),dict(state(),motion_diagnostics={'angular_speed_rad_s':2.})]
    waits=[]
    result=functional_retention_window(lambda:held.pop(0),waits.append,state(),duration=.2,samples=3)
    assert result['success'] and waits==[.1,.1]
    assert result['functional_retention_window']['samples_observed']==3
    waits=[]
    failed=state([0,0,.046])
    result=functional_retention_window(lambda:failed,waits.append,state(),duration=.2,samples=5)
    assert not result['success'] and len(waits)==1
    assert result['functional_retention_window']['samples_observed']==2


def test_initial_failure_is_not_waited_away_and_invalid_window_is_rejected():
    def forbidden(*args): raise AssertionError('failure must not be healed by waiting')
    result=functional_retention_window(forbidden,forbidden,state([0,0,.048]),duration=.2,samples=5)
    assert not result['success'] and result['functional_retention_window']['samples_observed']==1
    with pytest.raises(ValueError): functional_retention_window(forbidden,forbidden,state(),duration=0,samples=5)


def test_control_tick_quantization_must_not_shorten_declared_observation():
    initial=dict(state(),sample_time_s=1.)
    too_short=dict(state(),sample_time_s=1.16)
    row=functional_retention_window(lambda:too_short,lambda seconds:None,initial,duration=.2,samples=2)
    assert not row['success'] and row['functional_retention_window']['duration_observed_s']==pytest.approx(.16)


def test_real_contact_adapter_and_passive_window_on_independent_cad_fixture():
    import mujoco
    from types import SimpleNamespace
    from scripts.check_v12_pin_contact_tolerance import model_xml,MASS
    from simbench.value.pin_contact_v12 import evaluate_context_contact
    model=mujoco.MjModel.from_xml_string(model_xml(.002,.45));data=mujoco.MjData(model)
    bid=model.body('pin').id
    for _ in range(500):
        data.xfrc_applied[bid,0]=2*MASS*9.81
        mujoco.mj_step(model,data)
    ctx=SimpleNamespace(model=model,data=data,body_id=lambda name:model.body(name).id)
    def sample():
        return evaluate_context_contact(ctx,'pin','receiver',data.xpos[bid].copy(),
            data.xmat[bid].reshape(3,3)[:,2].copy(),np.zeros(3),np.eye(3),
            released=True,touching_finger=False,phase='inserted_after_release',
            config=replace(CONFIG,contact_robustness=True))
    first=sample()
    assert first['receiver_contacts'] and first['maximum_receiver_contact_penetration_m']>0
    assert first['maximum_receiver_contact_penetration_m']<first['numerical_contact_guard_m']
    data.xfrc_applied[:]=0
    def wait(seconds):
        for _ in range(int(np.ceil(seconds/model.opt.timestep-1e-12))): mujoco.mj_step(model,data)
    result=functional_retention_window(sample,wait,first,duration=.2,samples=5)
    assert result['success'] and result['functional_retention_window']['samples_observed']==5
    assert result['functional_retention_window']['duration_observed_s']>=.2-1e-9
