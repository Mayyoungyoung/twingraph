"""Sensor/plan contract regressions; no mocked result is physical evidence."""
from types import SimpleNamespace
import math

import numpy as np
import pytest

from simbench.assembly.control import down
from simbench.assembly.library import Session
from simbench.assembly.library import HANDLERS
from simbench.assembly.ports import validate_ports
from simbench.value.stage_v5 import stage_calls


def test_detect_after_placement_reads_new_pixels_instead_of_supply_cache(monkeypatch):
    session=SimpleNamespace(strict_rgbd_v12=True, parts=("end_stop",),
        perception_backend="rgbd_geometry", decision_observation={"backend":"rgbd_geometry",
            "objects":{"end_stop":dict(valid=True,position_m=[-.2,-.3,.818])},"sha256":"before"},
        observations={"end_stop":[np.array([-.2,-.3,.818])]})
    seen=[]
    def acquire(s,parts):
        seen.append(tuple(parts))
        fresh={"backend":"rgbd_geometry","objects":{"end_stop":dict(valid=True,
            position_m=[-.017,.085,.836])},"sha256":"new_pixels"}
        Session.set_decision_observation(s,fresh)
        return fresh
    monkeypatch.setattr("simbench.value.stage_v7.refresh_visual_observation",acquire)
    result=Session.observe_parts(session,required_parts=["end_stop"])
    assert result.ok and seen==[("end_stop",)]
    assert result.metrics["fresh_rgbd_capture"] and not result.metrics["frozen"]
    assert result.metrics["observation_sha256"]=="new_pixels"
    np.testing.assert_allclose(session.observations["end_stop"][0],[-.017,.085,.836])


def test_object_relative_grasp_rebinds_to_new_detected_orientation():
    checked=[]
    def ik(xyz,rotation):
        checked.append(rotation.copy());return np.zeros(7)
    session=SimpleNamespace(strict_rgbd_v12=True, grasp_specs={"end_stop":(.023,.026)},
        artifacts={}, ctx=SimpleNamespace(arm_qpos=np.zeros(7)), arm=SimpleNamespace(ik_with_restarts=ik))
    session.artifact=lambda key,kind,part,**fields:session.artifacts.update({key:dict(part=part,**fields)})
    for observed in (.2,-.4):
        session.artifacts["pose"]=dict(xyz=np.array([0.,0.,.818]),
            quat=np.array([math.cos(observed/2),0.,0.,math.sin(observed/2)]))
        result=Session.propose_grasps(session,"end_stop",yaws=[math.pi/2],yaw_frame="object",width=.022)
        assert result.ok
        grasp=session.artifacts["grasps"]["candidates"][0]
        assert grasp["yaw"]==pytest.approx(observed+math.pi/2)
        assert grasp["width"]==.022
        np.testing.assert_allclose(checked[-1],down(observed+math.pi/2),atol=1e-12)


def test_object_relative_source_path_uses_selected_grasp_world_yaw():
    choices=dict(yaw=math.pi/2,grasp_yaw_frame="object",height=.004,width=.022,
        clearance=1.,force=5.,speed=.004,press_force=2.,placement_yaw=math.pi/2)
    calls=stage_calls("end_stop",[0.,0.,.836],choices,0,v7=True,v12=True)
    grasp=next(c for c in calls if c.skill=="estimate_grasp")
    assert grasp.arguments["yaws"].frame=="object"
    path=next(c for c in calls if c.skill=="plan_path")
    assert path.arguments["yaw"].status=="deferred"
    assert path.arguments["yaw"].source_output=="grasp_yaw"


def test_deep_pin_alignment_starts_above_mouth_and_executes_absolute_depth():
    choice=dict(yaw=0.,height=.004,clearance=1.,force=9.,speed=.004,force_limit=8.,
        press_force=2.,placement_yaw=math.pi/2,pin_command_depth_m=.043,pin_press_extra_m=0.)
    target=dict(hole_entry_m=[0.,0.,.854],axis=[0.,0.,1.])
    calls=stage_calls("pin_left",[0.,0.,.858],choice,0,v7=True,v12=True,
        assembly_target=target,cad=dict(pin_shaft_offsets_m=[-.047,.006]))
    align=next(c for c in calls if c.skill=="move" and c.arguments.get("reference") is not None)
    assert align.arguments["target"].value[2]-.047==pytest.approx(.864)
    contact=next(c for c in calls if c.skill=="plan_path" and c.arguments.get("method") is not None)
    assert contact.arguments["pin_command_depth_m"].value==.043
    assert contact.arguments["hole_entry_m"].value==[0.,0.,.854]
    assert contact.arguments["target"].value[2]-.047==pytest.approx(.854-.043)
    insertion=next(c for c in calls if c.skill=="insert")
    assert 1 <= insertion.arguments["max_steps"].value <= 1000
    assert any(c.skill=="inspect" and c.arguments.get("what") is not None
               and c.arguments["what"].value=="pin_joint" for c in calls)


def test_insertion_step_budget_is_rejected_before_rollout():
    with pytest.raises(ValueError, match="max_steps"):
        validate_ports(HANDLERS["learned_insert"],
            dict(part="pin_left", artifact="insert", policy=None, max_steps=1800))
