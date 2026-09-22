"""V16 construction/controller regressions; these are not task-success labels."""
from types import SimpleNamespace
import math

import numpy as np
import pytest

from simbench.assembly.library import Session
from simbench.assembly.placement_catalog_v13 import placement_clearance_catalog
from simbench.assembly.printed_kit import planning_metadata
from simbench.value.full_task_v7 import _clean
from simbench.value.stage_v5 import stage_calls
from simbench.value.plan import PlanIR


def _pose(xyz, yaw=0.):
    return dict(valid=True, position_m=list(xyz),
                quat_wxyz=[math.cos(yaw/2), 0., 0., math.sin(yaw/2)],
                fit_residual_m=.0005)


class _Clear:
    def __init__(self, primitives):
        self.primitives=primitives
        self.provenance=dict(no_simulation_steps=True)

    def query(self, xyz, rotation, gap, receivers):
        return dict(min_clearance_m=.02, collision=False,
                    distance_is_lower_bound=False, pairs=[])


def test_object_frame_grasp_center_is_a_generic_executable_port():
    seen=[]
    session=SimpleNamespace(strict_rgbd_v12=True,
        grasp_specs={"end_stop":(.023,.026)}, artifacts={},
        ctx=SimpleNamespace(arm_qpos=np.zeros(7)),
        arm=SimpleNamespace(ik_with_restarts=lambda xyz,rotation: seen.append(np.asarray(xyz).copy()) or np.zeros(7)))
    session.artifact=lambda key,kind,part,**fields:session.artifacts.update({key:dict(part=part,**fields)})
    yaw=.6
    session.artifacts["pose"]=dict(xyz=np.array([.1,.2,.8]),
        quat=np.array([math.cos(yaw/2),0.,0.,math.sin(yaw/2)]))
    result=Session.propose_grasps(session,"end_stop",yaws=[0.],yaw_frame="object",
        center_offset=[.008,0.,0.])
    assert result.ok
    expected=np.array([.1,.2,.8])+np.array([math.cos(yaw)*.008,math.sin(yaw)*.008,.023])
    np.testing.assert_allclose(session.artifacts["grasps"]["candidates"][0]["xyz"],expected)
    np.testing.assert_allclose(seen[0],expected+[0.,0.,.10])


def test_catalogue_declares_and_binds_scene_relative_grasp_centres():
    cad=planning_metadata()
    parts=("carriage","end_stop","pin_left","pin_right","handle")
    obs=dict(objects={p:_pose([-.35+i*.06,-.22,.84]) for i,p in enumerate(parts)},
        fixtures={"guide_base":_pose([0.,0.,.812])},
        assembly_targets={p:_pose([.02+i*.02,.08,.84]) for i,p in enumerate(parts)},
        receiver_geometry=dict(rail=dict(entry_approach_m=[-.12,.08,.87],
            entry_m=[-.12,.08,.84],axis=[1.,0.,0.])))
    rows=placement_clearance_catalog(obs,cad,"carriage",source_axis_offsets=(0.,),
        height_offsets=(0.,),query_factory=_Clear)
    offsets={tuple(r["grasp_center_offset_body_m"]) for r in rows}
    assert offsets=={(0.,0.,0.),(.012,0.,0.),(-.012,0.,0.),(0.,.012,0.),(0.,-.012,0.)}
    for row in rows:
        expected=np.asarray(obs["objects"]["carriage"]["position_m"])+np.asarray(
            row["grasp_center_offset_body_m"])+[0.,0.,.027]
        np.testing.assert_allclose(row["source_eef_m"],expected)


def test_plan_ir_carries_grasp_center_without_task_specific_value_schema():
    choice=dict(yaw=math.pi/2,grasp_yaw_frame="object",center_offset=[.008,0.,0.],
        height=.004,width=.022,clearance=1.,force=5.,speed=.004,press_force=2.,
        placement_yaw=math.pi/2)
    calls=stage_calls("end_stop",[0.,0.,.836],choice,0,v7=True,v12=True)
    grasp=next(c for c in calls if c.skill=="estimate_grasp")
    assert grasp.arguments["center_offset"].value==[.008,0.,0.]
    assert grasp.arguments["center_offset"].frame=="object"


def test_plan_ir_semantic_reconstruction_includes_grasp_center():
    choice=dict(yaw=math.pi/2,grasp_yaw_frame="object",center_offset=[.008,0.,0.],
        height=.004,width=.022,clearance=1.,force=5.,speed=.004,press_force=2.,
        placement_yaw=math.pi/2)
    calls=stage_calls("end_stop",[0.,0.,.836],choice,0,v7=True,v12=True)
    prefix=dict(id="offset",part="end_stop",execution="program",skill_version="feedback.v12",
                steps=[dict(skill=calls[0].skill,params={k:a.value for k,a in calls[0].arguments.items()})],
                order=["end_stop"],choices={"end_stop":choice})
    PlanIR("offset",calls,1,prefix,"unknown",protocol="assembly.program.feedback.v2").validate(
        ("end_stop",))


def test_wipe_unload_is_free_space_after_contact_skill_release(monkeypatch):
    calls=[]
    session=SimpleNamespace(strict_rgbd_v12=False,functional_acceptance_v12=False,
        decision_observation={"objects":{"wipe_tool":{"valid":True,"position_m":[-.2,-.2,.82]}}},
        artifacts={},stage_passes={})
    def call(skill,**kwargs):
        calls.append((skill,kwargs))
        if skill=="plan_path":
            session.artifacts["wipe"]={"points":[[-.01,.085,.828],[.01,.085,.828]]}
        return SimpleNamespace(ok=True)
    session.call=call
    monkeypatch.setattr("simbench.value.full_task_v7.pick",lambda *a,**k:None)
    monkeypatch.setattr("simbench.value.full_task_v7.transfer_part",lambda *a,**k:None)
    _clean(session,0,1.5,8.)
    deltas=[row[1]["delta"] for row in calls if row[0]=="move" and "delta" in row[1]]
    assert deltas[:1]==[[0,0,.10]]
    unload=next(row for row in calls if row[0]=="move" and row[1].get("delta")==[0,0,.10])
    assert unload[1]["speed"]==.01
