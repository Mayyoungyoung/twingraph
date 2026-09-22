from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import numpy as np
import pytest

from simbench.assembly.gripper_clearance_v12 import GripperClearance,PANDA,SHELLS,PADS
from simbench.assembly.sensor_learning_v12 import pin_depth_contract,sensor_depth,limit_pin_eef_command
from simbench.assembly.control import down


def sensor_session(position=(0,0,.01)):
    ctx=SimpleNamespace(eef_mat=lambda:np.eye(3),eef_pos=lambda:np.asarray(position,float),
        obj_pos=lambda *a:(_ for _ in ()).throw(AssertionError("object truth is forbidden")))
    return SimpleNamespace(ctx=ctx,held="pin_left",strict_rgbd_v12=True,
        held_visual_transforms_v12={"pin_left":dict(local_position=np.zeros(3),local_rotation=np.eye(3),
            source="rgbd_at_grasp_plus_encoder_forward_kinematics")},
        pin_insertion_config=SimpleNamespace(shaft_tip_offset_m=-.047),
        planning_cad={"pin_head_seated_total_depth_m":.046})


def plan():
    return dict(axis=[0,0,-1],hole_entry_m=[0,0,0],command_depth_m=.043,
        press_extra_m=.001,maximum_total_depth_m=.044,absolute_depth_limit_m=.046)


def test_explicit_depth_references_observed_entry_not_contact_trigger():
    session=sensor_session();c=pin_depth_contract(session,"pin_left",plan())
    assert sensor_depth(session,"pin_left",[0,0,.01],c)==pytest.approx(.037)
    assert c["command_depth"]-sensor_depth(session,"pin_left",[0,0,.01],c)==pytest.approx(.006)
    command,clipped=limit_pin_eef_command(session,"pin_left",[0,0,-.02],c,c["command_depth"])
    assert command[2]==pytest.approx(.004)
    assert clipped==pytest.approx(.024)
    assert sensor_depth(session,"pin_left",command,c)==pytest.approx(.043)


def test_press_and_command_cannot_sum_past_real_head_seated_depth():
    session=sensor_session();p=plan();p.update(press_extra_m=.006,maximum_total_depth_m=.049)
    with pytest.raises(ValueError,match="CAD envelope"):pin_depth_contract(session,"pin_left",p)
    p=plan();p["maximum_total_depth_m"]+=.001
    with pytest.raises(ValueError,match="CAD envelope"):pin_depth_contract(session,"pin_left",p)


def test_receiver_plane_search_accumulates_axial_steps_smaller_than_ik_tolerance():
    from simbench.assembly.sensor_learning_v12 import advance_pin_search_command
    command=np.zeros(3)
    for _ in range(4):
        command=advance_pin_search_command(command,[0,0,0],[0,0,0],[.001,0,-.03],[0,0,-1],.00006)
    np.testing.assert_allclose(command,[.001,0,-.00024])


def test_full_shell_masks_are_live_and_meshes_not_rescaled():
    root=ET.parse(PANDA).getroot()
    for name in SHELLS+PADS:
        geom=root.find(f".//geom[@name='{name}']")
        assert geom.get("contype")==geom.get("conaffinity")=="1"
    for name in ("hand","finger"):
        assert root.find(f"asset/mesh[@name='{name}']").get("scale") is None


def test_static_query_detects_full_finger_boss_collision_and_has_no_fixed_palm_exclusion():
    # An actual declared boss primitive; this is a geometric counterexample,
    # not a task rollout or a substitute for the original CAD plate.
    query=GripperClearance({"end_stop":[dict(name="stop_boss",type="box",
        pos=[0,0,.023],size=[.011,.013,.008])]})
    fixture={"end_stop":dict(valid=True,position_m=[-.092,0,.024],quat_wxyz=[1,0,0,0])}
    yaw0=query.query([-.092,-.032,.053],down(0),.04,fixture)
    yaw90=query.query([-.092,-.032,.057],down(np.pi/2),.04,fixture)
    assert yaw0["collision"] and any(row["shell_involved"] for row in yaw0["pairs"] if row["distance_m"]<0)
    assert not yaw90["collision"]
    assert query.model.joint("query_gripper_pose").type[0]==0  # free root: palm-vs-fixed contact is not filtered


def test_missing_receiver_pose_remains_unknown():
    query=GripperClearance({"end_stop":[dict(name="plate",type="box",pos=[0,0,0],size=[.01,.01,.01])]})
    with pytest.raises(ValueError,match="valid RGB-D"):
        query.query([0,0,.2],down(0),.01,{"end_stop":dict(valid=False)})


def test_joint_evaluator_requires_both_original_receiver_intervals():
    import mujoco
    from simbench.assembly.skills_v12 import evaluate_pin_joint_engagement
    from simbench.value.stage_v12 import PIN_CONFIG
    # Independent geometry-unit fixture. No controller, detector or policy.
    xml='<mujoco><worldbody><body name="end_stop" pos="0 0 .018"/><body name="guide_base" pos="0 0 -.006"/><body name="pin_left" pos="0 0 .040"><freejoint/><inertial pos="0 0 0" mass=".004" diaginertia=".0001 .0001 .0001"/></body></worldbody></mujoco>'
    model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model)
    body_id=lambda name:mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name)
    ctx=SimpleNamespace(model=model,data=data,body_id=body_id,
        obj_pos=lambda name:data.xpos[body_id(name)].copy(),
        obj_axis=lambda name:data.xmat[body_id(name)].reshape(3,3)[:,2].copy(),
        obj_pose=lambda name:(data.xpos[body_id(name)].copy(),data.xquat[body_id(name)].copy()))
    cad=dict(pin_hole_offsets_m=[[0,0,.018],[0,0,.018]],base_hole_offsets_m=[[0,0,.006],[0,0,.006]],
        base_receiver_depth_m=.012,pin_base_minimum_depth_m=.006)
    session=SimpleNamespace(ctx=ctx,planning_cad=cad,pin_insertion_config=PIN_CONFIG,held="pin_left")
    mujoco.mj_forward(model,data)
    deep=evaluate_pin_joint_engagement(session,"pin_left","inserted_while_held")
    assert deep["success"] and set(deep["receivers"])=={"end_stop","guide_base"}
    data.qpos[2]=.075;mujoco.mj_forward(model,data)
    shallow=evaluate_pin_joint_engagement(session,"pin_left","inserted_while_held")
    assert shallow["receivers"]["end_stop"]["success"]
    assert not shallow["receivers"]["guide_base"]["success"] and not shallow["success"]


def test_catalog_rotates_with_observed_receiver_and_preserves_source_yaw():
    from scipy.spatial.transform import Rotation
    from simbench.assembly.printed_kit import planning_metadata
    from simbench.assembly.gripper_clearance_v12 import pin_clearance_catalog
    rotation=np.pi/4;q=Rotation.from_euler("z",rotation).as_quat()
    obs=dict(objects={"pin_left":dict(valid=True,position_m=[-.3,0,.85],quat_wxyz=[1,0,0,0],fit_residual_m=.0002)},
        fixtures={"guide_base":dict(valid=True,position_m=[.3,.5,.8],quat_wxyz=q[[3,0,1,2]].tolist(),fit_residual_m=.0002)})
    rows=pin_clearance_catalog(obs,planning_metadata(),"pin_left",source_yaws=(0.,),height_offsets=(.004,),command_depths=(.043,))
    assert all(row["yaw"]==0. for row in rows)
    passed=[row for row in rows if row["status"]=="necessary_pass"]
    assert passed and all(abs(np.sin(row["placement_yaw"]-rotation))>.99 for row in passed)
    assert all(not row["source_grasp_ik_checked"] for row in rows)


def test_observed_entry_uses_shared_axis_and_true_tilted_receiver_plane():
    from simbench.assembly.gripper_clearance_v12 import common_corridor_entry
    normal=np.array([np.sin(.03),0,np.cos(.03)])
    route=dict(axis=[0,0,1],stop_axis=normal.tolist(),common_axis_point_m=[.0004,0,0],stop_entry_m=[.0008,0,.036])
    entry=common_corridor_entry(route)
    assert entry[0]==pytest.approx(.0004)
    assert np.dot(entry-np.asarray(route["stop_entry_m"]),normal)==pytest.approx(0.,abs=1e-12)
    assert entry[2]>.036
    route["stop_entry_on_common_axis_m"]=[.0004,0,.036012]
    np.testing.assert_allclose(common_corridor_entry(route),route["stop_entry_on_common_axis_m"])
