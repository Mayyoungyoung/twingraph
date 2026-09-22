"""Development-only visual supply / explicitly calibrated two-layer fixture.

This pins the receiver at the CAD mating pose before a fresh reset. It is not
an assembled end-stop success, a dynamic visual-fixture test, or a full task.
"""
import argparse,json,time
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
import mujoco


def run(seed,out,source_yaw,goal_yaw,height,depth,friction_scale=1.):
    from simbench.value.stage_v12 import make_scene
    from simbench.value.stage_v7 import refresh_visual_observation
    from simbench.assembly.control import HOME
    from simbench.assembly.library import Session
    from simbench.core.sim_context import MjContext
    from simbench.assembly.skills_v12 import configure_v12_skills,evaluate_pin_joint_engagement
    from simbench.assembly.sensor_learning_v12 import gripper_fixture_contacts
    from simbench.value.stage_v5 import stage_calls
    from simbench.value.plan import execute_calls,argument
    from simbench.assembly.gripper_clearance_v12 import PANDA
    import hashlib
    out=Path(out);out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    if not .9<=friction_scale<=1.1:raise ValueError("development contact perturbation must stay within +/-10%")
    _,old,path,_=make_scene(seed,out/"source_scene",preinstalled_end_stop=False)
    root=ET.parse(path).getroot();stop=root.find(".//body[@name='end_stop']")
    base=root.find(".//body[@name='guide_base']")
    bp=np.fromstring(base.get("pos"),sep=" ");bq=np.fromstring(base.get("quat","1 0 0 0"),sep=" ")
    bR=Rotation.from_quat(bq[[1,2,3,0]]).as_matrix();cad=old.planning_cad
    mate=cad["end_stop_mating_pose_in_base"];sp=bp+bR@np.asarray(mate["position_m"])
    for joint in list(stop.findall("freejoint")):stop.remove(joint)
    stop.set("pos"," ".join(map(str,sp)));stop.set("quat"," ".join(map(str,bq)))
    contact_parameters=[]
    for body in (base,stop):
        for geom in body.findall("geom"):
            if geom.get("contype","1")==geom.get("conaffinity","1")=="0":continue
            friction=np.fromstring(geom.get("friction",".45 .01 .0001"),sep=" ")
            before=friction.copy();friction[0]*=friction_scale
            geom.set("friction"," ".join(map(str,friction)))
            contact_parameters.append(dict(geom=geom.get("name"),before=before.tolist(),after=friction.tolist()))
    path=out/"source_scene"/"fixed_calibrated_two_layer.xml";ET.ElementTree(root).write(path,encoding="unicode")
    ctx=MjContext(path,control_freq=50);ctx.reset();ctx.data.qpos[ctx.arm_qadr]=HOME
    mujoco.mj_forward(ctx.model,ctx.data);ctx.hold_arm();ctx.set_finger_ctrl(.04)
    for _ in range(80):ctx.step()
    session=Session(ctx,seed=seed,out=out,parts=old.parts,grasp_specs=old.grasp_specs,capabilities=old.capabilities)
    for key in ("stage_targets","task_version","pin_insertion_config","visual_templates_fn","planning_cad",
                "perception_estimator_fn","capture_detector_fn","detector_size"):
        if hasattr(old,key):setattr(session,key,getattr(old,key))
    configure_v12_skills(session);observation=refresh_visual_observation(session)
    (out/"observation.json").write_text(json.dumps(observation,indent=2),encoding="utf-8")
    part="pin_left";entry=sp+bR@np.asarray(cad["pin_hole_offsets_m"][0]);axis=bR[:,2]
    target=entry-axis*(depth+cad["pin_shaft_offsets_m"][0]);session.stage_targets[part]=target.tolist()
    choices=dict(yaw=source_yaw,placement_yaw=goal_yaw,height=height,clearance=.99,force=6.,
        speed=.004,force_limit=8.,press_force=2.,pin_command_depth_m=depth,pin_press_extra_m=0.)
    protocol=dict(seed=seed,choices=choices,panda_xml_sha256=hashlib.sha256(PANDA.read_bytes()).hexdigest(),
        calibration=dict(base_position_m=bp.tolist(),stop_position_m=sp.tolist(),hole_entry_m=entry.tolist(),axis=axis.tolist()),
        setup="fixed nominal CAD two-layer fixture; RGB-D supply grasp",complete_task_trial=False,
        fixed_receiver_is_not_an_assembly_success=True,target_source="explicit calibration, not simulator object truth",
        gripper_model="full unscaled conservative palm/finger collision hulls and pads",
        receiver_friction_scale=friction_scale,contact_parameters=contact_parameters)
    repo=Path(__file__).resolve().parents[1]
    source_paths=["scripts/check_v12_pin_full_gripper.py","simbench/assets/panda/panda.xml",
        "simbench/assembly/sensor_learning_v12.py","simbench/assembly/skills_v12.py",
        "simbench/assembly/control.py","simbench/assembly/grasp_v12.py","simbench/assembly/library.py",
        "simbench/assembly/printed_kit.py","simbench/assembly/ports.py","simbench/value/stage_v5.py",
        "simbench/value/stage_v12.py","simbench/value/pin_geometry.py","simbench/value/pin_contact_v12.py"]
    protocol["source_file_sha256"]={name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in source_paths}
    (out/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    success=False;error=None;acceptance=None
    try:
        calls=stage_calls(part,target,choices,0,v7=True,v12=True,
            assembly_target=dict(hole_entry_m=entry.tolist(),axis=axis.tolist()),cad=cad)
        calls[0].arguments["required_parts"]=argument([part])
        execute_calls(session,SimpleNamespace(prefix={"part":part,"targets":{part:target.tolist()}}),calls)
        acceptance=evaluate_pin_joint_engagement(session,part,"inserted_after_release")
        success=bool(acceptance["success"])
        if not success:error="released shaft did not simultaneously occupy both real receiver layers"
    except Exception as exc:
        error=f"{type(exc).__name__}: {session.results[-1].get('skill')} / {session.results[-1].get('reason')}" if session.results else f"{type(exc).__name__}: {exc}"
    report=dict(**protocol,success=success,error=error,steps=session.results,
        acceptance=acceptance,gripper_fixture_contact=gripper_fixture_contacts(session),
        actor_artifact=session.artifacts.get("learned_insertion_result"),
        end_state_evaluator_only={p:dict(position_m=ctx.obj_pos(p).tolist(),axis=ctx.obj_axis(p).tolist())
            for p in (part,"end_stop","guide_base")},wall_seconds=time.perf_counter()-started)
    (out/"summary.json").write_text(json.dumps(report,indent=2,default=lambda x:np.asarray(x).tolist()),encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("success","error","wall_seconds")}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--seed",type=int,default=1600);p.add_argument("--out",required=True)
    p.add_argument("--source-yaw",type=float,default=0.);p.add_argument("--goal-yaw",type=float,default=np.pi/2)
    p.add_argument("--height",type=float,default=.004);p.add_argument("--depth",type=float,default=.043)
    p.add_argument("--friction-scale",type=float,default=1.)
    a=p.parse_args();run(a.seed,a.out,a.source_yaw,a.goal_yaw,a.height,a.depth,a.friction_scale)
