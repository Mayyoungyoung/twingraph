"""Declared CAD fixture/full-gripper sweep audit, not a physical success test."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from simbench.assembly.control import down
from simbench.assembly.gripper_clearance_v12 import GripperClearance, receiver_primitives_from_model


def run(directory):
    import mujoco
    from simbench.value.stage_v7 import StageV7Spec
    from simbench.value.stage_v12 import write_scene
    out=Path(directory)
    out.mkdir(parents=True,exist_ok=False)
    # Compile the original CAD only; there is no reset/settle/rollout, and no
    # MjData object pose is read as a sensor or as this declared calibration.
    scene=write_scene(StageV7Spec.sample(1600,"L1"),out/"cad_scene")
    model=mujoco.MjModel.from_xml_path(str(scene))
    primitives=receiver_primitives_from_model(model)
    query=GripperClearance(primitives)
    poses={"guide_base":dict(valid=True,position_m=[0,0,0],quat_wxyz=[1,0,0,0]),
           "end_stop":dict(valid=True,position_m=[-.092,0,.024],quat_wxyz=[1,0,0,0])}
    rows=[]
    for y in (-.032,.032):
        for yaw in (0.,np.pi/2,np.pi,-np.pi/2):
            for height in (-.004,0.,.004,.007):
                for depth in (.008,.020,.030,.036,.040,.042,.044,.046):
                    entry=np.array([-.092,y,.042])
                    origin=entry+[0,0,.047-depth]
                    eef=origin+[0,0,.006+height]
                    phases=[]
                    def check(phase,p,gap):
                        row=query.query(p,down(yaw),gap,poses)
                        phases.append(dict(phase=phase,eef_position_m=p.tolist(),finger_joint_gap_m=gap,**row))
                    for value in np.linspace(-.030,depth,int(np.ceil((depth+.030)/.002))+1):
                        check("insertion",entry+np.array([0,0,.047-value+.006+height]),.0095)
                    for gap in np.linspace(.0095,.04,9): check("opening",eef,float(gap))
                    for lift in np.linspace(0.,.10,11): check("retraction",eef+[0,0,lift],.04)
                    worst=min(phases,key=lambda row:row["min_clearance_m"])
                    rows.append(dict(pin_side="left" if y<0 else "right",yaw_rad=float(yaw),
                        grasp_height_offset_m=height,command_depth_m=depth,
                        min_clearance_m=worst["min_clearance_m"],worst_phase=worst["phase"],
                        worst_eef_position_m=worst["eef_position_m"],worst_pairs=worst["pairs"][:6],
                        sampled_collision_free=all(not row["collision"] for row in phases),
                        samples=len(phases),by_phase={name:min(row["min_clearance_m"] for row in phases if row["phase"]==name)
                            for name in ("insertion","opening","retraction")}))
    report=dict(schema="twingraph.full_gripper_static_clearance.r7",declared_fixture_poses=poses,
        pose_source="explicit nominal mating CAD calibration, not camera or simulator truth",
        measured_physical_success=False,advances_physics=False,robot_IK_checked=False,
        limitations="discrete static convex-hull clearance only; source grasp feasibility, actual grip and tracking require separate checks",
        geometry_provenance=query.provenance,receiver_primitives=primitives,rows=rows)
    (out/"clearance.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    summary=[{k:row[k] for k in ("yaw_rad","grasp_height_offset_m","command_depth_m","min_clearance_m","worst_phase","sampled_collision_free")}
             for row in rows if row["pin_side"]=="left" and row["command_depth_m"] in (.008,.042,.046)]
    print(json.dumps(dict(cases=len(rows),nominal_left_examples=summary),indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--out",required=True)
    run(parser.parse_args().out)
