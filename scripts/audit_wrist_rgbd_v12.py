"""Offline wrist visibility audit. Truth scores output, never fits the detector.

An optional recorded state is copied into a NEW scene for sensor diagnostics.
This is not an execution result and never modifies the source experiment.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from simbench.value import stage_v7, stage_v12
from simbench.value.wrist_rgbd_v12 import CAMERA_NAME


def audit(session, label, out):
    stage_v7.refresh_visual_observation(session, parts=session.parts)
    frames, calibrations = stage_v7.capture_detector(session)
    Image.fromarray(frames[CAMERA_NAME]["rgb"]).save(out / f"{label}.png")
    np.savez_compressed(out / f"{label}_frames.npz", **{
        f"{view}_{key}":value for view,frame in frames.items() for key,value in frame.items()})
    transform = calibrations[CAMERA_NAME].matrix()
    cid = mujoco.mj_name2id(session.ctx.model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    record = dict(label=label, camera_position=transform[:3, 3].tolist(),
        calibration=calibrations[CAMERA_NAME].manifest(), acquisition=session.decision_observation["acquisition"],
        camera_FK_max_error=float(max(np.max(np.abs(transform[:3,3]-session.ctx.data.cam_xpos[cid])),
            np.max(np.abs(transform[:3,:3]-session.ctx.data.cam_xmat[cid].reshape(3,3))))), objects={})
    for part, row in session.decision_observation["objects"].items():
        truth, quaternion = session.ctx.obj_pose(part)
        truth_yaw = float(Rotation.from_quat(quaternion[[1,2,3,0]]).as_euler("xyz")[2])
        pred_yaw = 2*np.arctan2(row["quat_wxyz"][3],row["quat_wxyz"][0]) if row["valid"] else None
        record["objects"][part] = dict(valid=row["valid"], pose=row["position_m"],
            truth=truth.tolist(), error_m=float(np.linalg.norm(np.asarray(row["position_m"])-truth)) if row["valid"] else None,
            yaw_error_mod_pi_rad=float(abs((pred_yaw-truth_yaw+np.pi/2)%np.pi-np.pi/2)) if row["valid"] else None,
            geometry=row.get("geometry_agreement"), hypotheses=row.get("diagnostic_hypotheses"))
    (out/f"{label}_observation.json").write_text(json.dumps(session.decision_observation,indent=2))
    print(json.dumps(dict(label=label,objects={p:dict(valid=r["valid"],error_m=r["error_m"]) for p,r in record["objects"].items()})),flush=True)
    return record


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",default="results/wrist_camera_dev")
    parser.add_argument("--recorded-state")
    args=parser.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for installed,seed in [(False,1600),(False,1601),(False,1603),(False,1604),(True,1620)]:
        label=f"{seed}_installed_{installed}"
        _,session,_,_=stage_v12.make_scene(seed,out/label,preinstalled_end_stop=installed)
        rows.append(audit(session,label,out))
    label="CAD_assembled_handle_fixture_visibility_only"
    _,session,_,_=stage_v12.make_scene(1600,out/label,preinstalled_end_stop=True)
    center=np.asarray(session.stage_targets["carriage"],float)
    session.ctx.set_obj_pose("carriage",center[:2],center[2],yaw=0.)
    handle=center+np.array([0.,0.,.048])
    session.ctx.set_obj_pose("handle",handle[:2],handle[2],yaw=0.)
    for _ in range(30): session.ctx.step()
    record=audit(session,label,out)
    record["evidence_scope"]="CAD assembly initial fixture for sensor visibility, not execution success evidence"
    rows.append(record)
    if args.recorded_state:
        label="recorded_1604_installation_visibility_only"
        _,session,_,_=stage_v12.make_scene(1604,out/label)
        state=np.load(args.recorded_state)
        for key in ("qpos","qvel","ctrl"):
            getattr(session.ctx.data,key)[:]=state[key]
        session.ctx.data.time=float(state["time"])
        session.ctx.model.geom_friction[:]=state["geom_friction"]
        mujoco.mj_forward(session.ctx.model,session.ctx.data)
        # Empty-hand acquisition collision-checks and executes HOME if needed.
        record=audit(session,label,out)
        record["evidence_scope"]="sensor visibility from recorded state in a new clone; not execution success evidence"
        rows.append(record)
    (out/"audit.json").write_text(json.dumps(rows,indent=2))


if __name__=="__main__": main()
