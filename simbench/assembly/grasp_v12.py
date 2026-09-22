"""Continuous grasp/withdrawal feasibility, using vision for part attachment.

Environment collision geometry remains the declared digital-twin adapter.
The real holder is tested at every withdrawal sample, never exempted.
"""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from .control import down


def continuation(arm, start, goal, rotation, seed, spacing=.0025):
    points=np.linspace(start,goal,max(2,int(np.ceil(np.linalg.norm(goal-start)/spacing))+1))
    q=np.asarray(seed,float).copy(); joints=[]
    for point in points[1:]:
        q=arm.ik(point,rotation,seed=q,max_iter=160)
        joints.append(q.copy())
    return joints


def withdrawal_slice(grasp, height):
    """Consume the same checked branch across one or more declared lifts."""
    previous=float(grasp.get("withdrawal_progress_m",0.))
    limit=float(grasp["withdrawal_height_m"])
    end=previous+float(height)
    if height <= 0 or end > limit+1e-8:
        raise ValueError("requested lift exceeds its bound withdrawal branch")
    knots=np.vstack([grasp["approach_joints_v12"][-1],grasp["lift_joints_v12"]])
    coordinates=np.linspace(0.,limit,len(knots))
    path=[knots[i].copy() for i in range(1,len(knots))
          if previous+1e-9 < coordinates[i] < end-1e-9]
    i=min(int(np.searchsorted(coordinates,end,side="left")),len(knots)-1)
    alpha=(end-coordinates[i-1])/(coordinates[i]-coordinates[i-1])
    path.append(knots[i-1]+alpha*(knots[i]-knots[i-1]))
    return path,end


def plan_grasp(session, part, pose, xyz, yaw, width):
    arm,ctx=session.arm,session.ctx
    R=down(yaw); hover=np.asarray(xyz)+[0,0,.10]
    quat=np.asarray(pose["quat"],float)
    object_R=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
    axis=object_R[:,2]
    if axis[2]<0: axis=-axis
    # Pins leave their real holder along the observed shaft axis.
    lift_axis=axis if part.startswith("pin_") else np.array([0.,0.,1.])
    failures=[]
    for restart,seed in enumerate(arm.restart_seeds(24)):
        try:
            qhover=arm.ik(hover,R,seed=seed)
            reach=arm.check_joint_path([qhover])
            if not reach["valid"]:
                failures.append(dict(restart=restart,phase="reach_hover",check=reach)); continue
            approach=continuation(arm,hover,np.asarray(xyz),R,qhover)
            approach_check=arm.check_joint_path(approach,start_q=qhover)
            if not approach_check["valid"]:
                failures.append(dict(restart=restart,phase="approach",check=approach_check)); continue
            qgrasp=approach[-1]
            # FK plus the visual pose supplies the hypothetical attachment.
            arm.scratch.qpos[:]=ctx.data.qpos
            arm.scratch.qpos[ctx.arm_qadr]=qgrasp
            mujoco.mj_forward(ctx.model,arm.scratch)
            eef=arm.scratch.site_xpos[ctx.eef_site_id].copy()
            hand_R=arm.scratch.site_xmat[ctx.eef_site_id].reshape(3,3).copy()
            attachment=dict(position=hand_R.T@(np.asarray(pose["xyz"])-eef),
                            rotation=hand_R.T@object_R)
            lift_goal=np.asarray(xyz)+lift_axis*.10
            lifted=continuation(arm,np.asarray(xyz),lift_goal,R,qgrasp)
            lift_check=arm.check_joint_path(lifted,held=part,start_q=qgrasp,
                held_transform=attachment,finger_gap=width/2,step=.02)
            if not lift_check["valid"]:
                failures.append(dict(restart=restart,phase="withdrawal",check=lift_check)); continue
            return dict(q_hover=qhover,approach_joints_v12=approach,lift_joints_v12=lifted,
                lift_axis_v12=lift_axis,withdrawal_height_m=.10,
                continuous_grasp_check=dict(ik_restart=restart,reach=reach,approach=approach_check,
                    withdrawal=lift_check,attachment_source="RGB-D pose and planned encoder FK",
                    collision_geometry="digital twin environment; holder not exempted"))
        except ValueError as exc:
            failures.append(dict(restart=restart,phase="IK continuation",reason=str(exc)))
    raise ValueError(f"continuous grasp unresolved after {len(failures)} branch attempts: {failures[-3:]}")
