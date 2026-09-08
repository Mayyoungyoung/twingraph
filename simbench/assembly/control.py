"""Full-pose IK in scratch data and bounded smooth servo execution.

Planning never writes the live robot state. Live execution writes only the
Panda actuator commands. Parts are never welded, teleported or driven.
"""

import numpy as np
import mujoco
from pathlib import Path
from scipy.spatial.transform import Rotation
from ..core.sim_context import MjContext
from .scene import SCENE

HOME = np.array([0.0, -0.55, 0.0, -2.25, 0.0, 1.70, 0.7854])
DEFAULT_POLICY = Path(__file__).resolve().parent / "checkpoints" / "insert_bc.pt"


def down(yaw=0.0):
    # local finger closing axis is x; yaw=0 closes along world y.
    return Rotation.from_euler("z", yaw).as_matrix() @ np.array(
        [[0.0, 1, 0], [1, 0, 0], [0, 0, -1]]
    )


def make_context(seed=0):
    np.random.seed(seed)
    ctx = MjContext(SCENE, control_freq=50)
    ctx.reset()
    ctx.data.qpos[ctx.arm_qadr] = HOME
    mujoco.mj_forward(ctx.model, ctx.data)
    ctx.hold_arm()
    ctx.set_finger_ctrl(0.04)
    for _ in range(80):
        ctx.step()
    return ctx


class PoseController:
    def __init__(self, ctx):
        self.ctx = ctx
        self.scratch = mujoco.MjData(ctx.model)
        self.dofs = [int(ctx.model.jnt_dofadr[j]) for j in ctx.arm_joint_ids]
        self.limits = ctx.model.jnt_range[ctx.arm_joint_ids].copy()
        self.rotation = down()
        self.trace = []

    def ik(self, xyz, rotation=None, seed=None, max_iter=160):
        ctx = self.ctx
        d = self.scratch
        m = ctx.model
        d.qpos[:] = ctx.data.qpos
        d.qpos[ctx.arm_qadr] = ctx.arm_qpos if seed is None else seed
        goal = np.asarray(xyz)
        R = self.rotation if rotation is None else rotation
        jp = np.zeros((3, m.nv))
        jr = np.zeros((3, m.nv))
        for _ in range(max_iter):
            mujoco.mj_forward(m, d)
            ep = goal - d.site_xpos[ctx.eef_site_id]
            er = Rotation.from_matrix(
                R @ d.site_xmat[ctx.eef_site_id].reshape(3, 3).T
            ).as_rotvec()
            if np.linalg.norm(ep) < 0.00012 and np.linalg.norm(er) < 0.002:
                return d.qpos[ctx.arm_qadr].copy()
            mujoco.mj_jacSite(m, d, jp, jr, ctx.eef_site_id)
            J = np.vstack([jp[:, self.dofs], jr[:, self.dofs] * 0.35])
            error = np.r_[ep, er * 0.35]
            dq = J.T @ np.linalg.solve(J @ J.T + np.eye(6) * 0.00006, error)
            dq *= min(1.0, 0.12 / max(np.max(np.abs(dq)), 1e-9))
            d.qpos[ctx.arm_qadr] = np.clip(
                d.qpos[ctx.arm_qadr] + dq,
                self.limits[:, 0] + 0.02,
                self.limits[:, 1] - 0.02,
            )
        if seed is None:
            return self.ik(goal, R, seed=HOME, max_iter=max_iter)
        raise ValueError(
            f"IK unreachable: xyz={goal.tolist()}, residual={np.linalg.norm(ep):.5f}"
        )

    def check_joint_path(self, joints, held=None, step=0.035):
        """Discrete robot/environment and carried-part clearance check in scratch data."""
        ctx = self.ctx
        d = self.scratch
        m = ctx.model
        d.qpos[:] = ctx.data.qpos
        q0 = ctx.arm_qpos.copy()
        samples = 0
        robot = {
            i
            for i in range(m.nbody)
            if m.body(i).name.startswith(("link", "finger"))
            or m.body(i).name
            in ("leftfinger", "rightfinger", "right_hand", "right_gripper", "eef")
        }
        held_id = ctx.body_id(held) if held else -1
        if held:
            jid = int(m.body_jntadr[held_id])
            qa = int(m.jnt_qposadr[jid])
            R0 = ctx.eef_mat()
            local = R0.T @ (ctx.obj_pos(held) - ctx.eef_pos())
            local_R = R0.T @ ctx.data.xmat[held_id].reshape(3, 3)
        for q1 in joints:
            for t in np.linspace(0, 1, max(2, int(np.max(np.abs(q1 - q0)) / step) + 1)):
                d.qpos[ctx.arm_qadr] = q0 + t * (q1 - q0)
                mujoco.mj_forward(m, d)
                if held:
                    R = d.site_xmat[ctx.eef_site_id].reshape(3, 3)
                    d.qpos[qa : qa + 3] = d.site_xpos[ctx.eef_site_id] + R @ local
                    quat = Rotation.from_matrix(R @ local_R).as_quat()
                    d.qpos[qa + 3 : qa + 7] = quat[[3, 0, 1, 2]]
                    mujoco.mj_forward(m, d)
                samples += 1
                for c in d.contact:
                    b1, b2 = map(int, m.geom_bodyid[[c.geom1, c.geom2]])
                    if c.dist >= -0.0008 or (
                        b1 not in robot and b2 not in robot and held_id not in (b1, b2)
                    ):
                        continue
                    if held_id in (b1, b2):
                        other = b2 if b1 == held_id else b1
                        if m.body(other).name in (
                            "finger_joint1_tip",
                            "finger_joint2_tip",
                            "leftfinger",
                            "rightfinger",
                        ):
                            continue  # intended carried-object / finger contact only
                    return dict(
                        valid=False,
                        samples=samples,
                        pair=[m.geom(c.geom1).name, m.geom(c.geom2).name],
                        penetration_m=float(-c.dist),
                    )
            q0 = q1
        return dict(valid=True, samples=samples, resolution_rad=step)

    def execute_joint(self, qgoal, duration=None):
        ctx = self.ctx
        q0 = ctx.arm_qpos.copy()
        duration = (
            max(1.0, float(np.max(np.abs(qgoal - q0))) * 2.5)
            if duration is None
            else duration
        )
        for t in np.linspace(0, 1, max(2, int(duration / ctx.control_dt))):
            s = t * t * t * (10 - 15 * t + 6 * t * t)
            ctx.set_arm_ctrl(q0 + s * (qgoal - q0))
            ctx.step()
        for _ in range(30):
            ctx.set_arm_ctrl(qgoal)
            ctx.step()
        return float(np.max(np.abs(ctx.arm_qpos - qgoal))) < 0.012

    def move(self, xyz, rotation=None, speed=0.10, tol=0.001, linear=True):
        ctx = self.ctx
        R = self.rotation if rotation is None else rotation
        goal = np.asarray(xyz, float)
        if not linear:
            ok = self.execute_joint(self.ik(goal, R))
        else:
            start = ctx.eef_pos()
            R0 = ctx.eef_mat()
            rot = Rotation.from_matrix(R @ R0.T).as_rotvec()
            duration = max(
                0.5,
                np.linalg.norm(goal - start) * 1.9 / speed,
                np.linalg.norm(rot) * 1.8 / 0.5,
            )
            q = ctx.arm_qpos.copy()
            for t in np.linspace(0, 1, max(2, int(duration / ctx.control_dt))):
                s = t * t * t * (10 - 15 * t + 6 * t * t)
                q = self.ik(
                    start + s * (goal - start),
                    Rotation.from_rotvec(s * rot).as_matrix() @ R0,
                    seed=q,
                    max_iter=35,
                )
                ctx.set_arm_ctrl(q)
                ctx.step()
            for _ in range(25):
                q = self.ik(goal, R, seed=q, max_iter=25)
                ctx.set_arm_ctrl(q)
                ctx.step()
            ok = np.linalg.norm(ctx.eef_pos() - goal) < tol
        self.rotation = R.copy()
        return bool(ok)

    def servo(self, xyz, rotation=None):
        q = self.ik(xyz, self.rotation if rotation is None else rotation, max_iter=35)
        self.ctx.set_arm_ctrl(q)
        self.ctx.step()

    def open(self):
        start = float(self.ctx.data.ctrl[self.ctx.finger_act_ids[0]])
        for q in np.linspace(start, 0.04, 35):
            self.ctx.set_finger_ctrl(q)
            self.ctx.step()
        for _ in range(15):
            self.ctx.step()
        return self.ctx.pad_span() > 0.075

    def close(self, part, force=3.0):
        ctx = self.ctx
        q = float(ctx.data.ctrl[ctx.finger_act_ids[0]])
        for _ in range(400):
            measured = ctx.grasp_contacts(part)
            if (
                measured["held"]
                and min(measured["left_n"], measured["right_n"]) >= force
            ):
                break
            q = max(0.0, q - 0.00016)
            ctx.set_finger_ctrl(q)
            ctx.step()
            if q == 0:
                break
        for _ in range(20):
            ctx.step()
        return ctx.grasp_contacts(part)

    def part_target(self, part, xyz):
        return np.asarray(xyz) + self.ctx.eef_pos() - self.ctx.obj_pos(part)

    def carry(self, part, xyz, speed=0.055):
        return self.move(self.part_target(part, xyz), speed=speed)
