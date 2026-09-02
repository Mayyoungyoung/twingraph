"""Extension skills: peg insertion, wipe, pull (and push re-registered).

These are the hardest primitives for rule-based scripting -- contact-
rich, tolerance-sensitive motion.  Each ships with a closed-loop
scripted implementation; ``peg_insert`` additionally accepts a learned
policy (imitation / RL, see simbench.skills.learned) that replaces the
scripted control loop while keeping the same skill interface.
"""
import os

import numpy as np

from . import base
from . import manipulation
from .base import SkillResult, register
from .motion import move_eef
from .settle import settle

# default trained insertion-policy checkpoint (produced by
# simbench.skills.learned.train_insert); peg_insert(mode='policy')
# auto-loads it when no policy object is passed.
DEFAULT_INSERT_POLICY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "learned",
    "checkpoints", "peg_insert.pt")


# -------------------------------------------------------------- peg_insert
@register(category="ext",
          description="插销：把持件插入指定孔口。mode='press' 为到点压入，"
                      "mode='thread' 为底部导向螺旋下降（抗卡摆动，规则闭环），"
                      "mode='policy' 由学习策略（模仿学习/强化学习）输出插入"
                      "动作序列，输入孔心 xy 与孔底 z，奖励显式考虑接触、"
                      "卡滞与姿态偏差。",
          inputs={"name": "被插零件名（已由 grasp 抓持）",
                  "hole_xy": "孔心 world xy",
                  "to_z": "零件底面目标 z（孔底/座面）",
                  "mode": "press | thread | policy",
                  "policy": "可选：学习策略对象（mode='policy' 时必填）",
                  "max_steps": "策略/闭环最大步数"},
          outputs={"depth_m": "实测插入深度 (m)",
                   "bot_z": "终态零件底面 z"},
          preconditions=[base.held_part],
          postconditions=["零件底面到达 to_z 附近（以 depth_m 判定）"],
          failure_policy="abort",
          impl=f"{base.IMPL_SCRIPT}/{base.IMPL_RL}/{base.IMPL_IL}",
          granularity=base.GRAN_COMPOSITE,
          decomposes=["grasp", "insert(thread/press) 或 policy(RL/IL 逐步循环)",
                      "release"],
          deps=["grasp", "manipulation.insert",
                "skills.learned.insert_env",
                "skills.learned.load_policy (mode=policy)"])
def peg_insert(ctx, arm, gripper, name, hole_xy, to_z, mode="thread",
               policy=None, policy_path=None, half=None, press_steps=15,
               max_steps=300, seat_tol=0.0015, verbose=False):
    """Insert the held part *name* into the hole at (hole_xy, to_z).

    - mode='press': descend to a target EEF point and hold (press fit).
    - mode='thread': the bottom-steered threaded descent with anti-jam
      wiggle (the measured-stable scripted recipe).
    - mode='policy': a learned policy drives the EEF step-by-step from
      the current state; the policy is trained to "insert the peg into
      the given xyz hole" with contact / jam / tilt feedback (see
      simbench.skills.learned).

    Postcondition: the part bottom reaches ``to_z`` (within a small
    tolerance); the result carries the achieved depth.
    """
    from . import perception
    if half is None:
        half = perception.part_meta(ctx, name)["half_h"]
    if mode == "policy":
        from . import learned
        from .learned import insert_env
        if policy is None:
            ckpt = policy_path or DEFAULT_INSERT_POLICY
            if not os.path.exists(ckpt):
                return SkillResult(
                    ok=False, metrics=dict(),
                    reason=f"mode='policy' needs a trained checkpoint at "
                           f"{ckpt} (run python -m simbench.skills."
                           f"learned.train_insert) or pass policy=/"
                           f"policy_path=")
            policy = learned.load_policy(ckpt)
        hole_xy = np.asarray(hole_xy, dtype=float)[:2]
        z0 = ctx.obj_pos(name)[2] - half
        bot_z = z0
        for _ in range(int(max_steps)):
            obs = insert_env.build_obs(ctx, name, hole_xy, to_z, half)
            act = policy.act(obs, deterministic=True)
            d = insert_env.action_to_delta(act)
            eef = ctx.eef_pos()
            arm.move_eef(eef + d, gain=8.0, tol=0.0, max_steps=1,
                         stall=False)
            bot_z = ctx.obj_pos(name)[2] - half
            if bot_z <= to_z + seat_tol:
                break
        settle(ctx, max_steps=10)
        bot_z = ctx.obj_pos(name)[2] - half
        depth = z0 - bot_z
        lat = float(np.linalg.norm(ctx.obj_pos(name)[:2] - hole_xy))
        ok = bot_z <= to_z + max(seat_tol, 0.002) and depth > 0.0
        if verbose:
            print(f"  [peg_insert policy {name}] depth={depth*1000:.2f}mm "
                  f"bot_z={bot_z:.4f} to_z={to_z:.4f} "
                  f"lat={lat*1000:.2f}mm {'ok' if ok else 'FAIL'}")
        return SkillResult(
            ok=bool(ok),
            metrics=dict(depth_m=float(max(depth, 0.0)),
                         bot_z=float(bot_z), lateral=lat))
    if mode == "press":
        eef = ctx.eef_pos()
        off = eef - ctx.obj_pos(name)
        target_eef = np.array([hole_xy[0] + off[0],
                               hole_xy[1] + off[1],
                               to_z + half + off[2]])
        z0 = ctx.obj_pos(name)[2] - half
        ok = manipulation.insert(ctx, arm, gripper, name=name,
                                 target_eef=target_eef, mode="press",
                                 press_steps=press_steps, verbose=verbose)
    elif mode == "thread":
        z0 = ctx.obj_pos(name)[2] - half
        ok = manipulation.insert(ctx, arm, gripper, name=name,
                                 mode="thread", ref_axis=hole_xy,
                                 to_z=to_z, half=half,
                                 max_steps=max_steps, verbose=verbose)
    else:
        raise ValueError(f"unknown peg_insert mode {mode!r}")
    bot_z_now = ctx.obj_pos(name)[2] - half
    depth = z0 - bot_z_now
    return SkillResult(ok=bool(ok),
                       metrics=dict(depth_m=float(max(depth, 0.0)),
                                    bot_z=float(bot_z_now)))


# -------------------------------------------------------------------- pull
@register(category="ext",
          description="拉动：抓持零件后沿指定方向向机器人侧拖拽（与 push "
                      "反向的闭环位移技能）。若未持件则先轻咬合抓取并验证"
                      "举升；拖动全程监控零件随动，滑脱即停。",
          inputs={"name": "被拉零件名",
                  "delta": "位移向量 (dx, dy)",
                  "to_target": "可选：绝对目标 xy（优先于 delta）",
                  "release": "结束后是否原位释放",
                  "speed": "拖拽速度 (m/s)"},
          outputs={"moved_m": "零件沿方向实测位移 (m)"},
          preconditions=["零件存在"],
          postconditions=["位移 >= 70% 请求量（以 moved_m 判定）"],
          failure_policy="continue", impl=base.IMPL_SCRIPT,
          deps=["grasp", "manipulation._drive_along", "Gripper"])
def pull(ctx, arm, gripper, name, delta=None, to_target=None,
         press=0.0015, speed=0.04, release=True, settle_steps=15,
         verbose=False):
    """Drag *name* along a direction (pull toward the robot).

    Exactly one of ``delta`` / ``to_target`` selects the displacement
    (to_target wins).  A not-yet-held part is grasped first with a
    lift-verified light bite; the drag is closed-loop on the live part
    (quits on stall -- a jam or a lost grip), then optionally released
    in place.
    """
    if to_target is not None:
        delta = (np.asarray(to_target, dtype=float)[:2]
                 - ctx.obj_pos(name)[:2])
    if delta is None:
        return SkillResult(ok=False, reason="pull requires delta "
                                            "or to_target",
                           metrics=dict())
    delta = np.asarray(delta, dtype=float)[:2]
    dist = float(np.linalg.norm(delta))
    if dist < 1e-6:
        return SkillResult(ok=True, metrics=dict(moved_m=0.0))
    d = delta / dist
    held = float(np.linalg.norm(ctx.eef_pos()[:2]
                                - ctx.obj_pos(name)[:2])) < 0.03
    if not held:
        ok = manipulation.grasp(ctx, arm, gripper, name, press=press,
                                verbose=verbose)
        if not ok:
            return SkillResult(ok=False, reason=f"grasp {name} failed "
                                                f"before pull",
                               metrics=dict())
    prog = manipulation._drive_along(ctx, arm, name, d, dist,
                                     speed=speed, tag="pull",
                                     verbose=verbose)
    if release:
        gripper.hold()
        settle(ctx, max_steps=10)
        gripper.open()
        settle(ctx, max_steps=settle_steps)
        move_eef(arm, ctx.eef_pos()
                 + np.array([0.0, 0.0, manipulation.DEFAULT_LIFT]),
                 tol=0.012, smooth=True)
        settle(ctx, max_steps=settle_steps)
    ok = prog >= 0.7 * dist
    if verbose:
        print(f"  [pull {name}] moved {prog * 1000:.1f}mm of "
              f"{dist * 1000:.1f}mm {'ok' if ok else 'FAIL'}")
    return SkillResult(ok=bool(ok), metrics=dict(moved_m=float(prog)))


# -------------------------------------------------------------------- wipe
@register(category="ext",
          description="擦拭：并指成刃，轻压贴面后在目标面上沿直线扫掠，"
                      "全程监控 pad 接触力处于目标力带（既保证贴实又不"
                      "压穿/挑翻被擦面），可作触垫预清洁等语义。",
          inputs={"at": "擦拭区域中心 xy",
                  "direction": "扫掠方向单位向量",
                  "length": "扫掠长度 (m)",
                  "z": "EEF 扫掠高度 (world z，通常=面高+pad 偏移)",
                  "f_band": "(f_lo, f_hi) 目标接触力带 (N)"},
          outputs={"force_mean": "扫掠全程平均接触力 (N)",
                   "force_max": "扫掠峰值接触力 (N)",
                   "swept_m": "实测扫掠位移 (m)"},
          preconditions=["场景已加载"],
          postconditions=["接触力处于力带且扫掠完成（以 force_max 判定）"],
          failure_policy="continue", impl=base.IMPL_SCRIPT,
          deps=["Gripper.close_to_span", "motion.move_eef"])
def wipe(ctx, arm, gripper, at, direction, length, z=None,
         f_band=(0.05, 2.0), speed=0.05, max_steps=600, verbose=False):
    """Wipe a surface with the closed pads (blade wipe).

    The pads close into a narrow blade, press lightly onto the target
    surface (``z`` = EEF sweep height) and sweep along ``direction``
    for ``length``.  The pad contact force is monitored every step:
    the sweep aborts when the force leaves ``f_band`` (either the
    blade lifted off or dug in).  Returns the force profile stats.
    """
    at = np.asarray(at, dtype=float)[:2]
    d = np.asarray(direction, dtype=float)[:2]
    d = d / np.linalg.norm(d)
    z = float(z) if z is not None else ctx.eef_pos()[2]

    def _pad_force():
        return (ctx.geom_contact_force("finger1_pad_collision")
                + ctx.geom_contact_force("finger2_pad_collision"))

    gripper.close_to_span(0.012)          # fingers together = blade
    start = at - d * (length / 2.0)
    end = at + d * (length / 2.0)
    if not move_eef(arm, np.array([start[0], start[1], z + 0.04])):
        return SkillResult(ok=False, reason="wipe hover failed",
                           metrics=dict())
    arm.move_eef(np.array([start[0], start[1], z]), tol=0.004)
    settle(ctx, max_steps=6)
    forces = []
    best, stall = 0.0, 0
    swept = 0.0
    f_lo, f_hi = f_band
    for _ in range(max_steps):
        f = _pad_force()
        forces.append(float(f))
        if f > f_hi:
            # dug in: lift the blade 1mm and stop pressing deeper
            e = ctx.eef_pos()
            arm.move_eef(np.array([e[0], e[1], e[2] + 0.001]),
                         gain=8.0, tol=0.0, max_steps=1, stall=False)
        eef = ctx.eef_pos()
        arm.move_eef(np.array([end[0], end[1], z]), gain=6.0,
                     tol=0.0, max_steps=1, stall=False, max_speed=speed)
        prog = float((ctx.eef_pos()[:2] - start) @ d)
        if prog > best + 1e-5:
            best, stall = prog, 0
        else:
            stall += 1
            if stall >= 40:
                break
        if prog >= length:
            break
        swept = max(swept, prog)
    force_mean = float(np.mean(forces)) if forces else 0.0
    force_max = float(max(forces)) if forces else 0.0
    ok = force_max <= f_hi and swept >= 0.7 * length
    if verbose:
        print(f"  [wipe] swept={swept * 1000:.1f}mm "
              f"F_mean={force_mean:.2f}N F_max={force_max:.2f}N "
              f"{'ok' if ok else 'FAIL'}")
    move_eef(arm, ctx.eef_pos() + np.array([0.0, 0.0, 0.06]),
             tol=0.012, smooth=True)
    return SkillResult(ok=bool(ok), metrics=dict(
        force_mean=force_mean, force_max=force_max, swept_m=swept))
