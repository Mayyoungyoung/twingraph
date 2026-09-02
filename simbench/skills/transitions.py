"""Transition skills: smooth connections between exec and plan skills.

Each transition is a first-class, composable skill extracted from the
inline recipes of grasp/place (retreat, pre-align, approach) or the
arm controller (home).  They keep the measured-stable loop parameters
of the inline originals, so chaining them reproduces the legacy
behavior while exposing the seams for the planner:

  plan_grasp_pose -> approach -> grasp(approach_direct=True)
  place/release    -> retreat_lift -> return_home
  carry hover      -> pre_align -> place(descend only)
"""
import numpy as np

from . import base
from .base import SkillResult, register
from .motion import move_eef, home as _motion_home
from .settle import settle


# ----------------------------------------------------------- retreat_lift
@register(category=base.CAT_TRANS,
          description="放置/释放后垂直抬升退避：从当前 EEF 位置爬升指定"
                      "高度并静置，避免 pads 扫掠已就位零件。",
          inputs={"height": "抬升高度 (m, 默认 0.07)",
                  "tol": "抬升到位容差 (m)",
                  "settle_steps": "抬升后静置步数"},
          outputs={"lift_m": "实测爬升量 (m)"},
          preconditions=["场景已加载，机器人可控"],
          postconditions=["EEF 爬升量 >= height（skill 以 lift_m 度量判定）"],
          failure_policy="continue", impl=base.IMPL_SCRIPT,
          deps=["motion.move_eef", "settle"])
def retreat_lift(ctx, arm, gripper, height=0.07, tol=0.012, gain=10.0,
                 settle_steps=10, verbose=False):
    """Lift the EEF vertically by ``height`` (release-clear transition).

    Postcondition: the EEF sits ``height`` above its entry point, so
    the next waypoint move cannot sweep pads across the released part.
    """
    start = ctx.eef_pos()
    ok = arm.move_eef(start + np.array([0.0, 0.0, height]),
                      gain=gain, tol=tol, smooth=True)
    settle(ctx, max_steps=settle_steps)
    return SkillResult(ok=bool(ok),
                       metrics=dict(lift_m=float(ctx.eef_pos()[2]
                                                 - start[2])))


# ------------------------------------------------------------- return_home
@register(category=base.CAT_TRANS,
          description="回初始位：开爪（可选）+ 关节空间回 HOME。可选先"
                      "爬升到安全高度再回扫，规避关节回扫扫飞已装配件"
                      "（Task B 实测教训）。",
          inputs={"open_gripper": "回位前是否开爪",
                  "retreat_z": "可选：先平移到该绝对 z 高度再回扫"},
          outputs={"eef": "回位后 EEF 位置"},
          preconditions=["场景已加载"],
          postconditions=[],
          failure_policy="continue", impl=base.IMPL_SCRIPT,
          granularity=base.GRAN_COMPOSITE,
          decomposes=["grip_open(open_gripper=True 时)", "move(可选抬升)",
                      "关节回扫 HOME"],
          deps=["Gripper.open", "CartesianController.home",
                "motion.move_eef"])
def return_home(ctx, arm, gripper, open_gripper=True, retreat_z=None,
                verbose=False):
    """Open the gripper and servo the arm back to HOME.

    ``retreat_z`` (absolute world z) parks the EEF high above the
    assembly before the joint-space sweep -- the raw sweep issued at a
    low pose knocks seated parts off the rack (Task B wrap-up lesson).
    """
    if open_gripper and gripper is not None:
        gripper.open()
    if retreat_z is not None:
        e = ctx.eef_pos()
        move_eef(arm, np.array([e[0], e[1], float(retreat_z)]),
                 style="direct", tol=0.012)
    ok = _motion_home(arm)
    return SkillResult(ok=bool(ok),
                       metrics=dict(eef_z=float(ctx.eef_pos()[2])))


# --------------------------------------------------------------- pre_align
@register(category=base.CAT_TRANS,
          description="下降前预对齐：在当前位置高度上，将持件（或空爪 "
                      "EEF）对中到目标 xy，供后续垂直下降/放置直接"
                      "衔接（gain/迭代沿用 place 实测配方）。",
          inputs={"at": "目标 xy (world)",
                  "part": "可选：被持零件名（None=对齐 EEF 本身）",
                  "tol": "收敛容差 (m)",
                  "max_iters": "比例纠偏最大迭代数"},
          outputs={"resid": "收敛后的 xy 残差 (m)"},
          preconditions=["场景已加载"],
          postconditions=[],
          failure_policy="continue", impl=base.IMPL_SCRIPT,
          deps=["CartesianController.move_eef"])
def pre_align(ctx, arm, gripper, at, part=None, tol=0.0015, gain=6.0,
              max_iters=15, verbose=False):
    """Align a held part (or the EEF itself) over ``at`` at height.

    The proportional nudge loop of place/align, generalized: each
    iteration measures the live error and moves the EEF against it
    (one control step at a time), so a light part sliding inside the
    pads converges instead of being dragged.
    """
    target_xy = np.asarray(at, dtype=float)[:2]
    best = 9.9
    for _ in range(max_iters):
        ref = ctx.obj_pos(part)[:2] if part else ctx.eef_pos()[:2]
        err = ref - target_xy
        best = min(best, float(np.linalg.norm(err)))
        if np.linalg.norm(err) < tol:
            break
        e2 = ctx.eef_pos()
        arm.move_eef(e2 - np.array([err[0], err[1], 0.0]),
                     gain=gain, tol=0.0005, max_steps=10, stall=False)
    ok = best < 2.0 * tol
    if verbose:
        print(f"  [pre_align] resid={best * 1000:.2f}mm "
              f"{'ok' if ok else 'FAIL'}")
    return SkillResult(ok=bool(ok), metrics=dict(resid=best))


# ---------------------------------------------------------------- approach
@register(category=base.CAT_TRANS,
          description="抓取前接近：先到抓取点上方悬停，再垂直下降至抓取"
                      "高度（不闭合）。与 grasp(approach_direct=True) "
                      "组合即标准两段式抓取接近，避免 grasp 内部 "
                      "safe_z 二次抬升的下降-上升-抽搐。",
          inputs={"part": "目标零件名",
                  "grasp_pos": "可选：预计算抓取点（plan_grasp_pose "
                               "工件），缺省时现场估计",
                  "hover_lift": "悬停高度 (m)",
                  "tol": "下降容差 (m)"},
          outputs={"eef_z": "下降后 EEF 高度"},
          preconditions=["零件存在"],
          postconditions=[],
          failure_policy="continue", impl=base.IMPL_SCRIPT,
          granularity=base.GRAN_COMPOSITE,
          decomposes=["move(悬停) -> descend(精降)"],
          deps=["perception.estimate_grasp_pose", "motion.move_eef"])
def approach(ctx, arm, gripper, part, grasp_pos=None, hover_lift=0.07,
             tol=0.004, gain=6.0, verbose=False):
    """Hover above the grasp point and descend to it (pre-grasp leg).

    The descent is stall-tolerant (same convention as grasp): thin
    parts may stop the EEF slightly above the nominal target and the
    pads are still close enough to close.
    """
    if grasp_pos is None:
        from .perception import estimate_grasp_pose
        gp = estimate_grasp_pose(ctx, part)
        if gp is None:
            return SkillResult(ok=False,
                               reason=f"no grasp pose for '{part}'",
                               metrics=dict())
        grasp_pos = gp["pos"]
    target = np.asarray(grasp_pos, dtype=float)
    if not move_eef(arm, target + np.array([0.0, 0.0, hover_lift]),
                    smooth=True):
        return SkillResult(ok=False, reason="hover move failed",
                           metrics=dict())
    arm.move_eef(target, gain=gain, tol=tol)
    if verbose:
        eef = ctx.eef_pos()
        print(f"  [approach {part}] eef {np.round(eef, 4)} "
              f"(target {np.round(target, 4)})")
    return SkillResult(ok=True,
                       metrics=dict(eef_z=float(ctx.eef_pos()[2])))
