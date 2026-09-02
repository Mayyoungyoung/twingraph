"""Skill-library registrations: execution + planning atomic skills.

This module gives the *execution* and *planning* primitives that already
live in :mod:`simbench.skills.perception` / :mod:`.planning` /
:mod:`.motion` / :mod:`.manipulation` a first-class contract in the
:class:`~simbench.skills.base.SkillRegistry`, so that:

  - every skill declares its category, inputs/outputs, pre/post
    conditions, failure policy, implementation method and dependencies
    (the ``SkillSpec`` -- the skill-inventory deliverable);
  - skills run through one gated entry point (``REGISTRY.run``) that
    checks preconditions BEFORE executing and verifies postconditions
    AFTER, returning a uniform :class:`~simbench.skills.base.SkillResult`;
  - skills compose: ``run_chain([...])`` sequences them, and the
    plan->exec artifact handshake (``plan_grasp_pose`` emits a grasp
    pose that ``grasp`` consumes) is preserved.

The wrappers are thin: they adapt the conventional
``fn(ctx, arm, gripper, **params)`` signature onto the existing
implementations and never duplicate the measured-stable control recipes.
The executor keeps its own richer handlers (with semantic-landmark
resolution) and takes dispatch precedence; these registrations are the
contract / inventory layer plus a generic dispatch path.

Transition skills live in :mod:`.transitions`; extension skills (peg
insertion / wipe / pull) in :mod:`.extension`.
"""
import numpy as np

from . import base
from . import perception
from . import planning
from . import manipulation
from .motion import move_eef
from .settle import settle
from .base import SkillResult, register

_IMPL = base.IMPL_SCRIPT
_MOTION = base.IMPL_MOTION
_OPT = base.IMPL_OPT


def _part(params_part=None, name=None):
    """Canonical part-name resolution (accept ``part`` or ``name``)."""
    return params_part if params_part is not None else name


# =====================================================================
# EXECUTION SKILLS  (category="exec": they drive the physics)
# =====================================================================

@register(name="detect", category=base.CAT_EXEC,
          description="零件检测：在失败模型下感知零件位姿（位置+偏航），"
                      "可注入高斯噪声/漏检/误检，输出观测供 plan_grasp_pose "
                      "消费。是执行链的感知入口。",
          inputs={"part": "零件名（body）",
                  "faults": "可选 FailureModel（漏检/误检/噪声源）",
                  "step": "可选步序号（供故障模型归因）"},
          outputs={"found": "是否检测到", "pos": "world xyz 观测",
                   "yaw": "偏航观测 (rad)", "outcome": "ok|miss|false"},
          preconditions=[base.part_exists],
          postconditions=["检测到则 outcome!=miss 且 pos 非空"],
          failure_policy="retry", impl=_IMPL,
          deps=["perception.detect_part", "faults.FailureModel(可选)"])
def detect(ctx, arm, gripper, part=None, name=None, faults=None, step=None,
           verbose=False):
    """Detect a part under the failure model (single perception).

    One-shot by design: a missed/false outcome is REPORTED (outcome in
    metrics), and retries belong to the composition layer (failure_policy
    = "retry" lets run_chain / the planner re-invoke the atom).
    """
    pname = _part(part, name)
    found, pos, yaw, outcome = perception.detect_part(
        ctx, pname, faults=faults, step=step)
    metrics = dict(found=bool(found), outcome=outcome)
    if found:
        metrics["pos"] = np.asarray(pos, dtype=float)
        metrics["yaw"] = float(yaw)
    if verbose:
        print(f"  [detect {pname}] {outcome} "
              f"{np.round(pos, 4) if found else ''}")
    return SkillResult(ok=bool(found), metrics=metrics,
                       reason="" if found else f"detect {outcome}")


@register(name="inspect", category=base.CAT_EXEC,
          description="质量判定（原子）：对（已就位的）零件做一次几何检查并"
                      "给出 OK/NG 结论，不驱动任何运动。kind='seat' 测 xy/z "
                      "落位误差与倾斜角；kind='pin_engage' 测底部进入孔的深度"
                      "与孔轴偏差。可叠加传感器测量噪声。检查的是“装配质量”"
                      "（落座/压入是否达标），不是“零件存在性”（那是 detect）。",
          inputs={"part": "被检零件名", "kind": "seat|pin_engage",
                  "target_xy": "seat: 名义 world xy", "target_z": "seat: 可选名义 z",
                  "xy_tol": "seat: xy 容差 (m)", "z_tol": "seat: z 容差 (m)",
                  "tilt_tol": "倾斜容差 (deg)", "axis_xy": "pin_engage: 孔轴 xy",
                  "top_ref_z": "pin_engage: 孔口参考 z", "half": "pin_engage: 零件半高",
                  "min_depth": "pin_engage: 最小啮合深度 (m)",
                  "bot_off_tol": "pin_engage: 底端偏移容差 (m)",
                  "noise_std": "可选测量噪声 std (m)"},
          outputs={"ok": "判定结论", "tilt": "倾斜角 (deg)",
                   "xy_err": "seat: xy 误差 (m)", "z_err": "seat: z 误差 (m)",
                   "depth": "pin_engage: 啮合深度 (m)", "bot_off": "pin_engage: 底端偏移 (m)"},
          preconditions=[base.part_exists],
          postconditions=["metrics 含判定量且 ok 反映容差判定"],
          failure_policy="continue", impl=_IMPL,
          deps=["MjContext.obj_pos", "MjContext.obj_tilt",
                "MjContext.obj_axis"])
def inspect(ctx, arm, gripper, part=None, name=None, kind="seat",
            target_xy=None, target_z=None, xy_tol=0.005, z_tol=0.005,
            tilt_tol=8.0, axis_xy=None, top_ref_z=None, half=0.0,
            min_depth=0.0, bot_off_tol=0.006, noise_std=0.0, rng=None,
            verbose=False):
    """Single quality verdict on an (already seated) part; no motion."""
    pname = _part(part, name)
    r = np.random if rng is None else rng
    pos = np.asarray(ctx.obj_pos(pname), dtype=float)
    if noise_std > 0.0:
        pos = pos + r.normal(0.0, noise_std, 3)
    tilt = float(ctx.obj_tilt(pname))
    metrics = dict(tilt=tilt, pos=pos)
    ok = tilt < tilt_tol
    if kind == "pin_engage":
        if half <= 0.0 or axis_xy is None or top_ref_z is None:
            return SkillResult(ok=False, reason="inspect pin_engage needs "
                                                "axis_xy/top_ref_z/half",
                               metrics=metrics)
        axis = ctx.obj_axis(pname)
        bot = pos - float(half) * axis
        depth = float(top_ref_z) - bot[2]
        bot_off = float(np.linalg.norm(bot[:2]
                                      - np.asarray(axis_xy)[:2]))
        metrics["depth"] = depth
        metrics["bot_off"] = bot_off
        ok = depth >= min_depth and bot_off < bot_off_tol
    elif target_xy is not None:
        txy = np.asarray(target_xy, dtype=float)[:2]
        xy_err = float(np.linalg.norm(pos[:2] - txy))
        metrics["xy_err"] = xy_err
        ok = ok and xy_err < xy_tol
        if target_z is not None:
            z_err = float(pos[2] - float(target_z))
            metrics["z_err"] = z_err
            ok = ok and abs(z_err) <= z_tol
    if verbose:
        print(f"  [inspect {pname}] kind={kind} "
              f"{ {k: (round(v, 5) if isinstance(v, float) else v) for k, v in metrics.items()} } "
              f"{'OK' if ok else 'NG'}")
    return SkillResult(ok=bool(ok), metrics=metrics)


@register(name="move", category=base.CAT_EXEC,
          description="末端移动：按规划航点风格（direct/safe_z/clearance）"
                      "把 EEF 移动到 world xyz 目标，可选梯形速度顺滑。"
                      "是执行类的通用位移原语，消费 plan_path 的航点风格。",
          inputs={"to": "目标 world xyz", "style": "航点风格",
                  "tol": "到位容差 (m)", "gain": "P 增益",
                  "max_speed": "速度上限 (m/s)", "smooth": "梯形顺滑"},
          outputs={"reached": "是否到位", "eef": "终态 EEF xyz"},
          preconditions=["场景已加载，机器人可控"],
          postconditions=["EEF 与目标距离 <= tol（以 reached 判定）"],
          failure_policy="retry", impl=_MOTION,
          deps=["motion.move_eef", "planning.plan_path",
                "CartesianController.move_eef"])
def move(ctx, arm, gripper, to=None, style="safe_z", tol=0.008, gain=None,
         max_speed=None, smooth=False, verbose=False):
    """Move the EEF to a world xyz target through planned waypoints."""
    if to is None:
        return SkillResult(ok=False, reason="move requires 'to'",
                           metrics=dict())
    target = np.asarray(to, dtype=float)
    ok = move_eef(arm, target, style=style, tol=tol, gain=gain,
                  max_speed=max_speed, smooth=smooth)
    return SkillResult(ok=bool(ok),
                       metrics=dict(reached=bool(ok),
                                    eef=ctx.eef_pos().copy()))


# =====================================================================
# MINIMAL EXECUTION ATOMS: single-responsibility primitives that the
# composite skills (grasp / place / insert) decompose into.  Each does
# ONE thing (close fingers / descend / verify lift / ...) with no
# bundled perception or motion, so a planner can compose them freely.
# =====================================================================

@register(name="grip_open", category=base.CAT_EXEC,
          description="张开夹爪（最小原子）：仅将手指伺服到全开，不做任何"
                      "移动或感知。释放/接近前的手指原语。",
          inputs={"max_steps": "开爪 slew 最大步数"},
          outputs={"span": "张开后 pad 间距 (m)"},
          preconditions=["场景已加载，夹爪可控"],
          postconditions=["pad 间距达到全开（以 span 度量）"],
          failure_policy="continue", impl=_IMPL,
          deps=["Gripper.open"])
def grip_open(ctx, arm, gripper, max_steps=80, verbose=False):
    """Open the fingers (minimal finger primitive)."""
    gripper.open(max_steps=max_steps)
    span = gripper.span()
    if verbose:
        print(f"  [grip_open] span={(span or 0)*1000:.1f}mm")
    return SkillResult(ok=True, metrics=dict(span=span))


@register(name="grip_close", category=base.CAT_EXEC,
          description="闭合夹爪（最小原子）：仅将手指闭到指定开度/压入量/"
                      "接触力，不做接近或感知。三种口径：给 part/outer_d 按"
                      "外径+press 闭合并过冲咬合；给 span 闭到该 pad 间距；"
                      "给 force_stop 闭到 pad 接触力达标。即“grasp 只负责闭合"
                      "夹爪”的那个原子。",
          inputs={"part": "可选零件名（用其外径闭合）", "outer_d": "可选外径 (m)",
                  "span": "可选目标 pad 间距 (m)", "press": "压入量 (m)",
                  "force_stop": "可选：闭到该接触力 (N)"},
          outputs={"span": "闭合后 pad 间距 (m)", "pad_force": "pad 接触力 (N)"},
          preconditions=[],
          postconditions=["手指已闭合（以 span/pad_force 度量）"],
          failure_policy="continue", impl=_IMPL,
          deps=["Gripper.close_on_part", "Gripper.close_to_span",
                "perception.part_meta(可选)"])
def grip_close(ctx, arm, gripper, part=None, name=None, outer_d=None,
               span=None, press=0.0015, force_stop=None, verbose=False):
    """Close the fingers (minimal finger primitive)."""
    if force_stop is not None:
        gripper.close_to_span(span if span is not None else 0.012,
                              force_stop=force_stop)
    elif span is not None:
        gripper.close_to_span(span)
    else:
        pname = _part(part, name)
        od = outer_d
        if od is None and pname is not None:
            od = perception.part_meta(ctx, pname)["outer_d"]
        if od is None:
            gripper.close_to_span(0.012)
        else:
            gripper.close_on_part(od, press=press)
    sp = gripper.span()
    try:
        f = (ctx.geom_contact_force("finger1_pad_collision")
             + ctx.geom_contact_force("finger2_pad_collision"))
    except ValueError:
        f = 0.0
    if verbose:
        print(f"  [grip_close] span={(sp or 0)*1000:.1f}mm padF={f:.2f}N")
    return SkillResult(ok=True, metrics=dict(span=sp, pad_force=float(f)))


@register(name="descend", category=base.CAT_EXEC,
          description="垂直下降（最小原子）：保持当前 xy，把 EEF 下降到目标 "
                      "z（或给定 xyz）。接近后/放置前的精降原语，是 grasp/"
                      "place/insert 分解中的下降步。",
          inputs={"to_z": "目标 world z（保持 xy）", "to": "可选完整 xyz 目标",
                  "gain": "P 增益", "tol": "到位容差 (m)"},
          outputs={"reached": "是否到位", "eef_z": "终态 EEF z"},
          preconditions=["场景已加载，机器人可控"],
          postconditions=["EEF z 到达 to_z（以 reached 判定）"],
          failure_policy="retry", impl=_MOTION,
          deps=["CartesianController.move_eef"])
def descend(ctx, arm, gripper, to_z=None, to=None, gain=6.0, tol=0.002,
            max_steps=200, verbose=False):
    """Vertical descent holding xy (minimal motion primitive)."""
    if to is not None:
        target = np.asarray(to, dtype=float)
    elif to_z is not None:
        e = ctx.eef_pos()
        target = np.array([e[0], e[1], float(to_z)])
    else:
        return SkillResult(ok=False, reason="descend requires to_z or to",
                           metrics=dict())
    ok = arm.move_eef(target, gain=gain, tol=tol, max_steps=max_steps)
    return SkillResult(ok=bool(ok),
                       metrics=dict(reached=bool(ok),
                                    eef_z=float(ctx.eef_pos()[2])))


@register(name="lift_verify", category=base.CAT_EXEC,
          description="举升校验（最小原子，纯判定）：比较零件当前 z 与参考 "
                      "z，确认已被抬升 >= min_lift，判抓取是否抓稳。不驱动"
                      "任何运动，是 grasp 分解的收尾校验原语。",
          inputs={"part": "零件名", "z_ref": "抬升前参考 z (m)",
                  "min_lift": "最小抬升量 (m)"},
          outputs={"ok": "抬升达标", "z_now": "当前零件 z", "lift": "实测抬升量"},
          preconditions=[base.part_exists],
          postconditions=["z_now - z_ref >= min_lift（以 ok 判定）"],
          failure_policy="continue", impl=_IMPL,
          deps=["MjContext.obj_pos"])
def lift_verify(ctx, arm, gripper, part=None, name=None, z_ref=None,
                min_lift=0.03, verbose=False):
    """Verify a part was lifted (minimal predicate; no motion)."""
    pname = _part(part, name)
    z_now = float(ctx.obj_pos(pname)[2])
    if z_ref is None:
        return SkillResult(ok=False, reason="lift_verify requires z_ref",
                           metrics=dict(z_now=z_now))
    lift = z_now - float(z_ref)
    ok = lift >= float(min_lift)
    if verbose:
        print(f"  [lift_verify {pname}] lift={lift*1000:.1f}mm "
              f"{'ok' if ok else 'FAIL'}")
    return SkillResult(ok=bool(ok), metrics=dict(z_now=z_now, lift=lift))


@register(name="release", category=base.CAT_EXEC,
          description="释放（最小原子）：张开夹爪让持件原地落下并静置，不做"
                      "抬升退避（退避见过渡类 retreat_lift）。是 place 分解的"
                      "释放原语。",
          inputs={"settle_steps": "释放后静置步数"},
          outputs={"ok": "已释放", "span": "张开后 pad 间距"},
          preconditions=[],
          postconditions=["夹爪全开（以 span 度量）"],
          failure_policy="continue", impl=_IMPL,
          deps=["Gripper.open", "settle"])
def release(ctx, arm, gripper, settle_steps=20, verbose=False):
    """Open the gripper and let the held part drop in place (minimal)."""
    gripper.open()
    settle(ctx, max_steps=settle_steps)
    return SkillResult(ok=True, metrics=dict(span=gripper.span()))


@register(name="grasp", category=base.CAT_EXEC,
          description="抓取：检测/估计（或消费 plan_grasp_pose 工件）→ 接近"
                      "→ 下降 → 闭爪 → 举升校验的闭环抓取。前置零件存在，"
                      "后置以举升量判定抓稳。是操作链的核心执行技能。",
          inputs={"part": "零件名", "grasp_pose": "可选预估计抓取位姿工件",
                  "press": "压入量 (m)", "lift": "举升高度 (m)",
                  "tol": "下降容差 (m)", "squeeze": "力反馈深咬 (N)",
                  "approach_yaw": "接近前腕部偏航"},
          outputs={"ok": "举升校验通过", "held_dist": "EEF-零件距离 (m)"},
          preconditions=[base.part_exists],
          postconditions=["零件被抬升 >= VERIFY_LIFT（以 ok 判定）"],
          failure_policy="retry", impl=_IMPL,
          granularity=base.GRAN_COMPOSITE,
          decomposes=["detect", "plan_grasp_pose", "approach",
                      "grip_close", "retreat_lift", "lift_verify"],
          deps=["perception.estimate_grasp_pose", "motion.move_eef",
                "Gripper.close_on_part", "plan_grasp_pose(可选)"])
def grasp(ctx, arm, gripper, part=None, name=None, **kw):
    """Closed-loop detect+approach+close+lift-verified grasp."""
    pname = _part(part, name)
    verbose = kw.pop("verbose", False)
    ok = manipulation.grasp(ctx, arm, gripper, pname, verbose=verbose, **kw)
    try:
        held = float(np.linalg.norm(ctx.eef_pos() - ctx.obj_pos(pname)))
    except ValueError:
        held = 9.9
    return SkillResult(ok=bool(ok), metrics=dict(held_dist=held),
                       reason="" if ok else f"grasp {pname} failed")


@register(name="place", category=base.CAT_EXEC,
          description="放置：把持件搬运到目标 xy/z 并释放落位。基于持件时"
                      "实测的 EEF-零件偏移推导落点，抵消抓取检测噪声；支持"
                      "对中/低空搬运/腕部扶正/多种释放策略。前置仍持件，"
                      "后置以落位 xy 容差判定。",
          inputs={"part": "零件名", "at": "目标 world xy",
                  "target_z": "可选目标 z", "align": "落前对中",
                  "live_align": "活体比例对中", "release": "释放策略",
                  "carry_style": "搬运航点风格", "low_carry": "低空搬运"},
          outputs={"ok": "落位 xy 在容差内", "pos": "终态零件 xyz"},
          preconditions=[base.held_part],
          postconditions=["零件 xy 与目标距离 < 0.012m（以 ok 判定）"],
          failure_policy="continue", impl=_IMPL,
          granularity=base.GRAN_COMPOSITE,
          decomposes=["transport", "pre_align", "descend", "release",
                      "retreat_lift"],
          deps=["manipulation.place", "motion.move_eef", "Gripper.open"])
def place(ctx, arm, gripper, part=None, name=None, at=None, target_xy=None,
          **kw):
    """Place the held part so its body lands at the target xy/z."""
    pname = _part(part, name)
    xy = at if at is not None else target_xy
    if xy is None:
        xy = ctx.obj_pos(pname)[:2]
    xy = np.asarray(xy, dtype=float)[:2]
    verbose = kw.pop("verbose", False)
    ok = manipulation.place(ctx, arm, gripper, pname, xy,
                            verbose=verbose, **kw)
    return SkillResult(ok=bool(ok),
                       metrics=dict(pos=ctx.obj_pos(pname).copy()),
                       reason="" if ok else f"place {pname} off-target")


@register(name="transport", category=base.CAT_EXEC,
          description="搬运（组合）：把持件沿给定航点（或到目标点的规划航点）"
                      "低速平移到目标上方悬停，供后续 place/insert 精降；全程"
                      "监控仍持件，滑脱即失败。由原子 move（航点执行）与持件"
                      "校验组合而成。",
          inputs={"part": "零件名", "waypoints": "航点列表 (world xyz)",
                  "to": "可选：目标 xyz（缺 waypoints 时按 style 规划）",
                  "style": "航点风格", "carry_speed": "搬运速度上限",
                  "tol": "航点到位容差"},
          outputs={"ok": "到达且仍持件", "held": "搬运后是否仍持件"},
          preconditions=[base.held_part],
          postconditions=["搬运后 EEF-零件 xy 距离 < 0.03m（以 held 判定）"],
          failure_policy="continue", impl=_MOTION,
          granularity=base.GRAN_COMPOSITE,
          decomposes=["move(低速航点执行)", "持件校验(谓词，滑脱即败)"],
          deps=["CartesianController.move_eef", "motion.move_eef",
                "plan_path(可选)"])
def transport(ctx, arm, gripper, part=None, name=None, waypoints=None,
              to=None, style="safe_z", carry_speed=0.12, tol=0.004,
              smooth=True, verbose=False):
    """Carry the held part along waypoints (or a planned path) to a hover."""
    pname = _part(part, name)
    if waypoints is None:
        if to is None:
            return SkillResult(ok=False,
                               reason="transport requires waypoints or 'to'",
                               metrics=dict())
        wps = list(planning.plan_path(ctx.eef_pos(),
                                      np.asarray(to, dtype=float),
                                      style=style))
    else:
        wps = [np.asarray(w, dtype=float) for w in waypoints]
    ok = True
    for w in wps:
        if not arm.move_eef(w, tol=tol, gain=6.0, max_speed=carry_speed,
                            smooth=smooth):
            ok = False
            break
    held = float(np.linalg.norm(ctx.eef_pos()[:2]
                                - ctx.obj_pos(pname)[:2])) < 0.03
    if verbose:
        print(f"  [transport {pname}] {'ok' if (ok and held) else 'FAIL'} "
              f"held={held}")
    return SkillResult(ok=bool(ok and held), metrics=dict(held=bool(held)),
                       reason="" if (ok and held) else "transport slip/stall")


@register(name="push", category=base.CAT_EXEC,
          description="推动：并指成刃，从零件背后沿方向闭环推抵，实时监测"
                      "零件随动位移，卡滞即停（不硬磨）。用于纠偏/送料等"
                      "非抓持的水平位移。以位移达请求量 70% 判定成功。",
          inputs={"part": "零件名", "delta": "位移 (dx,dy)",
                  "to_target": "可选绝对目标 xy（优先）",
                  "push_z": "推动线 world z", "speed": "推动速度"},
          outputs={"ok": "位移 >= 70% 请求量", "moved": "实测位移向量"},
          preconditions=[base.part_exists],
          postconditions=["零件沿方向位移 >= 70% 请求量（以 ok 判定）"],
          failure_policy="continue", impl=_IMPL,
          deps=["manipulation.push", "motion.move_eef", "Gripper"])
def push(ctx, arm, gripper, part=None, name=None, **kw):
    """Blade-push a part horizontally (closed-loop, stall-aware)."""
    pname = _part(part, name)
    verbose = kw.pop("verbose", False)
    before = ctx.obj_pos(pname)[:2].copy()
    ok = manipulation.push(ctx, arm, gripper, pname, verbose=verbose, **kw)
    moved = ctx.obj_pos(pname)[:2] - before
    return SkillResult(ok=bool(ok), metrics=dict(moved=moved.copy()),
                       reason="" if ok else "push under-delivered")


@register(name="insert", category=base.CAT_EXEC,
          description="插入装配：mode='press' 到点压入并保压；mode='thread' "
                      "底部导向螺旋下降（比例纠偏+抗卡摆动），把持件装入孔/"
                      "轴。前置仍持件。是插销/套轴等装配的执行原语（学习版"
                      "见扩展类 peg_insert mode='policy'）。",
          inputs={"part": "零件名", "mode": "press|thread",
                  "target_eef": "press 模式目标 EEF xyz",
                  "ref_axis": "thread 模式参考轴 (callable/xy)",
                  "to_z": "thread 模式零件底面目标 z", "half": "零件半高"},
          outputs={"ok": "插入完成", "bot_z": "终态零件底面 z"},
          preconditions=[base.held_part],
          postconditions=["零件底面到达 to_z 附近（thread 模式）"],
          failure_policy="abort", impl=_OPT,
          granularity=base.GRAN_COMPOSITE,
          decomposes=["pre_align", "descend(thread+抗卡摆动)", "release"],
          deps=["manipulation.insert", "perception.part_meta",
                "CartesianController.move_eef"])
def insert(ctx, arm, gripper, part=None, name=None, mode="press", half=None,
           **kw):
    """Insert the held part via press-fit or bottom-steered threading."""
    pname = _part(part, name)
    verbose = kw.pop("verbose", False)
    if half is None:
        try:
            half = perception.part_meta(ctx, pname)["half_h"]
        except ValueError:
            half = None
    ok = manipulation.insert(ctx, arm, gripper, name=pname, mode=mode,
                             half=half, verbose=verbose, **kw)
    bot_z = float(ctx.obj_pos(pname)[2] - (half or 0.0))
    return SkillResult(ok=bool(ok), metrics=dict(bot_z=bot_z),
                       reason="" if ok else f"insert({mode}) {pname} failed")


# =====================================================================
# PLANNING SKILLS  (category="plan": computation only, emit artifacts,
# generate MULTIPLE candidate branches, score, filter, select the best)
# =====================================================================

@register(name="plan_grasp_pose", category=base.CAT_PLAN,
          description="抓取位姿估计（多候选）：由检测观测+零件元数据生成 "
                      "n_yaw 个接近角候选抓取位姿，按接近线碰撞/零件类型做"
                      "可行性过滤，按行程+偏航失配+碰撞惩罚评分，排序选最优"
                      "作为抓取工件（保留全部候选分支用于归因/数据导出）。",
          inputs={"part": "零件名", "detect_result": "可选外部观测 "
                  "(found,pos,yaw,outcome)", "noise_std": "检测噪声 std",
                  "n_yaw": "候选接近角数量（2=旧两轴）",
                  "obstacles": "可选静态障碍 AABB 列表", "yaw": "可选指定偏航"},
          outputs={"grasp_pose": "选中的抓取工件 {pos,yaw,outer_d,...}",
                   "n_candidates": "候选数", "candidates": "候选分支摘要"},
          preconditions=[base.part_exists],
          postconditions=["至少一个可行候选（否则 ok=False，估计失败）"],
          failure_policy="retry", impl=_MOTION,
          deps=["planning.grasp_pose_candidates", "perception.part_meta",
                "perception.detect"])
def plan_grasp_pose(ctx, arm, gripper, part=None, name=None,
                    detect_result=None, noise_std=0.0, n_yaw=8,
                    obstacles=None, yaw=None, grasp_dz=None, rng=None,
                    verbose=False):
    """Multi-candidate grasp-pose estimation + scoring + best selection."""
    pname = _part(part, name)
    meta = perception.part_meta(ctx, pname)
    cands = planning.grasp_pose_candidates(
        ctx, pname, meta, detect_result=detect_result,
        noise_std=noise_std, rng=rng, n_yaw=n_yaw, obstacles=obstacles)
    feasible = [c for c in cands if c.get("feasible")]
    if not feasible:
        return SkillResult(ok=False, metrics=dict(n_candidates=len(cands)),
                           reason=f"no feasible grasp for {pname}")
    best = dict(feasible[0])
    if grasp_dz is not None:
        best["pos"] = np.array([best["pos"][0], best["pos"][1],
                                ctx.obj_pos(pname)[2] + float(grasp_dz)])
    if yaw is not None:
        best["approach_yaw"] = float(yaw)
    summary = [{k: c[k] for k in ("pos", "yaw", "score", "feasible",
                                  "reason") if k in c} for c in cands]
    if verbose:
        print(f"  [plan_grasp_pose {pname}] best of {len(cands)} "
              f"(feasible {len(feasible)}) score={best['score']:.2f}")
    return SkillResult(ok=True, metrics=dict(
        grasp_pose=best, n_candidates=len(cands),
        n_feasible=len(feasible), candidates=summary))


@register(name="plan_path", category=base.CAT_PLAN,
          description="路径规划（多候选）：由起点到目标生成多个航点候选"
                      "（高度变体×侧向 via 偏移），对每个候选做线段-AABB "
                      "碰撞门过滤，按路径长+航点数评分，选最优可行路径；"
                      "全碰撞时换 style 重规划。产出 waypoints 工件供 "
                      "move/transport 消费。",
          inputs={"to": "目标 world xyz", "cur": "可选起点（默认 EEF）",
                  "style": "direct|safe_z|clearance", "lift": "抬升量 (m)",
                  "safe_z": "clearance 绝对高度", "obstacles": "静态障碍 AABB",
                  "via_offsets": "侧向 via 偏移候选"},
          outputs={"waypoints": "选中路径航点", "style": "路径风格",
                   "score": "路径评分", "n_candidates": "候选数",
                   "candidates": "候选分支+碰撞命中摘要"},
          preconditions=["场景已加载，机器人可控"],
          postconditions=["至少一个无碰撞候选（否则 ok=False，需重规划）"],
          failure_policy="retry", impl=_MOTION,
          deps=["planning.path_candidates", "planning.check_path",
                "planning.seg_aabb_hit"])
def plan_path(ctx, arm, gripper, to=None, cur=None, style="safe_z",
              lift=0.08, safe_z=None, obstacles=None, margin=0.005,
              via_offsets=(0.0, 0.03, -0.03), verbose=False):
    """Multi-candidate path planning + collision gate + best selection."""
    if to is None:
        return SkillResult(ok=False, reason="plan_path requires 'to'",
                           metrics=dict())
    start = ctx.eef_pos() if cur is None else np.asarray(cur, dtype=float)
    goal = np.asarray(to, dtype=float)
    obstacles = obstacles or []
    styles = [style] + [s for s in ("clearance", "safe_z") if s != style]
    scored = []
    for st in styles:
        cands = planning.path_candidates(start, goal, style=st, lift=lift,
                                         safe_z=safe_z,
                                         via_offsets=via_offsets)
        for wp in cands:
            hits = planning.check_path([start] + list(wp), obstacles,
                                       margin=margin)
            scored.append(dict(waypoints=[w.tolist() for w in wp.waypoints],
                               style=st, score=float(wp.score),
                               valid=not hits, hits=[h[1] for h in hits]))
        feasible = [s for s in scored if s["valid"]]
        if feasible:
            break
    feasible = [s for s in scored if s["valid"]]
    if not feasible:
        return SkillResult(ok=False, metrics=dict(n_candidates=len(scored),
                                                  candidates=scored),
                           reason="all path candidates collide")
    best = min(feasible, key=lambda s: s["score"])
    if verbose:
        print(f"  [plan_path] {len(best['waypoints'])} wps style={best['style']} "
              f"(best of {len(scored)}, feasible {len(feasible)})")
    return SkillResult(ok=True, metrics=dict(
        waypoints=[np.asarray(w, dtype=float) for w in best["waypoints"]],
        style=best["style"], score=best["score"],
        n_candidates=len(scored), n_feasible=len(feasible),
        candidates=scored))
