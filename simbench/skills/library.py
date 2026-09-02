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
                  "step": "可选步序号（供故障模型归因）",
                  "retries": "漏检重试次数（默认 0）"},
          outputs={"found": "是否检测到", "pos": "world xyz 观测",
                   "yaw": "偏航观测 (rad)", "outcome": "ok|miss|false"},
          preconditions=[base.part_exists],
          postconditions=["检测到则 outcome!=miss 且 pos 非空"],
          failure_policy="retry", impl=_IMPL,
          deps=["perception.detect_part", "faults.FailureModel(可选)"])
def detect(ctx, arm, gripper, part=None, name=None, faults=None, step=None,
           retries=0, verbose=False):
    """Detect a part under the failure model (perception entry point)."""
    pname = _part(part, name)
    attempts = int(retries) + 1
    last = (False, None, None, "miss")
    for _ in range(attempts):
        found, pos, yaw, outcome = perception.detect_part(
            ctx, pname, faults=faults, step=step)
        last = (found, pos, yaw, outcome)
        if found:
            break
    found, pos, yaw, outcome = last
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
          description="质量检测：对（已就位的）零件做 seat 几何判定——测量 "
                      "xy 位置误差、z 高度误差与倾斜角，与容差比较给出 "
                      "OK/NG 结论，可叠加传感器测量噪声。用于阶段终检或"
                      "中段纠偏触发。",
          inputs={"part": "被检零件名", "target_xy": "名义 world xy",
                  "target_z": "可选名义 world z", "xy_tol": "xy 容差 (m)",
                  "z_tol": "z 容差 (m)", "tilt_tol": "倾斜容差 (deg)",
                  "noise_std": "可选测量噪声 std (m)"},
          outputs={"xy_err": "xy 误差 (m)", "z_err": "z 误差 (m)",
                   "tilt": "倾斜角 (deg)", "ok": "判定结论"},
          preconditions=[base.part_exists],
          postconditions=["metrics 含 xy_err/tilt 且 ok 反映容差判定"],
          failure_policy="continue", impl=_IMPL,
          deps=["MjContext.obj_pos", "MjContext.obj_tilt"])
def inspect(ctx, arm, gripper, part=None, name=None, target_xy=None,
            target_z=None, xy_tol=0.005, z_tol=0.005, tilt_tol=8.0,
            noise_std=0.0, rng=None, verbose=False):
    """Seat-geometry quality inspection of a part (OK/NG verdict)."""
    pname = _part(part, name)
    r = np.random if rng is None else rng
    pos = np.asarray(ctx.obj_pos(pname), dtype=float)
    if noise_std > 0.0:
        pos = pos + r.normal(0.0, noise_std, 3)
    tilt = float(ctx.obj_tilt(pname))
    metrics = dict(tilt=tilt, pos=pos)
    ok = tilt < tilt_tol
    if target_xy is not None:
        txy = np.asarray(target_xy, dtype=float)[:2]
        xy_err = float(np.linalg.norm(pos[:2] - txy))
        metrics["xy_err"] = xy_err
        ok = ok and xy_err < xy_tol
        if target_z is not None:
            z_err = float(pos[2] - float(target_z))
            metrics["z_err"] = z_err
            ok = ok and abs(z_err) <= z_tol
    if verbose:
        print(f"  [inspect {pname}] tilt={tilt:.2f}deg "
              f"xy_err={metrics.get('xy_err', 0)*1000:.2f}mm "
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
          description="搬运：持件沿给定航点（或到目标点的规划航点）低速"
                      "平移到目标上方悬停，供后续 place/insert 精降。全程"
                      "监控仍持件，滑脱即失败。衔接规划与放置的搬运腿。",
          inputs={"part": "零件名", "waypoints": "航点列表 (world xyz)",
                  "to": "可选：目标 xyz（缺 waypoints 时按 style 规划）",
                  "style": "航点风格", "carry_speed": "搬运速度上限",
                  "tol": "航点到位容差"},
          outputs={"ok": "到达且仍持件", "held": "搬运后是否仍持件"},
          preconditions=[base.held_part],
          postconditions=["搬运后 EEF-零件 xy 距离 < 0.03m（以 held 判定）"],
          failure_policy="continue", impl=_MOTION,
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
