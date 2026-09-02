#!/usr/bin/env python3
"""Record one demo mp4 per ATOMIC skill (the granularity audit output).

    MUJOCO_GL=egl python -m simbench.record_skills [--skills exec_move,..]
    MUJOCO_GL=egl python -m simbench.record_skills --all

Each demo shows the atomic skill's typical pre-state, its execution
(through the REGISTERED atom via REGISTRY.run -- never the composite
wrappers), and its result overlaid as banner text (metrics / OK-NG).
Pure computation / verdict atoms (detect, inspect, lift_verify,
plan_grasp_pose, plan_path) get a visualisation leg (e.g. the EEF drives
to the detected / planned pose) plus a metrics overlay so "what it did"
is visible.  Videos land in ``results/skill_videos/<cat>_<name>.mp4``
(exec_*/plan_*/trans_*/ext_*), and the skill->video index is written to
``results/skill_videos/index.md`` and ``docs/skill_videos_index.md``.
"""
import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np                                  # noqa: E402

from simbench.core.sim_context import MjContext     # noqa: E402
from simbench.core.controller import CartesianController, Gripper  # noqa: E402
from simbench.core.camera import VideoRecorder      # noqa: E402
from simbench.skills.settle import settle           # noqa: E402
from simbench.skills import manipulation, motion, perception  # noqa: E402
from simbench.skills import REGISTRY, base          # noqa: E402

SCENE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "scenes", "taskA_gearbox.xml")
OUT_DIR = os.path.join(_REPO, "results", "skill_videos")
CUBE = "test_cube"
CUBE_XY = np.array([-0.02, 0.24])
CUBE_Z = 0.8155


class Harness:
    """One scene + recorder for one skill demo."""

    def __init__(self, skill, cat):
        os.makedirs(OUT_DIR, exist_ok=True)
        self.skill, self.cat = skill, cat
        self.path = os.path.join(OUT_DIR, f"{cat}_{skill}.mp4")
        self.ctx = MjContext(SCENE)
        self.ctx.reset()
        self.arm = CartesianController(self.ctx)
        self.gripper = Gripper(self.ctx)
        settle(self.ctx, max_steps=100)
        self.rec = VideoRecorder(self.ctx, self.path,
                                 title=f"{cat} {skill}", camera="agentview",
                                 every=2, repeat=3)
        self.ctx.on_control_step = self.rec
        self.rec.set_stage("pre-state")

    def stage(self, s):
        self.rec.set_stage(str(s)[:60])

    def hold(self, sec=1.0):
        for _ in range(int(sec / self.ctx.control_dt)):
            self.ctx.step()

    def settle(self, sec=0.6):
        settle(self.ctx, max_steps=int(sec / self.ctx.control_dt))

    def result(self, ok=True, tag=None):
        self.rec.add_result(f"S-{tag or self.skill}", bool(ok), "")

    def run_atom(self, skill, **params):
        """Execute the REGISTERED atom (the one being demonstrated)."""
        res = REGISTRY.run(skill, self.ctx, self.arm, self.gripper,
                           verbose=False, **params)
        return res

    def close(self):
        self.rec.close()


def _move_eef(h, xyz, style="direct", tol=0.008, max_speed=None):
    motion.move_eef(h.arm, np.asarray(xyz, dtype=float), style=style,
                    tol=tol, max_speed=max_speed)


def _pick_cube(h):
    """Setup helper: pick the test cube (composite grasp, not recorded)."""
    h.stage("prep: pick cube")
    ok = manipulation.grasp(h.ctx, h.arm, h.gripper, CUBE,
                            press=0.0025, verbose=False)
    h.settle()
    return ok


# =====================================================================
# EXEC atoms
# =====================================================================
def demo_detect(h):
    h.stage("detect test_cube")
    res = h.run_atom("detect", part=CUBE)
    h.hold(0.6)
    p = res.metrics.get("pos")
    h.stage(f"found pos=({p[0]:.3f},{p[1]:.3f}) outcome={res.metrics.get('outcome')}")
    h.hold(1.2)
    # visualisation leg: drive the EEF above the DETECTED pose
    h.stage("viz: eef -> detected pose")
    _move_eef(h, [p[0], p[1], p[2] + 0.13], tol=0.006)
    h.settle()
    h.stage(f"result: detect {'OK' if res.ok else 'NG'}")
    h.hold(1.5)
    h.result(res.ok)
    return res.ok


def demo_inspect(h):
    # two verdicts on the seated cube: NG (tight tol) then OK (spec tol)
    h.stage("inspect test_cube (seat)")
    res1 = h.run_atom("inspect", part=CUBE, kind="seat", target_xy=CUBE_XY,
                      target_z=CUBE_Z, xy_tol=0.0005, z_tol=0.001,
                      tilt_tol=2.0)
    h.hold(1.0)
    xy = res1.metrics.get("xy_err", 0.0)
    h.stage(f"tight tol -> {'OK' if res1.ok else 'NG'} xy_err={xy*1000:.2f}mm")
    h.hold(1.5)
    res2 = h.run_atom("inspect", part=CUBE, kind="seat", target_xy=CUBE_XY,
                      target_z=CUBE_Z, xy_tol=0.006, z_tol=0.006,
                      tilt_tol=5.0)
    h.hold(1.2)
    h.stage(f"spec tol -> {'OK' if res2.ok else 'NG'} tilt="
            f"{res2.metrics.get('tilt', 0):.2f}deg")
    h.hold(1.5)
    h.result(res2.ok, tag="inspect")
    return res2.ok


def demo_move(h):
    a = np.array([-0.16, 0.16, 0.96])
    b = np.array([0.02, 0.24, 0.93])
    h.stage(f"move -> ({a[0]:.2f},{a[1]:.2f})")
    _move_eef(h, a, style="safe_z", tol=0.006)
    h.hold(0.4)
    h.stage(f"move -> ({b[0]:.2f},{b[1]:.2f})")
    res = h.run_atom("move", to=b, style="safe_z", tol=0.006)
    h.settle()
    h.stage(f"result: move {'OK' if res.ok else 'NG'}")
    h.hold(1.5)
    h.result(res.ok)
    return res.ok


def demo_grip_open(h):
    h.stage("grip_open")
    res = h.run_atom("grip_open")
    h.hold(1.0)
    sp = res.metrics.get("span")
    h.stage(f"result: open span={(sp or 0)*1000:.1f}mm")
    h.hold(1.5)
    h.result(res.ok)
    return res.ok


def demo_grip_close(h):
    # pre-state: hover + descend over the cube (prep, not the atom)
    gp = perception.estimate_grasp_pose(h.ctx, CUBE)
    gpos = gp["pos"]
    _move_eef(h, gpos + np.array([0, 0, 0.08]), style="safe_z", tol=0.006)
    h.arm.move_eef(gpos, gain=6.0, tol=0.004)
    h.settle()
    h.stage("grip_close on test_cube")
    res = h.run_atom("grip_close", part=CUBE, press=0.0025)
    h.hold(1.0)
    sp = res.metrics.get("span")
    f = res.metrics.get("pad_force", 0.0)
    h.stage(f"result: span={(sp or 0)*1000:.1f}mm padF={f:.2f}N")
    h.hold(1.5)
    # effect leg: the closed grip must carry the cube up
    h.stage("effect: lift with closed grip")
    h.arm.move_eef(h.ctx.eef_pos() + np.array([0, 0, 0.05]), gain=8.0,
                   tol=0.01)
    h.settle()
    h.stage(f"result: grip holds cube (z={h.ctx.obj_pos(CUBE)[2]:.3f})")
    h.hold(1.2)
    h.result(res.ok)
    return res.ok


def demo_release(h):
    if not _pick_cube(h):
        return False
    h.stage("release (drop in place)")
    res = h.run_atom("release", settle_steps=25)
    h.hold(0.6)
    h.stage(f"result: released, cube z={h.ctx.obj_pos(CUBE)[2]:.3f}")
    h.hold(1.5)
    h.result(res.ok)
    return res.ok


def demo_lift_verify(h):
    z_ref = float(h.ctx.obj_pos(CUBE)[2])   # floor z BEFORE the pick
    if not _pick_cube(h):
        return False
    h.stage("lift_verify (is the cube held aloft?)")
    res = h.run_atom("lift_verify", part=CUBE, z_ref=z_ref, min_lift=0.03)
    h.hold(1.0)
    lift = res.metrics.get("lift", 0.0)
    h.stage(f"result: lift={lift*1000:.1f}mm -> "
            f"{'OK held' if res.ok else 'NG dropped'}")
    h.hold(1.5)
    h.result(res.ok)
    return res.ok


def demo_push(h):
    h.stage("push test_cube +5cm (blade)")
    before = h.ctx.obj_pos(CUBE)[:2].copy()
    res = h.run_atom("push", part=CUBE, delta=(0.05, 0.0), speed=0.04)
    h.settle(1.0)
    moved = h.ctx.obj_pos(CUBE)[:2] - before
    h.stage(f"result: moved {np.linalg.norm(moved)*1000:.0f}mm "
            f"{'OK' if res.ok else 'NG'}")
    h.hold(1.5)
    h.result(res.ok)
    return res.ok


def demo_transport(h):
    if not _pick_cube(h):
        return False
    e = h.ctx.eef_pos()
    w1 = e + np.array([0.0, 0.0, 0.03])
    w2 = np.array([e[0] + 0.12, e[1] - 0.08, e[2]])
    w3 = np.array([e[0] + 0.12, e[1] - 0.08, e[2] + 0.02])
    h.stage("transport: carry along waypoints")
    res = h.run_atom("transport", part=CUBE,
                     waypoints=[w1, w2, w3], carry_speed=0.12, tol=0.006)
    h.settle()
    held = res.metrics.get("held", False)
    h.stage(f"result: carry {'OK held' if res.ok else 'NG'} held={held}")
    h.hold(1.5)
    h.result(res.ok)
    return res.ok


def demo_descend(h):
    p = np.array([CUBE_XY[0], CUBE_XY[1], 0.97])
    _move_eef(h, p, style="safe_z", tol=0.006)
    h.hold(0.4)
    h.stage("descend (hold xy, drop to z=0.85)")
    res = h.run_atom("descend", to_z=0.85, gain=6.0, tol=0.003)
    h.settle()
    z = h.ctx.eef_pos()[2]
    h.stage(f"result: eef_z={z:.3f} {'OK' if res.ok else 'NG'}")
    h.hold(1.5)
    h.result(res.ok)
    return res.ok


# =====================================================================
# PLAN atoms (computation; visualised by driving the EEF along the
# chosen candidate)
# =====================================================================
def demo_plan_grasp_pose(h):
    h.stage("plan_grasp_pose: candidates for gear")
    res = h.run_atom("plan_grasp_pose", part="gear", n_yaw=6)
    h.hold(0.8)
    gp = res.metrics.get("grasp_pose", {})
    n = res.metrics.get("n_candidates", 0)
    h.stage(f"best of {n}: pos=({gp.get('pos', [0,0,0])[0]:.3f},"
            f"{gp.get('pos', [0,0,0])[1]:.3f}) yaw={gp.get('yaw', 0):.2f}")
    h.hold(1.2)
    h.stage("viz: eef -> chosen grasp pose")
    _move_eef(h, gp["pos"] + np.array([0, 0, 0.08]), style="safe_z",
              tol=0.006)
    h.settle()
    h.stage(f"result: plan_grasp_pose {'OK' if res.ok else 'NG'}")
    h.hold(1.5)
    h.result(res.ok)
    return res.ok


def demo_plan_path(h):
    tgt = np.array([0.0, 0.0, 0.85])
    h.stage("plan_path: route to tray centre")
    res = h.run_atom("plan_path", to=tgt, style="safe_z")
    h.hold(0.8)
    wps = res.metrics.get("waypoints", [])
    n = res.metrics.get("n_candidates", 0)
    sc = res.metrics.get("score", 0.0)
    h.stage(f"best of {n}: {len(wps)} waypoints score={sc:.3f}")
    h.hold(1.0)
    h.stage("viz: travel the planned waypoints")
    for w in wps:
        _move_eef(h, w, style="direct", tol=0.007)
    h.settle()
    h.stage(f"result: plan_path {'OK' if res.ok else 'NG'}")
    h.hold(1.5)
    h.result(res.ok)
    return res.ok


# =====================================================================
# TRANS atoms
# =====================================================================
def demo_retreat_lift(h):
    _move_eef(h, np.array([CUBE_XY[0], CUBE_XY[1], 0.86]), style="safe_z",
              tol=0.006)
    h.hold(0.4)
    h.stage("retreat_lift +0.12m")
    res = h.run_atom("retreat_lift", height=0.12, tol=0.012)
    h.settle()
    lm = res.metrics.get("lift_m", 0.0)
    h.stage(f"result: lift={lm*1000:.0f}mm {'OK' if res.ok else 'NG'}")
    h.hold(1.5)
    h.result(res.ok)
    return res.ok


def demo_pre_align(h):
    if not _pick_cube(h):
        return False
    target = np.array([0.0, -0.12])          # clear target point
    e = h.ctx.eef_pos()
    _move_eef(h, np.array([target[0] + 0.03, target[1], e[2]]),
              style="safe_z", tol=0.006)
    h.settle()
    h.stage("pre_align: held cube -> target (offset 3cm)")
    res = h.run_atom("pre_align", at=target, part=CUBE, tol=0.0015,
                     max_iters=25)
    h.settle()
    resid = res.metrics.get("resid", 9.9)
    h.stage(f"result: resid={resid*1000:.1f}mm "
            f"{'OK' if res.ok else 'NG'}")
    h.hold(1.5)
    h.result(res.ok)
    return res.ok


# =====================================================================
# EXT atoms
# =====================================================================
def demo_pull(h):
    if not _pick_cube(h):
        return False
    h.stage("pull: drag held cube -x")
    # NOTE the registered pull atom names its body param `name` (see
    # extension.py), unlike the exec atoms that use `part`
    res = h.run_atom("pull", name=CUBE, delta=(-0.05, 0.0), speed=0.04)
    h.settle()
    mv = res.metrics.get("moved_m", 0.0)
    h.stage(f"result: pulled {mv*1000:.0f}mm {'OK' if res.ok else 'NG'}")
    h.hold(1.5)
    h.result(res.ok)
    return res.ok


def demo_wipe(h):
    at = np.array([0.15, 0.0])           # empty fixture-tray interior
    h.stage("wipe: press blade onto tray floor")
    _move_eef(h, np.array([at[0], at[1], 0.92]), style="safe_z", tol=0.01)

    def _pf():
        return (h.ctx.geom_contact_force("finger1_pad_collision")
                + h.ctx.geom_contact_force("finger2_pad_collision"))

    # one full move toward the floor (contact blocks it), then a fine
    # servo to land the pad force in a light-contact band
    h.arm.move_eef(np.array([at[0], at[1], 0.802]), gain=6.0, tol=0.002,
                   max_steps=400, stall=False)
    for _ in range(300):
        ff = _pf()
        if 0.05 <= ff <= 1.2:
            break
        e = h.ctx.eef_pos()
        dz = -0.0002 if ff < 0.05 else 0.0003
        h.arm.move_eef(e + np.array([0, 0, dz]), gain=6.0, tol=0.0,
                       max_steps=1, stall=False)
    zz = float(h.ctx.eef_pos()[2])
    h.settle()
    h.stage("wipe: blade sweep +x 45mm (tray floor)")
    res = h.run_atom("wipe", at=at, direction=(1.0, 0.0), length=0.045,
                     z=zz, f_band=(0.05, 3.0), speed=0.06)
    h.settle()
    fm = res.metrics.get("force_max", 0.0)
    sw = res.metrics.get("swept_m", 0.0)
    h.stage(f"result: swept {sw*1000:.0f}mm Fmax={fm:.2f}N "
            f"{'OK' if res.ok else 'NG'}")
    h.hold(1.5)
    h.result(res.ok)
    return res.ok


DEMOS = [
    ("exec", "detect", demo_detect, "感知零件位姿（单次判定）"),
    ("exec", "inspect", demo_inspect, "落座质量判定 OK/NG（NG 紧容差 + OK 规格容差两例）"),
    ("exec", "move", demo_move, "EEF 航点移动到目标点"),
    ("exec", "descend", demo_descend, "保持 xy 垂直下降到目标 z"),
    ("exec", "grip_open", demo_grip_open, "张开夹爪到全开"),
    ("exec", "grip_close", demo_grip_close, "闭合夹爪咬合零件（含举升效果验证）"),
    ("exec", "release", demo_release, "张开夹爪原地释放持件"),
    ("exec", "lift_verify", demo_lift_verify, "举升校验：判零件被抬升达标（纯判定）"),
    ("exec", "push", demo_push, "并指刃状推动零件 5cm"),
    ("exec", "transport", demo_transport, "持件沿航点搬运（组合，视频演示其效果）"),
    ("plan", "plan_grasp_pose", demo_plan_grasp_pose, "多候选抓取位姿生成/评分/选优（EEF 走向所选位姿可视化）"),
    ("plan", "plan_path", demo_plan_path, "多候选路径规划 + 碰撞门（EEF 沿所选航点行进可视化）"),
    ("trans", "retreat_lift", demo_retreat_lift, "垂直抬升退避 0.12m"),
    ("trans", "pre_align", demo_pre_align, "持件对中到目标 xy（3cm 偏置收敛）"),
    ("ext", "pull", demo_pull, "抓持后沿方向拖拽并释放"),
    ("ext", "wipe", demo_wipe, "并指压表+力带扫掠擦拭"),
]


def write_index(results):
    lines = ["# 原子技能演示视频索引（Skill Videos Index）", "",
             "> 与 `docs/skill_inventory.md` 一一对应；由 "
             "`simbench/record_skills.py` 生成。视频文件位于本目录"
             "（results/，不入库）。", ""]
    lines.append("| 技能 | 类别 | 粒度 | 视频路径 | 演示内容概述 | 录制结果 |")
    lines.append("|---|---|---|---|---|---|")
    by_name = {r["name"]: r for r in REGISTRY.inventory()}
    for cat, name, fn, desc in DEMOS:
        gr = by_name.get(name, {}).get("granularity_cn", "原子")
        exists = os.path.exists(os.path.join(OUT_DIR, f"{cat}_{name}.mp4"))
        lines.append(f"| `{name}` | {cat} | {gr} | "
                     f"results/skill_videos/{cat}_{name}.mp4 | {desc} "
                     f"| {'已录制' if exists else '缺失'} |")
    lines.append("")
    for p in (os.path.join(OUT_DIR, "index.md"),
              os.path.join(_REPO, "docs", "skill_videos_index.md")):
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
        with open(p, "w") as f:
            f.write("\n".join(lines))
    print(f"wrote index -> {OUT_DIR}/index.md, docs/skill_videos_index.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--skills", default="",
                    help="comma list of skill names to record")
    args = ap.parse_args()
    names = {d[1] for d in DEMOS} if args.all else \
        {s.strip() for s in args.skills.split(",") if s.strip()}
    results = {}
    for cat, name, fn, desc in DEMOS:
        if name not in names:
            continue
        h = Harness(name, cat)
        print(f"== recording {cat}_{name} ...", flush=True)
        try:
            ok = fn(h)
            results[name] = bool(ok)
            print(f"   -> {'PASS' if ok else 'FAIL'} {h.path}", flush=True)
        except Exception as exc:
            results[name] = False
            print(f"   -> EXCEPTION {exc}", flush=True)
        finally:
            h.close()
    write_index(results)
    ok_n = sum(1 for v in results.values() if v)
    print(f"recorded {len(results)} demos, {ok_n} PASS")


if __name__ == "__main__":
    main()
