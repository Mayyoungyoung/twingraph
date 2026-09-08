"""Record every atom in a purpose-built scene with truthful spatial explanations.

Preparations run through the same controller/skills before recording. Each clip
restores a physically reached checkpoint; recording never teleports any object.
"""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .demo_scenes import make_demo_session, PIN_TARGET
from .demo_visuals import Visuals, DemoRecorder, joint_path_points
from .library import CATALOG
from .task import pick, transfer_part
from .scene import CENTER
from .control import DEFAULT_POLICY


def serial(x):
    if isinstance(x, Path):
        return str(x)
    return np.asarray(x).tolist()


NOTES = {
    "observe_parts": "半透明彩球＝检测中心；每个目标独立着色（仿真位姿观测）",
    "estimate_pose": "红 / 绿 / 蓝箭头＝估计的 X / Y / Z 轴；球心＝位置估计",
    "propose_grasps": "青 / 橙夹爪框＝两个实际生成的抓取候选；箭头＝接近方向",
    "select_grasp": "绿色夹爪框＝选中的抓取位姿；灰色＝另一候选",
    "plan_transfer": "青线＝关节规划经正运动学得到的末端轨迹；绿球＝终点",
    "plan_linear": "青线＝已求解逆运动学的直线路径；绿球＝目标位置",
    "execute_joint_path": "青线＝计划轨迹；黄线＝机械臂实际末端轨迹；灰块＝障碍物",
    "execute_cartesian_path": "青线＝直线计划；黄线＝实际末端运动；绿球＝终点",
    "open_gripper": "释放桌上的小方块；绿色点＝真实指垫接触",
    "approach": "绿色夹持目标；青线＝接近方向；黄线＝实际末端运动",
    "close_gripper": "夹住 40 mm 方块；绿色点＝真实指垫接触",
    "verify_grasp": "验证左右指垫都接触同一方块；绿色点＝真实接触点",
    "lift": "青箭头＝目标抬升；黄线＝被夹方块的实际运动",
    "lower": "青箭头＝目标下移；黄线＝被夹方块的实际运动",
    "retreat": "释放后空夹爪上撤；青箭头＝退出方向；黄线＝实际末端运动",
    "home": "绿球＝初始关节位姿对应的末端位置；黄线＝实际回位轨迹",
    "orient_wrist": "红 / 绿 / 蓝＝末端坐标轴；保持位置并转动手腕 60°",
    "align_axis": "绿线＝孔的目标轴线；黄线＝插销实际位置；消除横向偏差",
    "plan_insertion": "绿线＝孔轴；青线＝插入参数给定的目标方向与深度",
    "slide_insert": "青线＝导轨插入方向；黄线＝滑块实际运动",
    "guarded_descent": "插销故意偏离孔口；橙点＝碰到孔座；达到力阈值停止下降",
    "press_seat": "肩面压靠孔座；橙点＝真实支撑接触；同时检查高度与接触",
    "inspect_seat": "绿球 / 轴线＝装配目标；检查真实位置误差与倾角",
    "move_constrained": "青线＝目标行程；黄线＝滑块沿真实导轨的受约束运动",
    "verify_stroke": "青线＝要求区间；黄线＝前序往复运动的实测记录",
    "measure_clearance": "黄色点＝滑块边缘；绿色点＝导轨内缘；青线连接实际余量",
    "plan_recovery": "青箭头＝当前接触状态下规划的上撤方向；绿球＝恢复目标",
    "retract_contact": "从真实孔口接触中上撤；青箭头＝目标；黄线＝实际运动",
    "spiral_search": "绿线＝孔轴；黄线＝真实寻孔路径；橙点＝外部接触",
    "learned_insert": "模仿学习策略闭环插销；绿线＝目标孔轴；黄线＝实际运动",
}


class Suite:
    def __init__(self, out, record=False, only=None):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.record = record
        self.only = set(only or [])
        self.rows = []

    def scene(self, kind):
        return make_demo_session(kind, self.out / "scenes")

    def run(
        self,
        s,
        state,
        name,
        params=None,
        part=None,
        target=None,
        path=None,
        after=None,
        trace=None,
    ):
        s.restore(state)
        if self.only and name not in self.only:
            return
        params = params or {}
        v = Visuals(s, name, part=part, target=target, path=path, note=NOTES[name])
        if trace is not None:
            v.trace = list(trace)
        rec = (
            DemoRecorder(s, self.out / "skills" / (name + ".mp4"), v)
            if self.record
            else None
        )
        if rec:
            rec.hold(0.8)
            s.ctx.on_control_step = rec
        physics_before = s.ctx.snapshot()
        start = len(s.results)
        ok, error = False, ""
        try:
            result = s.call(name, **params)
            v.metrics = result.metrics
            if after:
                after(v)
            ok = True
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            s.ctx.on_control_step = None
            if rec:
                # Computation-only skills get a progressive artifact reveal.
                rec.hold(
                    4.0 if s.ctx.data.time == physics_before["time"] else 1.7,
                    reveal=s.ctx.data.time == physics_before["time"],
                )
                (self.out / "previews").mkdir(exist_ok=True)
                Image.fromarray(rec.peak_image).save(
                    self.out / "previews" / (name + ".png")
                )
                # Verify renderer annotations never change simulation data.
                a = s.ctx.snapshot()
                decorated, bare = rec.frame(), rec.frame(decorate=False)
                b = s.ctx.snapshot()
                assert a["time"] == b["time"]
                assert all(
                    np.array_equal(a["data"][k], b["data"][k]) for k in a["data"]
                )
                overlay_pixels = int(
                    np.count_nonzero(
                        np.max(
                            np.abs(
                                decorated[:907].astype(int) - bare[:907].astype(int)
                            ),
                            axis=2,
                        )
                        > 12
                    )
                )
                rec.close()
            else:
                overlay_pixels = 0
            row = dict(
                skill=name,
                label=CATALOG[name].label,
                scene=Path(s.ctx.scene_path).name,
                success=ok,
                error=error,
                metrics=v.metrics,
                sim_seconds=s.ctx.data.time - physics_before["time"],
                trace_samples=len(v.trace),
                overlay_pixels=overlay_pixels,
                visualization=NOTES[name],
                target=None if v.target is None else v.target,
                planned_path=v.path,
                actual_trace=v.trace,
                result_steps=s.results[start:],
            )
            if rec:
                row.update(
                    video=f"skills/{name}.mp4",
                    seconds=rec.frames / 25,
                    annotation_geoms_max=rec.peak_geoms,
                    sha256=hashlib.sha256(rec.path.read_bytes()).hexdigest(),
                )
            self.rows.append(row)
            (self.out / "verification.json").write_text(
                json.dumps(self.rows, default=serial, ensure_ascii=False, indent=2)
            )
        if not ok:
            raise RuntimeError(name + ": " + error)

    def perception(self):
        s = self.scene("perception")
        initial = s.snapshot()
        self.run(s, initial, "observe_parts")
        s.restore(initial)
        s.call("observe_parts")
        observed = s.snapshot()
        self.run(s, observed, "estimate_pose", {"part": "cube"})
        s.restore(observed)
        s.call("estimate_pose", part="cube")
        posed = s.snapshot()
        self.run(s, posed, "propose_grasps", {"part": "cube"})
        s.restore(posed)
        s.call("propose_grasps", part="cube")
        self.run(s, s.snapshot(), "select_grasp", {"part": "cube", "index": 0})

    def cube(self):
        s = self.scene("cube")
        initial_home = s.ctx.eef_pos()
        s.call("observe_parts")
        s.call("estimate_pose", part="cube")
        s.call("propose_grasps", part="cube")
        s.call("select_grasp", part="cube")
        goal = s.artifacts["grasp"]["xyz"].copy()
        s.call("plan_transfer", target=goal + [0, 0, 0.12])
        s.call("execute_joint_path")
        hover = s.snapshot()
        self.run(
            s,
            hover,
            "approach",
            {"part": "cube"},
            target=goal,
            path=np.linspace(s.ctx.eef_pos(), goal, 30),
        )
        s.restore(hover)
        s.call("approach", part="cube")
        at_cube = s.snapshot()
        self.run(s, at_cube, "close_gripper", {"part": "cube"}, part="cube")
        s.restore(at_cube)
        s.call("close_gripper", part="cube")
        gripped = s.snapshot()
        self.run(s, gripped, "verify_grasp", {"part": "cube"}, part="cube")
        self.run(s, gripped, "open_gripper", part="cube")
        self.run(
            s,
            gripped,
            "lift",
            {"part": "cube", "height": 0.15},
            part="cube",
            target=goal + [0, 0, 0.15],
        )
        s.restore(gripped)
        s.call("lift", part="cube", height=0.15)
        lifted = s.snapshot()
        self.run(
            s,
            lifted,
            "lower",
            {"part": "cube", "height": 0.10},
            part="cube",
            target=s.ctx.obj_pos("cube") - [0, 0, 0.10],
        )
        self.run(s, lifted, "orient_wrist", {"yaw": float(np.pi / 3)})
        s.restore(gripped)
        s.call("open_gripper")
        self.run(
            s,
            s.snapshot(),
            "retreat",
            target=s.ctx.eef_pos() + [0, 0, 0.12],
            params={"height": 0.12},
        )
        self.run(s, hover, "home", target=initial_home)
        s.restore(hover)
        start = s.ctx.eef_pos()
        target = start + [0.23, 0.08, 0.035]
        self.run(
            s,
            hover,
            "plan_linear",
            {"target": target},
            target=target,
            after=lambda v: setattr(
                v,
                "path",
                np.linspace(
                    s.artifacts["linear"]["start"], s.artifacts["linear"]["target"], 50
                ),
            ),
        )
        s.restore(hover)
        s.call("plan_linear", target=target)
        self.run(
            s,
            s.snapshot(),
            "execute_cartesian_path",
            target=target,
            path=np.linspace(start, target, 50),
        )

    def obstacle(self):
        s = self.scene("obstacle")
        s.arm.move([-0.18, -0.18, 0.94], linear=False)
        start = s.snapshot()
        target = np.array([0.14, 0.09, 0.96])
        self.run(
            s,
            start,
            "plan_transfer",
            {"target": target, "clearance": 1.08},
            target=target,
            after=lambda v: setattr(
                v, "path", joint_path_points(s, s.artifacts["transfer"])
            ),
        )
        s.restore(start)
        s.call("plan_transfer", target=target, clearance=1.08)
        path = joint_path_points(s, s.artifacts["transfer"])
        self.run(s, s.snapshot(), "execute_joint_path", target=target, path=path)

    def pin(self, policy):
        s = self.scene("pin")
        part, target = "pin_left", PIN_TARGET.copy()
        pick(s, part)
        transfer_part(s, part, target + [0.012, 0, 0.069])
        off_axis = s.snapshot()
        aligned_target = target + [0, 0, 0.069]
        self.run(
            s,
            off_axis,
            "align_axis",
            {"part": part, "target": aligned_target},
            part=part,
            target=aligned_target,
        )
        s.restore(off_axis)
        s.call("align_axis", part=part, target=aligned_target)
        aligned = s.snapshot()
        self.run(
            s,
            aligned,
            "plan_insertion",
            {"part": part, "target": target},
            part=part,
            target=target,
            after=lambda v: setattr(
                v,
                "path",
                np.linspace(s.ctx.obj_pos(part), s.artifacts["insert"]["target"], 50),
            ),
        )
        # A deliberately misaligned descent must stop on the solid receiver top.
        s.restore(off_axis)
        self.run(
            s,
            off_axis,
            "guarded_descent",
            {"part": part, "target_z": target[2], "force_stop": 2.0},
            part=part,
            target=target,
            path=np.linspace(s.ctx.obj_pos(part), target + [0.012, 0, 0], 50),
        )
        s.restore(off_axis)
        s.call("guarded_descent", part=part, target_z=target[2], force_stop=2.0)
        contacted = s.snapshot()
        self.run(
            s,
            contacted,
            "plan_recovery",
            {"part": part, "height": 0.018},
            part=part,
            after=lambda v: setattr(
                v,
                "target",
                s.artifacts["recovery"]["target"]
                - (s.ctx.eef_pos() - s.ctx.obj_pos(part)),
            ),
        )
        s.restore(contacted)
        s.call("plan_recovery", part=part, height=0.018)
        recovery_goal = s.ctx.obj_pos(part) + [0, 0, 0.018]
        self.run(
            s,
            s.snapshot(),
            "retract_contact",
            {"part": part},
            part=part,
            target=recovery_goal,
        )
        s.restore(contacted)
        s.call("plan_recovery", part=part, height=0.012)
        s.call("retract_contact", part=part)
        s.call("plan_insertion", part=part, target=target)
        self.run(
            s, s.snapshot(), "spiral_search", {"part": part}, part=part, target=target
        )
        s.restore(aligned)
        s.call("plan_insertion", part=part, target=target)
        self.run(
            s,
            s.snapshot(),
            "learned_insert",
            {"part": part, "policy": str(policy)},
            part=part,
            target=target,
        )
        s.restore(aligned)
        s.call("guarded_descent", part=part, target_z=target[2], force_stop=3.0)
        self.run(
            s,
            s.snapshot(),
            "press_seat",
            {"part": part, "target_z": target[2]},
            part=part,
            target=target,
        )
        s.call("open_gripper")
        s.call("retreat")
        self.run(
            s,
            s.snapshot(),
            "inspect_seat",
            {"part": part, "target": target},
            part=part,
            target=target,
        )

    def rail(self):
        s = self.scene("rail")
        part = "carriage"
        pick(s, part)
        entry = np.r_[CENTER + [-0.155, 0], 0.854]
        transfer_part(s, part, entry)
        s.call("lower", part=part, height=0.030)
        s.call("align_axis", part=part, target=np.r_[entry[:2], 0.8255])
        target = np.r_[CENTER + [0.030, 0], 0.8255]
        s.call(
            "plan_insertion",
            part=part,
            target=target,
            axis=(1, 0, 0),
            speed=0.025,
            force_limit=18.0,
        )
        entry_state = s.snapshot()
        self.run(
            s,
            entry_state,
            "slide_insert",
            {"part": part},
            part=part,
            target=target,
            path=np.linspace(s.ctx.obj_pos(part), target, 50),
        )
        s.restore(entry_state)
        s.call("slide_insert", part=part)
        s.call("open_gripper")
        s.call("retreat")
        seated = s.snapshot()
        self.run(s, seated, "measure_clearance", part=part)
        s.restore(seated)
        pick(s, part, lift=False)
        gripped = s.snapshot()
        start = s.ctx.obj_pos(part)
        left = start.copy()
        left[0] = CENTER[0] - 0.04
        self.run(
            s,
            gripped,
            "move_constrained",
            {"part": part, "target_x": left[0]},
            part=part,
            target=left,
            path=np.linspace(start, left, 50),
        )
        s.restore(gripped)
        trace = []
        s.ctx.on_control_step = lambda ctx: trace.append(ctx.obj_pos(part).copy())
        s.call("move_constrained", part=part, target_x=float(CENTER[0] - 0.04))
        s.call("move_constrained", part=part, target_x=float(CENTER[0] + 0.06))
        s.ctx.on_control_step = None
        right = left.copy()
        right[0] = CENTER[0] + 0.06
        self.run(
            s,
            s.snapshot(),
            "verify_stroke",
            {"minimum": 0.09},
            part=part,
            path=np.linspace(left, right, 50),
            trace=trace,
        )


def record_cube_grasp(out):
    """An additional three-atom example, not an extra atomic skill."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    s = make_demo_session("cube", out / "scenes")
    for name in ("observe_parts", "estimate_pose", "propose_grasps", "select_grasp"):
        s.call(name, **({} if name == "observe_parts" else {"part": "cube"}))
    goal = s.artifacts["grasp"]["xyz"].copy()
    s.call("plan_transfer", target=goal + [0, 0, 0.12])
    s.call("execute_joint_path")
    v = Visuals(
        s,
        "approach",
        target=goal,
        path=np.linspace(s.ctx.eef_pos(), goal, 30),
        note=NOTES["approach"],
    )
    rec = DemoRecorder(s, out / "cube_grasp.mp4", v)
    first = len(s.results)
    try:
        for name, params in [
            ("approach", {"part": "cube"}),
            ("close_gripper", {"part": "cube"}),
            ("lift", {"part": "cube", "height": 0.15}),
        ]:
            if name != "approach":
                v = Visuals(
                    s,
                    name,
                    part="cube",
                    note=NOTES[name],
                    target=goal + [0, 0, 0.15] if name == "lift" else None,
                )
                rec.visuals = v
            rec.hold(0.8)
            s.ctx.on_control_step = rec
            result = s.call(name, **params)
            v.metrics = result.metrics
            s.ctx.on_control_step = None
            rec.hold(1.6)
        Image.fromarray(rec.frame()).save(out / "cube_grasp.png")
        (out / "cube_grasp_verification.json").write_text(
            json.dumps(
                dict(
                    success=True,
                    composite_of=["approach", "close_gripper", "lift"],
                    seconds=rec.frames / 25,
                    results=s.results[first:],
                ),
                default=serial,
                indent=2,
            )
        )
    finally:
        s.ctx.on_control_step = None
        rec.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/standalone_skills")
    p.add_argument("--record", action="store_true")
    p.add_argument("--cube-grasp", action="store_true")
    p.add_argument(
        "--groups", nargs="+", default=["perception", "cube", "obstacle", "pin", "rail"]
    )
    p.add_argument("--only", nargs="+")
    p.add_argument("--policy", default=str(DEFAULT_POLICY))
    a = p.parse_args()
    if a.cube_grasp:
        record_cube_grasp(a.out)
        return
    suite = Suite(a.out, a.record, a.only)
    for group in a.groups:
        print("DEMO GROUP " + group, flush=True)
        if group == "pin":
            suite.pin(a.policy)
        else:
            getattr(suite, group)()
    print(
        json.dumps(
            dict(
                success=all(r["success"] for r in suite.rows),
                skills=len(suite.rows),
                out=a.out,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
