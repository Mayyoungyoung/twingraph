"""Eleven close-up, fixed-camera demonstrations of the public atom interfaces.

All preparation and execution use the existing controllers. Decorations only
touch MjvScene; computation-only videos reveal real outputs without moving time.
"""

import argparse
import hashlib
import json
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .demo_scenes import make_demo_session, PIN_TARGET
from .demo_visuals import Visuals, joint_path_points, CYAN, GREEN, YELLOW, ORANGE
from .interfaces import PUBLIC_SKILLS
from .control import DEFAULT_POLICY
from .task import pick, transfer_part
from .scene import CENTER


MODES = dict(
    detect="observe_parts",
    estimate_pose="estimate_pose",
    estimate_grasp="propose_grasps",
    plan_path="plan_transfer",
    move="execute_joint_path",
    grasp="close_gripper",
    place="open_gripper",
    insert="learned_insert",
    press="press_seat",
    measure="measure_clearance",
    inspect="inspect_seat",
)
NOTES = {
    "detect": "半透明彩球：检测到的对象；实心点：观测中心（仿真位姿观测）",
    "estimate_pose": "红 / 绿 / 蓝粗箭头：物体 X / Y / Z 轴；球心：估计位置",
    "estimate_grasp": "青 / 橙夹爪框：实际生成的两种抓法；箭头：接近方向",
    "plan_path": "青 / 橙 / 紫粗线：不同高度的候选路径；绿球：同一个目标",
    "move": "青色粗线：计划路径；黄色轨迹：机械臂实际运动；灰块：障碍物",
    "grasp": "观察两侧手指闭合；绿色点：与方块的真实接触",
    "place": "方块已位于支撑面：张开手指，并检查释放后的位置",
    "insert": "观察销轴进入孔座；旁侧尺寸箭头显示实际插入进度（行为克隆策略）",
    "press": "压靠位移很小：观察接触力上升，同时检查肩面高度",
    "measure": "黄点 / 绿点：滑块 / 导轨边缘的上方投影；引线指向真实测量位置",
    "inspect": "绿环：装配目标；检查实际位置误差与倾角是否满足要求",
}


class AtomVisuals(Visuals):
    def __init__(self, session, atom, part=None, target=None, path=None):
        # Pin references are drawn beside the contact, so thick lines do not hide it.
        super().__init__(
            session,
            MODES[atom],
            part=part,
            target=None if atom in ("insert", "press", "inspect", "place") else target,
            path=path,
            note=NOTES[atom],
        )
        self.atom = atom
        self.goal = None if target is None else np.asarray(target).copy()
        self.routes = []
        self.done = False

    def line(self, scene, a, b, color=CYAN, radius=0.002, arrow=False):
        super().line(scene, a, b, color, radius * 2.8, arrow)

    def axes(self, scene, xyz, R=np.eye(3), length=0.16):
        # A longer direction reference protrudes beyond the 40 mm teaching cube.
        super().axes(scene, xyz, R, length)

    def sphere(self, scene, p, radius=0.015, color=GREEN):
        if self.atom in ("insert", "press") and radius == 0.006:
            radius = 0.0025  # Contact dots must not conceal the narrow pin.
        return super().sphere(scene, p, radius, color)

    def draw(self, scene):
        before = scene.ngeom
        trace = self.trace
        if self.atom in ("insert", "press", "inspect", "place"):
            self.trace = []
        mode = self.mode
        if self.atom == "measure":
            self.mode = "custom_clearance"
        super().draw(scene)
        self.mode = mode
        self.trace = trace
        if self.atom == "plan_path":
            colors = (CYAN, ORANGE, [0.72, 0.35, 1.0, 0.95])
            for i, route in enumerate(self.routes):
                self.polyline(scene, route, colors[i % 3], 0.0022, self.reveal)
            if self.routes:
                self.sphere(scene, self.routes[0][0], 0.014, CYAN)
        if self.atom in ("insert", "press"):
            current = self.position()
            offset = np.array([0.055, -0.018, 0.0])
            bottom = self.goal + offset
            top = self.start + offset
            self.line(scene, bottom, top, [0.15, 0.9, 1.0, 0.6], 0.0013)
            self.line(
                scene, top, current + offset, YELLOW, 0.0018, self.atom == "insert"
            )
            self.line(scene, current, current + offset, CYAN, 0.0006)
            self.line(scene, self.goal, bottom, GREEN, 0.0006)
            self.sphere(scene, bottom, 0.004, GREEN)
        if self.atom in ("place", "inspect") and self.goal is not None:
            radius = 0.032 if self.atom == "place" else 0.024
            angles = np.linspace(0, 2 * np.pi, 64)
            z = 0.802 if self.atom == "place" else 0.857
            ring = np.c_[
                self.goal[0] + radius * np.cos(angles),
                self.goal[1] + radius * np.sin(angles),
                np.full(len(angles), z),
            ]
            self.polyline(scene, ring, GREEN, 0.001)
        if self.atom == "measure" and self.done:
            p = self.position()
            for sign in (-1, 1):
                a = np.array([p[0] + 0.027, p[1] + sign * 0.023, p[2]])
                b = np.array([a[0], CENTER[1] + sign * 0.027, p[2]])
                offset = np.array([0.06, 0.0, 0.095])
                self.line(scene, a, a + offset, YELLOW, 0.0005)
                self.line(scene, b, b + offset, GREEN, 0.0005)
                self.line(scene, a + offset, b + offset, CYAN, 0.00035)
                self.sphere(scene, a + offset, 0.0015, YELLOW)
                self.sphere(scene, b + offset, 0.0015, GREEN)
        self.geom_counts[-1] = scene.ngeom - before

    def status(self):
        s, m = self.s, self.metrics
        if self.atom == "detect":
            return (
                f"检测结果：{len(s.observations)} 个对象" if m else "正在读取场景观测"
            )
        if self.atom == "estimate_pose" and m:
            p = m["position_m"]
            return f"位置：({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) m"
        if self.atom == "estimate_grasp" and m:
            return f"抓取候选：{m['feasible_candidates']} 个；保留不同朝向"
        if self.atom == "plan_path" and m:
            return f"已生成并通过离散检查：{len(self.routes)} 条路径；规划阶段机械臂保持不动"
        if self.atom == "move":
            error = np.linalg.norm(s.ctx.eef_pos() - self.goal) * 1000
            return f"距离目标：{error:.1f} mm" + (
                "  ·  到达检查通过" if self.done else ""
            )
        if self.atom in ("grasp", "place"):
            c = s.ctx.grasp_contacts(self.part)
            return (
                f"夹爪宽度：{s.ctx.pad_span()*1000:.1f} mm    左 / 右接触：{c['left_n']:.2f} / {c['right_n']:.2f} N"
                + ("    检查通过" if self.done else "")
            )
        if self.atom == "insert":
            travel = (self.start[2] - self.position()[2]) * 1000
            return f"实际推进：{travel:.1f} mm    外部接触：{s.external_force(self.part):.2f} N" + (
                "    插入完成" if self.done else ""
            )
        if self.atom == "press":
            error = abs(self.position()[2] - self.goal[2]) * 1000
            return (
                f"外部接触：{s.external_force(self.part):.2f} N    停止力阈值：2.5 N    高度误差：{error:.2f} mm"
                + ("    压靠通过" if self.done else "")
            )
        if self.atom == "measure" and m:
            return f"最小侧向间隙：{float(m['value'])*1000:.2f} mm    输出带单位的数值，尚未作验收判定"
        if self.atom == "inspect" and m:
            return f"位置误差：{m['position_error_m']*1000:.3f} / 1.500 mm    倾角：{m['tilt_deg']:.3f} / 3.000°    验收通过"
        return "输入已就绪"


class AtomRecorder:
    def __init__(self, s, path, visuals):
        self.ctx, self.path, self.visuals = s.ctx, Path(path), visuals
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.renderer = mujoco.Renderer(s.ctx.model, height=1000, width=1600)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[3:] = 0
        # Clear instructional views: avoid the wrist shadow concealing the hole.
        font = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
        self.title_font = ImageFont.truetype(font, 32)
        self.text_font = ImageFont.truetype(font, 25)
        self.writer = imageio.get_writer(
            str(path), fps=25, codec="libx264", quality=9, macro_block_size=2
        )
        self.frames, self.n = 0, 0

    def frame(self, decorate=True):
        self.renderer.update_scene(
            self.ctx.data, camera="task_view", scene_option=self.option
        )
        self.renderer._scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        if decorate:
            self.visuals.draw(self.renderer._scene)
        im = Image.fromarray(self.renderer.render())
        draw = ImageDraw.Draw(im)
        draw.rectangle([0, 850, 1600, 1000], fill=(23, 28, 34))
        atom = self.visuals.atom
        draw.text(
            (28, 863),
            f"当前技能：{PUBLIC_SKILLS[atom]['label']}  |  {atom}",
            font=self.title_font,
            fill=(248, 250, 252),
        )
        draw.text((28, 912), NOTES[atom], font=self.text_font, fill=(174, 218, 240))
        draw.text(
            (28, 954), self.visuals.status(), font=self.text_font, fill=(231, 239, 244)
        )
        if atom == "measure" and self.visuals.done:
            draw.rounded_rectangle([34, 490, 390, 688], radius=10, fill=(23, 28, 34))
            draw.text(
                (52, 502), "间隙示意（放大）", font=self.text_font, fill=(238, 242, 248)
            )
            draw.rectangle([72, 558, 159, 604], fill=(255, 199, 25))
            draw.rectangle([248, 558, 345, 604], fill=(25, 230, 89))
            draw.line([164, 582, 243, 582], fill=(55, 211, 245), width=5)
            draw.line([164, 572, 164, 592], fill=(55, 211, 245), width=4)
            draw.line([243, 572, 243, 592], fill=(55, 211, 245), width=4)
            value = self.visuals.metrics["value"] * 1000
            draw.text(
                (111, 628), f"{value:.2f} mm", font=self.text_font, fill=(238, 242, 248)
            )
        if atom == "press":
            force = self.visuals.s.external_force(self.visuals.part)
            draw.rounded_rectangle([1190, 55, 1568, 180], radius=10, fill=(23, 28, 34))
            draw.text(
                (1210, 65),
                f"外部接触力 {force:.2f} N",
                font=self.text_font,
                fill=(238, 242, 248),
            )
            draw.rectangle([1210, 117, 1548, 139], fill=(75, 83, 93))
            draw.rectangle(
                [1210, 117, 1210 + 338 * min(1.0, force / 2.5), 139],
                fill=(45, 218, 145),
            )
            draw.text(
                (1210, 143),
                "0                  2.5 N 停止",
                font=self.text_font,
                fill=(200, 210, 222),
            )
        return np.asarray(im)

    def emit(self):
        self.writer.append_data(self.frame())
        self.frames += 1

    def __call__(self, ctx):
        self.visuals.sample()
        self.n += 1
        if self.n % 2 == 0:
            self.emit()

    def hold(self, seconds, reveal=False):
        for i in range(int(seconds * 25)):
            self.visuals.reveal = (
                min(1.0, (i + 1) / max(1.0, seconds * 25 * 0.65)) if reveal else 1.0
            )
            self.emit()

    def close(self):
        self.writer.close()
        self.renderer._gl_context.make_current()
        self.renderer._mjr_context.free()
        self.renderer._gl_context.free()


class Suite:
    def __init__(self, out, only=None, record=True):
        self.out, self.only, self.record = Path(out), set(only or PUBLIC_SKILLS), record
        self.out.mkdir(parents=True, exist_ok=True)
        self.rows = []

    def scene(self, kind):
        return make_demo_session(kind, self.out / "scenes", view="front")

    def run(self, s, atom, params=None, part=None, target=None, path=None, after=None):
        if atom not in self.only:
            return
        v = AtomVisuals(s, atom, part, target, path)
        rec = (
            AtomRecorder(s, self.out / "skills" / (atom + ".mp4"), v)
            if self.record
            else None
        )
        before, first = s.ctx.snapshot(), len(s.results)
        if rec:
            rec.hold(1.0)
            s.ctx.on_control_step = rec
        try:
            result = s.call(atom, **(params or {}))
            v.metrics, v.done = result.metrics, True
            if after:
                after(v)
            s.ctx.on_control_step = None
            computation = s.ctx.data.time == before["time"]
            if computation:
                for key, value in before["data"].items():
                    np.testing.assert_array_equal(s.ctx.snapshot()["data"][key], value)
            if rec:
                rec.hold(5.0 if computation else 2.2, reveal=computation)
                state = s.ctx.snapshot()
                decorated, bare = rec.frame(), rec.frame(False)
                for key, value in state["data"].items():
                    np.testing.assert_array_equal(s.ctx.snapshot()["data"][key], value)
                overlay = int(
                    np.count_nonzero(
                        np.max(
                            np.abs(
                                decorated[:850].astype(int) - bare[:850].astype(int)
                            ),
                            axis=2,
                        )
                        > 12
                    )
                )
                (self.out / "previews").mkdir(exist_ok=True)
                Image.fromarray(decorated).save(self.out / "previews" / (atom + ".png"))
            else:
                overlay = 0
            row = dict(
                atom=atom,
                label=PUBLIC_SKILLS[atom]["label"],
                success=result.ok,
                scene=Path(s.ctx.scene_path).name,
                camera="fixed_front_closeup",
                resolution=[1600, 1000],
                fps=25,
                seconds=rec.frames / 25 if rec else None,
                sim_seconds=s.ctx.data.time - before["time"],
                computation_only=computation,
                overlay_pixels=overlay,
                metrics=result.metrics,
                results=s.results[first:],
                route_count=len(v.routes),
                line_scale=2.8,
            )
            if rec:
                rec.close()
                rec = None
                row["video"] = "skills/" + atom + ".mp4"
                row["sha256"] = hashlib.sha256(
                    (self.out / row["video"]).read_bytes()
                ).hexdigest()
            self.rows.append(row)
            (self.out / "verification.json").write_text(
                json.dumps(
                    self.rows,
                    ensure_ascii=False,
                    indent=2,
                    default=lambda x: np.asarray(x).tolist(),
                )
            )
        finally:
            s.ctx.on_control_step = None
            if rec:
                rec.close()

    def perception(self):
        s = self.scene("perception")
        # Preparation is not part of the video and still uses the real sensor backend.
        self.run(s, "detect")
        if not s.observations:
            s.call("detect")
        self.run(s, "estimate_pose", dict(part="cube"), part="cube")
        if "pose" not in s.artifacts:
            s.call("estimate_pose", part="cube")
        self.run(s, "estimate_grasp", dict(part="cube"), part="cube")

    def obstacle(self):
        s = self.scene("obstacle")
        s.arm.move([-0.18, -0.18, 0.94], linear=False)
        target = np.array([0.14, 0.09, 0.96])

        def reveal(v):
            v.routes = [
                joint_path_points(s, c["path"])
                for c in s.artifacts["transfer_candidates"]["candidates"]
                if c["status"] == "necessary_pass"
            ]

        self.run(
            s,
            "plan_path",
            dict(target=target, clearance=1.08),
            target=target,
            after=reveal,
        )
        if "transfer" not in s.artifacts:
            s.call("plan_path", target=target, clearance=1.08)
        self.run(
            s,
            "move",
            dict(path="transfer"),
            target=target,
            path=joint_path_points(s, s.artifacts["transfer"]),
        )

    def cube(self):
        s = self.scene("cube")
        s.call("detect")
        s.call("estimate_pose", part="cube")
        s.call("estimate_grasp", part="cube")
        s.call("select_grasp", part="cube", index=0)
        goal = s.artifacts["grasp"]["xyz"]
        s.call("plan_path", target=goal + [0, 0, 0.10])
        s.call("move", path="transfer")
        s.call("move", part="cube", grasp="grasp")
        self.run(s, "grasp", dict(part="cube"), part="cube")
        if s.held is None:
            s.call("grasp", part="cube")
        target = s.ctx.obj_pos("cube").copy()
        self.run(
            s, "place", dict(part="cube", target=target), part="cube", target=target
        )

    def pin(self):
        s = self.scene("pin")
        part, target = "pin_left", PIN_TARGET.copy()
        pick(s, part)
        transfer_part(s, part, target + [0, 0, 0.069])
        s.call("move", reference="object", part=part, target=target + [0, 0, 0.069])
        s.call("plan_path", method="contact", part=part, target=target)
        aligned = s.snapshot()
        self.run(
            s,
            "insert",
            dict(part=part, strategy="learned", policy=str(DEFAULT_POLICY)),
            part=part,
            target=target,
        )
        s.restore(aligned)
        s.call("move", mode="guarded", part=part, target_z=target[2], force_stop=3.0)
        self.run(
            s, "press", dict(part=part, target_z=target[2]), part=part, target=target
        )
        if "press" not in self.only:
            s.call("press", part=part, target_z=target[2])
        s.call("place", part=part, target=target)
        s.call("move", delta=[0, 0, 0.10])
        self.run(s, "inspect", dict(part=part, target=target), part=part, target=target)

    def rail(self):
        s = self.scene("rail")
        pick(s, "carriage")
        entry = np.r_[CENTER + [-0.155, 0], 0.854]
        transfer_part(s, "carriage", entry)
        s.call("move", part="carriage", delta=[0, 0, -0.030])
        s.call(
            "move", reference="object", part="carriage", target=np.r_[entry[:2], 0.8255]
        )
        target = np.r_[CENTER + [0.030, 0], 0.8255]
        s.call(
            "plan_path",
            method="contact",
            part="carriage",
            target=target,
            axis=(1, 0, 0),
            speed=0.025,
            force_limit=18.0,
        )
        s.call("insert", part="carriage")
        s.call("press", part="carriage", target_z=0.824)
        s.call("place", part="carriage", target=np.r_[target[:2], 0.824])
        s.call("move", delta=[0, 0, 0.10])
        self.run(
            s, "measure", dict(quantity="clearance", part="carriage"), part="carriage"
        )

    def all(self):
        for atoms, method in [
            ({"detect", "estimate_pose", "estimate_grasp"}, self.perception),
            ({"plan_path", "move"}, self.obstacle),
            ({"grasp", "place"}, self.cube),
            ({"insert", "press", "inspect"}, self.pin),
            ({"measure"}, self.rail),
        ]:
            if atoms & self.only:
                method()
        assert {r["atom"] for r in self.rows} == self.only


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/atomic_closeup")
    p.add_argument("--only", nargs="+", choices=list(PUBLIC_SKILLS))
    p.add_argument("--no-record", action="store_true")
    a = p.parse_args()
    suite = Suite(a.out, a.only, record=not a.no_record)
    suite.all()
    print(json.dumps(dict(success=True, atoms=len(suite.rows), out=a.out)))


if __name__ == "__main__":
    main()
