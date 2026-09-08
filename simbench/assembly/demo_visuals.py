"""Read-only MuJoCo scene decorations sourced from real skill artifacts/state.

The geoms below live only in MjvScene, never in MjModel or MjData. Planning
reveals can advance presentation time without advancing simulation time.
"""

from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw, ImageFont
from .library import CATALOG

CYAN = np.array([0.05, 0.8, 1.0, 0.9])
GREEN = np.array([0.1, 1.0, 0.35, 0.85])
YELLOW = np.array([1.0, 0.78, 0.1, 0.95])
ORANGE = np.array([1.0, 0.35, 0.08, 0.9])
GREY = np.array([0.7, 0.72, 0.78, 0.35])
RGB = ([1.0, 0.18, 0.18, 0.9], [0.2, 1.0, 0.25, 0.9], [0.2, 0.5, 1.0, 0.9])


def joint_path_points(s, plan):
    """Forward kinematics of the actual joint-interpolated path, in scratch data."""
    d = mujoco.MjData(s.ctx.model)
    d.qpos[:] = s.ctx.data.qpos
    q0 = np.asarray(plan["start_q"])
    points = []
    for q1 in plan["joints"]:
        for t in np.linspace(0, 1, 32):
            # Quintic time scaling changes timing, not this geometric curve.
            d.qpos[s.ctx.arm_qadr] = q0 + t * (q1 - q0)
            mujoco.mj_forward(s.ctx.model, d)
            points.append(d.site_xpos[s.ctx.eef_site_id].copy())
        q0 = q1
    return np.array(points)


class Visuals:
    def __init__(self, session, mode, part=None, target=None, path=None, note=""):
        self.s = session
        self.mode = mode
        self.part = part
        self.target = None if target is None else np.asarray(target, float)
        self.path = [] if path is None else np.asarray(path, float)
        self.trace = []
        self.note = note
        self.metrics = {}
        self.reveal = 1.0
        self.geom_counts = []
        self.contact_samples = []
        self.start = self.position()

    def position(self):
        return (
            self.s.ctx.obj_pos(self.part) if self.part else self.s.ctx.eef_pos()
        ).copy()

    def sample(self):
        self.trace.append(self.position())
        if self.part:
            self.contact_samples.append(self.s.ctx.grasp_contacts(self.part))

    def add(self, scene, kind, pos, size, color, mat=None):
        if scene.ngeom >= scene.maxgeom:
            raise RuntimeError("visualization geometry capacity exceeded")
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            g,
            kind,
            np.asarray(size, float),
            np.asarray(pos, float),
            np.eye(3).ravel() if mat is None else np.asarray(mat).ravel(),
            np.asarray(color, np.float32),
        )
        g.category = mujoco.mjtCatBit.mjCAT_DECOR
        g.segid = -1
        g.specular = 0.15
        scene.ngeom += 1
        return g

    def sphere(self, scene, p, radius=0.015, color=GREEN):
        return self.add(scene, mujoco.mjtGeom.mjGEOM_SPHERE, p, [radius] * 3, color)

    def line(self, scene, a, b, color=CYAN, radius=0.002, arrow=False):
        if np.linalg.norm(np.asarray(b) - a) < 1e-7:
            return
        kind = mujoco.mjtGeom.mjGEOM_ARROW if arrow else mujoco.mjtGeom.mjGEOM_CAPSULE
        g = self.add(scene, kind, [0, 0, 0], [1, 1, 1], color)
        mujoco.mjv_makeConnector(g, kind, radius, *np.asarray(a), *np.asarray(b))

    def polyline(self, scene, points, color=CYAN, radius=0.0018, reveal=1.0):
        points = np.asarray(points)
        points = points[: max(1, int(len(points) * reveal))]
        if len(points) > 180:
            points = points[np.linspace(0, len(points) - 1, 180).astype(int)]
        for a, b in zip(points[:-1], points[1:]):
            self.line(scene, a, b, color, radius)

    def axes(self, scene, xyz, R=np.eye(3), length=0.07):
        for i, color in enumerate(RGB):
            self.line(scene, xyz, xyz + R[:, i] * length, color, 0.003, True)

    def grasp(self, scene, candidate, color):
        p = np.asarray(candidate["xyz"])
        yaw = candidate["yaw"]
        closing = np.array([-np.sin(yaw), np.cos(yaw), 0])
        width = candidate["width"] + 0.010
        left, right = p - closing * width / 2, p + closing * width / 2
        up = np.array([0, 0, 0.048])
        self.line(scene, left, left + up, color, 0.0028)
        self.line(scene, right, right + up, color, 0.0028)
        self.line(scene, left + up, right + up, color, 0.0028)
        self.line(scene, p + [0, 0, 0.09], p + [0, 0, 0.055], color, 0.003, True)

    def contacts(self, scene):
        ctx = self.s.ctx
        bid = ctx.body_id(self.part)
        for i, c in enumerate(ctx.data.contact):
            bodies = ctx.model.geom_bodyid[[c.geom1, c.geom2]]
            if bid not in bodies:
                continue
            f = np.zeros(6)
            mujoco.mj_contactForce(ctx.model, ctx.data, i, f)
            if f[0] > 0.05:
                other = int(bodies[0] if bodies[1] == bid else bodies[1])
                color = GREEN if "finger" in ctx.model.body(other).name else ORANGE
                self.sphere(scene, c.pos, 0.006, color)

    def draw(self, scene):
        before = scene.ngeom
        s, ctx = self.s, self.s.ctx
        mode = self.mode
        if mode == "observe_parts":
            colors = (
                [0.15, 0.9, 1.0, 0.28],
                [0.5, 1.0, 0.15, 0.28],
                [1.0, 0.2, 0.7, 0.28],
            )
            for i, (part, points) in enumerate(s.observations.items()):
                if (i + 1) / len(s.observations) > self.reveal + 0.01:
                    continue
                xyz = np.median(points, axis=0)
                self.sphere(scene, xyz, 0.037, colors[i % len(colors)])
                self.sphere(scene, xyz, 0.005, colors[i % len(colors)][:3] + [1.0])
        elif mode == "estimate_pose" and "pose" in s.artifacts:
            p = s.artifacts["pose"]
            q = np.asarray(p["quat"])
            self.axes(scene, p["xyz"], Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix())
            self.sphere(scene, p["xyz"], 0.025, [0.1, 0.9, 1.0, 0.2])
        elif mode in ("propose_grasps", "select_grasp") and "grasps" in s.artifacts:
            for i, candidate in enumerate(s.artifacts["grasps"]["candidates"]):
                if (i + 1) / len(
                    s.artifacts["grasps"]["candidates"]
                ) > self.reveal + 0.01:
                    continue
                selected = mode == "select_grasp" and "grasp" in s.artifacts
                chosen = selected and np.isclose(
                    candidate["yaw"], s.artifacts["grasp"]["yaw"]
                )
                self.grasp(
                    scene,
                    candidate,
                    GREEN if chosen else GREY if selected else (CYAN, ORANGE)[i % 2],
                )
        if len(self.path):
            self.polyline(scene, self.path, CYAN, reveal=self.reveal)
            self.sphere(scene, self.path[0], 0.009, CYAN)
        if self.target is not None:
            self.sphere(
                scene,
                self.target,
                0.033 if mode == "approach" else 0.016,
                [0.1, 1.0, 0.35, 0.3],
            )
            if (
                self.part
                and self.part.startswith("pin_")
                and mode not in ("plan_recovery", "retract_contact")
            ):
                from .demo_scenes import PIN_TARGET

                p = PIN_TARGET.copy()
                self.line(scene, p - [0, 0, 0.02], p + [0, 0, 0.11], GREEN, 0.0018)
                angles = np.linspace(0, 2 * np.pi, 40)
                circle = np.c_[
                    p[0] + 0.012 * np.cos(angles),
                    p[1] + 0.012 * np.sin(angles),
                    np.full(40, 0.856),
                ]
                self.polyline(scene, circle, GREEN, 0.001)
            if mode == "inspect_seat" and self.part:
                self.line(scene, self.position(), self.target, ORANGE, 0.002)
        if len(self.trace) > 1:
            self.polyline(scene, self.trace, YELLOW, 0.0017)
        if (
            mode
            in (
                "close_gripper",
                "verify_grasp",
                "open_gripper",
                "guarded_descent",
                "press_seat",
                "spiral_search",
                "learned_insert",
                "retract_contact",
            )
            and self.part
        ):
            self.contacts(scene)
        if (
            mode
            in (
                "approach",
                "lift",
                "lower",
                "retreat",
                "align_axis",
                "plan_recovery",
                "retract_contact",
            )
            and self.target is not None
        ):
            offset = (
                np.array([0.065, -0.085, 0])
                if mode in ("approach", "lift", "lower", "retreat")
                else np.zeros(3)
            )
            # Offset dimension arrows remain readable beside the hand; leaders
            # tie both ends to the true positions. Yellow traces stay unshifted.
            self.line(
                scene, self.start + offset, self.target + offset, CYAN, 0.004, True
            )
            if np.any(offset):
                self.line(scene, self.start, self.start + offset, CYAN, 0.0008)
                self.line(scene, self.target, self.target + offset, CYAN, 0.0008)
        if mode == "orient_wrist":
            self.axes(scene, ctx.eef_pos(), ctx.eef_mat(), 0.10)
        if mode == "measure_clearance":
            from .scene import CENTER

            p = ctx.obj_pos("carriage")
            # Actual edge-to-guide clearance, with offset leaders for readability.
            for sign in (-1, 1):
                a = np.array([p[0], p[1] + sign * 0.023, 0.853])
                b = np.array([p[0], CENTER[1] + sign * 0.027, 0.853])
                self.sphere(scene, a, 0.0025, YELLOW)
                self.sphere(scene, b, 0.0025, GREEN)
                self.line(scene, a, b, CYAN, 0.0012)
                self.line(scene, b, b + [0.035, sign * 0.025, 0.02], CYAN, 0.001)
        self.geom_counts.append(scene.ngeom - before)

    def description(self):
        ctx, m = self.s.ctx, self.metrics
        if self.mode in ("close_gripper", "verify_grasp", "open_gripper"):
            f = ctx.grasp_contacts(self.part)
            return f"{self.note}  |  夹爪宽度 {ctx.pad_span()*1000:.1f} mm · 左/右接触 {f['left_n']:.1f}/{f['right_n']:.1f} N"
        if self.mode in (
            "guarded_descent",
            "press_seat",
            "spiral_search",
            "learned_insert",
        ):
            return f"{self.note}  |  外部接触 {self.s.external_force(self.part):.2f} N"
        if self.mode == "inspect_seat" and m:
            return f"{self.note}  |  位置误差 {m['position_error_m']*1000:.2f} mm"
        if self.mode == "measure_clearance" and m:
            return f"{self.note}  |  最小侧向余量 {m['lateral_margin_m']*1000:.2f} mm"
        if self.mode == "verify_stroke" and m:
            return f"{self.note}  |  实测行程 {m['measured_range_m']*1000:.1f} mm"
        return self.note


class DemoRecorder:
    def __init__(self, session, path, visuals):
        self.ctx = session.ctx
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.visuals = visuals
        self.renderer = mujoco.Renderer(self.ctx.model, height=1000, width=1600)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[3:] = 0
        self.font = ImageFont.truetype(
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 28
        )
        self.small = ImageFont.truetype(
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 23
        )
        self.writer = imageio.get_writer(
            str(path), fps=25, codec="libx264", quality=8, macro_block_size=2
        )
        self.n = self.frames = 0
        self.peak_image = None
        self.peak_geoms = -1

    def frame(self, decorate=True):
        self.renderer.update_scene(
            self.ctx.data, camera="task_view", scene_option=self.option
        )
        if decorate:
            self.visuals.draw(self.renderer._scene)
        im = Image.fromarray(self.renderer.render())
        draw = ImageDraw.Draw(im)
        draw.rectangle([0, 907, 1600, 1000], fill=(25, 29, 34))
        spec = CATALOG[self.visuals.mode]
        draw.text(
            (24, 914),
            f"{spec.label}  |  {spec.name}",
            font=self.font,
            fill=(247, 249, 251),
        )
        draw.text(
            (24, 959), self.visuals.description(), font=self.small, fill=(197, 222, 234)
        )
        return np.asarray(im)

    def emit(self):
        frame = self.frame()
        self.writer.append_data(frame)
        self.frames += 1
        n = self.visuals.geom_counts[-1]
        if n >= self.peak_geoms:
            self.peak_geoms = n
            self.peak_image = frame.copy()

    def __call__(self, ctx):
        self.visuals.sample()
        self.n += 1
        if self.n % 2 == 0:
            self.emit()

    def hold(self, seconds=2.0, reveal=False):
        count = max(1, int(seconds * 25))
        for i in range(count):
            self.visuals.reveal = min(1.0, (i + 1) / (count * 0.7)) if reveal else 1.0
            self.emit()

    def close(self):
        self.writer.close()
        self.renderer._gl_context.make_current()
        self.renderer._mjr_context.free()
        self.renderer._gl_context.free()
