"""Robot-only surface preparation and complete assembly, with honest fault cases."""

import argparse
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont
from .scene import SCENE, CENTER, geom, fmt
from .control import HOME, DEFAULT_POLICY
from .library import Session, PARTS, GRASP, DEFAULT_CAPABILITIES, SkillFailure
from ..core.sim_context import MjContext
from .task import pick, transfer_part, assemble
from .recording import Recorder

TOOL_HOME = np.array([-0.31, 0.20, 0.804])


def make_session(out, isolated=False, recorder=False, failure=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    root = ET.parse(SCENE).getroot()
    world = root.find("worldbody")
    if isolated:
        for body in list(world.findall("body")):
            if body.get("name") != "guide_base":
                world.remove(body)
    assets = SCENE.parent.parent / "assets/panda"
    root.find("compiler").set("meshdir", os.path.relpath(assets, out.resolve()))
    root.find("include").set(
        "file", os.path.relpath(assets / "panda.xml", out.resolve())
    )
    tool = ET.SubElement(world, "body", name="wipe_tool", pos=fmt(TOOL_HOME))
    ET.SubElement(tool, "freejoint", name="wipe_tool_free")
    geom(
        tool,
        "wipe_pad",
        [0, 0, 0],
        [0.018, 0.012, 0.004],
        ".16 .66 .68 1",
        friction=".25 .01 .001",
        mass=".025",
        solref=".03 1",
        solimp=".8 .95 .002",
    )
    geom(
        tool,
        "wipe_handle",
        [0, 0, 0.047],
        [0.012, 0.014, 0.022],
        ".85 .65 .25 1",
        friction="1 .02 .001",
        mass=".045",
    )
    # Cosmetic stem connects the two rigidly attached collision shapes.
    geom(
        tool,
        "wipe_stem",
        [0, 0, 0.014],
        [0.006, 0.007, 0.011],
        ".85 .65 .25 1",
        contype="0",
        conaffinity="0",
        mass="0",
    )
    cam = world.find("camera")
    eye = np.array([0.90, -1.18, 1.65])
    target = np.array([-0.16, 0.0, 1.075])
    fovy = 40
    if isolated:
        eye = np.array([0.48, -0.35, 1.21])
        target = np.array([0.055, 0.075, 0.91])
        fovy = 39
    z = (eye - target) / np.linalg.norm(eye - target)
    x = np.cross([0, 0, 1], z)
    x /= np.linalg.norm(x)
    cam.set("pos", fmt(eye))
    cam.set("xyaxes", fmt(np.r_[x, np.cross(z, x)]))
    cam.set("fovy", str(fovy))
    root.find("visual/global").set("offwidth", "1920")
    root.find("visual/global").set("offheight", "1200")
    root.find("visual/quality").set("offsamples", "4")
    root.find("visual/headlight").set("ambient", ".35 .35 .35")
    scene = out / "scene.xml"
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(scene, encoding="unicode")
    ctx = MjContext(scene, control_freq=50)
    ctx.reset()
    ctx.data.qpos[ctx.arm_qadr] = HOME
    mujoco.mj_forward(ctx.model, ctx.data)
    ctx.hold_arm()
    ctx.set_finger_ctrl(0.04)
    for _ in range(80):
        ctx.step()
    parts = ("wipe_tool",) if isolated else (*PARTS, "wipe_tool")
    s = Session(
        ctx,
        out=out,
        parts=parts,
        grasp_specs={**GRASP, "wipe_tool": (0.047, 0.028)},
        capabilities={**DEFAULT_CAPABILITIES, "wipe_tool": ("wipe",)},
    )
    if recorder:
        s.rec = FullRecorder(
            s,
            out / ("wipe.mp4" if isolated else "assembly.mp4"),
            failure=failure,
            isolated=isolated,
        )
        ctx.on_control_step = s.rec
    return s


class FullRecorder(Recorder):
    def __init__(self, s, path, failure=False, isolated=False):
        # Keep the existing recorder/controller hooks; sampling every 80 ms is
        # presented at 25 fps for explicitly labelled 2x full-task playback.
        super().__init__(
            s.ctx, path, width=1920, height=1200, every=2 if isolated else 4
        )
        if not isolated:
            self.writer.close()
            import imageio.v2 as imageio

            self.writer = imageio.get_writer(
                str(path), fps=25, codec="libx264", quality=8, macro_block_size=2
            )
        self.s = s
        self.failure = failure
        self.isolated = isolated
        self.current_atom = None
        self.font = ImageFont.truetype(
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 34
        )
        self.small = ImageFont.truetype(
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 27
        )
        self.timeline = []
        self.outcome = None

    def set_skill(self, name, label):
        self.current_atom = name
        self.label = label
        self.timeline.append(
            dict(
                frame=self.frames,
                atom=name,
                label=label,
                sim_time=float(self.ctx.data.time),
            )
        )

    def frame(self):
        self.renderer.update_scene(
            self.ctx.data, camera="task_view", scene_option=self.option
        )
        self.renderer._scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        if self.current_atom == "wipe" and hasattr(self.s, "wipe_telemetry"):
            from .demo_visuals import Visuals, GREEN, ORANGE

            t = self.s.wipe_telemetry
            v = Visuals(self.s, "wipe")
            for i in range(0, len(t["grid"]), 8):
                xy = t["grid"][i]
                v.sphere(
                    self.renderer._scene,
                    [*xy, t["surface_z"] + 0.0008],
                    0.0014,
                    GREEN if t["covered"][i] else ORANGE,
                )
        im = Image.fromarray(self.renderer.render())
        draw = ImageDraw.Draw(im)
        draw.rectangle([0, 1090, 1920, 1200], fill=(23, 28, 34))
        draw.text((28, 1100), self.label, font=self.font, fill=(246, 248, 252))
        note = (
            "1×播放 · 轨迹模仿学习 + 接触力反馈"
            if self.isolated
            else "2×播放 · 固定正面镜头 · Panda 独立完成"
        )
        if self.current_atom == "wipe" and hasattr(self.s, "wipe_telemetry"):
            t = self.s.wipe_telemetry
            note += f"    表面力 {t['force']:.2f} N · 接触覆盖 {100*t['coverage']:.1f}%"
        draw.text((28, 1151), note, font=self.small, fill=(178, 218, 236))
        if self.failure:
            draw.rounded_rectangle([25, 23, 845, 85], radius=8, fill=(60, 31, 28))
            draw.text(
                (42, 33),
                "孔位偏差测试：右侧插销目标偏移 8 mm",
                font=self.small,
                fill=(255, 211, 157),
            )
        if self.outcome:
            draw.rounded_rectangle([25, 105, 1125, 173], radius=8, fill=(23, 28, 34))
            draw.text(
                (42, 118),
                self.outcome,
                font=self.small,
                fill=(130, 244, 171) if not self.failure else (255, 160, 140),
            )
        return np.asarray(im)


def prepare_wipe(s):
    pick(s, "wipe_tool")
    s.call("plan_path", method="surface", part="wipe_tool", center=CENTER.tolist())
    start = np.asarray(s.artifacts["wipe"]["points"][0])
    transfer_part(s, "wipe_tool", start + [0, 0, 0.075])
    s.call("move", reference="object", part="wipe_tool", target=start + [0, 0, 0.001])


def clean(s):
    prepare_wipe(s)
    s.call("wipe", part="wipe_tool")
    s.call("move", part="wipe_tool", delta=[0, 0, 0.1])
    transfer_part(s, "wipe_tool", TOOL_HOME + [0, 0, 0.06])
    s.call(
        "move", mode="guarded", part="wipe_tool", target_z=TOOL_HOME[2], force_stop=3.0
    )
    s.call("press", part="wipe_tool", target_z=TOOL_HOME[2], force_stop=2.0)
    s.call("place", part="wipe_tool", target=TOOL_HOME, tol=0.002)
    s.call("move", delta=[0, 0, 0.12])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/full_current")
    p.add_argument("--record", action="store_true")
    p.add_argument("--failure", action="store_true")
    p.add_argument("--wipe-only", action="store_true")
    a = p.parse_args()
    s = make_session(
        a.out,
        isolated=a.wipe_only,
        recorder=a.record and not a.wipe_only,
        failure=a.failure,
    )
    ok = False
    error = ""
    try:
        if a.wipe_only:
            prepare_wipe(s)
            if a.record:
                s.rec = FullRecorder(s, Path(a.out) / "wipe.mp4", isolated=True)
                s.ctx.on_control_step = s.rec
            s.call("wipe", part="wipe_tool")
        else:
            clean(s)
            assemble(s, None, pin_offset=0.008 if a.failure else 0.0)
        ok = True
    except (SkillFailure, ValueError) as exc:
        error = str(exc)
        print(error, flush=True)
    finally:
        report = dict(
            success=ok,
            expected_failure=a.failure,
            error=error,
            steps=len(s.results),
            atoms=sorted({r["atom"] for r in s.results if r["atom"]}),
            actuators=s.ctx.model.nu,
            sim_seconds=float(s.ctx.data.time),
            wiping=s.artifacts.get("wipe_result"),
            last_step=s.results[-1] if s.results else None,
        )
        if s.rec:
            s.rec.outcome = (
                "擦拭完成"
                if a.wipe_only and ok
                else (
                    "装配完成：位置、抓持释放与行程验收通过"
                    if ok
                    else "装配失败：接触或终态验收未通过，已停止任务"
                )
            )
            if not ok and s.results and "height_error_m" in s.results[-1]["metrics"]:
                mm = s.results[-1]["metrics"]["height_error_m"] * 1000
                s.rec.outcome = (
                    f"装配失败：插销未落座，高度误差 {mm:.1f} mm；任务已停止"
                )
            s.rec.pause(4.0)
            Image.fromarray(s.rec.frame()).save(Path(a.out) / "final.png")
            report.update(
                video_frames=s.rec.frames,
                video_seconds=s.rec.frames / 25,
                playback_rate=1 if a.wipe_only else 2,
            )
            (Path(a.out) / "timeline.json").write_text(
                json.dumps(s.rec.timeline, ensure_ascii=False, indent=2)
            )
            s.rec.close()
        (Path(a.out) / "verification.json").write_text(
            json.dumps(report, indent=2, default=lambda x: np.asarray(x).tolist())
        )
        print(json.dumps(report, default=lambda x: np.asarray(x).tolist()), flush=True)
    if not ok and not a.failure:
        raise SystemExit(1)
    if ok and a.failure:
        raise SystemExit("Expected physical failure did not occur")


if __name__ == "__main__":
    main()
