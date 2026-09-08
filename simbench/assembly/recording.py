"""One fixed MuJoCo camera with only a bottom current-skill caption."""

from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class Recorder:
    def __init__(self, ctx, path, width=1600, height=1000, every=2):
        self.ctx = ctx
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.renderer = mujoco.Renderer(ctx.model, height=height, width=width)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[3:] = 0
        self.font = ImageFont.truetype(
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 27
        )
        self.writer = imageio.get_writer(
            str(path),
            fps=1 / (ctx.control_dt * every),
            codec="libx264",
            quality=7,
            macro_block_size=2,
        )
        self.label = "初始化"
        self.n = 0
        self.frames = 0
        self.every = every

    def frame(self):
        self.renderer.update_scene(
            self.ctx.data, camera="task_view", scene_option=self.option
        )
        im = Image.fromarray(self.renderer.render())
        draw = ImageDraw.Draw(im)
        draw.rectangle([0, im.height - 55, im.width, im.height], fill=(30, 31, 33))
        draw.text(
            (24, im.height - 47), self.label, font=self.font, fill=(245, 245, 245)
        )
        return np.asarray(im)

    def set_skill(self, name, label):
        self.label = label

    def __call__(self, ctx):
        self.n += 1
        if self.n % self.every:
            return
        frame = self.frame()
        self.writer.append_data(frame)
        self.frames += 1

    def pause(self, seconds=0.7):
        # A planning/inspection skill has no physical motion. Show its computed
        # outcome for readability without advancing or changing the simulator.
        frame = self.frame()
        for _ in range(max(1, int(seconds / (self.ctx.control_dt * self.every)))):
            self.writer.append_data(frame)
            self.frames += 1

    def close(self):
        self.writer.close()
        if hasattr(self.renderer, "close"):
            self.renderer.close()
        else:
            self.renderer._gl_context.make_current()
            self.renderer._mjr_context.free()
            self.renderer._gl_context.free()
