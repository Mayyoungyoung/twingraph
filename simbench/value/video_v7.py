"""Record one independent v7 candidate rollout with two synchronized views."""
import argparse
import json
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
try:
    import cv2
except ImportError:  # pragma: no cover - imageio is the normal path
    cv2 = None
from PIL import Image, ImageDraw, ImageFont

from . import stage_v7
from .physical import PhysicalRunner, perturbation


class TwinRecorder:
    def __init__(self, session, path, score=None, width=640, height=480,
                 frame_stride=1):
        self.session = session; self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width=width; self.height=height
        self.frame_stride=max(1, int(frame_stride)); self._control_steps=0
        # Dropped frames use a lower playback rate, preserving the physical
        # trajectory duration instead of silently speeding up the clip.
        self.fps=20.0 / self.frame_stride
        self.renderer = mujoco.Renderer(session.ctx.model, height=height, width=width)
        self._cv_writer = None
        try:
            self.writer = imageio.get_writer(str(self.path), fps=self.fps, codec="libx264", quality=7, macro_block_size=2)
        except (ValueError, RuntimeError):
            if cv2 is None:
                raise
            # Windows environments may ship imageio without an ffmpeg plugin.
            # OpenCV writes the same real frames to an ordinary MP4.
            self.writer = None
            self._cv_writer = cv2.VideoWriter(
                str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps,
                (self.width * 2, self.height + 70),
            )
            if not self._cv_writer.isOpened():
                raise RuntimeError(f"cannot open video writer: {self.path}")
        self.label="初始化"; self.score=score; self.frames=0
        try: self.font=ImageFont.truetype("DejaVuSans.ttf", 20)
        except OSError: self.font=ImageFont.load_default()

    def set_skill(self, name, label): self.label=label

    def _view(self, camera):
        self.renderer.update_scene(self.session.ctx.data, camera=camera)
        return self.renderer.render().copy()

    def frame(self):
        left=self._view("task_view"); right=self._view("top_view")
        canvas=Image.new("RGB", (self.width*2, self.height+70), (28,31,35))
        canvas.paste(Image.fromarray(left), (0,0)); canvas.paste(Image.fromarray(right), (self.width,0))
        draw=ImageDraw.Draw(canvas)
        dirty=self.session.dirty_state or {}
        initial=float(dirty.get("initial_dirty_amount",0.0)); remain=float(dirty.get("remaining_dirty_amount",0.0))
        clean=1-remain/max(initial,1e-12) if initial else 0.0
        peak=max(self.session.stroke_peak_forces, default=0.0)
        draw.rectangle([0,self.height,self.width*2,self.height+70], fill=(28,31,35))
        text=f"{self.label} | task_view / top_view | score={self.score if self.score is not None else 'NA'}"
        draw.text((14,self.height+8), text, font=self.font, fill=(245,245,245))
        draw.text((14,self.height+38), f"cleaning_ratio={clean:.3f} residual={remain:.3f} | stroke_peak_N={peak:.2f}", font=self.font, fill=(180,225,238))
        return np.asarray(canvas)

    def _append(self, frame):
        if self.writer is not None:
            self.writer.append_data(frame)
        else:
            self._cv_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def __call__(self, ctx):
        self._control_steps += 1
        if (self._control_steps - 1) % self.frame_stride:
            return
        self._append(self.frame()); self.frames+=1

    def pause(self, seconds=.5):
        frame=self.frame()
        for _ in range(max(1,int(seconds*20))): self._append(frame); self.frames+=1

    def close(self):
        if self.writer is not None: self.writer.close()
        if self._cv_writer is not None: self._cv_writer.release()
        close=getattr(self.renderer,"close",None) or getattr(self.renderer,"free",None)
        if close: close()


def record_one(seed, out, candidate_index=0, level="L0", score=None, timeout=360.):
    out=Path(out); scene_dir=out/"scene"; spec,s,xml,targets=stage_v7.make_scene(seed, scene_dir, role="development", level=level)
    plans,_=stage_v7.build_pool(s,targets,seed,n=max(1,candidate_index+1)); plan=plans[candidate_index]
    rec=TwinRecorder(s, out/f"candidate_{candidate_index+1:02d}.mp4", score=score); s.rec=rec; s.ctx.on_control_step=rec
    runner=PhysicalRunner(s,timeout=timeout); result=runner.run(plan, perturbation(seed,0,"development"), keep_trace=True); rec.pause(1.5); rec.close()
    row=dict(seed=seed, candidate_id=plan.id, candidate_index=candidate_index, model_score=score,
             video=str((out/f"candidate_{candidate_index+1:02d}.mp4").resolve()), result=result,
             source="real MuJoCo state trajectory")
    (out/f"candidate_{candidate_index+1:02d}.json").write_text(json.dumps(row,indent=2,ensure_ascii=False,default=lambda x: np.asarray(x).tolist()))
    return row


def main():
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);p.add_argument("--seed",type=int,required=True);p.add_argument("--candidate",type=int,default=0);p.add_argument("--level",choices=("L0","L1","L2"),default="L0");p.add_argument("--score",type=float);a=p.parse_args()
    print(json.dumps(record_one(a.seed,a.out,a.candidate,a.level,a.score),ensure_ascii=False,default=lambda x: np.asarray(x).tolist()))


if __name__=="__main__": main()
