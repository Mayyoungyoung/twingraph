"""Record a new physical replay of a selected plan; never part of timed tests."""
import argparse
import contextlib
import json
from pathlib import Path

import subprocess
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from simbench.value import stage_v9
from simbench.value.physical import PhysicalRunner,perturbation
from simbench.value.system_v11 import ClosedLoop,FrozenValue,run_staged,save
from simbench.value.v9_candidates import bind


class Recorder:
    def __init__(self,session,path,title):
        self.session=session;self.path=Path(path);self.title=title;self.label="initialization"
        self.renderer=mujoco.Renderer(session.ctx.model,height=480,width=640)
        self.encoder_log=self.path.with_suffix(".encoder.log").open("w")
        self.writer=subprocess.Popen(["ffmpeg","-y","-loglevel","error","-f","rawvideo",
            "-pix_fmt","rgb24","-s","1280x566","-r","20","-i","-","-an",
            "-c:v","libx264","-preset","veryfast","-crf","22","-threads","1",
            "-pix_fmt","yuv420p","-movflags","+faststart",str(path)],stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,stderr=self.encoder_log)
        self.stride=10;self.calls=0;self.frames=0;self.start=float(session.ctx.data.time)
        self.speed=20*self.stride*session.ctx.control_dt
        try:self.font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",19)
        except OSError:self.font=ImageFont.load_default()

    def set_skill(self,name,label):
        part=label.rsplit("·",1)[-1].strip() if "·" in label else ""
        self.label=f"{name or 'verification'} {part}"

    def pause(self,seconds): pass

    def frame(self,status=None):
        canvas=Image.new("RGB",(1280,566),(22,29,38))
        for i,camera in enumerate(("task_view","top_view")):
            self.renderer.update_scene(self.session.ctx.data,camera=camera)
            canvas.paste(Image.fromarray(self.renderer.render().copy()),(i*640,40))
        draw=ImageDraw.Draw(canvas)
        draw.text((16,10),f"TwinGraph | {self.title} | independent simulation | {self.speed:g}x playback",font=self.font,fill="white")
        elapsed=float(self.session.ctx.data.time)-self.start
        draw.text((16,534),f"t = {elapsed:.1f} s   |   {status or self.label}",font=self.font,fill=(161,217,234))
        return np.asarray(canvas)

    def __call__(self,ctx):
        self.calls+=1
        if self.calls%self.stride==0:
            self.writer.stdin.write(self.frame().tobytes());self.frames+=1

    def close(self,result):
        frame=self.frame("FUNCTIONAL TASK PASS" if result["success"] else "TASK FAILED - see trace")
        Image.fromarray(frame).save(self.path.with_suffix(".png"))
        for _ in range(40):self.writer.stdin.write(frame.tobytes())
        self.writer.stdin.close();code=self.writer.wait();self.encoder_log.close()
        close=getattr(self.renderer,"close",None) or getattr(self.renderer,"free",None)
        if close: close()
        if code:raise RuntimeError(f"video encoder failed: {code}")


def main():
    p=argparse.ArgumentParser();p.add_argument("--method-root",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--candidate");p.add_argument("--domain",choices=["online","deployment"],default="deployment")
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    request=json.loads((a.method_root / "request.json").read_text())
    chosen=a.candidate or json.loads((a.method_root / "summary.json").read_text())["selected"]
    proposal=next(r for r in request["pool"] if r["name"]==chosen)
    expected=json.loads((a.method_root / "twins" / chosen / "result.json").read_text())["boundaries"]
    value=FrozenValue(a.checkpoint);boundaries=[]
    monitor=ClosedLoop(request["seed"],expected,value,a.out / "closed_loop") if a.domain=="deployment" else None
    with (a.out / "console.log").open("w",encoding="utf-8") as log,contextlib.redirect_stdout(log):
        _,session,_,_=stage_v9.make_scene(request["seed"],a.out / "scene",role=a.domain)
        session.full_task_controller=lambda s,**params:run_staged(s,**params,monitor=monitor,boundaries=boundaries)
        rec=Recorder(session,a.out / "execution.mp4",f'seed {request["seed"]} / {chosen}')
        session.rec=rec;session.ctx.on_control_step=rec
        result=PhysicalRunner(session,timeout=1800.).run(bind(session,session.stage_targets,proposal),
            perturbation(request["seed"],0,a.domain),keep_trace=True)
        save(a.out / "result.json",dict(result=result,seed=request["seed"],proposal=proposal,
            boundaries=boundaries,playback_speed=rec.speed,physical_frames=rec.frames,
            note="New physical replay for visualization; not counted in timed test",domain=a.domain))
        rec.close(result)
    print(json.dumps(dict(success=result["success"],error=result["error"],video=str(a.out / "execution.mp4"))))


if __name__=="__main__":main()
