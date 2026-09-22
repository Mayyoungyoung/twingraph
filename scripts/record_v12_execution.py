"""V12 recording labels expose the stopping stage and unrun functional test."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scripts.record_v11_execution import Recorder


class RecorderV12(Recorder):
    def frame(self, status=None):
        """Display cameras are separate from the wrist sensor input."""
        canvas = Image.new("RGB", (1280, 566), (22, 29, 38))
        self.renderer.update_scene(self.session.ctx.data, camera="task_view")
        canvas.paste(Image.fromarray(self.renderer.render().copy()), (0, 40))
        close = mujoco.MjvCamera()
        close.type = mujoco.mjtCamera.mjCAMERA_FREE
        close.lookat[:] = [.06, .085, .84]
        close.distance = .42
        close.azimuth = 135.
        close.elevation = -38.
        self.renderer.update_scene(self.session.ctx.data, camera=close)
        canvas.paste(Image.fromarray(self.renderer.render().copy()), (640, 40))
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 10), f"TwinGraph | {self.title} | simulation | {self.speed:g}x playback", font=self.font, fill="white")
        draw.text((652, 46), "Assembly detail (display only)", font=self.font, fill=(35, 48, 65))
        camera = "wrist_rgbd"
        if mujoco.mj_name2id(self.session.ctx.model, mujoco.mjtObj.mjOBJ_CAMERA, camera) >= 0:
            self.renderer.update_scene(self.session.ctx.data, camera=camera)
            view = Image.fromarray(self.renderer.render().copy()).resize((256, 192))
            canvas.paste(view, (8, 316))
            draw.rectangle((8, 290, 264, 316), fill=(22, 29, 38))
            draw.text((13, 292), "Wrist sensor view", font=self.font, fill="white")
        elapsed = float(self.session.ctx.data.time) - self.start
        draw.text((16, 534), f"t = {elapsed:.1f} s   |   {status or self.label}", font=self.font, fill=(161, 217, 234))
        return np.asarray(canvas)

    def close(self, result):
        if result["success"]:
            status="FUNCTIONAL TASK PASS | release, stroke and retained pins verified"
        else:
            reason=result.get("error", "unknown stop").split(":", 1)[0][:65]
            tested=bool(result.get("stage_passes", {}).get("functional_test_pass"))
            status=f"STOPPED: {reason} | functional test: {'passed earlier' if tested else 'not passed / possibly not reached'}"
        frame=self.frame(status)
        Image.fromarray(frame).save(self.path.with_suffix(".png"))
        for _ in range(60): self.writer.stdin.write(frame.tobytes())
        self.writer.stdin.close(); code=self.writer.wait(); self.encoder_log.close()
        close=getattr(self.renderer,"close",None) or getattr(self.renderer,"free",None)
        if close: close()
        if code: raise RuntimeError(f"video encoder failed: {code}")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--domain", choices=["online","deployment","development"], default="deployment")
    args=parser.parse_args()
    request=json.loads(args.request.read_text())
    proposal=next(p for p in request["pool"] if p["name"]==args.candidate)
    from simbench.value.system_v12 import rollout
    result=rollout(request["seed"],proposal,args.out,domain=args.domain,
        level=request.get("level","L1"),record=True)
    print(json.dumps(dict(success=result["success"],error=result["error"],out=str(args.out))))


if __name__=="__main__": main()
