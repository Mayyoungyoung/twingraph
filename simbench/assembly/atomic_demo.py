"""A supported Place demonstration using the public atom interface and fixed camera."""

import argparse
import json
from pathlib import Path
from PIL import Image
from .demo_scenes import make_demo_session
from .recording import Recorder
from .task import pick


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/atomic_place")
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    s = make_demo_session("cube", out / "scene")
    pick(s, "cube", lift=False)
    target = s.ctx.obj_pos("cube").copy()
    rec = Recorder(s.ctx, out / "place.mp4") if args.record else None
    s.rec = rec
    s.ctx.on_control_step = rec
    try:
        result = s.call("place", part="cube", target=target)
        report = dict(
            atom="place",
            success=result.ok,
            metrics=result.metrics,
            held_after=s.held,
            object_xyz=s.ctx.obj_pos("cube").tolist(),
        )
        if rec:
            rec.label = "放置 | place · 已释放；支撑与位置检查通过"
            rec.pause(1.0)
            Image.fromarray(rec.frame()).save(out / "place.png")
            report["video_seconds"] = rec.frames * s.ctx.control_dt * rec.every
        (out / "verification.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report))
    finally:
        s.rec = None
        s.ctx.on_control_step = None
        if rec:
            rec.close()


if __name__ == "__main__":
    main()
