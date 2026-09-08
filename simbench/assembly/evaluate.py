"""Repeated complete assembly trials with observation noise and real kit scatter."""

import argparse
import contextlib
import json
from pathlib import Path
import time
import numpy as np
import mujoco
from .control import make_context, DEFAULT_POLICY
from .library import Session, SkillFailure, PARTS
from .task import assemble


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4,5")
    ap.add_argument("--noise", type=float, default=0.00025)
    ap.add_argument("--scatter", type=float, default=0.0005)
    ap.add_argument("--policy", default=str(DEFAULT_POLICY))
    ap.add_argument("--out", default="results/tabletop/evaluation")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for seed in map(int, a.seeds.split(",")):
        ctx = make_context(seed)
        rng = np.random.default_rng(seed + 17001)
        # Episode initialization only: scatter supplied parts, then let gravity
        # settle them before any task execution or demonstration begins.
        for part in PARTS:
            bid = ctx.body_id(part)
            jid = int(ctx.model.body_jntadr[bid])
            qa = int(ctx.model.jnt_qposadr[jid])
            ctx.data.qpos[qa : qa + 2] += rng.uniform(-a.scatter, a.scatter, 2)
            angle = rng.uniform(-0.004, 0.004)
            ctx.data.qpos[qa + 3 : qa + 7] = [
                np.cos(angle / 2),
                0,
                0,
                np.sin(angle / 2),
            ]
        mujoco.mj_forward(ctx.model, ctx.data)
        for _ in range(100):
            ctx.step()
        s = Session(ctx, out=out / f"seed_{seed}", seed=seed, noise=a.noise)
        started = time.perf_counter()
        ok = False
        error = ""
        with (out / f"seed_{seed}.log").open("w") as f, contextlib.redirect_stdout(f):
            try:
                assemble(s, a.policy)
                ok = True
            except (SkillFailure, ValueError) as exc:
                error = str(exc)
        metrics = {
            r["params"].get("part"): r["metrics"]
            for r in s.results
            if r["skill"] == "inspect_seat"
        }
        row = dict(
            seed=seed,
            success=ok,
            error=error,
            steps=len(s.results),
            noise_std_m=a.noise,
            kit_scatter_m=a.scatter,
            wall_seconds=time.perf_counter() - started,
            final_inspections=metrics,
        )
        (out / f"seed_{seed}.json").write_text(json.dumps(row, indent=2))
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
