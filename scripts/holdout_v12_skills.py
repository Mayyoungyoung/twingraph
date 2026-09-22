"""Predeclared independent skill trials against a frozen V12 runtime.

This harness adds recording only; it does not alter controllers, the scene,
acceptance thresholds or policy weights. Every scheduled outcome is retained.
"""
import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(seed, skill, output, record):
    from scripts import check_v12_skills as check
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    recorder = None
    if record:
        from scripts.record_v11_execution import Recorder
        from PIL import Image

        class IsolatedRecorder(Recorder):
            def close(self, result):
                reason = (result.get("error") or "").split(":", 1)[0][:65]
                status = ("ISOLATED SKILL PASS; not a complete assembly trial" if result["success"]
                          else "ISOLATED SKILL STOP: " + reason)
                frame = self.frame(status)
                Image.fromarray(frame).save(self.path.with_suffix(".png"))
                for _ in range(50): self.writer.stdin.write(frame.tobytes())
                self.writer.stdin.close(); code = self.writer.wait(); self.encoder_log.close()
                close = getattr(self.renderer, "close", None) or getattr(self.renderer, "free", None)
                if close: close()
                if code: raise RuntimeError(f"video encoder failed: {code}")

        original = check.make_scene

        def recorded_scene(*args, **kwargs):
            nonlocal recorder
            result = original(*args, **kwargs)
            session = result[1]
            recorder = IsolatedRecorder(session, output / "execution.mp4",
                                        f"white printed kit / isolated {skill} / seed {seed}")
            session.rec = recorder
            session.ctx.on_control_step = recorder
            return result

        check.make_scene = recorded_scene
    report = None
    try:
        report = check.run(seed, skill, output, fixture_mode="free_visual")
    except Exception as exc:
        report = dict(seed=seed, skill=skill, success=False, error=repr(exc),
                      complete_task_trial=False, setup="harness_or_scene_error", steps=[])
        (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    finally:
        if recorder is not None and report is not None: recorder.close(report)
    return report


def batch(output):
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists():
        raise FileExistsError("Refusing to overwrite an existing holdout protocol or outcomes")
    runtime = Path.cwd().resolve()
    source = {str(p.relative_to(runtime)): digest(p) for p in sorted((runtime / "simbench").rglob("*.py"))}
    source.update({str(p.relative_to(runtime)): digest(p) for p in
                   sorted((runtime / "simbench/assembly/checkpoints").glob("*v12.npz"))})
    trials = [dict(seed=s, skill="pin", record=s == 1680) for s in range(1680, 1690)]
    trials += [dict(seed=s, skill="wipe", record=s == 1690) for s in range(1690, 1695)]
    protocol = dict(runtime=str(runtime), created_unix=time.time(), source_sha256=source,
                    trials=trials, fixed_before_outcomes=True,
                    no_controller_updates_from_outcomes=True, complete_task_trial=False,
                    scope="Unseen simulated white-part layout seeds; fixed skill choices and geometry. Not RealSense hardware, not unseen materials or full assembly.")
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    def lane(index):
        for row in trials[index::3]:
            directory = output / f"{row['skill']}{row['seed']}"
            directory.mkdir(parents=True)
            command = ["taskset", "-c", str(6+index), sys.executable, str(Path(__file__).resolve()),
                       "--worker", "--seed", str(row["seed"]), "--skill", row["skill"], "--out", str(directory)]
            if row["record"]: command += ["--record"]
            env = dict(os.environ, PYTHONPATH=str(runtime), MUJOCO_GL="egl",
                       OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
            with (directory / "execution.jsonl").open("w") as log:
                process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            if not (directory / "summary.json").exists():
                (directory / "summary.json").write_text(json.dumps(dict(**row, success=False,
                    error=f"worker exit {process.returncode}; see execution.jsonl")), encoding="utf-8")
            report = json.loads((directory / "summary.json").read_text())
            print(json.dumps(dict(seed=row["seed"], skill=row["skill"], success=report["success"],
                                  error=(report.get("error") or "")[:100])), flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lane, range(3)))
    rows = []
    for trial in trials:
        path = output / f"{trial['skill']}{trial['seed']}" / "summary.json"
        row = json.loads(path.read_text())
        rows.append(dict(**trial, success=row["success"], error=row.get("error"),
                         result_sha256=digest(path), summary=str(path.relative_to(output))))
    result = dict(trials=rows, complete_task_trial=False, protocol_sha256=digest(output / "protocol.json"),
                  by_skill={skill: dict(total=sum(x["skill"] == skill for x in rows),
                                        success=sum(x["skill"] == skill and x["success"] for x in rows))
                            for skill in ("pin", "wipe")})
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["by_skill"]), flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--out", required=True)
    parser.add_argument("--worker", action="store_true"); parser.add_argument("--record", action="store_true")
    parser.add_argument("--seed", type=int); parser.add_argument("--skill", choices=("pin", "wipe"))
    args=parser.parse_args()
    if args.worker: worker(args.seed,args.skill,args.out,args.record)
    else: batch(args.out)
