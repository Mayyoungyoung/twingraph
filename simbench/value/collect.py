"""Collect complete physical suffix outcomes, one immutable candidate group at a time."""
import argparse
import hashlib
import concurrent.futures
import contextlib
from dataclasses import asdict
import json
import multiprocessing
import os
from pathlib import Path
import platform
import time
import traceback
import numpy as np
import mujoco
from simbench.assembly.library import SkillFailure
from simbench.assembly.candidates import fingerprint
from .plan import PROTOCOL, digest, execute_prefix, execute_suffix, plain
from .scenarios import TaskSpec, make_task, build_plans, observation, render_observation


def dump(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(plain(data), indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def source_hash():
    root = Path(__file__).resolve().parents[1]
    hasher = hashlib.sha256()
    for folder in ("value", "assembly", "core"):
        for file in sorted((root / folder).glob("*.py")):
            hasher.update(str(file.relative_to(root)).replace("\\", "/").encode())
            hasher.update(file.read_bytes())
    return hasher.hexdigest()


def collect_group(seed, out, repeats=4, render=True):
    spec = TaskSpec.sample(seed)
    group = Path(out) / f"group_{seed:06d}"
    group.mkdir(parents=True, exist_ok=True)
    request = dict(spec=asdict(spec), repeats=repeats, render=render, protocol=PROTOCOL)
    if (group / "complete.json").exists():
        completed = json.loads((group / "complete.json").read_text())
        if completed.get("request_sha256") != digest(request):
            raise ValueError("resume settings disagree with completed group")
        return completed
    started = time.perf_counter()
    with (group / "steps.log").open(
        "w", encoding="utf-8"
    ) as log, contextlib.redirect_stdout(log):
        session, scene_path = make_task(spec, group)
        plans = build_plans(session, spec)
        if not plans:
            raise RuntimeError("no candidates generated")
        observed = observation(session, spec)
        images = render_observation(session, group) if render else []
        snapshot = session.snapshot()
        inputs = dict(
            schema="twingraph.group.v1",
            group_id=spec.config_id,
            split_group=spec.config_id,
            protocol=PROTOCOL,
            task=asdict(spec),
            observation=observed,
            images=images,
            candidates=[p.to_dict() for p in plans],
            snapshot_sha256=fingerprint(session),
            source_sha256=source_hash(),
        )
        # Save inputs before the FIRST rollout, and verify no later mutation.
        input_hash = digest(inputs)
        dump(group / "inputs.json", inputs)
        # npz contains only numeric arrays, no executable pickle.
        np.savez_compressed(
            group / "initial_physics.npz", **snapshot["physics"]["data"]
        )
        outcomes = []
        try:
            for repeat in range(repeats):
                rng = np.random.default_rng(np.random.SeedSequence([seed, repeat, 731]))
                trial = dict(
                    repeat=repeat,
                    friction_scale=float(rng.uniform(0.85, 1.15)),
                    actuator_gain_scale=float(rng.uniform(0.985, 1.015)),
                )
                base_gain = session.ctx.model.actuator_gainprm.copy()
                base_bias = session.ctx.model.actuator_biasprm.copy()
                for plan in plans:
                    session.restore(snapshot)
                    session.results.clear()
                    assert fingerprint(session) == inputs["snapshot_sha256"]
                    # Deterministic pairing: all candidates see the same model
                    # perturbation. Candidates remain bound to nominal snapshot;
                    # perturb after stale-plan check, using a trial-specific copy.
                    import copy

                    trial_plan = copy.deepcopy(plan)
                    session.ctx.model.geom_friction[:] *= trial["friction_scale"]
                    session.ctx.model.actuator_gainprm[:] = (
                        base_gain * trial["actuator_gain_scale"]
                    )
                    session.ctx.model.actuator_biasprm[:] = (
                        base_bias * trial["actuator_gain_scale"]
                    )
                    trial_plan.prefix["start_state"] = fingerprint(session)
                    start_time = session.ctx.data.time
                    wall = time.perf_counter()
                    prefix, suffix, failed_stage, error, valid = (
                        False,
                        None,
                        "prefix",
                        "",
                        True,
                    )
                    try:
                        execute_prefix(session, trial_plan)
                        prefix = True
                        failed_stage = "suffix"
                        execute_suffix(session, trial_plan)
                        session.hold(0.25)
                        session.call(
                            "inspect", part="pin_left", target=spec.target, tol=0.0015
                        )
                        suffix = True
                        failed_stage = None
                    except SkillFailure as exc:
                        error = str(exc)
                        if prefix:
                            suffix = False
                    except ValueError as exc:
                        if str(exc).startswith("IK unreachable:"):
                            error = str(exc)
                            if prefix:
                                suffix = False
                        else:
                            error = traceback.format_exc()
                            valid = False
                    except Exception:
                        # Programming, serialization and infrastructure errors
                        # are NEVER converted into negative training labels.
                        error = traceback.format_exc()
                        valid = False
                    finally:
                        session.ctx.model.actuator_gainprm[:] = base_gain
                        session.ctx.model.actuator_biasprm[:] = base_bias
                    outcomes.append(
                        dict(
                            candidate_id=plan.id,
                            trial=trial,
                            valid=valid,
                            prefix_success=prefix if valid else None,
                            suffix_success=suffix if valid else None,
                            full_success=bool(prefix and suffix) if valid else None,
                            failure_stage=failed_stage,
                            error=error,
                            sim_seconds=float(session.ctx.data.time - start_time),
                            wall_seconds=time.perf_counter() - wall,
                            recoveries=0,
                            final_position=session.ctx.obj_pos("pin_left").tolist(),
                            held=session.held,
                            executed_steps=len(session.results),
                        )
                    )
                    dump(
                        group / "outcomes.json",
                        dict(input_sha256=input_hash, trials=outcomes),
                    )
        finally:
            session.restore(snapshot)
        if digest(inputs) != input_hash:
            raise RuntimeError("rollout mutated ranker inputs")
        invalid = [r for r in outcomes if not r["valid"]]
        if invalid:
            raise RuntimeError(
                f"{len(invalid)} invalid trials; first: {invalid[0]['error']}"
            )
        summary = dict(
            group_id=spec.config_id,
            seed=seed,
            candidates=len(plans),
            trials=len(outcomes),
            prefix_successes=sum(r["prefix_success"] for r in outcomes),
            full_successes=sum(r["full_success"] for r in outcomes),
            wall_seconds=time.perf_counter() - started,
            input_sha256=input_hash,
            request_sha256=digest(request),
        )
        dump(group / "complete.json", summary)
        return summary


def collect_worker(*args):
    try:
        return collect_group(*args)
    except Exception:
        # EGL exceptions contain ctypes pointers and cannot cross a process
        # queue. Preserve the real diagnostic as plain text instead.
        raise RuntimeError(traceback.format_exc()) from None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/value/data")
    p.add_argument("--groups", type=int, default=80)
    p.add_argument("--start-seed", type=int, default=1000)
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--no-render", action="store_true")
    a = p.parse_args()
    if min(a.groups, a.repeats, a.workers) < 1:
        p.error("groups, repeats and workers must be positive")
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        schema="twingraph.dataset.v1",
        protocol=PROTOCOL,
        request=vars(a),
        mujoco=mujoco.__version__,
        numpy=np.__version__,
        python=platform.python_version(),
        seed_range=[a.start_seed, a.start_seed + a.groups],
        label_source="physical_mujoco_rollouts",
        scope="parameterized sliding-stage pin subtask; full release/retreat suffix",
    )
    dump(out / "collection_request.json", manifest)
    summaries, errors = [], []
    # Forking an initialized EGL loader is unsafe; initialize each worker.
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=a.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = {
            pool.submit(
                collect_worker, seed, str(out), a.repeats, not a.no_render
            ): seed
            for seed in range(a.start_seed, a.start_seed + a.groups)
        }
        for f in concurrent.futures.as_completed(futures):
            try:
                summary = f.result()
                summaries.append(summary)
                print(json.dumps(summary), flush=True)
            except Exception as exc:
                error = dict(seed=futures[f], error=str(exc))
                errors.append(error)
                print(json.dumps(error), flush=True)
            dump(
                out / "manifest.json",
                {
                    **manifest,
                    "completed": sorted(summaries, key=lambda x: x["seed"]),
                    "errors": errors,
                },
            )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
