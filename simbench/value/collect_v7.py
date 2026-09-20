"""Collect complete v7 task outcomes with a frozen simulated observation."""
import argparse
import contextlib
import json
import multiprocessing
import concurrent.futures
import time
import traceback
import hashlib
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .collect import dump
from .plan import digest
from .physical import PhysicalRunner, perturbation
from .skill_graph import compile_graph


def trial_spec(seed, repeat, domain, friction_span=.08, gain_span=.015):
    namespaces = {"train": 881, "val": 883, "test": 887,
                  "twin": 889, "target": 893, "development": 877}
    if domain not in namespaces or repeat < 0:
        raise ValueError("unknown v7 disturbance domain")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(repeat), namespaces[domain]]))
    nominal = repeat == 0
    return dict(
        domain=domain, repeat=int(repeat),
        friction_scale=1. if nominal else float(rng.uniform(1-friction_span, 1+friction_span)),
        actuator_gain_scale=1. if nominal else float(rng.uniform(1-gain_span, 1+gain_span)),
        # The v7 runner changes friction and actuator gain. Mass, damping and
        # perception draws were previously advertised but never applied.
    )


def _source_manifest():
    root = Path(__file__).parents[2]
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for d in ("simbench/value", "simbench/assembly", "simbench/core")
            for p in sorted((root / d).glob("*.py"))}


def collect_one(seed, out, split="development", level="L1", n=12, repeats=1, timeout=360.):
    from . import stage_v7 as stage
    directory = Path(out) / f"group_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    source = _source_manifest()
    request = dict(seed=int(seed), split=split, level=level, n=int(n), repeats=int(repeats),
                   timeout_seconds=float(timeout), source_sha256=digest(source),
                   sensor_backend="simulated_pose_sensor_proxy", task_scope=stage.TASK_SCOPE)
    if (directory / "complete.json").exists():
        complete = json.loads((directory / "complete.json").read_text())
        if complete["request_sha256"] != digest(request):
            raise ValueError("resume request/source mismatch")
        return complete
    existing = [p for p in directory.iterdir() if p.name not in {"request.json"}]
    if existing:
        raise FileExistsError("preserve incomplete v7 group; choose a new output directory")
    dump(directory / "request.json", request); dump(directory / "source.json", source)
    started = time.perf_counter()
    with (directory / "steps.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        spec, session, xml, targets = stage.make_scene(seed, directory, role="development" if split == "development" else split, level=level)
        vision = stage.save_vision(directory / "vision.npz", stage.capture_vision(session))
        t = time.perf_counter(); plans, counts = stage.build_pool(session, targets, seed, n=n); candidate_seconds = time.perf_counter() - t
        observation = stage.observed(session, targets)
        runner = PhysicalRunner(session, timeout=timeout)
        inputs = dict(schema="twingraph.group.v7", group_id=spec.config_id, split_group=spec.config_id,
                      declared_split=split, level=level, task=asdict(spec), observation=observation,
                      sensor_manifest=session.sensor_manifest, dirty_initial=session.dirty_state,
                      vision=vision, candidates=[p.to_dict() for p in plans], pool_counts=counts,
                      source_sha256=digest(source), snapshot_sha256=runner.initial,
                      candidate_generation_seconds=candidate_seconds)
        ih = digest(inputs); graphs = [compile_graph(observation, p) for p in plans]
        dump(directory / "inputs.json", inputs); dump(directory / "skill_graphs.json", dict(input_sha256=ih, graphs=graphs))
        trials = []
        domain = "development" if split == "development" else split
        for repeat in range(repeats):
            trial = trial_spec(seed, repeat, domain)
            for plan, graph in zip(plans, graphs):
                row = runner.run(graph, trial, keep_trace=False)
                trials.append(row); dump(directory / "outcomes.json", dict(input_sha256=ih, trials=trials))
        complete = dict(request_sha256=digest(request), input_sha256=ih, source_sha256=digest(source),
                        seed=int(seed), split=split, level=level, candidates=len(plans), trials=len(trials),
                        repeats=repeats, successes=sum(bool(t["success"]) for t in trials),
                        cleaning_passes=sum(bool(t.get("stage_passes", {}).get("cleaning_pass")) for t in trials),
                        assembly_passes=sum(bool(t.get("stage_passes", {}).get("assembly_pass")) for t in trials),
                        functional_passes=sum(bool(t.get("stage_passes", {}).get("functional_test_pass")) for t in trials),
                        wall_seconds=time.perf_counter()-started,
                        physical_wall_seconds=sum(t["wall_seconds"]+t["restore_seconds"] for t in trials),
                        physics_steps=sum(t["physics_steps"] for t in trials))
        if _source_manifest() != source:
            raise RuntimeError("source changed during v7 collection")
        dump(directory / "complete.json", complete)
        return complete


def worker(request):
    try:
        return collect_one(**request)
    except Exception as exc:
        directory = Path(request["out"]) / f"group_{request['seed']}"; directory.mkdir(parents=True, exist_ok=True)
        row = dict(request=request, error=traceback.format_exc(), exception_type=type(exc).__name__, message=str(exc))
        dump(directory / "failure.json", row); return row


def main():
    p = argparse.ArgumentParser(); p.add_argument("--out", required=True); p.add_argument("--seed", type=int, required=True)
    p.add_argument("--groups", type=int, default=1); p.add_argument("--n", type=int, default=12); p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--workers", type=int, default=1); p.add_argument("--split", choices=("train", "val", "test", "development"), default="development")
    p.add_argument("--level", choices=("L0", "L1", "L2"), default="L1"); p.add_argument("--timeout", type=float, default=360.)
    a = p.parse_args(); requests = [dict(seed=a.seed+i, out=a.out, split=a.split, level=a.level, n=a.n, repeats=a.repeats, timeout=a.timeout) for i in range(a.groups)]
    results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for future in concurrent.futures.as_completed([pool.submit(worker, r) for r in requests]):
            row=future.result(); results.append(row); print(json.dumps(row), flush=True)
    dump(Path(a.out) / f"batch_{a.split}_{a.seed}.json", results)
    if any("error" in r for r in results): raise SystemExit(1)


if __name__ == "__main__":
    main()
