"""Collect robust full-program outcomes and pre-execution RGB-D observations."""
import argparse
import concurrent.futures
import contextlib
from dataclasses import asdict
import json
import multiprocessing
from pathlib import Path
import time
import traceback

import numpy as np

from .collect import dump
from .physical_v6 import RobustPhysicalRunner
from .plan import digest
from .skill_graph import compile_graph


def trial_spec(seed, repeat, domain, friction_span=.08, gain_span=.015,
               mass_span=.04, damping_span=.06):
    namespaces = {"train": 811, "reference": 813, "development": 810,
                  "twin": 817, "target": 819}
    if domain not in namespaces or repeat < 0:
        raise ValueError("unknown v6 disturbance domain")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(repeat), namespaces[domain]]))
    nominal = repeat == 0
    return dict(
        domain=domain, repeat=int(repeat),
        friction_scale=1. if nominal else float(rng.uniform(1-friction_span, 1+friction_span)),
        actuator_gain_scale=1. if nominal else float(rng.uniform(1-gain_span, 1+gain_span)),
        mass_scale=1. if nominal else float(rng.uniform(1-mass_span, 1+mass_span)),
        damping_scale=1. if nominal else float(rng.uniform(1-damping_span, 1+damping_span)),
        # Even the nominal trial has realistic detector scatter; repeat draws
        # widen the envelope instead of granting perfect simulator perception.
        position_noise_std_m=.00035 if nominal else float(rng.uniform(.00025, .00125)),
        perception_seed=int(rng.integers(0, 2**31-1)),
    )


def collect_one(seed, out, split="train", domain="train", n=12, repeats=3, timeout=240.):
    from . import stage_v6 as stage
    from .value_v6 import source_manifest
    directory = Path(out) / f"group_{seed}"
    existed = directory.exists() and any(directory.iterdir())
    directory.mkdir(parents=True, exist_ok=True)
    source = source_manifest()
    request = dict(seed=seed, split=split, domain=domain, n=n, repeats=repeats,
        timeout_seconds=timeout, source_sha256=digest(source), friction_span=.08,
        gain_span=.015, mass_span=.04, damping_span=.06,
        position_noise_std_m=[.00025, .00125])
    if (directory / "complete.json").exists():
        complete = json.loads((directory / "complete.json").read_text())
        if complete["request_sha256"] != digest(request):
            raise ValueError("resume request/source mismatch")
        return complete
    if existed:
        raise FileExistsError("preserve incomplete attempt; choose a new output directory")
    dump(directory / "request.json", request)
    dump(directory / "source.json", source)
    started = time.perf_counter()
    with (directory / "steps.log").open("w") as log, contextlib.redirect_stdout(log):
        spec, session, xml, targets = stage.make_scene(seed, directory, role="collection")
        scene_seconds = time.perf_counter() - started
        vision_arrays = stage.capture_vision(session)
        vision = stage.save_vision(directory / "vision.npz", vision_arrays)
        begin = time.perf_counter()
        plans, counts = stage.build_pool(session, targets, seed, n=n)
        candidate_seconds = time.perf_counter() - begin
        observation = stage.observed(session, targets)
        runner = RobustPhysicalRunner(session, timeout=timeout)
        inputs = dict(schema="twingraph.group.v6", group_id=spec.config_id,
            split_group=spec.config_id, declared_split=split, task=asdict(spec),
            observation=observation, vision=vision,
            candidates=[p.to_dict() for p in plans], pool_counts=counts,
            source_sha256=digest(source), snapshot_sha256=runner.initial,
            scene_seconds=scene_seconds, candidate_generation_seconds=candidate_seconds)
        ih = digest(inputs)
        graphs = [compile_graph(observation, p) for p in plans]
        dump(directory / "inputs.json", inputs)
        dump(directory / "skill_graphs.json", dict(input_sha256=ih, graphs=graphs))
        np.savez_compressed(directory / "initial_physics.npz", **runner.snapshot["physics"]["data"])
        trials = []
        for repeat in range(repeats):
            perturbation = trial_spec(seed, repeat, domain)
            for graph in graphs:
                trials.append(runner.run(graph, perturbation, keep_trace=False))
                dump(directory / "outcomes.json", dict(input_sha256=ih, trials=trials))
        complete = dict(request_sha256=digest(request), input_sha256=ih,
            source_sha256=digest(source), seed=seed, split=split,
            candidates=len(plans), trials=len(trials), repeats=repeats,
            successes=sum(t["success"] for t in trials),
            robust_positive_candidates=sum(
                sum(t["success"] for t in trials if t["candidate_id"] == p.id) / repeats >= .5
                for p in plans),
            timeouts=sum(t["timeout"] for t in trials),
            wall_seconds=time.perf_counter()-started,
            physical_wall_seconds=sum(t["wall_seconds"]+t["restore_seconds"] for t in trials),
            physics_steps=sum(t["physics_steps"] for t in trials))
        if source_manifest() != source:
            raise RuntimeError("source changed during collection")
        if complete["timeouts"]:
            dump(directory / "censored.json", complete)
            raise RuntimeError("censored group retained")
        dump(directory / "complete.json", complete)
        return complete


def worker(request):
    directory = Path(request["out"]) / f"group_{request['seed']}"
    existed = directory.exists() and any(directory.iterdir())
    try:
        return collect_one(**request)
    except Exception as exc:
        row = dict(request=request, error=traceback.format_exc(),
                   exception_type=type(exc).__name__, message=str(exc))
        directory.mkdir(parents=True, exist_ok=True)
        row["phase"] = "after_inputs" if (directory / "inputs.json").exists() else "before_inputs"
        if not existed and not (directory / "complete.json").exists():
            dump(directory / "failure.json", row)
        return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True); p.add_argument("--seed", type=int, required=True)
    p.add_argument("--groups", type=int, default=1); p.add_argument("--n", type=int, default=12)
    p.add_argument("--repeats", type=int, default=3); p.add_argument("--workers", type=int, default=8)
    p.add_argument("--split", choices=("train", "val", "test", "development"), default="train")
    p.add_argument("--domain", choices=("train", "reference", "development"), default="train")
    p.add_argument("--timeout", type=float, default=240.)
    a = p.parse_args()
    requests = [dict(seed=a.seed+i, out=a.out, split=a.split, domain=a.domain,
        n=a.n, repeats=a.repeats, timeout=a.timeout) for i in range(a.groups)]
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,
            mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = {pool.submit(worker, r): r for r in requests}
        for future in concurrent.futures.as_completed(pending):
            row = future.result(); results.append(row); print(json.dumps(row), flush=True)
    dump(Path(a.out) / f"batch_{a.split}_{a.seed}.json", results)
    if any("error" in r for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
