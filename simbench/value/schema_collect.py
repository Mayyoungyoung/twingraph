"""Collect complete graph-bound continuations in the original sliding stage."""
import argparse
import concurrent.futures
import contextlib
from dataclasses import asdict
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
import traceback

import numpy as np

from .collect import dump
from .physical import PhysicalRunner, perturbation
from .plan import digest, plain, execute_calls
from .skill_graph import compile_graph
from . import stage_assembly as stage


def source_manifest():
    root = Path(__file__).parents[2]
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for d in ("simbench/value", "simbench/assembly", "simbench/core") for p in sorted((root/d).glob("*.py"))}


def create_scene(seed, directory, checkpoint=2):
    spec, session, path, targets = stage.make_family(stage.FAMILY, seed, directory)
    if checkpoint not in (2, 3):
        raise ValueError("checkpoint must be after carriage/stop or after an additional real pin")
    if checkpoint == 3:
        # The existing production recipe selects its current grasp/route using
        # necessary geometry, never labels of the candidate continuations.
        from simbench.assembly.task import pick, transfer_part, release
        part = "pin_left"
        target = np.asarray(targets[part])
        begin = len(session.results)
        pick(session, part, terminal_targets=[dict(id="seated", part=part, xyz=target)])
        transfer_part(session, part, target + [0, 0, .069])
        session.call("move", reference="object", part=part, target=target + [0, 0, .069])
        session.call("move", mode="guarded", part=part, target_z=float(target[2]), force_stop=3.)
        session.call("press", part=part, target_z=float(target[2]))
        release(session)
        session.call("inspect", part=part, target=target)
        session.stage_checkpoint_trace.extend(plain(session.results[begin:]))
        session.stage_completed = (*stage.CHECKPOINT_PARTS, "pin_left")
        stage.check_preconditions(session, targets, session.stage_completed)
    return spec, session, path, targets


def collect_one(seed, out, split="train", domain="train", n=16, repeats=2, checkpoint=2):
    directory = Path(out)/f"group_{stage.FAMILY}_{seed}_{checkpoint}"
    directory.mkdir(parents=True, exist_ok=True)
    request = dict(seed=seed, split=split, domain=domain, n=n, repeats=repeats, checkpoint=checkpoint, timeout_seconds=180.)
    if (directory/"complete.json").exists():
        done = json.loads((directory/"complete.json").read_text())
        if done["request_sha256"] != digest(request):
            raise ValueError("resume request mismatch")
        return done
    started = time.perf_counter()
    with (directory/"steps.log").open("w") as log, contextlib.redirect_stdout(log):
        spec, session, path, targets = create_scene(seed, directory, checkpoint)
        checkpoint_seconds = time.perf_counter()-started
        plans, counts = stage.build_pool(session, targets, seed, n=n)
        observation = stage.observed(session, targets)
        runner = PhysicalRunner(session, timeout=180.)
        inputs = dict(schema="twingraph.group.v2", group_id=spec.config_id+f"_cp{checkpoint}",
            split_group=spec.config_id, declared_split=split, task=asdict(spec), checkpoint=checkpoint,
            checkpoint_trace=session.stage_checkpoint_trace, checkpoint_seconds=checkpoint_seconds,
            observation=observation, candidates=[p.to_dict() for p in plans], pool_counts=counts,
            source_sha256=digest(source_manifest()), snapshot_sha256=runner.initial)
        ih = digest(inputs)
        graphs = [compile_graph(observation, p) for p in plans]
        dump(directory/"inputs.json", inputs)
        dump(directory/"skill_graphs.json", dict(input_sha256=ih, graphs=graphs))
        dump(directory/"source.json", source_manifest())
        np.savez_compressed(directory/"initial_physics.npz", **runner.snapshot["physics"]["data"])
        trials = []
        for repeat in range(repeats):
            trial = perturbation(seed, repeat, domain)
            for graph in graphs:
                trials.append(runner.run(graph, trial, keep_trace=True))
                dump(directory/"outcomes.json", dict(input_sha256=ih, trials=trials))
        result = dict(request_sha256=digest(request), input_sha256=ih, source_sha256=inputs["source_sha256"],
            family=stage.FAMILY, seed=seed, checkpoint=checkpoint, split=split, candidates=len(plans),
            trials=len(trials), full_successes=sum(t["success"] for t in trials),
            prefix_successes=sum(t["prefix_success"] for t in trials),
            timeouts=sum(t["timeout"] for t in trials), plan_lengths=sorted({len(p.calls) for p in plans}),
            wall_seconds=time.perf_counter()-started)
        if result["timeouts"]:
            dump(directory/"censored.json", result)
            raise RuntimeError("censored paired group retained; do not train as physical failure")
        dump(directory/"complete.json", result)
    return result


def worker(request):
    try:
        return collect_one(**request)
    except Exception:
        result = dict(request=request, error=traceback.format_exc())
        directory = Path(request["out"])/f"group_{stage.FAMILY}_{request['seed']}_{request['checkpoint']}"
        directory.mkdir(parents=True, exist_ok=True)
        dump(directory/"failure.json", result)
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True); p.add_argument("--seed", type=int, required=True)
    p.add_argument("--groups", type=int, default=1); p.add_argument("--n", type=int, default=16)
    p.add_argument("--repeats", type=int, default=2); p.add_argument("--workers", type=int, default=4)
    p.add_argument("--split", default="train"); p.add_argument("--domain", default="train")
    p.add_argument("--checkpoints", nargs="+", type=int, default=[2])
    p.add_argument("--alternate-checkpoints", action="store_true")
    a = p.parse_args()
    requests = [dict(seed=a.seed+i, out=a.out, split=a.split, domain=a.domain, n=a.n,
                     repeats=a.repeats, checkpoint=cp) for i in range(a.groups)
                for cp in ([a.checkpoints[(a.seed+i) % len(a.checkpoints)]] if a.alternate_checkpoints else a.checkpoints)]
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for result in pool.map(worker, requests):
            results.append(result)
            print(json.dumps(result), flush=True)
    dump(Path(a.out)/f"batch_{a.seed}_{a.split}.json", results)
    if any("error" in row for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
