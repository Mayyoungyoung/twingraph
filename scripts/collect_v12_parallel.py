"""Complete physical candidate matrices with isolated spawn workers.

Concurrency is for collecting labels. Its wall times are explicitly tagged;
use run_v12_system for sequential, independently timed online comparisons.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time
import traceback


def worker(job):
    from simbench.value.system_v12 import rollout, save
    seed, proposal, directory, domain, level = job
    path = Path(directory) / "result.json"
    try:
        if path.exists():
            result = json.loads(path.read_text())
        else:
            result = rollout(seed, proposal, directory, domain=domain, level=level)
            result["timing_mode"] = "concurrent_label_collection_not_online_decision_latency"
            save(path, result)
        if not result.get("valid"):
            raise RuntimeError(result.get("invalid_reason", "invalid physical trial"))
        return dict(name=proposal["name"], success=result["success"], error=result["error"],
                    wall_seconds=result["total_wall_seconds"], valid=True)
    except Exception:
        error = traceback.format_exc()
        save(Path(directory) / "orchestration_error.json", dict(error=error))
        return dict(name=proposal["name"], valid=False, orchestration_error=error)


def main():
    from simbench.value.system_v12 import collect, require_frozen_source, save
    from simbench.value.plan import digest
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--n", type=int, default=48)
    parser.add_argument("--level", choices=("L0", "L1", "L2"), default="L1")
    parser.add_argument("--domain", choices=("train", "development", "online"), default="train")
    args = parser.parse_args()
    if not 1 <= args.workers <= 20: parser.error("workers must be between 1 and 20")
    source = require_frozen_source()
    for seed in args.seeds:
        started = time.perf_counter()
        root = args.out / f"seed_{seed}" / "collect"
        collect(seed, root, n=args.n, level=args.level, domain=args.domain, names=["__initialize_only__"])
        request = json.loads((root / "request.json").read_text())
        if (request["seed"] != seed or request["domain"] != args.domain
                or request["level"] != args.level or len(request["pool"]) != args.n):
            raise ValueError(f"frozen collection request differs from CLI: {root}")
        request["collection"] = dict(workers=args.workers, process_start="spawn", tasks_per_worker=1,
            collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            timing="concurrent physical labels; not sequential online decision timing")
        save(root / "request.json", request)
        jobs, results = [], []
        for proposal in request["pool"]:
            directory = root / "candidates" / proposal["name"]
            path = directory / "result.json"
            if not path.exists():
                jobs.append((seed, proposal, str(directory), args.domain, args.level))
                continue
            result = json.loads(path.read_text())
            graph_path = directory / "input_graph.json"
            expected_hash = request.get("graph_sha256", {}).get(proposal["name"])
            if (not result.get("valid") or result.get("proposal") != proposal
                    or result.get("runtime_sha256") != source["sha256"]
                    or result.get("seed") != seed
                    or result.get("domain") != args.domain
                    or not graph_path.is_file()
                    or digest(json.loads(graph_path.read_text())) != result.get("input_graph_sha256")
                    or (expected_hash is not None and result.get("input_graph_sha256") != expected_hash)):
                raise ValueError(f"existing physical result failed resume audit: {path}")
            results.append(dict(name=proposal["name"], success=result["success"],
                                error=result["error"], wall_seconds=result["total_wall_seconds"],
                                valid=True, reused=True))
        save(root / "progress.json", dict(seed=seed, n=args.n, completed=len(results),
                                          reused=len(results), pending=len(jobs), trials=results))
        if jobs:
            with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
                for result in pool.imap_unordered(worker, jobs, chunksize=1):
                    results.append(result)
                    save(root / "progress.json", dict(seed=seed, n=args.n, completed=len(results),
                                                      reused=sum(r.get("reused", False) for r in results),
                                                      pending=args.n-len(results), trials=results))
                    console={**result}
                    if console.get("error"):
                        console["error"]=console["error"].split(" {")[0][:240]
                    print(json.dumps(dict(seed=seed, completed=len(results), **console)), flush=True)
        summary = dict(seed=seed, pool_size=args.n, complete=len(results) == args.n and all(r["valid"] for r in results),
            successes=sum(bool(r.get("success")) for r in results), trials=results,
            reused_results=sum(bool(r.get("reused")) for r in results),
            collection_wall_seconds=time.perf_counter() - started, runtime_sha256=source["sha256"],
            timing_note="this invocation only; parallel data collection, not online decision latency")
        save(root / "collection_summary.json", summary)
        require_frozen_source()
        if not summary["complete"]: raise RuntimeError("incomplete matrix: inspect orchestration errors")


if __name__ == "__main__":
    main()
