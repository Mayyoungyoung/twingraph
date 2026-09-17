"""Run paired v6 Top-4 and exhaustive digital-twin screening in parallel."""
import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import traceback

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
if os.name != "nt":
    os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker(request):
    output = Path(request["output"])
    record = dict(seed=request["seed"], output=str(output), status="started")
    started = time.perf_counter()
    try:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("preserve existing v6 system attempt")
        output.mkdir(parents=True, exist_ok=True)
        import torch
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        from simbench.value.physical_v6 import RobustPhysicalRunner
        from simbench.value.system_v5 import SystemConfig, run_pair
        from simbench.value.value_v6 import ValueScorer
        from simbench.value import stage_v6
        scorer = ValueScorer(request["checkpoint"], "cpu")
        if scorer.checkpoint_sha256 != request["checkpoint_sha256"]:
            raise ValueError("checkpoint changed after dispatch")
        config = SystemConfig(seed=request["seed"], n=12, k=4,
            validation_repeats=3, target_repeats=3, accept_rate=.5,
            timeout=240., friction_span=.08, gain_span=.015, render=True)
        pair = run_pair(config, request["planner"], scorer, output,
                        stage=stage_v6, runner_factory=RobustPhysicalRunner)
        record.update(status="completed", target_successes=pair["target_successes"],
                      decision_seconds=pair["decision_seconds"],
                      pair_sha256=sha(output / "pair.json"))
    except Exception as exc:
        record.update(status="error", exception_type=type(exc).__name__,
                      message=str(exc), traceback=traceback.format_exc())
    record["wall_seconds"] = time.perf_counter() - started
    write(output / "worker_status.json", record)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--planner", default="experiments/value_v5/planner_record.json")
    p.add_argument("--out", required=True); p.add_argument("--seed", type=int, default=71400)
    p.add_argument("--groups", type=int, default=12); p.add_argument("--workers", type=int, default=12)
    a = p.parse_args()
    output = Path(a.out)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("preserve existing v6 system batch")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = str(Path(a.checkpoint).resolve())
    planner = read(a.planner)
    binding = dict(checkpoint=checkpoint, checkpoint_sha256=sha(checkpoint),
                   planner=planner, planner_sha256=hashlib.sha256(
                       json.dumps(planner, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
    requests = [dict(seed=a.seed+i, output=str((output / f"case_{a.seed+i}").resolve()), **binding)
                for i in range(a.groups)]
    write(output / "dispatch.json", dict(schema="twingraph.system_dispatch.v6",
        seeds=[r["seed"] for r in requests], workers=a.workers,
        checkpoint_sha256=binding["checkpoint_sha256"], planner_sha256=binding["planner_sha256"],
        requests=[{k: v for k, v in r.items() if k != "planner"} for r in requests]))
    started = time.perf_counter(); records = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,
            mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = {pool.submit(worker, request): request for request in requests}
        for future in concurrent.futures.as_completed(pending):
            row = future.result(); records.append(row)
            print(json.dumps({k: row.get(k) for k in ("seed", "status", "wall_seconds")}), flush=True)
    policy_rows = []
    for record in records:
        for policy in ("top_k", "full"):
            path = Path(record["output"]) / policy / "result.json"
            if path.is_file():
                policy_rows.append(read(path))
    summaries = {}
    for policy in ("top_k", "full"):
        rows = [r for r in policy_rows if r["policy"] == policy]
        requested_trials = a.groups * 3
        summaries[policy] = dict(requested_configurations=a.groups,
            completed_configurations=len(rows), target_successes=sum(r["target_successes"] for r in rows),
            requested_target_trials=requested_trials,
            target_success_rate=sum(r["target_successes"] for r in rows) / requested_trials,
            mean_decision_seconds=float(sum(r["seconds"]["decision"] for r in rows) / len(rows)) if rows else None,
            mean_total_seconds=float(sum(r["seconds"]["total"] for r in rows) / len(rows)) if rows else None,
            twin_validation_calls=sum(r["simulation_calls"]["twin_validation"] for r in rows),
            target_execution_calls=sum(r["simulation_calls"]["target_execution"] for r in rows))
    result = dict(schema="twingraph.system_batch.v6", status="completed" if
        len(records) == a.groups and all(r["status"] == "completed" for r in records) else "error",
        checkpoint_sha256=binding["checkpoint_sha256"], planner_sha256=binding["planner_sha256"],
        wall_seconds=time.perf_counter()-started, records=sorted(records, key=lambda r: r["seed"]),
        summaries=summaries)
    write(output / "system_batch.json", result)
    write(output / "timing_rows.json", dict(schema="twingraph.system_rows.v6", rows=policy_rows))
    print(json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True)
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
