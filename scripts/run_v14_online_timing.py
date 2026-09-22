"""Fresh-rollout wall-clock timing for methods frozen before V14 blind test."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.evaluate_v14_blind import load_ensemble
from scripts.train_value_v12 import dump, predict
from simbench.value.graph_value_v12 import encode_graph
from simbench.value.plan import digest
from simbench.value.planner_v12 import normalized_graph, propose
from simbench.value.system_v12 import make_scene, rollout


def ordered_pool(method, pool, scores, baselines, random_seed, layout):
    by_name = {p["name"]: p for p in pool}
    names = sorted(by_name)
    if method == "locked_contextual_model":
        order = [pool[i]["name"] for i in np.argsort(-scores, kind="stable")]
    elif method == "train_best_fixed_top2":
        prefix = baselines["best_fixed_top2_order"]
        order = list(prefix) + [n for n in names if n not in prefix]
    elif method == "reference_first":
        prefix = baselines["reference_first_order"]
        order = list(prefix) + [n for n in names if n not in prefix]
    elif method == "uniform_random":
        rng = np.random.default_rng(np.random.SeedSequence([random_seed, layout]))
        order = [names[i] for i in rng.permutation(len(names))]
    else:
        raise ValueError(method)
    return [by_name[n] for n in order]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix-root", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--baseline-lock", type=Path, required=True)
    p.add_argument("--model-lock", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(); torch.set_num_threads(1)
    split = json.loads(args.split.read_text()); timing = split["online_timing"]
    baselines = json.loads(args.baseline_lock.read_text())
    _, models = load_ensemble(args.model_lock, args.device)
    if args.out.exists() and any(args.out.rglob("result.json")):
        raise FileExistsError("online timing output already contains trials; use a fresh directory")
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for layout in timing["layout_seeds"]:
        archived = json.loads((args.matrix_root / f"seed_{layout}" / "end_stop" / "request.json").read_text())
        generation_started = time.perf_counter()
        _, session, _, _ = make_scene(layout, args.out / f"seed_{layout}" / "planning",
                                      domain="development", level=split["level"])
        pool, _ = propose(session.decision_observation, cad=session.planning_cad,
                          n=split["candidate_count"], seed=layout)
        graphs = [normalized_graph(session, proposal) for proposal in pool]
        generation_wall = time.perf_counter() - generation_started
        if digest(pool) != digest(archived["pool"]):
            raise ValueError(f"fresh candidate pool differs from frozen blind request: {layout}")
        encoding_started = time.perf_counter()
        encoded = [encode_graph(graph) for graph in graphs]
        encoding_wall = time.perf_counter() - encoding_started
        rows = [dict(encoded=value) for value in encoded]
        inference_started = time.perf_counter()
        scores = np.mean([predict(model, rows, args.device) for model in models], axis=0)
        inference_wall = time.perf_counter() - inference_started
        for method in timing["methods"]:
            planning_overhead = (encoding_wall + inference_wall
                                 if method == "locked_contextual_model" else 0.0)
            ordered = ordered_pool(method, pool, scores, baselines,
                                   timing["random_seed"], layout)
            trials = []
            for rank, proposal in enumerate(ordered[:timing["max_candidates_per_method"]], 1):
                root = args.out / f"seed_{layout}" / method / f"rank_{rank}_{proposal['name']}"
                started = time.perf_counter()
                result = rollout(layout, proposal, root, domain="development",
                                 level=split["level"], stop_after=split["stop_after"])
                measured = time.perf_counter() - started
                trials.append(dict(rank=rank, candidate=proposal["name"],
                                   success=bool(result["success"]), valid=bool(result["valid"]),
                                   measured_rollout_wall_seconds=measured,
                                   reported_rollout_wall_seconds=float(result["total_wall_seconds"]),
                                   simulator_seconds=float(result.get("sim_seconds", result.get("physics_steps", 0))),
                                   error=result.get("error", "")))
                if result["success"]:
                    break
            rollout_wall = sum(t["measured_rollout_wall_seconds"] for t in trials)
            records.append(dict(layout=layout, method=method, success=bool(trials[-1]["success"]),
                                calls=len(trials), generation_wall_seconds=generation_wall,
                                encoding_wall_seconds=(encoding_wall if method == "locked_contextual_model" else 0.0),
                                inference_wall_seconds=(inference_wall if method == "locked_contextual_model" else 0.0),
                                rollout_wall_seconds=rollout_wall,
                                end_to_end_wall_seconds=generation_wall + planning_overhead + rollout_wall,
                                trials=trials))
            dump(args.out / "ONLINE_TIMING_PROGRESS.json", dict(records=records, split_online_timing=timing))
    summary = {}
    for method in timing["methods"]:
        group = [r for r in records if r["method"] == method]
        summary[method] = dict(layouts=len(group), success=float(np.mean([r["success"] for r in group])),
                               calls=float(np.mean([r["calls"] for r in group])),
                               generation_wall_seconds=float(np.mean([r["generation_wall_seconds"] for r in group])),
                               encoding_wall_seconds=float(np.mean([r["encoding_wall_seconds"] for r in group])),
                               inference_wall_seconds=float(np.mean([r["inference_wall_seconds"] for r in group])),
                               rollout_wall_seconds=float(np.mean([r["rollout_wall_seconds"] for r in group])),
                               end_to_end_wall_seconds=float(np.mean([r["end_to_end_wall_seconds"] for r in group])))
    report = dict(schema="twingraph.contextual_online_timing.v14.v1", records=records,
                  summary=summary, same_pool_fresh_rollouts=True,
                  scope="physical execution prefix through end_stop only", full_task_claim=False)
    dump(args.out / "ONLINE_TIMING.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
