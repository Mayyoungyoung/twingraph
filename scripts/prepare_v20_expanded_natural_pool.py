#!/usr/bin/env python3
"""Freeze a larger nested natural pool and retain every earlier trial.

The V12 Halton proposal sequence is nested: increasing N does not alter the
earlier named proposals. This script checks that fact for every layout before
writing the expanded request. It reads no rollout label when generating plans.
The separate copy mode later carries ALL earlier results into the expanded
matrix, including failures and invalid trials; nothing is cherry-picked.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

from simbench.value.full_flow_graph_value_v20 import encode_graph
from simbench.value.plan import digest
from simbench.value.planner_v12 import normalized_graph, propose
from simbench.value.system_v11 import save
from simbench.value.system_v12 import make_scene, require_frozen_source


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def freeze(base_path, out_root, n):
    base = read(base_path)
    seed = int(base["seed"])
    old = {candidate["name"]: candidate for candidate in base["pool"]}
    if not len(old) == len(base["pool"]) < n <= 512:
        raise ValueError("expanded pool must add candidates to a unique base pool")
    root = Path(out_root) / f"seed_{seed}" / "collect"
    if (root / "request.json").exists():
        raise ValueError(f"expanded request already frozen: {root}")
    source = require_frozen_source()
    if source["sha256"] != base["runtime_sha256"]:
        raise ValueError("base and expanded pools require one physical runtime")
    started = time.perf_counter()
    _, session, _, _ = make_scene(seed, root / "decision",
        domain=base["domain"], level=base["level"])
    if session.decision_observation["sha256"] != base["initial_observation"]["sha256"]:
        raise ValueError("initial random layout changed")
    pool, grounding = propose(session.decision_observation,
        cad=session.planning_cad, n=n, seed=seed)
    expanded = {candidate["name"]: candidate for candidate in pool}
    if not all(name in expanded and digest(candidate) == digest(expanded[name])
               for name, candidate in old.items()):
        raise ValueError("larger grounded pool changed an earlier complete plan")
    hashes = {}
    for candidate in pool:
        graph = normalized_graph(session, candidate)
        encode_graph(graph)
        hashes[candidate["name"]] = digest(graph)
    save(root / "request.json", dict(seed=seed, level=base["level"],
        domain=base["domain"], pool=pool, graph_sha256=hashes,
        source=dict(grounding=grounding, construction="nested_natural_expansion_before_new_rollouts",
            base_request_sha256=digest(base), base_candidate_count=len(old),
            old_names_unchanged=True, outcome_labels_read_during_generation=False,
            all_old_results_must_be_retained=True),
        geometry_version=session.task_version,
        initial_observation=session.decision_observation,
        runtime_sha256=source["sha256"],
        generation_seconds=time.perf_counter()-started))
    save(root / "runtime_sources.json", source)
    return dict(seed=seed, old=len(old), expanded=n, runtime_sha256=source["sha256"])


def copy_all(base_path, out_root):
    base = read(base_path)
    seed = int(base["seed"])
    root = Path(out_root) / f"seed_{seed}" / "collect"
    expanded = read(root / "request.json")
    if expanded["source"]["base_request_sha256"] != digest(base):
        raise ValueError("expanded request is bound to a different base request")
    hashes = expanded["graph_sha256"]
    validated = []
    for candidate in base["pool"]:
        name = candidate["name"]
        old_dir = Path(base_path).parent / "candidates" / name
        new_dir = root / "candidates" / name
        result_path = old_dir / "result.json"
        graph_path = old_dir / "input_graph.json"
        if not result_path.is_file() or not graph_path.is_file():
            raise ValueError(f"base pool is not complete: {name}")
        result = read(result_path)
        graph = read(graph_path)
        if (result.get("runtime_sha256") != expanded["runtime_sha256"]
                or result.get("proposal") != candidate
                or digest(graph) != hashes[name]
                or result.get("input_graph_sha256") != hashes[name]):
            raise ValueError(f"base trial cannot be reused with expanded graph: {name}")
        for filename in ("result.json", "input_graph.json"):
            destination = new_dir / filename
            if destination.exists():
                raise ValueError(f"refusing to overwrite an expanded trial: {destination}")
        validated.append((old_dir, new_dir))
    for old_dir, new_dir in validated:
        new_dir.mkdir(parents=True, exist_ok=True)
        for filename in ("result.json", "input_graph.json"):
            shutil.copy2(old_dir / filename, new_dir / filename)
    copied = len(validated)
    save(root / "base_result_reuse.json", dict(base_request_sha256=digest(base),
        source_directory=str(Path(base_path).parent), copied_all_base_candidates=copied,
        selection_rule="all names, regardless of full success or failure"))
    return dict(seed=seed, copied_all_base_candidates=copied)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-request", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=96)
    parser.add_argument("--copy-all-base-results", action="store_true")
    args = parser.parse_args()
    seeds = [int(read(path)["seed"]) for path in args.base_request]
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate base layout")
    for path in args.base_request:
        result = (copy_all(path, args.out) if args.copy_all_base_results
                  else freeze(path, args.out, args.n))
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
