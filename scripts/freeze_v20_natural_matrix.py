#!/usr/bin/env python3
"""Freeze independent full-task natural pools before any candidate rollout.

This invokes the existing V12 generator unchanged. It records every generated
plan and its normalized graph hash before outcomes exist. Seeds and fold roles
are declared on the command line, never inferred from rollout labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from simbench.value.full_flow_graph_value_v20 import encode_graph
from simbench.value.plan import digest
from simbench.value.planner_v12 import normalized_graph
from simbench.value.system_v12 import collect, make_scene, require_frozen_source, save


def fold(seed: int) -> str:
    return "test" if seed % 5 == 0 else "validation" if seed % 5 == 1 else "train"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--n", type=int, default=48)
    parser.add_argument("--level", choices=("L0", "L1", "L2"), default="L0")
    parser.add_argument("--domain", choices=("train", "development", "online"), default="online")
    args = parser.parse_args()
    seeds = list(args.seeds)
    if len(seeds) != len(set(seeds)) or seeds != sorted(seeds):
        parser.error("seeds must be unique and in ascending predeclared order")
    if not 1 <= args.n <= 512:
        parser.error("candidate count outside generator contract")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("output directory must be empty; frozen requests are immutable")
    source = require_frozen_source()
    manifest = dict(schema="twingraph.natural_full_task_matrix_freeze.v20.r1",
                    task_scope="complete_functional_task",
                    generator="simbench.value.planner_v12.propose",
                    candidate_count=args.n, level=args.level, domain=args.domain,
                    seeds=seeds, fold_rule="seed modulo 5: 0 test, 1 validation, 2/3/4 train",
                    runtime_sha256=source["sha256"],
                    selection_policy="retain every attempted layout; train only on natural mixed pools; report unconditional coverage",
                    outcome_labels_read=False, layouts=[])
    for seed in seeds:
        root = args.out / f"seed_{seed}" / "collect"
        collect(seed, root, n=args.n, level=args.level, domain=args.domain,
                names=["__freeze_without_rollout__"])
        request_path = root / "request.json"
        request = json.loads(request_path.read_text(encoding="utf-8"))
        if any((root / "candidates").glob("*/result.json")):
            raise RuntimeError("a candidate executed before matrix freeze")
        _, session, _, _ = make_scene(seed, root / "graph_binding",
                                      domain=args.domain, level=args.level)
        if session.decision_observation["sha256"] != request["initial_observation"]["sha256"]:
            raise RuntimeError("independent scene regeneration changed the observation")
        hashes = {}
        for proposal in request["pool"]:
            graph = normalized_graph(session, proposal)
            encode_graph(graph)
            hashes[proposal["name"]] = digest(graph)
        if len(hashes) != args.n:
            raise RuntimeError("plan or graph collision during freeze")
        request["graph_sha256"] = hashes
        request["freeze_policy"] = "entire pool before any result; no label filtering"
        save(request_path, request)
        manifest["layouts"].append(dict(seed=seed, fold=fold(seed),
            request_sha256=hashlib.sha256(request_path.read_bytes()).hexdigest(),
            initial_observation_sha256=request["initial_observation"]["sha256"],
            candidate_names=list(hashes), graph_sha256=hashes))
        print(json.dumps(dict(seed=seed, fold=fold(seed), plans=len(hashes),
                              runtime_sha256=source["sha256"])), flush=True)
    save(args.out / "freeze_manifest.json", manifest)
    require_frozen_source()


if __name__ == "__main__":
    main()
