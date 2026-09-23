#!/usr/bin/env python3
"""Freeze label-blind recombinations of generic atomic-skill choices.

Each donor is an ordinary observation-grounded proposal. Independent seeded
permutations choose donors for each manipulated role, cleaning controller and
assembly order. The construction never opens a twin result or stage label.
Every recombined complete plan is compiled and graph-validated before use.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np

from simbench.value.plan import digest
from simbench.value.planner_v12 import normalized_graph, propose
from simbench.value.system_v11 import save
from simbench.value.system_v12 import make_scene, require_frozen_source


def recombine(base, *, seed, n):
    if not 1 <= n <= len(base):
        raise ValueError("recombined pool cannot exceed its donor pool")
    roles = tuple(base[0]["choices"])
    if any(set(item["choices"]) != set(roles) for item in base):
        raise ValueError("donor plans have incompatible manipulated roles")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 20260923]))
    source_index = {role: rng.permutation(len(base))[:n].tolist() for role in roles}
    clean_index = rng.permutation(len(base))[:n].tolist()
    order_index = rng.permutation(len(base))[:n].tolist()
    candidates = []
    for i in range(n):
        candidate = deepcopy(base[order_index[i]])
        candidate["name"] = f"recombined_{i:03d}"
        candidate["source"] = "label_blind_atomic_skill_port_recombination_v20"
        candidate["rationale"] = "independent seeded donor choice per generic manipulated role"
        for role in roles:
            donor = base[source_index[role][i]]
            candidate["choices"][role] = deepcopy(donor["choices"][role])
            candidate["necessary_geometry"][role] = deepcopy(donor["necessary_geometry"][role])
        cleaning = base[clean_index[i]]
        for key in ("wipe_variant", "wipe_force", "wipe_duration"):
            candidate[key] = cleaning[key]
        candidates.append(candidate)
    if len({digest({k:v for k,v in item.items() if k not in ("name","source","rationale")})
            for item in candidates}) != n:
        raise ValueError("duplicate executable recombination")
    return candidates, dict(role_donor_indices=source_index, cleaning_donor_indices=clean_index,
                            order_donor_indices=order_index)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--base-n", type=int, default=48)
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--level", choices=("L0", "L1", "L2"), default="L1")
    p.add_argument("--domain", choices=("train", "development", "online"), default="train")
    args = p.parse_args()
    if args.n > args.base_n:
        p.error("n cannot exceed base-n")
    source = require_frozen_source()
    for seed in args.seeds:
        root = args.out / f"seed_{seed}" / "collect"
        request_path = root / "request.json"
        if request_path.exists():
            raise ValueError(f"frozen candidate request already exists: {request_path}")
        started = time.perf_counter()
        _, session, _, _ = make_scene(seed, root / "decision", domain=args.domain, level=args.level)
        base, grounding = propose(session.decision_observation,cad=session.planning_cad,
                                  n=args.base_n,seed=seed)
        pool, permutation = recombine(base,seed=seed,n=args.n)
        graph_hashes = {}
        for candidate in pool:
            graph = normalized_graph(session,candidate)
            graph_hashes[candidate["name"]] = digest(graph)
        generation_seconds = time.perf_counter()-started
        save(root / "generation_input.json",dict(seed=seed,level=args.level,
            observation=session.decision_observation,cad=session.planning_cad,
            runtime_sha256=source["sha256"]))
        save(request_path,dict(seed=seed,level=args.level,domain=args.domain,
            pool=pool,source=dict(base=grounding,recombination=permutation,
                construction="label_blind_rolewise_permutation_before_twin_rollout",
                online_llm_call=False),
            graph_sha256=graph_hashes,geometry_version=session.task_version,
            initial_observation=session.decision_observation,
            runtime_sha256=source["sha256"],generation_seconds=generation_seconds))
        save(root / "runtime_sources.json",source)
        print(json.dumps(dict(seed=seed,n=len(pool),base_n=len(base),
            runtime_sha256=source["sha256"],generation_seconds=generation_seconds)),
            flush=True)


if __name__ == "__main__":
    main()
