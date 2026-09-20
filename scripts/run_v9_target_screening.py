"""Actual sequential target-environment screening with ranks frozen first."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

from scripts.analyze_v9_candidate_matrix import rank_model, rule_score
from scripts.collect_v9_candidate_matrix import collect, CONDITIONS
from simbench.value import stage_v9
from simbench.value.v9_candidates import proposals, reference_choices, SOURCE
from simbench.value import stage_v5, stage_v7


def reference_proposal():
    return dict(name="reference", source=SOURCE,
                order=list(stage_v5.legal_orders()[0]), choices=reference_choices(),
                wipe_variant=0, wipe_force=1.5, wipe_duration=14.,
                stroke_minimum=stage_v7.TASK_STROKE_MINIMUM_M)


def run(seed, condition, method, k, out, checkpoint=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    pool = proposals() if method != "fixed_reference" else [reference_proposal()]
    names = [item["name"] for item in pool]
    if method == "fixed_reference":
        order = names
        observation = None
        observation_hash = None
    else:
        _, preview, _, _ = stage_v9.make_scene(seed, out / "pre_execution_scene", role="target")
        observation = preview.decision_observation
        observation_hash = observation["sha256"]
        entry = {item["name"]: dict(observation=observation, proposal=item) for item in pool}
        if method == "simple_rule":
            order = sorted(names, key=lambda name: -rule_score(observation, entry[name]["proposal"]))
        elif method == "random":
            order = random.Random(1771 + seed).sample(names, len(names))
        elif method == "numeric_logistic" or method == "existing_value_v6_architecture":
            if checkpoint is None:
                raise ValueError("model method requires checkpoint")
            order = rank_model(entry, names, checkpoint)
        elif method == "full_12":
            order = names
        else:
            raise ValueError(method)
    rank_seconds = time.perf_counter() - start
    # Seal the entire ranked list before the first physical target rollout.
    (out / "selection_before_execution.json").write_text(json.dumps(dict(
        seed=seed, condition=condition, method=method, k=k, order=order,
        observation_sha256=observation_hash, checkpoint=checkpoint,
        checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() if checkpoint else None,
        candidate_pool_sha256=hashlib.sha256(json.dumps(pool, sort_keys=True).encode()).hexdigest(),
        ranking_wall_seconds=rank_seconds), indent=2))
    attempts = []
    for name in order[:k]:
        proposal = next(p for p in pool if p["name"] == name)
        directory = out / f"attempt_{len(attempts):02d}_{name}"
        row = collect(seed, proposal, condition, directory)
        if observation_hash is not None:
            detail = json.loads((directory / "result.json").read_text())
            if detail["pre_execution_observation"]["sha256"] != observation_hash:
                raise RuntimeError("target initial observation changed across independent sessions")
        attempts.append(row)
        (out / "attempts.json").write_text(json.dumps(attempts, indent=2))
        if row["success"]:
            break
    result = dict(seed=seed, condition=condition, method=method, k=k,
                  task_version=stage_v9.TASK_VERSION, candidate_source="script_curated_v9_r1",
                  selected_before_execution=order, observation_sha256=observation_hash,
                  success=bool(attempts[-1]["success"]), verification_count=len(attempts),
                  verification_wall_seconds=sum(a["wall_seconds"] for a in attempts),
                  full_system_wall_seconds=time.perf_counter() - start, attempts=attempts)
    (out / "result.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--condition", choices=tuple(CONDITIONS), default="nominal")
    parser.add_argument("--method", choices=("fixed_reference", "simple_rule", "random", "full_12",
                                           "numeric_logistic", "existing_value_v6_architecture"), required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    print(json.dumps(run(args.seed, args.condition, args.method, args.k,
                         args.out, args.checkpoint)), flush=True)


if __name__ == "__main__":
    main()
