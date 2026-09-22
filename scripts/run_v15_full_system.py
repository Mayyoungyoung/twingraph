#!/usr/bin/env python3
"""Codex strategies -> generic value Top-K -> complete-task twin verification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from scripts.v15_llm_contract import (
    compile_response,
    failure_summary,
    file_sha,
    load_response,
    planner_request,
)
from simbench.value.generic_graph_value_v15 import ValueRankerV15
from simbench.value.plan import digest
from simbench.value.provenance_v12 import fingerprint
from simbench.value.system_v11 import save
from simbench.value.system_v12 import make_scene, rollout
from simbench.value.planner_v12 import normalized_graph, propose


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prepare(seed, root, protocol, *, n, level, domain, round_index):
    round_root = root / f"round_{round_index}"
    round_root.mkdir(parents=True, exist_ok=True)
    previous = []
    failure_summaries = []
    if round_index:
        prior = load_json(root / "round_0" / "compiled_plans.json")
        previous = [row["proposal"]["name"] for row in prior["plans"]]
        failure_summaries = load_json(root / "round_0" / "failure_summaries.json")
    started = time.perf_counter()
    _, session, _, _ = make_scene(seed, round_root / "decision", domain=domain, level=level)
    observation = session.decision_observation
    pool, grounding = propose(observation, cad=session.planning_cad, n=n, seed=seed)
    generation_seconds = time.perf_counter() - started
    pool_record = dict(seed=seed, n=n, level=level, domain=domain, pool=pool,
        pool_sha256=digest(pool), grounding=grounding, runtime=fingerprint(),
        observation_sha256=observation["sha256"], planning_cad=session.planning_cad,
        planning_cad_sha256=digest(session.planning_cad),
        generation_seconds=generation_seconds, excluded_previous_names=previous)
    save(round_root / "search_pool.json", pool_record)
    request = planner_request(protocol["task_spec"], observation, session.planning_cad,
                              round_index=round_index, failure_summaries=failure_summaries)
    request.update(search_pool_contract=dict(n=n, candidate_output=protocol["candidate_generation"]["round_candidate_n"],
        excluded_previous_names=previous, grounding="strategy slots matched to distinct complete plans"))
    save(round_root / "llm_request.json", request)
    return session, pool, grounding, previous, request, generation_seconds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("simbench/configs/value_v15_full_system.json"))
    parser.add_argument("--response", type=Path)
    parser.add_argument("--round", type=int, choices=(0, 1), default=0)
    parser.add_argument("--search-n", type=int, default=48)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--level", choices=("L0", "L1", "L2"), default="L2")
    parser.add_argument("--domain", choices=("train", "development", "online"), default="online")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--audited-legacy-full-task", action="store_true",
        help="Accept the frozen V12 rollout format, which predates evaluation_scope but always executes run_staged")
    parser.add_argument("--physical-base-runtime-sha256")
    args = parser.parse_args()
    protocol = load_json(args.protocol)
    declared = protocol["candidate_generation"]
    if args.search_n != declared["search_pool_n"] or args.k != declared["top_k"]:
        parser.error("search-n/K must match the frozen V15 protocol")
    root = args.out / f"seed_{args.seed}"
    session, pool, grounding, previous, request, generation_seconds = prepare(
        args.seed, root, protocol, n=args.search_n, level=args.level,
        domain=args.domain, round_index=args.round)
    round_root = root / f"round_{args.round}"
    if args.prepare_only:
        print(json.dumps(dict(status="awaiting_codex_response", request=str(round_root / "llm_request.json"),
                              request_sha256=file_sha(round_root / "llm_request.json")), ensure_ascii=False))
        return
    if args.response is None:
        parser.error("--response is required unless --prepare-only is used")
    response = load_response(args.response, args.round)
    if response["candidate_count"] != declared["round_candidate_n"]:
        raise ValueError("Codex response candidate count differs from frozen protocol")
    save(round_root / "llm_response.json", response)
    compile_started = time.perf_counter()
    plans, bindings = compile_response(pool, response, excluded_names=previous)
    graphs = [normalized_graph(session, proposal) for proposal in plans]
    for proposal, graph in zip(plans, graphs):
        save(round_root / "graphs" / f"{proposal['name']}.json", graph)
    compilation_seconds = time.perf_counter() - compile_started
    compiled = dict(response_sha256=file_sha(round_root / "llm_response.json"),
        request_sha256=file_sha(round_root / "llm_request.json"), pool_sha256=digest(pool),
        bindings=bindings, plans=[dict(proposal=proposal, graph_sha256=digest(graph))
                                  for proposal, graph in zip(plans, graphs)],
        complete_plan_count=len(plans), compilation_seconds=compilation_seconds,
        external_llm_api_call=False,
        llm_latency_note="Codex response was authored interactively and replayed; no external API latency was measured")
    save(round_root / "compiled_plans.json", compiled)
    ranker = ValueRankerV15(args.checkpoint)
    began = time.perf_counter(); scores = np.asarray(ranker.score(graphs), float)
    ranking_seconds = time.perf_counter() - began
    order = np.argsort(-scores, kind="stable").tolist()
    save(round_root / "ranking.json", dict(model_sha256=ranker.sha256, input_schema=protocol["value_model"]["input_schema"],
        scores=[float(x) for x in scores], order=order, k=args.k,
        ranking_seconds=ranking_seconds, selected=[plans[i]["name"] for i in order[:args.k]]))
    trials = []; failures = []; verified = None
    verification_started = time.perf_counter()
    for rank, index in enumerate(order[:args.k], 1):
        proposal = plans[index]
        candidate_root = round_root / "twins" / proposal["name"]
        result_path = candidate_root / "result.json"
        result = load_json(result_path) if result_path.exists() else rollout(
            args.seed, proposal, candidate_root, domain=args.domain, level=args.level, record=args.record)
        modern_scope = bool(result.get("full_task_label")) and result.get("evaluation_scope") == "complete_functional_task"
        legacy_scope = (args.audited_legacy_full_task and
                        {"cleaning_pass", "assembly_pass", "functional_test_pass"}.issubset(
                            result.get("stage_passes", {})))
        if not result.get("valid") or not (modern_scope or legacy_scope):
            raise RuntimeError("invalid, censored or non-full-task twin result")
        row = dict(rank=rank, plan_index=index, name=proposal["name"], score=float(scores[index]),
                   success=bool(result["success"]), error=result.get("error"),
                   seconds=float(result["total_wall_seconds"]), result=str(result_path),
                   result_sha256=file_sha(result_path))
        trials.append(row)
        if result["success"]:
            verified = dict(**row, proposal=proposal, graph_sha256=digest(graphs[index]))
            save(round_root / "verified_plan.json", verified)
            break
        failures.append(failure_summary(result, rank, scores[index]))
        save(round_root / "progress.json", dict(trials=trials, verified=verified))
    verification_seconds = time.perf_counter() - verification_started
    save(round_root / "failure_summaries.json", failures)
    status = "verified_success" if verified else ("awaiting_round_1_codex_response" if args.round == 0 else "two_round_budget_exhausted")
    summary = dict(schema="twingraph.full_system_result.v15.r1", seed=args.seed, round=args.round,
        status=status, complete_candidate_count=len(plans), top_k=args.k, trials=trials,
        verified=verified, failures=failures, runtime_sha256=fingerprint()["sha256"],
        physical_base_runtime_sha256=args.physical_base_runtime_sha256,
        evaluation_scope=("audited_frozen_v12_complete_functional_task" if args.audited_legacy_full_task
                          else "complete_functional_task"),
        checkpoint_sha256=ranker.sha256, generation_seconds=generation_seconds,
        codex_response_compilation_seconds=compilation_seconds, ranking_seconds=ranking_seconds,
        verification_seconds=verification_seconds,
        measured_pipeline_seconds=generation_seconds + compilation_seconds + ranking_seconds + verification_seconds,
        external_llm_api_call=False, external_llm_latency_seconds=None,
        timing_note="fresh serial complete-task twin verification; interactive Codex authoring latency is not included")
    save(round_root / "summary.json", summary)
    if not verified and args.round == 0:
        next_request = planner_request(protocol["task_spec"], session.decision_observation, session.planning_cad,
                                       round_index=1, failure_summaries=failures)
        next_request["avoid_grounded_names"] = [row["proposal"]["name"] for row in compiled["plans"]]
        save(root / "round_1_pending_request.json", next_request)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
