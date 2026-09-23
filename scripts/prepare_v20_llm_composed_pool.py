#!/usr/bin/env python3
"""Ask an LLM to compose complete grounded atomic plans without rollout labels.

The model selects among pre-execution solver solutions for each manipulated
role, legal task order, and cleaning ports. This script never reads results.
The response and each bound full-flow graph are frozen before twin execution.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

from simbench.assembly.graph import catalog_graph
from simbench.assembly.interfaces import PUBLIC_SKILLS
from simbench.value.full_flow_graph_value_v20 import encode_graph
from simbench.value.plan import digest
from simbench.value.planner_v12 import normalized_graph
from simbench.value.system_v11 import save
from simbench.value.system_v12 import make_scene, require_frozen_source


FLAGS = (
    "cleaning_pass", "assembly_pass", "functional_test_pass",
    "fixture_capture_pass", "final_seat_pass",
    "final_release_and_retraction_pass",
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prompt_input(base_request, candidate_count, *, graph_schema=True):
    """Expose task, generic skills and legal pre-execution port solutions."""
    pool = base_request["pool"]
    if not pool or len({item["name"] for item in pool}) != len(pool):
        raise ValueError("base candidate request is empty or contains duplicates")
    roles = tuple(pool[0]["choices"])
    if any(set(item["choices"]) != set(roles) for item in pool):
        raise ValueError("base proposals have inconsistent manipulated roles")
    if not 1 <= candidate_count <= len(pool):
        raise ValueError("candidate count must fit the grounded donor library")
    obs = base_request["initial_observation"]
    def short(value):
        if isinstance(value, float):
            return round(value, 6)
        if isinstance(value, dict):
            return {key: short(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [short(item) for item in value]
        return value
    objects = {name: {key: row.get(key) for key in
               ("position_m", "quat_wxyz", "valid", "quality", "fit_residual_m")}
               for name, row in obs["objects"].items()}
    options = {}
    for role in roles:
        scalar_keys = sorted({key for item in pool for key,value in item["choices"][role].items()
                              if isinstance(value,(int,float,str,bool))})
        fields = ["donor_index", *scalar_keys, "geometry_status", "geometry_margin_m"]
        rows = []
        for i,item in enumerate(pool):
            condition = item["necessary_geometry"][role]
            clearance = condition.get("min_clearance_m")
            required = condition.get("required_clearance_m")
            margin = (float(clearance)-float(required)
                      if isinstance(clearance,(int,float)) and isinstance(required,(int,float))
                      else None)
            rows.append([i, *[short(item["choices"][role].get(key)) for key in scalar_keys],
                         condition.get("status"), short(margin)])
        options[role] = dict(fields=fields, rows=rows)
    request = dict(schema=("twingraph.llm_atomic_composition_request.v20.r2" if graph_schema
                           else "twingraph.llm_atomic_composition_request.v20.r1"),
        observation_sha256=obs["sha256"], base_pool_sha256=digest(pool),
        candidate_count=candidate_count,
        task=dict(scope="complete_functional_task",
            success_definition="all six final physical acceptance predicates true in one complete rollout",
            final_acceptance_flags=list(FLAGS),
            required_stroke_m=pool[0]["stroke_minimum"],
            target_geometry=short(obs.get("assembly_targets", {})),
            legal_orders=[list(order) for order in sorted({tuple(item["order"]) for item in pool})]),
        observation=dict(objects=short(objects),
                         fixture_centers={name:short(row.get("position_m")) for name,row in
                             obs.get("fixtures", {}).items()}),
        role_options=options,
        cleaning_options=[dict(index=i, wipe_variant=item["wipe_variant"],
            wipe_force=item["wipe_force"], wipe_duration=item["wipe_duration"])
            for i, item in enumerate(pool)],
        order_options=[dict(index=i, order=item["order"]) for i, item in enumerate(pool)],
        output_contract=dict(
            schema_file="adjacent prompt .schema.json generated for the scene roles",
            donor_indices=f"integer in [0,{len(pool)-1}]",
            exact_role_names=list(roles),
            rule="Compose complete plans from listed pre-execution options only. "
                 "Choose diverse valid skill/port solutions; do not infer outcomes or use rollout labels."))
    if graph_schema:
        registry = catalog_graph()
        request["baseline_donor_index"] = next(
            (i for i,item in enumerate(pool) if item["name"] == "grounded_000"), 0)
        request["generic_atomic_skill_graph"] = dict(
            nodes=[{key:node[key] for key in
                    ("name", "kind", "inputs", "outputs", "description")}
                   for node in registry["nodes"]],
            edges=[{key:edge[key] for key in ("source", "target", "supplies")}
                   for edge in registry["edges"]],
            edge_rule="conditional capability/dependency alternatives; complete bound plans are validated")
    else:
        request["generic_atomic_skills"] = PUBLIC_SKILLS
    return request


def response_schema(roles):
    """Strict model output contract, generated from arbitrary scene role names."""
    donor = {role: {"type": "integer"} for role in roles}
    return dict(type="object", additionalProperties=False,
        required=["observation_sha256", "base_pool_sha256", "plans"],
        properties=dict(observation_sha256=dict(type="string"),
            base_pool_sha256=dict(type="string"),
            plans=dict(type="array", items=dict(type="object", additionalProperties=False,
                required=["name", "order_donor", "cleaning_donor", "role_donors"],
                properties=dict(name=dict(type="string"), order_donor=dict(type="integer"),
                    cleaning_donor=dict(type="integer"),
                    role_donors=dict(type="object", additionalProperties=False,
                        required=list(roles), properties=donor))))))


def compose(base_request, model_response, candidate_count):
    base = base_request["pool"]
    observation_sha = base_request["initial_observation"]["sha256"]
    if set(model_response) != {"observation_sha256", "base_pool_sha256", "plans"}:
        raise ValueError("LLM response has unexpected fields")
    if model_response["observation_sha256"] != observation_sha:
        raise ValueError("LLM response belongs to a different observation")
    if model_response["base_pool_sha256"] != digest(base):
        raise ValueError("LLM response belongs to a different donor pool")
    rows = model_response["plans"]
    if not isinstance(rows, list) or len(rows) != candidate_count:
        raise ValueError("LLM candidate count differs from frozen request")
    roles = set(base[0]["choices"])
    result = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {
                "name", "order_donor", "cleaning_donor", "role_donors"}:
            raise ValueError("malformed complete plan composition")
        if row["name"] != f"llm_{i:03d}":
            raise ValueError("candidate names must be consecutive llm_NNN identifiers")
        if not isinstance(row["role_donors"], dict) or set(row["role_donors"]) != roles:
            raise ValueError("plan must bind every manipulated role exactly once")
        indices = [row["order_donor"], row["cleaning_donor"], *row["role_donors"].values()]
        if any(type(index) is not int or not 0 <= index < len(base) for index in indices):
            raise ValueError("LLM selected a donor outside the grounded library")
        candidate = deepcopy(base[row["order_donor"]])
        candidate["name"] = row["name"]
        candidate["source"] = "llm_composed_grounded_atomic_ports_v20"
        candidate["rationale"] = "model-selected legal order and independent pre-execution role/cleaning ports"
        for role, donor_index in row["role_donors"].items():
            donor = base[donor_index]
            candidate["choices"][role] = deepcopy(donor["choices"][role])
            candidate["necessary_geometry"][role] = deepcopy(donor["necessary_geometry"][role])
        clean = base[row["cleaning_donor"]]
        for key in ("wipe_variant", "wipe_force", "wipe_duration"):
            candidate[key] = clean[key]
        result.append(candidate)
    signatures = [digest({k:v for k,v in item.items()
                          if k not in ("name", "source", "rationale")}) for item in result]
    if len(set(signatures)) != len(result):
        raise ValueError("LLM repeated an executable complete plan")
    return result


def compile_pool(base_request, model_response, *, candidate_count, out, model,
                 llm_seconds, prompt_path):
    source = require_frozen_source()
    if source["sha256"] != base_request["runtime_sha256"]:
        raise ValueError("LLM donor pool was generated under a different physical runtime")
    seed = int(base_request["seed"])
    root = Path(out) / f"seed_{seed}" / "collect"
    if (root / "request.json").exists():
        raise ValueError("frozen LLM candidate request already exists")
    started = time.perf_counter()
    prompt = read(prompt_path)
    baseline = prompt.get("baseline_donor_index")
    if baseline is not None:
        first = model_response["plans"][0]
        if (first["order_donor"] != baseline or first["cleaning_donor"] != baseline
                or any(index != baseline for index in first["role_donors"].values())):
            raise ValueError("LLM omitted the nominal solver baseline from its first plan")
    pool = compose(base_request, model_response, candidate_count)
    _, session, _, _ = make_scene(seed, root / "decision",
        domain=base_request["domain"], level=base_request["level"])
    if session.decision_observation["sha256"] != base_request["initial_observation"]["sha256"]:
        raise ValueError("fresh scene observation differs from frozen LLM request")
    hashes = {}
    for candidate in pool:
        graph = normalized_graph(session, candidate)
        encode_graph(graph)
        hashes[candidate["name"]] = digest(graph)
    save(root / "prompt_input.json", prompt)
    save(root / "llm_response.json", model_response)
    save(root / "request.json", dict(seed=seed, level=base_request["level"],
        domain=base_request["domain"], pool=pool,
        source=dict(construction="llm_selected_grounded_atomic_role_ports_before_twin",
            model=model, llm_call=True, llm_seconds=float(llm_seconds),
            prompt_sha256=digest(read(prompt_path)), response_sha256=digest(model_response),
            base_request_sha256=digest(base_request)),
        graph_sha256=hashes, geometry_version=session.task_version,
        initial_observation=session.decision_observation,
        runtime_sha256=source["sha256"],
        generation_seconds=float(llm_seconds)+time.perf_counter()-started))
    save(root / "runtime_sources.json", source)
    return dict(seed=seed, candidates=len(pool), runtime_sha256=source["sha256"],
                graph_hashes_frozen=len(hashes), llm_seconds=float(llm_seconds))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-request", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--emit-prompt", action="store_true")
    parser.add_argument("--response", type=Path)
    parser.add_argument("--prompt", type=Path)
    parser.add_argument("--model", default="codex-cli")
    parser.add_argument("--llm-seconds", type=float, default=0.)
    args = parser.parse_args()
    base = read(args.base_request)
    if args.emit_prompt == bool(args.response):
        parser.error("choose exactly one of --emit-prompt or --response")
    if args.emit_prompt:
        request = prompt_input(base, args.n)
        save(args.out, request)
        save(args.out.with_suffix(".schema.json"),
             response_schema(request["output_contract"]["exact_role_names"]))
        print(json.dumps(dict(prompt=str(args.out), bytes=args.out.stat().st_size,
            output_schema=str(args.out.with_suffix(".schema.json")),
            observation_sha256=request["observation_sha256"],
            base_pool_sha256=request["base_pool_sha256"])))
    else:
        if args.prompt is None:
            parser.error("--response requires --prompt")
        frozen_prompt = read(args.prompt)
        schema = frozen_prompt.get("schema")
        if schema not in ("twingraph.llm_atomic_composition_request.v20.r1",
                          "twingraph.llm_atomic_composition_request.v20.r2"):
            raise ValueError("unsupported frozen LLM prompt schema")
        if frozen_prompt != prompt_input(base, args.n, graph_schema=schema.endswith("r2")):
            raise ValueError("LLM prompt differs from the frozen donor request")
        print(json.dumps(compile_pool(base, read(args.response), candidate_count=args.n,
            out=args.out, model=args.model, llm_seconds=args.llm_seconds,
            prompt_path=args.prompt)))


if __name__ == "__main__":
    main()
