"""Read-only, predeclared N=12/24/48 coverage diagnostic for one V12 generator.

This is an external report analyzer, not a new experiment or parameter search.
It imports only the frozen pure proposer and provenance utility. It never
constructs a scene, executes a plan, trains a model, or imputes a missing label.
"""
import argparse
from fractions import Fraction
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys


BUDGETS = (12, 24, 48)
K = 4
SCHEMA = "twingraph.same_generator_pool_coverage.v12"
FROZEN_RUNTIME_SHA256 = "c6b79761f3555a682f402f4f408b0f94d35b4c50b4fad65f5c2251ac1f8824da"


def digest(value, *, compact=False):
    options = dict(sort_keys=True, allow_nan=False)
    if compact:
        options["separators"] = (",", ":")
    return hashlib.sha256(json.dumps(value, **options).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def is_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def canonical(value):
    """Normalize JSON tuples/lists without tolerating changed floating values."""
    return json.loads(json.dumps(value, allow_nan=False))


def random_top_k(n, feasible, k=K):
    """Exact probability of at least one success, uniform without replacement."""
    if any(type(v) is not int for v in (n, feasible, k)) or not 0 <= feasible <= n or not 1 <= k <= n:
        raise ValueError("invalid finite-pool random selection parameters")
    total = math.comb(n, k)
    misses = math.comb(n-feasible, k) if n-feasible >= k else 0
    hit = Fraction(total-misses, total)
    return dict(hit_probability=float(hit), hit_fraction=f"{hit.numerator}/{hit.denominator}",
                hit_subsets=total-misses, total_subsets=total,
                expected_feasible_selected=k*feasible/n)


class Snapshot:
    def __init__(self):
        self.files = {}

    def read(self, path):
        path = Path(path).resolve()
        raw = path.read_bytes()
        fingerprint = hashlib.sha256(raw).hexdigest()
        if str(path) in self.files and self.files[str(path)] != fingerprint:
            raise ValueError("input changed while being read")
        self.files[str(path)] = fingerprint
        return json.loads(raw)

    def verify(self):
        if any(sha(path) != expected for path, expected in self.files.items()):
            raise ValueError("input changed during analysis; wait for immutable completed evidence")


def _load_generator(runtime):
    runtime = Path(runtime).resolve()
    sys.dont_write_bytecode = True  # Imports must not add cache files to frozen source.
    sys.path.insert(0, str(runtime))
    planner = importlib.import_module("simbench.value.planner_v12")
    provenance = importlib.import_module("simbench.value.provenance_v12")
    if Path(planner.__file__).resolve() != runtime/"simbench/value/planner_v12.py" or provenance.ROOT.resolve() != runtime:
        raise ValueError("imported generator is not from the explicitly bound frozen runtime; use a fresh process")
    return planner.propose, provenance.fingerprint, provenance.fingerprint()


def _verify_pool(pool, n):
    if not isinstance(pool, list) or len(pool) != n:
        raise ValueError(f"expected exactly {n} candidates")
    names = [p.get("name") for p in pool]
    expected = {f"grounded_{i:03d}" for i in range(n)}
    if len(set(names)) != n or set(names) != expected:
        raise ValueError("candidate names do not identify the declared generation-order subset")
    return {p["name"]: p for p in pool}


def _verify_export(snapshot, root, directory, seed, runtime, names):
    """Validate exported bytes, never follow an archive's original source path."""
    top = snapshot.read(root/"export_manifest.json")
    entry = snapshot.read(directory/"export_manifest.json")
    if (top.get("schema") != "twingraph.actual_online_all_twin_matrix_export.v12"
            or top.get("complete") is not True or top.get("expected_pool_size") != 48
            or top.get("source_runtime_sha256") != runtime
            or seed not in top.get("expected_seeds", ())
            or top.get("training_or_simulation_executed") is not False
            or top.get("original_outcomes_and_times_modified") is not False):
        raise ValueError("incomplete or unbound all_twin export")
    matches = [e for e in top.get("layouts", ()) if e.get("seed") == seed]
    if matches != [entry] or (entry.get("complete") is not True or entry.get("pool_size") != 48
            or entry.get("source_runtime_sha256") != runtime
            or entry.get("source_method") != "all_twin" or entry.get("source_domain") != "online"):
        raise ValueError("export layout manifest disagrees with its parent")
    required = {directory/name for name in ("request.json", "runtime_sources.json", "source_online_summary.json")}
    required |= {directory/"candidates"/name/file for name in names for file in ("result.json", "input_graph.json")}
    exported = {}
    for row in entry.get("files", ()):
        target = (root/row["exported"]).resolve()
        if not target.is_relative_to(directory) or target in exported:
            raise ValueError("invalid or duplicate export path")
        snapshot.read(target)
        if snapshot.files[str(target)] != row.get("sha256"):
            raise ValueError("exported label bytes differ from audited manifest")
        exported[target] = row["sha256"]
    if set(exported) != required:
        raise ValueError("export does not bind every requested graph and outcome")
    if (entry.get("source_request_sha256") != exported[directory/"request.json"]
            or entry.get("source_summary_sha256") != exported[directory/"source_online_summary.json"]):
        raise ValueError("export request or summary hash mismatch")
    summary = snapshot.read(directory/"source_online_summary.json")
    if (summary.get("valid") is not True or summary.get("seed") != seed
            or summary.get("method") != "all_twin" or summary.get("runtime_sha256") != runtime
            or summary.get("pool_size") != 48):
        raise ValueError("export lacks a complete valid all_twin summary")
    trials = summary.get("trials", [])
    if len(trials) != 48 or {r.get("name") for r in trials} != set(names):
        raise ValueError("all_twin summary is missing candidate labels")
    return {r["name"]: r for r in trials}


def analyze(root, seeds, runtime):
    """Only explicitly named layout directories are opened, never other splits."""
    root = Path(root).resolve()
    if not seeds or any(type(s) is not int for s in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError("explicit, unique integer layout seeds required")
    proposer, fingerprint, frozen = _load_generator(runtime)
    if frozen["sha256"] != FROZEN_RUNTIME_SHA256:
        raise ValueError("runtime differs from this diagnostic's predeclared frozen r6")
    from simbench.value.skill_graph import interface_hash
    expected_interface = interface_hash()
    snapshot = Snapshot()
    rows, layout_bindings = [], []
    for seed in seeds:
        directory = root/f"seed_{seed}"/"collect"
        request = snapshot.read(directory/"request.json")
        provenance = snapshot.read(directory/"runtime_sources.json")
        if provenance != frozen or digest(provenance.get("files"), compact=True) != provenance.get("sha256"):
            raise ValueError("matrix source/CAD/policy hashes differ from the bound frozen generator runtime")
        if request.get("runtime_sha256") != frozen["sha256"] or request.get("seed") != seed:
            raise ValueError("matrix seed or runtime binding mismatch")
        if request.get("method") not in (None, "all_twin"):
            raise ValueError("Top-K or early-stop outcomes cannot substitute for a complete matrix")
        domain = request.get("domain")
        if domain not in ("train", "validation", "online"):
            raise ValueError("undeclared physical label domain")
        pool = request.get("pool")
        archived = _verify_pool(pool, 48)
        candidates = directory/"candidates"
        if {p.name for p in candidates.iterdir() if p.is_dir()} != set(archived):
            raise ValueError("incomplete matrix: candidate directories differ from requested pool")
        observation = request.get("initial_observation", {})
        observed_sha = observation.get("sha256")
        if not is_hash(observed_sha) or digest({k:v for k,v in observation.items() if k != "sha256"}) != observed_sha:
            raise ValueError("saved initial observation hash mismatch")
        source = request.get("source", {})
        if source.get("observation_sha256") != observed_sha or source.get("requested") != 48:
            raise ValueError("proposal source is not bound to the initial observation and N48")
        exported_trials = None
        if request.get("method") == "all_twin":
            exported_trials = _verify_export(snapshot, root, directory, seed, frozen["sha256"], archived)
        elif (directory/"export_manifest.json").exists():
            raise ValueError("native matrix unexpectedly claims an online export")
        cad, interfaces, labels, label_files = None, set(), {}, []
        for name, proposal in archived.items():
            result_path = candidates/name/"result.json"
            graph_path = result_path.with_name("input_graph.json")
            result, graph = snapshot.read(result_path), snapshot.read(graph_path)
            if result.get("valid") is not True or type(result.get("success")) is not bool:
                raise ValueError("missing or invalid physical label; never impute failure")
            if result.get("seed") != seed or result.get("runtime_sha256") != frozen["sha256"] or result.get("domain") != domain:
                raise ValueError("candidate seed/runtime/domain differs from matrix")
            if result.get("proposal") != proposal or graph.get("proposal") != proposal:
                raise ValueError("executed or graphed proposal differs from request")
            if result.get("input_graph_sha256") != digest(graph):
                raise ValueError("result is not bound to its archived input graph")
            if (result.get("geometry_version") != request.get("geometry_version")
                    or graph.get("task_geometry_version") != request.get("geometry_version")
                    or not isinstance(request.get("geometry_version"), str)):
                raise ValueError("geometry version mismatch")
            if result.get("initial_observation", {}).get("sha256") != observed_sha:
                raise ValueError("candidate was executed from a different initial observation")
            execution_obs = result["initial_observation"]
            if digest({k:v for k,v in execution_obs.items() if k != "sha256"}) != observed_sha:
                raise ValueError("candidate observation contents do not match their hash")
            assembly = graph.get("assembly", {})
            if assembly.get("observation", {}).get("perception", {}).get("observation_sha256") != observed_sha:
                raise ValueError("graph uses a different initial observation")
            interface = assembly.get("interface_sha256")
            if not is_hash(interface) or interface != expected_interface:
                raise ValueError("graph interface hash differs from the frozen skill interface")
            interfaces.add(interface)
            if "planning_cad" not in graph or digest(graph["planning_cad"]) != source.get("cad_sha256"):
                raise ValueError("saved CAD missing or not bound to proposal source")
            if cad is None:
                cad = graph["planning_cad"]
            elif cad != graph["planning_cad"]:
                raise ValueError("CAD differs within the candidate pool")
            if exported_trials is not None:
                trial = exported_trials[name]
                if (type(trial.get("success")) is not bool or trial["success"] != result["success"]
                        or trial.get("index") != pool.index(proposal)):
                    raise ValueError("exported summary label or original candidate index mismatch")
            labels[name] = result["success"]
            label_files.append(dict(candidate=name, result_sha256=snapshot.files[str(result_path)],
                                    graph_sha256=snapshot.files[str(graph_path)]))
        if len(interfaces) != 1:
            raise ValueError("mixed skill interfaces within one layout")
        regeneration = []
        for n in BUDGETS:
            regenerated, regenerated_source = proposer(observation, cad=cad, n=n, seed=seed, priors=source.get("priors"))
            regenerated, regenerated_source = canonical(regenerated), canonical(regenerated_source)
            generated = _verify_pool(regenerated, n)
            if any(generated[name] != archived[name] for name in generated):
                raise ValueError(f"non-nested or unreproducible same-name proposal at N={n}; do not reuse N48 labels")
            if n == 48 and (regenerated != pool or regenerated_source != source):
                raise ValueError("N48 generation order or full provenance cannot be reproduced from saved observation/CAD")
            ordered_names = [f"grounded_{i:03d}" for i in range(n)]
            count = sum(labels[name] for name in ordered_names)
            rows.append(dict(seed=seed, n=n, k=K, feasible_candidates=count, has_feasible_candidate=bool(count),
                subset_rule="generation index 0..N-1, never shuffled list prefix",
                candidate_names=ordered_names, random_top4=random_top_k(n, count)))
            regeneration.append(dict(n=n, exact_same_name_proposals=True,
                shuffled_generation_order=[p["name"] for p in regenerated],
                canonical_subset_sha256=digest([generated[name] for name in ordered_names])))
        layout_bindings.append(dict(seed=seed, runtime_sha256=frozen["sha256"], initial_observation_sha256=observed_sha,
            cad_sha256=source["cad_sha256"], interface_sha256=next(iter(interfaces)),
            label_origin="actual online all_twin byte-preserved export" if exported_trials is not None else "native complete candidate matrix",
            request_sha256=snapshot.files[str(directory/"request.json")], label_files=label_files,
            regeneration=regeneration))
    snapshot.verify()
    if fingerprint() != frozen:
        raise ValueError("frozen generator source/CAD/policy changed during analysis")
    aggregate = []
    for n in BUDGETS:
        subset = [row for row in rows if row["n"] == n]
        mean_probability = sum((Fraction(r["random_top4"]["hit_fraction"]) for r in subset), Fraction())/len(subset)
        aggregate.append(dict(n=n, k=K, layouts=len(subset), layouts_with_solution=sum(r["has_feasible_candidate"] for r in subset),
            mean_feasible_candidates=sum(r["feasible_candidates"] for r in subset)/len(subset),
            mean_exact_random_top4_hit_probability=float(mean_probability),
            mean_random_top4_hit_fraction=f"{mean_probability.numerator}/{mean_probability.denominator}"))
    return dict(schema=SCHEMA, complete=True, expected_seeds=list(seeds), budgets=list(BUDGETS), k=K,
        purpose="same r6 generator budget coverage only; main N48/K4 and checkpoint selection unchanged",
        no_new_physical_experiments=True, no_model_training_or_selection=True, no_latency_claim=True,
        interpretation="Feasible means the archived full-task acceptance label. Analytic uniform random subset coverage is not independent deployment success.",
        runtime_sha256=frozen["sha256"], analyzer_sha256=sha(__file__), rows=rows, aggregate=aggregate,
        layout_bindings=layout_bindings, input_files=snapshot.files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Complete native matrix or audited all_twin export root")
    parser.add_argument("--seeds", type=int, nargs="+", required=True, help="Only these explicit layout directories are opened")
    parser.add_argument("--runtime", type=Path, required=True, help="Read-only frozen source matching all label runtime hashes")
    parser.add_argument("--out", type=Path, required=True, help="New JSON report path outside source/evidence trees")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() or any(out.is_relative_to(tree.resolve()) for tree in (args.root, args.runtime)):
        parser.error("output must be new and separate from both input evidence and frozen runtime")
    report = analyze(args.root, args.seeds, args.runtime)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(dict(complete=True, layouts=len(args.seeds), aggregate=report["aggregate"], report=str(out))))


if __name__ == "__main__":
    main()
