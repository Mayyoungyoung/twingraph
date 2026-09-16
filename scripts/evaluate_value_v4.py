"""Frozen graph-value evaluation, input-only latency, and independent deployment."""
import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
try:
    import torch
except ImportError:  # Input-manifest audits and fake-weight tests need no GPU runtime.
    torch = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simbench.value.collect import dump
from simbench.value.plan import PlanIR, digest, initial_artifacts
from simbench.value.skill_graph import compile_graph, validate_graph, interface_hash
from simbench.value.schema_collect import create_scene, source_manifest
from simbench.value.physical import PhysicalRunner, perturbation
from simbench.value import stage_assembly as stage
from simbench.assembly.library import SkillFailure


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def implementation():
    sources = source_manifest()
    root = Path(__file__).resolve().parents[1]
    for name in ("evaluate_value_v4.py", "analyze_value_v4.py"):
        sources[f"scripts/{name}"] = sha(root/"scripts"/name)
    return dict(sha256=digest(sources), files=sources)


def sync(device):
    if str(device).startswith("cuda"):
        if torch is None:
            raise RuntimeError("CUDA measurement requires Torch")
        torch.cuda.synchronize(device)


def make_scorer(checkpoint, device):
    from simbench.value.schema_rank import SchemaScorer
    return SchemaScorer(checkpoint, device)


def finite_or(value, default):
    return float(value) if value is not None and np.isfinite(value) else default


def freeze(models, output, test_seeds, checkpoints=(2,), n=8, k=4,
           deployment_seeds=None, alternate_checkpoints=False):
    if Path(output).exists():
        raise ValueError("refuse to overwrite an existing freeze")
    if not test_seeds or len(set(test_seeds)) != len(test_seeds) or n < k or k < 1:
        raise ValueError("invalid frozen test seeds or candidate budget")
    if not checkpoints or any(cp not in (2, 3) for cp in checkpoints):
        raise ValueError("only robot-executed checkpoints 2 and 3 are supported")
    if len(set(checkpoints)) != len(checkpoints):
        raise ValueError("duplicate checkpoints")
    tasks = [dict(seed=seed, checkpoint=checkpoints[seed % len(checkpoints)]) for seed in sorted(test_seeds)] if alternate_checkpoints else [
        dict(seed=seed, checkpoint=cp) for seed in sorted(test_seeds) for cp in sorted(checkpoints)]
    deployment_seeds = list(deployment_seeds or sorted(test_seeds)[:4])
    if len(deployment_seeds) != 4 or len(set(deployment_seeds)) != 4 or not set(deployment_seeds) <= set(test_seeds):
        raise ValueError("freeze exactly four distinct deployment configurations from test")
    deployment_tasks = [next(t for t in tasks if t["seed"] == seed) for seed in deployment_seeds]
    if torch is None:
        raise RuntimeError("freezing real model weights requires Torch")
    rows = []
    for path in sorted(Path(models).glob("*/best.pt")):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved.get("schema") != "twingraph.value.schema.v1" or saved["interface_sha256"] != interface_hash():
            raise ValueError("incompatible model in freeze directory")
        raw = {k:v for k,v in saved["validation"].items() if k != "rows"}
        calibrated = {k:v for k,v in saved.get("validation_calibrated", saved["validation"]).items() if k != "rows"}
        rows.append(dict(name=path.parent.name, path=str(path.resolve()), sha256=sha(path),
                         kind=saved["kind"], seed=saved["seed"], epoch=saved["epoch"],
                         validation=raw, validation_calibrated=calibrated,
                         calibration=saved.get("calibration", dict(scale=1., bias=0.)),
                         source_sha256=saved["source_sha256"], split_sha256=saved["split_sha256"]))
    if not rows:
        raise ValueError("no schema checkpoints found")
    if len({r["split_sha256"] for r in rows}) != 1:
        raise ValueError("models must share the same train/validation split")
    def key(row):
        v = row["validation"]
        return (finite_or(v.get("hit4"), -1.), finite_or(v.get("quality4"), -1.),
                finite_or(v.get("hit1"), -1.), -finite_or(row["validation_calibrated"].get("brier"), float("inf")))
    selected = max(rows, key=key)
    result = dict(schema="twingraph.schema_experiment.freeze.v1", selected=selected["name"],
        selection="validation raw-score Hit4, quality4, Hit1, then negative calibrated Brier; missing Hit sorts below observed Hit",
        models=rows, interface_sha256=interface_hash(), created_unix=time.time(),
        test=dict(family=stage.FAMILY, seeds=sorted(test_seeds), checkpoints=list(checkpoints), tasks=tasks,
                  assignment="checkpoint indexed by seed modulo checkpoint count" if alternate_checkpoints else "Cartesian product",
                  alternate_checkpoints=alternate_checkpoints, n=n, repeats=2),
        deployment=dict(seeds=deployment_seeds, tasks=deployment_tasks, n=n, k=k,
                        online_repeats=2, deployment_repeats=2, accept_rate=.5,
                        selection="success fraction descending, original rank ascending"),
        implementation=implementation())
    dump(output, result)
    print(json.dumps(dict(selected=result["selected"], freeze=str(output), models=len(rows))))
    return result


def verify_freeze(frozen):
    if frozen.get("schema") != "twingraph.schema_experiment.freeze.v1":
        raise ValueError("unsupported freeze schema")
    if frozen["interface_sha256"] != interface_hash():
        raise ValueError("execution interface changed after freeze")
    for row in frozen["models"]:
        if sha(row["path"]) != row["sha256"]:
            raise ValueError(f"frozen weights changed: {row['name']}")


def frozen_tasks(frozen):
    design = frozen["test"]
    return design.get("tasks") or [dict(seed=seed,checkpoint=cp) for seed in design["seeds"] for cp in design["checkpoints"]]


def admissible_setup_error(error, require_trace=False):
    """Only existing robot setup skill/IK failures are dataset attrition."""
    text = str(error)
    if any(marker in text for marker in ("unexpected keyword", "unknown skill", "invalid parameter choice", "missing a required argument")):
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    final = lines[-1] if lines else ""
    solver_exhausted = "no materialized candidate; expand solver budget or verify unknowns"
    physical = (isinstance(error, SkillFailure) or isinstance(error, ValueError) and text.startswith("IK unreachable:")
                or isinstance(error, ValueError) and text == solver_exhausted
                or "SkillFailure:" in final or final.startswith("ValueError: IK unreachable:")
                or final == "ValueError: " + solver_exhausted)
    context = not require_trace or any(marker in text for marker in ("in create_scene", "in prepare_checkpoint"))
    return bool(physical and context)


def test_inputs(roots, frozen):
    """Only input JSONs are opened: usable for timing before opening outcomes."""
    items = []
    for root in roots:
        for directory in sorted(Path(root).glob("group_*")):
            path = directory/"inputs.json"
            failure_path = directory/"failure.json"
            if not path.exists():
                if not failure_path.exists():
                    continue
                failed = json.loads(failure_path.read_text())
                request = failed.get("request",{})
                if request.get("split") != "test":
                    continue
                if not admissible_setup_error(failed.get("error",""),require_trace=True):
                    raise ValueError("test failure is not an admissible physical checkpoint-setup failure")
                items.append(dict(path=failure_path,inputs=None,seed=request["seed"],checkpoint=request["checkpoint"],
                                  setup_failure=failed,disposition="checkpoint_setup_failed"))
                continue
            inp = json.loads(path.read_text())
            if inp["declared_split"] != "test":
                continue
            if failure_path.exists():
                raise ValueError("test group contains both inputs and a failure record; resolve execution provenance")
            if not (directory/"complete.json").exists():
                raise ValueError("test input group is incomplete")
            task = inp["task"]
            if task["family"] != frozen["test"]["family"]:
                raise ValueError("unexpected test family")
            items.append(dict(path=path, inputs=inp, seed=task["seed"], checkpoint=inp["checkpoint"],
                              disposition="checkpoint_reached"))
    expected = {(t["seed"],t["checkpoint"]) for t in frozen_tasks(frozen)}
    actual = [(i["seed"], i["checkpoint"]) for i in items]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("test configurations do not exactly match the frozen design")
    return sorted(items, key=lambda i:(i["seed"], i["checkpoint"]))


def shortest_initial_scores(plans):
    lengths = []
    for plan in plans:
        artifacts = initial_artifacts(plan)
        if not artifacts:
            raise ValueError("shortest_initial baseline requires a materialized initial joint path")
        # Only the actually consumed first initial trajectory is compared.
        path = None
        for call in plan.calls:
            from simbench.assembly.interfaces import resolve
            name, params = resolve(call.skill, {k:a.value for k,a in call.arguments.items()})
            if name == "execute_joint_path" and params.get("artifact", "transfer") in artifacts:
                path = artifacts[params.get("artifact", "transfer")]
                break
        if path is None:
            raise ValueError("initial trajectory has no execution consumer")
        points = np.vstack([path["start_q"], path["joints"]])
        lengths.append(float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()))
    return -np.asarray(lengths)


def baseline_scores(method, plans):
    if method in {"source_order","exhaustive"}:
        return -np.arange(len(plans), dtype=float)
    if method == "shortest_initial":
        return shortest_initial_scores(plans)
    raise ValueError("unknown baseline")


def build_graphs(inp):
    started = time.perf_counter()
    graphs = [compile_graph(inp["observation"], p) for p in inp["candidates"]]
    return graphs, time.perf_counter()-started


def rank_method(method, graphs, scorers, k):
    if method in scorers:
        return scorers[method].rank(graphs, min(k, len(graphs)))
    started = time.perf_counter()
    plans = [validate_graph(g) for g in graphs]
    check = time.perf_counter()-started
    t = time.perf_counter()
    scores = baseline_scores(method, plans)
    features = time.perf_counter()-t
    t = time.perf_counter()
    order = np.argsort(-scores, kind="stable")
    result = dict(scores=scores.tolist(), logits=None, order=[plans[i].id for i in order],
        top_k=[dict(candidate_id=plans[i].id, score=float(scores[i]), graph=graphs[i],
                    graph_sha256=digest(graphs[i]), plan=plans[i].to_dict()) for i in order[:k]],
        seconds=dict(integrity=check, encoding=features, inference=0.))
    result["seconds"]["export"] = time.perf_counter()-t
    result["seconds"]["total"] = time.perf_counter()-started
    return result


def evaluate(frozen, roots, out, device):
    from simbench.value.schema_learning import load_groups, sigmoid
    verify_freeze(frozen)
    items = test_inputs(roots, frozen)
    reached = [i for i in items if i["inputs"] is not None]
    groups = load_groups(roots, splits=("test",)) if reached else []
    expected = {i["inputs"]["group_id"] for i in reached}
    if {g["id"] for g in groups} != expected:
        raise ValueError("loaded label groups differ from frozen test inputs")
    if any(g["repeats"] != frozen["test"].get("repeats",2) for g in groups):
        raise ValueError("test repetitions differ from frozen design")
    scorers = {r["name"]:make_scorer(r["path"], device) for r in frozen["models"]}
    methods = {}
    source_records = []
    for group in groups:
        inp = group["inputs"]
        source_records.append(dict(group_id=group["id"], input_sha256=group["input_sha256"],
                                   collection_source_sha256=inp["source_sha256"], repeats=group["repeats"]))
        base = dict(group_id=group["id"], config_id=json.dumps(group["config"]), seed=group["seed"],
                    family=inp["task"]["family"], checkpoint=inp["checkpoint"],
                    candidate_ids=[p.id for p in group["plans"]], reference=group["y"].tolist(),
                    plan_lengths=[len(p.calls) for p in group["plans"]],
                    repeats=group["repeats"], input_sha256=group["input_sha256"])
        for method in (*scorers, "source_order", "shortest_initial"):
            ranked = rank_method(method, group["graphs"], scorers, 4)
            if method in scorers:
                z = ranked["logits"]
                methods.setdefault(method, []).append(dict(base, scores=ranked["scores"],
                    ranking_scores=z, probability_metrics=True, calibrated=True))
                methods.setdefault(method+"_raw", []).append(dict(base, scores=sigmoid(np.asarray(z)).tolist(),
                    ranking_scores=z, probability_metrics=True, calibrated=False))
            else:
                methods.setdefault(method, []).append(dict(base, scores=ranked["scores"], probability_metrics=False))
    attrition = dict(expected_configurations=len({i["seed"] for i in items}),expected_groups=len(items),
        reached_groups=len(reached),failed_setup_groups=len(items)-len(reached),
        disposition=[dict(seed=i["seed"],checkpoint=i["checkpoint"],status=i["disposition"],
                          failure=i.get("setup_failure")) for i in items],
        note="Candidate-ranking metrics condition on a physically reached checkpoint; setup failures count as zero in end-to-end feasible-hit denominators.")
    result = dict(schema="twingraph.value.predictions.v4", freeze_sha256=digest(frozen),
        selected=frozen["selected"], methods=methods, sources=source_records, implementation=implementation(),
        attrition=attrition,
        ranking_note="learned policies rank raw logits; positive affine calibration changes probabilities only",
        baseline_note="source_order uses the first K source candidates without learned screening; shortest_initial sums joint-space segment lengths in radians for the materialized initial trajectory only",
        label_note="finite paired physical repetitions, not known true success probabilities")
    from scripts.analyze_value_v4 import analyze, save_outputs
    analyzed = analyze(result) if groups else dict(schema="twingraph.value.ranking_audit.v1",methods={},
        status="no_physically_reached_test_checkpoint",attrition=attrition)
    end_to_end = {}
    for name in methods or [*scorers,*[m+"_raw" for m in scorers],"source_order","shortest_initial"]:
        rows = analyzed["methods"].get(name,{}).get("rows",[])
        by_task = {(r["seed"],next(g["inputs"]["checkpoint"] for g in groups if g["id"]==r["group_id"])):r for r in rows}
        end_to_end[name] = {}
        for k in (1,2,4):
            by_config = {}
            for item in items:
                value = by_task.get((item["seed"],item["checkpoint"]),{}).get(f"feasible{k}",False)
                by_config.setdefault(item["seed"],[]).append(float(value))
            end_to_end[name][f"feasible{k}"] = float(np.mean([np.mean(v) for v in by_config.values()]))
    result["end_to_end_feasible_hit_including_setup_failures"] = end_to_end
    analyzed["end_to_end_feasible_hit_including_setup_failures"] = end_to_end
    dump(out/"predictions.json", result)
    save_outputs(analyzed, out)
    print(json.dumps(dict(predictions=str(out/"predictions.json"), methods=list(methods), groups=len(groups))))
    return result


def benchmark(frozen, roots, out, device, repeats=5):
    verify_freeze(frozen)
    if repeats < 3:
        raise ValueError("at least three measured repetitions are required")
    manifest = test_inputs(roots, frozen)  # deliberately never load_groups/outcomes
    inputs = [item for item in manifest if item["inputs"] is not None]
    if not inputs:
        result = dict(freeze_sha256=digest(frozen),summary={},rows=[],status="no_reached_checkpoint_to_benchmark",
                      failed_setup_groups=len(manifest))
        dump(out/"module_timing.json",result)
        return result
    scorers = {r["name"]:make_scorer(r["path"], device) for r in frozen["models"]}
    methods = [*scorers, "source_order", "shortest_initial"]
    rows = []
    for item in inputs:
        inp = item["inputs"]
        warm_graphs, _ = build_graphs(inp)
        for method in methods:
            rank_method(method, warm_graphs, scorers, frozen["deployment"]["k"])
        for repeat in range(repeats):
            order = np.random.default_rng(item["seed"]+repeat).permutation(methods)
            for method in order:
                sync(device)
                graphs, construction = build_graphs(inp)
                ranked = rank_method(method, graphs, scorers, frozen["deployment"]["k"])
                sync(device)
                rows.append(dict(group_id=inp["group_id"], config_id=inp["split_group"], seed=item["seed"],
                    checkpoint=item["checkpoint"], method=str(method), repeat=repeat,
                    graph_construction=construction, **ranked["seconds"],
                    module_total=construction+ranked["seconds"]["total"]))
    summary = {}
    for method in methods:
        selected = [r for r in rows if r["method"] == method]
        summary[method] = {}
        for key in ("graph_construction", "integrity", "encoding", "inference", "export", "total", "module_total"):
            # First aggregate repeated timing within a checkpoint, then siblings.
            configs = {}
            for gid in sorted({r["group_id"] for r in selected}):
                siblings = [r for r in selected if r["group_id"] == gid]
                configs.setdefault(siblings[0]["config_id"], []).append(np.median([r[key] for r in siblings]))
            values = [np.mean(v) for v in configs.values()]
            summary[method][key] = dict(median_seconds=float(np.median(values)), p95_seconds=float(np.quantile(values,.95)))
    result = dict(freeze_sha256=digest(frozen), implementation=implementation(), summary=summary, rows=rows,
        expected_groups=len(manifest),reached_groups=len(inputs),failed_setup_groups=len(manifest)-len(inputs),
        weight_load_seconds={name:s.load_seconds for name,s in scorers.items()},
        note="Resident warm models; CUDA synchronized; every measurement recompiles and checks graphs, encodes inputs, infers and exports. Only inputs.json is read, no labels. Scene/checkpoint creation and candidate generation excluded. total is scorer time; module_total adds graph construction. Weight loading is not process cold start.")
    dump(out/"module_timing.json", result)
    return result


def _deployment_worker(request):
    for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
        os.environ[name]="1"
    if torch is not None:
        torch.set_num_threads(1)
    return deployment(request["frozen"],Path(request["out"]),request["device"],
                      workers=1,selected_tasks=[request["task"]],write_summary=False)


def render_terminal(session, directory, method, seed, checkpoint, execution):
    """Save the last actual deployment state without stepping or posing bodies."""
    if not execution:
        return dict(status="no_independent_execution_to_render",seconds=0.)
    started=time.perf_counter();renderer=None
    try:
        import mujoco
        import imageio.v3 as imageio
        renderer=mujoco.Renderer(session.ctx.model,height=600,width=960)
        renderer.update_scene(session.ctx.data,camera="task_view")
        path=directory/f"{method}_final.png"
        imageio.imwrite(path,renderer.render())
        last=execution[-1]
        metadata=dict(seed=seed,checkpoint=checkpoint,method=method,success=last["success"],
            candidate_id=last["candidate_id"],input_graph_sha256=last["input_graph_sha256"],
            trial=last["trial"],trial_sha256=last["trial_sha256"],image=path.name,
            observation="raw terminal frame of the last actual independent deployment repetition; no rerun, state assignment or image editing")
        dump(directory/f"{method}_final.json",metadata)
        return dict(status="saved",image=str(path),metadata=str(directory/f"{method}_final.json"),seconds=time.perf_counter()-started)
    except Exception as exc:
        diagnostic=dict(status="render_failed",error=repr(exc),seconds=time.perf_counter()-started,
                        note="Render failure does not change the physical trial label")
        dump(directory/f"{method}_render_diagnostic.json",diagnostic)
        return diagnostic
    finally:
        if renderer is not None:
            try:
                if hasattr(renderer,"close"):renderer.close()
                else:
                    renderer._mjr_context.free();renderer._gl_context.free()
            except Exception:
                pass


def deployment(frozen, out, device, workers=1, selected_tasks=None, write_summary=True):
    verify_freeze(frozen)
    overall_start = time.perf_counter()
    config = frozen["deployment"]
    all_tasks = config.get("tasks") or [dict(seed=seed,checkpoint=config["checkpoint"]) for seed in config["seeds"]]
    tasks = all_tasks if selected_tasks is None else selected_tasks
    if not tasks or any(t not in all_tasks for t in tasks) or len({(t["seed"],t["checkpoint"]) for t in tasks})!=len(tasks):
        raise ValueError("deployment subset must contain distinct frozen tasks")
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers > 1:
        requests = [dict(frozen=frozen,out=str(out),device=device,task=task) for task in tasks]
        previous_env = {name:os.environ.get(name) for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")}
        for name in previous_env:os.environ[name]="1"
        summaries = []
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=min(workers,len(tasks)),
                    mp_context=multiprocessing.get_context("spawn")) as pool:
                futures = [pool.submit(_deployment_worker,r) for r in requests]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result["freeze_sha256"] != digest(frozen):
                        raise ValueError("worker used a different freeze")
                    summaries.append(result)
                    print(json.dumps(dict(completed_deployment_configurations=len(summaries),total=len(tasks))),flush=True)
        finally:
            for name,value in previous_env.items():
                if value is None:os.environ.pop(name,None)
                else:os.environ[name]=value
        merged = dict(freeze_sha256=digest(frozen),implementation=implementation(),
            decisions=sorted([row for result in summaries for row in result["decisions"]],key=lambda r:(r["seed"],r["method"])),
            parallel_workers=min(workers,len(tasks)),overall_wall_seconds=time.perf_counter()-overall_start,
            note="Configurations run in independent processes, one Torch/OMP/BLAS thread each. Per-policy times are measured under this declared concurrency; overall_wall_seconds is actual parallel elapsed time, not a sum of worker durations. Exhaustive validates every candidate under its larger actual budget; no theoretical N/K speedup is reported.")
        if write_summary:dump(out/"deployment_summary.json",merged)
        return merged
    row = next(r for r in frozen["models"] if r["name"] == frozen["selected"])
    scorers = {row["name"]:make_scorer(row["path"], device)}
    methods = [row["name"], "shortest_initial", "source_order", "exhaustive"]
    source = implementation()
    results = []
    for task in tasks:
        seed, checkpoint = task["seed"], task["checkpoint"]
        directory = out/f"deployment_{seed}_cp{checkpoint}"
        directory.mkdir(parents=True, exist_ok=True)
        with (directory/"steps.log").open("w") as stream, contextlib.redirect_stdout(stream):
            started = time.perf_counter()
            try:
                failure_path = directory/"failure.json"
                if failure_path.exists():
                    previous = json.loads(failure_path.read_text())
                    if previous.get("kind")=="admissible_checkpoint_setup_failure":
                        if previous["freeze_sha256"]!=digest(frozen) or previous["implementation_sha256"]!=source["sha256"]:
                            raise ValueError("saved setup failure differs from this execution request")
                        results.extend(json.loads((directory/f"{m}_decision.json").read_text()) for m in methods)
                        continue
                try:
                    spec, session, _, targets = create_scene(seed, directory, checkpoint)
                except Exception as exc:
                    if not admissible_setup_error(exc):
                        raise
                    setup_seconds=time.perf_counter()-started
                    failure=dict(kind="admissible_checkpoint_setup_failure",seed=seed,checkpoint=checkpoint,
                        error=traceback.format_exc(),freeze_sha256=digest(frozen),implementation_sha256=source["sha256"])
                    for method in methods:
                        result=dict(seed=seed,method=method,checkpoint=checkpoint,config_id=f"setup_failure_seed_{seed}",
                            status="checkpoint_setup_failed",request=dict(freeze_sha256=digest(frozen)),failure=failure,
                            validation_calls=0,deployment_attempts=0,deployment_successes=0,
                            requested_deployment_attempts=config["deployment_repeats"],deployment_success_rate=0.,
                            k=config["n"] if method=="exhaustive" else config["k"],
                            online_budget=(config["n"] if method=="exhaustive" else config["k"])*config["online_repeats"],
                            timing=dict(shared_checkpoint_seconds=setup_seconds,policy_wall_seconds=0.),
                            physics_steps=dict(validation=0,deployment=0))
                        dump(directory/f"{method}_decision.json",result);results.append(result)
                    dump(failure_path,failure)
                    continue
                checkpoint_seconds = time.perf_counter()-started
                begin = time.perf_counter()
                plans, counts = stage.build_pool(session, targets, seed, n=config["n"])
                observation = stage.observed(session, targets)
                candidate_seconds = time.perf_counter()-begin
                inp = dict(observation=observation, candidates=[p.to_dict() for p in plans])
                graphs, graph_seconds = build_graphs(inp)
                runner = PhysicalRunner(session, timeout=180.)
                lookup = {p.id:g for p,g in zip(plans,graphs)}
                dump(directory/"inputs.json", dict(inp, source_sha256=source["sha256"], seed=seed,
                     checkpoint=checkpoint, snapshot_sha256=runner.initial, pool_counts=counts))
                # Kernel warm-up is separate from policy decision cost.
                rank_method(row["name"], graphs, scorers, config["k"])
                for method in methods:
                    path = directory/f"{method}_decision.json"
                    request = dict(freeze_sha256=digest(frozen), implementation_sha256=source["sha256"],
                                   seed=seed, checkpoint=checkpoint, method=method, config=config, input_sha256=digest(inp))
                    if path.exists():
                        previous = json.loads(path.read_text())
                        if previous["request_sha256"] != digest(request):
                            raise ValueError("deployment resume request mismatch")
                        results.append(previous)
                        continue
                    decision_start = time.perf_counter()
                    policy_k = len(plans) if method=="exhaustive" else config["k"]
                    ranking = rank_method(method, graphs, scorers, policy_k)
                    trials, candidates = [], []
                    for rank, top in enumerate(ranking["top_k"]):
                        candidate_trials = []
                        for repeat in range(config["online_repeats"]):
                            trial = runner.run(lookup[top["candidate_id"]], perturbation(seed,repeat,"online"),keep_trace=True)
                            if trial["timeout"]:
                                raise RuntimeError("censored online rollout: rerun under adequate wall-time budget")
                            candidate_trials.append(trial); trials.append(trial)
                        rate = float(np.mean([t["success"] for t in candidate_trials]))
                        candidates.append(dict(candidate_id=top["candidate_id"], rank=rank, success_rate=rate))
                    accepted = [c for c in candidates if c["success_rate"] >= config["accept_rate"]]
                    chosen = min(accepted,key=lambda c:(-c["success_rate"],c["rank"])) if accepted else None
                    decision_seconds = time.perf_counter()-decision_start
                    deploy_start = time.perf_counter()
                    execution = []
                    if chosen:
                        for repeat in range(config["deployment_repeats"]):
                            trial = runner.run(lookup[chosen["candidate_id"]],perturbation(seed,repeat,"deployment"),keep_trace=True)
                            if trial["timeout"]:
                                raise RuntimeError("censored deployment rollout: rerun under adequate wall-time budget")
                            execution.append(trial)
                    deployment_seconds = time.perf_counter()-deploy_start
                    terminal_render = render_terminal(session,directory,method,seed,checkpoint,execution)
                    result = dict(request=request,request_sha256=digest(request),seed=seed,method=method,
                        checkpoint=checkpoint,config_id=spec.config_id,ranking=ranking,
                        validated=trials,candidate_results=candidates,chosen=chosen,deployment=execution,
                        k=policy_k,online_budget=policy_k*config["online_repeats"],
                        validation_calls=len(trials),deployment_attempts=len(execution),
                        deployment_successes=sum(t["success"] for t in execution),
                        requested_deployment_attempts=config["deployment_repeats"],
                        terminal_render=terminal_render,
                        deployment_success_rate=sum(t["success"] for t in execution)/config["deployment_repeats"],
                        timing=dict(shared_checkpoint_seconds=checkpoint_seconds,shared_candidate_seconds=candidate_seconds,
                            shared_graph_construction_seconds=graph_seconds,decision_seconds=decision_seconds,
                            independent_deployment_seconds=deployment_seconds,
                            policy_wall_seconds=decision_seconds+deployment_seconds,
                            validation_rollout_seconds=sum(t["wall_seconds"] for t in trials),
                            validation_restore_seconds=sum(t["restore_seconds"] for t in trials)),
                        physics_steps=dict(validation=sum(t["physics_steps"] for t in trials),
                                           deployment=sum(t["physics_steps"] for t in execution)),
                        status="deployed" if chosen else "unresolved_no_accepted_candidate")
                    dump(path,result);results.append(result)
            except Exception as exc:
                dump(directory/"failure.json",dict(seed=seed,error=repr(exc),freeze_sha256=digest(frozen)))
                raise
    summary = dict(freeze_sha256=digest(frozen),implementation=source,decisions=[
        {k:r[k] for k in ("seed","method","checkpoint","config_id","status","validation_calls","k","online_budget",
                          "deployment_attempts","deployment_successes","requested_deployment_attempts",
                          "deployment_success_rate","timing","physics_steps")} for r in results],
        parallel_workers=1,overall_wall_seconds=time.perf_counter()-overall_start,
        note="Actual graph-bound simulation, no outcome-table lookup. Disturbances are paired across candidates/methods and independent across namespaces. source_order is unscreened first-K; exhaustive validates the entire pool under its larger actual budget. No theoretical N/K speedup. Shared checkpoint/candidate/graph preparation is measured once per configuration and reported separately. Physical setup failures remain zero-success requests; implementation errors abort.")
    if write_summary:dump(out/"deployment_summary.json",summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("freeze","evaluate","benchmark","deployment"))
    parser.add_argument("--freeze",required=True);parser.add_argument("--models")
    parser.add_argument("--data",nargs="+");parser.add_argument("--out",default="runs/v4/evidence")
    parser.add_argument("--device",default="cuda");parser.add_argument("--repeats",type=int,default=5)
    parser.add_argument("--workers",type=int,default=1,help="Independent configuration workers for deployment only")
    parser.add_argument("--test-seeds",nargs="+",type=int,default=list(range(51300,51308)))
    parser.add_argument("--checkpoints",nargs="+",type=int,default=[2])
    parser.add_argument("--alternate-checkpoints",action="store_true",
                        help="Freeze one checkpoint per seed, indexed by seed modulo len(checkpoints); 2 3 means even=2, odd=3")
    parser.add_argument("--deployment-seeds",nargs="+",type=int)
    parser.add_argument("--n",type=int,default=8);parser.add_argument("--k",type=int,default=4)
    args=parser.parse_args()
    if torch is None:parser.error("model evaluation commands require Torch")
    torch.set_num_threads(2)
    if args.mode=="freeze":
        if not args.models:parser.error("freeze requires --models")
        freeze(args.models,args.freeze,args.test_seeds,args.checkpoints,args.n,args.k,args.deployment_seeds,args.alternate_checkpoints)
        return
    frozen=json.loads(Path(args.freeze).read_text());verify_freeze(frozen)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if args.mode in {"evaluate","benchmark"} and not args.data:
        parser.error("evaluate/benchmark requires --data")
    if args.mode=="evaluate":evaluate(frozen,args.data,out,args.device)
    elif args.mode=="benchmark":benchmark(frozen,args.data,out,args.device,args.repeats)
    else:deployment(frozen,out,args.device,workers=args.workers)


if __name__=="__main__":
    main()
