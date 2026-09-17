"""Measured task-to-plan-to-value-to-twin-to-independent-target experiment.

No outcome table is accepted by this API. Every validation and target result
comes from an actual rollout. A target owns a new model, data and Session; no
snapshot from a twin is passed to it. Target disturbances are sampled only
after selection and never enter the scorer's nominal observation.
"""
from dataclasses import asdict, dataclass, is_dataclass
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .plan import PlanIR, digest, plain
from .skill_graph import compile_graph, validate_graph, interface_hash
from .physical import PhysicalRunner
from simbench.assembly.candidates import fingerprint


@dataclass(frozen=True)
class SystemConfig:
    seed: int
    n: int = 12
    k: int = 4
    validation_repeats: int = 2
    target_repeats: int = 1
    accept_rate: float = .5
    timeout: float = 240.
    friction_span: float = .03
    gain_span: float = .005
    render: bool = True

    def validate(self):
        if not 1 <= self.k <= self.n or min(self.validation_repeats, self.target_repeats) < 1:
            raise ValueError("invalid candidate or rollout budget")
        if not 0 < self.accept_rate <= 1 or self.timeout <= 0:
            raise ValueError("invalid acceptance rate or timeout")
        if not 0 <= self.friction_span < 1 or not 0 <= self.gain_span < 1:
            raise ValueError("invalid physical disturbance range")
        return self


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(plain(value), indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def trial_for(config, repeat, role):
    """Same declared uncertainty envelope, independent role-specific draws."""
    namespaces = {"twin": 5107, "target": 7901}
    if role not in namespaces:
        raise ValueError("unsupported physical role")
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, repeat, namespaces[role]]))
    return dict(domain="v5_" + role, namespace=namespaces[role], repeat=int(repeat),
                friction_scale=float(rng.uniform(1-config.friction_span, 1+config.friction_span)),
                actuator_gain_scale=float(rng.uniform(1-config.gain_span, 1+config.gain_span)))


def semantic_payload(plan):
    """Exact selected task/control program, independent of initial-path rebinding.

    Call IDs and deferred references are normalized by position. All argument
    values, units, frames, skill order, roles and boundary remain protected.
    Only the initial externally materialized free trajectory and start-state
    binding are outside this semantic payload; both are separately recorded.
    """
    plan = plan if isinstance(plan, PlanIR) else PlanIR.from_dict(plan)
    indices = {c.id: i for i, c in enumerate(plan.calls)}
    calls = []
    for call in plan.calls:
        args = {}
        for key, argument in call.arguments.items():
            item = plain(asdict(argument))
            if item["source_call"] is not None:
                item["source_call"] = indices[item["source_call"]]
            args[key] = item
        calls.append(dict(skill=call.skill, arguments=args, roles=call.roles, kind=call.kind))
    return dict(protocol=plan.protocol, boundary=plan.boundary, calls=calls,
                order=plan.prefix.get("order"), choices=plan.prefix.get("choices"),
                task_scope=plan.prefix.get("task_scope"),
                semantic_program_id=plan.prefix.get("semantic_program_id"),
                initial_route_index=plan.prefix.get("initial_route_index"))


def assert_independent(twin, target):
    """Fail closed on shared simulator objects or important state buffers."""
    if twin is target or twin.ctx is target.ctx:
        raise ValueError("target must own an independent Session and context")
    if twin.ctx.model is target.ctx.model or twin.ctx.data is target.ctx.data:
        raise ValueError("target must own an independent MjModel and MjData")
    checked = []
    for owner, names in (("data", ("qpos", "qvel", "ctrl")),
                         ("model", ("geom_friction", "actuator_gainprm", "actuator_biasprm"))):
        for name in names:
            left = getattr(getattr(twin.ctx, owner), name)
            right = getattr(getattr(target.ctx, owner), name)
            if np.shares_memory(left, right):
                raise ValueError(f"target shares simulator memory: {owner}.{name}")
            checked.append(f"{owner}.{name}")
    return dict(distinct_session=True, distinct_context=True, distinct_model=True,
                distinct_data=True, disjoint_buffers=checked,
                snapshot_transfer=False, target_initialization="separate_scene_factory_call")


def validate_planner_record(record):
    """Require honest task/LLM provenance, not an inferred LLM attribution."""
    required = {"task_text", "source", "provider", "model", "proposed_orders", "raw_response"}
    if not isinstance(record, dict) or required - set(record):
        raise ValueError("planner record needs task_text, source, provider, model, proposed_orders, raw_response")
    if record["source"] not in {"llm_api", "llm_proxy", "authored_control"}:
        raise ValueError("declare planner source as llm_api, llm_proxy, or authored_control")
    if any(not isinstance(record[x], str) or not record[x].strip()
           for x in ("task_text", "provider", "model", "raw_response")):
        raise ValueError("planner text and model provenance must be nonempty")
    orders = record["proposed_orders"]
    if not isinstance(orders, list) or not orders or any(
            not isinstance(order, list) or not order or any(not isinstance(p, str) for p in order)
            for order in orders):
        raise ValueError("planner proposed_orders must be a nonempty list of part-order lists")
    return copy.deepcopy(record)


def normalize_ranking(result, plans, graphs, k):
    """Recompute ranking from raw logits and bind all scorer exports exactly."""
    ids = [p.id for p in plans]
    logits = np.asarray(result.get("logits"), dtype=float)
    if logits.shape != (len(plans),) or not np.isfinite(logits).all():
        raise ValueError("scorer must return one finite raw logit per input candidate")
    order = np.argsort(-logits, kind="stable").tolist()
    ranked_ids = [ids[i] for i in order]
    if result.get("order") != ranked_ids:
        raise ValueError("scorer order disagrees with raw logits and stable source-order ties")
    exported = result.get("top_k", [])
    if [r["candidate_id"] for r in exported] != ranked_ids[:k]:
        raise ValueError("scorer TopK disagrees with raw-logit ranking")
    by_id = {p.id: (p, g) for p, g in zip(plans, graphs)}
    for row in exported:
        plan, graph = by_id[row["candidate_id"]]
        if row.get("graph_sha256") != digest(graph) or digest(row.get("graph")) != digest(graph):
            raise ValueError("scorer graph differs from the executable candidate")
        if digest(row.get("plan")) != digest(plan.to_dict()):
            raise ValueError("scorer plan differs from the executable candidate")
    return order


def choose_candidate(candidate_ids, trial_rows, accept_rate):
    """Direct empirical success probability; source-order ties, no reward weights.

    A censored/invalid rollout is not positive evidence. It stays in the fixed
    validation budget denominator, so a timeout cannot increase acceptance.
    """
    rows = []
    for index, candidate_id in enumerate(candidate_ids):
        trials = [r for r in trial_rows if r["candidate_id"] == candidate_id]
        if not trials:
            raise ValueError("selected subset has an unvalidated candidate")
        positives = sum(bool(r.get("success")) and bool(r.get("valid"))
                        and not bool(r.get("timeout")) for r in trials)
        rate = positives / len(trials)
        rows.append(dict(candidate_id=candidate_id, source_index=index,
                         successful_trials=positives, trials=len(trials), success_rate=rate,
                         timeouts=sum(bool(r.get("timeout")) for r in trials),
                         accepted=positives > 0 and rate >= accept_rate))
    ordered = sorted(rows, key=lambda r: (-r["success_rate"], r["source_index"]))
    selected = next((r["candidate_id"] for r in ordered if r["accepted"]), None)
    return selected, rows


def render_terminal(session, path, execution):
    """Render the actual terminal data, without stepping or replaying success."""
    import mujoco
    import imageio.v3 as iio
    renderer = None
    try:
        renderer = mujoco.Renderer(session.ctx.model, height=600, width=960)
        renderer.update_scene(session.ctx.data, camera="task_view")
        iio.imwrite(path, renderer.render())
        result = dict(path=str(path), sha256=file_sha(path), source="actual_target_terminal_state",
                      candidate_id=execution["candidate_id"], success=execution["success"],
                      graph_sha256=execution["input_graph_sha256"], trial_sha256=execution["trial_sha256"],
                      simulation_time_s=float(session.ctx.data.time), no_replay=True)
        write_json(Path(path).with_suffix(".json"), result)
        return result
    except Exception as exc:
        return dict(path=None, error=f"{type(exc).__name__}: {exc}", source="render_failure")
    finally:
        if renderer is not None:
            close = getattr(renderer, "close", None) or getattr(renderer, "free", None)
            if close:
                close()


def rollout_with_state_trace(runner, graph, trial):
    """Collect measured simulator state at every control step, without replay."""
    ctx = runner.session.ctx
    previous = getattr(ctx, "on_control_step", None)
    records = {name: [] for name in ("time_s", "qpos", "qvel", "ctrl")}
    def sample(current):
        records["time_s"].append(float(current.data.time))
        for name in ("qpos", "qvel", "ctrl"):
            records[name].append(getattr(current.data, name).copy())
    def callback(current):
        if previous is not None:
            previous(current)
        sample(current)
    ctx.on_control_step = callback
    try:
        row = runner.run(graph, trial, keep_trace=True)
        if not records["time_s"] or records["time_s"][-1] != float(ctx.data.time):
            sample(ctx)
    finally:
        ctx.on_control_step = previous
    arrays = {name: np.asarray(value) for name, value in records.items()}
    return row, arrays


def write_state_trace(path, arrays, scene_path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return dict(path=str(path), sha256=file_sha(path), samples=len(arrays["time_s"]),
                columns=list(arrays), scene_path=str(scene_path), scene_sha256=file_sha(scene_path),
                sampling="post-control-step observed time/qpos/qvel/ctrl; terminal sample included",
                units="MuJoCo SI convention; free-joint quaternions dimensionless; actuator controls per scene XML")


def _scene_record(spec, session, path, role):
    value = asdict(spec) if is_dataclass(spec) else plain(spec)
    return dict(role=role, spec=value, scene_path=str(path), scene_sha256=file_sha(path),
                initial_state_sha256=fingerprint(session),
                initial_simulation_time_s=float(session.ctx.data.time),
                initial_positions={name: session.ctx.obj_pos(name).tolist() for name in session.parts})


def run_system(config, planner_record, scorer, output, method="top_k", *, stage=None,
               runner_factory=PhysicalRunner, clock=time.perf_counter):
    """Run one actual full or TopK policy, including independent target execution.

    ``stage`` exposes make_scene, observed, build_pool, rebind_plan. The scorer
    exposes rank(observation, plans, k) with input-order logits and bound TopK.
    A failed setup remains a reported unsuccessful requested configuration.
    Other programming/integrity failures raise instead of becoming labels.
    """
    config.validate()
    planner = validate_planner_record(planner_record)
    if method not in {"top_k", "full"}:
        raise ValueError("method must be top_k or full")
    if stage is None:
        from . import stage_v5 as stage
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        raise ValueError("refuse to overwrite a completed system run")
    total_started = clock()
    seconds = {key: 0. for key in ("scene_setup", "candidate_generation", "graph_construction", "integrity",
        "encoding", "inference", "ranking_other", "validation", "selection", "target_setup",
        "target_rebinding", "deployment", "rendering", "artifact_writing", "total")}
    planner.update(executable_interface_sha256=interface_hash(),
                   binding_note="LLM symbolic part orders are consumed by the task-specific stage_v5 atomic-skill compiler; proposed_skill_programs are preserved provenance. Grasp/motion solvers instantiate physical parameters and initial joint trajectories; the LLM does not compute joint waypoints.")
    if "skill_catalog" in planner:
        catalog_sha = digest(planner["skill_catalog"])
        if planner.get("skill_catalog_sha256", catalog_sha) != catalog_sha:
            raise ValueError("planner skill catalog hash disagrees with preserved input")
        planner["skill_catalog_sha256"] = catalog_sha
    write_json(output / "planner.json", planner)
    from .program_input_v5 import source_manifest
    sources = source_manifest()
    entrypoint = Path(__file__).resolve().parents[2] / "scripts" / "run_value_v5_system.py"
    if entrypoint.exists():
        sources["scripts/run_value_v5_system.py"] = file_sha(entrypoint)
    write_json(output / "source.json", sources)
    result = dict(schema="twingraph.system_run.v5", config=asdict(config), method=method, policy=method,
        n=config.n, k=config.k if method == "top_k" else config.n,
        config_id=f"stage_v5_{config.seed}", group_id=f"stage_v5_{config.seed}",
        planner_sha256=digest(planner), paired_namespace=dict(twin=5107, target=7901),
        implementation_sha256=digest(sources),
        selection_rule="observed success fraction descending, original pool index ascending",
        acceptance_rule="at least one valid uncensored success and fraction >= accept_rate; timeout stays denominator",
        perception_scope="privileged simulator pose observation; no real-camera perception evaluated",
        target_scope="independent MuJoCo target as real-system proxy; no hardware experiment",
        task_scope="full five-part seating assembly; functional sliding-stroke test excluded",
        uncertainty=dict(friction_scale=[1-config.friction_span, 1+config.friction_span],
                         actuator_gain_scale=[1-config.gain_span, 1+config.gain_span],
                         bounds_known=True, target_draw_hidden_from_scorer=True),
        simulation_calls=dict(twin_validation=0, target_execution=0),
        selected_candidate_id=None, success=False, target_successes=0,
        requested_target_trials=config.target_repeats, target_trials=[], validation=[], images=[],
        seconds=seconds, status="started")
    def finish(status):
        result["status"] = status
        seconds["decision"] = sum(seconds[key] for key in ("candidate_generation", "graph_construction",
            "integrity", "encoding", "inference", "ranking_other", "validation", "selection"))
        seconds["total"] = clock() - total_started
        seconds["other"] = max(0., seconds["total"] - sum(
            value for key, value in seconds.items() if key not in {"decision", "total", "other"}))
        result["timing_boundaries"] = dict(total="own nominal scene construction through final target render and intermediate artifact writes; excludes final result JSON write",
            decision="candidate generation through selection, including integrity and all scoring work; excludes scene setup, writes and target execution",
            validation="all actual nominal-twin runner calls including their resets and graph validation",
            deployment="actual independent-target runner calls including their own resets and graph validation")
        write_json(output / "result.json", result)
        return result
    started = clock()
    try:
        spec, twin, scene_path, targets = stage.make_scene(config.seed, output / "twin_scene", role="twin")
    except Exception as exc:
        from simbench.assembly.library import SkillFailure
        if not isinstance(exc, SkillFailure) and not (isinstance(exc, ValueError) and str(exc).startswith("IK unreachable:")):
            raise
        seconds["scene_setup"] = clock() - started
        result["setup_error"] = f"{type(exc).__name__}: {exc}"
        return finish("twin_setup_failed")
    seconds["scene_setup"] = clock() - started
    result["twin"] = _scene_record(spec, twin, scene_path, "twin")
    observation = stage.observed(twin, targets)
    # Later protocols may add a pre-execution sensor artifact while retaining
    # this audited orchestration.  The default v5 stage has no such hook and is
    # therefore byte-for-byte equivalent in its inputs and execution path.
    vision_arrays = None
    vision_manifest = None
    if hasattr(stage, "capture_vision"):
        started = clock()
        vision_arrays = stage.capture_vision(twin)
        vision_manifest = stage.save_vision(output / "vision.npz", vision_arrays)
        seconds["artifact_writing"] += clock() - started
        result["perception_scope"] = "pre-execution fixed-camera RGB-D plus structured detector-state contract"
        result["vision"] = vision_manifest
    started = clock()
    try:
        plans, counts = stage.build_pool(twin, targets, config.seed, n=config.n,
                                        orders=planner["proposed_orders"], precheck=True)
    except Exception as exc:
        expected = getattr(stage, "CandidateGenerationError", None)
        if expected is None or not isinstance(exc, expected):
            raise
        seconds["candidate_generation"] = clock() - started
        result["actual_n"] = 0
        result["candidate_generation_error"] = f"{type(exc).__name__}: {exc}"
        return finish("no_materializable_candidates")
    seconds["candidate_generation"] = clock() - started
    result["candidate_counts"] = counts
    result["actual_n"] = len(plans)
    if not plans:
        return finish("no_materializable_candidates")
    if len({p.id for p in plans}) != len(plans):
        raise ValueError("duplicate candidate IDs")
    started = clock()
    graphs = [compile_graph(observation, p) for p in plans]
    seconds["graph_construction"] = clock() - started
    started = clock()
    for graph, plan in zip(graphs, plans):
        if digest(validate_graph(graph).to_dict()) != digest(plan.to_dict()):
            raise ValueError("compiled graph does not recover the executable PlanIR")
    seconds["integrity"] += clock() - started
    inputs = dict(schema="twingraph.system_inputs.v6" if vision_manifest else "twingraph.system_inputs.v5",
                  observation=observation, candidates=[p.to_dict() for p in plans],
                  planner_sha256=digest(planner))
    if vision_manifest:
        inputs["vision"] = vision_manifest
    result["inputs_sha256"] = digest(inputs)
    started = clock()
    write_json(output / "inputs.json", inputs)
    seconds["artifact_writing"] += clock() - started
    if method == "top_k":
        if scorer is None:
            raise ValueError("TopK policy requires a trained value scorer")
        started = clock()
        rank_kwargs = dict(k=min(config.k, len(plans)))
        if vision_arrays is not None:
            rank_kwargs["vision"] = copy.deepcopy(vision_arrays)
        ranked = scorer.rank(copy.deepcopy(observation), copy.deepcopy(plans), **rank_kwargs)
        rank_wall = clock() - started
        claimed = ranked.get("seconds", {})
        scorer_phases = ("graph_construction", "integrity", "encoding", "inference")
        for name in scorer_phases:
            value = float(claimed.get(name, 0.))
            if not np.isfinite(value) or value < 0:
                raise ValueError("invalid scorer timing")
            seconds[name] += value
        seconds["ranking_other"] = max(0., rank_wall-sum(float(claimed.get(x, 0.)) for x in scorer_phases))
        started = clock()
        order = normalize_ranking(ranked, plans, graphs, min(config.k, len(plans)))
        seconds["integrity"] += clock() - started
        selected_indices = set(order[:config.k])
        result["ranking"] = {key: value for key, value in ranked.items() if key != "top_k"}
        result["ranking"]["top_k"] = [plans[i].id for i in order[:config.k]]
    else:
        selected_indices = set(range(len(plans)))
    # Iterate in source order under both methods, including tie breaking.
    subset = [(p, g) for i, (p, g) in enumerate(zip(plans, graphs)) if i in selected_indices]
    result["validated_candidate_ids"] = [p.id for p, _ in subset]
    started = clock()
    runner = runner_factory(twin, timeout=config.timeout)
    seconds["validation"] += clock() - started
    for plan, graph in subset:
        for repeat in range(config.validation_repeats):
            started = clock()
            trial = (stage.trial_for(config, repeat, "twin")
                     if hasattr(stage, "trial_for") else trial_for(config, repeat, "twin"))
            row, trace = rollout_with_state_trace(runner, graph, trial)
            seconds["validation"] += clock() - started
            if row["candidate_id"] != plan.id or row["input_graph_sha256"] != digest(graph):
                raise ValueError("physical result is not bound to submitted twin graph")
            result["simulation_calls"]["twin_validation"] += 1
            result["validation"].append(row)
            started = clock()
            row["state_trace"] = write_state_trace(output / "validation" / f"{plan.id}_{repeat}.npz", trace, scene_path)
            write_json(output / "validation" / f"{plan.id}_{repeat}.json", row)
            seconds["artifact_writing"] += clock() - started
    started = clock()
    selected, summary = choose_candidate([p.id for p, _ in subset], result["validation"], config.accept_rate)
    seconds["selection"] = clock() - started
    result["candidate_validation"] = summary
    result["selected_candidate_id"] = selected
    if selected is None:
        return finish("abstained_no_validated_plan")
    chosen = next(p for p in plans if p.id == selected)
    result["selected_graph_sha256"] = digest(graphs[plans.index(chosen)])
    result["selected_semantic_sha256"] = digest(semantic_payload(chosen))
    # Create a fresh target for each actual repeat; never restore any twin data.
    for repeat in range(config.target_repeats):
        started = clock()
        try:
            target_spec, target, target_path, target_goals = stage.make_scene(
                config.seed, output / f"target_scene_{repeat}", role="target")
        except Exception as exc:
            from simbench.assembly.library import SkillFailure
            if not isinstance(exc, SkillFailure) and not (isinstance(exc, ValueError) and str(exc).startswith("IK unreachable:")):
                raise
            seconds["target_setup"] += clock() - started
            result["target_trials"].append(dict(repeat=repeat, success=False, status="target_setup_failed",
                                                  error=f"{type(exc).__name__}: {exc}"))
            continue
        seconds["target_setup"] += clock() - started
        isolation = assert_independent(twin, target)
        target_scene = _scene_record(target_spec, target, target_path, "target")
        started = clock()
        try:
            rebound = stage.rebind_plan(target, target_goals, chosen)
        except Exception as exc:
            from simbench.assembly.library import SkillFailure
            if not isinstance(exc, SkillFailure) and not (isinstance(exc, ValueError) and str(exc).startswith("IK unreachable:")):
                raise
            seconds["target_rebinding"] += clock() - started
            result["target_trials"].append(dict(repeat=repeat, success=False, status="target_rebinding_failed",
                scene=target_scene, isolation=isolation, selected_candidate_id=selected,
                error=f"{type(exc).__name__}: {exc}"))
            continue
        if isinstance(rebound, tuple):
            rebound, rebind_details = rebound
        else:
            rebind_details = {}
        if digest(semantic_payload(rebound)) != result["selected_semantic_sha256"]:
            raise ValueError("target rebinding changed selected task choices or action parameters")
        target_observation = stage.observed(target, target_goals)
        target_graph = compile_graph(target_observation, rebound)
        validate_graph(target_graph)
        seconds["target_rebinding"] += clock() - started
        started = clock()
        write_json(output / f"target_input_{repeat}.json", dict(observation=target_observation,
            plan=rebound.to_dict(), graph=target_graph, selected_candidate_id=selected,
            semantic_sha256=result["selected_semantic_sha256"]))
        seconds["artifact_writing"] += clock() - started
        started = clock()
        target_runner = runner_factory(target, timeout=config.timeout)
        trial = (stage.trial_for(config, repeat, "target")
                 if hasattr(stage, "trial_for") else trial_for(config, repeat, "target"))
        execution, trace = rollout_with_state_trace(target_runner, target_graph, trial)
        seconds["deployment"] += clock() - started
        if execution["candidate_id"] != rebound.id or execution["input_graph_sha256"] != digest(target_graph):
            raise ValueError("physical result is not bound to submitted target graph")
        result["simulation_calls"]["target_execution"] += 1
        target_row = dict(repeat=repeat, status="executed", success=bool(execution["success"] and
            execution["valid"] and not execution["timeout"]), scene=target_scene, isolation=isolation,
            selected_candidate_id=selected, rebound_candidate_id=rebound.id,
            semantic_sha256=result["selected_semantic_sha256"], rebind_details=rebind_details,
            execution=execution)
        result["target_trials"].append(target_row)
        started = clock()
        execution["state_trace"] = write_state_trace(output / f"target_state_trace_{repeat}.npz", trace, target_path)
        write_json(output / f"target_execution_{repeat}.json", target_row)
        seconds["artifact_writing"] += clock() - started
        if config.render:
            started = clock()
            result["images"].append(render_terminal(target, output / f"target_terminal_{repeat}.png", execution))
            seconds["rendering"] += clock() - started
    result["target_successes"] = sum(row["success"] for row in result["target_trials"])
    result["success"] = result["target_successes"] == config.target_repeats
    result["target_success_rate"] = result["target_successes"] / config.target_repeats
    return finish("executed_independent_target")


def run_pair(config, planner_record, scorer, output, **kwargs):
    """Actually run both policies; compare identical generated inputs and trials."""
    started = time.perf_counter()
    topk = run_system(config, planner_record, scorer, Path(output) / "top_k", "top_k", **kwargs)
    full = run_system(config, planner_record, None, Path(output) / "full", "full", **kwargs)
    if topk.get("inputs_sha256") != full.get("inputs_sha256"):
        raise ValueError("paired policies did not receive identical candidate inputs")
    left = {(r["candidate_id"], r["trial_sha256"]): r for r in topk["validation"]}
    right = {(r["candidate_id"], r["trial_sha256"]): r for r in full["validation"]}
    if not set(left) <= set(right):
        raise ValueError("TopK twin trials are not paired with exhaustive twin trials")
    comparisons = []
    for key, row in left.items():
        other = right[key]
        comparisons.append(dict(candidate_id=key[0], trial_sha256=key[1],
            same_success=row["success"] == other["success"],
            same_timeout=row["timeout"] == other["timeout"],
            same_final_positions=digest(row["final_positions"]) == digest(other["final_positions"])))
    result = dict(schema="twingraph.system_pair.v5", config=asdict(config),
        top_k=str(Path(output) / "top_k" / "result.json"), full=str(Path(output) / "full" / "result.json"),
        input_sha256=topk.get("inputs_sha256"), paired_trial_checks=comparisons,
        decision_seconds=dict(top_k=topk["seconds"]["decision"], full=full["seconds"]["decision"]),
        target_successes=dict(top_k=topk["target_successes"], full=full["target_successes"]),
        requested_target_trials_per_policy=config.target_repeats,
        actual_wall_seconds=time.perf_counter()-started,
        execution_schedule="top_k then full, sequential, no result-cache reuse")
    write_json(Path(output) / "pair.json", result)
    write_json(Path(output) / "timing_rows.json", dict(schema="twingraph.system_run.v5",
        time_boundaries=topk["timing_boundaries"], rows=[topk, full]))
    return result
