"""Printed-kit physical experiments; every candidate is a real fresh rollout."""
from copy import deepcopy
from pathlib import Path
import contextlib
import time
import numpy as np

from simbench.assembly.control import HOME
from simbench.assembly.library import Result, SkillFailure
from .full_task_v7 import _clean, _stroke
from .physical import PhysicalRunner, perturbation
from .plan import execute_calls, digest
from .planner_v12 import assembly_program, normalized_graph, propose
from .system_v11 import save, observe_boundary, discrepancy, twin_checkpoint, restore_twin_checkpoint
from .v9_candidates import bind
from .provenance_v12 import fingerprint as source_fingerprint

RUNTIME_AT_IMPORT=source_fingerprint()
METHODS=("all_twin","random_top_k","value_top_k","random_early_stop","value_early_stop",
         "geometry_top_k","geometry_early_stop")
VALUE_METHODS=("value_top_k","value_early_stop")


def geometry_scores(graphs):
    """Cheap, predeclared geometric baseline using the same graph as value.

    Rank by worst available nominal clearance relative to visual reserve,
    with the number of checked stages as a secondary preference. This is a
    heuristic score, not a predicted success probability or rollout label.
    """
    from .skill_graph import validate_geometry_conditions
    scores=[]
    for graph in graphs:
        rows=validate_geometry_conditions(graph)
        if not rows:
            raise ValueError("geometry baseline requires graph-bound CAD conditions")
        margins=[]
        for row in rows.values():
            value=row.get("min_clearance_m"); reserve=row.get("required_clearance_m")
            if value is not None and reserve is not None:
                margins.append(float(np.clip(value/max(float(reserve),1e-6),-10.,10.)))
        minimum=min(margins) if margins else -11.
        scores.append(minimum+.001*sum(r.get("status")=="necessary_pass" for r in rows.values()))
    return np.asarray(scores,float)


def verification_schedule(method, graphs, value, seed, k, *, replan=0):
    """Decide order/budget before observing any candidate outcome."""
    if method not in METHODS: raise ValueError(method)
    n=len(graphs)
    if n < 1 or not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("nonempty candidate pool and positive integer k required")
    scores=None
    if method in VALUE_METHODS:
        if value is None: raise ValueError(f"{method} requires a value checkpoint")
        scores=np.asarray(value.score(graphs),dtype=float)
        if scores.shape!=(n,) or not np.isfinite(scores).all():
            raise ValueError("value scores must be finite and match candidate pool")
        order=np.argsort(-scores,kind="stable").tolist()
    elif method.startswith("geometry"):
        scores=geometry_scores(graphs)
        order=np.argsort(-scores,kind="stable").tolist()
    elif method.startswith("random"):
        entropy=[seed,1291]+([replan] if replan else [])
        order=np.random.default_rng(np.random.SeedSequence(entropy)).permutation(n).tolist()
    else: order=list(range(n))
    budget=n if method=="all_twin" or method.endswith("early_stop") else min(k,n)
    progressive=(dict(initial_k=min(k,n),batch_size=min(k,n),max_candidates=n,
        policy="ordered_batches_first_verified_success") if method in ("value_early_stop","geometry_early_stop") else None)
    return scores,order,budget,progressive


def verification_batches(order, attempted, progressive):
    """Only opened batches are logged; unattempted members are not rollouts."""
    if progressive is None: return []
    size=progressive["batch_size"]
    return [dict(batch=start//size+1,rank_start=start,rank_stop=min(start+size,len(order)),
                 attempted_indices=order[start:min(start+size,attempted)])
            for start in range(0,attempted,size)]


def require_frozen_source():
    current=source_fingerprint()
    if current["sha256"]!=RUNTIME_AT_IMPORT["sha256"]:
        raise RuntimeError("runtime source/CAD/policy changed during process; restart in a new experiment directory")
    return current


def make_scene(seed, directory, *, domain="online", level="L1"):
    from . import stage_v12
    from simbench.assembly.skills_v12 import configure_v12_skills
    from .supply_layout_v13 import apply as supply_layout
    result = stage_v12.make_scene(seed, directory, role=domain, level=level,
        scene_layout_hook=supply_layout)
    configure_v12_skills(result[1])
    from .stage_v7 import refresh_visual_observation
    refresh_visual_observation(result[1], parts=result[1].parts)
    return result


def _receiver_observation(session):
    from .stage_v7 import refresh_visual_observation
    observation = refresh_visual_observation(session, parts=session.parts)
    if not observation.get("fixtures", {}).get("guide_base", {}).get("valid"):
        raise SkillFailure("receiver RGB-D unknown; cannot bind assembly targets")
    return observation


def _require_stop_corridor(observation):
    relation = observation.get("fixture_relations", {}).get("end_stop_to_base", {})
    if not relation.get("observable"):
        raise SkillFailure("two-layer receiver relation unobserved; reobserve before inserting")
    if not relation.get("geometric_route_exists"):
        raise SkillFailure("end_stop placement leaves no shared shaft route into base; reposition predecessor")
    return relation


def _prepare_receiver_targets(session, part, proposal):
    """Bind a stage from a fresh released observation before compiling it.

    The same choice program is preserved. Only its receiver-relative targets
    are rebound from the declared visual feedback contract; no already
    compiled PlanIR is silently rewritten and no body-pose oracle is read.
    """
    observation = _receiver_observation(session)
    targets = deepcopy(observation.get("assembly_targets", {}))
    if set(targets) != set(proposal["choices"]):
        raise SkillFailure("incomplete observed receiver/CAD target bindings")
    if part.startswith("pin_"):
        relation = _require_stop_corridor(observation)
        for pin, row in relation["holes"].items():
            outward = np.asarray(row["axis"], float)
            outward /= np.linalg.norm(outward)
            point = np.asarray(row["common_axis_point_m"], float)
            stop_entry = np.asarray(row["stop_entry_m"], float)
            entry = np.asarray(row.get("stop_entry_on_common_axis_m",
                point + outward * float(np.dot(stop_entry - point, outward))), float)
            depth = float(row["minimum_total_depth_m"])
            tip = float(session.planning_cad["pin_shaft_offsets_m"][0])
            targets[pin].update(hole_entry_m=entry.tolist(), axis=outward.tolist(),
                position_m=(entry - outward * (depth + tip)).tolist(),
                minimum_total_depth_m=depth,
                source="fresh RGB-D shared two-layer shaft corridor")
    if part == "handle":
        from .cad_rgbd_v12 import _observed_transform
        measured = _observed_transform(observation["objects"].get("carriage", {}))
        if measured is None:
            raise SkillFailure("installed carriage RGB-D unknown before handle placement")
        position, rotation = measured
        targets["handle"]["position_m"] = (position + rotation[:, 2] * float(
            session.planning_cad["handle_seat_center_offset_from_carriage_m"])).tolist()
        targets["handle"]["source"] = "fresh RGB-D carriage post + static CAD mating"
    session.receiver_execution_targets = targets
    session.stage_targets = {name:list(row["position_m"]) for name,row in targets.items()}
    bindings = getattr(session, "receiver_binding_history", [])
    bindings.append(dict(stage=part, observation_sha256=observation["sha256"], targets=deepcopy(targets),
        source="wrist_RGBD_and_static_CAD", object_pose_oracle=False))
    session.receiver_binding_history = bindings
    save(Path(session.out)/"receiver_binding_history.json", bindings)
    return observation


def run_staged(session, *, order, choices, wipe_variant=0, wipe_force=1.5,
               wipe_duration=14., stroke_minimum=.02, completed=(), monitor=None, boundaries=None):
    required_stroke=float(session.planning_cad.get("functional_stroke_minimum_m", .02))
    if not np.isclose(stroke_minimum, required_stroke,rtol=0,atol=1e-12):
        raise ValueError("candidate cannot change functional task requirement")
    proposal = dict(order=list(order), choices=deepcopy(choices), wipe_variant=wipe_variant,
                    wipe_force=wipe_force, wipe_duration=wipe_duration, stroke_minimum=stroke_minimum)
    done = list(completed)
    boundaries = boundaries if boundaries is not None else []
    if not done:
        session.stage_passes = dict(cleaning_pass=False, assembly_pass=False,
            functional_test_pass=False, fixture_capture_pass=False,
            final_seat_pass=False, final_release_and_retraction_pass=False)

    def boundary(stage, failure=None):
        nonlocal proposal
        session.stage_completed=tuple(done)
        row = observe_boundary(session, stage, done)
        if failure is not None: row["failure"] = str(failure)
        boundaries.append(row)
        if monitor:
            replacement = monitor(session, row, proposal)
            if replacement is not None:
                for part in done:
                    if part in proposal["choices"] and replacement["choices"][part] != proposal["choices"][part]:
                        raise ValueError("cannot rewrite completed action")
                proposal = deepcopy(replacement)
                return True
        return False
    if "cleaning" not in done:
        _clean(session, wipe_variant, wipe_force, wipe_duration)
        done.append("cleaning"); boundary("cleaning")
    while any(part not in done for part in proposal["order"]):
        part = next(p for p in proposal["order"] if p not in done)
        try:
            _prepare_receiver_targets(session, part, proposal)
            plan = assembly_program(session, proposal)
            save(Path(session.out)/"stage_programs"/f"{len(boundaries):02d}_{part}.json",
                dict(observation_sha256=session.decision_observation["sha256"], plan=plan.to_dict(),
                     binding=session.receiver_binding_history[-1]))
            calls = [c for c in plan.calls if c.roles.get("manipulated") == part and not c.id.startswith("accept_")]
            execute_calls(session, plan, calls)
            if part == "end_stop":
                # Released is not synonymous with ready for the next skill.
                # Do not mark this predecessor complete if both holes cannot
                # accept a shared straight shaft through the base.
                _require_stop_corridor(session.decision_observation)
        except SkillFailure as exc:
            if monitor is None: raise
            # Recoverable failure is also a decision boundary. Do not mark
            # the failed part complete or silently restart completed actions.
            # A held object requires an explicit recovery procedure: the
            # monitor stops rather than moving/releasing it without a plan.
            if session.held is None: session.call("move", target="home")
            if not boundary(f"failed_{part}", failure=exc): raise
            continue
        done.append(part); boundary(part)
    plan = assembly_program(session, proposal)
    execute_calls(session, plan, [c for c in plan.calls if c.id.startswith("accept_") or c.id == "final_home"])
    session.stage_passes["assembly_pass"] = True
    _stroke(session, stroke_minimum)
    for part in ("pin_left", "pin_right"):
        session.call("inspect", what="pin_joint", part=part, phase="retained_after_stroke")
    from simbench.assembly.skills_v12 import evaluate_end_stop_fixture, evaluate_functional_seat
    fixture_ok, fixture_metrics = evaluate_end_stop_fixture(session)
    session.artifacts["final_fixture_acceptance"] = fixture_metrics
    session.stage_passes["fixture_capture_pass"] = bool(fixture_ok)
    # Stroke release, retraction and home all advance physics. Earlier seat
    # checks cannot establish that the completed assembly remains engaged.
    final_seats = {}
    for part in ("carriage", "handle"):
        seat_ok, seat_metrics = evaluate_functional_seat(session, part, session.stage_targets[part])
        final_seats[part] = dict(success=bool(seat_ok), **seat_metrics)
    session.artifacts["final_seat_acceptance"] = final_seats
    session.stage_passes["final_seat_pass"] = all(row["success"] for row in final_seats.values())
    session.stage_passes["final_release_and_retraction_pass"] = bool(
        session.held is None and np.max(np.abs(session.ctx.arm_qpos - HOME)) < .02)
    if not all(session.stage_passes.values()):
        raise SkillFailure("complete functional task predicate failed")
    done.extend(("stroke", "retention")); boundary("finished")
    return Result(True, dict(stage_passes=deepcopy(session.stage_passes), completed=done))


def run_mechanism_prefix(session, *, stop_after, order, choices, wipe_variant=0,
                         wipe_force=1.5, wipe_duration=14., stroke_minimum=.02,
                         completed=(), **_):
    """Execute an assembly prefix for candidate-coverage development only.

    This deliberately does not set the complete-task ``stage_passes`` flags.
    A positive result means that every declared stage through ``stop_after``
    executed and passed its own physical checks; it is never a full-task
    success label.  The input is still the same complete proposal consumed by
    the value/twin pipeline, so the experiment can diagnose whether useful
    alternatives exist before spending time on later common-prefix failures.
    """
    del wipe_variant, wipe_force, wipe_duration
    required_stroke=float(session.planning_cad.get("functional_stroke_minimum_m", .02))
    if not np.isclose(stroke_minimum, required_stroke, rtol=0, atol=1e-12):
        raise ValueError("candidate cannot change functional task requirement")
    if stop_after not in order:
        raise ValueError("mechanism stop stage must occur in the executable order")
    if completed:
        raise ValueError("fresh-scene mechanism evidence cannot claim a completed prefix")
    proposal=dict(order=list(order), choices=deepcopy(choices), wipe_variant=0,
                  wipe_force=1.5, wipe_duration=14., stroke_minimum=stroke_minimum)
    done=[]
    for part in proposal["order"]:
        _prepare_receiver_targets(session, part, proposal)
        plan=assembly_program(session, proposal)
        save(Path(session.out)/"mechanism_stage_programs"/f"{len(done):02d}_{part}.json",
             dict(observation_sha256=session.decision_observation["sha256"],
                  plan=plan.to_dict(), binding=session.receiver_binding_history[-1]))
        calls=[c for c in plan.calls if c.roles.get("manipulated")==part
               and not c.id.startswith("accept_")]
        execute_calls(session, plan, calls)
        if part=="end_stop":
            _require_stop_corridor(session.decision_observation)
        done.append(part)
        if part==stop_after:
            break
    if done[-1] != stop_after:
        raise RuntimeError("mechanism prefix did not reach its declared stop stage")
    return Result(True, dict(scope="assembly_prefix", stop_after=stop_after,
                             completed=done, full_task_success=False))


def rollout(seed, proposal, directory, *, domain="online", checkpoint=None,
            completed=(), monitor=None, level="L1", record=False, stop_after=None):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    provenance=require_frozen_source()
    started = time.perf_counter(); boundaries = []
    monitor_seconds=0.;monitor_timings=[]
    def timed_monitor(session,actual,current):
        nonlocal monitor_seconds
        began=time.perf_counter()
        try:
            return monitor(session,actual,current)
        finally:
            elapsed=time.perf_counter()-began
            monitor_seconds+=elapsed
            monitor_timings.append(dict(stage=actual.get("stage"),wall_seconds=elapsed))
    with (directory / "console.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        _, session, _, _ = make_scene(seed, directory / "scene", domain=domain, level=level)
        if checkpoint is not None: restore_twin_checkpoint(session, checkpoint)
        initial = deepcopy(session.decision_observation)
        graph = normalized_graph(session, proposal, completed=completed)
        save(directory / "input_graph.json", graph)
        if stop_after is None:
            session.full_task_controller = lambda s, **params: run_staged(s, **params,
                completed=completed, monitor=timed_monitor if monitor is not None else None, boundaries=boundaries)
        else:
            if monitor is not None or checkpoint is not None or completed:
                raise ValueError("mechanism-prefix rollout requires a fresh scene without replanning")
            session.full_task_controller = lambda s, **params: run_mechanism_prefix(
                s, stop_after=stop_after, completed=(), **params)
        recorder = None
        if record:
            from scripts.record_v12_execution import RecorderV12
            recorder = RecorderV12(session, directory / "execution.mp4",
                f"V12 / seed {seed} / {proposal['name']} / {domain}")
            session.rec = recorder; session.ctx.on_control_step = recorder
        result = PhysicalRunner(session, timeout=900. if record else 600.,
            excluded_wall_seconds=(lambda:monitor_seconds) if monitor is not None else None).run(
            bind(session, session.stage_targets, proposal), perturbation(seed, 0, domain), keep_trace=True)
        result.update(seed=seed, boundaries=boundaries, initial_observation=initial, proposal=proposal,
            total_wall_seconds=time.perf_counter()-started, domain=domain,
            input_graph_sha256=digest(graph), source="actual printed-kit MuJoCo execution",
            checkpoint_sync="exact_simulator_checkpoint" if checkpoint is not None else "paired_seed_initial_scene",
            geometry_version=session.task_version, independent_hardware=False,
            fixture_acceptance=deepcopy(session.artifacts.get("final_fixture_acceptance")),
            final_seat_acceptance=deepcopy(session.artifacts.get("final_seat_acceptance")),
            monitor_wall_seconds=monitor_seconds,monitor_timings=monitor_timings,
            resource_censored=bool(result.get("timeout")),
            runtime_sha256=provenance["sha256"],
            evaluation_scope=("complete_functional_task" if stop_after is None else
                              f"assembly_prefix_through_{stop_after}"),
            full_task_label=bool(stop_after is None))
        if result.get("timeout"):
            result.update(valid=False,invalid_reason="execution wall-time budget exhausted; resource-censored, not a physical feasibility label")
        if source_fingerprint()["sha256"]!=provenance["sha256"]:
            result.update(valid=False, invalid_reason="source/CAD/policy changed during physical rollout")
        save(directory / "result.json", result)
        if recorder: recorder.close(result)
    return result


class ClosedLoop:
    def __init__(self, seed, expected, value, directory, *, max_replans=1, k=4, n=48, level="L1", method="value_top_k"):
        self.seed=seed; self.expected={r["stage"]:r for r in expected}; self.value=value
        self.directory=Path(directory); self.max_replans=max_replans; self.k=k; self.n=n
        self.level=level; self.events=[]; self.replans=0; self.method=method

    def __call__(self, session, actual, current):
        # This is a replanning threshold, not a millimetre-level task success
        # test. Increase it with observed uncertainty; missing pending state
        # remains a reason to reobserve rather than assume agreement.
        sigmas=[float(r.get("fit_residual_m") or .003) for r in actual["observation"]["objects"].values() if r.get("valid")]
        limit=max(.012,3*max(sigmas,default=.004))
        comparison=discrepancy(self.expected.get(actual["stage"]),actual,position_limit=limit)
        event=dict(stage=actual["stage"],comparison=comparison,action="continue")
        if actual.get("failure") is not None: event["failure"] = actual["failure"]
        self.events.append(event)
        if actual["stage"] != "finished" and comparison["replan"]:
            if session.held is not None or self.replans >= self.max_replans:
                event["action"]="stop_recovery_or_budget_required"
                save(self.directory/"events.json",self.events)
                raise SkillFailure(event["action"])
            self.replans+=1
            try:
                pool,source=propose(session.decision_observation,cad=session.planning_cad,
                    n=self.n,seed=self.seed,completed=actual["completed"])
            except ValueError as exc:
                event.update(action="stop_reobserve_required",error=str(exc))
                save(self.directory/"events.json",self.events)
                raise SkillFailure(str(exc)) from exc
            for proposal in pool:
                for part in actual["completed"]:
                    if part in current["choices"]: proposal["choices"][part]=deepcopy(current["choices"][part])
                done=[p for p in current["order"] if p in actual["completed"]]
                proposal["order"]=done+[p for p in proposal["order"] if p not in done]
            graphs=[normalized_graph(session,p,completed=actual["completed"]) for p in pool]
            scores,order,budget,progressive=verification_schedule(
                self.method,graphs,self.value,self.seed,self.k,replan=self.replans)
            root=self.directory/f"replan_{self.replans}"
            save(root/"request.json",dict(pool=pool,source=source,scores=list(scores) if scores is not None else None,order=order,
                method=self.method,k=budget,progressive=progressive,
                completed=actual["completed"],synchronization="exact simulator checkpoint"))
            state=twin_checkpoint(session)
            selected=None; selected_boundaries=None; trials=[]
            for i in order[:budget]:
                trial=rollout(self.seed,pool[i],root/pool[i]["name"],checkpoint=state,
                    completed=actual["completed"],level=self.level)
                if not trial.get("valid"):
                    raise RuntimeError("invalid replanning trial; physical evidence cannot be used")
                if trial["success"] and selected is None:
                    selected=i; selected_boundaries=trial["boundaries"]
                trials.append(dict(index=i,name=pool[i]["name"],success=trial["success"]))
                save(root/"progress.json",dict(trials=trials,selected=selected,
                    verification_batches=verification_batches(order,len(trials),progressive)))
                if selected is not None and self.method!="all_twin": break
            if selected is not None:
                self.expected={r["stage"]:r for r in selected_boundaries}
                event.update(action="verified_remaining_suffix",selected=pool[selected]["name"])
                save(self.directory/"events.json",self.events)
                return pool[selected]
            event["action"]="stop_no_verified_suffix"
            save(self.directory/"events.json",self.events)
            raise SkillFailure(event["action"])
        save(self.directory/"events.json",self.events)
        return None


def collect(seed, directory, *, n=48, level="L1", names=None, record=False, domain="train"):
    """Collect complete matrix labels; do not rank using any of these labels."""
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    request_path=directory/"request.json"
    provenance=require_frozen_source()
    if request_path.exists():
        import json
        request=json.loads(request_path.read_text());pool=request["pool"]
        if request["seed"]!=seed or len(pool)!=n: raise ValueError("incompatible resume request")
        if request.get("runtime_sha256")!=provenance["sha256"]:
            raise ValueError("resume source/CAD/policy mismatch; use a new experiment directory")
    else:
        with (directory/"generation.log").open("w") as log,contextlib.redirect_stdout(log):
            _,session,_,_=make_scene(seed,directory/"decision",domain=domain,level=level)
            save(directory/"generation_input.json",dict(seed=seed,level=level,
                observation=session.decision_observation,cad=session.planning_cad,
                runtime_sha256=provenance["sha256"]))
            pool,source=propose(session.decision_observation,cad=session.planning_cad,n=n,seed=seed)
        save(request_path,dict(seed=seed,level=level,domain=domain,pool=pool,source=source,
            geometry_version=session.task_version,initial_observation=session.decision_observation,
            runtime_sha256=provenance["sha256"]))
        save(directory/"runtime_sources.json",provenance)
    results=[]
    for proposal in pool:
        if names and proposal["name"] not in names: continue
        root=directory/"candidates"/proposal["name"]
        if (root/"result.json").exists():
            import json
            result=json.loads((root/"result.json").read_text())
        else:
            result=rollout(seed,proposal,root,domain=domain,level=level,record=record)
        if not result.get("valid"):
            raise RuntimeError(f"invalid trial {proposal['name']}: {result.get('invalid_reason', result.get('error'))}")
        results.append(dict(name=proposal["name"],success=result["success"],error=result["error"],wall_seconds=result["total_wall_seconds"]))
        save(directory/"progress.json",dict(seed=seed,n=n,completed=len(results),trials=results))
    return results


def run_method(seed, method, directory, value, *, n=48, k=4, level="L1", record=False):
    if method not in METHODS:
        raise ValueError(method)
    started=time.perf_counter(); directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    provenance=require_frozen_source()
    with (directory/"generation.log").open("w") as log,contextlib.redirect_stdout(log):
        _,session,_,_=make_scene(seed,directory/"decision",level=level)
        pool,source=propose(session.decision_observation,cad=session.planning_cad,n=n,seed=seed)
        graphs=[normalized_graph(session,p) for p in pool]
    generation_seconds=time.perf_counter()-started
    t=time.perf_counter()
    scores,order,budget,progressive=verification_schedule(method,graphs,value,seed,k)
    ranking_seconds=time.perf_counter()-t
    save(directory/"request.json",dict(seed=seed,method=method,n=n,k=budget,pool=pool,source=source,progressive=progressive,
        level=level,domain="online",runtime_sha256=provenance["sha256"],
        initial_observation=session.decision_observation,geometry_version=session.task_version,
        scores=list(scores) if scores is not None else None,order=order,
        model_sha256=getattr(value,"sha256",None),generation_seconds=generation_seconds,ranking_seconds=ranking_seconds))
    save(directory/"runtime_sources.json",provenance)
    trials=[]; selected=None; expected=None
    for i in order[:budget]:
        result=rollout(seed,pool[i],directory/"twins"/pool[i]["name"],level=level)
        if not result.get("valid"):
            raise RuntimeError("invalid online trial; restart from a consistent frozen runtime")
        trials.append(dict(index=i,name=pool[i]["name"],success=result["success"],error=result["error"],wall_seconds=result["total_wall_seconds"]))
        if result["success"] and selected is None: selected=i; expected=result["boundaries"]
        save(directory/"progress.json",dict(trials=trials,selected=selected,
            verification_batches=verification_batches(order,len(trials),progressive)))
        if selected is not None and method!="all_twin": break
    decision_seconds=time.perf_counter()-started
    deployment=None;events=[]
    if selected is not None:
        monitor=ClosedLoop(seed,expected,value,directory/"closed_loop",k=k,n=n,level=level,method=method)
        deployment=rollout(seed,pool[selected],directory/"deployment",domain="deployment",monitor=monitor,level=level,record=record)
        if not deployment.get("valid"):
            raise RuntimeError("invalid deployment trial; evidence excluded")
        events=monitor.events
    result=dict(seed=seed,method=method,k=budget,pool_size=n,trials=trials,
        progressive=progressive,verification_batches=verification_batches(order,len(trials),progressive),
        valid=True,runtime_sha256=provenance["sha256"],
        selected=pool[selected]["name"] if selected is not None else None,twin_success=selected is not None,
        execution_success=bool(deployment and deployment["success"]),
        execution_error=deployment["error"] if deployment else "no_verified_plan",
        generation_seconds=generation_seconds,ranking_seconds=ranking_seconds,decision_seconds=decision_seconds,
        execution_seconds=deployment["total_wall_seconds"] if deployment else 0.,
        total_wall_seconds=time.perf_counter()-started,events=events,
        note="actual online printed-kit simulation, not hardware or cached timing")
    save(directory/"summary.json",result)
    return result
