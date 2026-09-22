"""Measured end-to-end screening and stage-boundary replanning experiment.

Candidate verification and target deployment use separate MjModel/MjData.
Replanning twins use a checkpoint copy in this simulation experiment; this
privileged synchronization is explicitly recorded and is not a real-robot twin.
"""
from copy import deepcopy
from pathlib import Path
import contextlib
import hashlib
import json
import time

import numpy as np

from simbench.assembly.control import HOME
from simbench.assembly.library import Result, SkillFailure
from . import stage_v7, stage_v9
from .full_task_v7 import _clean, _stroke, _update_execution_targets
from .physical import PhysicalRunner, perturbation
from .plan import execute_calls, digest, plain
from .planner_v11 import assembly_program, normalized_graph, propose, value_features
from .v9_candidates import bind


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2), encoding="utf-8")


def clone_geometry(session):
    model=session.ctx.model
    return digest({name:getattr(model,name).tolist() for name in
                   ("body_pos","body_quat","geom_pos","geom_quat","geom_size","qpos0")})


def twin_checkpoint(session):
    state=session.snapshot()
    state["clone_geometry_sha256"]=clone_geometry(session)
    return state


def restore_twin_checkpoint(session,state):
    if state.get("clone_geometry_sha256") != clone_geometry(session):
        raise ValueError("independent twin geometry differs from deployment checkpoint")
    state=deepcopy(state)
    # MjContext's normal restore deliberately rejects any other XML path.
    # Independently constructed models have a different path but must match
    # the full static geometry before a checked checkpoint transfer is allowed.
    state["physics"]["scene_path"]=session.ctx.scene_path
    session.restore(state)


def observe_boundary(session, stage, completed):
    stage_v7.refresh_visual_observation(session, parts=session.parts)
    return dict(stage=stage, completed=list(completed), held=session.held,
                observation=deepcopy(session.decision_observation),
                robot_joints=session.ctx.arm_qpos.tolist(),
                stage_passes=deepcopy(session.stage_passes))


def discrepancy(expected, actual, position_limit=.006, joint_limit=.15):
    """Only compare valid observations, never treat missing data as agreement."""
    differences = {}; unknown = []
    if expected is None:
        return dict(replan=True, reason="missing_twin_prediction", differences={}, unknown=[])
    for p, row in actual["observation"]["objects"].items():
        ref = expected["observation"]["objects"].get(p, {})
        if not row.get("valid") or not ref.get("valid") or row.get("position_m") is None or ref.get("position_m") is None:
            unknown.append(p); continue
        differences[p] = float(np.linalg.norm(np.asarray(row["position_m"])-ref["position_m"]))
    joint_error = float(np.max(np.abs(np.asarray(actual["robot_joints"])-expected["robot_joints"])))
    changed = actual["held"] != expected["held"] or actual["completed"] != expected["completed"]
    exceeded = any(v > position_limit for v in differences.values()) or joint_error > joint_limit
    # Unknown installed parts can be occluded; missing pending parts require
    # reobservation before generating a fresh plan, rather than an invented pose.
    pending_unknown = [p for p in unknown if p not in actual["completed"] and p != "wipe_tool"]
    return dict(replan=bool(changed or exceeded or pending_unknown),
                reason="state_changed" if changed else "prediction_error" if exceeded else
                       "pending_observation_unknown" if pending_unknown else "within_limits",
                differences=differences, joint_error_rad=joint_error, unknown=unknown,
                position_limit_m=position_limit, joint_limit_rad=joint_limit)


def run_staged(session, *, order, choices, wipe_variant=0, wipe_force=1.5,
               wipe_duration=14., stroke_minimum=.08, completed=(), monitor=None, boundaries=None):
    if stroke_minimum != .08:
        raise ValueError("task stroke requirement is immutable")
    proposal = dict(order=list(order), choices=deepcopy(choices), wipe_variant=wipe_variant,
                    wipe_force=wipe_force, wipe_duration=wipe_duration, stroke_minimum=stroke_minimum)
    done = list(completed)
    boundaries = boundaries if boundaries is not None else []
    if not done:
        session.stage_passes = dict(cleaning_pass=False, assembly_pass=False,
            functional_test_pass=False, final_release_and_retraction_pass=False)

    def boundary(stage):
        nonlocal proposal
        row = observe_boundary(session, stage, done)
        boundaries.append(row)
        if monitor:
            replacement = monitor(session, row, proposal)
            if replacement is not None:
                # Completed actions are never repeated or declared successful
                # through replanning. Only remaining executable choices change.
                for p in done:
                    if p in proposal["choices"] and replacement["choices"][p] != proposal["choices"][p]:
                        raise ValueError("replanning attempted to rewrite a completed action")
                proposal = deepcopy(replacement)

    if "cleaning" not in done:
        _clean(session, wipe_variant, wipe_force, wipe_duration)
        done.append("cleaning"); boundary("cleaning")
    while any(p not in done for p in proposal["order"]):
        p = next(p for p in proposal["order"] if p not in done)
        plan = assembly_program(session, proposal)
        if p.startswith("pin_") or p == "handle":
            _update_execution_targets(plan, session)
        calls = [c for c in plan.calls if c.roles.get("manipulated") == p and not c.id.startswith("accept_")]
        execute_calls(session, plan, calls)
        done.append(p); boundary(p)
    # Final local seating checks remain identical to V9; this experiment does
    # not loosen acceptance after seeing outcomes. Functional tests are required.
    plan = assembly_program(session, proposal)
    execute_calls(session, plan, [c for c in plan.calls if c.id.startswith("accept_") or c.id == "final_home"])
    session.stage_passes["assembly_pass"] = True
    _stroke(session, stroke_minimum)
    for p, y in (("pin_left", -.032), ("pin_right", .032)):
        session.call("inspect", what="pin", part=p, hole_part="end_stop",
                     hole_offset_m=[0., y, 0.], minimum_insertion_depth_m=.006,
                     phase="retained_after_stroke")
    session.stage_passes["final_release_and_retraction_pass"] = bool(
        session.held is None and np.max(np.abs(session.ctx.arm_qpos-HOME)) < .02)
    if not all(session.stage_passes.values()):
        raise SkillFailure("complete functional task predicate failed")
    done.extend(("stroke", "retention")); boundary("finished")
    return Result(True, dict(stage_passes=deepcopy(session.stage_passes), completed=done))


class FrozenValue:
    def __init__(self, checkpoint):
        import torch
        from .value_v6 import RobustProgramNet
        torch.set_num_threads(1)
        self.torch = torch
        checkpoint = Path(checkpoint)
        data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if data["method"] not in ("v6_bce", "v6_bce_pair"):
            raise ValueError("expected frozen V6 numerical program model")
        self.model = RobustProgramNet(data["input_dim"], "none")
        self.model.load_state_dict(data["state_dict"]); self.model.eval()
        self.sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    def score(self, graphs):
        x = self.torch.as_tensor(np.stack([value_features(g) for g in graphs]))
        with self.torch.no_grad():
            return self.torch.sigmoid(self.model(x)).numpy().tolist()


def rollout(seed, proposal, directory, *, domain="online", checkpoint=None,
            completed=(), monitor=None, record=False, level="L1"):
    started = time.perf_counter(); directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    boundaries = []; recorder = None
    with (directory / "console.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        spec, session, scene, targets = stage_v9.make_scene(seed, directory / "scene", role=domain, level=level)
        if checkpoint is not None:
            restore_twin_checkpoint(session,checkpoint)
        initial = deepcopy(session.decision_observation)
        graph = normalized_graph(session, proposal, completed=completed)
        save(directory / "input_graph.json", graph)
        def controller(s, **params):
            return run_staged(s, **params, completed=completed, monitor=monitor, boundaries=boundaries)
        session.full_task_controller = controller
        if record:
            from .video_v7 import TwinRecorder
            recorder = TwinRecorder(session, directory / "execution.mp4", width=480, height=360, frame_stride=5)
            session.rec = recorder; session.ctx.on_control_step = recorder
        plan = bind(session, session.stage_targets, proposal)
        trial = perturbation(seed, 0, domain)
        try:
            result = PhysicalRunner(session, timeout=600.).run(plan, trial, keep_trace=True)
            if recorder: recorder.pause(.5)
        finally:
            if recorder: recorder.close()
        result.update(boundaries=boundaries, initial_observation=initial, proposal=proposal,
                      total_wall_seconds=time.perf_counter()-started, domain=domain,
                      checkpoint_sync="exact_simulator_checkpoint" if checkpoint is not None else "paired_seed_initial_scene",
                      source="independent MuJoCo physical execution", video=record)
        save(directory / "result.json", result)
    return result


class ClosedLoop:
    def __init__(self, seed, expected, value, directory, *, max_replans=1, top_k=4, level="L1"):
        self.seed=seed; self.expected={r["stage"]:r for r in expected}; self.value=value
        self.directory=Path(directory); self.max_replans=max_replans; self.top_k=top_k
        self.level=level; self.events=[]; self.replans=0

    def __call__(self, session, actual, current):
        comparison = discrepancy(self.expected.get(actual["stage"]), actual)
        event = dict(stage=actual["stage"], comparison=comparison, action="continue")
        self.events.append(event)
        if actual["stage"] != "finished" and comparison["replan"]:
            if session.held is not None:
                event["action"]="stop_held_object_requires_recovery"
                save(self.directory / "events.json", self.events)
                raise SkillFailure("replanning requires a released boundary")
            if self.replans >= self.max_replans:
                event["action"]="stop_replan_budget_exhausted"
                save(self.directory / "events.json", self.events)
                raise SkillFailure("closed-loop replan budget exhausted")
            self.replans += 1
            try:
                pool, source = propose(session.decision_observation,completed=actual["completed"])
            except ValueError as exc:
                event.update(action="stop_reobserve_required", reason=str(exc))
                save(self.directory / "events.json", self.events)
                raise SkillFailure(str(exc)) from exc
            for p in pool:
                for part in actual["completed"]:
                    if part in current["choices"]: p["choices"][part] = deepcopy(current["choices"][part])
                # Preserve completed-prefix order while permitting unexecuted pin exchange.
                done = [p for p in current["order"] if p in actual["completed"]]
                p["order"] = done + [x for x in p["order"] if x not in done]
            graphs=[normalized_graph(session,p,completed=actual["completed"]) for p in pool]
            scores=self.value.score(graphs); order=np.argsort(-np.asarray(scores), kind="stable").tolist()
            root=self.directory / f"replan_{self.replans}"
            save(root / "request.json", dict(source=source, pool=pool, scores=scores, order=order,
                 completed=actual["completed"], synchronization="exact simulator checkpoint; not sensor-estimated twin"))
            state=twin_checkpoint(session)
            for i in order[:self.top_k]:
                result=rollout(self.seed,pool[i],root / pool[i]["name"],domain="online",checkpoint=state,
                               completed=actual["completed"],level=self.level)
                if result["success"]:
                    self.expected={r["stage"]:r for r in result["boundaries"]}
                    event.update(action="replan_verified_remaining_suffix", selected=pool[i]["name"],
                                 completed_preserved=actual["completed"])
                    save(self.directory / "events.json",self.events)
                    return pool[i]
            event["action"]="stop_no_verified_suffix"
            save(self.directory / "events.json",self.events)
            raise SkillFailure("no remaining program passed twin validation")
        save(self.directory / "events.json",self.events)
        return None


def run_method(seed, method, root, value, *, k=4, level="L1", record=False):
    started=time.perf_counter(); root=Path(root); root.mkdir(parents=True,exist_ok=True)
    with (root / "generation.log").open("w",encoding="utf-8") as log, contextlib.redirect_stdout(log):
        _,session,_,_=stage_v9.make_scene(seed,root / "decision",role="online",level=level)
        pool,source=propose(session.decision_observation)
        graphs=[normalized_graph(session,p) for p in pool]
    generation_seconds=time.perf_counter()-started
    ranking_start=time.perf_counter()
    scores=value.score(graphs) if method=="value_top_k" else None
    if scores is not None: order=np.argsort(-np.asarray(scores),kind="stable").tolist()
    elif method in ("random_top_k","random_early_stop"):
        order=np.random.default_rng(np.random.SeedSequence([seed,119])).permutation(len(pool)).tolist()
    else: order=list(range(len(pool)))
    ranking_seconds=time.perf_counter()-ranking_start
    budget=len(pool) if method in ("all_twin","random_early_stop") else min(k,len(pool))
    save(root / "request.json",dict(seed=seed,method=method,k=budget,source=source,pool=pool,
        scores=scores,order=order,model_sha256=value.sha256,
        graph_nodes=len(graphs[0]["assembly"]["nodes"]),
        graph_edges=len(graphs[0]["assembly"]["edges"]),
        generation_seconds=generation_seconds,ranking_seconds=ranking_seconds,
        task="V9 frozen functional criteria; V11 stage observation controller"))
    trials=[]; selected=None; expected=None
    for i in order[:budget]:
        result=rollout(seed,pool[i],root / "twins" / pool[i]["name"],level=level)
        trials.append(dict(index=i,name=pool[i]["name"],success=result["success"],
                           wall_seconds=result["total_wall_seconds"],error=result["error"]))
        if result["success"] and selected is None: selected=i; expected=result["boundaries"]
        save(root / "progress.json",dict(trials=trials,selected=selected))
        if selected is not None and method!="all_twin": break
    decision_seconds=time.perf_counter()-started
    deployment=None; events=[]
    if selected is not None:
        monitor=ClosedLoop(seed,expected,value,root / "closed_loop",top_k=k,level=level)
        deployment=rollout(seed,pool[selected],root / "deployment",domain="deployment",
                           monitor=monitor,record=record,level=level)
        events=monitor.events
    summary=dict(seed=seed,method=method,k=budget,pool_size=len(pool),trials=trials,
        selected=pool[selected]["name"] if selected is not None else None,
        twin_success=selected is not None,execution_success=bool(deployment and deployment["success"]),
        execution_error=deployment["error"] if deployment else "no_verified_plan",
        generation_seconds=generation_seconds,ranking_seconds=ranking_seconds,
        decision_seconds=decision_seconds,execution_seconds=deployment["total_wall_seconds"] if deployment else 0.,
        total_wall_seconds=time.perf_counter()-started,events=events,
        note="Actual online measurements; independent perturbed simulation deployment; no hardware")
    save(root / "summary.json",summary)
    return summary
