"""Independent-scene and binding regression tests without learned weights."""
import copy
from dataclasses import replace
import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from simbench.value import system_v5 as system
from simbench.value.plan import PlanIR, Call, argument, digest


def make_plan(name):
    call = Call("call0", "estimate_grasp", {"yaws": argument([0.]),
                "height_offset": argument(0.)}, {"manipulated": "pin"})
    return PlanIR(name, [call], 1, dict(id=name, part="pin", execution="program",
        start_state="initial", order=["pin"], choices={"pin": {"yaw": 0., "height": 0.}},
        steps=[dict(skill="estimate_grasp", params={"yaws": [0.], "height_offset": 0.})], task_scope="test",
        semantic_program_id="semantic_" + name, initial_route_index=0),
        protocol="assembly.program.feedback.v2")


def model_session():
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody><body name='pin'><freejoint/><geom type='sphere' size='.01' mass='1'/></body></worldbody></mujoco>")
    data = mujoco.MjData(model)
    return SimpleNamespace(ctx=SimpleNamespace(model=model, data=data))


def test_real_mujoco_instances_do_not_share_state():
    twin, target = model_session(), model_session()
    result = system.assert_independent(twin, target)
    twin.ctx.data.qpos[0] = 123
    twin.ctx.model.geom_friction[:] = 0
    assert target.ctx.data.qpos[0] == 0
    assert np.all(target.ctx.model.geom_friction[:, 0] > 0)
    assert result["distinct_model"] and result["snapshot_transfer"] is False


def test_independence_rejects_shared_model_or_data():
    twin, target = model_session(), model_session()
    target.ctx.model = twin.ctx.model
    with pytest.raises(ValueError, match="independent MjModel"):
        system.assert_independent(twin, target)


def test_semantic_rebinding_preserves_controls_and_route():
    left = make_plan("one")
    left.calls[0].arguments["force"] = argument(3., unit="N")
    right = copy.deepcopy(left)
    right.id = right.prefix["id"] = "different_state"
    right.prefix["start_state"] = "target_start"
    right.prefix["initial_artifacts"] = {"transfer": {"joints": [[1, 2, 3]]}}
    assert digest(system.semantic_payload(left)) == digest(system.semantic_payload(right))
    right.calls[0].arguments["force"].value = 3.1
    assert digest(system.semantic_payload(left)) != digest(system.semantic_payload(right))
    right = copy.deepcopy(left)
    right.prefix["initial_route_index"] = 1
    assert digest(system.semantic_payload(left)) != digest(system.semantic_payload(right))


def test_timeout_cannot_be_positive_evidence_or_shrink_denominator():
    rows = [dict(candidate_id="a", valid=True, success=True, timeout=False),
            dict(candidate_id="a", valid=True, success=True, timeout=True),
            dict(candidate_id="b", valid=True, success=True, timeout=False),
            dict(candidate_id="b", valid=False, success=True, timeout=False)]
    selected, summary = system.choose_candidate(["a", "b"], rows, .75)
    assert selected is None
    assert [r["success_rate"] for r in summary] == [.5, .5]
    assert system.choose_candidate(["b", "a"], rows, .5)[0] == "b"


def test_disturbance_namespaces_are_paired_between_policies_only():
    config = system.SystemConfig(912, render=False)
    first = system.trial_for(config, 0, "twin")
    assert first == system.trial_for(config, 0, "twin")
    target = system.trial_for(config, 0, "target")
    assert first["namespace"] != target["namespace"]
    assert first["friction_scale"] != target["friction_scale"]
    nominal = system.trial_for(replace(config, friction_span=0, gain_span=0), 0, "target")
    assert nominal["friction_scale"] == nominal["actuator_gain_scale"] == 1


def test_rank_uses_raw_logits_and_rejects_scorer_payload_tampering():
    plans = [make_plan(str(i)) for i in range(2)]
    graphs = [{"plan": p.to_dict()} for p in plans]
    result = dict(logits=[50., 51.], scores=[1., 1.], order=["1", "0"],
                  top_k=[dict(candidate_id="1", plan=plans[1].to_dict(),
                              graph=graphs[1], graph_sha256=digest(graphs[1]))])
    assert system.normalize_ranking(result, plans, graphs, 1) == [1, 0]
    result["top_k"][0]["plan"]["calls"][0]["skill"] = "grasp"
    with pytest.raises(ValueError, match="scorer plan differs"):
        system.normalize_ranking(result, plans, graphs, 1)


class FakeStage:
    def __init__(self):
        self.sessions = []
        self.rebinding_starts = []

    def make_scene(self, seed, directory, role):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "scene.xml"
        path.write_text("same independent scene", encoding="utf-8")
        session = model_session()
        session.role = role
        session.parts = ("pin",)
        session.ctx.obj_pos = lambda name: session.ctx.data.qpos[:3].copy()
        self.sessions.append(session)
        return {"seed": seed}, session, path, {"pin": [0, 0, 0]}

    def observed(self, session, targets):
        return dict(objects={"pin": {"position": [0, 0, 0]}}, goals=targets)

    def build_pool(self, session, targets, seed, n, orders, precheck):
        assert orders == [["pin"]] and precheck
        return [make_plan(f"p{i}") for i in range(n)], {"requested": n}

    def rebind_plan(self, session, targets, chosen):
        self.rebinding_starts.append(session.ctx.data.qpos.copy())
        result = copy.deepcopy(chosen)
        result.id = result.prefix["id"] = chosen.id + "_target"
        result.prefix["start_state"] = "target_initial"
        return result


class FakeRunner:
    calls = []

    def __init__(self, session, timeout):
        self.session = session

    def run(self, graph, trial, keep_trace):
        assert keep_trace
        plan = graph["plan"]
        name = plan["id"]
        self.calls.append((self.session.role, name, copy.deepcopy(trial)))
        # A successful twin ends far away from target initialization.
        self.session.ctx.data.qpos[:] = 42
        success = not name.startswith("p1")
        return dict(candidate_id=name, input_graph_sha256=digest(graph),
            trial=trial, trial_sha256=digest(trial), valid=True,
            success=success, timeout=False, final_positions={"pin": [42., 42., 42.]},
            executed_parameters=[dict(skill="actual_mock_dispatch", params={})])


class FakeScorer:
    def rank(self, observation, plans, k):
        graphs = [{"plan": p.to_dict(), "observation": observation} for p in plans]
        return dict(logits=[1., 3., 2.], order=["p1", "p2", "p0"],
            top_k=[dict(candidate_id=plans[i].id, plan=plans[i].to_dict(),
                        graph=graphs[i], graph_sha256=digest(graphs[i])) for i in (1, 2)[:k]],
            seconds={})


def test_actual_pair_orchestration_counts_and_target_initial_state(tmp_path, monkeypatch):
    monkeypatch.setattr(system, "compile_graph", lambda obs, p: {"plan": p.to_dict(), "observation": obs})
    monkeypatch.setattr(system, "validate_graph", lambda g: PlanIR.from_dict(g["plan"]))
    monkeypatch.setattr(system, "fingerprint", lambda s: digest(s.ctx.data.qpos))
    stage = FakeStage()
    FakeRunner.calls = []
    planner = dict(task_text="assemble", source="authored_control", provider="test", model="test",
                   proposed_orders=[["pin"]], raw_response="explicit test fixture")
    config = system.SystemConfig(44, n=3, k=2, validation_repeats=2, target_repeats=2, render=False)
    result = system.run_pair(config, planner, FakeScorer(), tmp_path,
                             stage=stage, runner_factory=FakeRunner)
    topk = json.loads((tmp_path / "top_k/result.json").read_text())
    full = json.loads((tmp_path / "full/result.json").read_text())
    assert topk["selected_candidate_id"] == "p2"
    assert full["selected_candidate_id"] == "p0"
    assert topk["simulation_calls"] == dict(twin_validation=4, target_execution=2)
    assert full["simulation_calls"] == dict(twin_validation=6, target_execution=2)
    assert len(FakeRunner.calls) == 14  # Every full trial ran, even overlapping TopK.
    assert len(stage.sessions) == 6  # Two twins and four independently constructed targets.
    assert all(start[0] == 0 for start in stage.rebinding_starts)
    assert all(row["same_success"] and row["same_final_positions"] for row in result["paired_trial_checks"])
    assert all(row["isolation"]["snapshot_transfer"] is False for row in topk["target_trials"])
    assert topk["success"] and full["success"]
    assert topk["ranking"]["top_k"] == ["p1", "p2"]
    assert topk["seconds"]["decision"] > 0


def test_target_rebinding_failure_counts_against_requested_budget(tmp_path, monkeypatch):
    from simbench.assembly.library import SkillFailure
    monkeypatch.setattr(system, "compile_graph", lambda obs, p: {"plan": p.to_dict(), "observation": obs})
    monkeypatch.setattr(system, "validate_graph", lambda g: PlanIR.from_dict(g["plan"]))
    monkeypatch.setattr(system, "fingerprint", lambda s: digest(s.ctx.data.qpos))
    stage = FakeStage()
    def cannot_bind(*args):
        raise SkillFailure("initial target approach is physically unavailable")
    stage.rebind_plan = cannot_bind
    planner = dict(task_text="assemble", source="authored_control", provider="test", model="test",
                   proposed_orders=[["pin"]], raw_response="explicit test fixture")
    result = system.run_system(system.SystemConfig(44, n=3, k=2, target_repeats=2, render=False),
        planner, FakeScorer(), tmp_path, stage=stage, runner_factory=FakeRunner)
    assert result["selected_candidate_id"] == "p2"
    assert result["target_successes"] == 0 and result["requested_target_trials"] == 2
    assert result["simulation_calls"]["target_execution"] == 0
    assert [row["status"] for row in result["target_trials"]] == ["target_rebinding_failed"] * 2


def test_orchestrator_rejects_changed_target_task_parameters(tmp_path, monkeypatch):
    monkeypatch.setattr(system, "compile_graph", lambda obs, p: {"plan": p.to_dict(), "observation": obs})
    monkeypatch.setattr(system, "validate_graph", lambda g: PlanIR.from_dict(g["plan"]))
    monkeypatch.setattr(system, "fingerprint", lambda s: digest(s.ctx.data.qpos))
    stage = FakeStage()
    original = stage.rebind_plan
    def changed_bind(*args):
        plan = original(*args)
        plan.calls[0].arguments["height_offset"].value = .1
        return plan
    stage.rebind_plan = changed_bind
    planner = dict(task_text="assemble", source="authored_control", provider="test", model="test",
                   proposed_orders=[["pin"]], raw_response="explicit test fixture")
    with pytest.raises(ValueError, match="changed selected task choices"):
        system.run_system(system.SystemConfig(44, n=3, k=2, render=False), planner,
            FakeScorer(), tmp_path, stage=stage, runner_factory=FakeRunner)


def test_real_five_part_graph_rebind_preserves_initial_state_and_selected_order(tmp_path):
    from simbench.value import stage_v5 as stage
    from simbench.assembly.candidates import fingerprint
    _, twin, _, goals = stage.make_scene(61000, tmp_path / "twin", role="twin")
    _, target, _, target_goals = stage.make_scene(61000, tmp_path / "target", role="target")
    system.assert_independent(twin, target)
    initial = fingerprint(twin)
    assert initial == fingerprint(target)
    assert twin.stage_completed == target.stage_completed == ()
    assert all(np.linalg.norm(twin.ctx.obj_pos(part) - goals[part]) > .01 for part in stage.PARTS)
    order = stage.legal_orders()[-1]
    plans, _ = stage.build_pool(twin, goals, 61000, n=1, orders=[order])
    assert plans[0].prefix["order"] == list(order)
    assert fingerprint(twin) == initial
    graph = system.compile_graph(stage.observed(twin, goals), plans[0])
    assert system.validate_graph(graph).id == plans[0].id
    rebound, _ = stage.rebind_plan(target, target_goals, plans[0])
    assert digest(system.semantic_payload(rebound)) == digest(system.semantic_payload(plans[0]))
    assert fingerprint(target) == rebound.prefix["start_state"] == initial
    assert system.compile_graph(stage.observed(target, target_goals), rebound) == graph
