"""A failed skill must remain pending across a monitored safe restart."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest

from simbench.assembly.control import HOME
from simbench.assembly.library import SkillFailure
from simbench.value import system_v12 as system


@pytest.mark.parametrize("held", [None, "carriage"])
def test_failed_stage_replans_only_at_released_state_without_repeating_prefix(monkeypatch, tmp_path, held):
    choices = {"carriage": {"force": 3.}}
    session = SimpleNamespace(planning_cad={"functional_stroke_minimum_m": .02}, held=held,
        stage_passes=dict(cleaning_pass=True, assembly_pass=False, functional_test_pass=False,
                          fixture_capture_pass=False, final_release_and_retraction_pass=False),
        ctx=SimpleNamespace(arm_qpos=np.array(HOME)), artifacts={},
        stage_targets={"carriage": [.105, .085, .824], "handle": [.105, .085, .872]},
        pin_insertion_config=SimpleNamespace(required_depth_m=.006))
    # Isolate safe restart semantics from camera/CAD binding and persistence.
    session.out = tmp_path
    session.decision_observation = {"sha256": "software-recovery-fixture"}
    session.receiver_binding_history = [{"source": "software fixture"}]
    monkeypatch.setattr(system, "_prepare_receiver_targets", lambda *args: session.decision_observation)
    commands, executed, events = [], [], []
    session.call = lambda *args, **kwargs: commands.append((args, kwargs))
    monkeypatch.setattr(system, "assembly_program", lambda s,p: SimpleNamespace(to_dict=lambda: {}, calls=[
        SimpleNamespace(id="place_carriage", roles={"manipulated":"carriage"}, force=p["choices"]["carriage"]["force"])]))
    def execute(s, plan, calls):
        if not calls: return
        executed.append(calls[0].force)
        if len(executed) == 1: raise SkillFailure("tactile grasp was not established")
    monkeypatch.setattr(system, "execute_calls", execute)
    monkeypatch.setattr(system, "observe_boundary", lambda s,stage,done: dict(stage=stage, completed=list(done),held=s.held))
    monkeypatch.setattr(system, "_stroke", lambda s,n: s.stage_passes.update(functional_test_pass=True))
    import simbench.assembly.skills_v12 as skills
    monkeypatch.setattr(skills, "evaluate_end_stop_fixture", lambda s: (True, {"test":True}))
    monkeypatch.setattr(skills, "evaluate_functional_seat", lambda s,part,target: (True, {"test":True}))
    def monitor(s, actual, current):
        events.append(deepcopy(actual))
        if not actual["stage"].startswith("failed_"): return
        assert actual["completed"] == ["cleaning"]
        if s.held is not None: raise SkillFailure("held object needs explicit recovery")
        replacement = deepcopy(current); replacement["choices"]["carriage"]["force"] = 6.
        return replacement
    if held:
        with pytest.raises(SkillFailure, match="explicit recovery"):
            system.run_staged(session,order=["carriage"],choices=choices,completed=["cleaning"],monitor=monitor)
        assert commands == [] and executed == [3.]
    else:
        result = system.run_staged(session,order=["carriage"],choices=choices,completed=["cleaning"],monitor=monitor)
        assert result.ok and executed == [3.,6.]
        assert commands[0] == (("move",), {"target":"home"})
        assert events[0]["stage"] == "failed_carriage" and "failure" in events[0]
        assert events[-1]["completed"] == ["cleaning","carriage","stroke","retention"]


@pytest.mark.parametrize("method, expected", [("all_twin",["a","b","c"]),("value_top_k",["b"])])
def test_replanning_preserves_method_budget_and_ranking(monkeypatch, tmp_path, method, expected):
    pool=[dict(name=name,choices={},order=[]) for name in ("a","b","c")]
    monkeypatch.setattr(system,"propose",lambda *args,**kwargs:(deepcopy(pool),{}))
    monkeypatch.setattr(system,"normalized_graph",lambda *args,**kwargs:{})
    monkeypatch.setattr(system,"twin_checkpoint",lambda s:{})
    calls=[]
    def trial(seed,proposal,*args,**kwargs):
        calls.append(proposal["name"])
        return dict(valid=True,success=True,boundaries=[])
    monkeypatch.setattr(system,"rollout",trial)
    value=SimpleNamespace(score=lambda graphs:[.1,.9,.2])
    monitor=system.ClosedLoop(1600,[],value,tmp_path,k=1,n=3,method=method)
    session=SimpleNamespace(held=None,decision_observation={},planning_cad={})
    actual=dict(stage="failed_carriage",observation={"objects":{}},completed=[])
    chosen=monitor(session,actual,dict(choices={},order=[]))
    assert calls==expected and chosen["name"]==expected[0]
