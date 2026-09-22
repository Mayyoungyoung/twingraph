"""Synthetic control-flow fixtures, never reported as physical outcomes."""
import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from simbench.assembly.library import SkillFailure
from simbench.value import system_v12 as system
from scripts.train_value_v12 import evaluate, random_metrics


@pytest.mark.parametrize("method,success_index,expected", [
    ("value_early_stop",2,[0,1,2]), ("value_early_stop",None,[0,1,2,3,4]),
    ("value_early_stop",0,[0]), ("value_top_k",2,[0,1]),
    ("all_twin",0,[0,1,2,3,4])])
def test_online_order_expansion_early_stop_and_fixed_budget(monkeypatch,tmp_path,method,success_index,expected):
    pool=[dict(name=f"p{i}",choices={},order=[]) for i in range(5)]
    session=SimpleNamespace(decision_observation={},planning_cad={},task_version="fixture")
    monkeypatch.setattr(system,"require_frozen_source",lambda:dict(sha256="a"*64))
    monkeypatch.setattr(system,"make_scene",lambda *a,**k:(None,session,None,None))
    monkeypatch.setattr(system,"propose",lambda *a,**k:(deepcopy(pool),{}))
    monkeypatch.setattr(system,"normalized_graph",lambda s,p:dict(input_only=p["name"]))
    score_inputs=[]
    def score(graphs):
        score_inputs.append(deepcopy(graphs))
        return [1.,.9,.8,.7,.6]
    value=SimpleNamespace(score=score,sha256="b"*64)
    calls=[]
    def rollout(seed,proposal,directory,**kwargs):
        i=int(proposal["name"][1:]);domain=kwargs.get("domain","online")
        calls.append((i,domain))
        return dict(valid=True,success=i==success_index,boundaries=[],error=None,total_wall_seconds=.01)
    monkeypatch.setattr(system,"rollout",rollout)
    result=system.run_method(1600,method,tmp_path,value,n=5,k=2)
    assert [i for i,domain in calls if domain=="online"]==expected
    selected=success_index in expected
    assert result["twin_success"] is selected
    assert len([i for i,domain in calls if domain=="deployment"])==int(selected)
    assert len(score_inputs)==int(method.startswith("value_"))
    if score_inputs: assert score_inputs[0]==[dict(input_only=f"p{i}") for i in range(5)]
    if method=="value_early_stop":
        assert result["k"]==5 and result["progressive"]["initial_k"]==2
        batches=result["verification_batches"]
        assert [i for b in batches for i in b["attempted_indices"]]==expected
        assert len(batches)==(len(expected)+1)//2
        assert json.loads((tmp_path/"progress.json").read_text())["verification_batches"]==batches
    if method=="value_top_k": assert result["k"]==2


def test_value_order_is_stable_and_random_baseline_uses_same_permutation():
    value=SimpleNamespace(score=lambda g:[.1,.9,.9,.2])
    scores,order,budget,meta=system.verification_schedule("value_early_stop",[{}]*4,value,1600,2)
    assert order==[1,2,3,0] and budget==4 and meta["batch_size"]==2
    a=system.verification_schedule("random_top_k",[{}]*4,None,1600,2)[1]
    b=system.verification_schedule("random_early_stop",[{}]*4,None,1600,2)[1]
    assert a==b
    expected=np.random.default_rng(np.random.SeedSequence([1600,1291,1])).permutation(4).tolist()
    assert system.verification_schedule("random_early_stop",[{}]*4,None,1600,2,replan=1)[1]==expected


@pytest.mark.parametrize("value", [None,SimpleNamespace(score=lambda g:[np.nan]*3),SimpleNamespace(score=lambda g:[.5])])
def test_progressive_requires_finite_full_pool_score_without_fallback(value):
    with pytest.raises(ValueError):
        system.verification_schedule("value_early_stop",[{}]*3,value,1600,2)


@pytest.mark.parametrize("success_index", [3,None])
def test_closed_loop_expands_same_value_order_and_logs_batches(monkeypatch,tmp_path,success_index):
    pool=[dict(name=f"p{i}",choices={},order=[]) for i in range(5)]
    monkeypatch.setattr(system,"propose",lambda *a,**k:(deepcopy(pool),{}))
    monkeypatch.setattr(system,"normalized_graph",lambda *a,**k:{})
    monkeypatch.setattr(system,"twin_checkpoint",lambda s:{})
    calls=[]
    def trial(seed,proposal,*a,**k):
        i=int(proposal["name"][1:]);calls.append(i)
        return dict(valid=True,success=i==success_index,boundaries=[])
    monkeypatch.setattr(system,"rollout",trial)
    value=SimpleNamespace(score=lambda graphs:[5.,4.,3.,2.,1.])
    monitor=system.ClosedLoop(1600,[],value,tmp_path,k=2,n=5,method="value_early_stop")
    session=SimpleNamespace(held=None,decision_observation={},planning_cad={})
    actual=dict(stage="failed_carriage",observation={"objects":{}},completed=[])
    if success_index is None:
        with pytest.raises(SkillFailure,match="stop_no_verified_suffix"):
            monitor(session,actual,dict(choices={},order=[]))
    else:
        assert monitor(session,actual,dict(choices={},order=[]))["name"]=="p3"
    assert calls==list(range(5 if success_index is None else 4))
    request=json.loads((tmp_path/"replan_1/request.json").read_text())
    progress=json.loads((tmp_path/"replan_1/progress.json").read_text())
    assert request["k"]==5 and request["progressive"]["batch_size"]==2
    assert [i for b in progress["verification_batches"] for i in b["attempted_indices"]]==calls


@pytest.mark.parametrize("success_index", [3,None])
def test_offline_progressive_full_pool_coverage_and_recorded_costs(success_index):
    rows=[dict(seed=1600,source="fixture",name=f"p{i}",y=float(i==success_index),
               outcomes=[dict(condition="fixture",success=i==success_index,seconds=i+1.)]) for i in range(5)]
    result=evaluate(rows,[5.,4.,3.,2.,1.],k=2)
    case=result["cases"][0]
    expected_calls=5 if success_index is None else 4
    assert case["success"] is False
    assert case["value_early_stop"]["success"]==case["all_twin"]["success"]==(success_index is not None)
    assert case["value_early_stop"]["calls"]==expected_calls
    assert case["value_early_stop"]["seconds"]==sum(range(1,expected_calls+1))
    assert case["random_early_stop"]==random_metrics([i==success_index for i in range(5)],list(range(1,6)),5)
