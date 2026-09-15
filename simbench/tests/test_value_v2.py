import copy
import pytest
import numpy as np
from simbench.tests.test_plan_value import example
from simbench.value.validation import select_and_validate
from simbench.value.physical import perturbation
from simbench.value.encode import encode_plan


def ranking(k=2,n=3):
    _,p=example();rows=[]
    for i in range(n):
        q=copy.deepcopy(p);q.id=str(i);q.prefix["id"]=q.id
        rows.append(dict(candidate_id=q.id,score=1-i*.1,plan=q.to_dict()))
    return dict(ranked=rows,top_k=rows[:k])


def test_no_expand_and_repeats_are_physical_budget():
    calls=[]
    def trial(p):calls.append(p.id);return False
    r=select_and_validate(ranking(k=1),trial,10,repeats=2,allow_expand=False)
    assert calls==["0","0"] and r["validation_calls"]==2 and r["unique_candidates"]==1
    assert r["unused_budget"]==8 and r["chosen"] is None
    assert "infeasible" not in r["status"]


def test_expansion_batches_and_incomplete_block():
    r=select_and_validate(ranking(k=1),lambda p:False,5,repeats=2,allow_expand=True)
    assert r["validation_calls"]==4 and len(r["batches"])==2
    assert r["unused_budget"]==1
    assert [x["repeat"] for x in r["validated"]]==[0,1,0,1]


def test_modes_choose_different_candidates():
    def trial(p):return dict(success=True,sim_seconds=10-int(p.id)*3)
    a=select_and_validate(ranking(),trial,4,repeats=2,mode="first_verified")
    b=select_and_validate(ranking(),trial,4,repeats=2,mode="best_within_budget")
    assert a["chosen"]["id"]=="0" and b["chosen"]["id"]=="1"
    assert a["validation_calls"]==2 and b["validation_calls"]==4


@pytest.mark.parametrize("budget",[0,1,2,3,4,5,7])
def test_budget_bound(budget):
    r=select_and_validate(ranking(),lambda p:True,budget,repeats=2,mode="best_within_budget",allow_expand=True)
    assert r["validation_calls"]<=budget


def test_duplicate_visits_are_rejected():
    r=ranking();r["ranked"].append(r["ranked"][0])
    with pytest.raises(ValueError,match="duplicate"):select_and_validate(r,lambda p:True,3)


def test_invalid_trial_is_never_failure_label():
    with pytest.raises(RuntimeError,match="invalid"):
        select_and_validate(ranking(),lambda p:dict(success=False,valid=False),1)


def test_namespaces_are_distinct_and_paired():
    domains=[perturbation(12,0,d) for d in ("train","online","deployment","reference")]
    assert len({x["friction_scale"] for x in domains})==4
    assert perturbation(12,0,"online")==perturbation(12,0,"online")


def test_object_renaming_preserves_input():
    obs,p=example();before=encode_plan(obs,p)
    old=next(iter(obs["objects"]));new="unseen_binding"
    def rename(x):
        if isinstance(x,dict):return {k:rename(v) for k,v in x.items()}
        if isinstance(x,list):return [rename(v) for v in x]
        return new if isinstance(x,str) and x==old else x
    from simbench.value.plan import PlanIR
    renamed=rename(obs);renamed["objects"][new]=renamed["objects"].pop(old)
    after=encode_plan(renamed,PlanIR.from_dict(rename(p.to_dict())))
    for key in before:np.testing.assert_array_equal(before[key],after[key])


def test_direct_fast_path_matches_full_network():
    import torch
    from simbench.tests.test_plan_value import model
    from simbench.value.encode import collate
    obs,plan=example();net=model();net.eval();batch=collate([encode_plan(obs,plan)])
    with torch.no_grad():
        torch.testing.assert_close(net(batch)["direct_logit"],net(batch,direct_only=True)["direct_logit"])


def test_candidate_reordering_preserves_individual_scores():
    import torch
    from simbench.tests.test_plan_value import model
    from simbench.value.encode import collate
    obs,p=example();q=copy.deepcopy(p)
    q.calls[-1].arguments["target"].value=[.2,.1,.3]
    a,b=encode_plan(obs,p),encode_plan(obs,q)
    net=model().eval()
    with torch.no_grad():
        before=net(collate([a,b]),direct_only=True)["direct_logit"]
        after=net(collate([b,a]),direct_only=True)["direct_logit"]
    torch.testing.assert_close(before,after.flip(0))
    assert not np.array_equal(a["numbers"],b["numbers"])


def test_unknown_validation_mode_and_fraction():
    with pytest.raises(ValueError):select_and_validate(ranking(),lambda p:True,3,mode="implicit")
    with pytest.raises(ValueError):select_and_validate(ranking(),lambda p:True,3,accept_rate=0)
