"""Value screening invariants; independent of learned task performance."""
import copy
from dataclasses import asdict
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from simbench.value.plan import Argument,Call,PlanIR,argument,from_pick
from simbench.value.encode import encode_plan,collate,sample_path
from simbench.value.network import ModelConfig,PlanValueNet
from simbench.value.losses import probability_loss,keep_loss
from simbench.value.dataset import split_name
from simbench.value.evaluate import subset_metrics
from simbench.value.rank import PlanRanker,select_and_validate


def example():
    obj=dict(position=[.1,.2,.8],quaternion=[1,0,0,0],geoms=[])
    obs=dict(robot=dict(joints=[0]*7),objects={"part":obj},goals=[dict(manipulated="part",position=[0,0,.85])])
    payload=dict(id="candidate",part="part",start_state="snapshot",grasp=dict(id="grasp",xyz=[0,0,.8],yaw=0.,width=.02,cost=123),
                 path=dict(joints=[[0]*7,[.1]*7],id="route",cost=321),control=dict(force=3.),terminal=dict(xyz=[0,0,.85]),
                 steps=[dict(skill="move",params=dict(target=[0,0,.8]))],status="necessary_pass",cost=987,bindings={},unknown=[])
    c=SimpleNamespace(id="candidate",part="part",status="necessary_pass",to_dict=lambda:copy.deepcopy(payload))
    suffix=[Call("suffix","place",dict(part=argument("part"),target=Argument([0,0,.85],"position",frame="world")),{"manipulated":"part"})]
    return obs,from_pick(c,suffix)


def model():
    torch.manual_seed(9)
    return PlanValueNet(ModelConfig(width=32,layers=2,heads=4,dropout=0,vision=False)).eval()


def test_plan_roundtrip_bindings_and_same_skill_edges():
    obs,p=example()
    restored=PlanIR.from_dict(p.to_dict())
    assert restored.to_dict()==p.to_dict()
    restored.calls[1].skill="move"
    assert restored.edges==[("prefix_0","suffix")]
    bad=p.to_dict()
    bad["prefix"]["grasp"]["xyz"][0]=9
    with pytest.raises(ValueError,match="disagree"):
        PlanIR.from_dict(bad)
    with pytest.raises(ValueError,match="frame"):
        Argument([0,0,0],kind="position").validate()


def test_prefix_cannot_see_suffix_even_indirectly():
    obs,p=example()
    changed=copy.deepcopy(p)
    changed.calls[1].arguments["target"].value=[9,8,7]
    other=copy.deepcopy(obs)
    other["goals"][0]["position"]=[1,2,3]
    net=model()
    with torch.no_grad():
        a=net(collate([encode_plan(obs,p)]))
        b=net(collate([encode_plan(other,changed)]))
    torch.testing.assert_close(a["prefix_logit"],b["prefix_logit"],atol=1e-6,rtol=1e-6)
    assert not torch.allclose(a["suffix_logit"],b["suffix_logit"])


def test_variable_length_batch_and_padding_values_are_inert():
    obs,p=example()
    long=copy.deepcopy(p)
    for i in range(5):
        call=copy.deepcopy(p.calls[-1])
        call.id=f"more_{i}"
        long.calls.append(call)
    a,b=encode_plan(obs,p),encode_plan(obs,long)
    net=model()
    batch=collate([a,b])
    with torch.no_grad():
        single=net(collate([a]))
        original=net(batch)
        batch["numbers"][batch["padding"]]=999
        batch["keys"][batch["padding"]]=333
        batch["stages"][batch["padding"]]=1
        modified=net(batch)
    for key in ("q","prefix_logit","suffix_logit"):
        torch.testing.assert_close(original[key][:1],single[key],atol=2e-6,rtol=2e-6)
        torch.testing.assert_close(original[key],modified[key],atol=2e-6,rtol=2e-6)


def test_costs_ids_outcomes_do_not_enter_features():
    obs,p=example()
    before=encode_plan(obs,p)
    p.id="different trace id"
    p.prefix["cost"]=-9999
    p.prefix["grasp"]["cost"]=9999
    p.calls[0].arguments["bound_grasp"].value["cost"]=9999
    obs["outcomes"]={"success":True}
    after=encode_plan(obs,p)
    for key in before:
        np.testing.assert_array_equal(before[key],after[key])


def test_object_order_does_not_change_tokens():
    obs,p=example()
    obs["objects"]["other"]=dict(position=[0,1,0],quaternion=[1,0,0,0],geoms=[])
    before=encode_plan(obs,p)
    obs["objects"]=dict(reversed(list(obs["objects"].items())))
    after=encode_plan(obs,p)
    for key in before:
        np.testing.assert_array_equal(before[key],after[key])


def test_conditional_loss_ignores_failed_prefixes_and_counts_trials():
    a=torch.zeros(3,requires_grad=True)
    b=torch.zeros(3,requires_grad=True)
    loss=probability_loss(dict(prefix_logit=a,suffix_logit=b),torch.tensor([4,4,4]),torch.tensor([0,4,2]),torch.tensor([0,3,1]))
    loss.backward()
    assert b.grad[0]==0 and a.grad[0]!=0 and b.grad[1]!=0
    with pytest.raises(ValueError):
        probability_loss(dict(prefix_logit=a,suffix_logit=b),torch.tensor([1,1,1]),torch.tensor([0,0,0]),torch.tensor([1,0,0]))


def test_keep_loss_boundary_and_all_failure_groups():
    z=torch.tensor([0.,1.,2.,3.],requires_grad=True)
    loss=keep_loss(z,torch.tensor([1.,0.,0.,0.]),torch.zeros(4,dtype=torch.long),k=2,margin=.1)
    assert float(loss)==pytest.approx(2.1)
    loss.backward()
    assert z.grad[0]<0 and z.grad[2]>0 and z.grad[3]==0
    assert keep_loss(z,torch.zeros(4),torch.zeros(4,dtype=torch.long),k=2)==0
    assert subset_metrics([0.,0.],[0])["hit"] is None
    assert split_name("same_configuration")==split_name("same_configuration")


def test_rank_returns_original_plans_and_budget_is_exact(tmp_path):
    obs,p=example()
    net=model()
    file=tmp_path/"model.pt"
    torch.save(dict(schema="twingraph.value.v1",model_config=asdict(net.config),state_dict=net.state_dict(),objective="dual",protocol=p.protocol),file)
    ranker=PlanRanker(file)
    p2=copy.deepcopy(p)
    p2.id="second"
    p2.status="unknown"
    ranked=ranker.rank(obs,[p,p2],k=5)
    assert ranked["returned_k"]==2 and ranked["top_k"][0]["plan"] in (p.to_dict(),p2.to_dict())
    calls=[]
    def reject(plan):
        calls.append(plan.id)
        return False
    result=select_and_validate(ranked,reject,budget=1)
    assert len(calls)==1 and result["chosen"] is None
    assert "infeasible" not in result["status"]
    with pytest.raises(ValueError):
        ranker.rank(obs,[p],k=0)


def test_path_sampling_keeps_endpoints():
    path=[[0,0],[1,0],[1,1]]
    sampled=sample_path(path,5)
    np.testing.assert_equal(sampled[0],path[0])
    np.testing.assert_equal(sampled[-1],path[-1])
