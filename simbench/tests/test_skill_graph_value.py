import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from simbench.value.research_scenarios import program
from simbench.value.skill_graph import compile_graph,validate_graph
from simbench.value.graph_encode import encode_graph,collate_graph
from simbench.value.graph_network import GraphValueNet,GraphConfig
from simbench.value.plan import PlanIR


def example(parts=("part_a",)):
    data=SimpleNamespace(qpos=np.zeros(7),qvel=np.zeros(7),ctrl=np.zeros(7))
    model=SimpleNamespace(geom_size=np.zeros((1,3)),geom_pos=np.zeros((1,3)),geom_friction=np.ones((1,3)))
    session=SimpleNamespace(parts=list(parts),ctx=SimpleNamespace(data=data,model=model))
    targets={p:[.04,i*.07,.85] for i,p in enumerate(parts)}
    choices={p:dict(yaw=0.,height=.003,clearance=1.,force=3.,speed=.006) for p in parts}
    plan=program(session,targets,list(parts),choices)
    obs=dict(robot=dict(joints=[0.]*7,eef=[0.,0.,1.]),
             objects={p:dict(position=[-.2,i*.07,.85],quaternion=[1.,0.,0.,0.],
                      geoms=[dict(type=6,size=[.01,.01,.02],position=[0.,0.,0.])]) for i,p in enumerate(parts)},
             goals=[dict(predicate="seated_released_retracted",manipulated=p,position=t) for p,t in targets.items()])
    return obs,plan


def test_compiler_uses_shared_ports_and_persistent_grasp_relation():
    obs,p=example();g=compile_graph(obs,p)
    assert validate_graph(g).to_dict()==p.to_dict()
    assert any(e["source"]==7 and e["target"]==15 and e["relation"]=="holding" for e in g["edges"])
    assert any(e["source"]==4 and e["target"]==5 and e["relation"]=="data" for e in g["edges"])
    # Legal move->move remains an execution adjacency.
    assert any(e["source"]==11 and e["target"]==12 and e["relation"]=="execution" for e in g["edges"])
    assert next(x for x in g["nodes"][7]["ports"] if x["name"]=="force")["unit"]=="N"
    assert all(x["status"]=="deferred" for n in g["nodes"] for x in n["writes"])
    assert all("value" not in x for n in g["nodes"] for x in n["writes"])


def test_scene_versions_carry_cross_object_dependencies():
    obs,p=example(("a","b"));g=compile_graph(obs,p)
    assert any(e["source"]<18<=e["target"] and e["relation"]=="scene" for e in g["edges"])
    assert len({r["version"] for n in g["nodes"] for r in n["writes"] if r["state"]=="scene"})>2


@pytest.mark.parametrize("field",["plan","edges","nodes","interface_sha256"])
def test_tampered_graph_cannot_execute(field):
    obs,p=example();g=compile_graph(obs,p)
    if field=="plan":g[field]["calls"][-1]["arguments"]["tol"]["value"]*=2
    elif field=="edges":g[field].pop()
    elif field=="nodes":g[field][7]["ports"][1]["value"]=99
    else:g[field]="stale"
    with pytest.raises(ValueError):validate_graph(g)


def test_identity_and_storage_order_are_not_features():
    obs,p=example(("a","b"));before=encode_graph(compile_graph(obs,p))
    mapping={"a":"new_a","b":"new_b"}
    def rename(x):
        if isinstance(x,dict):return {k:rename(v) for k,v in x.items()}
        if isinstance(x,list):return [rename(v) for v in x]
        return mapping.get(x,x) if isinstance(x,str) else x
    new=rename(obs);new["objects"]={mapping[k]:v for k,v in reversed(list(new["objects"].items()))}
    renamed_plan=rename(p.to_dict())
    renamed_plan["prefix"]["choices"]={mapping[k]:v for k,v in renamed_plan["prefix"]["choices"].items()}
    after=encode_graph(compile_graph(new,PlanIR.from_dict(renamed_plan)))
    for k in before:np.testing.assert_array_equal(before[k],after[k])


def test_model_padding_candidate_order_and_parameter_sensitivity():
    torch.manual_seed(9);a,p=example();b,q=example(("a","b"))
    one=encode_graph(compile_graph(a,p));two=encode_graph(compile_graph(b,q))
    model=GraphValueNet(GraphConfig(dropout=0.)).eval()
    with torch.no_grad():
        alone=model(collate_graph([one]))
        batch=model(collate_graph([one,two]))
        reverse=model(collate_graph([two,one]))
    torch.testing.assert_close(alone,batch[:1],rtol=1e-4,atol=1e-6)
    torch.testing.assert_close(batch,reverse.flip(0),rtol=1e-4,atol=1e-6)
    changed=copy.deepcopy(a);changed["goals"][0]["position"][0]+=.02
    enc=encode_graph(compile_graph(changed,p))
    assert not np.array_equal(one["numbers"],enc["numbers"])
    with torch.no_grad(): assert abs(float(model(collate_graph([enc]))-alone))>1e-8


def test_unknown_is_not_zero_and_future_labels_are_ignored():
    obs,p=example();g=compile_graph(obs,p);before=encode_graph(g)
    obs["success"]=True;obs["outcomes"]=[dict(future_pose=[0,0,0])]
    after=encode_graph(compile_graph(obs,p))
    for k in before:np.testing.assert_array_equal(before[k],after[k])
    assert 2 in before["statuses"]
    # Every future producer is earlier than its consumer.
    for node in g["nodes"]:
        for port in node["ports"]:
            if port["source"]:assert port["source"]["call"]<node["index"]
