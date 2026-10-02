import json
import numpy as np
import pytest

torch=pytest.importorskip('torch')
from scripts import evaluate_workflow_v32 as workflow
from simbench.value.generic_graph_value_v15 import AtomicValueNetV15,FEATURES,RELATIONS


def test_rollout_serializes_nested_numpy_physics_trace(tmp_path,monkeypatch):
    from types import SimpleNamespace
    observation={'objects':{}}
    session=SimpleNamespace(decision_observation=observation,state_observation_random_state={},
        parts={},artifacts={'pin_joint_engagement':{'position':np.array([1.,2.,3.])}})
    monkeypatch.setattr(workflow,'make_scene',lambda *a,**k:(None,session,None,None))
    plan=SimpleNamespace(validate=lambda parts:plan,to_dict=lambda:{})
    monkeypatch.setattr(workflow,'PlanIR',SimpleNamespace(from_dict=lambda value:plan))
    result=dict(valid=True,success=True,timeout=False,stage_passes={k:True for k in workflow.REQUIRED},
        trace=[{'pose':{'xyz':np.array([0.1,0.2,0.3]),'accepted':np.bool_(True)}}])
    monkeypatch.setattr(workflow,'PhysicalRunner',lambda *a,**k:SimpleNamespace(run=lambda *a,**k:result))
    monkeypatch.setattr(workflow,'fingerprint',lambda:{'sha256':'runtime'})
    monkeypatch.setattr(workflow,'perturbation',lambda *a:{'noise':np.float64(0.0)})
    request=dict(seed=3300,domain='id',level='test',position_noise_std_m=0.,yaw_noise_std_rad=0.,
        decision_observation=observation,observation_random_state={},runtime_sha256='runtime')
    entry=dict(name='llm_000',complete_candidate_plan_ir={},plan_sha256=workflow.digest({}))
    folder=tmp_path/'rollout'
    actual=workflow.rollout(request,entry,folder,0,'runtime')
    saved=json.loads((folder/'result.json').read_text())
    assert actual['success'] is True and saved['valid'] is True
    assert saved['trace'][0]['pose']=={'xyz':[0.1,0.2,0.3],'accepted':True}
    assert saved['base_hole_acceptance']['position']==[1.,2.,3.]


def test_three_policies_use_fresh_deployment_and_resume_without_duplicate_trials(tmp_path,monkeypatch):
    source=tmp_path/'source';layout=source/'seed_3300';layout.mkdir(parents=True)
    request=dict(seed=3300,runtime_sha256='original',task_version=workflow.TASK_VERSION,
        candidates=[dict(name=f'llm_{i:03d}',plan_sha256=f'plan{i}') for i in range(8)])
    (layout/'request.json').write_text(json.dumps(request))
    model=AtomicValueNetV15.build(width=4,message_layers=0);checkpoint=tmp_path/'model.pt'
    torch.save(dict(model_config=model.config,state_dict=model.state_dict(),temperature=1.),checkpoint)
    encoded=dict(x=np.zeros((2,len(FEATURES)),np.float32),relations=np.zeros((len(RELATIONS),2,2),np.float32),active=np.ones(2,np.float32))
    monkeypatch.setattr(workflow,'encode_candidates',lambda request:[encoded for _ in request['candidates']])
    calls=[]
    def fake_rollout(request,entry,out,repeat,runtime):
        calls.append((entry['name'],repeat))
        return dict(name=entry['name'],success=repeat==0 and entry['name']=='llm_001',seconds=1.,repeat=repeat)
    monkeypatch.setattr(workflow,'rollout',fake_rollout)
    job=(str(source),3300,str(checkpoint),str(tmp_path/'eval'),'current',['original','current'])
    result=workflow.evaluate_layout(job)
    assert 'error' not in result
    policies=result['policies']
    assert policies['full8']['twin_calls']==8
    assert policies['value_top4']['twin_calls']==2
    assert policies['value_top4']['verification_hit'] is True
    assert all(not r['workflow_success'] for r in policies.values())
    assert any(repeat==1 for _,repeat in calls)
    count=len(calls)
    second=workflow.evaluate_layout(job)
    assert 'error' not in second and len(calls)==count
