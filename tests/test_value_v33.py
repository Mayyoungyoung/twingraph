from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pytest
from scripts.value_features_v33 import encode_candidate,FEATURES,BASE_FEATURES


@pytest.fixture
def frozen():
    roots=[Path('results/dataset_1000'),Path('results/v32_remote/results/dataset_1000')]
    path=next((r/'seed_3334/request.json' for r in roots if (r/'seed_3334/request.json').exists()),None)
    if path is None:pytest.skip('audited frozen integration fixture unavailable')
    return json.loads(path.read_text(encoding='utf-8'))


def test_geometry_is_present_and_outcome_metadata_cannot_change_features(frozen):
    entry=frozen['candidates'][0];a=encode_candidate(frozen,entry)
    assert a['x'].shape[1]==len(FEATURES)
    assert a['x'][:,FEATURES.index('geometry_available')].any()
    altered=deepcopy(frozen);altered['seed']=999999;altered['success']=False
    altered['candidates'][0]['name']='irrelevant';altered['candidates'][0]['actual_result']={'success':True}
    b=encode_candidate(altered,altered['candidates'][0])
    for key in a:np.testing.assert_array_equal(a[key],b[key])
    base=encode_candidate(frozen,entry,False)
    np.testing.assert_array_equal(base['x'],a['x'][:,:len(BASE_FEATURES)])


def test_parameter_binding_detects_geometry_or_executable_tampering(frozen):
    bad=deepcopy(frozen['candidates'][0]);bad['proposal']['necessary_geometry']['carriage']['yaw']+=.1
    with pytest.raises(ValueError,match='geometry/plan'):encode_candidate(frozen,bad)
    bad=deepcopy(frozen['candidates'][0]);bad['assembly_plan_sha256']='bad'
    with pytest.raises(ValueError,match='hash'):encode_candidate(frozen,bad)


def test_top3_selection_counts_unsolvable_layout_and_independent_deployment():
    from scripts.train_value_v33 import validation_metrics
    rows=[]
    for seed in (1,2):
        for i in range(8):rows.append(dict(id=f'{seed}_{i}',seed=seed,label=int(seed==1 and i==1),
            robust=.25 if seed==1 and i==1 else 0.,nominal_steps=100))
    pred=np.tile(np.array([.5,.5,100.]),(1,16,1))
    metric=validation_metrics(rows,pred)
    assert metric['workflow_success']==.125
    assert metric['calls']==2.5 and metric['steps']==250


def test_soft_robust_training_smoke(tmp_path):
    pytest.importorskip('torch')
    from scripts.train_value_v33 import fit,temperature_scale
    from simbench.value.generic_graph_value_v15 import RELATIONS
    rows=[]
    for seed in range(4):
        for i in range(8):
            e=dict(x=np.random.default_rng(seed*8+i).normal(size=(3,len(FEATURES))).astype(np.float32),
                relations=np.zeros((len(RELATIONS),3,3),np.float32),active=np.ones(3,np.float32))
            rows.append(dict(id=f'{seed}_{i}',seed=seed,encoded=e,label=i%2,robust=(i%4)/3,
                nominal_steps=100+i,log_steps=np.log1p(100+i)))
    path,pred=fit(rows[:16],rows[16:],'enhanced_robust',7,tmp_path,max_epochs=1,min_epochs=1,patience=1)
    assert path.exists() and pred.shape==(16,3) and np.isfinite(pred).all()
    scaled=temperature_scale(pred,[1.3,1.2]);np.testing.assert_array_equal(scaled[:,2],pred[:,2])


def test_three_scenario_top3_budget_and_idempotent_resume(tmp_path,monkeypatch,frozen):
    pytest.importorskip('torch')
    from scripts import evaluate_workflow_v33 as wf
    seed=frozen['seed'];source=tmp_path/'source';folder=source/f'seed_{seed}';folder.mkdir(parents=True)
    (folder/'request.json').write_text(json.dumps(frozen))
    training=tmp_path/'training';training.mkdir();(training/'selection_frozen.json').write_text('{}')
    monkeypatch.setattr(wf,'load_ranker',lambda path:({},[]))
    monkeypatch.setattr(wf,'encode_candidates',lambda request:[])
    monkeypatch.setattr(wf,'score_candidates',lambda *a:(-np.arange(8),np.ones((8,3)),np.zeros(8)))
    calls=[]
    def fake(request,entry,out,repeat,runtime):
        calls.append((entry['name'],repeat));return dict(name=entry['name'],success=entry['name']=='llm_003' and repeat%10==0,
            seconds=1.,plan_sha256=entry['plan_sha256'],repeat=repeat)
    monkeypatch.setattr(wf,'rollout',fake)
    job=(str(source),str(training),str(tmp_path/'workflow'),seed)
    result=wf.evaluate_layout(job);assert 'error' not in result
    for scenario in result['scenarios']:
        assert scenario['policies']['value_top3']['twin_calls']==3
        assert scenario['policies']['full8']['twin_calls']==8
        assert scenario['policies']['random_top3']['twin_calls']<=3
        assert all(not r['workflow_success'] for r in scenario['policies'].values())
    assert {r for _,r in calls if r%10==1}=={21,31,41}
    n=len(calls);assert wf.evaluate_layout(job)==result and len(calls)==n
