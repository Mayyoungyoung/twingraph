import hashlib
import json
from pathlib import Path
import pytest
from scripts.record_v12_selected_replay import prepare_replay,validate_output,loop_k,STAGE_PASSES


def fixture(tmp_path,method='value_top_k'):
    root=tmp_path/'source';root.mkdir()
    checkpoint=tmp_path/'frozen.pt';checkpoint.write_bytes(b'preflight fixture; never deserialized')
    checkpoint_sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest();runtime='a'*64
    plan={'name':'grounded_001','order':['carriage'],'choices':{'carriage':{'speed':.02}}}
    progressive={'initial_k':1,'batch_size':1,'max_candidates':2} if method=='value_early_stop' else None
    budget=1 if method.endswith('_top_k') else 2
    request=dict(seed=1604,method=method,n=2,k=budget,level='L1',pool=[dict(plan,name='grounded_000'),plan],
        model_sha256=checkpoint_sha,runtime_sha256=runtime,progressive=progressive)
    summary=dict(seed=1604,method=method,pool_size=2,k=budget,selected='grounded_001',valid=True,
        twin_success=True,execution_success=True,runtime_sha256=runtime,
        trials=[{'name':'grounded_001','success':True}],events=[])
    result=dict(seed=1604,valid=True,success=True,runtime_sha256=runtime,proposal=plan,
        geometry_version='printed_functional_assembly_v12',stage_passes={k:True for k in STAGE_PASSES},
        boundaries=[{'stage':'finished','origin':'twin prediction'}])
    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    write(root/'request.json',request);write(root/'summary.json',summary)
    write(root/'twins/grounded_001/result.json',dict(result,domain='online'))
    write(root/'deployment/result.json',dict(result,domain='deployment',boundaries=[{'stage':'finished','origin':'deployment'}]))
    return root,checkpoint,request,summary,write


def test_uses_selected_twin_predictions_and_original_configuration(tmp_path):
    root,checkpoint,_,_,_=fixture(tmp_path)
    ready=prepare_replay(root/'summary.json',checkpoint)
    assert ready['proposal']['name']=='grounded_001'
    assert ready['expected'][0]['origin']=='twin prediction'
    assert ready['config']==dict(seed=1604,method='value_top_k',n=2,k=1,level='L1',max_replans=1)
    assert set(ready['source_files'])=={'summary','request','selected_twin','deployment','checkpoint'}


@pytest.mark.parametrize('field',['valid','twin_success','execution_success'])
def test_rejects_unsuccessful_source(tmp_path,field):
    root,checkpoint,_,summary,write=fixture(tmp_path);summary[field]=False;write(root/'summary.json',summary)
    with pytest.raises(ValueError,match='successful'): prepare_replay(root/'summary.json',checkpoint)


def test_checkpoint_and_runtime_must_match_saved_evidence(tmp_path):
    root,checkpoint,request,_,write=fixture(tmp_path)
    checkpoint.write_bytes(b'changed')
    with pytest.raises(ValueError,match='checkpoint differs'): prepare_replay(root/'summary.json',checkpoint)
    request['runtime_sha256']='b'*64;write(root/'request.json',request)
    with pytest.raises(ValueError,match='runtime mismatch'): prepare_replay(root/'summary.json',checkpoint)


def test_full_budget_methods_require_original_k_and_progressive_recovers_it(tmp_path):
    root,checkpoint,request,summary,_=fixture(tmp_path,'all_twin')
    with pytest.raises(ValueError,match='original-k'): prepare_replay(root/'summary.json',checkpoint)
    assert prepare_replay(root/'summary.json',checkpoint,original_k=1)['config']['k']==1
    request.update(method='value_early_stop',progressive={'initial_k':1,'batch_size':1});summary['method']='value_early_stop'
    assert loop_k(request,summary)==1
    with pytest.raises(ValueError,match='disagrees'): loop_k(request,summary,original_k=2)


def test_refuses_existing_output_or_output_inside_timed_source(tmp_path):
    root,_,_,_,_=fixture(tmp_path)
    with pytest.raises(FileExistsError): validate_output(root,root)
    with pytest.raises(ValueError,match='outside'): validate_output(root/'new_replay',root)
    assert validate_output(tmp_path/'replay',root)==(tmp_path/'replay').resolve()
