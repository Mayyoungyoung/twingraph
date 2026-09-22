"""Record one previously selected successful online plan with its ClosedLoop.

Preflight is the default; --run starts a new recorded deployment simulation.
This additional replay is excluded from the timed method comparison. It does
not rescreen the original candidate pool or overwrite any source evidence.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

METHODS=('all_twin','random_top_k','value_top_k','random_early_stop','value_early_stop')
VALUE_METHODS=('value_top_k','value_early_stop')
STAGE_PASSES=('cleaning_pass','assembly_pass','functional_test_pass','fixture_capture_pass',
              'final_seat_pass','final_release_and_retraction_pass')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def loop_k(request,summary,original_k=None):
    """Recover the original initial k, not a full-pool verification budget."""
    method=request['method'];n=int(request['n'])
    progressive=request.get('progressive')
    if method=='value_early_stop':
        if not isinstance(progressive,dict) or 'initial_k' not in progressive:
            raise ValueError('value_early_stop lacks its recorded progressive.initial_k')
        inferred=int(progressive['initial_k'])
        if inferred!=int(progressive.get('batch_size',-1)):
            raise ValueError('inconsistent progressive batch size')
    elif method.endswith('_top_k'):
        inferred=int(request['k'])
    else:
        if original_k is None:
            raise ValueError('this method records total budget N, not original k; supply --original-k from its original invocation')
        inferred=int(original_k)
    if original_k is not None and min(int(original_k),n)!=min(inferred,n):
        raise ValueError('--original-k disagrees with the recorded effective k')
    if not 1<=inferred<=n:
        raise ValueError('expected an original effective k between 1 and pool size')
    expected_budget=inferred if method.endswith('_top_k') else n
    if int(request['k'])!=expected_budget or int(summary['k'])!=expected_budget:
        raise ValueError('source verification budget disagrees with method/k')
    return inferred


def prepare_replay(summary_path,checkpoint_path,*,original_k=None):
    """Pure file validation only: no renderer, tensor model or physics imports."""
    summary_path=Path(summary_path).resolve();checkpoint_path=Path(checkpoint_path).resolve()
    root=summary_path.parent
    summary=load_json(summary_path);request_path=root/'request.json';request=load_json(request_path)
    if not (summary.get('valid') is True and summary.get('twin_success') is True
            and summary.get('execution_success') is True):
        raise ValueError('recording source must have a valid successful twin and deployment')
    method=summary.get('method');selected=summary.get('selected');seed=summary.get('seed')
    if method not in METHODS or method!=request.get('method') or seed!=request.get('seed'):
        raise ValueError('source seed/method mismatch')
    if not isinstance(seed,int) or isinstance(seed,bool): raise ValueError('invalid source seed')
    if not isinstance(selected,str) or re.fullmatch(r'grounded_\d+',selected) is None:
        raise ValueError('source must name an actual grounded selected candidate')
    pool=request.get('pool',[]);n=int(request['n'])
    if len(pool)!=n or int(summary['pool_size'])!=n: raise ValueError('candidate pool size mismatch')
    choices=[p for p in pool if p.get('name')==selected]
    if len(choices)!=1: raise ValueError('selected candidate is absent or duplicated in source pool')
    selected_trials=[t for t in summary.get('trials',[]) if t.get('name')==selected]
    if len(selected_trials)!=1 or selected_trials[0].get('success') is not True:
        raise ValueError('selected candidate lacks a successful original verification trial')
    proposal=choices[0]
    twin_path=root/'twins'/selected/'result.json';twin=load_json(twin_path)
    deployment_path=root/'deployment/result.json';deployment=load_json(deployment_path)
    runtime=summary.get('runtime_sha256')
    if not isinstance(runtime,str) or re.fullmatch(r'[0-9a-f]{64}',runtime) is None:
        raise ValueError('source lacks a frozen runtime hash')
    if request.get('runtime_sha256')!=runtime: raise ValueError('request/runtime mismatch')
    for label,result,domain in [('selected twin',twin,'online'),('deployment',deployment,'deployment')]:
        if not (result.get('valid') is True and result.get('success') is True):
            raise ValueError(f'{label} is not a valid successful source')
        if result.get('runtime_sha256')!=runtime or result.get('seed')!=seed:
            raise ValueError(f'{label} seed/runtime mismatch')
        if result.get('proposal')!=proposal or result.get('domain')!=domain:
            raise ValueError(f'{label} proposal/domain mismatch')
        if result.get('geometry_version')!='printed_functional_assembly_v12':
            raise ValueError(f'{label} geometry is not printed V12')
        if not all(result.get('stage_passes',{}).get(k) is True for k in STAGE_PASSES):
            raise ValueError(f'{label} lacks complete functional task acceptance')
        if not result.get('boundaries') or result['boundaries'][-1].get('stage')!='finished':
            raise ValueError(f'{label} lacks a complete boundary trace')
    model_sha=sha256(checkpoint_path);recorded_model=request.get('model_sha256')
    if recorded_model is not None and recorded_model!=model_sha:
        raise ValueError('checkpoint differs from the model frozen in the source request')
    if method in VALUE_METHODS and recorded_model is None:
        raise ValueError('value-method source lacks a frozen model hash')
    files=dict(summary=summary_path,request=request_path,selected_twin=twin_path,
               deployment=deployment_path,checkpoint=checkpoint_path)
    manifest_path=root/'runtime_sources.json'
    if manifest_path.is_file():
        if load_json(manifest_path).get('sha256')!=runtime:
            raise ValueError('source runtime manifest disagrees with summary')
        files['runtime_manifest']=manifest_path
    config=dict(seed=seed,method=method,n=n,k=loop_k(request,summary,original_k),
        level=request.get('level','L1'),max_replans=1)
    if config['level'] not in ('L0','L1','L2'): raise ValueError('unsupported source level')
    return dict(summary=summary,request=request,proposal=deepcopy(proposal),
        expected=deepcopy(twin['boundaries']),source_deployment=deployment,
        runtime_sha256=runtime,checkpoint_sha256=model_sha,source_checkpoint_hash_recorded=recorded_model is not None,
        source_files={key:dict(path=str(path),sha256=sha256(path)) for key,path in files.items()},
        source_root=str(root),config=config)


def validate_output(out,source_root):
    out=Path(out).resolve();source_root=Path(source_root).resolve()
    if out.exists(): raise FileExistsError('refusing to overwrite any existing replay directory')
    if out==source_root or out.is_relative_to(source_root):
        raise ValueError('recorded replay must be outside the timed source method directory')
    return out


def unchanged_sources(prepared):
    return {name:sha256(item['path'])==item['sha256'] for name,item in prepared['source_files'].items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--original-k',type=int,
        help='Original invocation k; required for all_twin/random_early_stop, whose request stores only budget N')
    parser.add_argument('--run',action='store_true',help='Start physics and recording; otherwise preflight only')
    args=parser.parse_args()
    prepared=prepare_replay(args.summary,args.checkpoint,original_k=args.original_k)
    out=validate_output(args.out,prepared['source_root'])
    from simbench.value.provenance_v12 import fingerprint
    runtime=fingerprint()
    if runtime['sha256']!=prepared['runtime_sha256']:
        raise ValueError('current source/CAD/policies differ from the successful frozen source')
    config=prepared['config'];selected=prepared['proposal']['name']
    protocol=dict(schema='twingraph.v12.additional_selected_recorded_replay.v1',
        additional_recorded_replay=True,excluded_from_timing=True,hardware_experiment=False,
        selection='original summary.selected; no candidate override or rescreening',
        selected=selected,closed_loop=config,runtime_sha256=runtime['sha256'],
        checkpoint_sha256=prepared['checkpoint_sha256'],
        source_checkpoint_hash_recorded=prepared['source_checkpoint_hash_recorded'],
        source_files=prepared['source_files'],launcher_sha256=sha256(__file__),
        domain='deployment',record=True,source_events=prepared['summary'].get('events',[]),
        limitations=['New physical replay, not the original timed trajectory; results may differ',
            'Live ClosedLoop starts from the selected twin predictions, not successful deployment observations',
            'Recorder uses display cameras plus a wrist inset; detector continues using its frozen wrist RGB-D interface',
            'Recorded rollout uses the existing 900-second active recording timeout rather than the 600-second unrecorded timeout',
            'Replanning, if triggered, retains exact simulator checkpoint synchronization; not hardware evidence'])
    if not args.run:
        print(json.dumps(dict(preflight='passed',physics_started=False,output_created=False,
            selected=selected,closed_loop=config,runtime_sha256=runtime['sha256'],out=str(out)),ensure_ascii=False))
        return
    out.mkdir(parents=True,exist_ok=False)
    save_json(out/'protocol.json',protocol)
    save_json(out/'runtime_sources.json',runtime)
    save_json(out/'selected_proposal.json',prepared['proposal'])
    save_json(out/'selected_twin_expected_boundaries.json',prepared['expected'])
    try:
        from simbench.value.graph_value_v12 import ValueRankerV12
        from simbench.value.system_v12 import ClosedLoop,rollout
        value=ValueRankerV12(args.checkpoint)
        if value.sha256!=prepared['checkpoint_sha256']: raise ValueError('checkpoint changed while loading')
        monitor=ClosedLoop(config['seed'],prepared['expected'],value,out/'closed_loop',
            max_replans=config['max_replans'],k=config['k'],n=config['n'],level=config['level'],method=config['method'])
        result=rollout(config['seed'],prepared['proposal'],out/'deployment',domain='deployment',
            monitor=monitor,level=config['level'],record=True)
        input_checks=unchanged_sources(prepared)
        runtime_unchanged=fingerprint()['sha256']==prepared['runtime_sha256']
        result_path=out/'deployment/result.json';video=out/'deployment/execution.mp4'
        report=dict(schema=protocol['schema'],additional_recorded_replay=True,excluded_from_timing=True,
            seed=config['seed'],method=config['method'],selected=selected,
            source_execution_success=True,replay_execution_success=bool(result['success']),
            replay_valid=bool(result.get('valid') and all(input_checks.values()) and runtime_unchanged),
            error=result.get('error'),resource_censored=result.get('resource_censored'),
            closed_loop_config=config,closed_loop_events=monitor.events,
            source_inputs_unchanged=input_checks,runtime_unchanged=runtime_unchanged,
            runtime_sha256=prepared['runtime_sha256'],checkpoint_sha256=prepared['checkpoint_sha256'],
            source_files=prepared['source_files'],result_sha256=sha256(result_path),
            video=str(video),video_sha256=sha256(video) if video.is_file() else None,
            stage_passes=result.get('stage_passes'),
            recorded_replay_wall_seconds=result.get('total_wall_seconds'),
            monitor_wall_seconds=result.get('monitor_wall_seconds'),
            note='Additional recorded replay is excluded from all source timing and success-rate aggregates')
        save_json(out/'recording_summary.json',report)
        print(json.dumps(dict(success=report['replay_execution_success'],valid=report['replay_valid'],
            out=str(out),video=str(video),excluded_from_timing=True),ensure_ascii=False))
    except Exception as exc:
        save_json(out/'recording_failure.json',dict(error_type=type(exc).__name__,error=str(exc),
            additional_recorded_replay=True,excluded_from_timing=True,source_files=prepared['source_files']))
        raise


if __name__=='__main__': main()
