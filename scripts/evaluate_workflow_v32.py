"""Fresh three-policy workflows on the frozen test layouts, including deployment."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
from copy import deepcopy
import json
from pathlib import Path
import multiprocessing
import time
import traceback
import numpy as np
from scripts.collect_sliding_assembly_v23 import REQUIRED, TASK_VERSION
from scripts.train_value_v32 import read, write, sha, bootstrap
from simbench.assembly.library import DEFAULT_CAPABILITIES
from simbench.value.plan import PlanIR, digest, plain
from simbench.value.physical import PhysicalRunner, perturbation
from simbench.value.provenance_v12 import fingerprint
from simbench.value.system_v12 import make_scene
from simbench.value.skill_graph import compile_graph
from simbench.value.generic_graph_value_v15 import AtomicValueNetV15, encode_graph, collate


POLICIES=('value_top4','random_top4','full8')


def encode_candidates(request):
    observation=deepcopy(request['decision_observation']); observation.setdefault('robot',{})
    observation['goals']=[dict(predicate='assembly_through_handle',both_pins_through_base=True)]
    for part,capabilities in DEFAULT_CAPABILITIES.items():
        if part in observation['objects']: observation['objects'][part]['capabilities']=list(capabilities)
    return [encode_graph({'assembly':compile_graph(observation,PlanIR.from_dict(e['assembly_plan_ir']))}) for e in request['candidates']]


def rollout(request,entry,out,repeat,runtime):
    out.mkdir(parents=True,exist_ok=False); began=time.perf_counter()
    kwargs=dict(domain=request['domain'],level=request['level'],observation_backend='mujoco_state_pose',
        position_noise_std_m=request['position_noise_std_m'],yaw_noise_std_rad=request['yaw_noise_std_rad'])
    _,session,_,_=make_scene(request['seed'],out/'scene',**kwargs)
    if session.decision_observation!=request['decision_observation'] or session.state_observation_random_state!=request['observation_random_state']:
        raise ValueError('fresh observation differs from original frozen request')
    session.sliding_assembly_v23=True; session.end_stop_place_acceptance_v12='stable_supported'; session.required_stage_passes=REQUIRED
    plan=PlanIR.from_dict(entry['complete_candidate_plan_ir']).validate(session.parts)
    if digest(plan.to_dict())!=entry['plan_sha256']: raise ValueError('plan changed')
    trial=perturbation(request['seed'],repeat,request['domain'])
    with (out/'console.log').open('w') as log, contextlib.redirect_stdout(log):
        try: result=PhysicalRunner(session,timeout=1800.).run(plan,trial,keep_trace=True)
        except Exception as exc:
            result=dict(valid=False,success=False,timeout=False,error=f'{type(exc).__name__}: {exc}',
                software_exception=traceback.format_exc(),stage_passes=deepcopy(getattr(session,'stage_passes',{})))
    result.update(candidate_name=entry['name'],source_plan_sha256=entry['plan_sha256'],source_runtime_sha256=request['runtime_sha256'],
        runtime_sha256=runtime,repeat=repeat,trial=trial,task_version=TASK_VERSION,
        base_hole_acceptance=deepcopy(session.artifacts.get('pin_joint_engagement')),
        fresh_total_wall_seconds=time.perf_counter()-began)
    if result.get('timeout'): result.update(valid=False,invalid_reason='resource-censored execution')
    if fingerprint()['sha256']!=runtime: result.update(valid=False,invalid_reason='runtime changed during evaluation')
    accepted=all(result.get('stage_passes',{}).get(k) is True for k in REQUIRED)
    if bool(result['success'])!=accepted: result.update(valid=False,invalid_reason='functional predicate mismatch')
    write(out/'result.json',plain(result))
    if not result['valid']: raise ValueError(f'invalid fresh trial at {out}: {result.get("invalid_reason",result.get("error"))}')
    return dict(name=entry['name'],success=bool(result['success']),seconds=result['fresh_total_wall_seconds'],
        plan_sha256=entry['plan_sha256'],repeat=repeat,result=str(out/'result.json'))


def evaluate_layout(job):
    source,seed,checkpoint_path,out,runtime,allowed=job
    import torch
    torch.set_num_threads(1)
    try:
        out=Path(out); out.mkdir(parents=True,exist_ok=True)
        request_path=Path(source)/f'seed_{seed}'/'request.json'
        request=read(request_path)
        if request['runtime_sha256'] not in allowed or request['task_version']!=TASK_VERSION: raise ValueError('unregistered source runtime/task')
        if len(request['candidates'])!=8: raise ValueError('expected eight frozen candidates')
        loading=time.perf_counter(); saved=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        model=AtomicValueNetV15.build(**saved['model_config']); model.load_state_dict(saved['state_dict']); model.eval()
        load_seconds=time.perf_counter()-loading
        random_order=np.random.default_rng(320927+seed).permutation(8).tolist()
        policy_order=np.random.default_rng(321027+seed).permutation(POLICIES).tolist()
        manifest=dict(seed=seed,checkpoint_sha256=sha(checkpoint_path),request_sha256=digest(request),
            source_runtime_sha256=request['runtime_sha256'],execution_runtime_sha256=runtime,
            random_order=random_order,policy_order=policy_order,validation_repeat=0,deployment_repeat=1,
            original_plan_generation_seconds=request.get('source',{}).get('wall_seconds'),
            planning_mode='Frozen pre-execution plans reused as the common planner stub; no new LLM time claimed')
        if (out/'manifest.json').exists() and read(out/'manifest.json')!=manifest: raise ValueError('workflow manifest changed')
        write(out/'manifest.json',manifest)
        reports={}
        for policy in policy_order:
            folder=out/policy
            if (folder/'summary.json').exists(): reports[policy]=read(folder/'summary.json'); continue
            if folder.exists() and any(folder.iterdir()): raise ValueError(f'interrupted timed policy must be archived before an explicit rerun: {folder}')
            folder.mkdir(exist_ok=True); began=time.perf_counter()
            started=time.perf_counter(); frozen=read(request_path); plan_load_seconds=time.perf_counter()-started
            encode_seconds=inference_seconds=0.; probabilities=None
            started=time.perf_counter()
            if policy=='value_top4':
                encoded=encode_candidates(frozen); encode_seconds=time.perf_counter()-started
                started=time.perf_counter()
                with torch.inference_mode():
                    probabilities=torch.sigmoid(model(collate(encoded))/saved['temperature']).tolist()
                inference_seconds=time.perf_counter()-started
                order=np.argsort(-np.asarray(probabilities),kind='stable').tolist(); budget=4
            elif policy=='random_top4': order=random_order; budget=4
            else: order=list(range(8)); budget=8
            write(folder/'selection.json',dict(policy=policy,order=order,budget=budget,probabilities=probabilities,
                outcome_labels_read_for_ranking=False,selection_frozen_at=time.time()))
            validations=[]; selected=None; validation_start=time.perf_counter()
            for index in order[:budget]:
                entry=frozen['candidates'][index]
                row=rollout(frozen,entry,folder/'validation'/entry['name'],0,runtime); validations.append(row)
                if row['success'] and selected is None: selected=entry
                if selected and policy!='full8': break
                write(folder/'progress.json',dict(validations=validations,selected=selected['name'] if selected else None))
            validation_seconds=time.perf_counter()-validation_start
            deployment=None; deployment_start=time.perf_counter()
            if selected: deployment=rollout(frozen,selected,folder/'deployment'/selected['name'],1,runtime)
            deployment_seconds=time.perf_counter()-deployment_start if selected else 0.
            report=dict(policy=policy,seed=seed,valid=True,verification_hit=selected is not None,
                workflow_success=bool(deployment and deployment['success']),twin_calls=len(validations),
                selected=selected['name'] if selected else None,plan_load_seconds=plan_load_seconds,
                encoding_seconds=encode_seconds,inference_seconds=inference_seconds,model_load_seconds=load_seconds if policy=='value_top4' else 0.,
                validation_seconds=validation_seconds,deployment_seconds=deployment_seconds,
                workflow_wall_seconds=time.perf_counter()-began,validations=validations,deployment=deployment)
            write(folder/'summary.json',report); reports[policy]=report
        write(out/'summary.json',dict(seed=seed,policies=reports))
        return dict(seed=seed,policies=reports)
    except Exception as exc:
        result=dict(seed=seed,error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        write(Path(out)/'error.json',result); return result


def summarize(rows):
    good=[r for r in rows if 'error' not in r]; metrics={}
    for policy in POLICIES:
        values=[r['policies'][policy] for r in good]
        if not values: continue
        metrics[policy]=dict(layouts=len(values),successes=sum(r['workflow_success'] for r in values),
            workflow_success=bootstrap([r['workflow_success'] for r in values]),
            verification_hit=float(np.mean([r['verification_hit'] for r in values])),
            mean_twin_calls=float(np.mean([r['twin_calls'] for r in values])),
            seconds={k:dict(mean=float(np.mean([r[k] for r in values])),median=float(np.median([r[k] for r in values])))
                for k in ('plan_load_seconds','encoding_seconds','inference_seconds','model_load_seconds','validation_seconds','deployment_seconds','workflow_wall_seconds')})
    paired={}
    if good:
        for baseline in ('random_top4','full8'):
            paired[baseline]=dict(success_difference=bootstrap([r['policies']['value_top4']['workflow_success']-r['policies'][baseline]['workflow_success'] for r in good]),
                seconds_difference=bootstrap([r['policies']['value_top4']['workflow_wall_seconds']-r['policies'][baseline]['workflow_wall_seconds'] for r in good]))
    return dict(complete=len(good)==13 and not any('error' in r for r in rows),completed_layouts=len(good),
        metrics=metrics,paired_value_minus_baseline=paired,rows=rows,
        timing_scope='Fresh measured wall time from frozen-plan loading through sequential verification and repeat-1 deployment; new LLM generation excluded; model cold load reported separately; up to four concurrent layouts.')


def evaluate(source,checkpoint,out,workers=4):
    manifest=read(source/'manifest.json'); runtime=fingerprint(); active=manifest.get('active_runtime_sha256',manifest['runtime_sha256'])
    if runtime['sha256']!=active: raise ValueError('active runtime mismatch')
    original=read(source/'revisions/original/runtime.json')
    delta={p for p in set(original['files'])|set(runtime['files']) if original['files'].get(p)!=runtime['files'].get(p)}
    if delta!={'simbench/assembly/placement_catalog_v13.py'}: raise ValueError('physical/encoder changes invalidate compatibility')
    frozen=read(checkpoint.parent/'selection_frozen.json')
    if sha(checkpoint)!=frozen['checkpoint_sha256']: raise ValueError('checkpoint changed after freeze')
    test_seeds=[j['seed'] for j in manifest['jobs'] if j['split']=='test']
    if len(test_seeds)!=13: raise ValueError('wrong held-out layout count')
    out.mkdir(parents=True,exist_ok=True)
    allowed=[original['sha256'],runtime['sha256']]
    jobs=[(str(source),s,str(checkpoint),str(out/f'seed_{s}'),runtime['sha256'],allowed) for s in test_seeds]
    rows=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(evaluate_layout,job) for job in jobs]):
            row=future.result();rows.append(row);write(out/'comparison.json',summarize(rows))
            print(json.dumps(dict(event='workflow_layout_complete',seed=row['seed'],error=row.get('error'))),flush=True)
    report=summarize(rows); write(out/'comparison.json',report)
    if not report['complete']: raise ValueError('workflow evaluation incomplete; see comparison.json')
    return report


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();evaluate(a.source,a.checkpoint,a.out,a.workers)
