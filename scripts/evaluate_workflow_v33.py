"""Three paired Top3/Full8 workflows on 50 new, fully frozen test layouts."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing
from pathlib import Path
import time,traceback
import numpy as np
from scripts.train_value_v32 import read,write,sha,bootstrap
from scripts.train_value_v33 import load_ranker,score_candidates
from scripts.value_features_v33 import encode_candidates
from scripts.evaluate_workflow_v32 import rollout
from scripts.collect_robust_value_v33 import RUNTIME
from simbench.value.plan import digest
from simbench.value.provenance_v12 import fingerprint

POLICIES=('value_top3','random_top3','full8')
SCENARIOS=((20,21),(30,31),(40,41))


def evaluate_layout(job):
    import torch
    torch.set_num_threads(1)
    source,training,root,seed=map(lambda x:x,job);out=Path(root)/f'seed_{seed}';out.mkdir(parents=True,exist_ok=True)
    try:
        path=Path(source)/f'seed_{seed}'/'request.json';request=read(path)
        if request['runtime_sha256']!=RUNTIME or len(request['candidates'])!=8:raise ValueError('test request mismatch')
        started=time.perf_counter();frozen,models=load_ranker(Path(training));cold=time.perf_counter()-started
        manifest=dict(seed=seed,request_sha256=digest(request),model_freeze_sha256=sha(Path(training)/'selection_frozen.json'),
            scenarios=SCENARIOS,original_plan_generation_seconds=request['source'].get('wall_seconds'))
        # JSON stores tuples as lists; compare the same serialized representation.
        manifest['scenarios']=[list(x) for x in SCENARIOS]
        if (out/'manifest.json').exists() and read(out/'manifest.json')!=manifest:raise ValueError('evaluation manifest changed')
        write(out/'manifest.json',manifest);scenarios=[]
        for scenario,(verify_repeat,deploy_repeat) in enumerate(SCENARIOS):
            reports={};scenario_dir=out/f'scenario_{scenario}';scenario_dir.mkdir(exist_ok=True)
            policy_order=np.random.default_rng(331027+seed*7+scenario).permutation(POLICIES).tolist()
            random_order=np.random.default_rng(330927+seed*7+scenario).permutation(8).tolist()
            for policy in policy_order:
                folder=scenario_dir/policy
                if (folder/'summary.json').exists():reports[policy]=read(folder/'summary.json');continue
                if folder.exists() and any(folder.iterdir()):raise ValueError(f'interrupted timed policy must be explicitly archived: {folder}')
                folder.mkdir(exist_ok=True);began=time.perf_counter();start=time.perf_counter();current=read(path);load=time.perf_counter()-start
                encoding=inference=0.;pred=unc=scores=None
                if policy=='value_top3':
                    start=time.perf_counter();encoded=encode_candidates(current);encoding=time.perf_counter()-start
                    start=time.perf_counter();scores,pred,unc=score_candidates(encoded,frozen,models);inference=time.perf_counter()-start
                    order=np.argsort(-scores,kind='stable').tolist();budget=3
                elif policy=='random_top3':order=random_order;budget=3
                else:order=list(range(8));budget=8
                write(folder/'selection.json',dict(order=order,budget=budget,scores=scores.tolist() if scores is not None else None,
                    predictions=pred.tolist() if pred is not None else None,uncertainty=unc.tolist() if unc is not None else None,
                    outcome_labels_read=False,policy_order=policy_order,random_order=random_order,selected_at=time.time()))
                validations=[];selected=None;start=time.perf_counter()
                for index in order[:budget]:
                    entry=current['candidates'][index]
                    row=rollout(current,entry,folder/'validation'/entry['name'],verify_repeat,RUNTIME);validations.append(row)
                    if row['success'] and selected is None:selected=entry
                    write(folder/'progress.json',dict(validations=validations,selected=selected['name'] if selected else None))
                    if selected and policy!='full8':break
                validation=time.perf_counter()-start;start=time.perf_counter();deployment=None
                if selected:deployment=rollout(current,selected,folder/'deployment'/selected['name'],deploy_repeat,RUNTIME)
                deploy=time.perf_counter()-start if selected else 0.
                report=dict(seed=seed,scenario=scenario,policy=policy,verify_repeat=verify_repeat,deploy_repeat=deploy_repeat,
                    valid=True,verification_hit=selected is not None,workflow_success=bool(deployment and deployment['success']),
                    twin_calls=len(validations),selected=selected['name'] if selected else None,plan_load_seconds=load,
                    encoding_seconds=encoding,inference_seconds=inference,model_load_seconds=cold if policy=='value_top3' else 0.,
                    validation_seconds=validation,deployment_seconds=deploy,workflow_wall_seconds=time.perf_counter()-began,
                    validations=validations,deployment=deployment)
                write(folder/'summary.json',report);reports[policy]=report
            scenarios.append(dict(scenario=scenario,policies=reports))
        result=dict(seed=seed,scenarios=scenarios);write(out/'summary.json',result);return result
    except Exception as exc:
        result=dict(seed=seed,error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        write(out/'error.json',result);return result


def summarize(rows):
    good=[r for r in rows if 'error' not in r];metrics={};paired={}
    def means(policy,field):return [float(np.mean([s['policies'][policy][field] for s in r['scenarios']])) for r in good]
    if good:
        for policy in POLICIES:
            all_runs=[s['policies'][policy] for r in good for s in r['scenarios']]
            metrics[policy]=dict(layouts=len(good),scenarios=len(all_runs),successes=sum(r['workflow_success'] for r in all_runs),
                workflow_success=bootstrap(means(policy,'workflow_success')),verification_hit=float(np.mean(means(policy,'verification_hit'))),
                mean_twin_calls=float(np.mean(means(policy,'twin_calls'))),
                seconds={k:dict(mean=float(np.mean(means(policy,k))),layout_bootstrap=bootstrap(means(policy,k))) for k in
                    ('plan_load_seconds','encoding_seconds','inference_seconds','model_load_seconds','validation_seconds','deployment_seconds','workflow_wall_seconds')})
        for baseline in ('random_top3','full8'):
            paired[baseline]={field:bootstrap(np.asarray(means('value_top3',field))-means(baseline,field)) for field in ('workflow_success','workflow_wall_seconds')}
    return dict(complete=len(good)==50 and len(rows)==50,completed_layouts=len(good),metrics=metrics,
        paired_value_minus_baseline=paired,rows=rows,timing_scope='Measured fresh frozen-plan load, encoding, inference, serial twin calls and independent deployment; no new planning time; cold model load separate; four concurrent layouts.')


def evaluate(source,training,out):
    if fingerprint()['sha256']!=RUNTIME:raise ValueError('physical runtime changed')
    progress=read(source/'progress.json')
    if not progress['complete']:raise ValueError('all 50 candidate pools must be frozen before evaluation')
    if {r['seed'] for r in progress['rows']}!=set(range(4000,4050)):raise ValueError('wrong frozen test layouts')
    for row in progress['rows']:
        if digest(read(source/f"seed_{row['seed']}"/'request.json'))!=row['request_sha256']:raise ValueError('test request changed after freeze')
    if sha(training/'selection_frozen.json')!=read(source/'manifest.json')['selection_frozen_sha256']:raise ValueError('model changed after test freeze')
    out.mkdir(parents=True,exist_ok=True);rows=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(evaluate_layout,(str(source),str(training),str(out),seed)) for seed in range(4000,4050)]
        for future in as_completed(futures):
            row=future.result();rows.append(row);write(out/'comparison.json',summarize(rows));print(dict(seed=row['seed'],error=row.get('error')),flush=True)
    result=summarize(rows)
    if not result['complete']:raise ValueError('test workflow incomplete; inspect preserved errors')
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--training',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();evaluate(a.source,a.training,a.out)
