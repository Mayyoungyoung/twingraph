"""Additional actual budget/scale runs; strongest checkpoint chosen on validation."""
import argparse
import concurrent.futures
import json
import multiprocessing
from pathlib import Path
import torch
import numpy as np
from simbench.value.research_evaluate import OnlineExperiment
from simbench.value.collect import dump


def choose_models(root):
    models=json.loads((root/'models/online_models.json').read_text())
    candidates=[]
    for name,path in models.items():
        if name=='v1_fixed':continue
        saved=torch.load(path,map_location='cpu',weights_only=False)
        val=saved['validation'];parameters=sum(v.numel() for v in saved['state_dict'].values())
        # Retention/regret first; favor the smaller model on equal quality.
        candidates.append(((val['hit'] or 0.,-val['regret'],-parameters,-val['brier']),name,path))
    _,name,path=max(candidates)
    dump(root/'models/validation_recommendation.json',dict(method=name,checkpoint=path,
         rule='validation Hit@4 descending, Regret@4 ascending, parameter count ascending, Brier ascending; seed17 primary models',
         candidate_metrics=[dict(method=n,criterion=list(c)) for c,n,p in candidates]))
    return {name:path}


def worker(request):
    task,models,settings,out=request;torch.set_num_threads(1);experiment=OnlineExperiment(models,'cuda');rows=[]
    for n,k,budget in settings:
        for method in np.random.default_rng(task['seed']+n+budget).permutation(['geometry',*models]):
            method=str(method)
            directory=Path(out)/f"{task['family']}_{task['seed']}_{method}_n{n}_k{k}_b{budget}"
            row=experiment.run(task,method,directory,n=n,k=k,budget=budget,repeats=2,mode='best_within_budget',allow_expand=False)
            rows.append(row);print(json.dumps(dict(task=task,method=method,n=n,k=k,budget=budget,success=row['execution_success_rate'],seconds=row['timing']['decision_wall_seconds'])),flush=True)
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/value_v2');p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    root=Path(a.root);models=choose_models(root);out=root/'online_curves';out.mkdir(exist_ok=True)
    jobs=[]
    for family,base in [('rigid_connector_module',20040),('sliding_stage_pin',21040)]:
        jobs.append((dict(family=family,seed=base,checkpoint=0),models,[(16,1,2),(16,2,4),(16,4,8)],str(root/'online_budget')))
        jobs.append((dict(family=family,seed=base+4,checkpoint=0),models,[(16,4,8),(32,4,8),(64,4,8)],str(root/'online_scale')))
    dump(out/'request.json',dict(jobs=jobs,models=models,workers=a.workers,selection='validation only',
         limitation='scale runs measure deployment and actual cost; no complete 64-candidate reference labels'))
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        rows=[]
        for result in pool.map(worker,jobs):rows.extend(result);dump(out/'results.json',rows)


if __name__=='__main__':main()
