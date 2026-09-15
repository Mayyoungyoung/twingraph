"""Parallelize independent problem groups, never race rankers within a group."""
import argparse
import concurrent.futures
import json
import multiprocessing
from pathlib import Path
from simbench.value.research_evaluate import OnlineExperiment
from simbench.value.collect import dump


def run_task(request):
    task,models,out=request
    import torch
    torch.set_num_threads(1)
    experiment=OnlineExperiment(models,'cuda');rows=[]
    # Paired method order randomized without outcome access, to reduce clock bias.
    import numpy as np
    methods=['random','geometry',*models]
    rng=np.random.default_rng(task['seed']+921)
    for mode in ('first_verified','best_within_budget'):
        for method in rng.permutation(methods):
            name=f"{task['family']}_{task['seed']}_{task.get('checkpoint',0)}_{method}_{mode}"
            row=experiment.run(task,str(method),Path(out)/name,n=16,k=4,budget=8,repeats=2,mode=mode,allow_expand=False)
            rows.append(row)
            print(json.dumps(dict(task=task,method=str(method),mode=mode,success=row['execution_success_rate'],seconds=row['timing']['decision_wall_seconds'])),flush=True)
    row=experiment.run(task,'exhaustive',Path(out)/f"{task['family']}_{task['seed']}_exhaustive",n=16,repeats=2)
    rows.append(row)
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--models',default='results/value_v2/models/online_models.json')
    p.add_argument('--out',default='results/value_v2/online');p.add_argument('--workers',type=int,default=6)
    p.add_argument('--groups-per-family',type=int,default=3);a=p.parse_args()
    models=json.loads(Path(a.models).read_text());tasks=[dict(family=f,seed=base+i,checkpoint=0)
        for f,base in [('rigid_connector_module',20040),('sliding_stage_pin',21040)] for i in range(a.groups_per_family)]
    Path(a.out).mkdir(parents=True,exist_ok=True)
    dump(Path(a.out)/'execution_request.json',dict(tasks=tasks,models=models,workers=a.workers,k=4,budget=8,repeats=2,
        local_workers_per_decision=1,method_order='seeded random permutation',scope='independent physical simulation deployment'))
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        allrows=[]
        for rows in pool.map(run_task,[(t,models,a.out) for t in tasks]):
            allrows.extend(rows);dump(Path(a.out)/'results.json',allrows)


if __name__=='__main__':main()
