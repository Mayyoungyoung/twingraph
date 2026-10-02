"""Repeat every frozen train/validation candidate under predefined perturbations."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime,timezone
import multiprocessing
from pathlib import Path
import shutil
import time
from scripts.train_value_v32 import read,write,sha
from scripts.evaluate_workflow_v32 import rollout
from scripts.collect_sliding_assembly_v23 import REQUIRED
from simbench.value.plan import digest
from simbench.value.physical import perturbation
from simbench.value.provenance_v12 import fingerprint

RUNTIME='42a1eda6fe8996d4312daa40c0855795c6e9142d7911427195d14bd306fd65be'
REPEATS=(10,11,12)


def checked(path,request,entry,repeat):
    r=read(path)
    if not r.get('valid') or r.get('timeout') or r.get('software_exception'):
        raise ValueError(f'invalid trial requires diagnosis: {path}')
    if (r['source_plan_sha256']!=entry['plan_sha256'] or r['repeat']!=repeat
            or r['source_runtime_sha256']!=request['runtime_sha256'] or r['runtime_sha256']!=RUNTIME):
        raise ValueError(f'completed trial provenance mismatch: {path}')
    if r['trial']!=perturbation(request['seed'],repeat,request['domain']) or r['success']!=all(r['stage_passes'].get(k) is True for k in REQUIRED):
        raise ValueError(f'trial parameters or label mismatch: {path}')
    return dict(valid=True,success=r['success'],seconds=r['fresh_total_wall_seconds'],result=str(path),repeat=repeat)


def worker(job):
    request_path,name,repeat,folder=job
    request=read(Path(request_path));entry=next(e for e in request['candidates'] if e['name']==name)
    folder=Path(folder)
    if (folder/'result.json').exists():return checked(folder/'result.json',request,entry,repeat)
    if folder.exists():
        # Only incomplete executions can enter here; full valid/invalid results never repeat.
        archive=folder.parent/'interrupted';archive.mkdir(exist_ok=True)
        target=archive/(folder.name+'_'+str(time.time_ns()))
        shutil.move(str(folder),str(target))
        write(target/'interruption.json',dict(reason='previous process stopped before a complete result was saved',restarted_at=time.time()))
    result=rollout(request,entry,folder,repeat,RUNTIME)
    return checked(folder/'result.json',request,entry,repeat)


def collect(source,out,workers=8):
    if fingerprint()['sha256']!=RUNTIME:raise ValueError('physical runtime mismatch')
    baseline_audit=read(source.parent/'v32_value_training_20260927'/'dataset_audit.json')
    if not baseline_audit['passed'] or not baseline_audit['dataset_complete']:raise ValueError('audited source required')
    original=read(source/'manifest.json');out.mkdir(parents=True,exist_ok=True)
    jobs=[];inputs={};splits={}
    for j in original['jobs']:
        if j['split'] not in ('train','validation'):continue
        path=source/f"seed_{j['seed']}"/'request.json';request=read(path)
        if len(request['candidates'])!=8:raise ValueError('candidate pool changed')
        inputs[str(j['seed'])]=digest(request);splits[str(j['seed'])]=j['split']
        for entry in request['candidates']:
            for repeat in REPEATS:
                folder=out/f"seed_{j['seed']}"/entry['name']/f'repeat_{repeat}'
                jobs.append((str(path),entry['name'],repeat,str(folder)))
    manifest=dict(runtime_sha256=RUNTIME,source_manifest_sha256=sha(source/'manifest.json'),
        request_sha256=inputs,splits=splits,repeats=list(REPEATS),planned_trials=len(jobs),outcome_based_resampling=False)
    if len(jobs)!=2688:raise ValueError('wrong train/validation repeat count')
    if (out/'manifest.json').exists() and read(out/'manifest.json')!=manifest:raise ValueError('repeat protocol changed')
    write(out/'manifest.json',manifest)
    completed=positives=0;pending_jobs=[]
    for job in jobs:
        path=Path(job[3])/'result.json'
        if path.exists():
            request=read(Path(job[0]));entry=next(e for e in request['candidates'] if e['name']==job[1])
            r=checked(path,request,entry,job[2]);completed+=1;positives+=int(r['success'])
        else:pending_jobs.append(job)
    def progress(stage,error=None):
        write(out/'progress.json',dict(stage=stage,completed=completed,planned=len(jobs),positives=positives,
            updated_at=datetime.now(timezone.utc).isoformat(),error=error))
    progress('running')
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        iterator=iter(pending_jobs);pending={}
        def submit():
            job=next(iterator,None)
            if job is not None:pending[pool.submit(worker,job)]=job
        for _ in range(workers):submit()
        while pending:
            done,_=wait(pending,return_when=FIRST_COMPLETED)
            for f in done:
                job=pending.pop(f)
                try:r=f.result()
                except Exception as exc:
                    progress('failed',f'{job}: {type(exc).__name__}: {exc}')
                    for future in pending:future.cancel()
                    raise
                completed+=1;positives+=int(r['success']);progress('running')
                print(dict(completed=completed,planned=len(jobs),last=job[3]),flush=True);submit()
    progress('complete')
    return read(out/'progress.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=8);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (a.out/'collection.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);collect(a.source,a.out,a.workers)
