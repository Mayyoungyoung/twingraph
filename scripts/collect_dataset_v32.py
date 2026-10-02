"""Frozen-layout collection: model-composed candidates, raw outcomes and layout splits."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.collect_sliding_assembly_v23 import collect, TASK_VERSION
from simbench.value.provenance_v12 import fingerprint


def write(path, value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)


def worker(job):
    seed,out=job
    try:
        result=collect(seed,Path(out),candidate_n=8,pool_n=48,planner='llm',domain='online',
                       level='L0',timeout_seconds=1800.)
        return dict(seed=seed,out=str(out),attempted=result['attempted'],valid=result['valid'],
                    positives=result['successes'],negatives=result['failures'])
    except Exception as exc:
        return dict(seed=seed,out=str(out),error=f'{type(exc).__name__}: {exc}')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+')
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--pilot-report',type=Path)
    a=p.parse_args()
    if not 1<=a.workers<=8:p.error('workers must be in [1,8]')
    a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'manifest.json').exists():raise ValueError('choose a new output directory; never overwrite a collection')
    runtime=fingerprint()['sha256']
    if a.seeds is not None:
        seeds=a.seeds
        purpose='development_pilot_excluded_from_training'
    else:
        if a.pilot_report is None:p.error('large collection requires a passing pilot report')
        pilot=json.loads(a.pilot_report.read_text(encoding='utf-8'))
        if pilot.get('runtime_sha256')!=runtime or not pilot.get('ready_for_collection'):
            raise ValueError('pilot does not pass the current runtime readiness gate')
        seeds=list(range(3300,3425))
        purpose='value_training_data'
    if len(set(seeds))!=len(seeds):raise ValueError('duplicate layout seeds')
    permutation=np.random.default_rng(320032).permutation(len(seeds)).tolist()
    splits={seeds[i]:('train' if j<int(.8*len(seeds)) else 'validation' if j<int(.9*len(seeds)) else 'test')
            for j,i in enumerate(permutation)}
    jobs=[dict(seed=s,split=splits[s],out=str((a.out/f'seed_{s}').resolve())) for s in seeds]
    files=['collect_dataset_v32.py','collect_sliding_assembly_v23.py','llm_transport_v32.py']
    script_hashes={f:hashlib.sha256(Path(__file__).with_name(f).read_bytes()).hexdigest() for f in files}
    source=a.out/'source';source.mkdir()
    for f in files:(source/f).write_bytes(Path(__file__).with_name(f).read_bytes())
    manifest=dict(task_version=TASK_VERSION,runtime_sha256=runtime,script_sha256=script_hashes,
        purpose=purpose,candidates_per_layout=8,planned_samples=8*len(seeds),jobs=jobs,
        domain='online',position_noise_std_m=0.,yaw_noise_std_rad=0.,planner='llm',
        gate=dict(minimum_layouts=3,minimum_valid_fraction=.95,minimum_positive_fraction=.125,
                  minimum_mixed_layouts=2),
        workers=a.workers,
        stop_rule='stop at batch boundary on runtime change, invalid sample, or an all-negative batch of at least 3 layouts',
        outcome_based_resampling=False,forced_transport_mode=False)
    write(a.out/'manifest.json',manifest)
    rows=[]
    halt=None
    for begin in range(0,len(jobs),a.workers):
        batch=jobs[begin:begin+a.workers]
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            futures=[pool.submit(worker,(j['seed'],j['out'])) for j in batch]
            for future in as_completed(futures):
                rows.append(future.result())
                write(a.out/'progress.json',dict(results=rows,runtime_sha256=runtime))
        complete=[r for r in rows if 'error' not in r]
        last=[r for r in rows if r['seed'] in {j['seed'] for j in batch}]
        if fingerprint()['sha256']!=runtime:halt='runtime_changed'
        elif any('error' in r or r.get('valid')!=8 for r in last):halt='invalid_or_censored_sample'
        elif len(last)>=3 and sum(r['positives'] for r in last)==0:halt='all_negative_batch'
        summary=dict(runtime_sha256=runtime,completed_layouts=len(complete),attempted=sum(r['attempted'] for r in complete),
            valid=sum(r['valid'] for r in complete),positives=sum(r['positives'] for r in complete),
            negatives=sum(r['negatives'] for r in complete),mixed_layouts=sum(0<r['positives']<8 for r in complete),
            halt_reason=halt,results=rows,purpose=purpose)
        summary['ready_for_collection']=bool(not halt and len(complete)>=3 and summary['valid']>=.95*summary['attempted']
            and summary['positives']>=.125*summary['valid'] and summary['mixed_layouts']>=2
            and summary['negatives']>=.125*summary['valid'])
        write(a.out/'summary.json',summary)
        print(json.dumps(summary),flush=True)
        if halt:return 2
    return 0


if __name__=='__main__':raise SystemExit(main())
