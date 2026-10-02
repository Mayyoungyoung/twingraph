"""Resume the fixed 125-layout dataset, preserving all frozen trials and splits."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from scripts.collect_dataset_v32 import write, worker
from scripts.collect_sliding_assembly_v23 import collect
from simbench.value.provenance_v12 import fingerprint

REPAIR_FILE = 'simbench/assembly/placement_catalog_v13.py'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verified_existing(layout, seed, allowed):
    """A summary alone cannot make a layout complete."""
    if not (layout/'request.json').exists(): return None
    request=read(layout/'request.json')
    if request['runtime_sha256'] not in allowed: raise ValueError('unregistered runtime')
    entries=request['candidates']; rows=[]
    if len(entries)!=8: raise ValueError('layout must have eight frozen candidates')
    for entry in entries:
        p=layout/'candidates'/entry['name']/'result.json'
        if not p.exists(): return None
        result=read(p)
        if (not result['valid'] or result.get('timeout') or result.get('resource_censored')
                or result['runtime_sha256']!=request['runtime_sha256']
                or result['plan_sha256']!=entry['plan_sha256']):
            raise ValueError(f'invalid or mismatched existing sample: {p}')
        rows.append(result)
    return dict(seed=seed,out=str(layout.resolve()),attempted=8,valid=8,
        positives=sum(r['success'] for r in rows),negatives=sum(not r['success'] for r in rows),
        runtime_sha256=request['runtime_sha256'])


def resume_worker(job):
    seed, output, runtime = job
    layout=Path(output)
    try:
        if not (layout/'request.json').exists():
            row=worker((seed, output)); row['runtime_sha256']=runtime; return row
        # Resume only missing executions from the exact frozen candidate set.
        request=read(layout/'request.json')
        if request['runtime_sha256']!=runtime:
            raise ValueError('partial layout requires its original runtime')
        for index, entry in enumerate(request['candidates']):
            if (layout/'candidates'/entry['name']/'result.json').exists(): continue
            temporary=layout.parent/'recovery_runs'/f'seed_{seed}_{index}_{time.time_ns()}'
            collect(seed,temporary,candidate_n=8,pool_n=48,planner='llm',domain='online',level='L0',
                timeout_seconds=1800.,frozen_request=layout/'request.json',execute_start=index,execute_stop=index+1)
            target=layout/'candidates'/entry['name']
            if target.exists():
                archived=layout.parent/'interrupted_candidates'/f'seed_{seed}_{entry["name"]}_{time.time_ns()}'
                archived.parent.mkdir(exist_ok=True); target.rename(archived)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copytree(temporary/'candidates'/entry['name'],target)
        result=verified_existing(layout,seed,{runtime})
        if result is None: raise ValueError('incomplete frozen execution')
        return result
    except Exception as exc:
        return dict(seed=seed,out=str(layout),error=f'{type(exc).__name__}: {exc}')


def summarize(rows, runtime, target, halt=None, running=False):
    complete=[r for r in rows if 'error' not in r]
    return dict(runtime_sha256=runtime,completed_layouts=len(complete),attempted=sum(r['attempted'] for r in complete),
        valid=sum(r['valid'] for r in complete),positives=sum(r['positives'] for r in complete),
        negatives=sum(r['negatives'] for r in complete),mixed_layouts=sum(0<r['positives']<8 for r in complete),
        planned_samples=target,halt_reason=halt,results=rows,purpose='value_training_data',
        running=running,updated_at=datetime.now(timezone.utc).isoformat(),
        ready_for_collection=not halt and len(complete)>=3)


def run(out, original, workers):
    current=fingerprint(); manifest=read(out/'manifest.json')
    if manifest['planned_samples']!=1000 or len(manifest['jobs'])!=125:
        raise ValueError('expected original fixed 1000-sample manifest')
    baseline=read(original)
    if baseline['sha256']!=manifest['runtime_sha256']: raise ValueError('wrong original fingerprint')
    changed={p for p in set(baseline['files'])|set(current['files']) if baseline['files'].get(p)!=current['files'].get(p)}
    if changed!={REPAIR_FILE}: raise ValueError(f'unreviewed runtime changes: {sorted(changed)}')
    revision_dir=out/'revisions'/current['sha256']; revision_dir.mkdir(parents=True,exist_ok=True)
    if not (revision_dir/'manifest_before.json').exists():
        shutil.copy2(out/'manifest.json',revision_dir/'manifest_before.json')
        if (out/'summary.json').exists(): shutil.copy2(out/'summary.json',revision_dir/'summary_before.json')
    write(revision_dir/'runtime.json',current)
    for name in ('resume_dataset_v32.py','collect_dataset_v32.py','collect_sliding_assembly_v23.py','llm_transport_v32.py'):
        shutil.copy2(Path(__file__).with_name(name),revision_dir/name)
    shutil.copy2(Path(REPAIR_FILE),revision_dir/Path(REPAIR_FILE).name)
    revisions=manifest.setdefault('runtime_revisions',[])
    if not any(r['sha256']==current['sha256'] for r in revisions):
        revisions.append(dict(sha256=current['sha256'],parent_sha256=baseline['sha256'],changed_files=sorted(changed),
            reason='Use conditional predecessor poses for stage-specific pickup geometry; physical controllers, observations, labels and encoder unchanged',
            source=f'revisions/{current["sha256"]}',created_at=datetime.now(timezone.utc).isoformat()))
    manifest['active_runtime_sha256']=current['sha256']
    manifest['resume_policy']='Preserve complete trials and splits; resume frozen missing candidates; archive pre-freeze errors before retrying same seed'
    write(out/'manifest.json',manifest)
    allowed={manifest['runtime_sha256'],*[r['sha256'] for r in revisions]}
    rows=[]; pending=[]
    for job in manifest['jobs']:
        layout=out/f"seed_{job['seed']}"
        existing=verified_existing(layout,job['seed'],allowed)
        if existing is not None: rows.append(existing); continue
        if layout.exists() and any(layout.iterdir()) and not (layout/'request.json').exists():
            # Keep the original failed geometry scene and any model request.
            archive=out/'pre_freeze_failures'/f"seed_{job['seed']}_{time.time_ns()}"
            archive.parent.mkdir(exist_ok=True); layout.rename(archive)
        pending.append((job['seed'],str(layout),current['sha256']))
    write(out/'summary.json',summarize(rows,current['sha256'],1000,running=bool(pending)))
    for begin in range(0,len(pending),workers):
        if fingerprint()['sha256']!=current['sha256']: raise ValueError('runtime changed before batch')
        batch=pending[begin:begin+workers]; latest=[]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for future in as_completed([pool.submit(resume_worker,job) for job in batch]):
                row=future.result(); rows.append(row); latest.append(row)
                write(out/'progress.json',dict(results=rows,runtime_sha256=current['sha256']))
                write(out/'summary.json',summarize(rows,current['sha256'],1000,running=True))
                print(json.dumps(dict(event='layout_complete',**row)),flush=True)
        halt=None
        if fingerprint()['sha256']!=current['sha256']: halt='runtime_changed'
        elif any('error' in r or r.get('valid')!=8 for r in latest): halt='invalid_or_censored_sample'
        elif len(latest)>=3 and sum(r['positives'] for r in latest)==0: halt='all_negative_batch'
        done=sum(r.get('valid',0) for r in rows)==1000
        write(out/'summary.json',summarize(rows,current['sha256'],1000,halt,running=not halt and not done))
        if halt: return 2
    return 0


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--original-runtime',type=Path,required=True); parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    if not 1<=args.workers<=8: parser.error('workers must be 1..8')
    # Linux collection host: release the exclusive lock automatically on exit.
    import fcntl
    with (args.out/'resume.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        lock.write(str(os.getpid())); lock.flush()
        try: return run(args.out,args.original_runtime,args.workers)
        except Exception as exc:
            summary=read(args.out/'summary.json') if (args.out/'summary.json').exists() else {}
            summary.update(running=False,halt_reason='resume_exception',error=f'{type(exc).__name__}: {exc}')
            write(args.out/'summary.json',summary); raise


if __name__=='__main__': raise SystemExit(main())
