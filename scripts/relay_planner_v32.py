"""Run the model locally on remote pre-execution requests; relay no outcome data to it."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import tarfile
from scripts.llm_transport_v32 import infer

REMOTE='/home/jia/twingraph-v32-20260926'
LOCAL=Path('results/v32_remote')


def sync_results(name):
    """Copy completed candidates and frozen requests, then build training inputs."""
    remote(f'''import tarfile
from pathlib import Path
root=Path({REMOTE!r}); source=root/'results'/{name!r}
with tarfile.open(root/{(name+'_snapshot.tar.gz')!r},'w:gz',compresslevel=1) as archive:
 for p in source.rglob('*'):
  if not p.is_file() or p.suffix=='.tmp':continue
  rel=p.relative_to(source)
  if 'candidates' in rel.parts:
   i=rel.parts.index('candidates');candidate=source.joinpath(*rel.parts[:i+2])
   if not (candidate/'result.json').exists():continue
  archive.add(p,arcname=str(p.relative_to(root)))
''',timeout=900)
    target=LOCAL/(name+'_snapshot.tar.gz')
    subprocess.run(['scp',f'901:{REMOTE}/{name}_snapshot.tar.gz',str(target)],check=True,capture_output=True,timeout=1800)
    base=LOCAL.resolve()
    with tarfile.open(target) as archive:
        members=archive.getmembers()
        for member in members:
            destination=(base/member.name).resolve()
            if base not in destination.parents or not member.isfile():
                raise ValueError('unsafe snapshot member')
        archive.extractall(base,members=members)
    from scripts.export_dataset_v32 import export
    report=export(LOCAL/'results'/name,LOCAL/(name+'_value'))
    print(json.dumps(dict(synced=name,exported=report['exported'],positives=report['positives'])),flush=True)
    if name=='dataset_1000':
        from scripts.audit_dataset_v32 import audit
        checked=audit(LOCAL/'results'/name,LOCAL/(name+'_value'))
        (LOCAL/'dataset_1000_audit.json').write_text(json.dumps(checked,indent=2),encoding='utf-8')
        if not checked['passed']:raise ValueError('dataset audit failed; inspect dataset_1000_audit.json')
        print(json.dumps(dict(audited=name,samples=checked['samples'],passed=True)),flush=True)


def remote(code, *, timeout=45):
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','901','python3','-'],
                     input=code.encode('utf-8'),capture_output=True,timeout=timeout)
    if p.returncode:raise RuntimeError(p.stderr.decode('utf-8',errors='replace')[-1000:])
    return p.stdout.decode('utf-8')


def plan(relative):
    directory=LOCAL/relative
    directory.mkdir(parents=True,exist_ok=True)
    for name in ('request.json','response.schema.json'):
        subprocess.run(['scp',f'901:{REMOTE}/{relative}/{name}',str(directory/name)],check=True,capture_output=True,timeout=45)
    if not (directory/'response.json').is_file():infer(directory)
    subprocess.run(['scp',str(directory/'response.json'),f'901:{REMOTE}/{relative}/response.upload'],check=True,capture_output=True,timeout=45)
    remote(f'from pathlib import Path\np=Path({(REMOTE+"/"+relative)!r})\n(p/"response.upload").replace(p/"response.json")\n')
    return relative


def launch_bulk():
    return remote(f'''import os,json,subprocess
from pathlib import Path
root=Path({REMOTE!r})
summary=json.loads((root/'results/readiness/summary.json').read_text())
if not summary.get('ready_for_collection'):raise ValueError('readiness gate failed')
marker=root/'bulk_process.json'
if marker.exists():print(marker.read_text())
else:
 env={{**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TWINGRAPH_EXTERNAL_PLANNER':'1','MUJOCO_GL':'egl'}}
 cmd=['/home/jia/twingraph-v8-mj237/bin/python','-m','scripts.collect_dataset_v32','--out','results/dataset_1000','--pilot-report','results/readiness/summary.json','--workers','8']
 f=(root/'bulk.log').open('w')
 p=subprocess.Popen(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 marker.write_text(json.dumps(dict(pid=p.pid,command=cmd)))
 print(marker.read_text())
''')


def get_state():
    return json.loads(remote(f'''import json
from pathlib import Path
root=Path({REMOTE!r})
pending=[str(p.parent.relative_to(root)) for p in root.glob('results/*/seed_*/planner/request.json') if not (p.parent/'response.json').exists()]
summaries={{}}
progress={{}}
for name in ('readiness','dataset_1000'):
 p=root/'results'/name/'summary.json'
 if p.exists():summaries[name]=json.loads(p.read_text())
 rows=[]
 for p in sorted((root/'results'/name).glob('seed_*/progress.json')):
  r=json.loads(p.read_text());rows.append(dict(layout=p.parent.name,completed=len(r),valid=sum(x['valid'] for x in r),positives=sum(x['valid'] and x['success'] for x in r)))
 if rows:progress[name]=dict(completed=sum(r['completed'] for r in rows),positives=sum(r['positives'] for r in rows),layouts=rows)
print(json.dumps(dict(pending=pending,summaries=summaries,progress=progress)))
'''))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume-existing',action='store_true',help='Monitor the resumed collector without launching or exporting the old pilot')
    args=parser.parse_args()
    LOCAL.mkdir(parents=True,exist_ok=True)
    # The OS releases this lock if the relay exits or is interrupted.
    # Hold the handle for the entire function, including executor shutdown.
    relay_lock=(LOCAL/'relay.lock').open('a+b')
    if relay_lock.seek(0,2)==0:
        relay_lock.write(b'0');relay_lock.flush()
    relay_lock.seek(0)
    if os.name=='nt':
        import msvcrt
        msvcrt.locking(relay_lock.fileno(),msvcrt.LK_NBLCK,1)
    else:
        import fcntl
        fcntl.flock(relay_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    futures={}; finished=set(); attempts={}; bulk_started=args.resume_existing; connection_failed=False
    with ThreadPoolExecutor(max_workers=3) as pool:
        while True:
            try:
                state=get_state()
                if connection_failed:print('connection_restored',flush=True)
                connection_failed=False
            except (RuntimeError,subprocess.TimeoutExpired,json.JSONDecodeError) as exc:
                if not connection_failed:print('connection_retry '+str(exc),flush=True)
                connection_failed=True;time.sleep(10.);continue
            temporary=LOCAL/'status.tmp'
            temporary.write_text(json.dumps(state['progress'],indent=2),encoding='utf-8')
            temporary.replace(LOCAL/'status.json')
            for relative,future in list(futures.items()):
                if future.done():
                    try:
                        future.result();print(json.dumps(dict(model_completed=relative)),flush=True)
                        finished.add(relative)
                    except Exception as exc:
                        print(json.dumps(dict(model_error=relative,error=str(exc))),flush=True)
                        # Retry delivery of an already frozen response. A network
                        # interruption must not require selecting a different plan.
                        if attempts[relative]>=3:
                            finished.add(relative)
                    del futures[relative]
            for relative in state['pending']:
                if relative not in futures and relative not in finished:
                    attempts[relative]=attempts.get(relative,0)+1
                    futures[relative]=pool.submit(plan,relative)
            for name,summary in state['summaries'].items():
                (LOCAL/f'{name}_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
            pilot=state['summaries'].get('readiness')
            if pilot and pilot.get('completed_layouts',0)>=3 and not bulk_started:
                sync_results('readiness')
                if pilot.get('ready_for_collection'):
                    print('bulk_started '+launch_bulk(),flush=True);bulk_started=True
                else:
                    print('pilot_gate_failed',flush=True);return 2
            if pilot and pilot.get('halt_reason') and not args.resume_existing:
                if pilot.get('completed_layouts',0)<3:sync_results('readiness')
                print('pilot_halted '+str(pilot['halt_reason']),flush=True);return 2
            bulk=state['summaries'].get('dataset_1000')
            if bulk and (bulk.get('halt_reason') or bulk.get('completed_layouts')==125):
                sync_results('dataset_1000')
                print('bulk_finished '+json.dumps(bulk),flush=True);return 0 if not bulk.get('halt_reason') else 2
            time.sleep(10.)


if __name__=='__main__':raise SystemExit(main())
