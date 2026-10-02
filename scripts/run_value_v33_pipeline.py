"""Persistent V33 pipeline: await repeats, train, freeze new pools, evaluate, audit."""
import argparse
from datetime import datetime,timezone
import os,shutil,subprocess,sys,time
from pathlib import Path
from scripts.train_value_v32 import read,write,sha


def status(out,stage,**fields):
    write(out/'status.json',dict(stage=stage,pid=os.getpid(),updated_at=datetime.now(timezone.utc).isoformat(),**fields))


def run(source,test,out):
    out.mkdir(exist_ok=True,parents=True);snapshot=out/'source';snapshot.mkdir(exist_ok=True)
    files=['collect_robust_value_v33.py','value_features_v33.py','train_value_v33.py','freeze_test_value_v33.py',
        'evaluate_workflow_v33.py','finalize_value_v33.py','run_value_v33_pipeline.py',
        'train_value_v32.py','evaluate_workflow_v32.py','collect_sliding_assembly_v23.py','llm_transport_v32.py']
    for name in files:
        current=Path(__file__).with_name(name);target=snapshot/name
        if target.exists() and sha(target)!=sha(current):raise ValueError(f'pipeline source changed: {name}; explicit repair evidence required')
        if not target.exists():shutil.copy2(current,target)
    while True:
        p=out/'repeats/progress.json';progress=read(p) if p.exists() else {}
        if progress.get('stage')=='failed':raise ValueError('repeat collection halted: '+str(progress.get('error')))
        if progress.get('stage')=='complete':break
        status(out,'collecting_robust_repeats',completed=progress.get('completed',0),planned=2688);time.sleep(30)
    def execute(stage,module,args):
        for name in files:
            if sha(Path(__file__).with_name(name))!=sha(snapshot/name):raise ValueError(f'source changed before {stage}: {name}')
        status(out,stage)
        with (out/(stage+'.log')).open('a') as log:
            subprocess.run([sys.executable,'-u','-m',module,*args],stdout=log,stderr=subprocess.STDOUT,check=True)
    execute('training','scripts.train_value_v33',['--source',str(source),'--repeats',str(out/'repeats'),'--out',str(out/'training')])
    execute('freezing_test_candidates','scripts.freeze_test_value_v33',['--out',str(test),'--training',str(out/'training')])
    execute('fresh_evaluation','scripts.evaluate_workflow_v33',['--source',str(test),'--training',str(out/'training'),'--out',str(out/'workflow')])
    execute('final_audit','scripts.finalize_value_v33',['--source',str(test),'--out',str(out)])
    status(out,'complete',report=str(out/'REPORT_ZH.md'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--test',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (a.out/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(a.source,a.test,a.out)
        except Exception as exc:status(a.out,'failed',error=f'{type(exc).__name__}: {exc}');raise
