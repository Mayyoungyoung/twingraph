"""Exclusive local model relay for the new, outcome-blind V33 test pools."""
from concurrent.futures import ThreadPoolExecutor
import json,os,time,subprocess
from pathlib import Path
from scripts.relay_planner_v32 import remote,REMOTE,LOCAL,plan


def main():
    LOCAL.mkdir(exist_ok=True,parents=True)
    lock=(LOCAL/'relay.lock').open('a+b')
    if lock.seek(0,2)==0:lock.write(b'0');lock.flush()
    lock.seek(0)
    if os.name=='nt':
        import msvcrt
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    else:
        import fcntl
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    futures={};tries={}
    with ThreadPoolExecutor(max_workers=2) as pool:
        while True:
            try:
                state=json.loads(remote(f'''from pathlib import Path
import json
root=Path({REMOTE!r});base=root/'results/v33_test_top3_20260927'
pending=[str(p.parent.relative_to(root)) for p in base.glob('seed_*/planner/request.json') if not (p.parent/'response.json').exists()]
status=root/'results/v33_robust_top3_20260927/status.json'
print(json.dumps(dict(pending=pending,stage=json.loads(status.read_text()).get('stage') if status.exists() else None)))
'''))
            except (RuntimeError,subprocess.TimeoutExpired,json.JSONDecodeError) as exc:
                print('connection retry '+str(exc),flush=True);time.sleep(15);continue
            for name,future in list(futures.items()):
                if future.done():
                    try:future.result();print('delivered '+name,flush=True)
                    except Exception as exc:print('planner error '+name+' '+str(exc),flush=True)
                    del futures[name]
            for name in state['pending']:
                if name not in futures and tries.get(name,0)<3:
                    tries[name]=tries.get(name,0)+1;futures[name]=pool.submit(plan,name)
            if state['stage'] in ('fresh_evaluation','complete') and not futures:return
            time.sleep(10)


if __name__=='__main__':main()
