"""Fixed splits and resource-bounded collection. Never select by outcomes."""
import argparse
import concurrent.futures
import multiprocessing
import json
from pathlib import Path
from simbench.value.research_collect import worker
from simbench.value.collect import dump


def main():
    p=argparse.ArgumentParser();p.add_argument("--out",default="results/value_v2/data")
    p.add_argument("--workers",type=int,default=6);p.add_argument("--test",action="store_true")
    a=p.parse_args();requests=[]
    for family,base in (("rigid_connector_module",20000),("sliding_stage_pin",21000)):
        for split,start,count in (("test",40,6),) if a.test else (("train",10,12),("val",30,4)):
            for i in range(count):
                requests.append(dict(family=family,seed=base+start+i,out=a.out,n=16,repeats=2,
                                     domain="reference" if a.test else "train",split=split,checkpoint=0))
                if family=="rigid_connector_module" and i<2:
                    for cp in ([1,3] if a.test else [1,2]):
                        requests.append(dict(family=family,seed=base+start+i,out=a.out,n=16,repeats=2,
                                     domain="reference" if a.test else "train",split=split,checkpoint=cp))
    Path(a.out).mkdir(parents=True,exist_ok=True)
    dump(Path(a.out)/("test_request.json" if a.test else "development_request.json"),requests)
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        results=[]
        for f in concurrent.futures.as_completed([pool.submit(worker,r) for r in requests]):
            result=f.result();results.append(result);print(json.dumps(result),flush=True)
            dump(Path(a.out)/("test_progress.json" if a.test else "development_progress.json"),results)
    if any("error" in r for r in results):raise SystemExit(1)


if __name__=="__main__":main()
