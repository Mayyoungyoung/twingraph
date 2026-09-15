"""Process-cold / OS-cache-warm startup cost; no cached planning outcomes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

CHILD="""import json,sys,torch
from simbench.value.research_evaluate import OnlineExperiment
torch.set_num_threads(1)
request=json.loads(sys.argv[1])
experiment=OnlineExperiment(request,'cuda')
print(json.dumps(dict(model_load_and_warm_seconds=experiment.cold,checkpoint_sha256=experiment.checkpoint_hashes)))
"""


def main():
    p=argparse.ArgumentParser();p.add_argument('--models',default='results/value_v2/models/online_models.json');p.add_argument('--out',default='results/value_v2/cold_start.json');p.add_argument('--repeats',type=int,default=3);a=p.parse_args()
    models=json.loads(Path(a.models).read_text());methods=['random','geometry',*models];rows=[]
    rng=np.random.default_rng(993)
    for repeat in range(a.repeats):
        for method in rng.permutation(methods):
            request={str(method):models[str(method)]} if method in models else {}
            start=time.perf_counter();run=subprocess.run([sys.executable,'-c',CHILD,json.dumps(request)],capture_output=True,text=True,check=True)
            rows.append(dict(method=str(method),repeat=repeat,process_start_and_shutdown_seconds=time.perf_counter()-start,**json.loads(run.stdout)))
    result=dict(scope='fresh Python process including imports, lazy CUDA initialization, model loading, synthetic kernel warmup and process shutdown; OS file caches not flushed; no scene or physical rollout',
                workers=1,repeats=a.repeats,rows=rows,
                median_seconds={method:float(np.median([r['process_start_and_shutdown_seconds'] for r in rows if r['method']==method])) for method in methods})
    Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps(result['median_seconds']))


if __name__=='__main__':main()
