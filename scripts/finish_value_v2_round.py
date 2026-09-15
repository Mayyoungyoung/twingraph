"""Resume the already-dispatched round through actual training/test/execution."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    root=Path('results/value_v2');root.mkdir(parents=True,exist_ok=True)
    state=root/'round_state.json'
    def status(stage,**kw):state.write_text(json.dumps(dict(stage=stage,time=time.time(),**kw),indent=2));print(stage,flush=True)
    def run(args,log):
        with (root/log).open('w') as f:subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True)
    status('waiting_for_development_collection')
    while True:
        try:
            progress=json.loads((root/'data/development_progress.json').read_text())
            requests=json.loads((root/'data/development_request.json').read_text())
            if len(progress)==len(requests):
                if any('error' in r for r in progress):raise RuntimeError('collection errors; inspect saved progress')
                break
        except FileNotFoundError:pass
        time.sleep(15)
    # Configurations and hyperparameters are fixed before collecting locked labels.
    status('training_and_locked_reference_collection')
    with (root/'test_collection.log').open('w') as log:
        test=subprocess.Popen([sys.executable,'scripts/run_value_v2_collection.py','--test','--workers','6'],stdout=log,stderr=subprocess.STDOUT)
        run(['bash','scripts/train_value_v2.sh'],'training_pipeline.log')
        if test.wait()!=0:raise RuntimeError('locked collection failed')
    status('cache_locked_visuals')
    run([sys.executable,'-m','simbench.value.vision','--data',str(root/'data'),'--device','cuda'],'test_visual.log')
    # Model set fixed in advance, not selected on test performance.
    mapping={name:str(root/'models'/folder/'best.pt') for name,folder in
             [('prior','prior_17'),('mlp','mlp_17'),('residual','residual_17'),('no_vision','no_vision_17'),('vision','vision_17')]}
    mapping['v1_fixed']='results/value/direct/best.pt'
    (root/'models/online_models.json').write_text(json.dumps(mapping,indent=2))
    status('locked_offline_evaluation')
    run([sys.executable,'-m','simbench.value.research_evaluate','--models',str(root/'models/all_models.json'),
         '--data',str(root/'data'),'--out',str(root/'locked_metrics.json')],'locked_evaluation.log')
    status('independent_online_execution')
    run([sys.executable,'scripts/run_value_v2_online.py','--workers','6'],'online.log')
    run([sys.executable,'scripts/summarize_value_v2.py'],'summarize.log')
    status('complete')


if __name__=='__main__':main()
