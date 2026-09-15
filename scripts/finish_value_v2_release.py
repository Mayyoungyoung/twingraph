"""Finish actual curves, a fresh executable Top-K, and the reproducible release."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--code-commit',required=True)
    p.add_argument('--package-only',action='store_true',help='resume after completed physical experiments')
    p.add_argument('--prepared-plots',action='store_true',help='verify plots made in a separate existing analysis environment')
    a=p.parse_args()
    root=Path('results/value_v2');state=root/'release_state.json'
    def status(stage):
        state.write_text(json.dumps(dict(stage=stage,time=time.time()),indent=2));print(stage,flush=True)
    def run(args,log):
        with (root/log).open('w') as f:subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True)
    status('waiting_for_main_experiments')
    while json.loads((root/'round_state.json').read_text())['stage']!='complete':time.sleep(15)
    if not a.package_only:
        status('actual_budget_and_scale_curves')
        run([sys.executable,'scripts/run_value_v2_curves.py','--workers','4'],'curves.log')
        model=json.loads((root/'models/validation_recommendation.json').read_text())['checkpoint']
        status('fresh_configuration_execution')
        run([sys.executable,'-m','simbench.value.research_decision','--checkpoint',model,'--out',str(root/'new_configuration'),
             '--seed','30000','--n','32','--k','4','--budget','8','--repeats','2','--mode','best_within_budget'],'new_configuration.log')
    elif not all((root/name).exists() for name in ('online_curves/results.json','new_configuration/decision.json')):
        raise RuntimeError('physical curves/new-configuration execution are not complete')
    status('tables_and_release')
    run([sys.executable,'scripts/summarize_value_v2.py'],'summarize.log')
    run([sys.executable,'scripts/audit_value_v2_table.py'],'table_audit.json')
    if a.prepared_plots:
        out=Path('docs/evidence/value_v2');manifest=json.loads((out/'plot_manifest.json').read_text())
        digest=lambda obj:hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()
        assert manifest['locked_metrics_sha256']==digest(json.loads((root/'locked_metrics.json').read_text()))
        assert manifest['online_summary_sha256']==digest(json.loads((out/'online_summary.json').read_text()))
        for name,expected in manifest['files'].items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==expected
        (root/'plots.log').write_text('Verified externally prepared plot data and file hashes.\n')
    else:run([sys.executable,'scripts/plot_value_v2.py'],'plots.log')
    run([sys.executable,'scripts/package_value_v2.py','--code-commit',a.code_commit],'package.log')
    status('complete')


if __name__=='__main__':main()
