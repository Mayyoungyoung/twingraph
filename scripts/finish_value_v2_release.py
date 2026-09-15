"""Finish actual curves, a fresh executable Top-K, and the reproducible release."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--code-commit',required=True);a=p.parse_args()
    root=Path('results/value_v2');state=root/'release_state.json'
    def status(stage):
        state.write_text(json.dumps(dict(stage=stage,time=time.time()),indent=2));print(stage,flush=True)
    def run(args,log):
        with (root/log).open('w') as f:subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True)
    status('waiting_for_main_experiments')
    while json.loads((root/'round_state.json').read_text())['stage']!='complete':time.sleep(15)
    status('actual_budget_and_scale_curves')
    run([sys.executable,'scripts/run_value_v2_curves.py','--workers','4'],'curves.log')
    model=json.loads((root/'models/validation_recommendation.json').read_text())['checkpoint']
    status('fresh_configuration_execution')
    run([sys.executable,'-m','simbench.value.research_decision','--checkpoint',model,'--out',str(root/'new_configuration'),
         '--seed','30000','--n','32','--k','4','--budget','8','--repeats','2','--mode','best_within_budget'],'new_configuration.log')
    status('tables_and_release')
    run([sys.executable,'scripts/summarize_value_v2.py'],'summarize.log')
    run([sys.executable,'scripts/audit_value_v2_table.py'],'table_audit.json')
    run([sys.executable,'scripts/plot_value_v2.py'],'plots.log')
    run([sys.executable,'scripts/package_value_v2.py','--code-commit',a.code_commit],'package.log')
    status('complete')


if __name__=='__main__':main()
