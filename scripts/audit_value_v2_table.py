"""Read-only consistency audit of the release CSV against raw decision records."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--table',default='EXPERIMENT_TABLE.csv');a=p.parse_args()
    rows=list(csv.DictReader(Path(a.table).open(encoding='utf-8-sig')))
    if not rows:raise ValueError('empty experiment table')
    domains=set();total=0;identities=set();shared_inputs={};shared_perturbations={}
    for row in rows:
        record=json.loads(Path(row['artifact']).read_text());selection=record['selection']
        inputs=json.loads(Path(row['artifact']).with_name('inputs.json').read_text())
        configuration=(row['family'],row['seed'],row['checkpoint'])
        pool=configuration+(row['n'],)
        signature=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()
        assert shared_inputs.setdefault(pool,signature)==signature, 'methods did not receive identical pre-execution inputs'
        assert float(row['decision_seconds'])==record['timing']['decision_wall_seconds']
        assert float(row['independent_execution_success'])==record['execution_success_rate']
        assert int(row['rollouts'])==len(selection['validated'])<=int(row['budget'])
        assert int(row['unique_candidates'])==len({r['candidate_id'] for r in selection['validated']})
        top={r['candidate_id'] for r in selection['top_k']}
        if not selection['allow_expand']:assert all(r['candidate_id'] in top for r in selection['validated'])
        assert all(r['trial']['domain']=='online' for r in selection['validated'])
        assert all(r['trial']['domain']=='deployment' for r in record['deployment'])
        if selection['chosen']:assert all(r['candidate_id']==selection['chosen']['id'] for r in record['deployment'])
        assert all(r['valid'] for r in [*selection['validated'],*record['deployment']])
        for trial in [*selection['validated'],*record['deployment']]:
            assert abs(trial['prefix_parameter_solving_seconds']+trial['later_parameter_solving_seconds']-trial['deferred_solving_seconds'])<1e-8
            perturbation=trial['trial'];key=configuration+(perturbation['domain'],perturbation['repeat'])
            assert shared_perturbations.setdefault(key,perturbation)==perturbation, 'unpaired method/candidate perturbation'
        components=('candidate_generation_seconds','necessary_geometry_seconds','optional_geometry_seconds',
                    'render_seconds','visual_encoding_seconds','network_inference_seconds','numeric_features_seconds',
                    'snapshot_restore_seconds','twin_rollout_seconds')
        assert sum(record['timing'][k] for k in components)<=record['timing']['decision_wall_seconds']+.001
        identity=tuple(row[k] for k in ('runset','family','seed','checkpoint','method','protocol','n','k','budget','repeats'))
        assert identity not in identities;identities.add(identity);total+=int(row['rollouts'])+len(record['deployment'])
        domains.update(r['trial']['domain'] for r in [*selection['validated'],*record['deployment']])
        # Blank reference for a pool without complete reference labels is intentional.
        if int(row['n'])>16:assert not row.get('regret_at_k')
    print(json.dumps(dict(status='passed',decisions=len(rows),physical_rollouts=total,domains=sorted(domains),
                         shared_input_pools=len(shared_inputs),paired_perturbations=len(shared_perturbations),
                         audit='read-only CSV-to-raw consistency, shared inputs, paired perturbations, budget/identity/domain checks; no file modified')))


if __name__=='__main__':main()
