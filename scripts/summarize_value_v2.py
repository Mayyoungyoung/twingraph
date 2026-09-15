"""Post-execution tables; independent reference never enters the online policy."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def reference_groups(root):
    result={}
    for f in root.glob('data/group_*/complete.json'):
        d=f.parent;inputs=json.loads((d/'inputs.json').read_text())
        if inputs['declared_split']!='test':continue
        task=inputs['task'];key=(task['family'],task['seed'],inputs['checkpoint'])
        if key in result:raise ValueError('multiple locked references for a configuration')
        trials=json.loads((d/'outcomes.json').read_text())['trials']
        if any(t['trial']['domain']!='reference' or not t['valid'] for t in trials):
            raise ValueError('reference domain/validity mismatch')
        rates={p['id']:np.mean([t['success'] for t in trials if t['candidate_id']==p['id']]) for p in inputs['candidates']}
        result[key]=(inputs,rates)
    return result


def reference_quality(record,inputs,reference):
    """Called only after physical deployment; no reference used to choose plans."""
    key=(record['task']['family'],record['task']['seed'],record['task'].get('checkpoint',0))
    if key not in reference:return {}
    ref,rates=reference[key]
    plans=inputs['candidates']
    if {p['id'] for p in plans}!=set(rates):return {}  # larger pools lack exhaustive labels
    ref_plans={p['id']:p for p in ref['candidates']}
    for p in plans:
        other=ref_plans[p['id']]
        if p['calls']!=other['calls'] or p['prefix']['choices']!=other['prefix']['choices']:
            raise ValueError('same ID with different executed plan')
    best=float(max(rates.values()));selected=record['selection']
    top=max((rates[p['candidate_id']] for p in selected['top_k']),default=0.)
    chosen=float(rates[selected['chosen']['id']]) if selected['chosen'] else 0.
    return dict(reference_repeats=2,pool_type='all_failure' if best==0 else 'all_success' if min(rates.values())==1 else 'mixed',
                pool_best_reference=best,near_optimal_hit_at_k=float(top>=best-.1) if best>0 else None,
                feasible_hit_at_k=float(top>0),regret_at_k=best-top,
                selected_reference=chosen,selected_regret=best-chosen)


def mean_cluster(rows,key):
    values={}
    for row in rows:
        if row.get(key) is not None:values.setdefault((row['family'],row['seed']),[]).append(row[key])
    return float(np.mean([np.mean(v) for v in values.values()])) if values else None


def summarize(rows):
    summaries=[]
    keys=('runset','method','protocol','n','k','budget','repeats')
    for group in sorted({tuple(r[k] for k in keys) for r in rows}):
        common=[r for r in rows if tuple(r[k] for k in keys)==group]
        for family in ['all',*sorted({r['family'] for r in common})]:
            chosen=[r for r in common if family=='all' or r['family']==family]
            times=np.array([r['decision_seconds'] for r in chosen])
            summary=dict(zip(keys,group));summary.update(family=family,decisions=len(chosen),
                         configurations=len({(r['family'],r['seed']) for r in chosen}),
                         success=mean_cluster(chosen,'independent_execution_success'),
                         mean_seconds=float(times.mean()),median_seconds=float(np.median(times)),p95_seconds=float(np.quantile(times,.95)),
                         mean_rollouts=mean_cluster(chosen,'rollouts'),exhausted=mean_cluster(chosen,'budget_exhausted'))
            for key in ('near_optimal_hit_at_k','feasible_hit_at_k','regret_at_k','selected_regret'):
                summary[key]=mean_cluster(chosen,key)
            summaries.append(summary)
    return summaries


def paired(rows):
    outputs=[];settings=('runset','protocol','n','k','budget','repeats')
    for condition in sorted({tuple(r[k] for k in settings) for r in rows}):
        group=[r for r in rows if tuple(r[k] for k in settings)==condition]
        baseline={(r['family'],r['seed'],r['checkpoint']):r for r in group if r['method']=='geometry'}
        for method in sorted({r['method'] for r in group}-{'geometry','exhaustive'}):
            differences={}
            for row in group:
                key=(row['family'],row['seed'],row['checkpoint'])
                if row['method']!=method or key not in baseline:continue
                base=baseline[key]
                differences.setdefault(key[:2],[]).append([base['decision_seconds']-row['decision_seconds'],row['independent_execution_success']-base['independent_execution_success']])
            values=np.array([np.mean(v,axis=0) for v in differences.values()])
            if not len(values):continue
            interval=[[None,None],[None,None]]
            if len(values)>1:
                rng=np.random.default_rng(831);boot=values[rng.integers(0,len(values),(2000,len(values)))].mean(1)
                interval=np.quantile(boot,[.025,.975],axis=0).tolist()
            item=dict(zip(settings,condition));item.update(method=method,configurations=len(values),mean=values.mean(0).tolist(),
                 interval95=interval,columns=['seconds_saved_vs_geometry','success_difference_vs_geometry'])
            outputs.append(item)
    return outputs


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/value_v2');p.add_argument('--out',default='docs/evidence/value_v2');a=p.parse_args()
    root=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);refs=reference_groups(root)
    rows=[];identities=set()
    for f in sorted(root.glob('online*/**/decision.json')):
        r=json.loads(f.read_text());q=r['selection'];t=r['timing'];runset=f.relative_to(root).parts[0]
        inputs=json.loads((f.parent/'inputs.json').read_text())
        row=dict(artifact=str(f),runset=runset,family=r['task']['family'],seed=r['task']['seed'],checkpoint=r['task'].get('checkpoint',0),
                 method=r['method'],checkpoint_sha256=r['checkpoint_sha256'],input_sha256=r['input_sha256'],protocol=q['mode'],
                 n=r['pool_counts']['materializable'],k=q['k'],budget=q['budget'],repeats=q['repeats'],accept_rate=q['accept_rate'],allow_expand=q['allow_expand'],
                 independent_execution_success=r['execution_success_rate'],deployment_rollouts=len(r['deployment']),
                 deployment_physics_steps=sum(d['physics_steps'] for d in r['deployment']),decision_seconds=t['decision_wall_seconds'],
                 scene_setup_seconds=t['scene_setup_seconds'],model_cold_seconds=r['model_cold_start_seconds'],visual_cold_seconds=r['cold_visual_seconds'],
                 candidate_generation_seconds=t['candidate_generation_seconds'],necessary_geometry_seconds=t['necessary_geometry_seconds'],
                 optional_geometry_seconds=t['optional_geometry_seconds'],render_seconds=t['render_seconds'],visual_encoding_seconds=t['visual_encoding_seconds'],
                 network_seconds=t['network_inference_seconds'],numeric_features_seconds=t['numeric_features_seconds'],
                 restore_seconds=t['snapshot_restore_seconds'],twin_seconds=t['twin_rollout_seconds'],
                 deferred_solving_seconds_subset=t['deferred_solving_seconds_subset_of_rollout'],deployment_seconds=t['independent_execution_seconds'],
                 later_parameter_solving_seconds_subset=t.get('later_parameter_solving_seconds_subset_of_rollout'),
                 rollouts=q['validation_calls'],unique_candidates=q['unique_candidates'],physics_steps=r['physics_steps'],
                 timeouts=sum(d.get('timeout',False) for d in q['validated']),
                 budget_exhausted=int(q['budget_exhausted']),budget_fully_spent=int(q['budget_fully_spent']),accepted=int(q['chosen'] is not None),
                 **reference_quality(r,inputs,refs))
        identity=tuple(row[k] for k in ('runset','family','seed','checkpoint','method','protocol','n','k','budget','repeats'))
        if identity in identities:raise ValueError('duplicate experimental cell')
        identities.add(identity);rows.append(row)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    if rows:
        with (out/'EXPERIMENT_TABLE.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        Path('EXPERIMENT_TABLE.csv').write_bytes((out/'EXPERIMENT_TABLE.csv').read_bytes())
    (out/'online_summary.json').write_text(json.dumps(summarize(rows),indent=2))
    (out/'paired_intervals.json').write_text(json.dumps(paired(rows),indent=2))
    print(json.dumps(dict(decisions=len(rows),runsets=sorted({r['runset'] for r in rows}),output=str(out))))


if __name__=='__main__':main()

