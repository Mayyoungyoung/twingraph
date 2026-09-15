"""Artifact tables and group-clustered intervals; never fabricate missing rows."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def clustered_interval(records,key,seed=971):
    clusters={}
    for r in records:clusters.setdefault((r['task']['family'],r['task']['seed']),[]).append(r[key])
    values=np.array([np.mean(v) for v in clusters.values()]);rng=np.random.default_rng(seed)
    if len(values)<2:return [None,None]
    boot=rng.choice(values,(2000,len(values)),replace=True).mean(1)
    return np.quantile(boot,[.025,.975]).tolist()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/value_v2');p.add_argument('--out',default='docs/evidence/value_v2');a=p.parse_args()
    root=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for f in root.glob('online*/**/decision.json'):
        r=json.loads(f.read_text());q=r['selection'];t=r['timing']
        rows.append(dict(artifact=str(f),family=r['task']['family'],seed=r['task']['seed'],checkpoint=r['task'].get('checkpoint',0),
                         method=r['method'],protocol=q['mode'],n=r['pool_counts']['materializable'],k=q['k'],budget=q['budget'],repeats=q['repeats'],
                         independent_execution_success=r['execution_success_rate'],decision_seconds=t['decision_wall_seconds'],
                         optional_geometry_seconds=t['optional_geometry_seconds'],network_seconds=t['network_inference_seconds'],
                         twin_seconds=t['twin_rollout_seconds'],rollouts=q['validation_calls'],physics_steps=r['physics_steps'],
                         budget_exhausted=int(q['budget_exhausted']),budget_fully_spent=int(q['budget_fully_spent']),accepted=int(q['chosen'] is not None)))
    if rows:
        with (out/'EXPERIMENT_TABLE.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=[]
    for group in sorted({(r['method'],r['protocol'],r['n'],r['k'],r['budget']) for r in rows}):
        selected=[r for r in rows if (r['method'],r['protocol'],r['n'],r['k'],r['budget'])==group]
        times=np.array([r['decision_seconds'] for r in selected])
        summary.append(dict(method=group[0],protocol=group[1],n=group[2],k=group[3],budget=group[4],decisions=len(selected),
                     configurations=len({(r['family'],r['seed']) for r in selected}),
                     success=float(np.mean([r['independent_execution_success'] for r in selected])),
                     median_seconds=float(np.median(times)),p95_seconds=float(np.quantile(times,.95)),
                     mean_rollouts=float(np.mean([r['rollouts'] for r in selected])),
                     exhausted=float(np.mean([r['budget_exhausted'] for r in selected])) )
    (out/'online_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    # Paired bootstrap by geometry; checkpoint siblings are never independent.
    pairs=[]
    for protocol in sorted({r['protocol'] for r in rows}):
        for method in sorted({r['method'] for r in rows if r['method'] not in {'geometry','exhaustive'}}):
            diffs={}
            for r in rows:
                if r['protocol']!=protocol or r['method']!=method:continue
                base=[b for b in rows if b['method']=='geometry' and all(b[k]==r[k] for k in ('family','seed','checkpoint','protocol','n','k','budget'))]
                if not base:continue
                diffs.setdefault((r['family'],r['seed']),[]).append([base[0]['decision_seconds']-r['decision_seconds'],r['independent_execution_success']-base[0]['independent_execution_success']])
            values=np.array([np.mean(v,axis=0) for v in diffs.values()])
            if len(values)<2:continue
            rng=np.random.default_rng(831);b=values[rng.integers(0,len(values),(2000,len(values)))].mean(1)
            pairs.append(dict(method=method,protocol=protocol,configurations=len(values),mean=values.mean(0).tolist(),
                              interval95=np.quantile(b,[.025,.975],axis=0).tolist(),columns=['seconds_saved_vs_geometry','success_difference_vs_geometry']))
    (out/'paired_intervals.json').write_text(json.dumps(pairs,indent=2))


if __name__=='__main__':main()
