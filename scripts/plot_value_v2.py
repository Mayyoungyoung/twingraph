"""Scientific plots and offline strata, always sourced from completed experiments."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


LABELS={'geometry':'Geometry','prior_17':'Fitted geo prior','mlp_17':'MLP','residual_17':'Geo + residual','no_vision_17':'Plan Transformer','vision_17':'Vision + plan','v1_fixed':'Frozen v1','pin_mlp_17':'Pin-only MLP','fewshot_mlp_17':'Adapted MLP'}


def cluster(rows,key):
    values={}
    for r in rows:
        if r[key] is not None:values.setdefault((r['family'],r['seed']),[]).append(r[key])
    return float(np.mean([np.mean(v) for v in values.values()])) if values else None


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/value_v2');p.add_argument('--out',default='docs/evidence/value_v2');p.add_argument('--offline-only',action='store_true');a=p.parse_args()
    root=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    offline=json.loads((root/'locked_metrics.json').read_text());online=[] if a.offline_only else json.loads((out/'online_summary.json').read_text());rows=[]
    for name,ks in offline.items():
        for k,value in ks.items():
            source=value['rows']
            strata=[('all','all',source)]
            for field in ('family','length','pool_type','informative'):
                for v in sorted({r[field] for r in source},key=str):strata.append((field,str(v),[r for r in source if r[field]==v]))
            for field,v,selected in strata:
                row=dict(model=name,k=int(k),stratum=field,value=v,decision_groups=len(selected),configurations=len({(r['family'],r['seed']) for r in selected}))
                for key in ('hit','feasible','regret','brier'):row[key]=None if name=='geometry' and key=='brier' else cluster(selected,key)
                rows.append(row)
    with (out/'OFFLINE_RANKING_TABLE.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    names=[n for n in LABELS if n in offline]
    fig,axes=plt.subplots(1,2,figsize=(13,5.5))
    for axis,family in zip(axes,['sliding_stage_pin','rigid_connector_module']):
        x=np.arange(len(names));width=.38
        for offset,k in [(-.5,1),(.5,4)]:
            values=[100*cluster([r for r in offline[n][str(k)]['rows'] if r['family']==family],'hit') for n in names]
            axis.bar(x+offset*width,values,width,label=f'Near-optimal Hit@{k}')
        axis.set_xticks(x,[LABELS[n] for n in names],rotation=35,ha='right');axis.set_ylim(0,105);axis.set_ylabel('Configuration-weighted hit (%)');axis.set_title(family.replace('_',' '))
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.94),ncol=2,frameon=False)
    fig.suptitle('Locked simulation reference: 6 independent configurations per family; 2 repeats/candidate',fontsize=12,y=.99)
    fig.subplots_adjust(left=.06,right=.99,bottom=.27,top=.81,wspace=.16)
    fig.savefig(out/'locked_ranking.png',dpi=180);fig.savefig(out/'locked_ranking.svg');plt.close(fig)
    if a.offline_only:
        print(json.dumps(dict(offline_rows=len(rows),plots=str(out))));return
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for axis,family in zip(axes,['sliding_stage_pin','rigid_connector_module']):
        selected=[r for r in online if r['runset']=='online' and r['family']==family]
        for r in selected:
            axis.scatter(r['median_seconds'],100*r['success'],marker='o' if r['protocol']=='first_verified' else 's',s=40)
            axis.annotate(r['method']+(' A' if r['protocol']=='first_verified' else ' B'),(r['median_seconds'],100*r['success']),fontsize=7,xytext=(3,4),textcoords='offset points')
        axis.set_xlabel('Measured median decision time (s)');axis.set_ylabel('Independent deployment success (%)');axis.set_ylim(-3,110);axis.set_title(family.replace('_',' '));axis.grid(alpha=.2)
    fig.suptitle('Actual execution: A = first verified, B = best within budget (3 configurations/family)')
    fig.savefig(out/'execution_cost.png',dpi=180);fig.savefig(out/'execution_cost.svg');plt.close(fig)
    curve=[r for r in online if r['runset'] in {'online_budget','online_scale'} and r['family']!='all']
    if curve:
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        for family in sorted({r['family'] for r in curve}):
            for method in sorted({r['method'] for r in curve}):
                selected=sorted([r for r in curve if r['family']==family and r['method']==method and r['runset']=='online_budget'],key=lambda r:r['budget'])
                axes[0].plot([r['budget'] for r in selected],[100*r['success'] for r in selected],'o-',label=family.split('_')[0]+' / '+method)
                selected=sorted([r for r in curve if r['family']==family and r['method']==method and r['runset']=='online_scale'],key=lambda r:r['n'])
                axes[1].plot([r['n'] for r in selected],[r['median_seconds'] for r in selected],'o-',label=family.split('_')[0]+' / '+method)
        axes[0].set(xlabel='Physical rollout budget B',ylabel='Independent deployment success (%)',ylim=(-3,105));axes[1].set(xlabel='Nested candidate pool N',ylabel='Measured median decision time (s)')
        for axis in axes:axis.legend(fontsize=7);axis.grid(alpha=.2)
        fig.suptitle('Exploratory curves: tiny configuration counts; no 32/64-pool reference regret claim')
        fig.savefig(out/'budget_scale.png',dpi=180);fig.savefig(out/'budget_scale.svg');plt.close(fig)
    print(json.dumps(dict(offline_rows=len(rows),plots=str(out))))


if __name__=='__main__':main()
