"""Layout-held-out V32 full-program value learning and frozen offline evaluation."""
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import time
import numpy as np
from simbench.value.generic_graph_value_v15 import AtomicValueNetV15, FEATURES, RELATIONS, SCHEMA, collate, parameter_count


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8'); tmp.replace(path)


def grouped(rows):
    groups=defaultdict(list)
    for row in rows: groups[row['seed']].append(row)
    return {s:sorted(g,key=lambda r:r['id']) for s,g in sorted(groups.items())}


def binary_metrics(y, p):
    y=np.asarray(y,dtype=bool); p=np.asarray(p,dtype=float); positive=p[y]; negative=p[~y]
    q=np.clip(p,1.e-7,1-1.e-7)
    auc=float(((positive[:,None]>negative[None,:])+.5*(positive[:,None]==negative[None,:])).mean()) if len(positive) and len(negative) else None
    # Average precision at distinct thresholds handles score ties as a group.
    ap=0.; previous=0
    for threshold in sorted(set(p),reverse=True):
        selected=p>=threshold; tp=int(y[selected].sum())
        if y.sum(): ap+=(tp-previous)/int(y.sum())*tp/int(selected.sum())
        previous=tp
    return dict(accuracy=float(((p>=.5)==y).mean()),
        balanced_accuracy=float(.5*((positive>=.5).mean()+(negative<.5).mean())) if len(positive) and len(negative) else None,
        roc_auc=auc,average_precision=float(ap) if y.any() else None,brier=float(((p-y)**2).mean()),
        nll=float(-(y*np.log(q)+(~y)*np.log(1-q)).mean()))


def random_expected(rows, k):
    n=len(rows); m=sum(r['label'] for r in rows); k=min(k,n)
    miss=lambda j: math.comb(n-m,j)/math.comb(n,j) if n-m>=j else 0.
    hit=1-miss(k); calls=sum(miss(j) for j in range(k))
    if not m: seconds=k/n*sum(r.get('seconds',0.) for r in rows)
    else:
        negative_weight=sum(math.comb(n-1-m,j)/math.comb(n-1,j) if n-1-m>=j else 0. for j in range(k))/n
        seconds=sum(r.get('seconds',0.)*(hit/m if r['label'] else negative_weight) for r in rows)
    return dict(success_rate=hit,calls=calls,seconds=seconds)


def ordered(rows, order, k):
    chosen=[]
    for index in list(order)[:k]:
        chosen.append(rows[index])
        if rows[index]['label']: break
    return dict(success=bool(chosen and chosen[-1]['label']),calls=len(chosen),
        seconds=sum(r.get('seconds',0.) for r in chosen),selected=chosen[-1]['id'] if chosen and chosen[-1]['label'] else None)


def rank_metrics(rows, probabilities, k=4):
    scores={r['id']:float(p) for r,p in zip(rows,probabilities)}; layouts=[]; correct=total=0.
    for seed, group in grouped(rows).items():
        p=np.asarray([scores[r['id']] for r in group]); y=np.asarray([r['label'] for r in group],bool)
        order=np.argsort(-p,kind='stable').tolist()
        pos=p[y]; neg=p[~y]
        correct+=float(((pos[:,None]>neg[None,:])+.5*(pos[:,None]==neg[None,:])).sum()); total+=len(pos)*len(neg)
        layouts.append(dict(seed=seed,positives=int(y.sum()),value=ordered(group,order,k),random=random_expected(group,k),
            full=dict(success=bool(y.any()),calls=len(group),seconds=sum(r.get('seconds',0.) for r in group)),
            value_order=[group[i]['id'] for i in order]))
    return dict(**binary_metrics([r['label'] for r in rows],probabilities),
        pair_accuracy=correct/total if total else None,hit=float(np.mean([r['value']['success'] for r in layouts])),
        calls=float(np.mean([r['value']['calls'] for r in layouts])),layouts=layouts)


def select_key(report, epoch):
    return (-report['hit'],report['calls'],-(report['pair_accuracy'] or 0.),report['brier'],epoch)


def load_rows(exported, split, raw_root=None):
    rows=[]
    for line in (exported/'labels.jsonl').read_text().splitlines():
        row=json.loads(line)
        if row['split']!=split: continue
        with np.load(exported/row['input']) as z: encoded={k:z[k] for k in z.files}
        if any(not np.isfinite(a).all() for a in encoded.values()): raise ValueError('nonfinite features')
        row.update(encoded=encoded,input_sha256=sha(exported/row['input']))
        if raw_root:
            result=read(raw_root/f"seed_{row['seed']}"/'candidates'/row['id'].split('_',2)[2]/'result.json')
            if not result['valid'] or result.get('resource_censored'): raise ValueError('invalid test record')
            row['seconds']=float(result['total_wall_seconds'])
        rows.append(row)
    return sorted(rows,key=lambda r:(r['seed'],r['id']))


def score(model, rows, device, temperature=1.):
    import torch
    model.eval(); logits=[]
    with torch.inference_mode():
        for start in range(0,len(rows),32):
            logits.extend(model(collate([r['encoded'] for r in rows[start:start+32]],device)).cpu().tolist())
    logits=np.asarray(logits)
    return logits,1/(1+np.exp(-np.clip(logits/temperature,-50,50)))


def train_run(train, validation, initialization, device, out, *, max_epochs=80, min_epochs=20, patience=15):
    import torch
    from torch.nn import functional as F
    random.seed(initialization); np.random.seed(initialization); torch.manual_seed(initialization)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(initialization)
    model=AtomicValueNetV15.build(width=64,message_layers=2).to(device)
    x=np.concatenate([r['encoded']['x'] for r in train]); mean=x.mean(0); scale=np.maximum(x.std(0),.1)
    with torch.no_grad():
        model.mean.copy_(torch.as_tensor(mean,device=device)); model.scale.copy_(torch.as_tensor(scale,device=device))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.03)
    groups=list(grouped(train).values()); best=None; trace=[]; stale=0; began=time.perf_counter()
    for epoch in range(1,max_epochs+1):
        model.train(); shuffled=list(groups); random.Random(initialization*10000+epoch).shuffle(shuffled); losses=[]
        for start in range(0,len(shuffled),4):
            batch_groups=shuffled[start:start+4]; batch=[r for g in batch_groups for r in g]
            logits=model(collate([r['encoded'] for r in batch],device)); offset=0; terms=[]
            for group in batch_groups:
                z=logits[offset:offset+len(group)]; offset+=len(group)
                y=torch.tensor([r['label'] for r in group],dtype=torch.float32,device=device); pos=y>.5
                ranking=F.softplus(-(z[pos,None]-z[None,~pos])).mean() if pos.any() and (~pos).any() else z.sum()*0
                terms.append(F.binary_cross_entropy_with_logits(z,y)+.5*ranking)
            loss=torch.stack(terms).mean(); optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.); optimizer.step(); losses.append(float(loss.detach()))
        _,prob=score(model,validation,device); report=rank_metrics(validation,prob)
        key=select_key(report,epoch)
        trace.append(dict(epoch=epoch,loss=float(np.mean(losses)),validation={k:v for k,v in report.items() if k!='layouts'}))
        if best is None or key<best['key']:
            best=dict(key=key,epoch=epoch,state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},report=report); stale=0
        else: stale+=1
        write(out/f'trace_{initialization}.json',trace)
        print(json.dumps(dict(event='epoch',initialization=initialization,epoch=epoch,loss=trace[-1]['loss'],validation_hit4=report['hit'])),flush=True)
        if epoch>=min_epochs and stale>=patience: break
    model.load_state_dict(best['state'])
    return model,best,dict(initialization=initialization,selected_epoch=best['epoch'],selection_key=best['key'],
        trained_epochs=len(trace),training_seconds=time.perf_counter()-began,validation=best['report'])


def calibrate(logits, labels):
    import torch
    from torch.nn import functional as F
    z=torch.tensor(logits,dtype=torch.float64); y=torch.tensor(labels,dtype=torch.float64)
    log_t=torch.zeros((),dtype=torch.float64,requires_grad=True)
    optimizer=torch.optim.LBFGS([log_t],lr=.1,max_iter=60,line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad(); loss=F.binary_cross_entropy_with_logits(z/log_t.clamp(math.log(.25),math.log(4.)).exp(),y)
        loss.backward(); return loss
    optimizer.step(closure)
    t=float(log_t.detach().clamp(math.log(.25),math.log(4.)).exp())
    if float(F.binary_cross_entropy_with_logits(z/t,y))>=float(F.binary_cross_entropy_with_logits(z,y)): t=1.
    return t


def bootstrap(values):
    v=np.asarray(values,float); rng=np.random.default_rng(320927)
    means=v[rng.integers(0,len(v),size=(10000,len(v)))].mean(1)
    return dict(mean=float(v.mean()),layout_bootstrap_95=[float(x) for x in np.quantile(means,[.025,.975])])


def offline_test(rows, probabilities):
    reports={str(k):rank_metrics(rows,probabilities,k) for k in (1,2,4,8)}
    comparisons={}
    for k,report in reports.items():
        layouts=report['layouts']
        comparisons[k]=dict(value_success=bootstrap([r['value']['success'] for r in layouts]),
            random_expected_success=bootstrap([r['random']['success_rate'] for r in layouts]),
            full_success=bootstrap([r['full']['success'] for r in layouts]),
            value_calls=float(np.mean([r['value']['calls'] for r in layouts])),
            random_expected_calls=float(np.mean([r['random']['calls'] for r in layouts])),
            full_calls=8,value_twin_seconds_estimate=float(np.mean([r['value']['seconds'] for r in layouts])),
            random_twin_seconds_estimate=float(np.mean([r['random']['seconds'] for r in layouts])),
            full_twin_seconds_estimate=float(np.mean([r['full']['seconds'] for r in layouts])),
            paired_success_difference=bootstrap([r['value']['success']-r['random']['success_rate'] for r in layouts]))
    return dict(classification={k:v for k,v in reports['4'].items() if k not in ('layouts','hit','calls')},
        by_k=comparisons,layouts=reports['4']['layouts'],
        time_definition='Offline sum of per-candidate wall times recorded during parallel collection, not measured fresh workflow wall time.')


def train(exported, raw_root, out, audit_path):
    import torch
    audit=read(audit_path)
    if not audit.get('passed') or not audit.get('dataset_complete') or audit.get('samples')!=1000:
        raise ValueError('complete 1000-sample audit required')
    if (out/'selection_frozen.json').exists():
        if (out/'offline_test.json').exists(): return read(out/'training_report.json')
        raise ValueError('selection already frozen: resume evaluation explicitly, do not retrain')
    out.mkdir(parents=True,exist_ok=True); torch.set_num_threads(2)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda': torch.backends.cudnn.benchmark=False
    train_rows=load_rows(exported,'train'); validation=load_rows(exported,'validation')
    if len(train_rows)!=800 or len(validation)!=96 or len(grouped(train_rows))!=100 or len(grouped(validation))!=12:
        raise ValueError('unexpected fixed split')
    if set(grouped(train_rows))&set(grouped(validation)): raise ValueError('layout leakage')
    trials=[]; selected=None
    for initialization in (7,17,29):
        model,best,report=train_run(train_rows,validation,initialization,device,out); trials.append(report)
        torch.save(dict(state_dict=best['state'],model_config=model.config),out/f'initialization_{initialization}.pt')
        if selected is None or best['key']<selected['key']:
            selected=dict(key=best['key'],state=best['state'],initialization=initialization,epoch=best['epoch'])
        del model
    model=AtomicValueNetV15.build(width=64,message_layers=2).to(device); model.load_state_dict(selected['state'])
    logits,_=score(model,validation,device); temperature=calibrate(logits,[r['label'] for r in validation])
    checkpoint=dict(schema=SCHEMA,state_dict=selected['state'],model_config=model.config,temperature=temperature,
        features=FEATURES,relations=RELATIONS,task='V32 full assembly through released handle',
        train_seeds=list(grouped(train_rows)),validation_seeds=list(grouped(validation)),
        selected_initialization=selected['initialization'],selected_epoch=selected['epoch'])
    torch.save(checkpoint,out/'value_v32.pt')
    report=dict(checkpoint_sha256=sha(out/'value_v32.pt'),dataset_labels_sha256=sha(exported/'labels.jsonl'),
        audit_sha256=sha(audit_path),training_script_sha256=sha(__file__),model_parameters=parameter_count(model),device=device,
        torch_version=torch.__version__,gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,
        selected_initialization=selected['initialization'],selected_epoch=selected['epoch'],temperature=temperature,
        attempts=trials,train_seeds=list(grouped(train_rows)),validation_seeds=list(grouped(validation)),
        training_input_hashes={r['id']:r['input_sha256'] for r in train_rows+validation})
    write(out/'training_report.json',report)
    write(out/'selection_frozen.json',dict(checkpoint_sha256=report['checkpoint_sha256'],frozen_at=time.time(),
        selected_initialization=selected['initialization'],selected_epoch=selected['epoch'],temperature=temperature,
        test_used_for_selection=False,primary_k=4,supplemental_k=[1,2,4,8]))
    # Test data is loaded only after all model/epoch/calibration choices freeze.
    test=load_rows(exported,'test',raw_root)
    if len(test)!=104 or len(grouped(test))!=13: raise ValueError('unexpected fixed test split')
    if set(grouped(test))&(set(grouped(train_rows))|set(grouped(validation))): raise ValueError('test layout leakage')
    _,probabilities=score(model,test,device,temperature)
    write(out/'offline_test.json',offline_test(test,probabilities))
    write(out/'test_predictions.json',[dict(id=r['id'],seed=r['seed'],probability=float(p),label=r['label']) for r,p in zip(test,probabilities)])
    print(json.dumps(dict(event='training_and_offline_test_complete',selected_initialization=selected['initialization'],selected_epoch=selected['epoch'])),flush=True)
    return report


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--exported',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--audit',type=Path,required=True)
    a=p.parse_args();train(a.exported,a.source,a.out,a.audit)
