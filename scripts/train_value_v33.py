"""Leakage-checked V33 feature ablations, robust multi-head training and freeze."""
import argparse
from copy import deepcopy
from pathlib import Path
import time
import numpy as np
from scripts.train_value_v32 import read,write,sha,grouped,binary_metrics,calibrate
from scripts.value_features_v33 import encode_candidate,FEATURES,BASE_FEATURES,SCHEMA
from scripts.collect_robust_value_v33 import checked,REPEATS
from simbench.value.generic_graph_value_v15 import AtomicValueNetV15,collate
from simbench.value.plan import digest


def build(config):
    import torch
    model=AtomicValueNetV15.build(**config)
    model.plan[-1]=torch.nn.Linear(config['width'],3)
    return model


def load_data(source,repeats,out):
    manifest=read(source/'manifest.json');repeat_manifest=read(repeats/'manifest.json')
    if read(repeats/'progress.json')['stage']!='complete':raise ValueError('all robust repeats required')
    if sha(source/'manifest.json')!=repeat_manifest['source_manifest_sha256']:raise ValueError('source manifest changed')
    data=[];hashes={}
    for job in manifest['jobs']:
        if job['split'] not in ('train','validation'):continue
        request=read(source/f"seed_{job['seed']}"/'request.json')
        if digest(request)!=repeat_manifest['request_sha256'][str(job['seed'])]:raise ValueError('frozen request changed')
        for entry in request['candidates']:
            encoded=encode_candidate(request,entry)
            original=source/f"seed_{job['seed']}"/'candidates'/entry['name']/'result.json';nominal=read(original)
            if not nominal['valid'] or nominal['plan_sha256']!=entry['plan_sha256']:raise ValueError('invalid nominal label')
            trials=[]
            for repeat in REPEATS:
                path=repeats/f"seed_{job['seed']}"/entry['name']/f'repeat_{repeat}'/'result.json'
                checked(path,request,entry,repeat);r=read(path);trials.append(r)
                hashes[str(path)]=sha(path)
            hashes[str(original)]=sha(original)
            id=f"seed_{job['seed']}_{entry['name']}"
            dest=out/'inputs'/id;dest.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest/'features.npz',**encoded)
            data.append(dict(id=id,seed=job['seed'],split=job['split'],encoded=encoded,label=int(nominal['success']),
                robust_labels=[int(r['success']) for r in trials],robust=float(np.mean([r['success'] for r in trials])),
                nominal_steps=nominal['physics_steps'],log_steps=float(np.log1p(np.mean([r['physics_steps'] for r in trials]))),
                input_sha256=sha(dest/'features.npz'),request_sha256=digest(request),plan_sha256=entry['plan_sha256']))
    train=[r for r in data if r['split']=='train'];val=[r for r in data if r['split']=='validation']
    if len(train)!=800 or len(val)!=96 or set(grouped(train))&set(grouped(val)):raise ValueError('incorrect layout split')
    write(out/'data_audit.json',dict(passed=True,train=800,validation=96,repeat_trials=len(data)*3,
        test_loaded=False,feature_schema=SCHEMA,features=list(FEATURES),raw_result_sha256=hashes,
        rows=[{k:v for k,v in r.items() if k!='encoded'} for r in data]))
    return train,val


def predict(model,rows,device,cost_mean,cost_scale):
    import torch
    model.eval();values=[]
    with torch.inference_mode():
        for start in range(0,len(rows),32):
            encoded=[r['encoded'] for r in rows[start:start+32]]
            dim=model.config['input_dim']
            encoded=[{**e,'x':e['x'][:,:dim]} for e in encoded]
            values.extend(model(collate(encoded,device)).cpu().numpy())
    values=np.asarray(values)
    p=1/(1+np.exp(-np.clip(values[:,:2],-40,40)))
    return np.column_stack([p,np.expm1(np.clip(values[:,2]*cost_scale+cost_mean,0,20))])


def ranking(predictions,alpha=0.,beta=0.):
    # predictions: ensemble x candidates x [nominal probability, robust probability, steps].
    mean=predictions.mean(0);uncertainty=predictions[:,:,1].std(0)
    cost=mean[:,2]/max(float(np.median(mean[:,2])),1.)
    return np.log(np.maximum(mean[:,1],1e-6))+.25*np.log(np.maximum(mean[:,0],1e-6))-alpha*np.log(np.maximum(cost,.01))-beta*uncertainty


def temperature_scale(prediction,temperatures):
    result=prediction.copy()
    for head,t in enumerate(temperatures):
        p=np.clip(result[:,head],1e-7,1-1e-7)
        result[:,head]=1/(1+np.exp(-np.clip(np.log(p/(1-p))/t,-40,40)))
    return result


def validation_metrics(rows,predictions,alpha=0.,beta=0.):
    scores=ranking(predictions,alpha,beta);indices={r['id']:i for i,r in enumerate(rows)}
    success=[];calls=[];costs=[]
    for group in grouped(rows).values():
        ids=[indices[r['id']] for r in group];order=np.argsort(-scores[ids],kind='stable')[:3]
        selected=None;steps=0.;n=0
        for i in order:
            r=group[int(i)];n+=1;steps+=r['nominal_steps']
            if r['label']:selected=r;break
        success.append(selected['robust'] if selected else 0.);calls.append(n);costs.append(steps)
    p=predictions.mean(0)[:,1]
    return dict(workflow_success=float(np.mean(success)),calls=float(np.mean(calls)),steps=float(np.mean(costs)),
        robust_brier=float(np.mean((p-np.asarray([r['robust'] for r in rows]))**2)),layout_success=success)


def key(metrics,epoch=0):
    return (-metrics['workflow_success'],metrics['steps'],metrics['robust_brier'],epoch)


def fit(train,val,variant,seed,out,max_epochs=100,min_epochs=25,patience=20):
    import torch
    from torch.nn import functional as F
    torch.manual_seed(seed);np.random.seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    dim=len(BASE_FEATURES) if variant=='base_nominal' else len(FEATURES)
    model=build(dict(input_dim=dim,width=64,message_layers=2)).to(device)
    x=np.concatenate([r['encoded']['x'][:,:dim] for r in train]);scale=np.maximum(x.std(0),.001)
    with torch.no_grad():model.mean.copy_(torch.tensor(x.mean(0),device=device));model.scale.copy_(torch.tensor(scale,device=device))
    cost_mean=float(np.mean([r['log_steps'] for r in train]));cost_scale=max(float(np.std([r['log_steps'] for r in train])),.1)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.03)
    groups=list(grouped(train).values());best=None;stale=0;trace=[]
    for epoch in range(1,max_epochs+1):
        model.train();rng=np.random.default_rng(seed*1000+epoch);permutation=rng.permutation(len(groups));losses=[]
        for start in range(0,len(groups),4):
            selected=[groups[i] for i in permutation[start:start+4]];batch=[r for g in selected for r in g]
            encoded=[{**r['encoded'],'x':r['encoded']['x'][:,:dim]} for r in batch]
            z=model(collate(encoded,device));offset=0;terms=[]
            for group in selected:
                value=z[offset:offset+len(group)];offset+=len(group)
                nominal=torch.tensor([r['label'] for r in group],dtype=torch.float32,device=device)
                target=torch.tensor([r['robust'] if variant=='enhanced_robust' else r['label'] for r in group],dtype=torch.float32,device=device)
                costs=torch.tensor([(r['log_steps']-cost_mean)/cost_scale for r in group],device=device,dtype=torch.float32)
                gap=target[:,None]-target[None,:];positive=gap>0
                rank=(gap[positive]*F.softplus(-(value[:,1,None]-value[None,:,1])[positive])).mean() if positive.any() else value.sum()*0
                terms.append(.5*F.binary_cross_entropy_with_logits(value[:,0],nominal)+F.binary_cross_entropy_with_logits(value[:,1],target)+.5*rank+.1*F.smooth_l1_loss(value[:,2],costs))
            loss=torch.stack(terms).mean();optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);optimizer.step();losses.append(float(loss.detach()))
        prediction=predict(model,val,device,cost_mean,cost_scale);metric=validation_metrics(val,prediction[None])
        trace.append(dict(epoch=epoch,loss=float(np.mean(losses)),validation=metric));candidate=key(metric,epoch)
        if best is None or candidate<best['key']:
            best=dict(key=candidate,epoch=epoch,state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},metrics=metric);stale=0
        else:stale+=1
        write(out/f'{variant}_{seed}_trace.json',trace)
        if epoch%10==0:print(dict(variant=variant,seed=seed,epoch=epoch,validation=metric),flush=True)
        if epoch>=min_epochs and stale>=patience:break
    model.load_state_dict(best['state'])
    saved=dict(model_config=model.config,state_dict=best['state'],cost_mean=cost_mean,cost_scale=cost_scale,
        variant=variant,seed=seed,selected_epoch=best['epoch'],validation=best['metrics'],schema=SCHEMA)
    path=out/f'{variant}_{seed}.pt';torch.save(saved,path)
    return path,predict(model,val,device,cost_mean,cost_scale)


def train(source,repeats,out):
    import torch
    torch.set_num_threads(2);out.mkdir(parents=True,exist_ok=True)
    if (out/'selection_frozen.json').exists():
        frozen=read(out/'selection_frozen.json')
        if all(sha(out/name)==value for name,value in frozen['checkpoint_sha256'].items()):return frozen
        raise ValueError('frozen model changed')
    train_rows,val=load_data(source,repeats,out)
    candidates=[];ablations={}
    for variant in ('base_nominal','enhanced_nominal','enhanced_robust'):
        predictions=[];paths=[]
        for seed in (7,17,29):
            path,pred=fit(train_rows,val,variant,seed,out)
            temperatures=[]
            for head,field in enumerate(('label','robust')):
                p=np.clip(pred[:,head],1e-7,1-1e-7)
                temperatures.append(calibrate(np.log(p/(1-p)),[r[field] for r in val]))
            saved=torch.load(path,map_location='cpu',weights_only=False);saved['temperatures']=temperatures;torch.save(saved,path)
            pred=temperature_scale(pred,temperatures);paths.append(path);predictions.append(pred)
            if variant!='base_nominal':candidates.append(([path],pred[None]))
        ablations[variant]=validation_metrics(val,np.asarray(predictions))
        if variant!='base_nominal':candidates.append((paths,np.asarray(predictions)))
    # Development cross-validation uses train layouts only, with fixed folds.
    cv=[];seeds=sorted(grouped(train_rows));folds=np.array_split(np.random.default_rng(330927).permutation(seeds),3)
    cvdir=out/'train_cv';cvdir.mkdir(exist_ok=True)
    for fold,held in enumerate(folds):
        held=set(int(s) for s in held);tr=[r for r in train_rows if r['seed'] not in held];va=[r for r in train_rows if r['seed'] in held]
        path,pred=fit(tr,va,'enhanced_robust',107+fold,cvdir)
        cv.append(dict(fold=fold,held_layouts=sorted(held),metrics=validation_metrics(va,pred[None])))
    choices=[]
    for paths,pred in candidates:
        for alpha in (0.,.1,.25):
            for beta in (0.,.5):
                m=validation_metrics(val,pred,alpha,beta)
                choices.append(dict(paths=[p.name for p in paths],alpha=alpha,beta=beta,validation=m))
    selected=min(choices,key=lambda r:(key(r['validation']),len(r['paths']),r['alpha'],r['beta']))
    report=dict(selected=selected,ablations=ablations,train_cv=cv,selection_candidates=choices,
        train_layouts=sorted(grouped(train_rows)),validation_layouts=sorted(grouped(val)),test_used=False,
        data_audit_sha256=sha(out/'data_audit.json'),training_script_sha256=sha(__file__))
    write(out/'training_report.json',report)
    frozen=dict(**selected,checkpoint_sha256={name:sha(out/name) for name in selected['paths']},frozen_at=time.time(),
        test_used=False,budget=3,feature_schema=SCHEMA,training_report_sha256=sha(out/'training_report.json'))
    write(out/'selection_frozen.json',frozen);return frozen


def load_ranker(folder):
    import torch
    frozen=read(folder/'selection_frozen.json');models=[]
    for name,expected in frozen['checkpoint_sha256'].items():
        if sha(folder/name)!=expected:raise ValueError('checkpoint changed after freeze')
        saved=torch.load(folder/name,map_location='cpu',weights_only=False);model=build(saved['model_config'])
        model.load_state_dict(saved['state_dict']);model.eval();models.append((model,saved))
    return frozen,models


def score_candidates(encoded,frozen,models):
    rows=[dict(encoded=e) for e in encoded]
    pred=np.asarray([temperature_scale(predict(model,rows,'cpu',saved['cost_mean'],saved['cost_scale']),saved['temperatures']) for model,saved in models])
    return ranking(pred,frozen['alpha'],frozen['beta']),pred.mean(0),pred[:,:,1].std(0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--repeats',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();train(a.source,a.repeats,a.out)
