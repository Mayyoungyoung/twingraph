"""Small numeric models and single-head Transformer with locked group splits."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from .plan import PlanIR,digest
from .encode import encode_plan,collate
from .network import ModelConfig,PlanValueNet
from .evaluate import subset_metrics
from .collect import dump
from .program_audit import audit_program


def numeric_features(obs,plan,geometry):
    """Identity-free, task-relative numeric features; same features for MLP/residual."""
    stages=[]; current=None
    for call in plan.calls:
        a={k:v.value for k,v in call.arguments.items()}
        if call.skill=="estimate_grasp":
            current=dict(part=call.roles["manipulated"],yaw=a["yaws"][0],height=a["height_offset"])
            stages.append(current)
        elif current is not None:
            if call.skill=="grasp":current["force"]=a["force"]
            if call.skill=="plan_path" and "clearance" in a:current["clearance"]=a["clearance"]
            if call.skill=="move" and a.get("mode")=="guarded":current["speed"]=a["speed"]
    goals={g["manipulated"]:np.array(g["position"]) for g in obs["goals"]}
    rows=[];assembled=[]
    for stage in stages:
        part=stage["part"];obj=obs["objects"][part];target=goals[part]
        source=np.asarray(obj["position"]);others=[v for k,v in goals.items() if k!=part]
        previous=[goals[k] for k in assembled]
        nearest=min((float(np.linalg.norm(x[:2]-target[:2])) for x in others),default=.3)
        prev=min((float(np.linalg.norm(x[:2]-target[:2])) for x in previous),default=.3)
        size=np.max(np.asarray([g["size"] for g in obj["geoms"]]),axis=0)
        yaw=stage["yaw"]
        rows.append([np.sin(2*yaw),np.cos(2*yaw),stage["height"]*100,stage["force"],
                     stage["clearance"],stage["speed"]*100,*((target-source)*10),* (size*100),
                     nearest*10,prev*10,float(len(assembled)),target[2]*10])
        assembled.append(part)
    rows=np.asarray(rows,np.float32)
    if not len(rows):raise ValueError("numeric v2 model requires parameterized grasp stages")
    # Set summaries plus first/last stages retain program boundary/ordering.
    return np.r_[np.asarray(geometry,np.float32),rows.mean(0),rows.min(0),rows.max(0),rows[0],rows[-1],len(rows)].astype(np.float32)


def load_data(root,vision=False,include_test=False):
    groups=[];split_by_config={};ids=set();sources=set()
    for f in sorted(Path(root).glob("group_*/complete.json")):
        d=f.parent;inputs=json.loads((d/"inputs.json").read_text())
        split=inputs["declared_split"]
        gid=inputs["split_group"]
        if gid in split_by_config and split_by_config[gid]!=split:raise ValueError("trajectory siblings cross split")
        split_by_config[gid]=split
        if split=="test" and not include_test:continue
        if inputs["group_id"] in ids:raise ValueError("duplicate decision group")
        ids.add(inputs["group_id"]);sources.add(inputs["source_sha256"])
        ih=digest(inputs);out=json.loads((d/"outcomes.json").read_text());geo=json.loads((d/"geometry.json").read_text())
        if out["input_sha256"]!=ih or geo["input_sha256"]!=ih:raise ValueError("stale labels/geometry")
        plans=[PlanIR.from_dict(p) for p in inputs["candidates"]]
        for p in plans:audit_program(p,inputs["observation"]["objects"])
        records={p.id:[] for p in plans};paired={}
        for r in out["trials"]:
            if r["valid"] is not True:raise ValueError("invalid trial")
            a,b=r["prefix_success"],r["suffix_success"]
            if not isinstance(a,bool) or (not a and b is not None) or (a and not isinstance(b,bool)) or bool(a and b)!=r["success"]:
                raise ValueError("invalid conditional labels")
            if r["candidate_id"] not in records:raise ValueError("unknown candidate label")
            key=r["trial"]["repeat"];trial_hash=digest(r["trial"])
            if key in paired and paired[key]!=trial_hash:raise ValueError("unpaired perturbations")
            paired[key]=trial_hash;records[r["candidate_id"]].append(r)
        for rs in records.values():
            if sorted(r["trial"]["repeat"] for r in rs)!=sorted(paired):raise ValueError("missing or duplicate repetitions")
        visual=None
        if vision:
            cache=np.load(d/"visual.npz",allow_pickle=False)
            if str(cache["input_sha256"])!=ih:raise ValueError("stale visual cache")
            visual=cache["features"]
        groups.append(dict(id=inputs["group_id"],split_group=gid,split=split,path=str(d),inputs=inputs,plans=plans,
                    x=np.stack([numeric_features(inputs["observation"],p,g) for p,g in zip(plans,geo["features"])]),
                    geometry=np.asarray(geo["features"],np.float32),
                    y=np.array([np.mean([r["success"] for r in records[p.id]]) for p in plans],np.float32),
                    encoded=[encode_plan(inputs["observation"],p) for p in plans],visual=visual,
                    repeats=len(paired),input_sha256=ih))
    if not groups:raise ValueError("empty dataset")
    if len(sources)>1:
        review=Path(root)/"source_compatibility.json"
        approved=set(json.loads(review.read_text())["approved_sources"]) if review.exists() else set()
        if not sources.issubset(approved):raise ValueError("unreviewed mixed collection code versions")
    return groups


class NumericNet(nn.Module):
    def __init__(self,dim,kind="mlp"):
        super().__init__();self.kind=kind
        self.register_buffer("mean",torch.zeros(dim));self.register_buffer("std",torch.ones(dim))
        self.prior=nn.Linear(6,1)
        self.delta=nn.Sequential(nn.Linear(dim,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1))

    def forward(self,x):
        x=(x-self.mean)/self.std
        if self.kind=="prior":return self.prior(x[:,:6]).squeeze(-1)
        residual=self.delta(x).squeeze(-1)
        return residual+self.prior(x[:,:6]).squeeze(-1) if self.kind=="residual" else residual


@torch.inference_mode()
def predict(model,group,device,kind,chunk=2):
    model.eval()
    if kind in {"mlp","residual","prior"}:
        return model(torch.as_tensor(group["x"],device=device)).sigmoid().cpu().numpy()
    scores=[]
    for i in range(0,len(group["plans"]),chunk):
        enc=group["encoded"][i:i+chunk]
        vis=[group["visual"]]*len(enc) if model.config.vision else None
        scores.extend(model(collate(enc,vis,device),direct_only=True)["direct_logit"].sigmoid().cpu().tolist())
    return np.asarray(scores)


def metrics(groups,scores,k=4):
    rows=[]
    for g,s in zip(groups,scores):
        selected=np.argsort(-np.asarray(s),kind="stable")[:k]
        rows.append(dict(group_id=g["id"],split_group=g["split_group"],family=g["inputs"]["task"]["family"],
                         checkpoint=g["inputs"]["checkpoint"],length=len(g["plans"][0].calls),
                         pool_type="all_failure" if g["y"].max()==0 else "all_success" if g["y"].min()==1 else "mixed",
                         **subset_metrics(g["y"],selected,.1),brier=float(np.mean((s-g["y"])**2)),
                         scores=np.asarray(s).tolist(),reference=g["y"].tolist()))
    return dict(groups=len(rows),hit=float(np.mean([r["hit"] for r in rows if r["hit"] is not None])) if any(r["hit"] is not None for r in rows) else None,
                feasible=float(np.mean([r["feasible"] for r in rows])),regret=float(np.mean([r["regret"] for r in rows])),
                brier=float(np.mean([r["brier"] for r in rows])),rows=rows)


def load_model(path,device="cpu"):
    saved=torch.load(path,map_location=device,weights_only=False)
    if saved.get("schema")=="twingraph.value.v1":
        if saved.get("objective")!="direct":raise ValueError("fixed v1 transfer requires the trained direct head")
        model=PlanValueNet(ModelConfig(**saved["model_config"]));kind="transformer"
    else:
        kind=saved["kind"]
        model=NumericNet(saved["dim"],kind) if kind in {"mlp","residual","prior"} else PlanValueNet(ModelConfig(**saved["model_config"]))
    model.load_state_dict(saved["state_dict"]);model.to(device).eval()
    return model,kind,saved


def train(args):
    torch.set_num_threads(2);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    vision=args.kind=="vision";groups=load_data(args.data,vision=vision)
    if args.family:groups=[g for g in groups if g["inputs"]["task"]["family"]==args.family]
    if args.few_shot:
        # Fixed first configuration IDs, no outcome-based selection.
        connector=sorted({g["split_group"] for g in groups if g["split"]=="train" and g["inputs"]["task"]["family"]=="rigid_connector_module"})[:args.few_shot]
        groups=[g for g in groups if g["split"]!="train" or g["inputs"]["task"]["family"]!="rigid_connector_module" or g["split_group"] in connector]
    tr=[g for g in groups if g["split"]=="train"];va=[g for g in groups if g["split"]=="val"]
    if not tr or not va:raise ValueError("nonempty train/val required")
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    numeric=args.kind in {"mlp","residual","prior"}
    dim=tr[0]["x"].shape[1];cfg=ModelConfig(vision=vision)
    model=NumericNet(dim,args.kind) if numeric else PlanValueNet(cfg)
    model=model.to(args.device)
    if numeric:
        x=torch.as_tensor(np.concatenate([g["x"] for g in tr]),device=args.device)
        y=torch.as_tensor(np.concatenate([g["y"] for g in tr]),device=args.device)
        model.mean.copy_(x.mean(0));model.std.copy_(x.std(0).clamp_min(.01))
        if args.kind in {"residual","prior"}:
            opt=torch.optim.Adam(model.prior.parameters(),lr=.025)
            for _ in range(250):
                opt.zero_grad();z=model.prior(((x-model.mean)/model.std)[:,:6]).squeeze(-1)
                loss=nn.functional.binary_cross_entropy_with_logits(z,y)+.001*model.prior.weight.square().sum()
                loss.backward();opt.step()
            model.prior.requires_grad_(False)
            if args.kind=="residual":
                nn.init.zeros_(model.delta[-1].weight);nn.init.zeros_(model.delta[-1].bias)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.003 if numeric else .0002,weight_decay=.01)
    history=[];best=None;start=time.perf_counter()
    manifest=dict(train=[g["id"] for g in tr],val=[g["id"] for g in va],input_hashes={g["id"]:g["input_sha256"] for g in groups})
    dump(out/"splits.json",manifest)
    for epoch in range(args.epochs):
        model.train();losses=[]
        if numeric:
            optimizer.zero_grad();loss=nn.functional.binary_cross_entropy_with_logits(model(x),y)
            if args.kind!="prior":loss.backward();optimizer.step()
            losses=[float(loss.detach())]
        else:
            for gi in np.random.permutation(len(tr)):
                g=tr[gi]
                order=np.random.permutation(len(g["plans"]))
                for starti in range(0,len(order),args.batch_size):
                    ix=order[starti:starti+args.batch_size];enc=[g["encoded"][i] for i in ix]
                    vis=[g["visual"]]*len(ix) if vision else None
                    optimizer.zero_grad();output=model(collate(enc,vis,args.device),direct_only=True)
                    target=torch.as_tensor(g["y"][ix],device=args.device)
                    loss=nn.functional.binary_cross_entropy_with_logits(output["direct_logit"],target)
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
        scores=[predict(model,g,args.device,args.kind) for g in va]
        val=metrics(va,scores,args.k);selection=(val["hit"] or 0.,-val["regret"],-val["brier"])
        history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),val={k:v for k,v in val.items() if k!="rows"},seconds=time.perf_counter()-start))
        if best is None or selection>best:
            best=selection
            torch.save(dict(schema="twingraph.value.v2",kind=args.kind,dim=dim,model_config=asdict(cfg),
                            state_dict=model.state_dict(),epoch=epoch+1,training=vars(args),split_sha256=digest(manifest),
                            collection_sources=sorted({g['inputs']['source_sha256'] for g in groups}),
                            protocol="assembly.program.feedback.v2",validation=val),out/"best.pt")
        dump(out/"history.json",history)
        if epoch%5==0:print(json.dumps(history[-1]),flush=True)
    dump(out/"summary.json",dict(wall_seconds=time.perf_counter()-start,seed=args.seed,kind=args.kind,
                                parameters=sum(p.numel() for p in model.parameters()),training_groups=len(tr),validation_groups=len(va)))


def main():
    p=argparse.ArgumentParser();p.add_argument("--data",required=True);p.add_argument("--out",required=True)
    p.add_argument("--kind",choices=["mlp","prior","residual","no_vision","vision"],default="mlp")
    p.add_argument("--device",default="cuda");p.add_argument("--seed",type=int,default=17)
    p.add_argument("--epochs",type=int,default=60);p.add_argument("--batch-size",type=int,default=2)
    p.add_argument("--k",type=int,default=4);p.add_argument("--family");p.add_argument("--few-shot",type=int,default=0)
    train(p.parse_args())


if __name__=="__main__":main()
