"""Reproducible value-module comparisons using graph-derived execution inputs."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from .plan import PlanIR,digest
from .skill_graph import compile_graph,validate_graph,interface_hash
from .graph_encode import encode_graph,collate_graph
from .graph_network import GraphConfig,GraphValueNet
from .research_learning import NumericNet,numeric_features
from .encode import encode_plan,collate
from .network import ModelConfig,PlanValueNet
from .collect import dump


def load_groups(roots, include_test=False):
    groups=[];seen=set();splits={}
    for root in roots:
        for complete in sorted(Path(root).glob("group_*/complete.json")):
            d=complete.parent;inp=json.loads((d/"inputs.json").read_text())
            if inp["task"]["family"]!="sliding_stage_pin":continue
            split=inp["declared_split"]
            if split=="test" and not include_test:continue
            key=(inp["task"]["family"],inp["task"]["seed"])
            if key in splits and splits[key]!=split:raise ValueError("configuration crosses splits")
            splits[key]=split;gid=inp["group_id"]
            if gid in seen:raise ValueError("duplicate group")
            seen.add(gid);ih=digest(inp)
            out=json.loads((d/"outcomes.json").read_text());geo=json.loads((d/"geometry.json").read_text())
            if out["input_sha256"]!=ih or geo["input_sha256"]!=ih:raise ValueError("stale label binding")
            plans=[PlanIR.from_dict(p) for p in inp["candidates"]]
            records={p.id:[] for p in plans};trials={}
            for row in out["trials"]:
                if row["valid"] is not True or not isinstance(row["success"],bool):raise ValueError("invalid physical label")
                if bool(row["prefix_success"] and row["suffix_success"])!=row["success"]:raise ValueError("invalid conditional outcome")
                r=row["trial"]["repeat"];h=digest(row["trial"])
                if r in trials and trials[r]!=h:raise ValueError("unpaired disturbances")
                trials[r]=h;records[row["candidate_id"]].append(row)
            if not trials:raise ValueError("no outcomes")
            for rs in records.values():
                if sorted(r["trial"]["repeat"] for r in rs)!=sorted(trials):raise ValueError("incomplete/duplicate labels")
            graphs=[compile_graph(inp["observation"],p) for p in plans]
            if (d/"skill_graphs.json").exists():
                saved=json.loads((d/"skill_graphs.json").read_text())
                if saved["input_sha256"]!=ih or digest(saved["graphs"])!=digest(graphs):raise ValueError("pre-execution graph drift")
                for p,g in zip(plans,graphs):
                    if any(r.get("input_graph_sha256")!=digest(g) for r in records[p.id]):raise ValueError("executed graph mismatch")
            groups.append(dict(id=gid,seed=inp["task"]["seed"],split=split,inputs=inp,plans=plans,graphs=graphs,path=str(d),
                encoded=[encode_graph(g,check=False) for g in graphs],
                field=[encode_plan(inp["observation"],p) for p in plans],
                x=np.stack([numeric_features(inp["observation"],p,g) for p,g in zip(plans,geo["features"])]),
                geometry=np.asarray(geo["features"]),
                y=np.asarray([np.mean([r["success"] for r in records[p.id]]) for p in plans],np.float32),
                repeats=len(trials),input_sha256=ih))
    if not groups:raise ValueError("empty dataset")
    return groups


def metrics(groups,scores):
    rows=[]
    for g,s in zip(groups,scores):
        y=g["y"];order=np.argsort(-np.asarray(s),kind="stable");best=float(y.max())
        row=dict(group_id=g["id"],seed=g["seed"],n=len(y),
                 pool_type="all_failure" if best==0 else "all_success" if y.min()==1 else "mixed",
                 scores=np.asarray(s).tolist(),reference=y.tolist(),brier=float(np.mean((s-y)**2)))
        for k in (1,2,4):
            chosen=y[order[:k]]
            row.update({f"hit{k}":bool(chosen.max()>=best-.1) if best>0 else None,
                        f"feasible{k}":bool(chosen.max()>0),f"regret{k}":float(best-chosen.max()),
                        f"quality{k}":float(chosen.mean())})
        rows.append(row)
    keys=["brier"]+[f"{m}{k}" for k in (1,2,4) for m in ("hit","feasible","regret","quality")]
    return dict(groups=len(rows),**{k:float(np.mean([r[k] for r in rows if r[k] is not None]))
                   if any(r[k] is not None for r in rows) else None for k in keys},rows=rows)


def forward(model,kind,group,indices,device,relation_mode="correct"):
    if kind=="mlp":return model(torch.as_tensor(group["x"][indices],device=device))
    if kind=="field":return model(collate([group["field"][i] for i in indices],device=device),direct_only=True)["direct_logit"]
    return model(collate_graph([group["encoded"][i] for i in indices],device,relation_mode))


@torch.inference_mode()
def predict(model,kind,group,device="cuda",relation_mode="correct"):
    model.eval();scores=[];chunk=4 if kind=="field" else 32
    for start in range(0,len(group["plans"]),chunk):
        ix=np.arange(start,min(start+chunk,len(group["plans"])))
        scores.extend(forward(model,kind,group,ix,device,relation_mode).sigmoid().cpu().tolist())
    return np.asarray(scores)


def load_model(path,device="cpu"):
    saved=torch.load(path,map_location=device,weights_only=False);kind=saved["kind"]
    if kind=="mlp":model=NumericNet(saved["dim"],"mlp")
    elif kind=="field":model=PlanValueNet(ModelConfig(**saved["config"]))
    else:model=GraphValueNet(GraphConfig(**saved["config"]))
    model.load_state_dict(saved["state_dict"]);model.to(device).eval()
    if saved["interface_sha256"]!=interface_hash():raise ValueError("checkpoint execution interface changed")
    return model,kind,saved


def train(args):
    started=time.perf_counter();torch.set_num_threads(2)
    torch.manual_seed(args.seed);random.seed(args.seed);np.random.seed(args.seed)
    groups=load_groups(args.data);tr=[g for g in groups if g["split"]=="train"];va=[g for g in groups if g["split"]=="val"]
    if not tr or not va:raise ValueError("nonempty train/val required")
    cfg=GraphConfig(relations=args.kind=="graph")
    if args.kind=="mlp":model=NumericNet(tr[0]["x"].shape[1],"mlp")
    elif args.kind=="field":
        cfg=ModelConfig(width=64,layers=2,heads=4,vision=False);model=PlanValueNet(cfg)
    else:model=GraphValueNet(cfg)
    model.to(args.device)
    if args.kind=="mlp":
        x=torch.as_tensor(np.concatenate([g["x"] for g in tr]),device=args.device)
        model.mean.copy_(x.mean(0));model.std.copy_(x.std(0).clamp_min(.01))
    opt=torch.optim.AdamW(model.parameters(),lr=.003 if args.kind=="mlp" else .0005,weight_decay=.01)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    manifest={split:[dict(id=g["id"],seed=g["seed"],input_sha256=g["input_sha256"]) for g in gs] for split,gs in (("train",tr),("val",va))}
    dump(out/"split.json",manifest)
    source=digest({str(p.relative_to(Path(__file__).parents[2])):hashlib.sha256(p.read_bytes()).hexdigest()
                   for d in (Path(__file__).parent,Path(__file__).parents[1]/"assembly") for p in sorted(d.glob("*.py"))})
    best=float("inf");history=[]
    for epoch in range(args.epochs):
        model.train();losses=[]
        for gi in np.random.permutation(len(tr)):
            g=tr[gi];order=np.random.permutation(len(g["plans"]));batch=4 if args.kind=="field" else 16
            for start in range(0,len(order),batch):
                ix=order[start:start+batch];opt.zero_grad()
                z=forward(model,args.kind,g,ix,args.device)
                loss=nn.functional.binary_cross_entropy_with_logits(z,torch.as_tensor(g["y"][ix],device=args.device))
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
        validation=metrics(va,[predict(model,args.kind,g,args.device) for g in va])
        summary=dict(epoch=epoch+1,loss=float(np.mean(losses)),validation={k:v for k,v in validation.items() if k!="rows"},seconds=time.perf_counter()-started)
        history.append(summary)
        if validation["brier"]<best:
            best=validation["brier"]
            torch.save(dict(schema="twingraph.value.graph.v1",kind=args.kind,config=asdict(cfg),dim=tr[0]["x"].shape[1],
                       state_dict=model.state_dict(),epoch=epoch+1,training=vars(args),validation=validation,
                       interface_sha256=interface_hash(),source_sha256=source,split_sha256=digest(manifest)),out/"best.pt")
        dump(out/"history.json",history)
        if epoch%5==0:print(json.dumps(summary),flush=True)
    dump(out/"summary.json",dict(kind=args.kind,seed=args.seed,wall_seconds=time.perf_counter()-started,
         train_groups=len(tr),val_groups=len(va),training_rollouts=sum(len(g["plans"])*g["repeats"] for g in tr),
         validation_rollouts=sum(len(g["plans"])*g["repeats"] for g in va),parameters=sum(p.numel() for p in model.parameters()),
         source_sha256=source,interface_sha256=interface_hash()))


def main():
    p=argparse.ArgumentParser();p.add_argument("--data",nargs="+",required=True);p.add_argument("--out",required=True)
    p.add_argument("--kind",choices=["graph","sequence","mlp","field"],default="graph")
    p.add_argument("--seed",type=int,default=17);p.add_argument("--epochs",type=int,default=60);p.add_argument("--device",default="cuda")
    train(p.parse_args())


if __name__=="__main__":main()
