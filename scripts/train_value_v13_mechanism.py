"""Fit a scene-conditioned ranker on audited physical mechanism prefixes.

This is deliberately separate from the frozen V12 full-task checkpoint.  It
uses only archived pre-execution graphs as input and binary prefix outcomes as
labels.  Layout IDs and candidate names are audit metadata, never features.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from scripts.run_v13_mechanism_matrix import SCHEMA as MATRIX_SCHEMA
from scripts.train_value_v12 import dump, evaluate, predict
from simbench.value.graph_value_v12 import SCHEMA, ValueRankerV12, encode_graph
from simbench.value.plan import digest
from simbench.value.value_v12 import MODEL_KINDS, StageValueNet, collate, supervised_loss

TRAINING_SCHEMA="twingraph.mechanism_value_training.v13.v1"
SELECTION=("maximum validation Hit@K; minimum normalized first-success calls; maximum mean within-layout "
           "positive-negative pair accuracy; minimum Brier")


def mechanism_metrics(rows, predictions, k):
    report=evaluate(rows,predictions,k);scores=np.asarray(predictions,float);layout_scores=[]
    for seed in sorted({r["seed"] for r in rows}):
        indexes=[i for i,r in enumerate(rows) if r["seed"]==seed]
        comparisons=[scores[p]>scores[n] for p in indexes for n in indexes
                     if rows[p]["y"]>rows[n]["y"]]
        if comparisons: layout_scores.append(float(np.mean(comparisons)))
    report["mean_within_layout_pair_accuracy"]=float(np.mean(layout_scores))
    report["pair_accuracy_by_layout"]=dict(zip(map(str,sorted({r["seed"] for r in rows})),layout_scores))
    return report


def mechanism_selection_key(report):
    return (-float(report["success"]),float(report["normalized_first_success_calls"]),
            -float(report["mean_within_layout_pair_accuracy"]),float(report["brier"]))


def load_rows(roots, allowed_seeds, stop_after="end_stop"):
    allowed=set(allowed_seeds); rows=[]; manifest=[]; runtimes=set(); seen={}
    for root in map(Path,roots):
        for request_path in sorted(root.glob(f"seed_*/{stop_after}/request.json")):
            request=json.loads(request_path.read_text())
            seed=int(request["seed"])
            if seed not in allowed: continue
            if (request.get("schema")!=MATRIX_SCHEMA or request.get("stop_after")!=stop_after
                    or request.get("domain")!="development"):
                raise ValueError("incompatible mechanism matrix")
            runtime=request.get("runtime_sha256");runtimes.add(runtime)
            result_paths={p.parent.name:p for p in request_path.parent.glob("candidates/*/result.json")}
            names=[p["name"] for p in request["pool"]]
            if set(names)!=set(result_paths) or len(names)!=len(set(names)):
                raise ValueError(f"incomplete mechanism pool for layout {seed}")
            for proposal in request["pool"]:
                path=result_paths[proposal["name"]]; graph_path=path.with_name("input_graph.json")
                result=json.loads(path.read_text());graph=json.loads(graph_path.read_text())
                if (result.get("valid") is not True or not isinstance(result.get("success"),bool)
                        or result.get("evaluation_scope")!=f"assembly_prefix_through_{stop_after}"
                        or result.get("full_task_label") is not False
                        or result.get("runtime_sha256")!=runtime or result.get("seed")!=seed
                        or result.get("proposal")!=proposal or graph.get("proposal")!=proposal
                        or result.get("input_graph_sha256")!=digest(graph)):
                    raise ValueError(f"unbound or invalid mechanism label: {path}")
                graph_sha=digest(graph); key=(seed,graph_sha); label=float(result["success"])
                if key in seen:
                    if seen[key]!=label: raise ValueError("duplicate graph has conflicting physical labels")
                    continue
                seen[key]=label; encoded=encode_graph(graph)
                duration=float(result.get("sim_seconds",result.get("physics_steps",0)))
                if not np.isfinite(duration) or duration<0: raise ValueError("invalid deterministic rollout cost")
                local=np.zeros(8,np.float32)
                rows.append(dict(seed=seed,source=f"mechanism_{stop_after}",name=proposal["name"],graph=graph,
                    encoded=encoded,y=label,local_y=local,local_mask=local,
                    outcomes=[dict(condition="physical_prefix",success=result["success"],seconds=duration)]))
                manifest.append(dict(layout=seed,candidate=proposal["name"],graph_sha256=graph_sha,
                    result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    request_sha256=hashlib.sha256(request_path.read_bytes()).hexdigest(),
                    runtime_sha256=runtime,result=str(path),graph=str(graph_path)))
    if len(runtimes)!=1 or None in runtimes: raise ValueError("one hash-bound physical runtime is required")
    found={r["seed"] for r in rows}
    if found!=allowed: raise ValueError(f"missing layouts: {sorted(allowed-found)}")
    return rows,manifest,next(iter(runtimes))


def fit(rows, manifest, runtime, splits, out, epochs=120, device="cpu"):
    train=[r for r in rows if r["seed"] in splits["train"]]
    validation=[r for r in rows if r["seed"] in splits["validation"]]
    if set(splits["train"])&set(splits["validation"]): raise ValueError("layout leakage")
    if not train or not validation: raise ValueError("nonempty train and validation layouts required")
    for group,name in ((train,"train"),(validation,"validation")):
        by_seed={s:{r["y"] for r in group if r["seed"]==s} for s in splits[name]}
        if any(values!={0.,1.} for values in by_seed.values()):
            raise ValueError(f"every {name} layout must be a mixed candidate pool")
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    declaration=dict(schema=TRAINING_SCHEMA,input_schema=SCHEMA,label_scope="physical_prefix_through_end_stop",
        full_task_claim=False,train_layouts=splits["train"],validation_layouts=splits["validation"],
        k=int(splits.get("k",2)),epochs=int(epochs),training_seed=1313,model_kinds=list(MODEL_KINDS),
        selection_criterion=SELECTION,physical_runtime_sha256=runtime,
        prohibited_features=["layout_seed","candidate_name","rollout_result","simulator_truth"])
    dump(out/"data_manifest.json",dict(**declaration,records=manifest))
    np.savez_compressed(out/"encoded_dataset.npz",x=np.stack([r["encoded"]["x"] for r in rows]),
        relations=np.stack([r["encoded"]["relations"] for r in rows]),
        active=np.stack([r["encoded"]["active"] for r in rows]),y=np.asarray([r["y"] for r in rows]),
        layout=np.asarray([r["seed"] for r in rows]))
    reports={};k=declaration["k"]
    for kind in MODEL_KINDS:
        torch.manual_seed(1313);rng=np.random.default_rng(1313)
        model=StageValueNet(train[0]["encoded"]["x"].shape[-1],kind=kind).to(device)
        x=np.concatenate([r["encoded"]["x"] for r in train])
        model.mean.copy_(torch.as_tensor(x.mean(0),device=device));model.scale.copy_(torch.as_tensor(np.maximum(x.std(0),.1),device=device))
        opt=torch.optim.AdamW(model.parameters(),lr=.003 if kind=="linear" else .001,weight_decay=.03)
        best=None;state=None;chosen=None;history=[];started=time.perf_counter()
        for epoch in range(epochs):
            losses=[]
            for seed in rng.permutation(splits["train"]):
                group=[r for r in train if r["seed"]==seed]
                model.train();opt.zero_grad();pred=model(collate([r["encoded"] for r in group],device))
                labels=torch.as_tensor([r["y"] for r in group],dtype=torch.float32,device=device)
                zeros=torch.zeros((len(group),8),device=device)
                loss=supervised_loss(pred,labels,zeros,zeros,local_weight=0.)
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step();losses.append(float(loss))
            metric=mechanism_metrics(validation,predict(model,validation,device),k);key=mechanism_selection_key(metric)
            history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),validation_hit_at_k=metric["success"],
                validation_normalized_first_success_calls=metric["normalized_first_success_calls"],
                validation_pair_accuracy=metric["mean_within_layout_pair_accuracy"],validation_brier=metric["brier"]))
            if best is None or key<best:
                best=key;chosen=epoch+1;state={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
        model.load_state_dict(state);checkpoint=out/f"{kind}.pt"
        torch.save(dict(schema=SCHEMA,model_config=model.config,state_dict=state,kind=kind,
            label_domain="mechanism_prefix_end_stop_v13",selected_epoch=chosen,predeclared_k=k,
            train_layouts=splits["train"],validation_layouts=splits["validation"],
            physical_runtime_sha256=runtime,selection_criterion=SELECTION,full_task_claim=False),checkpoint)
        reports[kind]=dict(selected_epoch=chosen,selected_validation_key=list(best),training_seconds=time.perf_counter()-started,
            train=mechanism_metrics(train,predict(model,train,device),k),
            validation=mechanism_metrics(validation,predict(model,validation,device),k))
        dump(out/f"{kind}_history.json",history);dump(out/"metrics.json",reports)
    selected=min(reports,key=lambda kind:mechanism_selection_key(reports[kind]["validation"]))
    dump(out/"selection.json",dict(selected=selected,criterion=SELECTION,k=k,
        checkpoint_sha256=hashlib.sha256((out/f"{selected}.pt").read_bytes()).hexdigest(),
        validation=reports[selected]["validation"],full_task_claim=False))
    return reports,selected


def main():
    p=argparse.ArgumentParser();p.add_argument("--roots",nargs="+",required=True)
    p.add_argument("--splits",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=120);p.add_argument("--device",default="cpu");a=p.parse_args()
    torch.set_num_threads(1);splits=json.loads(a.splits.read_text())
    if set(splits)!={"train","validation","k"}: raise ValueError("split file must declare train, validation and k only")
    rows,manifest,runtime=load_rows(a.roots,splits["train"]+splits["validation"])
    reports,selected=fit(rows,manifest,runtime,splits,a.out,a.epochs,a.device)
    print(json.dumps(dict(selected=selected,metrics=reports[selected]),ensure_ascii=False,indent=2))


if __name__=="__main__":main()
