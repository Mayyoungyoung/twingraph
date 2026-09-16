"""Freeze before opening test; report module quality, real scoring cost and deployment."""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from simbench.value.collect import dump
from simbench.value.plan import digest
from simbench.value.skill_graph import compile_graph,validate_graph
from simbench.value.graph_encode import encode_graphs
from simbench.value.graph_learning import load_groups,load_model,predict,metrics
from simbench.value.port_pool import vectorize
from simbench.value.encode import encode_plan
from simbench.value.research_learning import numeric_features
from simbench.value.research_scenarios import make_family,build_pool,observed,geometric_features
from simbench.value.physical import PhysicalRunner,perturbation
from simbench.value.validation import select_and_validate


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def freeze(models,out):
    rows=[]
    for p in sorted(Path(models).glob("*/best.pt")):
        saved=torch.load(p,map_location="cpu",weights_only=False)
        rows.append(dict(name=p.parent.name,path=str(p.resolve()),sha256=sha(p),kind=saved["kind"],
                         seed=saved["training"]["seed"],epoch=saved["epoch"],validation=saved["validation"],
                         source_sha256=saved["source_sha256"],split_sha256=saved["split_sha256"]))
    if not rows:raise ValueError("no checkpoints to freeze")
    candidates=[r for r in rows if r["kind"] in {"graph","sequence","port_mlp"}]
    best=min(candidates,key=lambda r:r["validation"]["brier"])
    result=dict(schema="twingraph.graph_experiment.freeze.v1",selection="minimum validation Brier among graph-derived models",
                selected=best["name"],models=rows,created_unix=time.time())
    if Path(out).exists():raise ValueError("refuse to overwrite an existing test freeze")
    dump(out,result);print(json.dumps(dict(selected=best["name"],models=len(rows),validation_brier=best["validation"]["brier"])))


def prepare(g,kind,saved):
    if kind=="port_mlp":g["port_x"]=np.stack([vectorize(e,saved["vocabulary"]) for e in g["encoded"]])


def intervals(result):
    rng=np.random.default_rng(17);rows=result["rows"]
    result["intervals"]={}
    for k in ("hit1","hit2","hit4","quality4","regret4"):
        values=np.asarray([r[k] for r in rows if r[k] is not None],float)
        if len(values):
            boot=values[rng.integers(len(values),size=(5000,len(values)))].mean(1)
            result["intervals"][k]=np.quantile(boot,[.025,.975]).tolist()
    result["interval_note"]="empirical problem-group bootstrap; a degenerate 100% interval is not a reliability guarantee"


def evaluate(frozen,data,out,device):
    groups=[g for g in load_groups([data],include_test=True) if g["split"]=="test"]
    if sorted(g["seed"] for g in groups)!=list(range(41200,41212)):raise ValueError("locked test configurations incomplete/changed")
    results={}
    for row in frozen["models"]:
        model,kind,saved=load_model(row["path"],device)
        for g in groups:prepare(g,kind,saved)
        scores=[predict(model,kind,g,device) for g in groups]
        result=metrics(groups,scores);intervals(result);results[row["name"]]=result
        if kind=="graph" and row["seed"]==17:
            for mode in ("none","shuffle"):
                results[row["name"]+"_edges_"+mode]=metrics(groups,[predict(model,kind,g,device,mode) for g in groups])
    for name,scores in (
        ("geometry",[-g["geometry"][:,0]-.001*g["geometry"][:,4] for g in groups]),
        ("random",[np.random.default_rng(np.random.SeedSequence([g["seed"],51])).random(len(g["plans"])) for g in groups])):
        result=metrics(groups,scores)
        if name=="geometry":
            result["brier"]=None
            for r in result["rows"]:r["brier"]=None
        intervals(result);results[name]=result
    # Paired problem-level comparison uses exactly the same independent groups.
    selected=results[frozen["selected"]];geo=results["geometry"]
    paired={};rng=np.random.default_rng(39)
    for key in ("hit1","hit2","hit4","quality4"):
        delta=np.asarray([a[key]-b[key] for a,b in zip(selected["rows"],geo["rows"]) if a[key] is not None],float)
        paired[key]=dict(mean=float(delta.mean()),bootstrap95=np.quantile(delta[rng.integers(len(delta),size=(5000,len(delta)))].mean(1),[.025,.975]).tolist())
    dump(out/"test_metrics.json",dict(freeze_sha256=digest(frozen),results=results,selected_minus_geometry=paired))
    print(json.dumps({name:{k:r[k] for k in ("hit1","hit2","hit4","quality4","brier")} for name,r in results.items()}))


def sync(device):
    if device.startswith("cuda"):torch.cuda.synchronize(device)


def measured_score(s,targets,plans,obs,method,loaded,device):
    times={};started=time.perf_counter()
    t=time.perf_counter();graphs=[compile_graph(obs,p) for p in plans];times["graph_construction"]=time.perf_counter()-t
    t=time.perf_counter()
    for g in graphs:validate_graph(g)
    times["graph_integrity"]=time.perf_counter()-t
    times.update(optional_geometry=0.,typed_encoding=0.,network=0.,sort_export=0.)
    if method=="geometry" or loaded[method][1]=="mlp":
        t=time.perf_counter();geometry=np.asarray([geometric_features(s,p,targets) for p in plans]);times["optional_geometry"]=time.perf_counter()-t
    if method=="geometry":scores=-geometry[:,0]-.001*geometry[:,4]
    else:
        model,kind,saved=loaded[method];g=dict(plans=plans)
        t=time.perf_counter()
        if kind=="mlp":g["x"]=np.stack([numeric_features(obs,p,x) for p,x in zip(plans,geometry)])
        elif kind=="field":g["field"]=[encode_plan(obs,p) for p in plans]
        else:
            g["encoded"]=encode_graphs(graphs,check=False);prepare(g,kind,saved)
        times["typed_encoding"]=time.perf_counter()-t
        sync(device);t=time.perf_counter();scores=predict(model,kind,g,device);sync(device);times["network"]=time.perf_counter()-t
    t=time.perf_counter();order=np.argsort(-np.asarray(scores),kind="stable")
    ranked=[dict(candidate_id=plans[i].id,score=float(scores[i]),plan=plans[i].to_dict(),input_graph_sha256=digest(graphs[i])) for i in order]
    ranking=dict(ranked=ranked,top_k=ranked[:4]);times["sort_export"]=time.perf_counter()-t
    times["total"]=time.perf_counter()-started
    times["scoring_after_shared_graph"]=times["total"]-times["graph_construction"]-times["graph_integrity"]
    return ranking,graphs,times


def benchmark(frozen,out,device,repeats=5):
    names=list(dict.fromkeys(["geometry", "mlp_17", "field_17", "port_mlp_17", "sequence_17", "graph_17",frozen["selected"]]))
    loaded={};cold={}
    for row in frozen["models"]:
        if row["name"] in names:
            sync(device);t=time.perf_counter();loaded[row["name"]]=load_model(row["path"],device);sync(device)
            cold[row["name"]]=time.perf_counter()-t
    rows=[]
    for seed in range(41200,41212):
        d=out/f"timing_{seed}";d.mkdir(parents=True,exist_ok=True)
        with (d/"steps.log").open("w") as log,contextlib.redirect_stdout(log):
            _,s,_,targets=make_family("sliding_stage_pin",seed,d)
            plans,counts=build_pool(s,targets,seed,16);obs=observed(s,targets)
            for method in names:measured_score(s,targets,plans,obs,method,loaded,device)
            for repeat in range(repeats):
                for method in np.random.default_rng(seed+repeat).permutation(names):
                    _,_,times=measured_score(s,targets,plans,obs,method,loaded,device)
                    rows.append(dict(seed=seed,repeat=repeat,method=method,**times))
        dump(d/"candidate_cost.json",counts)
    summary={}
    for method in names:
        selected=[r for r in rows if r["method"]==method];summary[method]={}
        for key in ("graph_construction","graph_integrity","optional_geometry","typed_encoding","network","sort_export","total","scoring_after_shared_graph"):
            # Aggregate repeated measurements within each problem before quantiles.
            v=[np.median([r[key] for r in selected if r["seed"]==seed]) for seed in range(41200,41212)]
            summary[method][key]=dict(median_seconds=float(np.median(v)),p95_seconds=float(np.quantile(v,.95)))
    dump(out/"module_timing.json",dict(summary=summary,rows=rows,load_weights_seconds=cold,
         note="Resident synchronized GPU, one process, 2 Torch CPU threads; fresh encoding/proxy on every call. Graph construction/integrity is common to every method. Scene setup, necessary candidate geometry, physics and rendering excluded; no model here uses images. Weight load is not full process cold-start."))
    print(json.dumps({m:r["total"] for m,r in summary.items()}))


def deployment(frozen,out,device):
    methods=list(dict.fromkeys(["geometry","mlp_17",frozen["selected"]]));loaded={}
    for row in frozen["models"]:
        if row["name"] in methods:loaded[row["name"]]=load_model(row["path"],device)
    decisions=[]
    for seed in range(41200,41204):
        for method in methods:
            d=out/f"deployment_{seed}_{method}";d.mkdir(parents=True,exist_ok=True)
            with (d/"steps.log").open("w") as log,contextlib.redirect_stdout(log):
                _,s,_,targets=make_family("sliding_stage_pin",seed,d)
                plans,counts=build_pool(s,targets,seed,16);obs=observed(s,targets);runner=PhysicalRunner(s)
                ranking,graphs,times=measured_score(s,targets,plans,obs,method,loaded,device)
                lookup={p.id:g for p,g in zip(plans,graphs)};occurrences={}
                def validate(plan):
                    r=occurrences.get(plan.id,0);occurrences[plan.id]=r+1
                    return runner.run(lookup[plan.id],perturbation(seed,r,"online"),keep_trace=True)
                selected=select_and_validate(ranking,validate,8,repeats=2,mode="best_within_budget",allow_expand=False,accept_rate=.5)
                execution=[]
                if selected["chosen"]:
                    graph=lookup[selected["chosen"]["id"]]
                    execution=[runner.run(graph,perturbation(seed,r,"deployment"),keep_trace=True) for r in range(2)]
                top=dict(**ranking,top_k_graphs=[lookup[r["candidate_id"]] for r in ranking["top_k"]])
                dump(d/"top_k.json",top)
                result=dict(seed=seed,method=method,n=16,k=4,budget=8,repeats=2,
                     input_sha256=digest(dict(observation=obs,plans=[p.to_dict() for p in plans])),
                     scoring_times=times,candidate_cost=counts,selection=selected,deployment=execution)
                dump(d/"decision.json",result);decisions.append(result)
    dump(out/"deployment_summary.json",dict(decisions=[dict(seed=r["seed"],method=r["method"],
         chosen=bool(r["selection"]["chosen"]),rollouts=r["selection"]["validation_calls"],
         deployment_successes=sum(x["success"] for x in r["deployment"]),deployment_attempts=2,
         budget_status=r["selection"]["status"]) for r in decisions],
         note="Simulator deployment disturbances independent of online validation and offline reference; no real-robot claim."))


def main():
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["freeze","evaluate","benchmark","deployment"])
    p.add_argument("--models");p.add_argument("--freeze",required=True);p.add_argument("--data")
    p.add_argument("--out",default="runs/v3/evidence");p.add_argument("--device",default="cuda")
    a=p.parse_args();torch.set_num_threads(2)
    if a.mode=="freeze":freeze(a.models,a.freeze);return
    frozen=json.loads(Path(a.freeze).read_text())
    for row in frozen["models"]:
        if sha(row["path"])!=row["sha256"]:raise ValueError("frozen checkpoint changed")
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if a.mode=="evaluate":evaluate(frozen,a.data,out,a.device)
    elif a.mode=="benchmark":benchmark(frozen,out,a.device)
    else:deployment(frozen,out,a.device)


if __name__=="__main__":main()
