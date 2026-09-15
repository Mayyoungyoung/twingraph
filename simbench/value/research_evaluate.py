"""Locked-test ranking plus actual online validation and independent deployment.

No cached outcome lookup is used by an online policy. Offline reference is
read only AFTER the selected PlanIR has been independently executed.
"""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from .research_scenarios import make_family,build_pool,observed,program,geometric_features,render_observation
from .research_learning import load_data,load_model,predict,metrics,numeric_features
from .physical import PhysicalRunner,perturbation
from .validation import select_and_validate
from .plan import PlanIR,execute_calls,digest
from .encode import encode_plan,collate,NUMERIC
from .collect import dump
from .program_audit import audit_program


def synchronize(device):
    if str(device).startswith("cuda"):torch.cuda.synchronize(device)


def ranking_rows(plans,scores,k):
    order=np.argsort(-np.asarray(scores),kind="stable")
    rows=[dict(candidate_id=plans[i].id,score=float(scores[i]),plan=plans[i].to_dict()) for i in order]
    return dict(ranked=rows,top_k=rows[:k],requested_k=k,returned_k=min(k,len(rows)))


class OnlineExperiment:
    def __init__(self,models,device="cuda"):
        self.device=device;self.models={};self.cold={};self.vision=None
        self.checkpoint_hashes={}
        for name,path in models.items():
            synchronize(device);t=time.perf_counter()
            self.models[name]=load_model(path,device)
            synchronize(device);self.cold[name]=time.perf_counter()-t
            self.checkpoint_hashes[name]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
            # Warm kernels on synthetic tensors, never on a test candidate.
            model,kind,_=self.models[name]
            t=time.perf_counter()
            with torch.inference_mode():
                if kind in {"mlp","residual","prior"}:
                    model(torch.zeros((1,len(model.mean)),device=device))
                else:
                    dummy={k:np.ones(32,dtype=np.int64) for k in ('keys','categories','statuses','stages','positions')}
                    dummy['numbers']=np.zeros((32,NUMERIC),np.float32)
                    visuals=[np.zeros((3,512),np.float32)] if model.config.vision else None
                    model(collate([dummy],visuals,device),direct_only=True)
            synchronize(device);self.cold[name]+=time.perf_counter()-t
        if any(getattr(m,"config",None) and m.config.vision for m,_,_ in self.models.values()):
            from .vision import FrozenVision
            t=time.perf_counter();self.vision=FrozenVision(device);synchronize(device)
            with torch.inference_mode():self.vision.model(torch.zeros((1,3,224,224),device=device))
            synchronize(device)
            self.cold["visual_encoder"]=time.perf_counter()-t

    def run(self,task,method,out,*,n=16,k=4,budget=8,repeats=2,mode="best_within_budget",
            allow_expand=False,deployment_repeats=2,domain_seed=0):
        directory=Path(out);directory.mkdir(parents=True,exist_ok=True)
        request=dict(task=task,method=method,n=n,k=k,budget=budget,repeats=repeats,mode=mode,
                     allow_expand=allow_expand,deployment_repeats=deployment_repeats,domain_seed=domain_seed)
        file=directory/"decision.json"
        if file.exists():
            old=json.loads(file.read_text())
            if old["request_sha256"]!=digest(request):raise ValueError("online resume mismatch")
            return old
        with (directory/"steps.log").open("w") as log,contextlib.redirect_stdout(log):
            t=time.perf_counter()
            spec,s,_,targets=make_family(task["family"],task["seed"],directory)
            setup=time.perf_counter()-t;completed=[];cp=task.get("checkpoint",0)
            if cp:
                order=list(s.parts);choices={p:dict(yaw=0.,height=.003,clearance=1.035,force=3.,speed=.008) for p in order}
                warm=program(s,targets,order,choices);s.active_candidate_id=warm.id
                execute_calls(s,warm,warm.calls[:18*cp]);completed=order[:cp]
            start=time.perf_counter();plans,counts=build_pool(s,targets,task["seed"],n,completed)
            obs=observed(s,targets);runner=PhysicalRunner(s)
            contracts=[audit_program(p,obs["objects"]) for p in plans]
            times=dict(scene_setup_seconds=setup,candidate_generation_seconds=counts["candidate_generation_seconds"],
                       necessary_geometry_seconds=counts["necessary_geometry_seconds"],optional_geometry_seconds=0.,
                       render_seconds=0.,visual_encoding_seconds=0.,network_inference_seconds=0.,numeric_features_seconds=0.)
            input_record=dict(observation=obs,candidates=[p.to_dict() for p in plans],snapshot_sha256=runner.initial,contracts=contracts)
            dump(directory/"inputs.json",input_record)
            geometry=None;visual=None
            if method in {"geometry","exhaustive"} or (method in self.models and self.models[method][1] in {"mlp","residual","prior"}):
                t=time.perf_counter();geometry=np.asarray([geometric_features(s,p,targets) for p in plans])
                times["optional_geometry_seconds"]=time.perf_counter()-t
            if method in {"geometry","exhaustive"}:
                scores=-geometry[:,0]-.001*geometry[:,4]
            elif method=="random":
                scores=np.random.default_rng(np.random.SeedSequence([task["seed"],domain_seed,51])).random(len(plans))
            else:
                model,kind,saved=self.models[method]
                if getattr(model,"config",None) and model.config.vision:
                    t=time.perf_counter();images=render_observation(s,directory,obs["objects"])
                    times["render_seconds"]=time.perf_counter()-t
                    synchronize(self.device);t=time.perf_counter();visual=self.vision.encode([directory/i for i in images]);synchronize(self.device)
                    times["visual_encoding_seconds"]=time.perf_counter()-t
                t=time.perf_counter()
                features=np.stack([numeric_features(obs,p,g) for p,g in zip(plans,geometry)]) if geometry is not None else None
                times["numeric_features_seconds"]=time.perf_counter()-t
                synchronize(self.device);t=time.perf_counter()
                group=dict(plans=plans,x=features,encoded=[encode_plan(obs,p) for p in plans],visual=visual)
                scores=predict(model,group,self.device,kind)
                synchronize(self.device);times["network_inference_seconds"]=time.perf_counter()-t
            if method=="exhaustive":k=len(plans);budget=len(plans)*repeats;allow_expand=False;mode="best_within_budget"
            ranking=ranking_rows(plans,scores,k);dump(directory/"top_k.json",ranking)
            occurrences={}
            def validate(plan):
                r=occurrences.get(plan.id,0);occurrences[plan.id]=r+1
                return runner.run(plan,perturbation(task["seed"],domain_seed*1000+r,"online"),keep_trace=True)
            selected=select_and_validate(ranking,validate,budget,mode=mode,repeats=repeats,
                                         accept_rate=.5,allow_expand=allow_expand)
            times["decision_wall_seconds"]=time.perf_counter()-start
            times["snapshot_restore_seconds"]=sum(x["restore_seconds"] for x in selected["validated"])
            times["twin_rollout_seconds"]=sum(x["wall_seconds"] for x in selected["validated"])
            times["deferred_solving_seconds_subset_of_rollout"]=sum(x["deferred_solving_seconds"] for x in selected["validated"])
            deployment=[]
            if selected["chosen"]:
                chosen=PlanIR.from_dict(selected["chosen"])
                for r in range(deployment_repeats):
                    deployment.append(runner.run(chosen,perturbation(task["seed"],domain_seed*1000+r,"deployment"),keep_trace=True))
            success=sum(x["success"] for x in deployment)/deployment_repeats
            times["independent_execution_seconds"]=sum(x["wall_seconds"]+x["restore_seconds"] for x in deployment)
            result=dict(request=request,request_sha256=digest(request),task=task,method=method,selection=selected,
                        timing=times,model_cold_start_seconds=self.cold.get(method,0.),
                        cold_visual_seconds=self.cold.get("visual_encoder",0.) if method in self.models and getattr(self.models[method][0],"visual",None) is not None else 0.,
                        checkpoint_sha256=self.checkpoint_hashes.get(method),
                        execution_success_rate=success,deployment=deployment,
                        physics_steps=sum(x["physics_steps"] for x in selected["validated"]),pool_counts=counts,
                        input_sha256=digest(input_record),parallel_workers=1,cache_policy="resident weights; fresh geometry/render/vision each decision",
                        deployment_scope="independent simulation deployment, not real robot")
            dump(file,result)
            return result


def offline(data,models,out,device="cuda"):
    groups=load_data(data,vision=True,include_test=True);test=[g for g in groups if g["split"]=="test"]
    result={}
    for name,path in models.items():
        model,kind,saved=load_model(path,device)
        # v1 features/protocol remain frozen; executing v2 is handled by PlanIR.
        scores=[predict(model,g,device,kind) for g in test]
        result[name]={str(k):metrics(test,scores,k) for k in (1,2,4,8)}
    result["geometry"]={str(k):metrics(test,[-g["geometry"][:,0]-.001*g["geometry"][:,4] for g in test],k) for k in (1,2,4,8)}
    dump(out,result)


def main():
    p=argparse.ArgumentParser();p.add_argument("--models",required=True);p.add_argument("--out",required=True)
    p.add_argument("--data");p.add_argument("--task-file");p.add_argument("--device",default="cuda")
    p.add_argument("--n",type=int,default=16);p.add_argument("--k",type=int,default=4)
    p.add_argument("--budget",type=int,default=8);p.add_argument("--repeats",type=int,default=2)
    a=p.parse_args();models=json.loads(Path(a.models).read_text())
    if a.data:offline(a.data,models,a.out,a.device);return
    experiment=OnlineExperiment(models,a.device);tasks=json.loads(Path(a.task_file).read_text());out=Path(a.out)
    rows=[]
    for task in tasks:
        for mode in ("first_verified","best_within_budget"):
            for method in ("random","geometry",*models):
                result=experiment.run(task,method,out/f"{task['family']}_{task['seed']}_{task.get('checkpoint',0)}_{method}_{mode}",
                    n=a.n,k=a.k,budget=a.budget,repeats=a.repeats,mode=mode,allow_expand=mode=="first_verified")
                rows.append(result);print(json.dumps(dict(task=task,method=method,mode=mode,success=result["execution_success_rate"],seconds=result["timing"]["decision_wall_seconds"])),flush=True)
        rows.append(experiment.run(task,"exhaustive",out/f"{task['family']}_{task['seed']}_{task.get('checkpoint',0)}_exhaustive",
                                   n=a.n,repeats=a.repeats))
    dump(out/"results.json",rows)


if __name__=="__main__":main()
