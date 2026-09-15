"""Rank an immutable input file with v2 weights; never open outcome labels."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from .plan import PlanIR,digest
from .program_audit import audit_program
from .research_learning import load_model,numeric_features,predict
from .research_evaluate import ranking_rows,synchronize
from .encode import encode_plan
from .collect import dump


def rank_input(checkpoint,input_file,k=4,device="cpu"):
    path=Path(input_file);inputs=json.loads(path.read_text());ih=digest(inputs)
    model,kind,saved=load_model(checkpoint,device)
    plans=[PlanIR.from_dict(p) for p in inputs["candidates"]]
    obs=inputs["observation"]
    for plan in plans:audit_program(plan,obs["objects"])
    features=None;visual=None
    if kind in {"mlp","residual","prior"}:
        geometry=json.loads(path.with_name("geometry.json").read_text())
        if geometry["input_sha256"]!=ih:raise ValueError("stale geometric features")
        if len(geometry["features"])!=len(plans):raise ValueError("geometry/plan count mismatch")
        features=np.stack([numeric_features(obs,p,g) for p,g in zip(plans,geometry["features"])])
    if getattr(model,"config",None) and model.config.vision:
        cache=path.with_name("visual.npz")
        if cache.exists():
            from .vision import ENCODER
            stored=np.load(cache,allow_pickle=False)
            if str(stored["input_sha256"])!=ih or str(stored["encoder"])!=ENCODER:raise ValueError("stale visual features")
            visual=stored["features"]
        else:
            from .vision import FrozenVision
            visual=FrozenVision(device).encode([path.parent/i for i in inputs["images"]])
    synchronize(device);start=time.perf_counter()
    group=dict(plans=plans,x=features,encoded=[encode_plan(obs,p) for p in plans],visual=visual)
    scores=predict(model,group,device,kind);synchronize(device)
    result=ranking_rows(plans,scores,k)
    result.update(schema="twingraph.topk.v2",input_sha256=ih,checkpoint=str(checkpoint),
                  inference_seconds=time.perf_counter()-start,scoring_kind=kind,
                  status="requires_twin_validation",score_semantics="uncalibrated empirical-success ranking")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--checkpoint",required=True)
    p.add_argument("--input",required=True);p.add_argument("--out",required=True)
    p.add_argument("--k",type=int,default=4);p.add_argument("--device",default="cpu");a=p.parse_args()
    result=rank_input(a.checkpoint,a.input,a.k,a.device);dump(a.out,result)
    print(json.dumps(dict(selected=[r["candidate_id"] for r in result["top_k"]],inference_seconds=result["inference_seconds"])))


if __name__=="__main__":main()
