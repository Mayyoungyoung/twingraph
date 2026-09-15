"""Load a trained checkpoint and return the actual Top-K executable PlanIRs."""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from .encode import encode_plan, collate
from .network import ModelConfig, PlanValueNet
from .plan import PlanIR, digest


class PlanRanker:
    def __init__(self,checkpoint,device="cpu"):
        saved=torch.load(checkpoint,map_location=device,weights_only=False)
        if saved.get("schema")!="twingraph.value.v1":
            raise ValueError("unsupported value checkpoint")
        self.saved=saved
        self.device=device
        self.model=PlanValueNet(ModelConfig(**saved["model_config"])).to(device).eval()
        self.model.load_state_dict(saved["state_dict"])

    @torch.inference_mode()
    def rank(self,observation,plans,k=2,visual=None,batch_size=8):
        if k<1 or batch_size<1:
            raise ValueError("k and batch_size must be positive")
        if len({p.id for p in plans})!=len(plans):
            raise ValueError("duplicate candidate ids")
        for plan in plans:
            plan.validate(observation["objects"])
            if plan.protocol!=self.saved["protocol"]:
                raise ValueError("plan execution protocol differs from checkpoint")
        eligible=[p for p in plans if p.status!="conflict"]
        scores=[]
        started=time.perf_counter()
        for start in range(0,len(eligible),batch_size):
            batch_plans=eligible[start:start+batch_size]
            enc=[encode_plan(observation,p) for p in batch_plans]
            visuals=[visual]*len(enc) if self.model.config.vision else None
            if self.model.config.vision and visual is None:
                raise ValueError("this checkpoint requires observation visual features")
            result=self.model(collate(enc,visuals,self.device))
            q=result["q"] if self.saved["objective"]=="dual" else result["direct_logit"].sigmoid()
            for i,plan in enumerate(batch_plans):
                scores.append(dict(candidate_id=plan.id,score=float(q[i]),p_prefix=float(result["p_prefix"][i]) if self.saved["objective"]=="dual" else None,
                                   p_suffix_given_prefix=float(result["p_suffix"][i]) if self.saved["objective"]=="dual" else None,status=plan.status,plan=plan.to_dict()))
        if any(not np.isfinite(row["score"]) for row in scores):
            raise RuntimeError("nonfinite value scores")
        # Stable input order breaks exact ties, never candidate ID semantics.
        scores.sort(key=lambda r:-r["score"])
        return dict(schema="twingraph.topk.v1",requested_k=k,returned_k=min(k,len(scores)),
                    top_k=scores[:k],ranked=scores,inference_seconds=time.perf_counter()-started,
                    rejected_conflicts=[p.id for p in plans if p.status=="conflict"],
                    status="requires_twin_validation" if scores else "expand_candidates")


def select_and_validate(ranking,validate,budget):
    """validate(PlanIR) -> bool; caller owns snapshot isolation for each trial.

    Try the screened candidates first and expand in score order only when
    needed and authorized by the fixed total rollout budget. No feasible plan
    found in that budget means unresolved, never proven task infeasibility.
    """
    if budget<0:
        raise ValueError("validation budget must be nonnegative")
    checked=[]
    chosen=None
    for row in ranking["ranked"][:budget]:
        plan=PlanIR.from_dict(row["plan"])
        ok=validate(plan)
        if not isinstance(ok,bool):
            raise TypeError("validator must return a boolean")
        checked.append(dict(candidate_id=plan.id,success=ok))
        if ok:
            chosen=plan.to_dict()
            break
    return dict(top_k=ranking["top_k"],validated=checked,chosen=chosen,validation_calls=len(checked),
                status="validated" if chosen else "budget_exhausted_expand_or_resample")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--input",required=True,help="group inputs.json, with no outcome labels")
    p.add_argument("--out",default="results/value/top_k.json")
    p.add_argument("--k",type=int,default=2)
    p.add_argument("--device",default="cpu")
    a=p.parse_args()
    file=Path(a.input)
    inputs=json.loads(file.read_text())
    ranker=PlanRanker(a.checkpoint,a.device)
    visual=None
    if ranker.model.config.vision:
        cache=file.parent/"visual.npz"
        if cache.exists():
            features=np.load(cache,allow_pickle=False)
            if str(features["input_sha256"])!=digest(inputs) or str(features["encoder"])!=ranker.saved["vision_encoder"]:
                raise ValueError("stale or incompatible visual cache")
            visual=features["features"]
        else:
            from .vision import FrozenVision
            visual=FrozenVision(a.device).encode([file.parent/f for f in inputs["images"]])
    result=ranker.rank(inputs["observation"],[PlanIR.from_dict(c) for c in inputs["candidates"]],a.k,visual)
    from .collect import dump
    dump(a.out,result)
    print(json.dumps(dict(output=a.out,selected=[r["candidate_id"] for r in result["top_k"]],scores=[r["score"] for r in result["top_k"]],inference_seconds=result["inference_seconds"])))


if __name__=="__main__":
    main()

