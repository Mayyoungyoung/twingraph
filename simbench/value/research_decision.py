"""One new scene: executable Top-K, explicit budget, independent deployment."""
import argparse
import json
from .research_evaluate import OnlineExperiment


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True);p.add_argument("--out",required=True)
    p.add_argument("--family",choices=["rigid_connector_module","sliding_stage_pin"],default="rigid_connector_module")
    p.add_argument("--seed",type=int,default=30000);p.add_argument("--checkpoint-stage",type=int,default=0)
    p.add_argument("--n",type=int,default=32);p.add_argument("--k",type=int,default=4)
    p.add_argument("--budget",type=int,default=8);p.add_argument("--repeats",type=int,default=2)
    p.add_argument("--mode",choices=["first_verified","best_within_budget"],default="best_within_budget")
    p.add_argument("--allow-expand",action="store_true");p.add_argument("--device",default="cuda")
    a=p.parse_args();experiment=OnlineExperiment({"learned":a.checkpoint},a.device)
    result=experiment.run(dict(family=a.family,seed=a.seed,checkpoint=a.checkpoint_stage),"learned",a.out,
                          n=a.n,k=a.k,budget=a.budget,repeats=a.repeats,mode=a.mode,allow_expand=a.allow_expand)
    print(json.dumps(dict(status=result["selection"]["status"],rollouts=result["selection"]["validation_calls"],
                          independent_execution_success=result["execution_success_rate"],timing=result["timing"])))


if __name__=="__main__":main()
