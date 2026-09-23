"""Run real V12 printed-kit candidates or paired screening methods."""
import argparse
import json
from pathlib import Path
from simbench.value.system_v12 import METHODS, collect, run_method


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--seeds",type=int,nargs="+",required=True)
    parser.add_argument("--n",type=int,default=48)
    parser.add_argument("--k",type=int,default=4)
    parser.add_argument("--level",choices=["L0","L1","L2"],default="L1")
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--methods",nargs="+",choices=["collect",*METHODS],default=["collect"])
    parser.add_argument("--names",nargs="+")
    parser.add_argument("--domain",choices=["train","development","online"],default="train")
    parser.add_argument("--record",action="store_true")
    args=parser.parse_args()
    value=None
    if args.checkpoint:
        import torch
        saved=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
        schema=saved.get("schema") if isinstance(saved,dict) else None
        if schema=="twingraph.full_flow_graph_value.v20.r1":
            from simbench.value.full_flow_graph_value_v20 import ValueRankerV20
            value=ValueRankerV20(args.checkpoint)
        elif schema=="twingraph.atomic_graph_value.v15.r2":
            from simbench.value.generic_graph_value_v15 import ValueRankerV15
            value=ValueRankerV15(args.checkpoint)
        else:
            from simbench.value.graph_value_v12 import ValueRankerV12
            value=ValueRankerV12(args.checkpoint)
    if any(m in ("value_top_k","value_early_stop") for m in args.methods) and value is None:
        parser.error("value methods require --checkpoint")
    for seed in args.seeds:
        for method in args.methods:
            root=args.out/f"seed_{seed}"/method
            if method=="collect":
                result=collect(seed,root,n=args.n,level=args.level,names=args.names,record=args.record,domain=args.domain)
            else:
                if (root/"summary.json").exists(): continue
                result=run_method(seed,method,root,value,n=args.n,k=args.k,level=args.level,record=args.record)
            print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__": main()
