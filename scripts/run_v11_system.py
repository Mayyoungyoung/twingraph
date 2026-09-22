"""Run preregistered methods sequentially; no cached rollouts used for timings."""
import argparse
import json
from pathlib import Path
from simbench.value.system_v11 import FrozenValue, run_method


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--seeds",type=int,nargs="+",default=[1520,1521,1522])
    p.add_argument("--methods",nargs="+",choices=["all_twin","random_top_k","value_top_k","random_early_stop"],
                   default=["all_twin","random_top_k","value_top_k","random_early_stop"])
    p.add_argument("--k",type=int,default=4)
    p.add_argument("--level",choices=["L0","L1","L2"],default="L1")
    p.add_argument("--record",action="store_true")
    a=p.parse_args(); value=FrozenValue(a.checkpoint)
    for seed in a.seeds:
        for method in a.methods:
            root=a.out / f"seed_{seed}" / method
            if (root / "summary.json").exists(): continue
            result=run_method(seed,method,root,value,k=a.k,level=a.level,record=a.record)
            print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__": main()
