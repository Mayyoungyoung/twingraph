"""Describe distribution shift of frozen value inputs without changing scores."""
import argparse
import json
from pathlib import Path
import numpy as np
from simbench.value.planner_v11 import value_features
from simbench.value.system_v11 import FrozenValue


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();value=FrozenValue(a.checkpoint)
    mean=value.model.mean.detach().cpu().numpy();std=value.model.std.detach().cpu().numpy()
    rows=[]
    for path in sorted(a.root.glob("seed_*/all_twin/twins/*/input_graph.json")):
        g=json.loads(path.read_text());x=value_features(g)
        z=np.abs((x-mean)/std)
        rows.append(dict(seed=int(path.parts[-5].split("_")[-1]),candidate=g["proposal"]["name"],
                         probability=value.score([g])[0],max_standardized_deviation=float(z.max()),
                         near_constant_train_features_changed=np.flatnonzero((std<=1.01e-5)&(np.abs(x-mean)>1e-4)).tolist()))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(dict(model_sha256=value.sha256,rows=rows,
        limitation="Old V9 training kept grasp yaw constant. New 90-degree grasps are out of distribution; probabilities are not calibrated evidence of feasibility."),indent=2),encoding="utf-8")
    print(json.dumps(dict(rows=len(rows),max_z=max(r["max_standardized_deviation"] for r in rows),
        rows_changing_near_constant_features=sum(bool(r["near_constant_train_features_changed"]) for r in rows))))


if __name__=="__main__":main()
