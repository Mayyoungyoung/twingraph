"""Development-only acquisition and holder withdrawal; no insertion or trial relabeling."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from simbench.value.stage_v12 import make_scene
from simbench.assembly.skills_v12 import configure_v12_skills
from simbench.value.stage_v5 import stage_calls
from simbench.value.plan import execute_calls,argument


def run(seed,yaw,out,first_lift=None):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    _,session,_,targets=make_scene(seed,out)
    configure_v12_skills(session)
    choice=dict(yaw=yaw,height=0.,clearance=.99,force=6.,speed=.004,
                force_limit=12.,press_force=2.)
    if first_lift is not None: choice["lift_first_m"]=first_lift
    calls=stage_calls("pin_left",targets["pin_left"],choice,0,v7=True,v12=True)[:11 if first_lift is not None else 10]
    calls[0].arguments["required_parts"]=argument(["pin_left"])
    plan=SimpleNamespace(prefix={"part":"pin_left","targets":{"pin_left":targets["pin_left"]}})
    error=None
    try: execute_calls(session,plan,calls)
    except Exception as exc: error=str(exc)
    result=dict(seed=seed,yaw=yaw,first_lift_m=first_lift,success=error is None,error=error,
                scope="development-only RGBD acquisition and real-holder withdrawal",steps=session.results,
                grasp=session.artifacts.get("grasp"),grasps=session.artifacts.get("grasps"))
    (out / "summary.json").write_text(json.dumps(result,indent=2,default=lambda a:np.asarray(a).tolist()))
    print(json.dumps({k:v for k,v in result.items() if k not in ("steps","grasp","grasps")}))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--seed",type=int,default=1600)
    parser.add_argument("--yaw",type=float,default=0.);parser.add_argument("--out",required=True)
    parser.add_argument("--first-lift",type=float)
    a=parser.parse_args();run(a.seed,a.yaw,a.out,a.first_lift)
