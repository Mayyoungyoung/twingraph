"""Physical evidence for nesting, parameter effects and geometry non-mutation."""
import argparse
import contextlib
import copy
from pathlib import Path
import time
import numpy as np
from .research_scenarios import make_family,build_pool,program,observed,geometric_features,semantic_key
from .physical import PhysicalRunner,perturbation
from .collect import dump
from .encode import encode_plan
from simbench.assembly.candidates import fingerprint


def main():
    p=argparse.ArgumentParser();p.add_argument("--out",default="results/value_v2/checks");a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);result={}
    with (out/"steps.log").open("w") as log,contextlib.redirect_stdout(log):
        for family,seed in (("sliding_stage_pin",21000),("rigid_connector_module",20000)):
            d=out/family;d.mkdir(exist_ok=True);spec,s,_,targets=make_family(family,seed,d)
            pools=[];counts=[]
            for n in (16,32,64):
                plans,c=build_pool(s,targets,seed,n);pools.append(plans);counts.append(c)
            assert [p.id for p in pools[0]]==[p.id for p in pools[2]][:len(pools[0])]
            assert [p.id for p in pools[1]]==[p.id for p in pools[2]][:len(pools[1])]
            assert len({semantic_key(p) for p in pools[2]})==len(pools[2])
            before=fingerprint(s);start=time.perf_counter()
            geo=[geometric_features(s,p,targets) for p in pools[2]]
            geo_time=time.perf_counter()-start;assert before==fingerprint(s)
            order=list(s.parts);base={p:dict(yaw=0.,height=.003,clearance=1.035,force=3.,speed=.006) for p in order}
            trial_results=[];runner=PhysicalRunner(s)
            for tag,changes in (("base",{}),("speed",dict(speed=.008)),("force",dict(force=4.)),("route",dict(clearance=.98)),("grasp",dict(yaw=np.pi/2,height=0.))):
                s.restore(runner.snapshot)
                choices=copy.deepcopy(base);choices[order[0]].update(changes)
                plan=program(s,targets,order,choices);samples=[];counter=[0];original=s.ctx.step
                def step(*args,**kw):
                    result=original(*args,**kw);counter[0]+=1
                    if counter[0]%25==0:samples.append([float(s.ctx.data.time),*s.ctx.eef_pos(),*s.ctx.obj_pos(order[0]),*s.ctx.finger_qpos])
                    return result
                s.ctx.step=step
                try:row=runner.run(plan,perturbation(seed,0,"regression"),keep_trace=True)
                finally:s.ctx.step=original
                np.savez_compressed(d/f"{tag}_trajectory.npz",samples=np.asarray(samples))
                row["tag"]=tag;row["trajectory_columns"]=["sim_time","eef_x","eef_y","eef_z","part_x","part_y","part_z","finger0","finger1"]
                trial_results.append(row)
            result[family]=dict(pool_counts=counts,geometry_64_seconds=geo_time,geometry=geo,parameter_trials=trial_results)
    dump(out/"checks.json",result)


if __name__=="__main__":main()
