"""Replay predeclared proposals for controller development; excluded from training."""
import argparse
import contextlib
import json
from pathlib import Path
from scripts.collect_sliding_assembly_v23 import controller_plan, REQUIRED
from simbench.value.system_v12 import make_scene, save
from simbench.value.physical import PhysicalRunner, perturbation


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--request', default='results/v31_rear_push_0/request.json')
    p.add_argument('--index',type=int,default=0)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    request=json.loads(Path(a.request).read_text(encoding='utf-8'))
    proposal=request['candidates'][a.index]['proposal']
    _,s,_,_=make_scene(request['seed'],a.out/'scene',domain='regression',level='L0',observation_backend='mujoco_state_pose')
    s.sliding_assembly_v23=True
    s.end_stop_place_acceptance_v12='stable_supported'
    s.required_stage_passes=REQUIRED
    original=s.call
    def traced(name,**params):
        before=s.ctx.obj_pos('carriage').tolist()
        try:
            return original(name,**params)
        finally:
            print(json.dumps(dict(action=name,params=params,before=before,after=s.ctx.obj_pos('carriage').tolist()),default=str),flush=True)
    s.call=traced
    plan=controller_plan(s,proposal)
    with (a.out/'console.log').open('w',encoding='utf-8') as f, contextlib.redirect_stdout(f):
        result=PhysicalRunner(s,timeout=900).run(plan,perturbation(request['seed'],0,'regression'),keep_trace=True)
    save(a.out/'result.json',result)
    print(json.dumps(dict(success=result['success'],error=result['error'],wall_seconds=result['wall_seconds'])),flush=True)

if __name__=='__main__':
    main()
