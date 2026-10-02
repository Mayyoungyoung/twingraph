"""Read-only diagnostics of the first failed approach in a development replay."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from scripts.collect_sliding_assembly_v23 import controller_plan, REQUIRED
from simbench.value.system_v12 import make_scene, save
from simbench.value.physical import PhysicalRunner, perturbation

p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True)
p.add_argument('--index',type=int,default=0);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
request=json.loads(a.request.read_text(encoding='utf8'))
_,s,_,_=make_scene(request['seed'],a.out/'scene',domain='online',level='L0',observation_backend='mujoco_state_pose')
s.sliding_assembly_v23=True;s.end_stop_place_acceptance_v12='stable_supported';s.required_stage_passes=REQUIRED
result=PhysicalRunner(s,timeout=600).run(controller_plan(s,request['candidates'][a.index]['proposal']),perturbation(request['seed'],0,'online'),keep_trace=True)
contacts=[]
for i,c in enumerate(s.ctx.data.contact):
    f=np.zeros(6);mujoco.mj_contactForce(s.ctx.model,s.ctx.data,i,f)
    if f[0]>.05:
        contacts.append(dict(pair=[s.ctx.model.geom(g).name for g in (c.geom1,c.geom2)],normal_n=float(f[0]),penetration_m=float(c.dist)))
save(a.out/'diagnostic.json',dict(error=result['error'],eef=s.ctx.eef_pos().tolist(),
    goal=np.asarray(s.artifacts.get('grasp',{}).get('xyz',[])).tolist(),contacts=contacts,
    controller_error_rad=(s.ctx.data.ctrl[s.ctx.arm_act_ids]-s.ctx.arm_qpos).tolist()))
