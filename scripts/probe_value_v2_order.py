"""Development-only paired order intervention with identical per-object grasps."""
import contextlib
from pathlib import Path
from simbench.value.research_scenarios import ConnectorSpec,make_connector,program
from simbench.value.physical import PhysicalRunner,perturbation
from simbench.value.collect import dump
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--out',default='results/value_v2/spacing_probe040')
parser.add_argument('--spacing',type=float,default=.040)
args=parser.parse_args()
out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
rows=[]
with (out/'steps.log').open('w') as log,contextlib.redirect_stdout(log):
 for seed in [20000,20001]:
  spec=ConnectorSpec.sample(seed);spec.spacing=args.spacing
  d=out/str(seed);d.mkdir(exist_ok=True);s,_,targets=make_connector(spec,d)
  runner=PhysicalRunner(s)
  choices={p:dict(yaw=0.,height=.003,clearance=1.035,force=3.,speed=.008) for p in s.parts}
  choices['insert_0']['yaw']=1.5707963267948966
  for order in [['insert_0','insert_1','guard','locator'],['insert_1','insert_0','guard','locator']]:
   s.restore(runner.snapshot);p=program(s,targets,order,choices)
   r=runner.run(p,perturbation(seed,0,'train'),keep_trace=True)
   r['seed']=seed;r['spacing']=spec.spacing;r['order']=order
   r['finger_contacts']=[dict(pair=[s.ctx.model.geom(c.geom1).name,s.ctx.model.geom(c.geom2).name],distance=float(c.dist)) for c in s.ctx.data.contact if any('finger' in s.ctx.model.body(int(s.ctx.model.geom_bodyid[g])).name for g in [c.geom1,c.geom2])]
   rows.append(r);dump(out/'result.json',rows)
for r in rows:print(r['seed'],r['order'],r['success'],r['error'],r['finger_contacts'][:8],flush=True)
