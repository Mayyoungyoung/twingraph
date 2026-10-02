"""Freeze the new 50-layout shared candidate pools without executing any plan."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
import multiprocessing
from pathlib import Path
from scripts.train_value_v32 import read,write,sha
from scripts.collect_robust_value_v33 import RUNTIME
from scripts.collect_sliding_assembly_v23 import controller_plan,TASK_VERSION
from scripts.llm_transport_v32 import choose
from simbench.value.provenance_v12 import fingerprint
from simbench.value.plan import digest
from simbench.value.planner_v12 import propose,assembly_program
from simbench.value.system_v12 import make_scene


def freeze_one(job):
    seed,root=job;out=Path(root)/f'seed_{seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'request.json').exists():
        r=read(out/'request.json')
        if r['seed']!=seed or r['runtime_sha256']!=RUNTIME or len(r['candidates'])!=8:raise ValueError('frozen test request mismatch')
        if not (out/'freeze_receipt.json').exists() or read(out/'freeze_receipt.json')['request_sha256']!=digest(r):raise ValueError('frozen test request receipt mismatch')
        return dict(seed=seed,request_sha256=digest(r))
    if fingerprint()['sha256']!=RUNTIME:raise ValueError('physical runtime mismatch')
    _,planning,_,_=make_scene(seed,out/'planning',domain='online',level='L0',observation_backend='mujoco_state_pose',position_noise_std_m=0.,yaw_noise_std_rad=0.)
    planning.sliding_assembly_v23=True;planning.end_stop_place_acceptance_v12='stable_supported'
    observation=deepcopy(planning.decision_observation)
    pool,grounding=propose(observation,cad=planning.planning_cad,n=48,seed=seed,completed=('cleaning',))
    selected,model_source=choose(pool,observation,planning.planning_cad,out/'planner',8)
    entries=[];seen=set()
    for proposal in selected:
        proposal={k:deepcopy(v) for k,v in proposal.items() if k not in ('wipe_variant','wipe_force','wipe_duration')}
        if any(g.get('status')=='rejected' for g in proposal.get('necessary_geometry',{}).values()):raise ValueError('rejected geometry selected')
        complete=controller_plan(planning,proposal);assembly=assembly_program(planning,proposal)
        signature=digest(complete.to_dict()['calls'])
        if signature in seen:raise ValueError('duplicate frozen test plan')
        seen.add(signature)
        entries.append(dict(name=proposal['name'],proposal=proposal,complete_candidate_plan_ir=complete.to_dict(),
            plan_sha256=digest(complete.to_dict()),assembly_plan_ir=assembly.to_dict(),assembly_plan_sha256=digest(assembly.to_dict())))
    request=dict(schema='twingraph.frozen_test.v33',task_version=TASK_VERSION,seed=seed,level='L0',domain='online',
        runtime_sha256=RUNTIME,decision_observation=observation,observation_random_state=planning.state_observation_random_state,
        position_noise_std_m=0.,yaw_noise_std_rad=0.,candidates=entries,source=dict(grounding=grounding,**model_source),
        candidate_n=8,pool_n=48,outcome_labels_read=False,layout_selected_from_prior_outcomes=False)
    write(out/'request.json',request)
    write(out/'freeze_receipt.json',dict(request_sha256=digest(request),script_sha256=sha(__file__)))
    return dict(seed=seed,request_sha256=digest(request))


def freeze(root,training,workers=2):
    selected=read(training/'selection_frozen.json')
    if not all(sha(training/n)==h for n,h in selected['checkpoint_sha256'].items()):raise ValueError('model not frozen')
    root.mkdir(parents=True,exist_ok=True)
    manifest=dict(seeds=list(range(4000,4050)),split='test',candidates_per_layout=8,
        runtime_sha256=RUNTIME,selection_frozen_sha256=sha(training/'selection_frozen.json'),outcome_based_resampling=False)
    if (root/'manifest.json').exists() and read(root/'manifest.json')!=manifest:raise ValueError('test protocol changed')
    write(root/'manifest.json',manifest);rows=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(freeze_one,(s,str(root))) for s in manifest['seeds']]):
            rows.append(future.result());write(root/'progress.json',dict(complete=len(rows)==50,layouts=len(rows),rows=rows))
    return rows


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--training',type=Path,required=True)
    a=p.parse_args();freeze(a.out,a.training)
