"""Label-blind model selection of grounded atomic compositions, with provenance."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import time
import os
from simbench.assembly.graph import catalog_graph
from simbench.value.plan import digest


def choose(pool, observation, cad, out, count):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    roles=tuple(pool[0]['choices'])
    library=[dict(index=i,order=p['order'],choices=p['choices'],
                  geometry={k:{f:v.get(f) for f in ('status','min_clearance_m','required_clearance_m')}
                            for k,v in p['necessary_geometry'].items()}) for i,p in enumerate(pool)]
    graph=catalog_graph()
    request=dict(schema='twingraph.llm_transport.v32',candidate_count=count,
        task='Assemble carriage, end stop, both pins, and handle. Start at carriage; finish after releasing the installed handle. No wiping or sliding test.',
        task_goal='Physically supported carriage in rail, captured stop, pins through stop and base, seated handle, robot released and retracted.',
        transport_alternatives={
            'held_insert':['detect','estimate_pose','estimate_grasp','plan_path','move','grasp','move','plan_path:contact','insert','press','place'],
            'released_push':['detect','estimate_pose','estimate_grasp','plan_path','move','grasp','move','place','move','detect','estimate_pose','estimate_push_pose','plan_path','move','gripper:close_empty','plan_path:push','push']},
        observation_sha256=observation['sha256'],pool_sha256=digest(pool),
        observation={k:observation[k] for k in ('objects','fixtures','receiver_geometry','assembly_targets')},
        atomic_skills=[{**{k:n[k] for k in ('name','description','inputs','outputs')},
            'contracts':[dict(name=c['name'],requires=c['requires'],produces=c['produces'],effects=c['effects'])
                         for c in n['contracts']]} for n in graph['nodes']],
        atomic_dependencies=[{k:e[k] for k in ('source','target','supplies')} for e in graph['edges']],
        grounded_options=library,
        rules=['Choose the transport composition by selecting its carriage donor; neither mode is mandatory.',
               'Choose diverse plausible candidates using only task, observation and skill contracts.',
               'Each role donor binds all its grounded geometric and control ports; do not invent numeric values.',
               'No outcome labels, special successful seed, or predefined successful baseline are available.',
               'Necessary geometry is not a guarantee of physical success.'])
    schema=dict(type='object',additionalProperties=False,
        required=['observation_sha256','pool_sha256','candidates'],properties=dict(
            observation_sha256=dict(type='string'),pool_sha256=dict(type='string'),
            candidates=dict(type='array',minItems=count,maxItems=count,items=dict(type='object',additionalProperties=False,
                required=['name','order_donor','role_donors'],properties=dict(name=dict(type='string'),
                order_donor=dict(type='integer',minimum=0,maximum=len(pool)-1),
                role_donors=dict(type='object',additionalProperties=False,required=list(roles),
                    properties={r:dict(type='integer',minimum=0,maximum=len(pool)-1) for r in roles}))))))
    for name,value in (('request.json',request),('response.schema.json',schema)):
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    response_path=(out/'response.json').resolve()
    started=time.perf_counter()
    if os.environ.get('TWINGRAPH_EXTERNAL_PLANNER') == '1':
        while not response_path.is_file():
            if time.perf_counter()-started>1800:
                raise TimeoutError('external model response did not arrive; no rollout started')
            time.sleep(1.)
    else:
        infer(out)
    response=json.loads(response_path.read_text(encoding='utf-8'))
    if response['observation_sha256']!=request['observation_sha256'] or response['pool_sha256']!=request['pool_sha256']:
        raise ValueError('model response hashes do not bind the frozen observation/library')
    if len(response['candidates'])!=count:raise ValueError('wrong model candidate count')
    selected=[]; signatures=set()
    for i,row in enumerate(response['candidates']):
        indices=[row['order_donor'],*row['role_donors'].values()]
        if (row['name']!=f'llm_{i:03d}' or set(row['role_donors'])!=set(roles)
                or any(type(j) is not int or not 0<=j<len(pool) for j in indices)):
            raise ValueError('invalid model donor binding')
        p=deepcopy(pool[row['order_donor']])
        for role,j in row['role_donors'].items():
            p['choices'][role]=deepcopy(pool[j]['choices'][role])
            p['necessary_geometry'][role]=deepcopy(pool[j]['necessary_geometry'][role])
        p.update(name=row['name'],source='llm_grounded_atomic_composition_v32',rationale='model-selected from observation and generic skill graph')
        signature=digest(dict(order=p['order'],choices=p['choices']))
        if signature in signatures:raise ValueError('model returned duplicate executable compositions')
        signatures.add(signature);selected.append(p)
    metadata=dict(source='llm_grounded_atomic_composition_v32',online_llm_call=True,
        request_sha256=digest(request),response_sha256=digest(response),wall_seconds=time.perf_counter()-started,
        response_file=str(response_path),outcome_labels_read=False,transport_mode_forced=False)
    (out/'provenance.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return selected,metadata


def infer(out):
    """Model-only operation; receives a frozen request and no rollout files."""
    out=Path(out)
    request=json.loads((out/'request.json').read_text(encoding='utf-8'))
    executable=shutil.which('codex.exe') or shutil.which('codex')
    if not executable:raise RuntimeError('Codex planner executable is unavailable')
    response_path=(out/'response.json').resolve()
    command=[executable,'exec','-s','read-only','--ephemeral','--skip-git-repo-check',
             '--output-schema',str((out/'response.schema.json').resolve()),'-o',str(response_path),'-']
    prompt=('You are a robotic symbolic planner. Do not use tools or inspect files. Use ONLY the attached '
        'pre-execution JSON. Return exactly the requested number of unique complete candidate compositions '
        'named llm_000, llm_001, etc. Copy both hashes exactly. Choose order and independent role donors. '
        'The carriage donor chooses either held insertion or released pushing; choose based on task and skill graph. '
        'Do not impose a single transport mode. Physical success is unknown. Return only schema-compliant JSON.\n'
        +json.dumps(request,ensure_ascii=False,separators=(',',':')))
    with (out/'stdout.log').open('wb') as stdout, (out/'stderr.log').open('wb') as stderr:
        process=subprocess.run(command,input=prompt.encode('utf-8'),stdout=stdout,stderr=stderr,timeout=900,
                               cwd=out,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if process.returncode or not response_path.is_file():
        raise RuntimeError('model planner failed; inspect planner/stderr.log; no rollout started')
