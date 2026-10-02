"""Export all valid frozen candidates with separate pre-rollout features and labels."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from simbench.assembly.library import DEFAULT_CAPABILITIES
from simbench.value.plan import PlanIR, digest
from simbench.value.skill_graph import compile_graph
from simbench.value.generic_graph_value_v15 import encode_graph, SCHEMA, FEATURES
from simbench.value.provenance_v12 import fingerprint


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def export(root, out):
    manifest=read(root/'manifest.json')
    active=manifest.get('active_runtime_sha256',manifest['runtime_sha256'])
    if active!=fingerprint()['sha256']:
        raise ValueError('export with the frozen runtime that generated these candidates')
    allowed={manifest['runtime_sha256']}
    for revision in manifest.get('runtime_revisions',[]):
        if (revision['parent_sha256']!=manifest['runtime_sha256'] or
                revision['changed_files']!=['simbench/assembly/placement_catalog_v13.py']):
            raise ValueError('unreviewed runtime revision; encoder/controller compatibility unknown')
        allowed.add(revision['sha256'])
    out.mkdir(parents=True,exist_ok=True)
    records=[]; excluded=[]
    for job in manifest['jobs']:
        layout=root/f"seed_{job['seed']}"
        if not (layout/'request.json').is_file():continue
        request=read(layout/'request.json')
        if request['runtime_sha256'] not in allowed:
            raise ValueError('mixed runtime')
        planner_log=layout/'planner'/'stderr.log'
        header=''
        if planner_log.exists():
            with planner_log.open(encoding='utf-8',errors='replace') as stream:
                header=''.join(next(stream,'') for _ in range(14))
        model=re.search(r'^model: (.+)$',header,re.M)
        effort=re.search(r'^reasoning effort: (.+)$',header,re.M)
        for entry in request['candidates']:
            directory=layout/'candidates'/entry['name']
            if not (directory/'result.json').is_file():continue
            result=read(directory/'result.json')
            sample_id=f"seed_{job['seed']}_{entry['name']}"
            if not result['valid'] or result.get('resource_censored'):
                excluded.append(dict(id=sample_id,reason=result.get('invalid_reason','invalid')))
                continue
            raw=read(directory/'input_graph.json')
            if (digest(raw)!=result['input_graph_sha256'] or
                    result['plan_sha256']!=entry['plan_sha256'] or
                    result['runtime_sha256']!=request['runtime_sha256'] or
                    digest(raw['complete_candidate_plan_ir'])!=entry['plan_sha256'] or
                    digest(raw['assembly_plan_ir'])!=entry['assembly_plan_sha256']):
                raise ValueError(f'provenance mismatch: {sample_id}')
            observation=deepcopy(raw['decision_observation'])
            observation.setdefault('robot',{})
            observation['goals']=[dict(predicate='assembly_through_handle',both_pins_through_base=True)]
            for part,capabilities in DEFAULT_CAPABILITIES.items():
                if part in observation['objects']:
                    observation['objects'][part]['capabilities']=list(capabilities)
            graph=compile_graph(observation,PlanIR.from_dict(raw['assembly_plan_ir']))
            # Nothing from result, stage audit, or execution trace is passed to the encoder.
            encoded=encode_graph({'assembly':graph})
            target=out/'inputs'/sample_id;target.mkdir(parents=True,exist_ok=True)
            (target/'graph.json').write_text(json.dumps(graph,ensure_ascii=False),encoding='utf-8')
            np.savez_compressed(target/'features.npz',**encoded)
            records.append(dict(id=sample_id,seed=job['seed'],split=job['split'],
                runtime_sha256=result['runtime_sha256'],
                planner_model=model.group(1).strip() if model else 'unrecorded',
                planner_reasoning_effort=effort.group(1).strip() if effort else 'unrecorded',
                label=int(result['success']),input=f'inputs/{sample_id}/features.npz',
                raw_input_sha256=result['input_graph_sha256'],plan_sha256=entry['plan_sha256'],
                source=str(directory.relative_to(root)),
                transport_mode=entry['proposal']['choices']['carriage']['transport_mode']))
    temporary=out/'labels.jsonl.tmp'
    temporary.write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
    temporary.replace(out/'labels.jsonl')
    report=dict(schema=SCHEMA,runtime_sha256=active,purpose=manifest['purpose'],
        runtime_revisions=manifest.get('runtime_revisions',[]),
        samples_by_runtime={sha:sum(r['runtime_sha256']==sha for r in records)
            for sha in sorted({r['runtime_sha256'] for r in records})},
        samples_by_planner_model={model:sum(r['planner_model']==model for r in records)
            for model in sorted({r['planner_model'] for r in records})},
        exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        exported=len(records),positives=sum(r['label'] for r in records),
        negatives=sum(1-r['label'] for r in records),excluded=excluded,
        features=list(FEATURES),split_unit='layout',outcome_based_resampling=False,
        source_manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest())
    report['by_transport']={mode:dict(samples=sum(r['transport_mode']==mode for r in records),
        positives=sum(r['label'] for r in records if r['transport_mode']==mode))
        for mode in sorted({r['transport_mode'] for r in records})}
    report['by_layout']={str(seed):dict(samples=sum(r['seed']==seed for r in records),
        positives=sum(r['label'] for r in records if r['seed']==seed))
        for seed in sorted({r['seed'] for r in records})}
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (out/'exporter.py').write_bytes(Path(__file__).read_bytes())
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=export(a.source,a.out)
    print(json.dumps({k:r[k] for k in ('exported','positives','negatives','purpose')}))
