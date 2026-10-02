"""Audit frozen observations, outcome labels, layout splits and value tensors."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.collect_sliding_assembly_v23 import REQUIRED
from simbench.assembly.library import DEFAULT_CAPABILITIES
from simbench.value.plan import PlanIR, digest
from simbench.value.skill_graph import compile_graph
from simbench.value.generic_graph_value_v15 import encode_graph, FEATURES


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def audit(root, exported):
    manifest = read(root/'manifest.json')
    labels = [json.loads(line) for line in (exported/'labels.jsonl').read_text().splitlines()]
    errors = []; warnings = []; failures = Counter(); shapes = Counter()
    feature_groups = defaultdict(list); split_rows = defaultdict(list)
    ids = set(); group_splits = defaultdict(set); raw_count = 0
    for job in manifest['jobs']:
        layout = root/f"seed_{job['seed']}"
        if not (layout/'request.json').exists():
            continue
        request = read(layout/'request.json')
        entries = request['candidates']
        if len({e['plan_sha256'] for e in entries}) != len(entries):
            errors.append(f"duplicate frozen plan: {job['seed']}")
        if request.get('outcome_labels_read') or request.get('layout_selected_from_prior_outcomes'):
            errors.append(f"outcome based candidate selection: {job['seed']}")
        group_splits[request['split_group']].add(job['split'])
        raw_count += sum((layout/'candidates'/e['name']/'result.json').exists() for e in entries)
    for row in labels:
        sid = row['id']
        if sid in ids: errors.append(f'duplicate sample ID: {sid}')
        ids.add(sid); split_rows[row['split']].append(row)
        layout = root/f"seed_{row['seed']}"
        request = read(layout/'request.json')
        entry = next(e for e in request['candidates'] if sid == f"seed_{row['seed']}_{e['name']}")
        directory = layout/'candidates'/entry['name']
        result = read(directory/'result.json'); raw = read(directory/'input_graph.json')
        training = read(directory/'training_sample.json')
        def check(condition, message):
            if not condition: errors.append(f'{sid}: {message}')
        check(result['valid'] and not result.get('timeout') and not result.get('resource_censored'), 'invalid/censored label')
        check(training.get('software_exception') is None, 'software exception')
        check(row['label'] in (0, 1) and row['label'] == int(result['success']) == training['actual_execution_label'], 'label mismatch')
        check(bool(result['success']) == all(result['final_functional_predicates'].get(k) is True for k in REQUIRED), 'functional acceptance mismatch')
        check(digest(raw) == result['input_graph_sha256'] == row['raw_input_sha256'], 'raw input hash mismatch')
        check(digest(raw['complete_candidate_plan_ir']) == entry['plan_sha256'] == result['plan_sha256'] == row['plan_sha256'], 'controller plan mismatch')
        check(digest(raw['assembly_plan_ir']) == entry['assembly_plan_sha256'], 'atomic plan mismatch')
        check(raw['decision_observation'] == request['decision_observation'], 'observation changed after freeze')
        check(result['runtime_sha256'] == request['runtime_sha256'], 'runtime mismatch')
        check(row['split'] == next(j['split'] for j in manifest['jobs'] if j['seed'] == row['seed']), 'split mismatch')
        if row['label']:
            for pin in ('pin_left', 'pin_right'):
                acceptance = result['base_hole_acceptance'][pin]['inserted_after_release']
                check(acceptance['success'] and acceptance['released'], f'{pin} not released/accepted')
                for receiver in ('end_stop', 'guide_base'):
                    a = acceptance['receivers'].get(receiver)
                    check(bool(a and a['success']), f'{pin}/{receiver} acceptance failed')
        else:
            failures[result.get('failure_type') or 'unclassified'] += 1
        observation = deepcopy(raw['decision_observation'])
        observation.setdefault('robot', {})
        observation['goals'] = [dict(predicate='assembly_through_handle', both_pins_through_base=True)]
        for part, capabilities in DEFAULT_CAPABILITIES.items():
            if part in observation['objects']: observation['objects'][part]['capabilities'] = list(capabilities)
        graph = compile_graph(observation, PlanIR.from_dict(raw['assembly_plan_ir']))
        check(graph == read(exported/'inputs'/sid/'graph.json'), 'exported graph differs from pre-execution reconstruction')
        encoded = encode_graph({'assembly': graph})
        with np.load(exported/row['input']) as saved:
            for key, value in encoded.items():
                check(np.isfinite(saved[key]).all() and np.array_equal(value, saved[key]), f'{key}: tensor mismatch/nonfinite')
        shapes[str(encoded['x'].shape)] += 1
        signature = hashlib.sha256(b''.join(encoded[k].tobytes() for k in sorted(encoded))).hexdigest()
        feature_groups[signature].append(dict(id=sid, label=row['label'], split=row['split']))
    for group, splits in group_splits.items():
        if len(splits) != 1: errors.append(f'layout leakage: {group}: {splits}')
    collisions = [v for v in feature_groups.values() if len(v)>1]
    cross_split = [v for v in collisions if len({r['split'] for r in v})>1]
    conflicting = [v for v in collisions if len({r['label'] for r in v})>1]
    if cross_split: errors.append('identical feature tensors across splits')
    if conflicting: warnings.append('identical input tensors have conflicting labels')
    stats = {s:dict(samples=len(rows), positives=sum(r['label'] for r in rows),
        negatives=sum(1-r['label'] for r in rows), layouts=len({r['seed'] for r in rows})) for s,rows in split_rows.items()}
    complete=len(labels)==manifest['planned_samples']
    if complete:
        for split in ('train','validation','test'):
            expected=sum(8 for j in manifest['jobs'] if j['split']==split)
            if stats.get(split,{}).get('samples',0)!=expected: errors.append(f'final split count mismatch: {split}')
    return dict(passed=not errors, samples=len(labels), raw_results=raw_count,
        planned_samples=manifest['planned_samples'],dataset_complete=complete and not errors,
        positives=sum(r['label'] for r in labels), negatives=sum(1-r['label'] for r in labels),
        layouts=len({r['seed'] for r in labels}), by_split=stats, failure_types=dict(failures),
        tensor_shapes=dict(shapes), feature_dimension=len(FEATURES), unique_feature_tensors=len(feature_groups),
        samples_by_runtime=dict(Counter(read(root/f"seed_{r['seed']}"/'request.json')['runtime_sha256'] for r in labels)),
        duplicate_tensor_groups=collisions, conflicting_label_groups=conflicting,
        errors=errors, warnings=warnings,
        scope='simulation binary full-plan success value, state-pose observation, zero added pose noise',
        limitation='No claim of real-robot, RGB-D-noise, or new-task generalization; few validation/test layouts.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--exported',type=Path,required=True); parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); report=audit(args.source,args.exported)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report)); raise SystemExit(0 if report['passed'] else 1)
