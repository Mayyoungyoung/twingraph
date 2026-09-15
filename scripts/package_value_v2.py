"""Create a versioned, checksummed release from completed physical records."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/value_v2');p.add_argument('--out',default='results/value_v2/release');p.add_argument('--code-commit',required=True);a=p.parse_args()
    root=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    data=root/'data';files=[];groups=[];raw_trials=0;used_trials=0;wall=0.;steps=0
    for f in sorted(data.glob('group_*/complete.json')):
        d=f.parent;complete=json.loads(f.read_text());inputs=json.loads((d/'inputs.json').read_text());trials=json.loads((d/'outcomes.json').read_text())['trials']
        groups.append(dict(directory=d.name,group_id=inputs['group_id'],split_group=inputs['split_group'],
                      family=complete['family'],seed=complete['seed'],checkpoint=complete['checkpoint'],split=complete['split'],
                      candidates=complete['candidates'],trials=len(trials),successes=sum(t['success'] for t in trials),
                      plan_lengths=complete['plan_lengths'],source_sha256=inputs['source_sha256'],input_sha256=complete['input_sha256'],
                      wall_seconds=complete['wall_seconds'],physics_steps=sum(t['physics_steps'] for t in trials)))
        used_trials+=len(trials);wall+=complete['wall_seconds'];steps+=sum(t['physics_steps'] for t in trials)
        files.extend(x for x in d.iterdir() if x.is_file())
    files.extend(f for f in data.iterdir() if f.is_file())
    with tarfile.open(out/'two_family_v2.tar.gz','w:gz') as archive:
        for f in sorted(files):archive.add(f,arcname=f.relative_to(data))
    # Logs include early failures and interrupted development, never promoted to labels.
    for f in root.glob('**/outcomes.json'):
        if 'release' in f.parts:continue
        try:raw_trials+=len(json.loads(f.read_text())['trials'])
        except (json.JSONDecodeError,KeyError):pass
    manifest=dict(schema='twingraph.dataset.manifest.v2',code_commit=a.code_commit,created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                  groups=groups,decision_groups=len(groups),configurations=len({(g['family'],g['seed']) for g in groups}),
                  used_label_rollouts=used_trials,collection_worker_seconds=wall,collection_physics_steps=steps,
                  raw_outcome_record_count_including_archived_copies=raw_trials,
                  files={str(f.relative_to(data)):sha(f) for f in sorted(files)},archive_sha256=sha(out/'two_family_v2.tar.gz'),
                  source_compatibility=json.loads((data/'source_compatibility.json').read_text()),
                  split_unit='family + seed, with all checkpoints and wide/compact variants together',
                  reference='two paired independent reference-domain perturbations per candidate; coarse empirical rates')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    code=[f for f in Path('simbench').rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.suffix not in {'.pyc','.log'}]
    code.extend(f for f in Path('scripts').rglob('*') if f.suffix in {'.py','.sh'} and '__pycache__' not in f.parts)
    code.extend(Path(f) for f in ('docs/value-v2-protocol.md','RELATED_WORK_DELTA.md'))
    with tarfile.open(out/'two_family_v2_source.tar.gz','w:gz') as archive:
        for f in sorted(code):archive.add(f,arcname=f)
    (out/'source_hashes.json').write_text(json.dumps({str(f):sha(f) for f in sorted(code)},indent=2))
    execution_dirs=[f for f in root.iterdir() if f.is_dir() and (f.name.startswith('online') or f.name.startswith('new_configuration'))]
    with tarfile.open(out/'online_execution_v2.tar.gz','w:gz') as archive:
        for directory in sorted(execution_dirs):archive.add(directory,arcname=directory.name)
    development_dirs=[f for f in root.iterdir() if f.is_dir() and (f.name in {'pilot','pilot2','checks','contact_probe','regression','data_wide_v2a','timing_smoke','development_online'} or f.name.startswith('spacing_probe'))]
    with tarfile.open(out/'development_evidence_v2.tar.gz','w:gz') as archive:
        for directory in sorted(development_dirs):archive.add(directory,arcname=directory.name)
    if (root/'wide_v2a_source.tar.gz').exists():shutil.copy2(root/'wide_v2a_source.tar.gz',out/'wide_v2a_source.tar.gz')
    modeldir=out/'models';modeldir.mkdir(exist_ok=True);model_info={}
    for folder in sorted((root/'models').iterdir()):
        if not folder.is_dir() or not (folder/'best.pt').exists():continue
        dest=modeldir/folder.name;dest.mkdir(exist_ok=True)
        for name in ('best.pt','summary.json','splits.json','history.json'):
            shutil.copy2(folder/name,dest/name)
        model_info[folder.name]=dict(checkpoint_sha256=sha(dest/'best.pt'),summary=json.loads((folder/'summary.json').read_text()))
    (out/'model_manifest.json').write_text(json.dumps(model_info,indent=2))
    shutil.copy2(root/'models/validation_recommendation.json',out/'validation_recommendation.json')
    selected=json.loads((out/'validation_recommendation.json').read_text())
    shutil.copy2(selected['checkpoint'],out/'best_value_v2.pt')
    import torch,mujoco,torchvision,numpy
    env=dict(python=platform.python_version(),platform=platform.platform(),torch=torch.__version__,torchvision=torchvision.__version__,
             mujoco=mujoco.__version__,numpy=numpy.__version__,gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,
             cpu_quota=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),code_commit=a.code_commit,
             device_info=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip())
    (out/'environment.json').write_text(json.dumps(env,indent=2))
    (out/'release_checksums.json').write_text(json.dumps({str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file() and f.name!='release_checksums.json'},indent=2))
    print(json.dumps(dict(groups=len(groups),trials=used_trials,archive_mb=(out/'two_family_v2.tar.gz').stat().st_size/1e6,output=str(out))))


if __name__=='__main__':main()
