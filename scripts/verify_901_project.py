"""Read-only verification of relocated V33 evidence and archived cleanup receipts."""
import argparse
import hashlib
import json
from pathlib import Path


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def verify(root,full_archives=False):
    from scripts.collect_robust_value_v33 import RUNTIME
    from simbench.value.provenance_v12 import fingerprint
    code=root/'code';out=code/'results/v33_robust_top3_20260927'
    current=fingerprint()['sha256']
    assert current==RUNTIME, 'frozen physical source changed'
    assert read(out/'status.json')['stage']=='complete'
    comparison=read(out/'workflow/comparison.json')
    assert comparison['complete'] and comparison['completed_layouts']==50
    frozen=read(out/'training/selection_frozen.json')
    for name,value in frozen['checkpoint_sha256'].items():
        assert sha(out/'training'/name)==value,'frozen model changed'
    audit=read(out/'workflow_audit.json')
    assert audit['passed'] and audit['policy_workflows']==450
    for name,value in audit['result_sha256'].items():
        assert sha(out/name)==value,'result changed: '+name
    receipts=[]
    for path in sorted((root/'experiments/historical').glob('*.json')):
        data=read(path)
        if 'archive_sha256' not in data:
            continue
        assert data['deleted'] and not Path(data['source']).exists()
        archive=Path(data['archive'])
        assert archive.is_file() and archive.stat().st_size==data['archive_bytes']
        if full_archives:
            assert sha(archive)==data['archive_sha256'],'archive changed'
        receipts.append(path.name)
    complete=read(root/'maintenance/organization_complete.json')
    assert len(receipts)==len(complete['deleted'])
    return dict(passed=True,runtime_sha256=current,checkpoint_sha256=frozen['checkpoint_sha256'],
                verified_fresh_trial_files=len(audit['result_sha256']),
                preserved_layouts=50,preserved_policy_workflows=450,
                archived_and_deleted_directories=len(receipts),full_archive_hash_verification=full_archives)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('/home/jia/twingraph'))
    parser.add_argument('--full-archives',action='store_true');a=parser.parse_args()
    print(json.dumps(verify(a.root,a.full_archives),indent=2))
