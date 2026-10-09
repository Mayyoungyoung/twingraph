"""Consolidate the audited 901 TwinGraph run; archive records before deletion.

This administrative tool never connects to a robot. Default is a read-only plan.
Only the explicit, audited legacy checkout names below may be removed.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

HOME_ROOT = Path('/home/jia')
LATEST = HOME_ROOT / 'twingraph-v32-20260926'
ENV = HOME_ROOT / 'twingraph-v8-mj237'
DEST = HOME_ROOT / 'twingraph'
LEGACY_NAMES = (
    'twingraph-core-baseline-e27afa2', 'twingraph-core-r3-fixtures',
    'twingraph-skill-graph', 'twingraph-v11-system-20260921',
    'twingraph-v12-coverage-dev-r7', 'twingraph-v12-coverage-dev-r7b',
    'twingraph-v12-r7-pin-dev', 'twingraph-v12-r7-vision-dev',
    'twingraph-v12-ring-dev', 'twingraph-v12-system-20260921',
    'twingraph-v12-training-failure-audit', 'twingraph-v12-value-r7-tests',
    'twingraph-v12-white-frozen-r1', 'twingraph-v12-white-frozen-r2',
    'twingraph-v12-white-frozen-r3', 'twingraph-v12-wrist-dev-r4',
    'twingraph-v12-wrist-frozen-r5', 'twingraph-v12-wrist-frozen-r6',
    'twingraph-v13-mechanism-r1', 'twingraph-v13-value-regression-r1',
    'twingraph-v13-value-regression-r2', 'twingraph-v15-full-system-r1',
    'twingraph-v16-natural-coverage-r1', 'twingraph-v16-natural-coverage-r2',
    'twingraph-v16-natural-coverage-r3', 'twingraph-v16-natural-coverage-r4',
    'twingraph-v16-natural-coverage-r5', 'twingraph-v16-natural-coverage-r6',
    'twingraph-v16-natural-coverage-r7', 'twingraph-v17a-end-stop-place',
    'twingraph-v18-atomic-flow', 'twingraph-v19-cleaning',
    'twingraph-v20-full-flow', 'twingraph-v20-l0-value-r2',
    'twingraph-v20-next', 'twingraph-v3-venv', 'twingraph-v4-20260916',
    'twingraph-v5-20260916', 'twingraph-v5-formal-20260916',
    'twingraph-v5-publication-20260916', 'twingraph-v6-20260917',
    'twingraph-v8-credible-screening', 'twingraph-v8-py312',
    'twingraph-v8-results', 'twingraph-v8-runtime', 'twingraph-v8-test-venv',
    'RAL', 'RAL_archive',
)
EXCLUDE_TOP = {'scripts', 'simbench', 'tests', '.git', '.pytest_cache',
               'include', 'lib', 'lib64', 'bin', 'share', 'pyvenv.cfg'}


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def hash_stream(f):
    h = hashlib.sha256()
    for chunk in iter(lambda: f.read(1024 * 1024), b''):
        h.update(chunk)
    return h.hexdigest()


def safe_source(name):
    if name not in LEGACY_NAMES:
        raise ValueError('not an audited legacy checkout')
    p = HOME_ROOT / name
    if p.is_symlink() or p.resolve() != p or p.parent != HOME_ROOT:
        raise ValueError('unsafe checkout path: ' + str(p))
    if p in (LATEST, ENV, DEST, HOME_ROOT):
        raise ValueError('retained path cannot be deleted')
    return p


def no_running(checkouts):
    matches = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            cwd = os.readlink(str(proc / 'cwd'))
            cmd = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            if any(cwd == str(p) or cwd.startswith(str(p) + '/') or
                   str(p) + '/' in cmd for p in checkouts):
                matches.append((proc.name, cwd, cmd[:500]))
        except (OSError, PermissionError):
            pass
    if matches:
        raise RuntimeError('running processes use source directories: ' + repr(matches))


def plan():
    rows = []
    for name in LEGACY_NAMES:
        p = safe_source(name)
        if not p.exists():
            continue
        records = sorted(x.name for x in p.iterdir() if x.name not in EXCLUDE_TOP)
        # Never discard a Git repository not belonging to TwinGraph.
        if (p / '.git').exists():
            remote = subprocess.check_output(['git', '-C', str(p), 'remote', '-v'], text=True)
            if 'Mayyoungyoung/twingraph' not in remote:
                raise ValueError('unrelated repository: ' + str(p))
        rows.append(dict(path=str(p),records=records,git_history=(p / '.git').exists()))
    return dict(destination=str(DEST), retain_latest=str(LATEST), retain_environment=str(ENV),
                legacy=rows, robot_scripts_untouched='/home/jia/robot_panda/src/robot/scripts')


def archive_records(p, folder):
    archive = folder / (p.name + '.tar.gz')
    receipt_path = folder / (p.name + '.json')
    if archive.exists() or receipt_path.exists():
        raise RuntimeError('archive exists without completed removal; inspect before resuming: ' + str(p))
    history = None
    if (p / '.git').exists():
        # Partial clones may lazily fetch years of deleted datasets when asked
        # for --all bundles. Preserve the actual cached Git state and working
        # tree instead; missing historical objects remain on the Git remote.
        history = folder / (p.name + '.git-state.tar.gz')
        with tarfile.open(str(history),'w:gz',compresslevel=1) as stored:
            stored.add(str(p/'.git'),arcname='.git')
        # Even `git diff HEAD` can lazy-fetch missing blobs in a partial clone.
        # Cached .git plus the exact working files preserve its local changes
        # without requiring a network operation or reconstructing ancient data.
        worktree=folder/(p.name+'.working-source.tar.gz')
        with tarfile.open(str(worktree),'w:gz',compresslevel=1) as stored:
            for name in ('simbench','scripts','tests'):
                if (p/name).exists():
                    stored.add(str(p/name),arcname=name)
    selected = sorted(x for x in p.iterdir() if x.name not in EXCLUDE_TOP)
    files = {}
    with gzip.open(str(archive), 'wb', compresslevel=1) as compressed:
        with tarfile.open(fileobj=compressed, mode='w|', dereference=False) as tar:
            for item in selected:
                tar.add(str(item), arcname=item.relative_to(p).as_posix())
    # Verify every archived regular member against its original, before removal.
    with tarfile.open(str(archive), 'r:gz') as tar:
        for member in tar:
            if member.isfile():
                original = p / member.name
                if original.is_symlink() or not original.is_file():
                    raise ValueError('archive source file mismatch')
                with original.open('rb') as src:
                    expected = hash_stream(src)
                with tar.extractfile(member) as stored:
                    if hash_stream(stored) != expected:
                        raise ValueError('archive content mismatch: ' + member.name)
                files[member.name] = dict(size=member.size, sha256=expected)
            elif member.issym():
                if not (p/member.name).is_symlink() or os.readlink(str(p/member.name)) != member.linkname:
                    raise ValueError('symlink mismatch')
    with archive.open('rb') as f:
        archive_sha = hash_stream(f)
    receipt = dict(source=str(p),archive=str(archive),archive_sha256=archive_sha,
                   verified_files=files,git_state_archive=str(history) if history else None,
                   deleted=False,archive_bytes=archive.stat().st_size)
    write(receipt_path,receipt)
    return receipt,receipt_path


def apply(resume=False):
    if resume:
        info=json.loads((DEST/'maintenance/organization_plan.json').read_text())
    else:
        info = plan()
    if DEST.exists() and not resume:
        raise RuntimeError('destination already exists; do not overwrite an existing consolidation')
    no_running([LATEST,ENV] + [safe_source(Path(r['path']).name) for r in info['legacy']])
    DEST.mkdir(exist_ok=resume)
    admin = DEST / 'maintenance';admin.mkdir(exist_ok=resume)
    if not resume:write(admin/'organization_plan.json',info)
    records = DEST/'experiments';historical=records/'historical';historical.mkdir(parents=True,exist_ok=resume)
    # Move final records intact. Frozen code bytes and file names remain intact.
    code = DEST/'code'
    if not resume:
        LATEST.rename(code)
        (code/'results').rename(records/'final')
        (code/'results').symlink_to('../experiments/final',target_is_directory=True)
    videos=records/'videos';videos.mkdir(exist_ok=resume)
    if not resume:
        for name in ('demo_videos_v33_20261002','demo_videos_v33_smoke'):
            if (code/name).exists():
                (code/name).rename(videos/name)
    runtime = DEST/'runtime';runtime.mkdir(exist_ok=resume)
    newenv=runtime/'twin'
    if not resume:ENV.rename(newenv)
    # Entry point shebangs are relocatable after a mechanical path replacement.
    for entry in (newenv/'bin').iterdir():
        if entry.is_symlink() or not entry.is_file():
            continue
        data=entry.read_bytes()
        if str(ENV).encode() in data:
            entry.write_bytes(data.replace(str(ENV).encode(),str(newenv).encode()))
    subprocess.run([str(newenv/'bin/python'),'-c','import mujoco,torch,numpy; print(mujoco.__version__,torch.__version__,numpy.__version__)'],check=True)
    completed=[]
    for row in info['legacy']:
        p=safe_source(Path(row['path']).name)
        if not p.exists():
            receipt_path=historical/(p.name+'.json')
            receipt=json.loads(receipt_path.read_text())
            if not receipt['deleted']:raise ValueError('missing source without verified removal')
            completed.append(dict(name=p.name,archive_bytes=receipt['archive_bytes'],files=len(receipt['verified_files'])))
            continue
        receipt,receipt_path=archive_records(p,historical)
        no_running([p])
        # Resolve the exact allowlisted absolute path immediately before deletion.
        if safe_source(p.name)!=p or not p.is_dir():
            raise ValueError('deletion target changed')
        shutil.rmtree(str(p))
        receipt['deleted']=True;receipt['completed_at_epoch']=time.time();write(receipt_path,receipt)
        completed.append(dict(name=p.name,archive_bytes=receipt['archive_bytes'],files=len(receipt['verified_files'])))
        write(admin/'cleanup_progress.json',dict(completed=completed,planned=len(info['legacy'])))
        print(json.dumps(completed[-1]),flush=True)
    write(admin/'organization_complete.json',dict(code=str(code),results=str(records/'final'),runtime=str(newenv),deleted=completed))
    print('COMPLETE '+str(DEST),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--apply',action='store_true');a.add_argument('--resume',action='store_true');args=a.parse_args()
    if args.apply:
        apply(args.resume)
    else:
        print(json.dumps(plan(),ensure_ascii=False,indent=2))
