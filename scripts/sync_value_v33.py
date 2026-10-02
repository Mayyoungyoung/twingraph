"""Synchronize completed V33 evidence without duplicating intermediate feature arrays."""
from pathlib import Path
import subprocess,tarfile
from scripts.relay_planner_v32 import remote,REMOTE


def sync():
    name='v33_robust_top3_20260927'
    remote(f'''import json,tarfile
from pathlib import Path
root=Path({REMOTE!r});source=root/'results'/{name!r}
if json.loads((source/'status.json').read_text())['stage']!='complete':raise ValueError('study incomplete')
with tarfile.open(root/{(name+'.tar.gz')!r},'w:gz',compresslevel=1) as archive:
 for base in (source,root/'results/v33_test_top3_20260927'):
  for p in base.rglob('*'):
   if not p.is_file() or p.suffix=='.tmp':continue
   if base==source and p.relative_to(base).parts[:2]==('training','inputs'):continue
   archive.add(p,arcname=str(p.relative_to(root)))
''',timeout=1800)
    target=Path('results')/(name+'.tar.gz')
    subprocess.run(['scp',f'901:{REMOTE}/{name}.tar.gz',str(target)],check=True,timeout=3600)
    root=Path('.').resolve()
    with tarfile.open(target) as archive:
        members=archive.getmembers()
        for m in members:
            if not m.isfile() or root not in (root/m.name).resolve().parents:raise ValueError('unsafe archive member')
        archive.extractall(root,members=members)
    return root/'results'/name


if __name__=='__main__':print(sync())
