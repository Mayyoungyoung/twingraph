"""Fetch the completed value study, checkpoints and raw fresh workflow records."""
from pathlib import Path
import subprocess
import tarfile
from scripts.relay_planner_v32 import remote,REMOTE


def sync():
    name='v32_value_training_20260927'
    remote(f'''import json,tarfile
from pathlib import Path
root=Path({REMOTE!r}); source=root/'results'/{name!r}
status=json.loads((source/'status.json').read_text())
if status['stage']!='complete':raise ValueError('training/workflow study not complete')
with tarfile.open(root/{(name+'.tar.gz')!r},'w:gz',compresslevel=1) as archive:
 for p in source.rglob('*'):
  if not p.is_file() or p.suffix=='.tmp':continue
  rel=p.relative_to(source)
  if len(rel.parts)>=2 and rel.parts[:2]==('data','inputs'):continue
  archive.add(p,arcname=str(p.relative_to(root)))
''',timeout=900)
    target=Path('results')/(name+'.tar.gz')
    subprocess.run(['scp',f'901:{REMOTE}/{name}.tar.gz',str(target)],check=True,capture_output=True,timeout=1800)
    base=Path('.').resolve()
    with tarfile.open(target) as archive:
        members=archive.getmembers()
        for member in members:
            path=(base/member.name).resolve()
            if base not in path.parents or not member.isfile():raise ValueError('unsafe archive member')
        archive.extractall(base,members=members)
    return base/'results'/name


if __name__=='__main__':print(sync())
