"""Archive physical source/CAD/policies before collecting a frozen matrix."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tarfile
from simbench.value.provenance_v12 import ROOT,fingerprint,source_files


def main():
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    state=fingerprint()
    with tarfile.open(args.out/"physical_source.tar.gz","w:gz") as archive:
        for path in source_files(): archive.add(path,arcname=path.relative_to(ROOT).as_posix())
        for path in sorted((ROOT/"scripts").glob("*v12*.py")):
            archive.add(path,arcname=path.relative_to(ROOT).as_posix())
    state["runtime_versions"]={}
    for name in ("mujoco","numpy","torch","scipy","opencv-python","Pillow"):
        try:state["runtime_versions"][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:state["runtime_versions"][name]=None
    if args.checkpoint:
        state["value_checkpoint"]=dict(path=str(args.checkpoint),sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest())
    state["source_archive_sha256"]=hashlib.sha256((args.out/"physical_source.tar.gz").read_bytes()).hexdigest()
    (args.out/"manifest.json").write_text(json.dumps(state,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in state.items() if k!="files"},indent=2))


if __name__=="__main__":main()
