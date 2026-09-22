"""Physical source, CAD and policy fingerprint for reproducible V12 matrices."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def source_files():
    files=list((ROOT/"simbench").rglob("*.py"))
    # Nested robot models and collision meshes affect feasibility too. In
    # particular, fingertip pads cannot stand in for a modified Panda shell.
    files+=list((ROOT/"simbench").rglob("*.xml"))
    files+=list((ROOT/"simbench"/"configs").glob("*.json"))
    files+=list((ROOT/"simbench"/"assembly"/"checkpoints").glob("*.npz"))
    files+=list((ROOT/"simbench"/"assembly"/"checkpoints").glob("*.pt"))
    for suffix in ("*.stl", "*.obj", "*.msh", "*.png", "*.jpg", "*.jpeg"):
        files+=list((ROOT/"simbench"/"assets").rglob(suffix))
    return sorted(set(files))


def fingerprint():
    files={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files()}
    digest=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return dict(sha256=digest,files=files)
