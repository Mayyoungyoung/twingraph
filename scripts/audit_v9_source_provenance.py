"""Compare the archived V9 matrix source manifest to the requested base commit."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--commit", default="60f5689")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with tarfile.open(args.archive) as tar:
        name = next(m.name for m in tar.getmembers() if m.name.endswith("/summary.json"))
        historical = json.load(tar.extractfile(name))["source_sha256"]
    changed = {}
    missing = []
    for path, recorded_hash in historical.items():
        try:
            blob = subprocess.check_output(["git", "show", f"{args.commit}:{path}"], stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            missing.append(path)
            continue
        commit_hash = hashlib.sha256(blob).hexdigest()
        if commit_hash != recorded_hash:
            changed[path] = dict(archive_sha256=recorded_hash, base_commit_sha256=commit_hash)
    result = dict(archive=str(args.archive), base_commit=args.commit,
                  archived_source_file_count=len(historical), mismatched_files=changed,
                  absent_at_base_commit=missing)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(dict(count=len(changed), changed=list(changed), missing=missing), indent=2))


if __name__ == "__main__":
    main()
