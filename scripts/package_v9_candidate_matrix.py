"""Package complete raw V9 rollouts as per-layout, checksummed archives."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-layouts", type=int, default=12)
    args = parser.parse_args()
    root, out = Path(args.matrix), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    directories = sorted(root.glob("seed_*"))
    if len(directories) != args.expected_layouts:
        raise ValueError(f"expected {args.expected_layouts} layout directories, found {len(directories)}")
    rows = []
    for directory in directories:
        summary = json.loads((directory / "summary.json").read_text())
        if len(summary["rows"]) != 36:
            raise ValueError(f"incomplete {directory}: {len(summary['rows'])} rows")
        archive = out / f"{directory.name}.tar.gz"
        with tarfile.open(archive, "w:gz", compresslevel=6) as tar:
            tar.add(directory, arcname=directory.name)
        rows.append(dict(seed=int(directory.name.split("_")[-1]), rows=36,
                         archive=archive.name, bytes=archive.stat().st_size,
                         sha256=sha256(archive)))
    (out / "raw_archive_manifest.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(dict(archives=len(rows), total_bytes=sum(x["bytes"] for x in rows))))


if __name__ == "__main__":
    main()
