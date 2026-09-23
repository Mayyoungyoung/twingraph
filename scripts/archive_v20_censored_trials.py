#!/usr/bin/env python3
"""Preserve resource-censored complete-task attempts before an exact retry.

Only invalid trials explicitly marked resource-censored can be archived. All
physical successes and failures remain untouched; this never changes labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from simbench.value.plan import digest


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contained(path, root):
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"path escapes frozen collection directory: {path}")


def archive(root):
    root = Path(root).resolve()
    request = read(root / "request.json")
    pool = {item["name"]: item for item in request["pool"]}
    existing = root / "censored_attempts"
    if not root.name == "collect" or not pool:
        raise ValueError("expected a complete-task seed collection directory")
    candidates = []
    for name, proposal in pool.items():
        directory = root / "candidates" / name
        result_path = directory / "result.json"
        if not result_path.exists():
            continue
        result = read(result_path)
        if result.get("valid"):
            continue
        graph_path = directory / "input_graph.json"
        if (result.get("resource_censored") is not True
                or result.get("timeout") is not True
                or result.get("full_task_label") is not True
                or result.get("evaluation_scope") != "complete_functional_task"
                or result.get("proposal") != proposal
                or result.get("runtime_sha256") != request["runtime_sha256"]
                or not graph_path.is_file()
                or digest(read(graph_path)) != request["graph_sha256"][name]
                or result.get("input_graph_sha256") != request["graph_sha256"][name]):
            raise ValueError(f"invalid non-censored or mismatched result needs manual audit: {name}")
        attempts = sorted((existing / name).glob("attempt_*")) if (existing / name).exists() else []
        number = len(attempts) + 1
        destination = existing / name / f"attempt_{number:03d}"
        if destination.exists():
            raise ValueError(f"archive destination already exists: {destination}")
        files = [directory / filename for filename in
                 ("result.json", "input_graph.json", "console.log", "orchestration_error.json")]
        files = [path for path in files if path.exists()]
        scene = directory / "scene"
        if scene.exists():
            files.append(scene)
        for path in [directory, destination, *files]:
            contained(path, root)
        candidates.append((name, result, files, destination))
    if not candidates:
        raise ValueError("no resource-censored attempts to archive")
    records = []
    for name, result, files, destination in candidates:
        destination.mkdir(parents=True)
        saved = []
        for path in files:
            item = dict(name=path.name, sha256=sha(path) if path.is_file() else None)
            shutil.move(str(path), str(destination / path.name))
            saved.append(item)
        records.append(dict(name=name, archive=str(destination),
                            reason=result.get("invalid_reason"),
                            wall_seconds=result.get("total_wall_seconds"), files=saved))
    metadata_dir = existing / f"collection_before_retry_{len(list(existing.glob('collection_before_retry_*'))) + 1:03d}"
    contained(metadata_dir, root)
    metadata_dir.mkdir(parents=True)
    metadata = []
    for filename in ("progress.json", "collection_summary.json"):
        path = root / filename
        if path.is_file():
            contained(path, root)
            shutil.copy2(path, metadata_dir / filename)
            metadata.append(dict(name=filename, sha256=sha(path)))
    manifest = dict(schema="twingraph.v20_resource_censored_retry.v1",
                    seed=request["seed"], runtime_sha256=request["runtime_sha256"],
                    request_sha256=sha(root / "request.json"),
                    policy="archive only invalid resource-censored attempts; retry same frozen candidate and runtime",
                    archived=records, previous_collection_metadata=metadata)
    target = existing / f"retry_manifest_{len(list(existing.glob('retry_manifest_*.json'))) + 1:03d}.json"
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return dict(seed=request["seed"], archived=[r["name"] for r in records], manifest=str(target))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    args = parser.parse_args()
    for root in args.root:
        print(json.dumps(archive(root), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
