"""Archive v5 source, raw data, all models and actual system/development evidence.

This command never trains, selects a model, evaluates a policy, or publishes.
Every retained run file is accounted for. Archives are split below 95 MiB and
their extracted byte streams are checked against the inventory. Missing exact
formal source bytes fail closed; historical development gaps are explicit.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.package_value_v4 import (LIMIT, SourceResolver, archive, development,
    digest, environment, files_under, read, sha, source_mapping, write)


def safe_archive_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError(f"unsafe archive path: {name}")
    return path.as_posix()


class ExactSources(SourceResolver):
    """Extend v4 resolution to recursive retained snapshots and scripts."""
    def __init__(self, repo, run, commit, extra=()):
        super().__init__(repo, run, commit)
        self.snapshot_paths = sorted(set(self.snapshot_paths) | {
            path for path in run.rglob("*.tar*") if "source" in path.name.lower()
        } | {Path(path).resolve() for path in extra})
        for path in self.snapshot_paths:
            with tarfile.open(path, "r:*") as stream:
                for member in stream.getmembers():
                    if not member.isfile():
                        continue
                    name = safe_archive_name(member.name)
                    parts = PurePosixPath(name).parts
                    indices = [i for i, p in enumerate(parts) if p in {"simbench", "scripts", "tests"}]
                    if not indices:
                        continue
                    relative = PurePosixPath(*parts[min(indices):]).as_posix()
                    payload = stream.extractfile(member).read()
                    self.snapshot_bytes[(relative, hashlib.sha256(payload).hexdigest())] = payload


def collection_record(directory, run):
    request = read(directory / "request.json") if (directory / "request.json").exists() else None
    inp = read(directory / "inputs.json") if (directory / "inputs.json").exists() else None
    complete = read(directory / "complete.json") if (directory / "complete.json").exists() else None
    failure = read(directory / "failure.json") if (directory / "failure.json").exists() else None
    censored = read(directory / "censored.json") if (directory / "censored.json").exists() else None
    outcomes = read(directory / "outcomes.json") if (directory / "outcomes.json").exists() else {}
    if request is None and failure:
        request = failure.get("request")
    trials = outcomes.get("trials", [])
    if inp:
        if inp.get("schema") != "twingraph.group.v5":
            raise ValueError(f"unsupported collection input: {directory}")
        ih = digest(inp)
        if outcomes and outcomes.get("input_sha256") != ih:
            raise ValueError(f"outcomes/input hash mismatch: {directory}")
        if complete and complete.get("input_sha256") != ih:
            raise ValueError(f"complete/input hash mismatch: {directory}")
        if complete and request and complete.get("request_sha256") != digest(request):
            raise ValueError(f"complete/request hash mismatch: {directory}")
        ids = [candidate["id"] for candidate in inp["candidates"]]
        if len(ids) != len(set(ids)) or any(row["candidate_id"] not in ids for row in trials):
            raise ValueError(f"invalid candidate identities: {directory}")
        identities = [(row["candidate_id"], digest(row["trial"])) for row in trials]
        if len(identities) != len(set(identities)):
            raise ValueError(f"duplicate physical trial: {directory}")
        if complete and (complete["candidates"] != len(ids) or complete["trials"] != len(trials)
                or complete["all_successes"] != sum(row.get("success") is True for row in trials)):
            raise ValueError(f"completion counts disagree with raw trials: {directory}")
    split = inp.get("declared_split") if inp else (request or {}).get("split")
    return dict(path=directory.relative_to(run).as_posix(),
        status="completed" if complete else "failed" if failure else "censored" if censored else "incomplete",
        development=split == "development" or development(directory.relative_to(run)),
        split=split, seed=(request or {}).get("seed"), request=request,
        candidate_count=len(inp["candidates"]) if inp else 0, trials=len(trials),
        successes=sum(row.get("success") is True for row in trials),
        timeouts=sum(row.get("timeout") is True for row in trials),
        invalid_trials=sum(row.get("valid") is not True for row in trials),
        inputs_sha256=digest(inp) if inp else None,
        source_sha256=(inp or request or {}).get("source_sha256"),
        source_manifest_present=(directory/"source.json").exists(),
        failure_record=bool(failure), censored_record=bool(censored))


def system_record(path, run):
    row = read(path)
    if row.get("schema") != "twingraph.system_run.v5":
        raise ValueError("not a v5 system result")
    inp = path.parent / "inputs.json"
    if inp.exists() and row.get("inputs_sha256") != digest(read(inp)):
        raise ValueError(f"system/input hash mismatch: {path}")
    validation = row.get("validation", [])
    target = row.get("target_trials", [])
    executed = [item for item in target if item.get("status") == "executed"]
    if row["simulation_calls"] != dict(twin_validation=len(validation), target_execution=len(executed)):
        raise ValueError(f"system simulation counts disagree: {path}")
    if row["target_successes"] != sum(item.get("success") is True for item in target):
        raise ValueError(f"system target successes disagree: {path}")
    if row["target_successes"] > row["requested_target_trials"]:
        raise ValueError(f"target successes exceed requested denominator: {path}")
    for item in executed:
        checks = item.get("isolation", {})
        if not all(checks.get(key) is True for key in ("distinct_session", "distinct_context", "distinct_model", "distinct_data")) or checks.get("snapshot_transfer") is not False:
            raise ValueError(f"missing independent-target isolation evidence: {path}")
    return dict(path=path.relative_to(run).as_posix(), method=row["method"],
        development=development(path.relative_to(run)), status=row["status"],
        config_id=row["config_id"], inputs_sha256=row.get("inputs_sha256"),
        selected_candidate_id=row.get("selected_candidate_id"),
        twin_validation_trials=len(validation), target_execution_trials=len(executed),
        target_successes=row["target_successes"], requested_target_trials=row["requested_target_trials"],
        implementation_sha256=row.get("implementation_sha256"), seconds=row["seconds"])


def chunked_archives(stage, label, entries, generated=None, chunk_bytes=64*1024*1024):
    """Partition by raw bytes so incompressible traces remain below upload limit."""
    items = [(name, source, None) for name, source in entries.items()]
    items.extend((name, None, payload.encode() if isinstance(payload, str) else payload)
                 for name, payload in (generated or {}).items())
    groups, current, size = [], [], 0
    for item in sorted(items):
        name, source, payload = item
        safe_archive_name(name)
        length = Path(source).stat().st_size if source is not None else len(payload)
        if length >= LIMIT - 1024*1024:
            raise ValueError(f"single member too large for bounded archive; retain and split explicitly: {name}")
        if current and size + length + 2048 > chunk_bytes:
            groups.append(current)
            current, size = [], 0
        current.append(item)
        size += length + 2048
    if current:
        groups.append(current)
    result = {}
    for index, group in enumerate(groups):
        relative = f"archives/{label}_{index:03d}.tar.gz"
        result[relative] = archive(stage / relative,
            {name: source for name, source, _ in group if source is not None},
            {name: payload for name, source, payload in group if source is None})
    return result


def package(run, output, commit, freeze=None, snapshots=(), repo=None):
    repo = Path(repo or Path(__file__).resolve().parents[1]).resolve()
    run, output = Path(run).resolve(), Path(output).resolve()
    if not run.is_dir() or output in {run, repo} or run.is_relative_to(output):
        raise ValueError("run must exist and output must be a distinct directory")
    if output.exists() and any(output.iterdir()):
        raise ValueError("refuse to overwrite a nonempty package directory")
    retained = [p for p in files_under(run) if not p.is_relative_to(output)]
    if not retained:
        raise ValueError("run contains no retained evidence")
    # A directly selected development/pilot root carries its scope into every
    # child, including names such as system_preflight that lack a dev prefix.
    # Inspect only the selected root's name, not unrelated ancestor directories.
    development_root = development(Path(run.name))
    inventory = {p.relative_to(run).as_posix(): dict(sha256=sha(p), bytes=p.stat().st_size) for p in retained}
    groups = []
    directories = {p.parent for p in retained if p.name in {"request.json", "failure.json"} and p.parent.name.startswith("group_")}
    directories |= {p.parent for p in retained if p.name == "inputs.json" and read(p).get("schema") == "twingraph.group.v5"}
    for directory in sorted(directories):
        group = collection_record(directory, run)
        group["development"] = development_root or group["development"]
        if group["status"] == "incomplete" and not group["development"]:
            raise ValueError(f"formal group is incomplete; preserve a failure record before packaging: {directory}")
        groups.append(group)
    systems = [system_record(p, run) for p in retained if p.name == "result.json"
               and read(p).get("schema") == "twingraph.system_run.v5"]
    for system in systems:
        system["development"] = development_root or system["development"]
    development_roots = {run/g["path"] for g in groups if g["development"]}
    is_dev = lambda p: development_root or development(p.relative_to(run)) or any(p.is_relative_to(root) for root in development_roots)
    resolver = ExactSources(repo, run, commit, snapshots)
    source_refs = [p for p in retained if p.name == "source.json"]
    overlays, source_rows, mapping_by_dir = {}, [], {}
    for path in source_refs:
        mapping = source_mapping(read(path))
        key = digest(mapping)
        mapping_by_dir[path.parent] = key
        missing, resolved = [], []
        for relative, checksum in sorted(mapping.items()):
            safe_archive_name(relative)
            payload, origin = resolver.resolve(relative, checksum)
            if payload is None:
                missing.append(dict(path=relative, sha256=checksum))
            else:
                overlays[f"snapshots/{key}/{relative}"] = payload
                resolved.append(dict(path=relative, sha256=checksum, origin=origin))
        if missing and not is_dev(path):
            raise ValueError(f"exact formal source bytes unavailable: {path}: {missing}")
        overlays[f"snapshots/{key}/source.json"] = json.dumps(mapping, indent=2).encode()
        source_rows.append(dict(reference=path.relative_to(run).as_posix(), sha256=key,
            development=is_dev(path), exact_bytes_complete=not missing, missing=missing, resolved=resolved))
    for group in groups:
        if group["inputs_sha256"] and not group["development"] and mapping_by_dir.get(run/group["path"]) != group["source_sha256"]:
            raise ValueError(f"formal collection source binding missing/mismatched: {group['path']}")
        source_matches = [row for row in source_rows if row["sha256"] == group["source_sha256"] and row["exact_bytes_complete"]]
        group["exact_source_available"] = bool(source_matches)
        group["matching_source_references"] = [row["reference"] for row in source_matches]
        if not group["development"] and not source_matches:
            raise ValueError(f"formal collection request has no retained exact source manifest: {group['path']}")
    for system in systems:
        if not system["development"] and mapping_by_dir.get((run/system["path"]).parent) != system["implementation_sha256"]:
            raise ValueError(f"formal system source binding missing/mismatched: {system['path']}")
    models = []
    for path in retained:
        if path.name != "best.pt":
            continue
        summary_path = path.parent / "summary.json"
        summary = read(summary_path) if summary_path.exists() else {}
        if summary.get("checkpoint_sha256", sha(path)) != sha(path):
            raise ValueError(f"model summary/weight hash mismatch: {path}")
        if not is_dev(path):
            for name in ("summary.json", "source.json", "split.json", "history.json", "input_schema.json"):
                if not (path.parent/name).is_file():
                    raise ValueError(f"formal checkpoint missing provenance: {path.parent/name}")
            if summary.get("source_sha256") != mapping_by_dir.get(path.parent):
                raise ValueError(f"model/source binding mismatch: {path}")
            if summary.get("split_sha256") != digest(read(path.parent/"split.json")):
                raise ValueError(f"model/split binding mismatch: {path}")
        models.append(dict(path=path.relative_to(run).as_posix(), sha256=sha(path),
            development=is_dev(path), kind=summary.get("kind"), seed=summary.get("seed"),
            source_sha256=summary.get("source_sha256"), split_sha256=summary.get("split_sha256")))
    frozen = None
    if freeze:
        freeze = Path(freeze).resolve()
        frozen = dict(path=str(freeze), sha256=sha(freeze), value=read(freeze),
            interpretation="preserved caller-designated freeze; packager performs no selection")
        declared = []
        def checkpoint_refs(value, location="freeze"):
            if isinstance(value, list):
                for index, item in enumerate(value):
                    checkpoint_refs(item, f"{location}/{index}")
            elif isinstance(value, dict):
                if "checkpoint_sha256" in value:
                    declared.append(dict(location=location, sha256=value["checkpoint_sha256"]))
                elif str(value.get("path", "")).endswith(".pt") and "sha256" in value:
                    declared.append(dict(location=location, sha256=value["sha256"]))
                for key, item in value.items():
                    checkpoint_refs(item, f"{location}/{key}")
        checkpoint_refs(frozen["value"])
        available = {model["sha256"] for model in models}
        if any(ref["sha256"] not in available for ref in declared):
            raise ValueError("caller-designated freeze references a missing or changed checkpoint")
        frozen["verified_checkpoint_references"] = declared
    categories = {key: {} for key in ("development", "data", "models", "system", "evidence")}
    data_roots = {run/g["path"] for g in groups}
    model_roots = {(run/m["path"]).parent for m in models}
    system_roots = {(run/s["path"]).parent for s in systems}
    for path in retained:
        key = "development" if is_dev(path) else "data" if any(path.is_relative_to(root) for root in data_roots) else "models" if any(path.is_relative_to(root) for root in model_roots) else "system" if any(path.is_relative_to(root) for root in system_roots) else "evidence"
        categories[key][path.relative_to(run).as_posix()] = path
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.building-", dir=output.parent) as temporary:
        stage = Path(temporary)/"package"
        stage.mkdir()
        archives = {}
        for label, entries in categories.items():
            archives.update(chunked_archives(stage, label, entries))
        base = {p.relative_to(repo).as_posix(): p for folder in ("simbench", "scripts", "tests")
                for p in files_under(repo/folder, exclude_results=True)}
        base.update({p.name:p for pattern in ("requirements*.txt", "pyproject.toml", "README.md") for p in repo.glob(pattern)})
        # Keep the CLI's directly referenced configuration and instructions at
        # their repository paths, without duplicating the public evidence tree.
        reproduction_files = ("experiments/value_v5/protocol.json", "experiments/value_v5/planner_record.json",
            "docs/value-v5-design.md", "docs/value-v5-protocol.md", "docs/value-v5-running.md")
        base.update({name:repo/name for name in reproduction_files if (repo/name).is_file()})
        overlays["source_index.json"] = json.dumps(dict(recorded_code_commit=commit, manifests=source_rows,
            base_reproduction_files=[name for name in reproduction_files if name in base],
            reproduction="Extract base tree, then overlay snapshots/<source_sha256>/ files and verify each exact source.json before reproducing.",
            development_limits="Missing historical development source bytes are listed; no exact reproduction claim for those records."), indent=2).encode()
        archives.update(chunked_archives(stage, "source", base, overlays))
        extra = {f"retained_source_snapshots/{i:03d}_{p.name}": p for i,p in enumerate(resolver.snapshot_paths) if not p.is_relative_to(run)}
        archives.update(chunked_archives(stage, "external_source_snapshots", extra))
        if frozen:
            write(stage/"designated_freeze.json", frozen)
        archived_run = {member["path"]: member for name, members in archives.items()
                        if not name.startswith(("archives/source_", "archives/external_source_snapshots_")) for member in members}
        if set(archived_run) != set(inventory):
            raise ValueError("not every retained run file was archived exactly once")
        if sum(len(members) for name, members in archives.items() if not name.startswith(
                ("archives/source_", "archives/external_source_snapshots_"))) != len(inventory):
            raise ValueError("retained files occur more than once in run archives")
        for name, record in inventory.items():
            if archived_run[name]["sha256"] != record["sha256"] or sha(run/name) != record["sha256"]:
                raise ValueError(f"run file changed during packaging: {name}")
        artifacts = [dict(path=p.relative_to(stage).as_posix(), bytes=p.stat().st_size, sha256=sha(p)) for p in files_under(stage)]
        manifest = dict(schema="twingraph.value.package.v5", created_utc=datetime.now(timezone.utc).isoformat(),
            recorded_code_commit=commit, code_commit_scope="recorded identifier; exact source-byte manifests are authoritative",
            input_root_scope="development" if development_root else "mixed_or_formal",
            input_run=str(run), retained_files=inventory, groups=groups, systems=systems, models=models,
            designated_freeze=frozen and {key:value for key,value in frozen.items() if key!="value"},
            source_manifests=source_rows, archives=archives, artifacts=artifacts,
            categories={key:len(value) for key,value in categories.items()},
            group_statuses=dict(Counter(g["status"] for g in groups)),
            failed_development=[g for g in groups if g["development"] and g["status"]!="completed"],
            environment=environment(), archive_size_limit_bytes=LIMIT,
            notes=["All retained files, including failed development, timeouts, logs and state/image traces, are preserved.",
                   "This is an integrity package, not a publication-readiness or model-superiority claim.",
                   "No training, threshold fitting, ranking, simulation, selection or publishing occurs during packaging."])
        write(stage/"manifest.json",manifest)
        (stage/"manifest.sha256").write_text(sha(stage/"manifest.json")+"  manifest.json\n",encoding="utf-8")
        if output.exists():
            output.rmdir()  # Verified explicitly named empty output directory only.
        stage.rename(output)
    return dict(output=str(output), retained_files=len(inventory), groups=len(groups), models=len(models),
        systems=len(systems), archives=len(archives), manifest_sha256=sha(output/"manifest.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--freeze")
    parser.add_argument("--source-snapshot", nargs="*", default=[])
    args = parser.parse_args()
    print(json.dumps(package(args.run,args.out,args.code_commit,args.freeze,args.source_snapshot),indent=2))


if __name__ == "__main__":
    main()
