"""Package completed value-v4 evidence; never train, select, or evaluate models.

Example: python scripts/package_value_v4.py --run results/v4 \
    --out results/v4/release --code-commit <recorded-commit>

Run only after the frozen evaluation/deployment has finished. Formal source
bytes must match every retained source manifest. Historical development-source
gaps are reported explicitly and do not invalidate the separate formal data.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

LIMIT = 95 * 1024 * 1024
MODEL_FILES = ("best.pt", "history.json", "split.json", "source.json", "summary.json")
EXCLUDED = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".git", "results"}
REQUIRED_EVIDENCE = ("predictions.json", "ranking_audit.json", "module_timing.json", "deployment_summary.json")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    # This is the serialization used by simbench.value.plan.digest.
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def development(path):
    return any(re.match(r"^(pilot|development|dev|smoke)(?:$|[_\d-])", part.lower())
               for part in Path(path).parts)


def files_under(root, *, exclude_results=False):
    root = Path(root)
    if not root.exists():
        return []
    result = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        forbidden = EXCLUDED if exclude_results else EXCLUDED - {"results"}
        if any(part in forbidden for part in relative.parts) or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_symlink():
            raise ValueError(f"refuse ambiguous symlink in release: {path}")
        if path.is_file():
            result.append(path)
    return result


def bounded_file(path):
    if Path(path).stat().st_size > LIMIT:
        raise ValueError(f"artifact exceeds 95 MiB; nothing may be silently dropped: {path}")


def archive(output, entries, generated=None):
    """entries maps archive names to immutable source files; include every file."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with tarfile.open(output, "w:gz", compresslevel=6) as tar:
        for name, source in sorted(entries.items()):
            source = Path(source)
            bounded_file(source)
            before = sha(source)
            size = source.stat().st_size
            tar.add(source, arcname=name, recursive=False)
            if before != sha(source):
                raise ValueError(f"source changed during packaging: {source}")
            records.append(dict(path=name, bytes=size, sha256=before))
        for name, payload in sorted((generated or {}).items()):
            if isinstance(payload, str):
                payload = payload.encode()
            if len(payload) > LIMIT:
                raise ValueError(f"archive member exceeds 95 MiB: {name}")
            item = tarfile.TarInfo(name)
            item.size = len(payload)
            item.mode = 0o644
            tar.addfile(item, io.BytesIO(payload))
            records.append(dict(path=name, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()))
    bounded_file(output)
    # Check the written stream, rather than merely trusting its input manifest.
    expected = {r["path"]: r for r in records}
    with tarfile.open(output, "r:gz") as tar:
        members = tar.getmembers()
        if len(members) != len(expected):
            raise ValueError(f"duplicate or missing archive entry in {output}")
        for member in members:
            stream = tar.extractfile(member)
            if stream is None or hashlib.sha256(stream.read()).hexdigest() != expected[member.name]["sha256"]:
                raise ValueError(f"archive verification failed: {output}:{member.name}")
    return records


def copy_file(source, target):
    bounded_file(source)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if sha(source) != sha(target):
        raise ValueError(f"copy integrity failure: {source}")


def unique_json(paths, description):
    paths = sorted(set(paths))
    if not paths:
        raise ValueError(f"missing {description}")
    hashes = {sha(p) for p in paths}
    if len(hashes) != 1:
        raise ValueError(f"conflicting {description}: {[str(p) for p in paths]}")
    return paths[0], read(paths[0])


def source_mapping(value):
    value = value.get("files", value) if isinstance(value, dict) else value
    if not isinstance(value, dict) or not value:
        raise ValueError("source manifest must be a nonempty path-to-SHA256 object")
    for name, checksum in value.items():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise ValueError(f"invalid source manifest entry: {name}")
    return value


class SourceResolver:
    def __init__(self, repo, run, commit):
        self.repo, self.commit = repo, commit
        self.cache, self.snapshot_bytes = {}, {}
        self.snapshot_paths = sorted(run.glob("*source*snapshot*.tar*"))
        for path in self.snapshot_paths:
            with tarfile.open(path, "r:*") as tar:
                for item in tar.getmembers():
                    if not item.isfile():
                        continue
                    components = PurePosixPath(item.name).parts
                    if "simbench" not in components:
                        continue
                    relative = PurePosixPath(*components[components.index("simbench"):]).as_posix()
                    stream = tar.extractfile(item)
                    payload = stream.read()
                    self.snapshot_bytes[(relative, hashlib.sha256(payload).hexdigest())] = payload

    def resolve(self, relative, checksum):
        key = relative, checksum
        if key in self.cache:
            return self.cache[key]
        path = self.repo / relative
        if path.is_file() and sha(path) == checksum:
            payload, origin = path.read_bytes(), "current_source_tree"
        elif key in self.snapshot_bytes:
            payload, origin = self.snapshot_bytes[key], "retained_collection_source_snapshot"
        else:
            payload, origin = None, None
            if (self.repo / ".git").exists():
                revisions = [self.commit]
                history = subprocess.run(["git", "rev-list", "--all", "--", relative], cwd=self.repo,
                                         capture_output=True, text=True, check=False)
                if history.returncode == 0:
                    revisions.extend(history.stdout.splitlines())
                for revision in dict.fromkeys(revisions):
                    result = subprocess.run(["git", "show", f"{revision}:{relative}"], cwd=self.repo,
                                            capture_output=True, check=False)
                    if result.returncode == 0 and hashlib.sha256(result.stdout).hexdigest() == checksum:
                        payload, origin = result.stdout, f"git:{revision}"
                        break
        self.cache[key] = payload, origin
        return payload, origin


def group_record(directory, frozen):
    inp = read(directory / "inputs.json") if (directory / "inputs.json").exists() else None
    done = read(directory / "complete.json") if (directory / "complete.json").exists() else None
    failure = read(directory / "failure.json") if (directory / "failure.json").exists() else None
    outcome = read(directory / "outcomes.json") if (directory / "outcomes.json").exists() else {}
    trials = outcome.get("trials", []) if isinstance(outcome, dict) else outcome
    request = failure.get("request") if failure else None
    if inp:
        ih = digest(inp)
        if outcome and outcome.get("input_sha256") != ih:
            raise ValueError(f"raw outcomes do not match inputs: {directory}")
        if done and done.get("input_sha256") != ih:
            raise ValueError(f"completion does not match inputs: {directory}")
        task = inp.get("task", {})
        seed, checkpoint, split = task.get("seed"), inp.get("checkpoint"), inp.get("declared_split")
        if done and not request and trials:
            # Recover the recorded request only if its exact hash confirms it.
            domains = {t.get("trial", {}).get("domain") for t in trials}
            repeats = {t.get("trial", {}).get("repeat") for t in trials}
            if len(domains) == 1 and None not in repeats:
                inferred = dict(seed=seed, split=split, domain=next(iter(domains)),
                                n=inp.get("pool_counts", {}).get("requested"), repeats=len(repeats),
                                checkpoint=checkpoint, timeout_seconds=180.)
                if digest(inferred) == done.get("request_sha256"):
                    request = inferred
    else:
        seed = (request or {}).get("seed")
        checkpoint = (request or {}).get("checkpoint")
        split = (request or {}).get("split")
    if request is None and split == "test":
        tasks = frozen["test"].get("tasks", [])
        if {"seed": seed, "checkpoint": checkpoint} in tasks:
            request = dict(seed=seed, checkpoint=checkpoint, split=split,
                           n=frozen["test"]["n"], repeats=frozen["test"]["repeats"], source="frozen_test_design")
    status = "completed" if done else "failed_after_checkpoint" if failure and inp else "failed_before_candidate_inputs" if failure else "incomplete"
    error = str((failure or {}).get("error", ""))
    setup = bool(failure and not inp and ("in create_scene" in error or "in prepare_checkpoint" in error))
    counts = Counter((t.get("candidate_id"), t.get("trial", {}).get("domain"), t.get("trial", {}).get("repeat")) for t in trials)
    if any(n != 1 for n in counts.values()):
        raise ValueError(f"duplicate raw rollout identity: {directory}")
    return dict(directory=directory.name, seed=seed, checkpoint=checkpoint, split=split,
                configuration_id=inp.get("split_group") if inp else None, status=status,
                requested=request, checkpoint_reached=inp is not None,
                checkpoint_preparation_failure=setup,
                pre_input_failure_unclassified=bool(failure and not inp and not setup),
                candidates=len(inp.get("candidates", [])) if inp else 0,
                requested_candidates=(request or {}).get("n"),
                requested_rollouts=(request["n"] * request["repeats"] if request and request.get("n") is not None and request.get("repeats") is not None else None),
                rollouts=len(trials), successes=sum(t.get("success") is True for t in trials),
                prefix_successes=sum(t.get("prefix_success") is True for t in trials),
                timeouts=sum(t.get("timeout") is True for t in trials),
                invalid_rollouts=sum(t.get("valid") is not True for t in trials),
                measured_rollout_wall_seconds=sum(t["wall_seconds"] for t in trials if isinstance(t.get("wall_seconds"), (int, float))),
                rollout_wall_seconds_missing=sum(not isinstance(t.get("wall_seconds"), (int, float)) for t in trials),
                checkpoint_seconds=inp.get("checkpoint_seconds") if inp else None,
                checkpoint_atom_calls=len(inp.get("checkpoint_trace", [])) if inp else None,
                source_sha256=inp.get("source_sha256") if inp else None,
                source_manifest_present=(directory / "source.json").exists())


def aggregate(groups):
    def one(rows):
        value = dict(recorded_requested_groups=len(rows),
                     recorded_configurations=len({r["seed"] for r in rows if r["seed"] is not None}),
                     checkpoint_reached_groups=sum(r["checkpoint_reached"] for r in rows),
                     completed_groups=sum(r["status"] == "completed" for r in rows),
                     checkpoint_preparation_failures=sum(r["checkpoint_preparation_failure"] for r in rows),
                     unclassified_pre_input_failures=sum(r["pre_input_failure_unclassified"] for r in rows),
                     requested_rollouts_known=sum(r["requested_rollouts"] for r in rows if r["requested_rollouts"] is not None),
                     requested_rollouts_unknown_groups=sum(r["requested_rollouts"] is None for r in rows),
                     statuses=dict(Counter(r["status"] for r in rows)))
        for key in ("candidates", "rollouts", "successes", "prefix_successes", "timeouts", "invalid_rollouts"):
            value[key] = sum(r[key] for r in rows)
        return value
    return dict(total=one(groups), splits={str(split): one([r for r in groups if r["split"] == split])
                                          for split in sorted({r["split"] for r in groups}, key=str)})


def development_counts(paths, run):
    """Describe retained pilot outcomes separately, without treating them as test data."""
    rows = []
    for path in paths:
        if path.name != "outcomes.json":
            continue
        value = read(path)
        trials = value.get("trials", []) if isinstance(value, dict) else value
        if not isinstance(trials, list):
            rows.append(dict(path=path.relative_to(run).as_posix(), status="unrecognized_historical_outcome_schema"))
            continue
        rows.append(dict(path=path.relative_to(run).as_posix(), status="retained_raw_pilot_trials",
                         rollouts=len(trials), successes=sum(r.get("success") is True for r in trials),
                         timeouts=sum(r.get("timeout") is True for r in trials)))
    return dict(outcome_files=rows,
                observed_rollouts=sum(r.get("rollouts", 0) for r in rows),
                observed_successes=sum(r.get("successes", 0) for r in rows),
                observed_timeouts=sum(r.get("timeouts", 0) for r in rows),
                retained_failure_files=[p.relative_to(run).as_posix() for p in paths if p.name == "failure.json"])


def environment():
    versions = {}
    for package in ("torch", "mujoco", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20, check=False)
        gpu = dict(available=result.returncode == 0, query_output=result.stdout.strip() or None,
                   error=result.stderr.strip() or None)
    except (OSError, subprocess.TimeoutExpired) as exc:
        gpu = dict(available=False, query_output=None, error=str(exc))
    return dict(kind="packaging_host_observation", python=sys.version, executable=sys.executable,
                platform=platform.platform(), hostname=platform.node(), libraries=versions, gpu=gpu,
                note="These versions describe this packaging process; retained environment files, if present, document earlier execution hosts.")


def package(run, output, commit):
    repo = Path(__file__).resolve().parents[1]
    run, output = Path(run).resolve(), Path(output).resolve()
    if not run.is_dir() or output == run or output == repo or run.is_relative_to(output):
        raise ValueError("run must exist and output must be a distinct release directory")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refuse to overwrite a nonempty release: {output}")
    all_files = [p for p in files_under(run) if not p.is_relative_to(output)]
    formal = [p for p in all_files if not development(p.relative_to(run))]
    freeze_path, frozen = unique_json([p for p in formal if p.name in {"freeze.json", "frozen.json"}], "frozen model manifest")
    if frozen.get("schema") != "twingraph.schema_experiment.freeze.v1":
        raise ValueError("unsupported frozen experiment")
    model_rows = frozen.get("models", [])
    if len(model_rows) != 12 or len({r["name"] for r in model_rows}) != 12:
        raise ValueError("release requires all 12 independently trained frozen models")
    if frozen.get("selected") not in {r["name"] for r in model_rows}:
        raise ValueError("frozen selected model is absent; packager will not select a replacement")
    evidence = {}
    for name in REQUIRED_EVIDENCE:
        path, value = unique_json([p for p in formal if p.name == name], name)
        if "freeze_sha256" in value and value["freeze_sha256"] != digest(frozen):
            raise ValueError(f"evidence differs from frozen model manifest: {path}")
        evidence[name] = path
    csv_files = [p for p in formal if p.suffix.lower() == ".csv"]
    if not csv_files:
        raise ValueError("missing ranking CSV evidence")
    decisions = [(p, read(p)) for p in formal if p.name.endswith("_decision.json")]
    selected_decisions = [(p, row) for p, row in decisions if row.get("method") == frozen["selected"] and row.get("ranking")]
    png_files = [p for p in formal if p.suffix.lower() == ".png" and "final" in p.stem.lower()]
    if selected_decisions and not png_files:
        raise ValueError("reached deployment decisions exist but final PNG evidence is missing")
    groups = [group_record(p, frozen) for p in sorted((run / "data").glob("group_*")) if p.is_dir()]
    if not groups:
        raise ValueError("no formal collection groups to package")
    if any(g["status"] == "incomplete" for g in groups):
        raise ValueError("formal collection has running/incomplete groups; preserve failure records before release")
    expected_test = frozen["test"].get("tasks") or [dict(seed=seed, checkpoint=cp)
        for seed in frozen["test"].get("seeds", []) for cp in frozen["test"].get("checkpoints", [])]
    actual_test = [(g["seed"], g["checkpoint"]) for g in groups if g["split"] == "test"]
    if len(actual_test) != len(set(actual_test)) or set(actual_test) != {(t["seed"], t["checkpoint"]) for t in expected_test}:
        raise ValueError("formal raw test requests do not exactly cover the frozen test design")

    resolver = SourceResolver(repo, run, commit)
    source_refs = [p for p in all_files if p.name == "source.json"]
    overlays, source_rows = {}, []
    for path in source_refs:
        mapping = source_mapping(read(path))
        key = digest(mapping)
        is_dev = development(path.relative_to(run))
        unresolved, resolved = [], []
        for relative, checksum in sorted(mapping.items()):
            payload, origin = resolver.resolve(relative, checksum)
            if payload is None:
                unresolved.append(dict(path=relative, sha256=checksum))
            else:
                overlays[f"snapshots/{key}/{relative}"] = payload
                resolved.append(dict(path=relative, sha256=checksum, origin=origin))
        if unresolved and not is_dev:
            raise ValueError(f"formal source bytes unavailable for {path}: {unresolved}")
        overlays[f"snapshots/{key}/source.json"] = json.dumps(mapping, indent=2).encode()
        source_rows.append(dict(reference=path.relative_to(run).as_posix(), source_sha256=key,
                                development_only=is_dev, exact_bytes_complete=not unresolved,
                                resolved=resolved, unrecoverable_historical_source=unresolved))
    formal_refs = {row["reference"]: row for row in source_rows if not row["development_only"]}
    for group in groups:
        if group["checkpoint_reached"]:
            reference = f"data/{group['directory']}/source.json"
            if reference not in formal_refs or formal_refs[reference]["source_sha256"] != group["source_sha256"]:
                raise ValueError(f"formal group source manifest binding mismatch: {reference}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.building-", dir=output.parent) as temp:
        stage = Path(temp) / "release"
        stage.mkdir()
        archives = {}
        used = set()
        raw = {p.relative_to(run).as_posix(): p for p in all_files if p.is_relative_to(run / "data")}
        dev = {p.relative_to(run).as_posix(): p for p in all_files if development(p.relative_to(run))}
        execution_roots = {p.parent for p, _ in decisions}
        execution_roots.update(p.parent for p in formal if p.name == "failure.json" and any(x.startswith("deployment_") for x in p.relative_to(run).parts))
        execution = {p.relative_to(run).as_posix(): p for p in formal if any(p.is_relative_to(d) for d in execution_roots)}
        for name, entries in (("stage_graph_v4", raw), ("stage_graph_v4_execution", execution),
                              ("stage_graph_v4_development", dev)):
            relative = f"datasets/value/{name}.tar.gz"
            archives[relative] = archive(stage / relative, entries)
            used.update(entries.values())
        sources = {p.relative_to(repo).as_posix(): p for p in files_under(repo / "simbench", exclude_results=True)}
        sources.update({p.relative_to(repo).as_posix(): p for p in files_under(repo / "scripts")})
        sources.update({p.name: p for p in repo.glob("requirements*.txt")})
        sources.update({f"provenance/{p.name}": p for p in resolver.snapshot_paths})
        source_index = dict(base_tree="simbench/ and scripts/ contain packaging-time source and required assets",
                            exact_reproduction="Overlay snapshots/<source_sha256>/simbench on the base tree for the corresponding source.json; check every recorded SHA256.",
                            recorded_code_commit=commit, manifests=source_rows,
                            development_limit="Missing historical development source is explicitly listed; no exact reproduction claim applies to those records.")
        overlays["source_index.json"] = json.dumps(source_index, indent=2).encode()
        relative = "datasets/value/stage_graph_v4_source.tar.gz"
        archives[relative] = archive(stage / relative, sources, overlays)
        used.update(resolver.snapshot_paths)

        models = []
        for row in model_rows:
            name = row["name"]
            if Path(name).name != name or name in {".", ".."}:
                raise ValueError("invalid frozen model directory name")
            source_dir = run / "models" / name
            if not source_dir.is_dir():
                raise ValueError(f"frozen model directory missing: {source_dir}")
            for filename in MODEL_FILES:
                if not (source_dir / filename).is_file():
                    raise ValueError(f"required model artifact missing: {source_dir / filename}")
                copy_file(source_dir / filename, stage / "models" / "value" / "v4" / name / filename)
                used.add(source_dir / filename)
            if sha(source_dir / "best.pt") != row["sha256"]:
                raise ValueError(f"frozen model weight checksum mismatch: {name}")
            if digest(source_mapping(read(source_dir / "source.json"))) != row["source_sha256"]:
                raise ValueError(f"frozen model source checksum mismatch: {name}")
            if digest(read(source_dir / "split.json")) != row["split_sha256"]:
                raise ValueError(f"frozen model split checksum mismatch: {name}")
            models.append(dict(name=name, kind=row.get("kind"), seed=row.get("seed"), sha256=row["sha256"],
                               source_sha256=row["source_sha256"], split_sha256=row["split_sha256"]))
        alias = stage / "models" / "value" / "v4" / "best_graph_input.pt"
        copy_file(run / "models" / frozen["selected"] / "best.pt", alias)
        for name, path in evidence.items():
            copy_file(path, stage / "evidence" / name)
            used.add(path)
        copy_file(freeze_path, stage / "evidence" / "frozen.json")
        used.add(freeze_path)
        for path in [*csv_files, *png_files]:
            copy_file(path, stage / "evidence" / "original_paths" / path.relative_to(run))
            used.add(path)
        exported = [dict(source=p.relative_to(run).as_posix(), source_sha256=sha(p),
                         seed=row["seed"], checkpoint=row["checkpoint"], method=row["method"],
                         order=row["ranking"].get("order"), top_k=row["ranking"]["top_k"], chosen=row.get("chosen"))
                    for p, row in selected_decisions]
        write(stage / "evidence" / "selected_top_k.json", dict(freeze_sha256=digest(frozen),
              selected_model=frozen["selected"], decisions=exported,
              status="copied_from_frozen_selected_policy_decisions" if exported else "no_reached_selected_policy_decision"))
        # Preserve other logs, request files and environment evidence without
        # placing them in the formal training-data archive or drawing conclusions.
        for path in all_files:
            if path not in used:
                copy_file(path, stage / "evidence" / "run_provenance" / path.relative_to(run))
        decision_counts = []
        for path, row in decisions:
            online, deployed = row.get("validated", []), row.get("deployment", [])
            decision_counts.append(dict(path=path.relative_to(run).as_posix(), seed=row.get("seed"),
                checkpoint=row.get("checkpoint"), method=row.get("method"), status=row.get("status"),
                requested_online_budget=row.get("online_budget"), actual_online_rollouts=len(online),
                online_successes=sum(t.get("success") is True for t in online),
                online_timeouts=sum(t.get("timeout") is True for t in online),
                requested_deployment_attempts=row.get("requested_deployment_attempts"),
                actual_deployment_rollouts=len(deployed), deployment_successes=sum(t.get("success") is True for t in deployed),
                deployment_timeouts=sum(t.get("timeout") is True for t in deployed)))
        artifacts = []
        for path in files_under(stage):
            bounded_file(path)
            artifacts.append(dict(path=path.relative_to(stage).as_posix(), bytes=path.stat().st_size, sha256=sha(path)))
        manifest = dict(schema="twingraph.value.release.v4", created_utc=datetime.now(timezone.utc).isoformat(),
            recorded_code_commit=commit, code_commit_verification="recorded identifier; remote packaging tree need not be a Git checkout",
            freeze_sha256=digest(frozen), selected_model=frozen["selected"],
            selected_alias="models/value/v4/best_graph_input.pt", models=models,
            collection=aggregate(groups), groups=groups, frozen_test_design=frozen["test"],
            deployment_decisions=decision_counts, source_manifests=source_rows,
            development=dict(archive="datasets/value/stage_graph_v4_development.tar.gz", files=len(dev),
                             excluded_from_formal_collection_counts=True,
                             counts=development_counts(dev.values(), run),
                             source_gaps=[r for r in source_rows if r["development_only"] and not r["exact_bytes_complete"]]),
            environment=environment(), environment_evidence_files=[p.relative_to(run).as_posix() for p in all_files if "environment" in p.name.lower()],
            archive_size_limit_bytes=LIMIT, archives=archives, artifacts=artifacts,
            notes=["Counts describe recorded requests and actual raw rollouts, not an inferred unrecorded study design.",
                   "Incomplete/failed setup groups and censored outcomes remain in raw archives; timeouts are counted separately.",
                   "Model selection, metrics, top-k and deployment choices are copied from frozen evidence; the packager computes none of them."])
        write(stage / "manifest.json", manifest)
        (stage / "manifest.sha256").write_text(sha(stage / "manifest.json") + "  manifest.json\n", encoding="utf-8")
        if output.exists():
            output.rmdir()  # Only the verified, explicitly named empty output.
        stage.rename(output)
    return dict(output=str(output), models=len(models), raw_groups=len(groups),
                archive_sizes={name: (output / name).stat().st_size for name in archives},
                manifest_sha256=sha(output / "manifest.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.run, args.out, args.code_commit), indent=2))


if __name__ == "__main__":
    main()
