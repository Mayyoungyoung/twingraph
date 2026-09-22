"""Export audited, complete actual online all-twin labels without rerunning.

Source request, graph and result bytes are preserved. This only changes the
directory layout for the matrix loader, and never invents collection latency.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from scripts.analyze_v12_system import analyze


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def export_matrices(root, out, seeds, *, expected_pool_size=48):
    root, out = Path(root).resolve(), Path(out).resolve()
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("explicit unique expected layout seeds are required")
    if out == root or out in root.parents or root in out.parents:
        raise ValueError("export destination must be separate from the online source tree")
    if out.exists():
        raise ValueError("export destination already exists; source and prior evidence will not be overwritten")
    if not isinstance(expected_pool_size, int) or expected_pool_size < 1:
        raise ValueError("invalid required full pool size")
    audit = analyze(root, seeds=seeds, methods=("all_twin",), legacy=False)
    if audit["status"] != "complete":
        raise ValueError("cannot export incomplete or invalid actual online all_twin: " + json.dumps(
            {key:audit[key] for key in ("incomplete", "invalid", "paired_checks", "cross_layout_provenance_consistent")}))
    if any(row["pool_size"] != expected_pool_size or row["initial_twin_calls"] != expected_pool_size for row in audit["rows"]):
        raise ValueError("all_twin pool does not match the explicitly required complete candidate count")
    sources = []
    for row in audit["rows"]:
        directory = root/f"seed_{row['seed']}"/"all_twin"
        request = json.loads((directory/"request.json").read_text())
        paths = [directory/"request.json", directory/"runtime_sources.json", directory/"summary.json"]
        paths += [directory/"twins"/p["name"]/name for p in request["pool"] for name in ("result.json", "input_graph.json")]
        for path in paths:
            if not path.resolve().is_relative_to(directory.resolve()):
                raise ValueError("candidate path escapes its audited method directory")
        sources.append((row, directory, request, {str(p):sha(p) for p in paths}))
    # Source hashes now predate this second validation. Final hash checks
    # below make the audited/copy interval coherent even if another process
    # is still writing the source tree.
    if analyze(root, seeds=seeds, methods=("all_twin",), legacy=False) != audit:
        raise ValueError("online audit changed before export; wait for immutable completed evidence")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v12-matrix-export-", dir=out.parent) as temporary:
        staging = Path(temporary)/"matrix"
        staging.mkdir()
        exported = []
        for row, directory, request, source_hashes in sources:
            collect = staging/f"seed_{row['seed']}"/"collect"
            collect.mkdir(parents=True)
            copied = []
            def copy_exact(source, destination):
                # Recheck against the pre-copy audit snapshot: concurrent edits
                # cannot silently become a new label or altered wall time.
                expected = source_hashes[str(source)]
                if sha(source) != expected:
                    raise ValueError("online evidence changed during export")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                if sha(destination) != expected:
                    raise ValueError("export copy is not byte-identical")
                copied.append(dict(source=str(source), exported=destination.relative_to(staging).as_posix(), sha256=expected))
            for name in ("request.json", "runtime_sources.json"):
                copy_exact(directory/name, collect/name)
            copy_exact(directory/"summary.json", collect/"source_online_summary.json")
            timing = []
            for proposal in request["pool"]:
                source = directory/"twins"/proposal["name"]
                target = collect/"candidates"/proposal["name"]
                for name in ("result.json", "input_graph.json"):
                    copy_exact(source/name, target/name)
                result = json.loads((target/"result.json").read_text())
                timing.append(dict(candidate=proposal["name"],
                    source_recorded_timing_mode=result.get("timing_mode"),
                    unchanged_total_wall_seconds=result["total_wall_seconds"]))
            entry = dict(seed=row["seed"], pool_size=row["pool_size"], complete=True,
                source_method="all_twin", source_domain="online", source_runtime_sha256=row["runtime_sha256"],
                label_origin="complete actual online all_twin initial rollouts, exported byte-for-byte",
                source_request_sha256=source_hashes[str(directory/"request.json")],
                source_summary_sha256=source_hashes[str(directory/"summary.json")],
                export_timing_mode="format_only_no_physical_execution_no_new_latency",
                timing_note="Original result timing_mode and durations are unchanged. Missing timing_mode stays absent. Offline cost replay is not a new online decision measurement.",
                source_timings=timing, files=copied,
                excluded_from_labels="independent deployment and closed-loop suffix trials; these remain separate online evidence")
            dump(collect/"export_manifest.json", entry)
            exported.append(entry)
        # Detect a source modified after an individual file was copied.
        for _, _, _, source_hashes in sources:
            if any(sha(path) != expected for path, expected in source_hashes.items()):
                raise ValueError("online evidence changed before export completed")
        manifest = dict(schema="twingraph.actual_online_all_twin_matrix_export.v12", complete=True,
            created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), source_root=str(root),
            expected_seeds=list(seeds), expected_pool_size=expected_pool_size,
            source_runtime_sha256=audit["runtime_hashes"][0], layouts=exported,
            exporter_sha256=sha(__file__), audit_script_sha256=sha(Path(__file__).with_name("analyze_v12_system.py")),
            training_or_simulation_executed=False, original_outcomes_and_times_modified=False)
        dump(staging/"export_manifest.json", manifest)
        dump(staging/"source_online_audit.json", audit)
        staging.rename(out)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Actual online root containing seed_*/all_twin")
    parser.add_argument("--out", type=Path, required=True, help="New separate matrix root, must not exist")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--expected-pool-size", type=int, default=48)
    args = parser.parse_args()
    report = export_matrices(args.root, args.out, args.seeds, expected_pool_size=args.expected_pool_size)
    print(json.dumps(dict(complete=True, layouts=len(report["layouts"]),
        candidates=sum(r["pool_size"] for r in report["layouts"]), export=str(args.out))))


if __name__ == "__main__":
    main()
