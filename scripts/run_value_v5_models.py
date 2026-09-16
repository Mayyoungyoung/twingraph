"""Train exactly the prospectively declared v5 models on complete train/val data.

No held-out outcomes are opened and no model is selected here. A dispatch is an
immutable attempt: failures preserve their logs and cannot silently be resumed.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simbench.value.plan import digest


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def require_empty(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError("preserve existing dispatch/results; undocumented resume or overwrite is forbidden")
    path.mkdir(parents=True, exist_ok=True)


def validate_protocol(spec):
    from simbench.value.program_input_v5 import source_manifest, encoding_hash
    from simbench.value.skill_graph import interface_hash
    if spec.get("schema") != "twingraph.prospective_protocol.v5" or spec.get("status") != "FROZEN_BEFORE_FORMAL_COLLECTION":
        raise ValueError("require a prospectively frozen v5 protocol")
    source = source_manifest()
    if source != spec["sources"] or digest(source) != spec["source_sha256"]:
        raise ValueError("current source differs from prospective protocol")
    if encoding_hash() != spec["encoding_sha256"] or interface_hash() != spec["interface_sha256"]:
        raise ValueError("encoder or executable interface differs from prospective protocol")
    if digest(spec["geometry_sources"]) != spec["geometry_sha256"]:
        raise ValueError("geometry manifest binding mismatch")
    for name, expected in spec["geometry_sources"].items():
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or sha(path) != expected:
            raise ValueError(f"scene/geometry differs from prospective protocol: {name}")
    partitions = [spec["collection"][key] for key in ("train_seeds", "validation_seeds", "test_seeds")]
    partitions.append(spec["system"]["seeds"])
    flattened = [seed for partition in partitions for seed in partition]
    if any(not partition for partition in partitions) or len(set(flattened)) != len(flattened):
        raise ValueError("prospective configuration partitions must be nonempty and disjoint")
    declared = spec["models"]["kinds"]
    if len(declared) != 4 or len({m["name"] for m in declared}) != 4:
        raise ValueError("v5 requires the exact four prospectively declared unique models")
    if any(m["kind"] not in {"mlp", "linear"} or Path(m["name"]).name != m["name"] for m in declared):
        raise ValueError("unsupported model kind or output name")
    return dict(protocol_sha256=digest(spec), source_sha256=spec["source_sha256"],
                geometry_sha256=spec["geometry_sha256"])


def audit_trainval(roots, spec):
    from simbench.value.value_v5 import request_dispositions, load_groups
    dispositions = request_dispositions(roots, ("train", "val"))
    expected = {(split, seed) for split, key in (("train", "train_seeds"), ("val", "validation_seeds"))
                for seed in spec["collection"][key]}
    actual = [(r["request"]["split"], r["request"]["seed"]) for r in dispositions]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError("train/validation requested dispositions do not exactly match protocol")
    for row in dispositions:
        request = row["request"]
        if (row["source_sha256"] != spec["source_sha256"] or request["domain"] != "train"
                or request["n"] != spec["candidates"]["n"]
                or request["repeats"] != spec["collection"]["nominal_repeats"]
                or request["timeout_seconds"] != spec["collection"]["timeout_seconds"]):
            raise ValueError("requested source/nominal execution budget differs from protocol")
    # Full loader checks immutable graph, trial, source and nominal-label bindings.
    groups = load_groups(roots, ("train", "val"))
    if {g["collected_source_sha256"] for g in groups} != {spec["source_sha256"]}:
        raise ValueError("training data contains mixed source revisions")
    if len({g["config"] for g in groups}) != len(groups):
        raise ValueError("protocol requires one candidate pool per physical configuration")
    split = {s:[dict(group_id=g["id"], config_id=g["config"], input_sha256=g["input_sha256"])
                for g in groups if g["split"] == s] for s in ("train", "val")}
    if any(not rows for rows in split.values()):
        raise ValueError("at least one reached configuration is required in both train and validation")
    files = {}
    for row in dispositions:
        directory = Path(row["directory"])
        names = ("request.json", "source.json", "inputs.json", "skill_graphs.json", "outcomes.json", "complete.json") if row["status"] == "completed" else ("request.json", "source.json", "failure.json")
        files.update({str((directory/name).resolve()):sha(directory/name) for name in names})
    return dict(dispositions=dispositions, split=split, split_sha256=digest(split), files=files,
        requested=len(dispositions), reached=len(groups), failed_before_inputs=len(dispositions)-len(groups),
        train_candidates=sum(len(g["plans"]) for g in groups if g["split"] == "train"),
        validation_candidates=sum(len(g["plans"]) for g in groups if g["split"] == "val"))


def verify_files(files):
    if any(sha(path) != expected for path, expected in files.items()):
        raise ValueError("immutable source data changed during model batch")


def model_command(python, roots, directory, model, spec, device):
    return [str(python), "-m", "simbench.value.value_v5", "--data", *map(str, roots),
        "--out", str(directory), "--kind", model["kind"], "--seed", str(model["seed"]),
        "--epochs", str(spec["models"]["epochs"]), "--batch-size", str(spec["models"]["batch_size"]),
        "--lr", str(spec["models"]["learning_rate"]), "--device", device]


def verify_model(directory, model, spec, audit):
    import torch
    directory = Path(directory)
    required = ("best.pt", "summary.json", "history.json", "input_schema.json", "source.json", "split.json", "request_dispositions.json")
    if any(not (directory/name).is_file() for name in required):
        raise ValueError("model process did not finish all required checkpoint/summary artifacts")
    checkpoint = directory/"best.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    summary = read(directory/"summary.json")
    if (saved.get("schema") != "twingraph.value.v5" or saved["source_sha256"] != spec["source_sha256"]
            or saved["split_sha256"] != audit["split_sha256"] or saved["encoding_sha256"] != spec["encoding_sha256"]
            or saved["interface_sha256"] != spec["interface_sha256"] or saved["kind"] != model["kind"]
            or saved["seed"] != model["seed"] or saved["target"] != "direct_nominal_full_program_success"):
        raise ValueError("finished model differs from declared source/data/model configuration")
    if summary["checkpoint_sha256"] != sha(checkpoint) or digest(read(directory/"split.json")) != audit["split_sha256"]:
        raise ValueError("model checkpoint or training split binding mismatch")
    if (digest(read(directory/"source.json")) != spec["source_sha256"]
            or read(directory/"input_schema.json") != saved["input_schema"]
            or digest(read(directory/"request_dispositions.json")) != digest(audit["dispositions"])):
        raise ValueError("model encoder/source/request-disposition sidecar mismatch")
    history = read(directory/"history.json")
    if len(history) != spec["models"]["epochs"] or [r["epoch"] for r in history] != list(range(1,len(history)+1)):
        raise ValueError("model did not complete the declared epoch budget")
    for field, key in (("epochs", "epochs"), ("batch_size", "batch_size"), ("lr", "learning_rate")):
        if saved["training"][field] != spec["models"][key]:
            raise ValueError("checkpoint training hyperparameters differ from protocol")
    return dict(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=sha(checkpoint),
        files={name:sha(directory/name) for name in required}, epoch=saved["epoch"],
        validation_brier=saved["validation"]["brier"], input_dim=saved["input_schema"]["dim"],
        parameters=saved["parameters"], training_wall_seconds=summary["wall_seconds"])


def train_one(model, device, roots, output, python, spec, audit):
    name = model["name"]
    command = model_command(python, roots, output/name, model, spec, device)
    started = time.perf_counter()
    record = dict(name=name, kind=model["kind"], seed=model["seed"], device=device,
                  command=command, started_unix=time.time())
    try:
        validate_protocol(spec); verify_files(audit["files"])
        environment = os.environ.copy()
        environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
        log = output/"logs"/(name+".log"); log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("x", encoding="utf-8") as stream:
            process = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=False)
        record.update(returncode=process.returncode, log=str(log), log_sha256=sha(log))
        if process.returncode:
            raise RuntimeError(f"training process exited {process.returncode}; inspect preserved log")
        record.update(verify_model(output/name, model, spec, audit), status="completed")
        validate_protocol(spec); verify_files(audit["files"])
    except Exception as exc:
        record.update(status="failed", exception_type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    record["subprocess_and_verification_wall_seconds"] = time.perf_counter()-started
    write_new(output/"model_status"/(name+".json"), record)
    print(json.dumps({key:record[key] for key in ("name","device","status","subprocess_and_verification_wall_seconds")}), flush=True)
    return record


def run_batch(protocol, roots, output, devices=("cuda:0","cuda:1"), python=sys.executable, check_only=False):
    spec = read(protocol); binding = validate_protocol(spec)
    roots = [str(Path(root).resolve()) for root in roots]
    if not 1 <= len(devices) <= 2 or len(set(devices)) != len(devices) or any(d not in {"cuda:0","cuda:1"} for d in devices):
        raise ValueError("use one or two distinct declared CUDA devices")
    audit = audit_trainval(roots, spec)
    if check_only:
        return dict(status="validated_only", **binding, **{k:audit[k] for k in ("requested","reached","failed_before_inputs","train_candidates","validation_candidates")})
    output = Path(output).resolve(); require_empty(output)
    started = time.perf_counter()
    dispatch = dict(schema="twingraph.model_dispatch.v5", **binding, protocol_file_sha256=sha(protocol),
        script_sha256=sha(__file__), devices=list(devices), python=str(python), data=roots,
        models=spec["models"]["kinds"], created_unix=time.time(), dataset_audit=audit,
        execution="one sequential model lane per GPU, at most two simultaneous training subprocesses")
    write_new(output/"dispatch.json", dispatch)
    lanes = [spec["models"]["kinds"][index::len(devices)] for index in range(len(devices))]
    def lane(index):
        return [train_one(model, devices[index], roots, output, python, spec, audit) for model in lanes[index]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [executor.submit(lane,index) for index in range(len(devices))]
        rows = [row for future in futures for row in future.result()]
    batch = dict(schema="twingraph.model_batch.v5", **binding, dispatch_sha256=digest(dispatch),
        models=sorted(rows,key=lambda row:row["name"]), requested_models=len(rows),
        completed_models=sum(row["status"]=="completed" for row in rows),
        wall_seconds=time.perf_counter()-started,
        selection="No model selected here; use validation-only prediction export after all declared models finish.")
    batch["status"] = "completed" if batch["completed_models"] == len(rows) == 4 else "failed"
    write_new(output/"model_batch.json",batch)
    return batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True); parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--out", required=True); parser.add_argument("--devices", nargs="+", default=["cuda:0","cuda:1"])
    parser.add_argument("--python", default=sys.executable); parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    result = run_batch(args.protocol,args.data,args.out,args.devices,args.python,args.check_only)
    print(json.dumps({k:v for k,v in result.items() if k not in {"models"}}),flush=True)
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
