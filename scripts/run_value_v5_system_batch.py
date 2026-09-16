"""Run all frozen independent system cases, retaining actual clocks and failures.

Each worker runs Top-K then the full digital twin sequentially. Across cases,
the prospectively declared four workers share the same CPU/Torch thread policy.
This script opens no reference outcome dataset and cannot select a new model.
"""
import argparse
import concurrent.futures
import copy
import multiprocessing
import os
from pathlib import Path
import sys
import time
import traceback

# Apply before NumPy/Torch/MuJoCo are imported in both parent and spawned child.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
if os.name != "nt":
    os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_value_v5_models import digest, read, sha, write_new, require_empty, validate_protocol
from scripts.export_value_v5_predictions import verify_selection


def validate_dispatch(spec, selection, planner, selected_checkpoint=None):
    binding = validate_protocol(spec)
    paths = {name:row["path"] for name,row in selection["models"].items()}
    selected = selection["selected"]
    if selected_checkpoint is not None:
        paths[selected] = str(Path(selected_checkpoint).resolve())
    verify_selection(selection, paths, spec["source_sha256"])
    declared = {row["name"]:row for row in spec["models"]["kinds"]}
    if set(selection["models"]) != set(declared):
        raise ValueError("selection must contain exactly the four prospectively declared models")
    for name, row in selection["models"].items():
        if (row["kind"] != declared[name]["kind"] or row["seed"] != declared[name]["seed"]
                or row["source_sha256"] != spec["source_sha256"]
                or row["encoding_sha256"] != spec["encoding_sha256"]
                or row["interface_sha256"] != spec["interface_sha256"]):
            raise ValueError("frozen model metadata differs from source/protocol")
    if min(selection["models"], key=lambda name:(selection["models"][name]["validation_nominal_brier"],name)) != selected:
        raise ValueError("model selection does not follow frozen validation Brier rule")
    expected_test = dict(seeds=spec["collection"]["test_seeds"],n=spec["candidates"]["n"],repeats=spec["collection"]["nominal_repeats"])
    if (selection["test"] != expected_test or sorted(selection["validation_seeds"]) != sorted(spec["collection"]["validation_seeds"])
            or selection["primary_k"] != spec["candidates"]["primary_k"]):
        raise ValueError("model-selection data and candidate budgets differ from prospective protocol")
    from simbench.value.system_v5 import validate_planner_record
    validate_planner_record(planner)
    canonical = ROOT/"experiments"/"value_v5"/"planner_record.json"
    if digest(planner) != digest(read(canonical)) or planner.get("interface_sha256") != spec["interface_sha256"]:
        raise ValueError("planner record differs from the recorded protocol planner/interface")
    system = spec["system"]
    if (system["n"] != spec["candidates"]["n"] or system["k"] != spec["candidates"]["primary_k"]
            or system["independent_namespaces"] != dict(twin=5107,target=7901)
            or len(system["seeds"]) != 4 or system["workers"] != 4):
        raise ValueError("system budget differs from fixed four-case, four-worker protocol")
    return dict(**binding, selected=selected, checkpoint=str(Path(paths[selected]).resolve()),
        checkpoint_sha256=selection["models"][selected]["sha256"], model_selection_sha256=digest(selection),
        planner_sha256=digest(planner))


def worker(request):
    """One bounded case; programming failures remain explicit, never relabeled."""
    directory = Path(request["out"])
    started = time.perf_counter()
    record = dict(seed=request["seed"], config_id=f"stage_v5_{request['seed']}",
        requested_target_trials_per_policy=request["spec"]["system"]["target_repeats"],
        status="started", output=str(directory), started_unix=time.time())
    try:
        require_empty(directory)
        spec, binding = request["spec"], request["binding"]
        validate_protocol(spec)
        if sha(binding["checkpoint"]) != binding["checkpoint_sha256"] or digest(request["planner"]) != binding["planner_sha256"]:
            raise ValueError("source/checkpoint/planner changed before worker execution")
        write_new(directory/"request.json",request)
        t = time.perf_counter()
        import torch
        record["torch_import_wall_seconds"] = time.perf_counter()-t
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        from simbench.value.value_v5 import ValueScorer
        from simbench.value.system_v5 import SystemConfig, run_pair
        t = time.perf_counter()
        scorer = ValueScorer(binding["checkpoint"],"cpu")
        measured_load = time.perf_counter()-t
        record.update(value_scorer_initialization_wall_seconds=measured_load,
            scorer_reported_load_seconds=float(scorer.load_seconds),
            control_environment={name:os.environ.get(name) for name in
                ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","MUJOCO_GL")},
            torch_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads(),
            torch_version=torch.__version__)
        if scorer.checkpoint_sha256 != binding["checkpoint_sha256"] or scorer.saved["source_sha256"] != spec["source_sha256"]:
            raise ValueError("loaded scorer changed the frozen checkpoint/source")
        system = spec["system"]
        config = SystemConfig(seed=request["seed"],n=system["n"],k=system["k"],
            validation_repeats=system["validation_repeats"],target_repeats=system["target_repeats"],
            accept_rate=system["accept_rate"],timeout=spec["collection"]["timeout_seconds"],
            friction_span=system["friction_half_width"],gain_span=system["actuator_gain_half_width"],render=True)
        pair = run_pair(config,request["planner"],scorer,directory)
        rows = read(directory/"timing_rows.json")["rows"]
        for row in rows:
            load = measured_load if row["policy"] == "top_k" else 0.
            row["batch_binding"] = binding
            row["model_initialization_wall_seconds"] = load
            row["seconds"]["cold_start_decision"] = row["seconds"]["decision"]+load
            row["seconds"]["cold_start_total"] = row["seconds"]["total"]+load
            row["cold_start_scope"] = "separately measured checkpoint/model initialization plus resident policy clock; excludes interpreter and Torch import"
        write_new(directory/"timing_rows_with_initialization.json",dict(schema="twingraph.system_run.v5",
            rows=rows,time_boundaries=rows[0]["timing_boundaries"],model_initialization=record))
        validate_protocol(spec)
        if sha(binding["checkpoint"]) != binding["checkpoint_sha256"]:
            raise ValueError("frozen checkpoint changed during system execution")
        record.update(status="completed",pair_sha256=sha(directory/"pair.json"),
                      actual_pair_wall_seconds=pair["actual_wall_seconds"],
                      target_successes=pair["target_successes"])
    except Exception as exc:
        record.update(status="error",exception_type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    record["worker_wall_seconds"] = time.perf_counter()-started
    # Never overwrite a previous attempt even if the caller is misconfigured.
    if not (directory/"worker_status.json").exists():
        write_new(directory/"worker_status.json",record)
    return record


def combine(output, spec, binding, records, batch_wall):
    rows, incomplete = [], []
    for record in sorted(records,key=lambda row:row["seed"]):
        directory = Path(record["output"])
        enriched = directory/"timing_rows_with_initialization.json"
        if enriched.exists():
            rows.extend(read(enriched)["rows"])
        else:
            # Preserve any real finished policy, but never fabricate a missing rollout.
            for policy in ("top_k","full"):
                path=directory/policy/"result.json"
                if path.exists():
                    rows.append(read(path))
        if record["status"] != "completed":
            incomplete.append(record)
    counts = {}
    requested = len(spec["system"]["seeds"])*spec["system"]["target_repeats"]
    for policy in ("top_k","full"):
        actual = [row for row in rows if row["policy"] == policy]
        successes = sum(row["target_successes"] for row in actual)
        counts[policy] = dict(requested_configurations=len(spec["system"]["seeds"]),
            requested_target_trials=requested, recorded_policy_results=len(actual),
            missing_policy_results=len(spec["system"]["seeds"])-len(actual),
            independent_target_successes=successes,
            requested_target_success_fraction=successes/requested,
            actual_twin_validation_calls=sum(row["simulation_calls"]["twin_validation"] for row in actual),
            actual_target_execution_calls=sum(row["simulation_calls"]["target_execution"] for row in actual),
            interpretation="Missing/program-error runs remain unresolved requested attempts; they are not physical negative labels.")
    metadata = dict(schema="twingraph.system_run.v5", **binding, requested_seeds=spec["system"]["seeds"],
        requested_target_trials_per_policy=requested, workers=spec["system"]["workers"],
        batch_complete=not incomplete and len(rows)==2*len(spec["system"]["seeds"]),
        failures=incomplete, denominator_summary=counts, overall_parallel_wall_seconds=batch_wall,
        model_initialization_records=[{k:v for k,v in r.items() if k in
            {"seed","torch_import_wall_seconds","value_scorer_initialization_wall_seconds","scorer_reported_load_seconds","worker_wall_seconds"}}
            for r in records])
    resident=dict(metadata,rows=rows,time_boundaries=rows[0].get("timing_boundaries") if rows else None,
        clock_scope="resident scorer; separate model initialization available per row and in cold-start artifact")
    write_new(output/"timing_rows.json",resident)
    cold_rows=[]
    for row in rows:
        if "cold_start_total" not in row["seconds"]:
            continue
        changed=copy.deepcopy(row)
        changed["seconds"]["resident_total"]=row["seconds"]["total"]
        changed["seconds"]["resident_decision"]=row["seconds"]["decision"]
        changed["seconds"]["total"]=row["seconds"]["cold_start_total"]
        changed["seconds"]["decision"]=row["seconds"]["cold_start_decision"]
        cold_rows.append(changed)
    write_new(output/"timing_rows_cold_start.json",dict(metadata,rows=cold_rows,
        time_boundaries=dict(total="resident full policy total plus separately measured checkpoint/model initialization for Top-K",
            decision="resident candidate-generation-through-selection plus measured model initialization for Top-K"),
        clock_scope="checkpoint/model-load-inclusive; Python process and Torch imports remain separate, captured in batch/worker metadata"))
    batch=dict(schema="twingraph.system_batch.v5",**binding,records=records,
        denominator_summary=counts,wall_seconds=batch_wall,
        status="completed" if resident["batch_complete"] else "error",
        timing_rows_sha256=sha(output/"timing_rows.json"),cold_timing_rows_sha256=sha(output/"timing_rows_cold_start.json"))
    write_new(output/"system_batch.json",batch)
    return batch


def run_batch(protocol, selection_path, planner_path, output, selected_checkpoint=None, check_only=False):
    spec, selection, planner=read(protocol),read(selection_path),read(planner_path)
    binding=validate_dispatch(spec,selection,planner,selected_checkpoint)
    if check_only:
        return dict(status="validated_only",**binding,seeds=spec["system"]["seeds"],workers=spec["system"]["workers"])
    output=Path(output).resolve();require_empty(output)
    requests=[dict(seed=seed,out=str(output/f"case_{seed}"),spec=spec,binding=binding,planner=planner)
              for seed in spec["system"]["seeds"]]
    write_new(output/"dispatch.json",dict(schema="twingraph.system_dispatch.v5",**binding,
        protocol_file_sha256=sha(protocol),selection_file_sha256=sha(selection_path),planner_file_sha256=sha(planner_path),
        script_sha256=sha(__file__),created_unix=time.time(),requests=requests,
        execution="four case workers; each executes Top-K then full sequentially; CPU Torch1, common control environment"))
    started=time.perf_counter();records=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=spec["system"]["workers"],
            mp_context=multiprocessing.get_context("spawn")) as executor:
        pending={executor.submit(worker,request):request for request in requests}
        for future in concurrent.futures.as_completed(pending):
            request=pending[future]
            try:
                record=future.result()
            except Exception as exc:
                record=dict(seed=request["seed"],output=request["out"],status="error",
                    exception_type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),
                    requested_target_trials_per_policy=spec["system"]["target_repeats"])
            records.append(record)
            print({key:record[key] for key in ("seed","status")},flush=True)
    return combine(output,spec,binding,records,time.perf_counter()-started)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol",required=True);parser.add_argument("--selection",required=True)
    parser.add_argument("--planner",required=True);parser.add_argument("--out",required=True)
    parser.add_argument("--selected-checkpoint",help="relocated selected checkpoint, identical frozen SHA256 required")
    parser.add_argument("--check-only",action="store_true")
    args=parser.parse_args()
    result=run_batch(args.protocol,args.selection,args.planner,args.out,args.selected_checkpoint,args.check_only)
    print({k:v for k,v in result.items() if k!="records"},flush=True)
    if result["status"]=="error":raise SystemExit(1)


if __name__=="__main__":main()
