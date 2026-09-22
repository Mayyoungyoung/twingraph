"""Fixed method waves with four independent, equally provisioned layout jobs.

These are measured concurrent-service wall times, not isolated robot latency.
No wave starts until every layout in the preceding wave has finished. Candidate
selection remains entirely inside the unchanged run_v12_system entry point.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--cores", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--egl-device", type=int, default=1)
    parser.add_argument("--n", type=int, default=48)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--methods",nargs="+",default=["all_twin","random_top_k","value_top_k",
        "random_early_stop","value_early_stop","geometry_top_k","geometry_early_stop"])
    args = parser.parse_args()
    if len(args.seeds) != len(args.cores) or not 1 <= len(args.seeds) <= 4:
        parser.error("one dedicated CPU per layout, at most four concurrent layouts")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.cores)) != len(args.cores):
        parser.error("duplicate layout or CPU")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("refusing to overwrite or silently resume experiment evidence")
    args.out.mkdir(parents=True, exist_ok=True)
    from simbench.value.system_v12 import METHODS
    methods = args.methods
    if len(set(methods))!=len(methods) or any(m not in METHODS for m in methods):
        parser.error("methods must be distinct supported system methods")
    from simbench.value.provenance_v12 import fingerprint
    runtime = fingerprint()["sha256"]
    binding = dict(runtime_sha256=runtime,
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        seeds=args.seeds, cores=args.cores, methods=methods, n=args.n, k=args.k,
        egl_device=args.egl_device, timing_mode="fixed_waves_concurrent_service_wall_time",
        interpretation="Actual measured wall times under shared GPU load, not isolated request latency",
        waves=[], complete=False)
    def system_query(command):
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=10)
            return dict(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)
        except Exception as error:
            return dict(error=str(error))

    binding["hardware"] = dict(
        cpu_map=system_query(["lscpu", "-p=CPU,CORE,SOCKET"]),
        gpus=system_query(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version", "--format=csv,noheader"]))

    def save():
        (args.out / "resource_schedule.json").write_text(json.dumps(binding, indent=2))

    save()
    environment = {**os.environ, "PYTHONPATH": ".", "MUJOCO_GL": "egl",
        "MUJOCO_EGL_DEVICE_ID": str(args.egl_device), "CUDA_VISIBLE_DEVICES": str(args.egl_device),
        "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    for method in methods:
        if fingerprint()["sha256"] != runtime:
            raise RuntimeError("physical runtime changed between method waves")
        wave = dict(method=method, start_unix=time.time(), jobs=[])
        binding["waves"].append(wave)
        processes = []
        for seed, core in zip(args.seeds, args.cores):
            log = (args.out / f"{method}_{seed}.log").open("w")
            command = ["taskset", "-c", str(core), sys.executable, "scripts/run_v12_system.py",
                "--out", str(args.out), "--checkpoint", str(args.checkpoint), "--seeds", str(seed),
                "--n", str(args.n), "--k", str(args.k), "--level", "L1", "--methods", method]
            process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
            job = dict(seed=seed, core=core, pid=process.pid, command=command, start_unix=time.time())
            wave["jobs"].append(job)
            processes.append((process, log, job))
        save()
        next_sample = 0.
        while processes:
            if time.monotonic() >= next_sample:
                sample = dict(unix=time.time(), method=method, active_jobs=len(processes),
                    loadavg=Path("/proc/loadavg").read_text().strip(),
                    cpu_stat=[line for line in Path("/proc/stat").read_text().splitlines() if line.startswith("cpu")],
                    gpu=system_query(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,power.draw,clocks.sm", "--format=csv,noheader"]),
                    processes=[])
                for process, _, job in processes:
                    try:
                        status = Path(f"/proc/{process.pid}/status").read_text().splitlines()
                        sample["processes"].append(dict(seed=job["seed"], pid=process.pid,
                            stat=Path(f"/proc/{process.pid}/stat").read_text().strip(),
                            status=[line for line in status if line.startswith(("Threads:", "VmRSS:", "Cpus_allowed_list:"))]))
                    except FileNotFoundError:
                        pass  # A completed child is collected immediately below.
                with (args.out / "resource_telemetry.jsonl").open("a") as telemetry:
                    telemetry.write(json.dumps(sample)+"\n")
                next_sample = time.monotonic()+10.
            for process, log, job in processes[:]:
                code = process.poll()
                if code is None:
                    continue
                log.close()
                job.update(exit_code=code, end_unix=time.time())
                processes.remove((process, log, job))
                save()
            if processes:
                time.sleep(1)
        wave["end_unix"] = time.time()
        save()
        if any(job["exit_code"] != 0 for job in wave["jobs"]):
            raise RuntimeError(f"incomplete wave {method}; preserve outputs and inspect resource or program error")
        print(json.dumps(dict(completed_wave=method, wall_seconds=wave["end_unix"]-wave["start_unix"])), flush=True)
    binding["complete"] = True
    save()


if __name__ == "__main__":
    main()
