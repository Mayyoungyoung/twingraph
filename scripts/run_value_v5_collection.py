"""Collect only the exact prospective configuration budget and frozen source."""
import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from simbench.value.plan import digest
from simbench.value.program_input_v5 import source_manifest
from simbench.value.collect import dump
from simbench.value.collect_v5 import worker


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol",required=True);p.add_argument("--out",required=True)
    p.add_argument("--splits",nargs="+",choices=("train","val","test"),required=True)
    p.add_argument("--selection");p.add_argument("--thresholds")
    a=p.parse_args();spec=json.loads(Path(a.protocol).read_text())
    if digest(source_manifest())!=spec["source_sha256"]:
        raise ValueError("source differs from prospective freeze")
    for name,expected in spec["geometry_sources"].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            raise ValueError(f"scene/geometry differs from prospective freeze: {name}")
    if len(set(a.splits))!=len(a.splits):raise ValueError("duplicate requested split")
    binding=dict(protocol_sha256=digest(spec))
    if "test" in a.splits:
        if a.splits!=["test"] or not a.selection or not a.thresholds:
            raise ValueError("test collection requires prior model/threshold freeze and a separate invocation")
        selection=json.loads(Path(a.selection).read_text());thresholds=json.loads(Path(a.thresholds).read_text())
        if selection["source_sha256"]!=spec["source_sha256"] or selection["test"]!={
                "seeds":spec["collection"]["test_seeds"],"n":spec["candidates"]["n"],"repeats":spec["collection"]["nominal_repeats"]}:
            raise ValueError("frozen model/test budget differs from prospective protocol")
        if (thresholds.get("source_sha256")!=spec["source_sha256"] or
                thresholds.get("selected")!=selection["selected"] or
                thresholds.get("selection_metadata",{}).get("model_selection_sha256")!=digest(selection) or
                {k:v["model_sha256"] for k,v in thresholds.get("methods",{}).items()}!={
                    k:v["sha256"] for k,v in selection["models"].items()} or
                thresholds.get("primary_k")!=spec["candidates"]["primary_k"]):
            raise ValueError("classification threshold/source/model bindings differ from the prior freeze")
        for model in selection["models"].values():
            if hashlib.sha256(Path(model["path"]).read_bytes()).hexdigest()!=model["sha256"]:
                raise ValueError("frozen model weights changed before test collection")
        binding.update(model_selection_sha256=digest(selection),thresholds_sha256=digest(thresholds))
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    marker=out/("dispatch_"+"_".join(a.splits)+".json")
    if marker.exists():raise FileExistsError("retain original requested batch; do not silently retry configurations")
    requests=[]
    for split in a.splits:
        key={"train":"train_seeds","val":"validation_seeds","test":"test_seeds"}[split]
        requests.extend(dict(seed=seed,out=str(out),split=split,domain="reference" if split=="test" else "train",
            n=spec["candidates"]["n"],repeats=spec["collection"]["nominal_repeats"],
            timeout=spec["collection"]["timeout_seconds"]) for seed in spec["collection"][key])
    dump(marker,dict(schema="twingraph.collection_dispatch.v5",created_unix=time.time(),requests=requests,**binding))
    started=time.perf_counter();results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=spec["collection"]["workers"],
            mp_context=multiprocessing.get_context("spawn")) as pool:
        futures=[pool.submit(worker,r) for r in requests]
        for future in concurrent.futures.as_completed(futures):
            result=future.result();results.append(result)
            if "error" in result:
                print(json.dumps(dict(completed=len(results),requested=len(requests),seed=result["request"]["seed"],
                    exception_type=result["exception_type"],message=result["message"])),flush=True)
            else:
                print(json.dumps(dict(completed=len(results),requested=len(requests),seed=result["seed"],
                    success=result["nominal_successes"],candidates=result["candidates"],seconds=result["wall_seconds"])),flush=True)
    dump(out/("batch_"+"_".join(a.splits)+".json"),dict(results=results,wall_seconds=time.perf_counter()-started,**binding))
    if any("error" in r for r in results):raise SystemExit(1)


if __name__=="__main__":main()
