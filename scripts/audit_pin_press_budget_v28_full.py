#!/usr/bin/env python3
"""Three-layout, eight-candidate paired full-task V27/V28 press comparison."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import subprocess
import sys
import time

from simbench.assembly.sensor_learning_v12 import PIN_PRESS_FIXED_V27,PIN_PRESS_REMAINING_V28
from simbench.value.plan import digest
from simbench.value.provenance_v12 import fingerprint as runtime_fingerprint
from simbench.value.system_v12 import save


LAYOUTS=(
    dict(name="neighborhood_14100",seed=14100,family="neighborhood",parent=4100),
    dict(name="broad_4102",seed=4102,family="broad",parent=4102),
    dict(name="broad_4104",seed=4104,family="broad",parent=4104),
)
POLICIES=(PIN_PRESS_FIXED_V27,PIN_PRESS_REMAINING_V28)


def complete(directory):
    path=directory/"summary.json"
    if not path.exists():return None
    row=json.loads(path.read_text(encoding="utf-8"))
    return row if row.get("attempted")==8 and row.get("valid")==8 else None


def run_job(job,base):
    directory=base/job["name"]/job["policy"]
    if summary:=complete(directory):return dict(**job,reused=True,summary=summary)
    directory.parent.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,"-m","scripts.collect_sliding_assembly_v23","--seed",str(job["seed"]),
        "--out",str(directory),"--pool-n","48","--candidate-n","8","--level","L0",
        "--domain","online","--layout-family",job["family"],"--split-group",
        f"neighborhood_parent_{job['parent']}","--pin-press-budget-policy",job["policy"]]
    if job["family"]=="neighborhood":
        command.extend(["--neighborhood-parent-seed",str(job["parent"])])
    log=base/"logs"/f"{job['name']}__{job['policy']}.log";log.parent.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    with log.open("w",encoding="utf-8") as stream:
        completed=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=10800,check=False)
    summary=complete(directory)
    if completed.returncode or summary is None:
        raise RuntimeError(f"full paired run failed: {job}; see {log}")
    return dict(**job,reused=False,wall_seconds=time.perf_counter()-started,summary=summary)


def candidate_rows(directory):
    request=json.loads((directory/"request.json").read_text(encoding="utf-8"))
    rows=[]
    for entry in request["candidates"]:
        result=json.loads((directory/"candidates"/entry["name"]/"result.json").read_text(encoding="utf-8"))
        audit=result["stage_audit"]
        rows.append(dict(name=entry["name"],proposal_sha256=digest(entry["proposal"]),
            trial_sha256=result["trial_sha256"],success=bool(result["success"]),
            pins_completed=all(audit[p]["completed"] for p in ("pin_left","pin_right")),
            pin_left_completed=bool(audit["pin_left"]["completed"]),
            pin_right_completed=bool(audit["pin_right"]["completed"]),
            first_failure=(result.get("first_failed_atom") or {}).get("reason") or result.get("error"),
            total_wall_seconds=result.get("total_wall_seconds"),result=str(directory/"candidates"/
                entry["name"]/"result.json")))
    return request,rows


def compare(base,jobs):
    layouts=[]
    for layout in LAYOUTS:
        paths={p:base/layout["name"]/p for p in POLICIES}
        old_request,old=candidate_rows(paths[PIN_PRESS_FIXED_V27])
        new_request,new=candidate_rows(paths[PIN_PRESS_REMAINING_V28])
        pairs=[]
        for a,b in zip(old,new):
            if a["name"]!=b["name"] or a["proposal_sha256"]!=b["proposal_sha256"]:
                raise ValueError("paired candidate semantics changed")
            if a["trial_sha256"]!=b["trial_sha256"]:
                raise ValueError("paired physical trial changed")
            pairs.append(dict(candidate=a["name"],proposal_sha256=a["proposal_sha256"],
                              trial_sha256=a["trial_sha256"],old=a,new=b))
        if old_request["decision_observation"]!=new_request["decision_observation"]:
            raise ValueError("paired decision observation changed")
        layouts.append(dict(**layout,decision_observation_sha256=old_request[
            "decision_observation"]["sha256"],pairs=pairs))
    all_pairs=[p for layout in layouts for p in layout["pairs"]]
    def aggregate(side):
        rows=[p[side] for p in all_pairs]
        return dict(candidates=len(rows),pin_stages_passed=sum(r["pins_completed"] for r in rows),
            pin_left_completed=sum(r["pin_left_completed"] for r in rows),
            pin_right_completed=sum(r["pin_right_completed"] for r in rows),
            full_successes=sum(r["success"] for r in rows),
            failure_reasons=dict(Counter(r["first_failure"] for r in rows if not r["success"])),
            total_wall_seconds=sum(float(r["total_wall_seconds"] or 0.) for r in rows))
    rescued=[p["candidate"]+"@"+layout["name"] for layout in layouts for p in layout["pairs"]
             if not p["old"]["pins_completed"] and p["new"]["pins_completed"]]
    new_full=[p["candidate"]+"@"+layout["name"] for layout in layouts for p in layout["pairs"]
              if not p["old"]["success"] and p["new"]["success"]]
    broken=[p["candidate"]+"@"+layout["name"] for layout in layouts for p in layout["pairs"]
            if p["old"]["success"] and not p["new"]["success"]]
    return dict(schema="twingraph.pin_press_budget_v28.full_pair_summary.r1",layouts=layouts,
        old=aggregate("old"),new=aggregate("new"),rescued_pin_stages=rescued,
        new_full_positives=new_full,broken_old_full_successes=broken,jobs=jobs)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=Path("results/v28_pin_budget/full_pairs"))
    parser.add_argument("--workers",type=int,default=3)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    jobs=[dict(**layout,policy=policy) for layout in LAYOUTS for policy in POLICIES]
    manifest=dict(schema="twingraph.pin_press_budget_v28.full_pair.r1",
        runtime_sha256=runtime_fingerprint()["sha256"],layouts=list(LAYOUTS),policies=list(POLICIES),
        candidates_per_layout=8,layout_selection_frozen_before_v28_execution=True,
        same_initial_state_candidate_semantics_and_physical_trial_required=True,no_training=True)
    save(args.out/"manifest.json",manifest)
    rows=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run_job,job,args.out):job for job in jobs}
        for future in as_completed(futures):
            row=future.result();rows.append(row);save(args.out/"progress.json",rows)
            print(json.dumps(dict(name=row["name"],policy=row["policy"],
                successes=row["summary"]["successes"])),flush=True)
    summary=compare(args.out,rows);save(args.out/"summary.json",summary)


if __name__=="__main__":main()
