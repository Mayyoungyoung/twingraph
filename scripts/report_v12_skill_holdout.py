"""Read every predeclared skill trial and make a compact, auditable report."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def wilson(success, total):
    z=1.959963984540054; p=success/total; d=1+z*z/total
    center=(p+z*z/(2*total))/d
    radius=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/d
    return [center-radius,center+radius]


def report(directory):
    directory=Path(directory)
    protocol=json.loads((directory / "protocol.json").read_text())
    rows=[]
    for trial in protocol["trials"]:
        path=directory / f"{trial['skill']}{trial['seed']}" / "summary.json"
        raw=json.loads(path.read_text())
        steps=raw.get("steps",[])
        failure=next((x for x in steps if not x["ok"]),None)
        row=dict(seed=trial["seed"],skill=trial["skill"],success=raw["success"],
            failure_skill=failure["skill"] if failure else None,
            failure_reason=failure.get("reason") if failure else None,
            learned_policy_called=raw.get("learned_policy_called",False),
            wall_seconds=raw.get("wall_seconds"), recorded=trial["record"],
            raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if trial["skill"]=="pin":
            actor=next((x["metrics"] for x in steps if x["skill"]=="learned_insert"),{})
            search=actor.get("search",{})
            final=next((x["metrics"] for x in reversed(steps)
                        if x["skill"]=="inspect_pin_inserted"),{})
            row.update(actor_steps=actor.get("steps"),search_iterations=search.get("iterations"),
                inferred_entry=search.get("entry_detected"),peak_force_n=actor.get("peak_force_n"),
                held_contiguous_depth_m=actor.get("independent_insertion_evaluation",{}).get("valid_depth_span_m"),
                final_contiguous_depth_m=final.get("valid_depth_span_m"),
                final_mouth_continuity_pass=final.get("mouth_continuity_pass"))
        else:
            wipe=next((x["metrics"] for x in steps if x["skill"]=="wipe_surface"),{})
            row.update(coverage=wipe.get("coverage"),contact_fraction=wipe.get("contact_fraction"),
                       peak_force_n=wipe.get("peak_force_n"))
        rows.append(row)
    result=dict(protocol_sha256=hashlib.sha256((directory / "protocol.json").read_bytes()).hexdigest(),
        runtime=protocol["runtime"],scope=protocol["scope"],complete_task_trial=False,trials=rows,
        by_skill={})
    runtime=Path(protocol["runtime"])
    if runtime.is_dir():
        changed=[name for name,expected in protocol["source_sha256"].items()
                 if not (runtime/name).exists() or hashlib.sha256((runtime/name).read_bytes()).hexdigest()!=expected]
        result["frozen_source_check"]=dict(unchanged=not changed,changed_files=changed,
                                           checked_files=len(protocol["source_sha256"]))
    for skill in ("pin","wipe"):
        selected=[x for x in rows if x["skill"]==skill]
        n=len(selected); successes=sum(x["success"] for x in selected)
        result["by_skill"][skill]=dict(success=successes,total=n,rate=successes/n,
                                      wilson_95_interval=wilson(successes,n))
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--directory",required=True)
    args=parser.parse_args(); result=report(args.directory)
    (Path(args.directory) / "compact_report.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
