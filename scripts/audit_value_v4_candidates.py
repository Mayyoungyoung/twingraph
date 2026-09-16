"""Descriptive audit of frozen candidate inputs and actual physical outcomes.

No simulator, Torch, fitting, new labels, or model selection is used. Associations
are pooled observations from correlated candidates/repetitions, not causal tests.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def canonical_paths(plan):
    keys=("type","part","start_q","joints","target","rotation")
    return {name:{key:path[key] for key in keys}
            for name,path in plan["prefix"].get("initial_artifacts",{}).items()}


def executable_key(plan):
    # Candidate/call trace IDs and path-binding IDs are excluded, actual calls,
    # object roles, controls and numerical trajectory content are retained.
    calls=[{key:call[key] for key in ("skill","arguments","roles")} for call in plan["calls"]]
    return digest(dict(calls=calls,order=plan["prefix"]["order"],
                       choices=plan["prefix"]["choices"],paths=canonical_paths(plan)))


def failure_call(trial):
    part=None
    for step in trial.get("executed_parameters") or []:
        part=step.get("params",{}).get("part",part)
        if not step["ok"]:
            return step["skill"],part
    error=trial.get("error","")
    return ("independent_goal" if error=="independent task goal not satisfied" else error.split(":")[0] or "unrecorded_failure"),part


def rates(rows,factor):
    groups=defaultdict(list)
    for row in rows:groups[row[factor]].append(row["success"])
    return [dict(value=key,successes=sum(values),trials=len(values),rate=statistics.mean(values))
            for key,values in sorted(groups.items(),key=lambda pair:str(pair[0]))]


def audit(roots):
    groups=[];failures=[];incomplete=[];lineage=[];seen=set()
    for root in roots:
        for directory in sorted(Path(root).glob("group_*")):
            files=[directory/name for name in ("inputs.json","outcomes.json","complete.json")]
            if not all(path.exists() for path in files):
                failure=directory/"failure.json"
                if failure.exists():
                    saved=json.loads(failure.read_text());request=saved.get("request",{})
                    failures.append(dict(path=str(failure),split=request.get("split"),seed=request.get("seed"),
                        checkpoint=request.get("checkpoint"),error=saved.get("error",""),
                        sha256=hashlib.sha256(failure.read_bytes()).hexdigest()))
                else:incomplete.append(str(directory))
                continue
            inp,out,complete=[json.loads(path.read_text()) for path in files]
            group_id=inp["group_id"]
            if group_id in seen:raise ValueError("duplicate decision group")
            seen.add(group_id)
            ih=digest(inp)
            if out["input_sha256"]!=ih or complete["input_sha256"]!=ih:
                raise ValueError("input/label binding mismatch")
            if len(out["trials"])!=complete["trials"]:
                raise ValueError("incomplete physical trial count")
            groups.append((inp,out))
            lineage.append(dict(group_id=group_id,seed=inp["task"]["seed"],split=inp["declared_split"],
                checkpoint=inp["checkpoint"],input_sha256=ih,collection_source_sha256=inp["source_sha256"],
                files={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in files}))
    rows=[];group_rows=[];values=defaultdict(set);shared=Counter();waypoints=Counter();lengths=Counter()
    for inp,out in groups:
        plans={p["id"]:p for p in inp["candidates"]}
        if len(plans)!=len(inp["candidates"]):raise ValueError("duplicate candidate ID")
        repeats=defaultdict(dict);trial_hashes={}
        for trial in out["trials"]:
            cid=trial["candidate_id"];repeat=trial["trial"]["repeat"]
            if cid not in plans or repeat in repeats[cid] or type(trial["success"]) is not bool or trial.get("valid") is not True:
                raise ValueError("invalid or duplicate outcome")
            th=digest(trial["trial"])
            if th!=trial.get("trial_sha256") or repeat in trial_hashes and th!=trial_hashes[repeat]:
                raise ValueError("unpaired disturbances")
            trial_hashes[repeat]=th;repeats[cid][repeat]=trial["success"]
            plan=plans[cid];choices=plan["prefix"]["choices"];first=plan["prefix"]["order"][0]
            paths=plan["prefix"].get("initial_artifacts",{})
            if len(paths)!=1:raise ValueError("audit expects the declared single initial joint trajectory")
            path=next(iter(paths.values()));pins=[v["yaw"] for part,v in choices.items() if part.startswith("pin_")]
            if not pins:raise ValueError("audit expects a remaining pin manipulation")
            call,part=failure_call(trial) if not trial["success"] else (None,None)
            rows.append(dict(split=inp["declared_split"],seed=inp["task"]["seed"],checkpoint=inp["checkpoint"],
                success=trial["success"],prefix_success=trial["prefix_success"],timeout=trial["timeout"],
                force=choices[first]["force"],speed=choices[first]["speed"],clearance=choices[first]["clearance"],
                initial_route=path["id"],first_part=first,first_yaw=choices[first]["yaw"],first_height=choices[first]["height"],
                all_remaining_pins_yaw90=all(abs(y-math.pi/2)<1e-6 for y in pins),
                failed_call=call,failed_part=part,handle_reached=any(
                    step["skill"]=="close_gripper" and step.get("params",{}).get("part")=="handle"
                    for step in trial.get("executed_parameters") or [])))
        for cid in plans:
            if set(repeats[cid])!=set(trial_hashes):raise ValueError("missing candidate repetitions")
        ys=[statistics.mean(repeats[cid].values()) for cid in plans]
        keys=[executable_key(plan) for plan in plans.values()]
        trajectories={digest(canonical_paths(plan)) for plan in plans.values()}
        orders={tuple(plan["prefix"]["order"]) for plan in plans.values()}
        group_rows.append(dict(group_id=inp["group_id"],split=inp["declared_split"],seed=inp["task"]["seed"],
            checkpoint=inp["checkpoint"],candidates=len(plans),trials=len(out["trials"]),repeats=len(trial_hashes),
            unique_executable_plans=len(set(keys)),duplicate_executable_plans=len(keys)-len(set(keys)),
            unique_initial_trajectories=len(trajectories),structure_branches=len(orders),
            initial_grasp_branches=inp["pool_counts"]["grasp_branches"],
            pool_type="all_failure" if max(ys)==0 else "all_success" if min(ys)==1 else "mixed",
            empirical_label_counts=dict(Counter(str(y) for y in ys)),pool_counts={key:inp["pool_counts"].get(key,0)
                for key in ("raw_count","known_conflict","materialization_unknown","deduplicated","materializable")}))
        for plan in plans.values():
            choices=plan["prefix"]["choices"]
            for key in ("force","clearance","speed"):
                shared[key]+=len({v[key] for v in choices.values()})==1
            for part,choice in choices.items():
                for key,value in choice.items():values[(part,key)].add(value)
            for path in plan["prefix"]["initial_artifacts"].values():waypoints[str(len(path["joints"]))]+=1
            lengths[str(len(plan["calls"]))]+=1
    splits={}
    factors=("first_yaw","all_remaining_pins_yaw90","force","speed","clearance","initial_route","first_height")
    for split in ("train","val","test","all"):
        selected=[r for r in rows if split=="all" or r["split"]==split]
        selected_groups=[g for g in group_rows if split=="all" or g["split"]==split]
        failed=[f for f in failures if split=="all" or f["split"]==split]
        failures_only=[r for r in selected if not r["success"]]
        splits[split]=dict(reached_groups=len(selected_groups),failed_setup_groups=len(failed),
            requested_groups=len(selected_groups)+len(failed),candidates=sum(g["candidates"] for g in selected_groups),
            trials=len(selected),successes=sum(r["success"] for r in selected),failures=len(failures_only),
            prefix_successes=sum(r["prefix_success"] for r in selected),timeouts=sum(r["timeout"] for r in selected),
            failed_calls=dict(Counter(r["failed_call"] for r in failures_only)),
            failed_call_parts=[dict(skill=key[0],part=key[1],count=count) for key,count in
                Counter((r["failed_call"],r["failed_part"]) for r in failures_only).most_common()],
            handle_grasp_reached_trials=sum(r["handle_reached"] for r in selected),
            pool_types=dict(Counter(g["pool_type"] for g in selected_groups)),
            marginals={factor:rates(selected,factor) for factor in factors})
    return dict(schema="twingraph.value.candidate_audit.v1",splits=splits,groups=group_rows,setup_failures=failures,
        incomplete_groups=incomplete,source_lineage=lineage,
        diversity=dict(plan_lengths=lengths,initial_joint_waypoint_counts=waypoints,
            shared_control_across_all_remaining_parts=dict(shared),
            parameter_values=[dict(part=part,parameter=key,values=sorted(value)) for (part,key),value in sorted(values.items())]),
        checkpoint_marginals={str(cp):{factor:rates([r for r in rows if r["checkpoint"]==cp],factor)
            for factor in factors} for cp in sorted({r["checkpoint"] for r in rows})},
        notes=["Read-only descriptive audit of existing inputs and physical outcomes; no model fitting, tuning, new labels or simulations.",
            "Each candidate has paired repetitions and candidates share configurations; pooled trial fractions are associations, not causal effects or independent statistical samples.",
            "Force, speed and clearance marginals use the first manipulation's command; shared_control_across_all_remaining_parts reports when those same values are reused by every part.",
            "A failure is attributed to the first failed recorded skill; direct IK errors can have no failed skill record. Part is the most recent recorded explicit part parameter.",
            "First-yaw and all-pin-yaw associations are retrospective dataset diagnostics, not permission to tune on test or silently filter candidates.",
            "Materialization-unknown attempts are unresolved solver attempts, not physical failures; they remain in generator counts."])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",nargs="+",required=True);parser.add_argument("--out",required=True)
    args=parser.parse_args();result=audit(args.data)
    result["audit_source_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(dict(out=str(path),overall={key:result["splits"]["all"][key]
        for key in ("requested_groups","reached_groups","failed_setup_groups","candidates","trials","successes","failures")})))


if __name__=="__main__":main()
