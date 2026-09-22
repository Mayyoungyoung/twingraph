"""Declared physical-fixture perturbation followed by the unmodified V12 loop.

Default is preflight only. --run must be supplied to start physics. The source
must be a successful full rollout from the same frozen runtime. The fixture
operator changes one free body's position; only a fresh RGB-D observation is
passed to ClosedLoop. This is a simulator intervention, never hardware evidence.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np

FROZEN_R2_SHA = "ca9628612e85ccfc02e09ad0fd7f2f021de322c7a07060f96a1a5349db22a911"


def validate_request(source, stage, part, delta, *, runtime_sha, n, k, settle):
    if not source.get("valid") or not source.get("success"):
        raise ValueError("source must be a valid successful complete rollout")
    if source.get("runtime_sha256") != runtime_sha:
        raise ValueError("source and requested frozen runtime differ")
    if source.get("geometry_version") != "printed_functional_assembly_v12":
        raise ValueError("source is not the printed V12 scene")
    required=("cleaning_pass","assembly_pass","functional_test_pass",
              "fixture_capture_pass","final_release_and_retraction_pass")
    if not all(source.get("stage_passes",{}).get(key) for key in required):
        raise ValueError("source lacks all complete-task predicates")
    matches=[row for row in source.get("boundaries",[]) if row["stage"] == stage]
    if len(matches)!=1 or matches[0].get("held") is not None:
        raise ValueError("intervention must name one successful released boundary")
    if stage not in matches[0]["completed"] or stage in ("finished","stroke","retention"):
        raise ValueError("intervention is not an assembly release boundary")
    if part not in source["proposal"]["order"] or part in matches[0]["completed"]:
        raise ValueError("disturbed body must be a pending assembly part")
    # Loose supplied components only. Moving a pin alone would invalidate its
    # physical holder setup and test a different intervention.
    if part not in ("handle","carriage","end_stop"):
        raise ValueError("position intervention only supports loose supplied components")
    delta=np.asarray(delta,float)
    if delta.shape!=(3,) or not np.isfinite(delta).all() or delta[2]!=0:
        raise ValueError("use a finite horizontal displacement; no height edit")
    if not .001 <= np.linalg.norm(delta) <= .03:
        raise ValueError("fixture displacement must be between 1 and 30 mm")
    if not 1 <= k <= n <= 128 or not 0 <= settle <= 1:
        raise ValueError("invalid bounded search budget or settling interval")
    return deepcopy(matches[0])


def prefix_preserved(current, replacement, completed):
    done=[part for part in current["order"] if part in completed]
    return (all(replacement["choices"].get(part)==current["choices"][part] for part in done)
            and replacement["order"][:len(done)]==done)


def apply_fixture_shift(session, part, delta):
    """Privileged experiment fixture only; never called by the detector."""
    import mujoco
    ctx=session.ctx; model,data=ctx.model,ctx.data
    if session.held is not None:
        raise ValueError("cannot perturb a held state")
    body=ctx.body_id(part)
    if model.body_jntnum[body]!=1:
        raise ValueError("fixture intervention requires exactly one free joint")
    joint=int(model.body_jntadr[body])
    if int(model.jnt_type[joint])!=int(mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError("fixture intervention cannot move fixed geometry")
    address=int(model.jnt_qposadr[joint]); before=data.qpos.copy(); velocity=data.qvel.copy()
    data.qpos[address:address+3]+=np.asarray(delta,float)
    mujoco.mj_forward(model,data)
    changed=np.flatnonzero(np.abs(data.qpos-before)>1e-12).tolist()
    if any(i not in range(address,address+3) for i in changed):
        raise RuntimeError("fixture intervention modified unrelated generalized positions")
    if not np.array_equal(data.qvel,velocity):
        raise RuntimeError("fixture intervention changed velocity")
    return dict(kind="privileged_simulator_fixture_position_intervention",part=part,
        delta_world_m=list(map(float,delta)),free_joint_position_before_m=before[address:address+3].tolist(),
        free_joint_position_after_m=data.qpos[address:address+3].tolist(),changed_qpos_indices=changed,
        velocity_modified=False,geometry_modified=False,observation_modified=False,
        truth_usage="operator audit only; never copied into RGB-D observations or candidate features")


class DisturbedReleaseMonitor:
    def __init__(self, loop, directory, stage, part, delta, settle):
        self.loop=loop; self.directory=Path(directory); self.stage=stage; self.part=part
        self.delta=np.asarray(delta,float); self.settle=settle; self.injected=False
        self.audit=[]; self.intervention=None; self.injection_step=None; self.completed_at_injection=[]

    def __call__(self, session, actual, current):
        from simbench.value.system_v11 import observe_boundary,save
        if not self.injected and actual["stage"] == self.stage:
            if self.part in actual["completed"] or actual["held"] is not None:
                raise ValueError("runtime release/pending state differs from preflight")
            before=deepcopy(actual)
            self.injection_step=len(session.results)
            self.completed_at_injection=list(actual["completed"])
            self.intervention=apply_fixture_shift(session,self.part,self.delta)
            self.intervention.update(stage=self.stage,completed=self.completed_at_injection,
                settling_seconds=self.settle,physics_time_before_settle_s=float(session.ctx.data.time))
            save(self.directory/"fixture_intervention.json",self.intervention)
            save(self.directory/"observation_before_intervention.json",before)
            session.hold(self.settle)
            # Re-render and call the existing CAD RGB-D detector. Never set an
            # object's observed position to the known intervention coordinates.
            fresh=observe_boundary(session,self.stage,self.completed_at_injection)
            if fresh["observation"].get("backend")!="rgbd_geometry":
                raise RuntimeError("post-disturbance observation did not use RGB-D")
            actual.clear(); actual.update(fresh)  # Replace with the genuine detector output.
            save(self.directory/"observation_after_intervention.json",actual)
            self.injected=True
        row=dict(stage=actual["stage"],completed=list(actual["completed"]),
                 executed_step_count=len(session.results),observation_sha256=actual["observation"].get("sha256"),
                 intervention_applied=self.injected)
        self.audit.append(row)
        try:
            replacement=self.loop(session,actual,current)
            if replacement is not None:
                row["completed_prefix_preserved"]=prefix_preserved(current,replacement,actual["completed"])
                if not row["completed_prefix_preserved"]:
                    raise RuntimeError("current ClosedLoop rewrote a completed prefix")
            return replacement
        finally:
            save(self.directory/"monitor_audit.json",self.audit)


def summarize(result, monitor, loop, directory):
    directory=Path(directory); completed=set(monitor.completed_at_injection)
    def replay_audit(steps):
        acquisitions=Counter(row.get("params",{}).get("part") for row in (steps or [])
                             if row.get("skill")=="close_gripper")
        repeats={part:acquisitions[part] for part in completed if part not in ("cleaning","handle") and acquisitions[part]}
        # An installed handle is normally regrasped once for the final stroke.
        if "handle" in completed and acquisitions["handle"]>1: repeats["handle"]=acquisitions["handle"]-1
        if "cleaning" in completed and any(row.get("skill")=="wipe_surface" for row in (steps or [])):
            repeats["cleaning"]=1
        return dict(acquisition_counts=dict(acquisitions),repeated_completed_skills=repeats,
                    installed_handle_functional_stroke_regrasp_allowance=int("handle" in completed))
    rows=[]
    for path in sorted((directory/"closed_loop").glob("replan_*/*/result.json")):
        trial=json.loads(path.read_text())
        stages=[r["stage"] for r in trial.get("boundaries",[]) if r["stage"]!="finished"]
        rows.append(dict(result=str(path.relative_to(directory)),valid=trial.get("valid"),success=trial["success"],
            stages=stages,repeated_completed_assembly_stages=sorted(completed.intersection(stages)),
            checkpoint_sync=trial.get("checkpoint_sync"),trial=trial.get("trial"),
            execution_audit=replay_audit(trial.get("executed_parameters"))))
    stages=[r["stage"] for r in result.get("boundaries",[]) if not r["stage"].startswith("failed_")]
    duplicates=[stage for stage,count in Counter(stages).items() if count>1]
    actions=[row.get("action") for row in loop.events]
    requested=next((e for e in loop.events if e["stage"]==monitor.stage),{})
    deployment_audit=replay_audit((result.get("executed_parameters") or [])[monitor.injection_step or 0:])
    no_replay=bool(not duplicates and not deployment_audit["repeated_completed_skills"]
        and all(not row["repeated_completed_assembly_stages"] and not row["execution_audit"]["repeated_completed_skills"] for row in rows))
    return dict(schema="twingraph.v12.physical_replanning_audit.v1",
        overall_success=result["success"],error=result.get("error"),valid=result.get("valid"),
        intervention_applied=monitor.injected,completed_at_intervention=sorted(completed),
        trigger_at_intervention=requested,events=loop.events,physical_suffix_trials=rows,
        verified_suffix_count=actions.count("verified_remaining_suffix"),
        repeated_deployment_stage_boundaries=duplicates,
        completed_choices_preserved=all(row.get("completed_prefix_preserved",True) for row in monitor.audit),
        deployment_execution_audit=deployment_audit,no_completed_stage_reexecuted=no_replay,
        closed_loop_recovery_demonstrated=bool(result["success"] and monitor.injected
            and requested.get("action")=="verified_remaining_suffix" and no_replay),
        runtime_sha256=result.get("runtime_sha256"),
        limitations=["Explicit simulator position intervention, not a hardware disturbance experiment",
            "Replanning twins use privileged exact simulator checkpoint synchronization",
            "Current ClosedLoop retains its online trial perturbation on suffix rollouts; see each raw trial parameters",
            "A single controlled trial cannot establish a recovery success rate or benefit over open loop"])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True); parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--runtime-sha",default=FROZEN_R2_SHA)
    parser.add_argument("--stage",default="end_stop"); parser.add_argument("--part",default="handle")
    parser.add_argument("--delta",type=float,nargs=3,default=[.020,0.,0.]); parser.add_argument("--settle",type=float,default=.25)
    parser.add_argument("--n",type=int,default=48); parser.add_argument("--k",type=int,default=4)
    parser.add_argument("--method",choices=["random_top_k","value_top_k","all_twin"],default="random_top_k")
    parser.add_argument("--checkpoint",type=Path); parser.add_argument("--level",choices=["L0","L1","L2"],default="L1")
    parser.add_argument("--record",action="store_true"); parser.add_argument("--run",action="store_true")
    args=parser.parse_args()
    source=json.loads(args.source.read_text())
    validate_request(source,args.stage,args.part,args.delta,runtime_sha=args.runtime_sha,n=args.n,k=args.k,settle=args.settle)
    if args.method=="value_top_k" and not args.checkpoint: parser.error("value_top_k requires --checkpoint")
    from simbench.value.provenance_v12 import fingerprint
    if fingerprint()["sha256"]!=args.runtime_sha: raise ValueError("running source differs from frozen runtime")
    if args.out.exists() and any(args.out.iterdir()): raise FileExistsError("refusing to overwrite experiment evidence")
    from simbench.value.system_v11 import save
    protocol=dict(source_result=str(args.source),source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        runtime_sha256=args.runtime_sha,seed=source["seed"],proposal=source["proposal"],domain=source["domain"],
        level=args.level,stage=args.stage,part=args.part,delta_world_m=args.delta,settle_s=args.settle,
        method=args.method,n=args.n,k=args.k,max_replans=1,record=args.record,
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None,
        preflight_only=not args.run,physical_fixture_oracle=True,truth_in_detector_input=False)
    save(args.out/"protocol.json",protocol)
    if not args.run:
        print(json.dumps(dict(preflight="passed",physics_started=False,protocol=str(args.out/"protocol.json")))); return
    from simbench.value.system_v12 import ClosedLoop,rollout
    value=None
    if args.checkpoint:
        from simbench.value.graph_value_v12 import ValueRankerV12
        value=ValueRankerV12(args.checkpoint)
    loop=ClosedLoop(source["seed"],deepcopy(source["boundaries"]),value,args.out/"closed_loop",
        max_replans=1,k=args.k,n=args.n,level=args.level,method=args.method)
    monitor=DisturbedReleaseMonitor(loop,args.out,args.stage,args.part,args.delta,args.settle)
    result=rollout(source["seed"],source["proposal"],args.out/"execution",domain=source["domain"],
                   level=args.level,monitor=monitor,record=args.record)
    summary=summarize(result,monitor,loop,args.out)
    save(args.out/"summary.json",summary)
    print(json.dumps(summary,ensure_ascii=False))


if __name__=="__main__": main()
