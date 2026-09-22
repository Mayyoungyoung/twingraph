"""Controlled integration test of prediction-error-triggered suffix replanning.

Injects a declared 0.2 rad bias into the stored twin joint prediction only.
No physical state, sensor observation, acceptance threshold or label is edited.
This tests execution of the replan path, not recovery benefit under real faults.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from simbench.value.system_v11 import ClosedLoop,FrozenValue,rollout,save


def main():
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();source=json.loads(a.source.read_text());proposal=source["proposal"]
    expected=deepcopy(source["boundaries"])
    boundary=next(r for r in expected if r["stage"]=="end_stop")
    boundary["robot_joints"][0]+=.2
    save(a.out / "intervention.json",dict(kind="stored_twin_prediction_bias",stage="end_stop",
        field="robot_joints[0]",bias_rad=.2,physical_state_modified=False,sensor_modified=False,
        purpose="exercise actual independent suffix verification and resume without replaying completed actions",
        limitation="not evidence that replanning improves success under physical disturbances"))
    seed=1510
    value=FrozenValue(a.checkpoint)
    monitor=ClosedLoop(seed,expected,value,a.out / "closed_loop")
    result=rollout(seed,proposal,a.out / "execution",domain="online",monitor=monitor)
    stages=[r["stage"] for r in result["boundaries"]]
    summary=dict(success=result["success"],error=result["error"],events=monitor.events,
                 completed_stages=stages,no_repeated_completed_stages=len(stages)==len(set(stages)),
                 successful_replans=sum(e["action"]=="replan_verified_remaining_suffix" for e in monitor.events),
                 physical_verification_runs=len(list((a.out / "closed_loop").glob("replan_*/*/result.json"))))
    save(a.out / "summary.json",summary)
    print(json.dumps(summary,ensure_ascii=False))


if __name__=="__main__":main()
