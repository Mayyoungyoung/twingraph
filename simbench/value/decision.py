"""Live integration: generate, rank, validate from snapshots, execute winner."""
import argparse
import contextlib
from pathlib import Path
import time
from .collect import dump
from .scenarios import TaskSpec,make_task,build_plans,observation,render_observation
from .plan import execute_prefix,execute_suffix,PlanIR,PROTOCOL
from .rank import PlanRanker,select_and_validate
from simbench.assembly.library import SkillFailure


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--out",default="results/value/decision")
    p.add_argument("--seed",type=int,default=9000)
    p.add_argument("--k",type=int,default=2)
    p.add_argument("--budget",type=int,default=2)
    p.add_argument("--device",default="cuda")
    p.add_argument("--execute",action="store_true",help="execute validated plan in this simulation")
    a=p.parse_args()
    out=Path(a.out)
    out.mkdir(parents=True,exist_ok=True)
    ranker=PlanRanker(a.checkpoint,a.device)
    started=time.perf_counter()
    with (out/"steps.log").open("w") as log,contextlib.redirect_stdout(log):
        spec=TaskSpec.sample(a.seed)
        session,_=make_task(spec,out)
        plans=build_plans(session,spec)
        obs=observation(session,spec)
        visual=None
        images=[]
        if ranker.model.config.vision:
            from .vision import FrozenVision
            images=render_observation(session,out)
            visual=FrozenVision(a.device).encode([out/f for f in images])
        dump(out/"inputs.json",dict(observation=obs,candidates=[p.to_dict() for p in plans],images=images,protocol=PROTOCOL))
        ranking=ranker.rank(obs,plans,a.k,visual)
        dump(out/"top_k.json",ranking)
        initial=session.snapshot()
        failures=[]
        def validate(plan):
            session.restore(initial)
            try:
                execute_prefix(session,plan)
                execute_suffix(session,plan)
                return True
            except SkillFailure as exc:
                failures.append(dict(candidate_id=plan.id,error=str(exc)))
                return False
            finally:
                session.restore(initial)
        result=select_and_validate(ranking,validate,a.budget)
        result["failures"]=failures
        result["execution_success"]=None
        if a.execute and result["chosen"]:
            plan=PlanIR.from_dict(result["chosen"])
            try:
                execute_prefix(session,plan)
                execute_suffix(session,plan)
                result["execution_success"]=True
            except SkillFailure as exc:
                result["execution_success"]=False
                result["execution_error"]=str(exc)
        result["total_wall_seconds"]=time.perf_counter()-started
        dump(out/"decision.json",result)
    print(f"{result['status']}; validations={result['validation_calls']}; execution={result['execution_success']}")


if __name__=="__main__":
    main()
