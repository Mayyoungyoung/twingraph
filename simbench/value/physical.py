"""Independent, paired perturbation namespaces and measured physical execution."""
import copy
import time
import numpy as np
from .plan import digest, execute_prefix, execute_suffix
from simbench.assembly.candidates import fingerprint
from simbench.assembly.library import SkillFailure

DOMAINS={"train":731,"online":1753,"deployment":2909,"reference":4001,"regression":0}


def perturbation(seed, repeat, domain):
    rng=np.random.default_rng(np.random.SeedSequence([seed,repeat,DOMAINS[domain]]))
    return dict(domain=domain,repeat=int(repeat),
                friction_scale=1. if domain=="regression" else float(rng.uniform(.85,1.15)),
                actuator_gain_scale=1. if domain=="regression" else float(rng.uniform(.985,1.015)))


class PhysicalRunner:
    def __init__(self,session,timeout=90.):
        self.session=session; self.snapshot=session.snapshot(); self.initial=fingerprint(session)
        self.gain=session.ctx.model.actuator_gainprm.copy();self.bias=session.ctx.model.actuator_biasprm.copy()
        self.timeout=timeout

    def run(self,plan,trial,keep_trace=False):
        s=self.session; t=time.perf_counter(); s.restore(self.snapshot)
        s.ctx.model.actuator_gainprm[:]=self.gain;s.ctx.model.actuator_biasprm[:]=self.bias
        restore=time.perf_counter()-t
        if fingerprint(s)!=self.initial or plan.prefix["start_state"]!=self.initial:
            raise ValueError("trial initial-state or executable binding mismatch")
        s.ctx.model.geom_friction[:]*=trial["friction_scale"]
        s.ctx.model.actuator_gainprm[:,:3]*=trial["actuator_gain_scale"]
        s.ctx.model.actuator_biasprm[:,:3]*=trial["actuator_gain_scale"]
        p=copy.deepcopy(plan); p.prefix["start_state"]=fingerprint(s)
        s.results.clear(); s.arm.trace.clear()
        start_sim=float(s.ctx.data.time);wall=time.perf_counter()
        prefix=False;suffix=None;error="";timeout=False
        # Deadline is checked at control-step boundaries; no background rollouts.
        original_step=s.ctx.step
        def bounded_step(*args,**kw):
            if time.perf_counter()-wall>self.timeout:
                raise TimeoutError("physical rollout wall-time limit")
            return original_step(*args,**kw)
        s.ctx.step=bounded_step
        try:
            execute_prefix(s,p);prefix=True
            execute_suffix(s,p);suffix=True
        except TimeoutError as exc:
            error=str(exc);timeout=True
            if prefix:suffix=False
        except SkillFailure as exc:
            # Dispatch errors are programming errors, not empirical task failures.
            if not s.results or any(x in str(exc) for x in ("unexpected keyword","unknown skill","invalid parameter choice","missing a required argument")):
                raise RuntimeError(f"invalid program: {exc}") from exc
            error=str(exc)
            if prefix:suffix=False
        except ValueError as exc:
            if not str(exc).startswith("IK unreachable:"): raise
            error=str(exc)
            if prefix:suffix=False
        finally:
            s.ctx.step=original_step
            s.ctx.model.actuator_gainprm[:]=self.gain;s.ctx.model.actuator_biasprm[:]=self.bias
        elapsed=time.perf_counter()-wall
        steps=copy.deepcopy(s.results)
        sim=float(s.ctx.data.time-start_sim)
        result=dict(candidate_id=plan.id,trial=trial,trial_sha256=digest(trial),valid=True,
                    success=bool(prefix and suffix),full_success=bool(prefix and suffix),
                    prefix_success=prefix,suffix_success=suffix,error=error,timeout=timeout,
                    restore_seconds=restore,wall_seconds=elapsed,sim_seconds=sim,
                    physics_steps=int(round(sim/s.ctx.model.opt.timestep)),executed_steps=len(steps),
                    deferred_solving_seconds=sum(x["wall_seconds"] for x in steps if x["skill"] in {"plan_transfer","propose_grasps","select_grasp","estimate_pose"}),
                    final_positions={n:s.ctx.obj_pos(n).tolist() for n in s.parts},
                    executed_parameters=[dict(skill=x["skill"],params=x["params"],ok=x["ok"],metrics=x["metrics"],sim_seconds=x["sim_seconds"]) for x in steps] if keep_trace else None)
        return result
