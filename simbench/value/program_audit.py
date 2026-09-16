"""Cheap symbolic contract audit; unknown contact/path facts stay unknown."""
import inspect
from simbench.assembly.library import Session,HANDLERS
from simbench.assembly.interfaces import resolve
from simbench.assembly.contracts import State,check,apply_effects
from .plan import initial_artifacts


def audit_program(plan,objects):
    plan.validate(objects)
    if plan.prefix.get("execution")!="program":
        return dict(status="legacy_bound_prefix",unknown=["physical suffix feasibility"])
    state=State(parts=tuple(objects),grasp_epoch=plan.prefix.get("initial_grasp_epoch",0),
                artifacts=initial_artifacts(plan));unknown=[]
    producers={c.id:(i,c) for i,c in enumerate(plan.calls)}
    last_grasp=-1
    for i,call in enumerate(plan.calls):
        for arg in call.arguments.values():
            if arg.status!="deferred":continue
            pi,producer=producers[arg.source_call]
            if producer.roles.get("manipulated")!=call.roles.get("manipulated"):
                raise ValueError("deferred producer is bound to another object")
            if pi<last_grasp:
                raise ValueError("deferred producer belongs to an earlier grasp acquisition")
            if arg.source_output=="grasp_hover" and producer.skill!="select_grasp":
                raise ValueError("grasp hover requires a selected grasp producer")
        if call.skill=="grasp":last_grasp=i
        args={k:a.value for k,a in call.arguments.items()}
        name,params=resolve(call.skill,args)
        if name not in HANDLERS:raise ValueError(f"unknown implementation {name}")
        bound=inspect.signature(getattr(Session,name)).bind(None,**params);bound.apply_defaults()
        params=dict(bound.arguments);params.pop("self")
        verdict=check(HANDLERS[name],params,state,contact=None)
        if verdict["conflicts"]:raise ValueError(f"call {i}: {'; '.join(verdict['conflicts'])}")
        unknown.extend(dict(call=i,obligation=x) for x in verdict["unknown"])
        apply_effects(HANDLERS[name],params,state,step=i,symbolic=True)
    return dict(status="no_known_contract_conflict",unknown=unknown,
                interpretation="symbolic obligations are not execution success proofs")
