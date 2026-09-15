"""v2 families and real, parameterized executable programs.

The connector is a passive rigid housing with two rectangular keyed sockets,
two separately manipulated inserts, a guard bar and a locating pin. Housing
is already fixtured. No loose body is moved except by physics/robot controls.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import copy
import itertools
import math
import time
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from .plan import PlanIR, Call, Argument, argument, digest, plain
from .scenarios import TaskSpec, make_task, render_observation
from simbench.assembly.scene import geom, fmt, holed_plate
from simbench.assembly.demo_scenes import build_demo_scene
from simbench.assembly.library import Session
from simbench.assembly.control import HOME, down
from simbench.assembly.candidates import fingerprint, transfer_routes, release_space_proxy
from simbench.core.sim_context import MjContext

PROGRAM_PROTOCOL = "assembly.program.feedback.v2"


@dataclass
class ConnectorSpec:
    seed: int
    center: list
    spacing: float
    clearance: float
    depth: float
    key_yaws: list
    source_shift: list
    guard_offset: float
    family: str = "rigid_connector_module"

    @classmethod
    def sample(cls, seed):
        r = np.random.default_rng(seed)
        return cls(seed, r.uniform([-.02,.005],[.035,.05]).tolist(),
                   float(r.uniform(.065,.095)), float(r.uniform(.0008,.0014)),
                   float(r.uniform(.020,.026)),
                   [float(r.choice([0,math.pi/2])) for _ in range(2)],
                   r.uniform([-.012,-.012],[.012,.012]).tolist(),
                   float(r.uniform(.066,.079)))

    @property
    def config_id(self):
        row = asdict(self); row.pop("seed")
        return digest(row)[:20]


def rect_socket(parent, name, xy, half_hole, half_outer, top, depth, color):
    """Four solid walls; dimensions are linked to the matching stem."""
    hx,hy=half_hole; ox,oy=half_outer
    for sign in (-1,1):
        geom(parent,f"{name}_x{sign}",[xy[0]+sign*(hx+ox)/2,xy[1],top-depth/2],
             [(ox-hx)/2,oy,depth/2],color)
        geom(parent,f"{name}_y{sign}",[xy[0],xy[1]+sign*(hy+oy)/2,top-depth/2],
             [hx,(oy-hy)/2,depth/2],color)


def make_connector(spec, directory):
    path,_,_=build_demo_scene("cube",directory)
    root=ET.parse(path).getroot(); world=root.find("worldbody")
    world.remove(world.find("body[@name='cube']"))
    root.set("model",spec.family)
    cx,cy=spec.center; top=.854
    housing=ET.SubElement(world,"body",name="housing",pos="0 0 0")
    geom(housing,"housing_support",[cx,cy,.806],[spec.spacing/2+.033,.10,.006],".3 .36 .45 1")
    targets={}; specs={}; sources=[[-.245,-.22],[-.125,-.23],[-.23,-.065],[-.12,-.08]]
    for i in range(2):
        name=f"insert_{i}"; xy=[cx+(i-.5)*spec.spacing,cy]
        targets[name]=[*xy,top+.001]
        dims=np.array([.006,.004]); yaw=spec.key_yaws[i]
        socket_dims=dims if yaw==0 else dims[::-1]
        rect_socket(housing,f"socket_{i}",xy,socket_dims+spec.clearance,
                    [.024,.026],top,spec.depth+.009,".4 .46 .54 1")
        sx,sy=np.array(sources[i])+spec.source_shift
        b=ET.SubElement(world,"body",name=name,pos=fmt([sx,sy,.851]),
                        quat=fmt([math.cos(yaw/2),0,0,math.sin(yaw/2)]))
        ET.SubElement(b,"freejoint",name=name+"_free")
        geom(b,name+"_key",[0,0,-spec.depth/2-.001],[*dims,spec.depth/2],".75 .64 .35 1")
        geom(b,name+"_head",[0,0,.012],[.011,.011,.013],".28 .50 .70 1",friction="1 .02 .001")
        holder=ET.SubElement(world,"body",name=name+"_holder")
        rect_socket(holder,name+"_supply",[sx,sy],socket_dims+.0015,[.024,.024],
                    .850,spec.depth+.005,".43 .45 .48 1")
        specs[name]=(.013,.022)
    # Guard bridges the rear of the two sockets, on a passive locating ledge.
    gy=cy+spec.guard_offset
    guard=ET.SubElement(world,"body",name="guard",pos=fmt([* (np.array(sources[2])+spec.source_shift),.808]))
    ET.SubElement(guard,"freejoint",name="guard_free")
    geom(guard,"guard_bar",[0,0,0],[spec.spacing/2+.022,.012,.008],".65 .40 .21 1")
    geom(guard,"guard_grasp",[0,0,.020],[.010,.010,.012],".72 .45 .24 1",friction="1 .02 .001")
    geom(housing,"guard_ledge",[cx,gy,.834],[spec.spacing/2+.025,.014,.020],".4 .46 .54 1")
    targets["guard"]=[cx,gy,.862]; specs["guard"]=(.020,.020)
    # The pin engages a separate keyed flange beside the guard.
    px,py=cx+spec.spacing/2+.052,gy
    rect_socket(housing,"pin_socket",[px,py],[.004+spec.clearance]*2,
                [.013,.017],top,.030,".4 .46 .54 1")
    sx,sy=np.array(sources[3])+spec.source_shift
    pin=ET.SubElement(world,"body",name="locator",pos=fmt([sx,sy,.851]))
    ET.SubElement(pin,"freejoint",name="locator_free")
    geom(pin,"locator_stem",[0,0,-.013],[.0035,.0035,.012],".7 .72 .76 1")
    geom(pin,"locator_head",[0,0,.005],[.009,.009,.006],".7 .72 .76 1",friction="1 .02 .001")
    holder=ET.SubElement(world,"body",name="locator_holder")
    rect_socket(holder,"locator_supply",[sx,sy],[.005,.005],[.018,.018],.850,.029,".43 .45 .48 1")
    specs["locator"]=(.006,.018); targets["locator"]=[px,py,top+.001]
    ET.ElementTree(root).write(path,encoding="unicode")
    ctx=MjContext(path,control_freq=50); ctx.reset()
    ctx.data.qpos[ctx.arm_qadr]=HOME; mujoco.mj_forward(ctx.model,ctx.data)
    ctx.hold_arm(); ctx.set_finger_ctrl(.04)
    for _ in range(80): ctx.step()
    return Session(ctx,seed=spec.seed,parts=tuple(specs),grasp_specs=specs),path,targets


def make_family(family, seed, directory):
    if family=="sliding_stage_pin":
        spec=TaskSpec.sample(seed); session,path=make_task(spec,directory)
        targets={"pin_left":spec.target.tolist()}
    elif family=="rigid_connector_module":
        spec=ConnectorSpec.sample(seed); session,path,targets=make_connector(spec,directory)
    else: raise ValueError(family)
    return spec,session,path,targets


def observed(session, targets):
    ctx=session.ctx
    names=list(session.parts)+[n for n in ("receiver","housing")
                                if mujoco.mj_name2id(ctx.model,mujoco.mjtObj.mjOBJ_BODY,n)>=0]
    objects={}
    for name in names:
        bid=ctx.body_id(name); idx=np.where(ctx.model.geom_bodyid==bid)[0]
        objects[name]=dict(position=ctx.obj_pos(name).tolist(),quaternion=ctx.data.xquat[bid].tolist(),
                           geoms=[dict(type=int(ctx.model.geom_type[i]),size=ctx.model.geom_size[i].tolist(),
                                       position=ctx.model.geom_pos[i].tolist(),quaternion=ctx.model.geom_quat[i].tolist()) for i in idx])
    return dict(robot=dict(joints=ctx.arm_qpos.tolist(),fingers=ctx.finger_qpos.tolist(),eef=ctx.eef_pos().tolist()),
                objects=objects,goals=[dict(predicate="seated_released_retracted",manipulated=n,
                    position=t,position_tolerance=.0015,tilt_tolerance_deg=3.) for n,t in targets.items()],
                perception="simulator_privileged_pose_and_segmentation")


def stage_calls(part,target,choice,stage):
    calls=[]; prefix=f"stage{stage}_"
    def add(skill,**params):
        cid=prefix+str(len(calls))
        calls.append(Call(cid,skill,{k:v if isinstance(v,Argument) else argument(v) for k,v in params.items()},
                          {"manipulated":part},"checker" if skill=="inspect" else "executable"))
        return cid
    def xyz(v): return Argument(plain(v),"position",frame="world",unit="m")
    add("detect"); add("estimate_pose",part=part)
    add("estimate_grasp",part=part,yaws=argument([choice["yaw"]],unit="rad"),
        height_offset=argument(choice["height"],unit="m"))
    producer=add("select_grasp",part=part)
    add("plan_path",target=Argument([0,0,.10],"position","deferred","world","m",producer,"grasp_hover"),
        yaw=argument(choice["yaw"],unit="rad"),clearance=argument(choice["clearance"],unit="m"))
    add("move",path="transfer"); add("move",grasp="grasp",part=part)
    add("grasp",part=part,force=argument(choice["force"],unit="N"))
    add("inspect",what="grasp",part=part)
    held=add("move",part=part,delta=xyz([0,0,.10]))
    add("plan_path",target=Argument((np.array(target)+[0,0,.065]).tolist(),"position","deferred","world","m",held,"object_to_eef"),
        yaw=Argument(None,"scalar","deferred",unit="rad",source_call=held,source_output="grasp_yaw"),
        clearance=argument(choice["clearance"],unit="m"))
    add("move",path="transfer")
    add("move",reference="object",part=part,target=xyz(np.array(target)+[0,0,.065]))
    add("move",mode="guarded",part=part,target_z=argument(target[2],unit="m"),
        force_stop=argument(3.,unit="N"),speed=argument(choice["speed"],unit="m/s"))
    add("press",part=part,target_z=argument(target[2],unit="m"))
    add("place",part=part,target=xyz(target),tol=argument(.0015,unit="m"),settle=argument(.35,unit="s"))
    add("move",delta=xyz([0,0,.10]))
    add("inspect",part=part,target=xyz(target),tol=argument(.0015,unit="m"))
    return calls


def program(session, targets, order, choices):
    calls=[c for i,n in enumerate(order) for c in stage_calls(n,targets[n],choices[n],i)]
    for name,target in targets.items():
        calls.append(Call(f"accept_{len(calls)}","inspect",dict(part=argument(name),
                          target=Argument(target,"position",frame="world",unit="m"),tol=argument(.0015,unit="m")),
                          {"manipulated":name},"checker"))
    cid=digest(dict(order=order,choices=choices))[:20]
    boundary=10
    payload=dict(id=cid,part=order[0],execution="program",start_state=fingerprint(session),
                 steps=[dict(skill=c.skill,params={k:a.value for k,a in c.arguments.items()}) for c in calls[:boundary]],
                 order=list(order),choices=plain(choices),cost=0.,skill_version="feedback.v2")
    return PlanIR(cid,calls,boundary,payload,"unknown",protocol=PROGRAM_PROTOCOL).validate(session.parts)


def semantic_key(plan):
    """IDs never define diversity. Geometry rounded below controller tolerances."""
    if plan.prefix.get("execution")=="program":
        row=dict(order=plan.prefix["order"],choices=plan.prefix["choices"])
    else:
        row=dict(grasp=plan.prefix["grasp"],path=plan.prefix["path"],control=plan.prefix["control"],
                 calls=[(c.skill,{k:a.value for k,a in c.arguments.items() if not k.startswith("bound_")}) for c in plan.calls])
    def clean(v):
        if isinstance(v,dict): return {k:clean(x) for k,x in sorted(v.items()) if k not in {"id","cost","binding","bindings","start_state","q_hover"}}
        if isinstance(v,(list,tuple)): return [clean(x) for x in v]
        if isinstance(v,(float,np.floating)): return round(float(v),4)
        return v
    return digest(clean(row))


def build_pool(session,targets,seed,n=16,completed=(),precheck=True):
    """Fixed label-blind shuffle; N=16/32/64 are prefixes of the same pool."""
    started=time.perf_counter(); remaining=[p for p in session.parts if p not in completed]
    orders=[remaining]
    if "insert_0" in remaining and "insert_1" in remaining:
        orders.append(["insert_1","insert_0"]+[p for p in remaining if not p.startswith("insert_")])
    rng=np.random.default_rng(np.random.SeedSequence([seed,183]))
    grid=list(itertools.product(range(len(orders)),range(2),range(4),range(2),range(2),range(2),range(2)))
    rng.shuffle(grid)
    result=[];seen=set();conflicts=0; raw=0; checks=0.; witnesses=[]
    # First-stage checking is shared by all continuations with the same grasp/route.
    cache={}
    for oi,y,h,route,force,other,speed in grid:
        raw+=1; order=orders[oi]; choices={}
        for j,p in enumerate(order):
            choices[p]=dict(yaw=float((y if j==0 else (other if p.startswith("insert") else 0))*math.pi/2),
                            height=float(([-.003,0.,.003,.006][h]) if p.startswith("insert") else ([-.002,0.,.0015,.003][h])),
                            clearance=.98+route*.055,force=2.5+force*1.0,speed=.006+speed*.002)
        plan=program(session,targets,order,choices); key=semantic_key(plan)
        if key in seen: continue
        seen.add(key)
        first=order[0]; choice=choices[first]
        check_key=(first,choice["yaw"],choice["height"],choice["clearance"])
        if precheck and check_key not in cache:
            t=time.perf_counter()
            xyz=session.ctx.obj_pos(first)+[0,0,session.grasp_specs[first][0]+choice["height"]]
            routes,details=transfer_routes(session,xyz+[0,0,.10],clearance=choice["clearance"],yaw=choice["yaw"])
            cache[check_key]=any(r["status"]=="necessary_pass" for r in routes)
            witnesses.extend(details); checks+=time.perf_counter()-t
        if precheck and not cache[check_key]:
            conflicts+=1; continue
        result.append(plan)
        if len(result)>=n: break
    return result,dict(requested=n,raw_count=raw,known_conflict=conflicts,unresolved=len(result),
                       deduplicated=len(seen),materializable=len(result),
                       grasp_branches=len({(p.prefix["part"],p.prefix["choices"][p.prefix["part"]]["yaw"],p.prefix["choices"][p.prefix["part"]]["height"]) for p in result}),
                       structure_branches=len({tuple(p.prefix["order"]) for p in result}),
                       necessary_geometry_seconds=checks,candidate_generation_seconds=time.perf_counter()-started-checks,
                       optional_geometry_seconds=0.,known_conflict_witnesses=witnesses)


def geometric_features(session,plan,targets):
    """Strong geometry baseline: checks the nominal future assembly in scratch state.

    Prior assembled parts are positioned only in scratch MjData. This is an
    optional prediction, never a physical rollout or a live-state mutation.
    """
    from simbench.assembly.candidates import release_space_proxy
    values=[]; original=session.ctx.data
    scratch=mujoco.MjData(session.ctx.model)
    scratch.qpos[:]=original.qpos; scratch.qvel[:]=original.qvel
    try:
        session.ctx.data=scratch
        for part in plan.prefix["order"]:
            mujoco.mj_forward(session.ctx.model,scratch)
            c=plan.prefix["choices"][part]
            grasp=dict(xyz=session.ctx.obj_pos(part)+[0,0,session.grasp_specs[part][0]+c["height"]],
                       yaw=c["yaw"],width=session.grasp_specs[part][1])
            proxy=release_space_proxy(session,part,grasp,dict(xyz=targets[part]))
            values.append(proxy["penalty"])
            bid=session.ctx.body_id(part);qa=session.ctx.model.jnt_qposadr[session.ctx.model.body_jntadr[bid]]
            scratch.qpos[qa:qa+3]=targets[part]
    finally: session.ctx.data=original
    costs=np.array(values)
    choices=list(plan.prefix["choices"].values())
    return [float(costs.sum()),float(costs.max()),float(np.count_nonzero(costs)),float(len(costs)),
            float(np.mean([c["clearance"] for c in choices])),float(np.mean([c["force"] for c in choices]))]
