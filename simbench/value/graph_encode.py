"""Generic typed-port encoder for the executable skill graph.

Object/producer names are binding keys only. Context records are graph state
inputs, not a second task representation. No hand-selected product features.
"""
import json
import numpy as np
from .encode import bucket, NUMERIC
from .plan import JOINT_PATH_FIELDS
from .skill_graph import RELATIONS, validate_graph


def encode_graph(graph, check=True, cache=None):
    if check: validate_graph(graph)
    obs = graph["observation"]; objects = obs["objects"]
    nodes = []; tokens = []; owner = []
    cache={} if cache is None else cache

    def leaf(key, value, node, status="known", kind="record", unit="", frame=""):
        identity=(key,status,kind,unit,frame,tuple(value) if isinstance(value,(tuple,list)) else value)
        if identity in cache:
            tokens.append(cache[identity]);owner.append(node);return
        category = 0
        if isinstance(value, (bool, int, float)):
            values = [value]
        elif isinstance(value, (list, tuple)) and all(isinstance(x, (int, float, bool)) for x in value):
            values = value
        else:
            values = []
            if value is not None: category = bucket(str(value))
        if len(values) > 8:
            for i in range(0, len(values), 8):
                leaf(key+f"/chunk{i//8}", values[i:i+8], node, status, kind, unit, frame)
            return
        vals = np.asarray(values, np.float32)
        if not np.isfinite(vals).all(): raise ValueError("nonfinite graph port")
        numbers = np.zeros(NUMERIC, np.float32)
        for i, scale in enumerate((1., 10., 100.)):
            numbers[8*i:8*i+len(vals)] = np.tanh(vals*scale)
        tokens.append((bucket(f"{key}|{kind}|{unit}|{frame}"), category,
                       {"known":1,"deferred":2,"unknown":3}[status], numbers))
        cache[identity]=tokens[-1]
        owner.append(node)

    def visit(key, value, node, **meta):
        if isinstance(value, dict):
            for k,v in sorted(value.items()): visit(key+"/"+k,v,node,**meta)
        elif isinstance(value, (tuple,list)) and any(isinstance(x,(dict,list,tuple)) for x in value):
            # A supplied trajectory is the executable payload, not a sampled
            # summary: every waypoint remains available to the value module.
            for i,v in enumerate(value): visit(key+f"/{i}",v,node,**meta)
        elif isinstance(value,str) and value in objects:
            # Binding carries the corresponding observed geometry; no raw ID.
            visit(key+"/bound_object",objects[value],node,**meta)
        else: leaf(key,value,node,**meta)

    # Canonical order makes mere object dictionary permutation/renaming inert.
    contexts = [("robot", obs["robot"])]
    contexts += [("scene_object", x) for x in sorted(objects.values(),key=lambda x:json.dumps(x,sort_keys=True))]
    def bound_goal(g):
        return {k: objects[v] if isinstance(v,str) and v in objects else v for k,v in g.items()}
    contexts += [("goal",x) for x in sorted((bound_goal(g) for g in obs["goals"]),key=lambda x:json.dumps(x,sort_keys=True))]
    for kind, value in contexts:
        node = len(nodes); nodes.append(0)
        leaf("context_kind",kind,node);visit(kind,value,node)
    offset = len(nodes)
    for call in graph["nodes"]:
        node = len(nodes); nodes.append(call["index"]+1)
        for key in ("skill","implementation","kind","requires","effects","outputs"):
            visit(key,call[key],node)
        leaf("execution_boundary",int(call["index"]<graph["boundary"]),node)
        for role, obj in sorted(call["roles"].items()): visit("role/"+role,obj,node)
        for port in call["ports"]:
            key=port["name"]
            # Binding addresses have their information in typed producer edges.
            value = None if port["kind"] in {"artifact_ref","binding"} else port["value"]
            visit("port/"+key,value,node,status=port["status"],kind=port["kind"],unit=port["unit"],frame=port["frame"])
            if "materialized" in port:
                for field, (kind, unit, frame) in JOINT_PATH_FIELDS.items():
                    visit("materialized/"+key+"/"+field,port["materialized"][field],node,
                          status="known",kind=kind,unit=unit,frame=frame)
            if port["source"]:
                leaf("producer_output/"+key,port["source"]["output"],node,status="deferred")
        for direction in ("reads","writes"):
            for ref in call[direction]:
                typ="object" if ref["state"].startswith("object:") else ref["state"]
                leaf(direction+"/"+typ,ref["version"],node,status=ref["status"])
    # Multi-hot directed relations, each with an explicit reverse channel.
    relations=np.zeros((len(nodes),len(nodes),2*len(RELATIONS)),np.float32)
    for edge in graph["edges"]:
        a,b=edge["source"]+offset,edge["target"]+offset
        r=RELATIONS.index(edge["relation"])
        relations[a,b,2*r]=1.;relations[b,a,2*r+1]=1.
    return dict(keys=np.asarray([x[0] for x in tokens]),categories=np.asarray([x[1] for x in tokens]),
                statuses=np.asarray([x[2] for x in tokens]),numbers=np.stack([x[3] for x in tokens]),
                owner=np.asarray(owner),positions=np.asarray(nodes),relations=relations)


def encode_graphs(graphs,check=True):
    # Cache pure typed values only within one candidate pool. No cross-request
    # cache, labels, geometry proxies or rollout results are retained.
    cache={}
    return [encode_graph(g,check=check,cache=cache) for g in graphs]


def collate_graph(rows, device="cpu", relation_mode="correct"):
    import torch
    size=max(len(r["keys"]) for r in rows); nodes=max(len(r["positions"]) for r in rows)
    b={}
    for key in ("keys","categories","statuses","owner","numbers"):
        a=np.zeros((len(rows),size,NUMERIC) if key=="numbers" else (len(rows),size),
                   np.float32 if key=="numbers" else np.int64)
        for i,r in enumerate(rows):a[i,:len(r[key])]=r[key]
        b[key]=torch.as_tensor(a,device=device)
    positions=np.zeros((len(rows),nodes),np.int64);mask=np.ones((len(rows),nodes),bool)
    relations=np.zeros((len(rows),nodes,nodes,2*len(RELATIONS)),np.float32)
    for i,r in enumerate(rows):
        n=len(r["positions"]);positions[i,:n]=r["positions"];mask[i,:n]=False
        rel=r["relations"]
        if relation_mode=="shuffle":
            # Fixed label/ID-independent permutation, preserving edge types/counts.
            ix=np.random.default_rng(197).permutation(n);rel=rel[ix][:,ix]
        elif relation_mode=="none":rel=np.zeros_like(rel)
        elif relation_mode!="correct":raise ValueError(relation_mode)
        relations[i,:n,:n]=rel
    b.update(positions=torch.as_tensor(positions,device=device),node_padding=torch.as_tensor(mask,device=device),
             relations=torch.as_tensor(relations,device=device),
             padding=torch.arange(size,device=device)[None,:]>=torch.tensor([len(r["keys"]) for r in rows],device=device)[:,None])
    return b
