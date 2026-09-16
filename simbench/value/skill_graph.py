"""Compile executable skill interfaces into immutable, pre-execution value input.

The embedded PlanIR is the sole executable payload. Edges record dependencies;
they never authorize skipping calls, substitute for collision checks, or carry
post-rollout measurements. The compiler reads no labels or geometry scores.
"""
from dataclasses import asdict
import copy
import inspect
from simbench.assembly.library import HANDLERS, Session
from simbench.assembly.interfaces import resolve
from simbench.assembly.ports import validate_ports
from .plan import PlanIR, digest, plain
from .program_audit import audit_program

SCHEMA = "twingraph.executable_skill_graph.v1"
RELATIONS = ("execution", "data", "object_state", "holding", "scene", "robot", "observations")


def compile_graph(observation, plan):
    if not isinstance(plan, PlanIR):
        plan = PlanIR.from_dict(plan)
    audit = audit_program(plan, observation["objects"])
    if plan.prefix.get("execution") != "program":
        raise ValueError("graph v1 requires explicit program PlanIR; legacy scorer remains available")
    obs = copy.deepcopy(observation)
    # Whitelist initial observation fields. Rollout logs never enter this record.
    obs = {k: obs[k] for k in ("robot", "objects", "goals", "perception") if k in obs}
    versions = {}; artifacts = {}; nodes = []; edges = []; held = None
    index = {c.id: i for i, c in enumerate(plan.calls)}

    def state_ref(key):
        version, producer = versions.get(key, (0, None))
        return dict(state=key, version=version, producer=producer,
                    status="known" if producer is None else "deferred")

    def edge(source, target, relation, port=""):
        if source is not None and source != target:
            edges.append(dict(source=source, target=target, relation=relation, port=port))

    for i, call in enumerate(plan.calls):
        raw = {k: a.value for k, a in call.arguments.items()}
        name, params = resolve(call.skill, raw)
        bound = inspect.signature(getattr(Session, name)).bind(None, **params)
        bound.apply_defaults(); params = dict(bound.arguments); params.pop("self")
        spec = HANDLERS[name]; validate_ports(spec, params)
        part = params.get("part") or held or call.roles.get("manipulated")
        ports = []
        for declaration in spec.ports:
            key = declaration.name; value = plain(params[key])
            original = call.arguments.get(key)
            row = {**asdict(declaration), "value": value, "status": "known", "source": None}
            if original:
                row.update(status=original.status,
                           unit=original.unit or declaration.unit,
                           frame=original.frame or declaration.frame)
                if original.source_call:
                    row["source"] = dict(call=index[original.source_call], output=original.source_output)
                    edge(index[original.source_call], i, "data", key)
            if declaration.kind == "artifact_ref" and key != "as_":
                producer = artifacts.get(value)
                if producer is not None:
                    row.update(status="deferred", source=dict(call=producer, output=nodes[producer]["outputs"][0]))
                    edge(producer, i, "data", key)
            ports.append(row)
        reads = []; writes = []
        for template in spec.state_reads:
            if "{part}" in template and part is None: continue
            key = template.format(part=part); ref = state_ref(key); reads.append(ref)
            relation = "object_state" if key.startswith("object:") else key
            edge(ref["producer"], i, relation, key)
        for template in spec.state_writes:
            if "{part}" in template and part is None: continue
            key = template.format(part=part)
            versions[key] = (versions.get(key, (0, None))[0] + 1, i)
            writes.append(state_ref(key))
        if spec.produces:
            artifacts[params.get("as_", spec.produces)] = i
        if "held:part" in spec.effects: held = params.get("part")
        if "held:empty" in spec.effects: held = None
        if i: edge(i-1, i, "execution")
        nodes.append(dict(index=i, skill=call.skill, implementation=name, kind=spec.kind,
                          ports=ports, roles=copy.deepcopy(call.roles), reads=reads, writes=writes,
                          requires=list(spec.requires), effects=list(spec.effects),
                          outputs=[spec.produces] if spec.produces else [],
                          obligations=list(spec.obligations)))
    return dict(schema=SCHEMA, observation=obs, plan=plan.to_dict(),
                plan_sha256=digest(plan.to_dict()), nodes=nodes, edges=edges,
                boundary=plan.boundary, protocol=plan.protocol,
                interface_sha256=interface_hash(), contract_status=audit["status"])


def interface_hash():
    return digest({name: asdict(spec) for name, spec in sorted(HANDLERS.items())})


def validate_graph(graph):
    if graph.get("schema") != SCHEMA:
        raise ValueError("unsupported executable graph schema")
    plan = PlanIR.from_dict(graph["plan"])
    expected = compile_graph(graph["observation"], plan)
    if digest(expected) != digest(graph):
        raise ValueError("graph/PlanIR/interface mismatch; recompile before scoring or execution")
    return plan
