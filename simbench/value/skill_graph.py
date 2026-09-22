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
from .plan import PlanIR, digest, plain, initial_artifacts
from .program_audit import audit_program

SCHEMA = "twingraph.executable_skill_graph.v1"
RELATIONS = ("execution", "data", "object_state", "holding", "scene", "robot", "observations")


def compile_graph(observation, plan):
    if not isinstance(plan, PlanIR):
        plan = PlanIR.from_dict(plan)
    audit = audit_program(plan, observation["objects"])
    if plan.prefix.get("execution") not in {"program", "full_task_v7"}:
        raise ValueError("graph v1 requires explicit program PlanIR; legacy scorer remains available")
    obs = copy.deepcopy(observation)
    # Whitelist initial observation fields. Rollout logs never enter this record.
    obs = {k: obs[k] for k in ("robot", "objects", "goals", "perception",
        "fixtures", "assembly_targets", "receiver_geometry", "fixture_relations") if k in obs}
    versions = {}; artifacts = {}; nodes = []; edges = []; held = None
    materialized = initial_artifacts(plan)
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
            if declaration.kind == "binding" and value is not None:
                raise ValueError("explicit selection binding needs materialized selected data; graph v1 supports solver-default selection")
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
                elif value in materialized:
                    row["materialized"] = copy.deepcopy(materialized[value])
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


def validate_geometry_conditions(graph, plan=None):
    """Bind cheap CAD readouts to the exact executable full-task graph.

    This is a provenance/consistency check, not a proof of physical feasibility.
    Old graphs without a catalogue retain explicit missing-feature masks.
    """
    import math
    conditions = graph.get("geometry_conditions")
    if conditions is None:
        return {}
    if not isinstance(conditions, dict) or conditions.get("schema") != "twingraph.geometry_conditions.v13":
        raise ValueError("unsupported geometry_conditions schema")
    assembly = graph["assembly"]
    plan = validate_graph(assembly) if plan is None else plan
    observation_sha = assembly["observation"].get("perception", {}).get("observation_sha256")
    if not isinstance(observation_sha, str) or not observation_sha:
        raise ValueError("geometry_conditions requires a bound pre-execution observation")
    expected = dict(plan_sha256=assembly["plan_sha256"], choices_sha256=digest(plan.prefix["choices"]),
                    order_sha256=digest(plan.prefix["order"]), observation_sha256=observation_sha,
                    cad_sha256=digest(graph.get("planning_cad", {})))
    for key, value in expected.items():
        if conditions.get(key) != value:
            raise ValueError(f"geometry_conditions {key} differs from executable plan/observation/CAD")
    records = conditions.get("parts")
    if not isinstance(records, dict):
        raise ValueError("geometry_conditions requires per-part records")
    for part, row in records.items():
        if part not in plan.prefix["choices"] or not isinstance(row, dict):
            raise ValueError("geometry_conditions references an unknown plan part")
        if row.get("part", part) != part or row.get("status") not in ("necessary_pass", "unknown", "rejected"):
            raise ValueError("invalid geometry_conditions part/status")
        if row.get("executable_parameters_available") is False:
            raise ValueError("geometry_conditions selected row lacks executable parameters")
        for key in ("yaw", "height", "placement_yaw", "pin_command_depth_m", "pin_press_extra_m"):
            if key not in row:
                continue
            value, command = row[key], plan.prefix["choices"][part].get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"invalid geometry_conditions {part}.{key}")
            if command is None or value != command:
                raise ValueError(f"geometry_conditions {part}.{key} differs from executable choice")
        for key in ("min_clearance_m", "required_clearance_m", "pad_face_axial_overlap_m", "head_pad_axial_overlap_m", "grasp_width_m"):
            value = row.get(key)
            if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)):
                raise ValueError(f"invalid geometry_conditions metric {key}")
        choice = plan.prefix["choices"][part]
        if row.get("grasp_width_m") is not None and choice.get("width") is not None:
            if row["grasp_width_m"] != choice["width"]:
                raise ValueError(f"geometry_conditions {part}.grasp_width_m differs from executable choice")
        if "yaw" in row:
            frame = row.get("grasp_yaw_frame", "world")
            if frame not in ("world", "object") or frame != choice.get("grasp_yaw_frame", "world"):
                raise ValueError(f"geometry_conditions {part}.grasp_yaw_frame differs from executable choice")
            if frame == "object":
                quaternion = assembly["observation"]["objects"].get(part, {}).get("quaternion")
                world = row.get("evaluated_world_yaw_rad")
                if quaternion is None or world is None:
                    raise ValueError("object-relative geometry requires evaluated world yaw and observed quaternion")
                norm = math.sqrt(sum(float(v)**2 for v in quaternion))
                if len(quaternion) != 4 or not math.isfinite(norm) or norm < 1.e-8 or not math.isfinite(world):
                    raise ValueError("invalid object-relative geometry orientation")
                w, x, y, z = (float(v)/norm for v in quaternion)
                observed_yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
                delta = world - observed_yaw - row["yaw"]
                if abs(math.atan2(math.sin(delta), math.cos(delta))) > 1.e-9:
                    raise ValueError("geometry_conditions evaluated world yaw disagrees with observed object frame")
        required = row.get("required_clearance_m")
        minimum = row.get("min_clearance_m")
        if required is not None and required < 0:
            raise ValueError("negative geometry uncertainty reserve")
        if row["status"] == "necessary_pass" and (minimum is None or required is None or minimum < required):
            raise ValueError("geometry necessary_pass does not meet its declared clearance reserve")
    return records
