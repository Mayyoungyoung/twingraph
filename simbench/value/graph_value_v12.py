"""Value inputs derived only from validated executable skill graphs.

Stage pooling retains typed-port masks, execution order and directed state/data
dependencies. Fixed physical units and periodic angles replace the V6 unsafe
division by near-zero training standard deviations. This is a compact graph
model inspired by plan feasibility and skill affordance, not a PIGINet replica.
"""
import math
import numpy as np

from .plan import PlanIR
from .skill_graph import RELATIONS, validate_geometry_conditions, validate_graph

STAGES = ("cleaning", "carriage", "end_stop", "pin_left", "pin_right", "handle",
          "bidirectional_stroke", "pin_retention")
SCHEMA = "twingraph.graph_stage_ports.v12.r5_object_relative_grasp"
VALUE_RELATIONS = (*RELATIONS, "phase", "planned_before", "predicted_obstruction", "goal_dependency")
PART_STAGES = STAGES[1:6]
MAX_PATH_YAWS = 4
PORTS = (("height_offset", .01), ("clearance", 1.), ("force", 10.),
         ("speed", .01), ("force_limit", 20.), ("force_stop", 10.),
         ("target_z", 1.), ("tol", .01), ("settle", 1.),
         ("minimum_insertion_depth_m", .02), ("minimum", .1),
         ("press_force", 10.), ("lift_first_m", .1), ("width", .05),
         ("pin_command_depth_m", .05), ("pin_press_extra_m", .05))


def _vector(value, length=3):
    if value is None:
        return None
    result = np.asarray(value, float)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError("invalid declared state-plan geometry")
    return result


def _scalar_port(nodes, name, *, implementation=None):
    values = [p["value"] for node in nodes if implementation is None or node["implementation"] == implementation
              for p in node["ports"] if p["name"] == name and p["status"] == "known"
              and isinstance(p["value"], (int, float))]
    if any(not np.isfinite(v) for v in values):
        raise ValueError("non-finite state-plan port")
    return float(values[-1]) if values else None


def _yaw(quaternion):
    q = _vector(quaternion, 4)
    if q is None:
        return None
    if np.linalg.norm(q) < 1.e-8:
        raise ValueError("invalid observed/declared orientation")
    w, x, y, z = q/np.linalg.norm(q)
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def pickup_yaw_state(nodes, obj):
    """Read commanded frame and initial observed world yaw from typed ports.

    An object-relative command is rebound by the skill at its future camera
    observation. Its current world estimate is only a planning input; missing
    object orientation stays unknown and is never recovered from rollout truth.
    """
    for node in nodes:
        yaw_ports = [p for p in node["ports"] if p["name"] == "yaws"
                     and p["status"] == "known" and p["value"]]
        if not yaw_ports:
            continue
        port = yaw_ports[0]
        command = float(port["value"][0])
        frames = [p for p in node["ports"] if p["name"] == "yaw_frame"]
        frame = frames[0]["value"] if frames and frames[0]["status"] == "known" else port.get("frame", "world")
        frame = frame or "world"
        if frame not in ("world", "object") or not np.isfinite(command):
            raise ValueError("invalid grasp yaw frame or value")
        if port.get("frame") and port["frame"] != frame:
            raise ValueError("grasp yaw port and frame disagree")
        observed = _yaw(obj.get("quaternion"))
        world = command if frame == "world" else command + observed if observed is not None else None
        return command, frame, world
    return None, None, None


def declared_stage_targets(by_stage):
    """Read executed placement contracts, never prefix metadata or a rollout."""
    targets = {}
    for stage, nodes in by_stage.items():
        values = [p["value"] for node in nodes if node["skill"] == "place"
                  for p in node["ports"] if p["name"] == "target" and p["status"] == "known"
                  and isinstance(p["value"], (list, tuple))]
        if values:
            targets[stage] = _vector(values[-1])
    return targets


def state_plan_interactions(graph, by_stage, order, completed):
    """Cheap declared geometry, not collision checking or predicted outcomes.

    A preceding placement supplies a *conditional nominal* occupied position.
    Missing geometry has a known mask, and fit residual is a proxy uncertainty,
    not a calibrated probability. No result/checkpoint/simulator fields enter.
    """
    cad = graph.get("planning_cad", {})
    observation = graph["assembly"]["observation"]
    objects = observation["objects"]
    targets = declared_stage_targets(by_stage)
    assembly_targets = observation.get("assembly_targets", {})
    # Final functional targets may differ from a release waypoint (deep pins).
    # They remain conditional intentions, not observed achieved placements.
    for part, row in assembly_targets.items():
        if part in PART_STAGES and row.get("position_m") is not None:
            targets[part] = _vector(row["position_m"])
    extra, obstruction = {}, np.zeros((len(STAGES), len(STAGES)), np.float32)
    for stage in STAGES:
        part = "wipe_tool" if stage == "cleaning" else stage
        obj = objects.get(part, {})
        spec = cad.get("parts", {}).get(part, {})
        dims = _vector(spec.get("dimensions_m"))
        if dims is not None and np.any(dims <= 0):
            raise ValueError("CAD dimensions must be positive")
        sigma = obj.get("fit_residual_m")
        sigma = float(sigma) if sigma is not None else None
        if sigma is not None and (not np.isfinite(sigma) or sigma < 0):
            raise ValueError("invalid observed uncertainty")
        row = []
        def scalar(value, scale=1.):
            row.extend((_number(value, scale) if value is not None else 0., float(value is not None)))
        row.extend(_number(v, .1) for v in (dims if dims is not None else np.zeros(3)))
        row.append(float(dims is not None))
        nodes = by_stage[stage]
        # The controller's force is a command/budget, not measured material strength.
        grip = _scalar_port(nodes, "force", implementation="close_gripper")
        if grip is None:
            grip = _scalar_port(nodes, "force")
        contact = _scalar_port(nodes, "force_limit")
        press = _scalar_port(nodes, "force_stop")
        load = max(v for v in (contact, press) if v is not None) if contact is not None or press is not None else None
        scalar(grip / max(load, 1.e-6) if grip is not None and load is not None else None)
        # This ratio intentionally has no invented friction coefficient.
        width = _scalar_port(nodes, "width")
        scalar(width, .05)
        scalar(spec.get("grasp_reference", {}).get("width_m"), .05)
        scalar(_scalar_port(nodes, "height_offset"), .01)
        scalar(_scalar_port(nodes, "speed") * sigma if sigma is not None and _scalar_port(nodes, "speed") is not None else None, .0001)
        clearance = None
        if stage.startswith("pin_"):
            aperture = cad.get("pin_hole_width_m")
            radius = cad.get("pin_shaft_radius_m")
            if aperture is not None and radius is not None:
                clearance = float(aperture)/2 - float(radius)
        elif stage == "handle":
            bore, post = cad.get("handle_bore_radius_m"), cad.get("handle_post_radius_m")
            if bore is not None and post is not None:
                clearance = float(bore)-float(post)
        scalar(clearance, .01)
        scalar(clearance-sigma if clearance is not None and sigma is not None else None, .01)
        scalar(sigma/max(clearance, 1.e-6) if clearance is not None and sigma is not None else None)
        goal = observation.get("goals", [{}])[0]
        scalar(goal.get("stroke_minimum_m"), .1)
        scalar(goal.get("pin_minimum_depth_m"), .02)
        for name in ("pin_base_bridge_required", "fixture_capture_required"):
            scalar(float(bool(goal[name])) if name in goal else None)
        _, _, pickup = pickup_yaw_state(nodes, obj)
        placement = _scalar_port(nodes, "yaw")
        observed_yaw = _yaw(obj.get("quaternion"))
        goal_yaw = _yaw(assembly_targets.get(stage, {}).get("quat_wxyz"))
        delta_yaw = (placement-pickup+observed_yaw-goal_yaw
                     if all(v is not None for v in (placement, pickup, observed_yaw, goal_yaw)) else None)
        row.extend((math.sin(delta_yaw), math.cos(delta_yaw), 1.) if delta_yaw is not None else (0., 0., 0.))
        # Reserve explicit known masks for full hand sweep geometry. A CAD
        # missing these declarations must not silently stand for zero volume.
        hand = cad.get("gripper_geometry", {})
        scalar(hand.get("palm_radius_m"), .1)
        scalar(hand.get("finger_swept_radius_m"), .1)
        fixture = observation.get("fixtures", {}).get("guide_base", {})
        fixture_position = _vector(fixture.get("position_m")) if fixture.get("valid") else None
        row.extend(_number(v, .5) for v in (fixture_position if fixture_position is not None else np.zeros(3)))
        row.append(float(fixture_position is not None))
        relation = observation.get("fixture_relations", {}).get("end_stop_to_base", {})
        installed_stop = "end_stop" in completed
        row.extend((float(installed_stop), float(bool(relation.get("observable"))),
                    float(bool(relation.get("requires_guarded_verification")))))
        # A loose initial stop is expected to be unaligned. Only an already
        # completed stop can supply observed assembled-route supervision input.
        route_known = installed_stop and bool(relation.get("observable"))
        scalar(float(bool(relation.get("geometric_route_exists"))) if route_known else None)
        hole = relation.get("holes", {}).get(stage, {})
        scalar(hole.get("geometric_margin_m") if route_known else None, .01)
        scalar(hole.get("minimum_total_depth_m"), .05)
        receiver = observation.get("receiver_geometry", {}).get("base_holes", {})
        receiver = (receiver.get(stage, {}) if isinstance(receiver, dict) else
                    next((r for r in receiver if r.get("id") == stage), {}) if isinstance(receiver, list) else {})
        scalar(receiver.get("depth_m", receiver.get("depth")), .05)
        scalar(receiver.get("half_width_m", receiver.get("half_width")), .01)
        desired = assembly_targets.get(stage, {})
        entry = _vector(desired.get("hole_entry_m"))
        axis = _vector(desired.get("axis"))
        command_axis = None
        for port_name in ("hole_entry_m", "axis"):
            values = [p["value"] for node in nodes for p in node["ports"]
                      if p["name"] == port_name and p["status"] == "known" and p["value"] is not None]
            vector = _vector(values[-1]) if values else None
            row.extend(_number(v, .5 if port_name == "hole_entry_m" else 1.)
                       for v in (vector if vector is not None else np.zeros(3)))
            row.append(float(vector is not None))
            if port_name == "axis":
                command_axis = vector
        scalar(float(np.dot(command_axis, axis)/(np.linalg.norm(command_axis)*np.linalg.norm(axis)))
               if command_axis is not None and axis is not None and np.linalg.norm(command_axis)*np.linalg.norm(axis) > 1.e-8 else None)
        commanded = targets.get(stage)
        axial = (float(np.dot(commanded-entry, axis/np.linalg.norm(axis)))
                 if entry is not None and commanded is not None and axis is not None and np.linalg.norm(axis) > 1.e-8 else None)
        scalar(axial, .1)
        target = targets.get(stage)
        own_radius = float(np.linalg.norm(dims[:2])/2) if dims is not None else None
        # Explicitly expose how each early placement changes a later destination.
        # Circumscribed XY radii are coarse, orientation-invariant shape proxies;
        # negative separation is not a collision verdict (mating parts overlap).
        for other in PART_STAGES:
            before = other != stage and order.index(other) < order.index(stage)
            is_completed = other in completed
            predicted = before and not is_completed
            observed = objects.get(other, {})
            other_position = (_vector(observed.get("position")) if is_completed or not before
                              else targets.get(other))
            other_dims = _vector(cad.get("parts", {}).get(other, {}).get("dimensions_m"))
            known = other != stage and target is not None and other_position is not None
            delta = other_position-target if known else np.zeros(3)
            row.extend((float(before), float(is_completed), float(predicted), float(known)))
            row.extend(_number(v, .1) for v in delta)
            gap = (float(np.linalg.norm(delta[:2]))-own_radius-float(np.linalg.norm(other_dims[:2])/2)
                   if known and own_radius is not None and other_dims is not None else None)
            scalar(gap, .1)
            if gap is not None and gap < 0 and before:
                obstruction[STAGES.index(other), STAGES.index(stage)] = 1.
        extra[stage] = row
    return extra, obstruction


def _number(value, scale=1.):
    value = float(value) / scale
    if not np.isfinite(value):
        raise ValueError("non-finite graph value")
    return float(np.clip(value, -10., 10.))


def path_yaw_features(nodes):
    """Retain each source/placement path yaw in executable call order.

    Source grasp yaw and destination placement yaw are different decisions.
    Slots carry sin/cos, known, deferred and present bits. Refuse an unsupported
    larger plan rather than silently losing additional yaw-bearing paths.
    """
    ports = [p for node in nodes for p in node["ports"] if p["name"] == "yaw"]
    if len(ports) > MAX_PATH_YAWS:
        raise ValueError("too many path yaw ports for frozen stage feature schema")
    result = []
    for i in range(MAX_PATH_YAWS):
        if i >= len(ports):
            result.extend((0., 0., 0., 0., 0.))
            continue
        port = ports[i]
        known = port["status"] == "known" and isinstance(port["value"], (int, float))
        yaw = float(port["value"]) if known else 0.
        if not np.isfinite(yaw):
            raise ValueError("non-finite path yaw")
        result.extend((math.sin(yaw) if known else 0., math.cos(yaw) if known else 0.,
                       float(known), float(port["status"] == "deferred"), 1.))
    return result


def geometry_condition_features(record):
    """The same audited CAD catalogue that is available to cheap ranking.

    A known sampled number and a clearance certificate are different fields:
    unknown may contain a measured positive margin below its reserve, while a
    missing number always has value=0 AND known=0. No failed rollout label is
    read or inferred from these necessary geometry conditions.
    """
    record = record or {}
    values = [float(bool(record))]
    values.extend(float(record.get("status") == s) for s in ("necessary_pass", "unknown", "rejected"))
    for key in ("min_clearance_m", "required_clearance_m", "grasp_width_m"):
        value = record.get(key)
        values.extend((_number(value, .01) if value is not None else 0., float(value is not None)))
    minimum, required = record.get("min_clearance_m"), record.get("required_clearance_m")
    known = minimum is not None and required is not None
    values.extend((_number(minimum-required, .01) if known else 0., float(known)))
    overlap = record.get("pad_face_axial_overlap_m", record.get("head_pad_axial_overlap_m"))
    values.extend((_number(overlap, .01) if overlap is not None else 0., float(overlap is not None)))
    checked = record.get("source_grasp_ik_checked")
    values.extend((float(bool(checked)), float(checked is not None)))
    return values


def encode_graph(graph, check=True):
    """Return stage nodes and relation matrices without reading proposal names.

Rollout results, seed, condition IDs and simulator state are not input fields.
Unknown/deferred values have explicit masks. A changed graph must be recompiled.
"""
    assembly = graph["assembly"]
    plan = validate_graph(assembly) if check else PlanIR.from_dict(assembly["plan"])
    conditions = validate_geometry_conditions(graph, plan)
    if "full_plan" in graph:
        full = PlanIR.from_dict(graph["full_plan"])
        params = {k: a.value for k, a in full.calls[0].arguments.items()}
        for key in ("order", "choices"):
            if key in params and params[key] != plan.prefix[key]:
                raise ValueError("full-task envelope and executable graph disagree")
        for key in ("wipe_variant", "wipe_force", "wipe_duration"):
            if key in params and params[key] != graph["cleaning"][key]:
                raise ValueError("cleaning envelope and graph disagree")
    obs = assembly["observation"]
    objects = obs["objects"]
    aliases = {"stroke": "bidirectional_stroke", "retention": "pin_retention"}
    completed = {aliases.get(s, s) for s in graph.get("completion", {}).get("completed", ())}
    order = ["cleaning", *plan.prefix["order"], "bidirectional_stroke", "pin_retention"]
    by_stage = {s: [] for s in STAGES}
    owner = {}
    for node in assembly["nodes"]:
        part = node["roles"].get("manipulated")
        if part in by_stage:
            by_stage[part].append(node)
            owner[node["index"]] = STAGES.index(part)
    interactions, obstruction = state_plan_interactions(graph, by_stage, order, completed)
    out = []
    for stage in STAGES:
        row = []
        row.extend(float(stage == s) for s in STAGES)
        row.extend((order.index(stage) / 7., float(stage in completed), float(stage not in completed)))
        obj = objects.get("wipe_tool" if stage == "cleaning" else stage, {})
        position = obj.get("position")
        valid = bool(obj.get("valid", position is not None))
        row.extend(_number(v, .5) for v in (position or [0., 0., 0.]))
        row.extend((float(valid), float(position is not None), _number(obj.get("quality") or 0.),
                    _number(obj.get("fit_residual_m") or 0., .01), float(obj.get("fit_residual_m") is not None)))
        q = obj.get("quaternion")
        if q is None:
            row.extend((0., 0., 0.))
        else:
            w, x, y, z = np.asarray(q, float) / max(np.linalg.norm(q), 1.e-8)
            yaw = math.atan2(2 * (w*z+x*y), 1-2*(y*y+z*z))
            row.extend((math.sin(yaw), math.cos(yaw), 1.))
        nodes = by_stage[stage]
        # Mean port values are accompanied by known/deferred/present masks.
        ports = [p for node in nodes for p in node["ports"]]
        for name, scale in PORTS:
            ps = [p for p in ports if p["name"] == name]
            known = [p for p in ps if p["status"] == "known" and isinstance(p["value"], (int, float))]
            row.extend((_number(np.mean([p["value"] for p in known]), scale) if known else 0.,
                        float(bool(known)), float(any(p["status"] == "deferred" for p in ps)), float(bool(ps))))
        command_yaw, yaw_frame, world_yaw = pickup_yaw_state(nodes, obj)
        row.extend((math.sin(command_yaw), math.cos(command_yaw), 1.) if command_yaw is not None else (0., 0., 0.))
        row.extend(float(yaw_frame == frame) for frame in ("world", "object"))
        row.extend((math.sin(world_yaw), math.cos(world_yaw), 1.) if world_yaw is not None else (0., 0., 0.))
        row.extend(path_yaw_features(nodes))
        concrete = [p["value"] for p in ports if p["name"] == "target" and p["status"] == "known"
                    and isinstance(p["value"], (list, tuple)) and len(p["value"]) == 3]
        target = np.asarray(concrete[-1], float) if concrete else np.zeros(3)
        row.extend(_number(v, .5) for v in target)
        row.append(float(bool(concrete)))
        relative = target - np.asarray(position or [0., 0., 0.]) if concrete and position else np.zeros(3)
        row.extend(_number(v, .5) for v in relative)
        row.extend((_number(np.linalg.norm(relative), .5), float(bool(concrete) and position is not None)))
        row.extend((len(nodes)/24., sum(p["status"] == "deferred" for p in ports)/40.,
                    sum(p["value"] is None for p in ports)/40.))
        strategies = [p["value"] for p in ports if p["name"] == "strategy"]
        row.extend(float(s in strategies) for s in ("feedback", "cartesian", "joint", "joint_checked_v10", "learned"))
        lifts = [p["value"][2] for p in ports if p["name"] == "delta"
                 and isinstance(p["value"], (list, tuple)) and len(p["value"]) == 3 and p["value"][2] > 0]
        row.extend((_number(min(lifts), .1) if lifts else 0., len(lifts)/4.))
        clean = graph.get("cleaning", {}) if stage == "cleaning" else {}
        row.extend((_number(clean.get("wipe_variant", 0), 3.), _number(clean.get("wipe_force", 0), 10.),
                    _number(clean.get("wipe_duration", 0), 20.), _number(clean.get("minimum_coverage", 0))))
        # Relative observed scene state makes state-conditioned ranking possible.
        for part in ("carriage", "end_stop", "pin_left", "pin_right", "handle", "wipe_tool"):
            other = objects.get(part, {})
            pos = other.get("position")
            delta = np.asarray(pos) - np.asarray(position or [0., 0., 0.]) if pos is not None else np.zeros(3)
            row.extend(_number(v, .5) for v in delta)
            row.append(float(pos is not None and bool(other.get("valid", True))))
        robot = obs.get("robot", {})
        joints = robot.get("joints", ())
        for j in range(7):
            row.extend((math.sin(joints[j]), math.cos(joints[j]), 1.) if j < len(joints) else (0., 0., 0.))
        fingers = robot.get("fingers", ())
        row.extend(_number(fingers[j], .05) if j < len(fingers) else 0. for j in range(2))
        row.append(float(len(fingers) == 2))
        eef = robot.get("eef")
        row.extend(_number(v, .5) for v in (eef or [0., 0., 0.]))
        row.append(float(eef is not None))
        row.extend(interactions[stage])
        row.extend(geometry_condition_features(conditions.get(stage)))
        out.append(row)
    # Keep relation types distinct; message passing uses graph topology.
    relations = np.zeros((len(VALUE_RELATIONS), len(STAGES), len(STAGES)), np.float32)
    for edge in assembly["edges"]:
        if edge["source"] in owner and edge["target"] in owner:
            a, b = owner[edge["source"]], owner[edge["target"]]
            relations[RELATIONS.index(edge["relation"]), a, b] += 1.
    for a, b in graph.get("phase_edges", ()):
        if a in STAGES and b in STAGES:
            relations[VALUE_RELATIONS.index("phase"), STAGES.index(a), STAGES.index(b)] = 1.
    for a in STAGES:
        for b in STAGES:
            if order.index(a) < order.index(b):
                relations[VALUE_RELATIONS.index("planned_before"), STAGES.index(a), STAGES.index(b)] = 1.
    relations[VALUE_RELATIONS.index("predicted_obstruction")] = obstruction
    for part in PART_STAGES:
        relations[VALUE_RELATIONS.index("goal_dependency"), STAGES.index(part), STAGES.index("bidirectional_stroke")] = 1.
    for part in ("pin_left", "pin_right"):
        relations[VALUE_RELATIONS.index("goal_dependency"), STAGES.index(part), STAGES.index("pin_retention")] = 1.
    relations = np.minimum(relations, 1.)
    x = np.asarray(out, np.float32)
    if not np.isfinite(x).all():
        raise ValueError("non-finite graph encoding")
    return {"x": x, "relations": relations, "active": np.asarray([s not in completed for s in STAGES], np.float32)}


class ValueRankerV12:
    """Drop-in scorer: rank(graphs) returns probabilities in original order."""
    def __init__(self, checkpoint, device="cpu"):
        import hashlib
        from pathlib import Path
        import torch
        from .value_v12 import StageValueNet
        self.device = device
        self.sha256 = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        self.saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if self.saved.get("schema") != SCHEMA:
            raise ValueError(f"incompatible value input schema: checkpoint {self.saved.get('schema')} != encoder {SCHEMA}; retraining is required")
        self.model = StageValueNet(**self.saved["model_config"]).to(device)
        self.model.load_state_dict(self.saved["state_dict"])
        self.model.eval()

    def rank(self, graphs):
        import torch
        from .value_v12 import collate
        if not graphs:
            return np.empty(0, np.float32)
        encoded = [encode_graph(g) for g in graphs]
        if encoded[0]["x"].shape[-1] != self.saved["model_config"]["input_dim"]:
            raise ValueError("checkpoint input dimension differs from frozen encoder")
        with torch.inference_mode():
            return self.model(collate(encoded, self.device))["plan_logit"].sigmoid().cpu().numpy()

    __call__ = rank
    score = rank
