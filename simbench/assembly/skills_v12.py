"""Explicit V12 skill configuration and observation/evaluation boundaries.

Control consumes RGB-D, robot encoders/FK, and a simulated force/tactile adapter.
Independent acceptance may inspect the simulated physical assembly, and labels
its provenance. It is never fed into the pose detector or learned policy.
"""
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np
from scipy.spatial.transform import Rotation

CHECKPOINTS = Path(__file__).parent / "checkpoints"


@dataclass(frozen=True)
class EndStopSeatConfig:
    """Declared bounded controller envelope; not inferred material parameters."""
    schema: str = "twingraph.end_stop_seat_search.v1"
    maximum_attempts: int = 25
    maximum_radius_m: float = .003
    uncertainty_multiplier: float = 3.
    minimum_pose_indicator_m: float = .0005
    maximum_control_steps: int = 8000
    lift_speed_m_s: float = .015
    descent_speed_m_s: float = .004
    minimum_locator_inset_fraction: float = .75
    supporting_force_weight_fraction: float = .10
    gripper_base_contact_stop_n: float = .15
    control_tracking_tolerance_m: float = .0003
    release_acceptance: str = "independent released capture, fresh RGB-D common corridor and final real pin/base bridge"
    sensor_assumption: str = "latest pre-close RGB-D plus encoder FK assumes no slip; bilateral contact does not observe axial slip"

    def manifest(self):
        return asdict(self)


@dataclass(frozen=True)
class EndStopStablePlacementConfig:
    """Declared development-stage acceptance envelope for stable-supported placement.

    The observation window is an experiment configuration for this local mode,
    not a hardware safety guarantee.  Contact tolerances keep ordinary
    numerical contact jitter from becoming a new strict gate.
    """
    schema: str = "twingraph.end_stop_stable_supported.v17a"
    observation_window_s: float = .5
    window_samples: int = 10
    minimum_support_force_n: float = .01
    press_contact_force_n: float = .05
    finger_support_force_n: float = .01
    penetration_tolerance_m: float = .001

    def manifest(self):
        return asdict(self)


def end_stop_stable_placement_config(session):
    return getattr(session, "end_stop_stable_placement_config_v17a", None) or EndStopStablePlacementConfig()


def evaluate_end_stop_stable_support(session, part, target, *, after_retreat=False):
    """Independent stable-support acceptance; never a motion-correction input.

    V17-A end-stop local mode: the released stop must occupy the declared CAD
    mounting region, be genuinely supported by non-gripper fixtures, and stay
    that way over the declared short observation window.  Ideal-pose residual,
    seating-band depth and printed-locator capture are recorded diagnostics in
    this mode, not gates.
    """
    import mujoco
    if part != "end_stop":
        raise ValueError("stable-supported acceptance is only defined for end_stop")
    config = end_stop_stable_placement_config(session)
    ctx = session.ctx
    target = np.asarray(target, float)
    stop_id = ctx.body_id(part)

    def sample():
        position = ctx.obj_pos(part)
        region_ok, region_row = functional_geometry(part, position, target)
        support_force, finger_force, deepest = 0., 0., 0.
        supporting = []
        for index, contact in enumerate(ctx.data.contact):
            bodies = set(map(int, ctx.model.geom_bodyid[[contact.geom1, contact.geom2]]))
            if stop_id not in bodies or len(bodies) < 2:
                continue
            other = next(int(b) for b in bodies if b != stop_id)
            partner = ctx.model.body(other).name
            deepest = min(deepest, float(contact.dist))
            force = np.zeros(6); mujoco.mj_contactForce(ctx.model, ctx.data, index, force)
            if "finger" in partner:
                finger_force += max(0., float(force[0]))
                continue
            upward = abs(float(np.dot(contact.frame[:3], (0., 0., 1.))))
            if upward < .5 or float(contact.pos[2]) >= float(position[2]):
                continue
            if force[0] > 0:
                support_force += float(force[0]) * upward
                supporting.append(partner)
        row = dict(region_ok=bool(region_ok), stop_offset_m=region_row.get("stop_envelope_offset_m"),
                   support_force_n=support_force, supporting_bodies=sorted(set(supporting)),
                   finger_support_force_n=finger_force, deepest_penetration_m=deepest,
                   stop_position_m=position.tolist())
        row["supported"] = bool(support_force >= config.minimum_support_force_n)
        row["gross_penetration"] = bool(deepest < -config.penetration_tolerance_m)
        row["gripper_supported"] = bool(finger_force >= config.finger_support_force_n)
        row["ok"] = bool(row["region_ok"] and row["supported"] and not row["gross_penetration"]
                         and not (after_retreat and row["gripper_supported"]))
        return row

    rows = [sample()]
    retained = rows[0]["ok"]
    if after_retreat and config.observation_window_s > 0 and config.window_samples > 1:
        interval = config.observation_window_s / (config.window_samples - 1) / float(ctx.control_dt)
        for _ in range(config.window_samples - 1):
            for _ in range(max(1, int(np.ceil(interval - 1e-12)))):
                ctx.step()
            rows.append(sample())
            retained = retained and rows[-1]["ok"]
    metrics = dict(schema=config.schema,
        success=bool(retained), region_ok=rows[0]["region_ok"],
        supported=rows[0]["supported"], support_force_n=rows[0]["support_force_n"],
        supporting_bodies=rows[0]["supporting_bodies"],
        gripper_released=bool(getattr(session, "held", None) is None),
        finger_support_force_n=rows[0]["finger_support_force_n"],
        gripper_still_supporting=rows[0]["gripper_supported"],
        deepest_penetration_m=rows[0]["deepest_penetration_m"],
        gross_penetration=rows[0]["gross_penetration"],
        observation_window_s=float(config.observation_window_s) if after_retreat else 0.,
        window_samples=len(rows), retained_entire_window=bool(retained),
        window_rows=rows, region_criterion="declared CAD mounting envelope via functional_geometry",
        criterion=("released stop supported inside declared region over the declared "
                   "observation window; locator capture and hole alignment are diagnostics"),
        measurement_source="independent_simulator_contact_and_CAD_evaluator")
    return bool(retained), metrics


def functional_stroke_targets(current_x, minimum, bounds):
    """Choose two opposing movements inside calibrated usable rail limits."""
    low, high = map(float, bounds)
    minimum, current_x = float(minimum), float(current_x)
    if not low < high or minimum <= 0:
        raise ValueError("invalid functional stroke calibration")
    candidates = []
    for sign in (-1., 1.):
        first = float(np.clip(current_x + sign * 1.25 * minimum, low, high))
        second = float(np.clip(first - sign * 2.5 * minimum, low, high))
        first_distance = sign * (first-current_x)
        return_distance = -sign * (second-first)
        if first_distance >= 1.1 * minimum and return_distance >= 1.1 * minimum:
            candidates.append((min(first_distance, return_distance), first, second))
    if not candidates:
        raise ValueError("current visual state cannot provide bidirectional task travel inside calibrated guide bounds")
    _, first, second = max(candidates)
    return [first, second]


def configure_v12_skills(session, insertion_policy=None, wiping_policy=None):
    session.functional_acceptance_v12 = True
    session.strict_rgbd_v12 = True
    session.insertion_policy_v12 = str(insertion_policy or CHECKPOINTS / "insert_sensor_bc_v12.npz")
    session.wiping_policy_v12 = str(wiping_policy or CHECKPOINTS / "wipe_imitation_v12.npz")
    session.held_visual_transforms_v12 = {}
    session.arm.object_position = lambda part: control_position(session, part)
    session.arm.continuous_grasp_v12 = True
    session.end_stop_seat_config_v12 = EndStopSeatConfig()
    if hasattr(session, "planning_cad"):
        session.planning_cad["end_stop_seat_search"] = session.end_stop_seat_config_v12.manifest()
    return session


def bounded_end_stop_seat(session, part, target_z, force_stop):
    from .end_stop_seating_v12 import execute
    return execute(session, part, target_z, force_stop)


def printed_visual_templates(installed=False):
    """Compatibility entry point for white, color-independent CAD templates."""
    from simbench.value.cad_rgbd_v12 import templates
    return templates(installed=installed)


def visual_position(session, part):
    observation = session.decision_observation or {}
    if observation.get("backend") != "rgbd_geometry":
        raise ValueError("V12 requires rendered RGB-D; simulator-pose fallback is prohibited")
    row = observation.get("objects", {}).get(part, {})
    if not row.get("valid") or row.get("position_m") is None:
        raise ValueError(f"V12 has no valid RGB-D localization for {part}")
    xyz = np.asarray(row["position_m"], dtype=float)
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        raise ValueError(f"V12 invalid RGB-D position for {part}")
    return xyz


def validated_held_registration(session, part):
    """Validate sensor provenance and algebra, without claiming slip detection.

    A rigid grasp registration plus bilateral contact does not observe axial
    slip. A future independent held-pose/tactile estimator may invalidate the
    registration; missing/invalid registration must never reuse a free pose.
    """
    registration = getattr(session, "held_visual_transforms_v12", {}).get(part)
    if (session.held != part or not registration
            or registration.get("source") != "rgbd_at_grasp_plus_encoder_forward_kinematics"
            or registration.get("valid", True) is not True):
        raise ValueError(f"V12 held RGB-D/FK registration unknown for {part}")
    offset = np.asarray(registration.get("local_position"), float)
    local_R = np.asarray(registration.get("local_rotation"), float)
    eef = np.asarray(session.ctx.eef_pos(), float)
    R = np.asarray(session.ctx.eef_mat(), float)
    if any(x.shape != shape or not np.isfinite(x).all()
           for x, shape in ((offset, (3,)), (local_R, (3, 3)), (eef, (3,)), (R, (3, 3)))):
        raise ValueError("V12 nonfinite or malformed held registration / encoder FK")
    # Algebraic validity only, not a desired assembly orientation tolerance.
    for rotation in (local_R, R):
        if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.e-6, rtol=0.)
                or not np.isclose(np.linalg.det(rotation), 1., atol=1.e-6, rtol=0.)):
            raise ValueError("V12 held registration / encoder rotation is not SO(3)")
    return registration


def bind_grasp_observation(session, part):
    """Bind the latest pre-close image pose to FK at contact establishment.

    No fresh image is acquired here. Object motion during closure and later
    slip remain unobserved until a separate sensor acquires a held-part pose.
    """
    xyz = visual_position(session, part)
    R = np.asarray(session.ctx.eef_mat())
    row = session.decision_observation["objects"][part]
    quat = np.asarray(row["quat_wxyz"], dtype=float)
    if quat.shape != (4,) or not np.isfinite(quat).all() or np.linalg.norm(quat) < 1.e-8:
        raise ValueError("V12 invalid RGB-D quaternion at grasp registration")
    object_R = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
    session.held_visual_transforms_v12[part] = {
        "local_position": R.T @ (xyz - session.ctx.eef_pos()),
        "local_rotation": R.T @ object_R,
        "source": "rgbd_at_grasp_plus_encoder_forward_kinematics",
        "observation_sha256": session.decision_observation.get("sha256"),
        "valid": True,
        "image_acquisition": "latest decision RGB-D observation before contact establishment; not a new held image",
        "rigid_attachment_assumption_verified": False,
        "axial_slip_observable_from_bilateral_contact_and_fk": False,
    }


def control_position(session, part):
    if not getattr(session, "strict_rgbd_v12", False):
        return session.ctx.obj_pos(part).copy()
    if session.held == part:
        transform = validated_held_registration(session, part)
        return session.ctx.eef_pos() + session.ctx.eef_mat() @ transform["local_position"]
    return visual_position(session, part)


def functional_geometry(part, position, target, relative_position=None, cad=None):
    """Coarse occupied-region tests, separate from the final actuation test.

    Bounds correspond to CAD engagement regions, not desired pose precision.
    Pins use the independent finite-bore evaluator instead.
    """
    p, t = np.asarray(position, float), np.asarray(target, float)
    delta = p - t
    if part == "handle":
        if relative_position is None:
            return False, {"reason": "carriage-relative geometry missing"}
        rel = np.asarray(relative_position, float)
        # Ring inner r=6.1mm, post r=5.5mm; ring must surround post and
        # overlap its axial section. No body-origin target-height criterion.
        radial = float(np.linalg.norm(rel[:2]))
        overlap = max(0., min(rel[2] + .008, .062) - max(rel[2] - .008, .040))
        cad = cad or {}
        clearance = float(cad.get("handle_bore_radius_m", .0061)) - float(cad.get("handle_post_radius_m", .0055))
        ok = radial <= clearance and overlap >= .004
        return ok, {"post_ring_overlap_m": overlap, "radial_post_offset_m": radial,
                    "criterion": "ring surrounds post and overlaps its axial section"}
    if part == "carriage":
        # Shoe remains in rail envelope; x is free over the useful rail travel.
        ok = abs(delta[0]) <= .13 and abs(delta[1]) <= .010 and abs(delta[2]) <= .015
        return ok, {"guide_envelope_offset_m": delta.tolist(), "criterion": "shoe occupies guide envelope"}
    if part == "end_stop":
        ok = abs(delta[0]) <= .016 and abs(delta[1]) <= .020 and abs(delta[2]) <= .018
        return ok, {"stop_envelope_offset_m": delta.tolist(), "criterion": "stop occupies mounting region; pins checked separately"}
    return bool(np.linalg.norm(delta) <= .025), {"criterion": "coarse supported placement region", "offset_m": delta.tolist()}


def evaluate_functional_seat(session, part, target):
    """Simulator evaluator only. It provides no controller/perception features."""
    if part == "end_stop":
        ok, metrics = evaluate_end_stop_fixture(session)
        metrics["functional_test_still_required"] = True
        return ok, metrics
    pos = session.ctx.obj_pos(part)
    relative = pos - session.ctx.obj_pos("carriage") if part == "handle" else None
    ok, metrics = functional_geometry(part, pos, target, relative, getattr(session, "planning_cad", None))
    metrics["measurement_source"] = "independent_simulator_acceptance_evaluator"
    metrics["functional_test_still_required"] = part in ("carriage", "handle", "end_stop")
    return bool(ok), metrics


def end_stop_locator_capture(center_base, bottom_corners_base, support_force_n,
                             locator_xy=None, locator_radius_m=.004, locator_top_z_m=.017):
    """Functional lateral capture by the printed base, not ideal-pose error.

    Inputs are expressed in the fixed base CAD frame. Four 8 mm posts surround
    the nominal 24 x 86 mm stop; their X/Y center separations are 30.46/92.46 mm.
    At the current orientation the stop must intercept the rail centerline
    and meet a vertically overlapping post when translated in either rail
    direction. Each post is checked against its local leading bottom edge:
    a raised corner at the opposite end does not remove an existing blocker.
    Actual base support is a separate required contact fact.
    """
    center = np.asarray(center_base, float)
    corners = np.asarray(bottom_corners_base, float)
    posts = np.asarray(locator_xy if locator_xy is not None else
        [[x,y] for x in (-.10723,-.07677) for y in (-.04623,.04623)], float)
    low, high = corners[:, :2].min(axis=0), corners[:, :2].max(axis=0)
    enclosure = bool(np.all(center[:2] >= posts.min(axis=0)) and np.all(center[:2] <= posts.max(axis=0)))
    # The four input corners need not be in perimeter order. On a CCW
    # footprint an edge with increasing Y faces +X; decreasing Y faces -X.
    # Translation along X preserves each material point's Y and Z, so the
    # post's circular Y span clips exactly the possible contact part of that
    # leading edge. Its height is affine along the edge, even for a tilt.
    mean_xy = corners[:,:2].mean(axis=0)
    angles = np.arctan2(corners[:,1]-mean_xy[1], corners[:,0]-mean_xy[0])
    polygon = corners[np.argsort(angles)]
    post_checks = []
    for post in posts:
        direction = int(np.sign(post[0]-center[0]))
        local_bottom = float("inf")
        blocked = False
        for start, finish in zip(polygon, np.roll(polygon, -1, axis=0)):
            delta = finish-start
            if direction * delta[1] <= 1.e-12:
                continue
            t0, t1 = sorted(((post[1]-locator_radius_m-start[1])/delta[1],
                             (post[1]+locator_radius_m-start[1])/delta[1]))
            t0, t1 = max(0.,t0), min(1.,t1)
            if t1-t0 <= 1.e-12:
                continue  # touching the circle's Y tangent alone cannot block
            edge_low = min(start[2]+t0*delta[2], start[2]+t1*delta[2])
            local_bottom = min(local_bottom, float(edge_low))
            if edge_low >= locator_top_z_m:
                continue
            # Restrict to material strictly below the post top. A blocker
            # must also lie ahead along the requested translation direction,
            # rather than behind an edge that has already passed the post.
            if delta[2] > 1.e-12:
                t1 = min(t1, (locator_top_z_m-start[2])/delta[2])
            elif delta[2] < -1.e-12:
                t0 = max(t0, (locator_top_z_m-start[2])/delta[2])
            if t1-t0 <= 1.e-12:
                continue
            for t in (t0, t1):
                point = start+t*delta
                circle_dx = np.sqrt(max(0., locator_radius_m**2-(point[1]-post[1])**2))
                leading_circle_x = post[0]-direction*circle_dx
                travel = direction*(leading_circle_x-point[0])
                if travel >= -1.e-12:
                    blocked = True
        overlap = float(locator_top_z_m-local_bottom) if np.isfinite(local_bottom) else None
        post_checks.append(dict(post_xy_m=post.tolist(), direction_x=direction,
            local_bottom_height_m=float(local_bottom) if np.isfinite(local_bottom) else None,
            vertical_overlap_m=overlap, blocks_translation=bool(blocked)))
    blocked_negative = any(row["blocks_translation"] and row["direction_x"] < 0 for row in post_checks)
    blocked_positive = any(row["blocks_translation"] and row["direction_x"] > 0 for row in post_checks)
    rail_interception = bool(low[1] <= 0. <= high[1])
    best_overlap = {direction: max((row["vertical_overlap_m"] for row in post_checks
        if row["direction_x"] == direction and row["blocks_translation"]), default=0.)
        for direction in (-1,1)}
    vertical_overlap = min(best_overlap.values())
    supported = bool(float(support_force_n) > 1.e-6)
    ok = supported and enclosure and blocked_negative and blocked_positive and rail_interception and vertical_overlap > 0.
    return dict(success=bool(ok), supporting_base_contact=supported,
        support_force_n=float(support_force_n), center_inside_locator_enclosure=enclosure,
        blocked_negative_rail_direction=blocked_negative, blocked_positive_rail_direction=blocked_positive,
        intercepts_rail_centerline=rail_interception, locator_vertical_overlap_m=vertical_overlap,
        locator_post_checks=post_checks,
        highest_bottom_corner_clearance_m=float(locator_top_z_m-np.max(corners[:,2])),
        stop_center_base_m=center.tolist(), bottom_corners_base_m=corners.tolist(),
        criterion="actual base support and printed-post lateral capture; no ideal world-pose residual")


def evaluate_end_stop_fixture(session):
    """Independent evaluator: require functional capture A, report pin bridge B.

    No returned quantity changes a motion command or perception observation.
    A shallow pin in the end-stop alone is never labelled a pin/base bridge.
    The caller chooses the task boundary explicitly; B is diagnostic here.
    """
    from dataclasses import replace
    import mujoco
    from simbench.value.pin_geometry import evaluate_pin_state, evaluate_pin_context
    ctx = session.ctx
    base_position, base_quaternion = ctx.obj_pose("guide_base")
    stop_position, stop_quaternion = ctx.obj_pose("end_stop")
    def matrix(quaternion):
        flat = np.zeros(9); mujoco.mju_quat2Mat(flat, quaternion)
        return flat.reshape(3,3)
    base_R, stop_R = matrix(base_quaternion), matrix(stop_quaternion)
    bottom = np.array([[x,y,-.018] for x in (-.012,.012) for y in (-.043,.043)])
    world_bottom = bottom @ stop_R.T + stop_position
    bottom_base = (world_bottom - base_position) @ base_R
    center_base = (np.asarray(stop_position)-base_position) @ base_R
    base_id, stop_id = ctx.body_id("guide_base"), ctx.body_id("end_stop")
    support_force, supporting_contacts = 0., 0
    for index, contact in enumerate(ctx.data.contact):
        bodies = set(map(int,ctx.model.geom_bodyid[[contact.geom1,contact.geom2]]))
        if bodies != {base_id,stop_id}: continue
        upward_fraction = abs(float(np.dot(contact.frame[:3],base_R[:,2])))
        contact_base = (np.asarray(contact.pos)-base_position) @ base_R
        if upward_fraction < .5 or contact_base[2] >= center_base[2]: continue
        force = np.zeros(6); mujoco.mj_contactForce(ctx.model,ctx.data,index,force)
        if force[0] > 0:
            support_force += float(force[0]) * upward_fraction
            supporting_contacts += 1
    capture = end_stop_locator_capture(center_base,bottom_base,support_force)
    capture["supporting_contact_count"] = supporting_contacts
    # Base is 12 mm thick. B requires a shaft interval through half that plate;
    # this is a separate bridge diagnostic, not the actor's 8 mm stop entry.
    config = replace(session.pin_insertion_config, guide_length_m=.012,
        required_depth_m=.006, plate_hole_half_width_m=.004, guide_inner_radius_m=.004,
        bore_profile=(), aperture_shape="square", source="printed base: 8 mm square hole through 12 mm plate")
    bridge = {}; stop_engagement = {}
    for pin in ("pin_left","pin_right"):
        touching = False; pin_id = ctx.body_id(pin)
        for contact in ctx.data.contact:
            b1,b2 = map(int,ctx.model.geom_bodyid[[contact.geom1,contact.geom2]])
            if pin_id in (b1,b2):
                other = b2 if b1 == pin_id else b1
                touching |= "finger" in ctx.model.body(other).name and contact.dist <= 0
        bridge[pin] = []
        stop_engagement[pin] = evaluate_pin_context(ctx, pin,
            hole_offset_m=(0., -.032 if pin == "pin_left" else .032, 0.),
            released=session.held != pin, touching_finger=touching,
            config=session.pin_insertion_config)
        for y in (-.032,.032):
            entry = np.asarray(base_position) + base_R @ np.array([-.092,y,.006])
            verdict = evaluate_pin_state(ctx.obj_pos(pin),ctx.obj_axis(pin),entry,base_R[:,2],
                released=session.held != pin,touching_finger=touching,
                phase="inserted_after_release",config=config,hole_axes=base_R[:,:2].T)
            bridge[pin].append(verdict)
    matrix_ok = [[row["success"] and stop_engagement[pin]["success"] for row in bridge[pin]]
                 for pin in ("pin_left","pin_right")]
    bridged = bool((matrix_ok[0][0] and matrix_ok[1][1]) or (matrix_ok[0][1] and matrix_ok[1][0]))
    metrics = dict(schema="twingraph.fixture_capture.v12", functional_locator_capture=capture,
        pin_base_bridge_verified=bridged, pin_base_bridge_diagnostic=bridge,
        pin_stop_engagement=stop_engagement,
        pin_base_minimum_depth_m=config.required_depth_m,
        pin_base_bridge_is_required=False,
        measurement_source="independent_simulator_contact_and_CAD_evaluator",
        task_boundary="printed-post functional end-stop capture plus separately checked pin/stop insertion; base bridging reported explicitly")
    return bool(capture["success"]), metrics


def evaluate_pin_joint_engagement(session, part, phase="inserted_after_release"):
    """Independent two-layer acceptance; never use the result for correction.

    Both receiver occupancy/contact predicates must hold at the same sampled
    instant. The base check uses its own 12 mm CAD plate and entry plane, not
    the end-stop adapter's historically implicit +18 mm entry offset.
    """
    from dataclasses import replace
    from simbench.value.pin_contact_v12 import evaluate_context_contact, functional_retention_window
    if part not in ("pin_left", "pin_right"):
        raise ValueError("joint engagement requires a declared pin")
    ctx, cad = session.ctx, session.planning_cad
    index = ("pin_left", "pin_right").index(part)
    stop_offset = cad["pin_hole_offsets_m"][index]
    base_offset = cad["base_hole_offsets_m"][index]
    stop_config = session.pin_insertion_config
    base_config = replace(stop_config, guide_length_m=float(cad["base_receiver_depth_m"]),
        required_depth_m=float(cad["pin_base_minimum_depth_m"]),
        source="original printed base CAD: independently checked second-layer occupancy")

    def sample():
        origin, axis = ctx.obj_pos(part), ctx.obj_axis(part)
        released = session.held != part
        rows = {}
        for fixture, offset, config in (("end_stop", stop_offset, stop_config),
                                         ("guide_base", base_offset, base_config)):
            position, quat = ctx.obj_pose(fixture)
            q = np.asarray(quat, float)
            R = Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
            entry = np.asarray(position) + R @ np.asarray(offset)
            rows[fixture] = evaluate_context_contact(ctx, part, fixture, origin, axis,
                entry, R, released=released, touching_finger=False, phase=phase, config=config)
        return dict(success=all(row["success"] for row in rows.values()), part=part, phase=phase,
            released=bool(released), receivers=rows, sample_time_s=float(ctx.data.time),
            pin_base_bridge_is_required=True,
            measurement_source="independent simulator geometry/contact acceptance; never policy input")

    initial = sample()
    if phase == "inserted_while_held": return initial
    if phase not in ("inserted_after_release", "retained_after_stroke"):
        raise ValueError("unknown joint-engagement phase")
    def wait(seconds):
        for _ in range(max(1,int(np.ceil(seconds/ctx.control_dt-1e-12)))):
            ctx.step()
    result = functional_retention_window(sample,wait,initial,
        duration=stop_config.retention_window_s,samples=stop_config.retention_samples)
    result["functional_retention_window"]["method"] = (
        "sampled simultaneous stop/base occupancy, each original receiver's contact guard and no fingers; no continuous-time guarantee")
    return result


def bounded_pin_press(session, part, target_z, force_stop):
    """Single integration entry for the explicit-depth pin controller."""
    from .sensor_learning_v12 import bounded_pin_press as execute
    return execute(session,part,target_z,force_stop)
