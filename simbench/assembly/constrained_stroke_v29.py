"""Versioned constrained guide motion used by the completed assembly stroke.

V28 keeps the historical world-X handle target and scalar normal-force sum for
paired replay.  V29 tracks the carriage in the configured guide coordinate,
preserves the live handle/carriage grasp relation, and advances the command
with bounded lead over measured carriage progress.
"""
from __future__ import annotations

import numpy as np
import mujoco


STROKE_FIXED_V28 = "fixed_world_target_v28"
STROKE_PROGRESS_V29 = "guide_progress_v29"
STROKE_MOTION_POLICIES = {
    STROKE_FIXED_V28: dict(
        controller_version="constrained_stroke.fixed_world_target.v28",
        speed_m_s=.04,
        force_limit_n=14.,
    ),
    STROKE_PROGRESS_V29: dict(
        controller_version="constrained_stroke.guide_progress.v29",
        speed_m_s=.04,
        force_limit_n=14.,
        maximum_command_lead_m=.0012,
        soft_load_fraction=.70,
        maximum_unload_m=.00025,
        target_boundary_margin_m=.0015,
        completion_tolerance_m=.00035,
        no_progress_window_s=1.0,
    ),
}


def _unit(vector):
    value = np.asarray(vector, float)
    norm = float(np.linalg.norm(value))
    if norm <= 1.e-12:
        raise ValueError("guide axis must be nonzero")
    return value / norm


def guide_coordinate(position, axis=(1., 0., 0.), origin=(0., 0., 0.)):
    return float(np.dot(np.asarray(position, float) - np.asarray(origin, float), _unit(axis)))


def live_stroke_plan(tracked_position, minimum, bounds, *, axis=(1., 0., 0.),
                     origin=(0., 0., 0.), preferred_first_direction=None,
                     boundary_margin=None):
    """Plan legal opposing targets from the post-grasp tracked body state."""
    from .skills_v12 import functional_stroke_targets
    start = guide_coordinate(tracked_position, axis, origin)
    low, high = map(float, bounds)
    margin = float(STROKE_MOTION_POLICIES[STROKE_PROGRESS_V29][
        "target_boundary_margin_m"] if boundary_margin is None else boundary_margin)
    control_bounds = (low + margin, high - margin)
    if not control_bounds[0] < control_bounds[1]:
        raise ValueError("stroke boundary margin removes the legal guide interval")
    targets = None
    preferred = None if preferred_first_direction is None else float(
        np.sign(preferred_first_direction))
    if preferred in (-1., 1.):
        first = float(np.clip(start + preferred * 1.25 * float(minimum), *control_bounds))
        second = float(np.clip(first - preferred * 2.5 * float(minimum), *control_bounds))
        first_distance = preferred * (first - start)
        return_distance = -preferred * (second - first)
        if first_distance >= 1.1 * float(minimum) and return_distance >= 1.1 * float(minimum):
            targets = [first, second]
    if targets is None:
        targets = functional_stroke_targets(start, float(minimum), control_bounds)
    return dict(
        coordinate_frame="configured_guide_axis",
        guide_axis_world=_unit(axis).tolist(),
        guide_origin_world=np.asarray(origin, float).tolist(),
        tracked_start_coordinate_m=float(start),
        calibrated_bounds_m=list(map(float, bounds)),
        command_bounds_m=list(map(float, control_bounds)),
        target_boundary_margin_m=margin,
        targets_m=list(map(float, targets)),
        required_each_direction_m=float(minimum),
        preferred_first_direction=int(preferred) if preferred in (-1., 1.) else None,
    )


def contact_load_report(session, part, axis=(1., 0., 0.), exclude_bodies=()):
    """Resolve live contacts into world and configured-guide coordinates.

    ``mj_contactForce`` is expressed in each contact frame.  The first row of
    ``contact.frame`` is its world normal; the transformed vector is signed as
    the force acting on ``part``.  The legacy scalar is retained separately.
    """
    axis = _unit(axis)
    bid = session.ctx.body_id(part)
    excluded = set(map(str, exclude_bodies))
    vector = np.zeros(3)
    legacy_sum = 0.
    largest = 0.
    rows = []
    for index, contact in enumerate(session.ctx.data.contact):
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        body1, body2 = map(int, session.ctx.model.geom_bodyid[[geom1, geom2]])
        if bid not in (body1, body2):
            continue
        other = body2 if body1 == bid else body1
        other_name = str(session.ctx.model.body(other).name)
        if "finger" in other_name or other_name in ("eef", "right_gripper", "right_hand"):
            continue
        if other_name in excluded:
            continue
        local = np.zeros(6)
        mujoco.mj_contactForce(session.ctx.model, session.ctx.data, index, local)
        normal_force = max(0., float(local[0]))
        frame = np.asarray(contact.frame, float).reshape(3, 3)
        world = frame.T @ local[:3]
        # MuJoCo reports the contact-frame force on geom2.  Reverse it for a
        # tracked body on geom1 so vector sums have a consistent body sign.
        if bid == body1:
            world = -world
        vector += world
        legacy_sum += normal_force
        largest = max(largest, normal_force)
        rows.append(dict(
            contact_index=int(index),
            part_geom=str(session.ctx.model.geom(geom1 if body1 == bid else geom2).name),
            other_body=other_name,
            other_geom=str(session.ctx.model.geom(geom2 if body1 == bid else geom1).name),
            distance_m=float(contact.dist),
            normal_force_n=normal_force,
            normal_world=frame[0].tolist(),
            force_on_part_world_n=world.tolist(),
            guide_force_n=float(np.dot(world, axis)),
        ))
    axial = float(np.dot(vector, axis))
    lateral = vector - axial * axis
    resultant = float(np.linalg.norm(vector))
    return dict(
        part=str(part),
        force_frame="world vector resolved from MuJoCo contact frames; guide projection uses configured axis",
        guide_axis_world=axis.tolist(),
        legacy_unsigned_normal_sum_n=float(legacy_sum),
        resultant_world_n=vector.tolist(),
        resultant_magnitude_n=resultant,
        guide_axis_force_n=axial,
        lateral_resultant_n=float(np.linalg.norm(lateral)),
        largest_single_contact_n=float(largest),
        hard_load_n=float(max(resultant, largest)),
        contacts=rows,
    )


def _trace_row(session, step, command_progress, tracked_part, start_tracked,
               axis, origin, driven_report, tracked_report):
    tracked = session.ctx.obj_pos(tracked_part).copy()
    return dict(
        step=int(step),
        sim_time_s=float(session.ctx.data.time),
        command_progress_m=float(command_progress),
        tracked_position_world_m=tracked.tolist(),
        tracked_coordinate_m=guide_coordinate(tracked, axis, origin),
        actual_progress_m=float(np.dot(tracked - start_tracked, axis)),
        driven_load=driven_report,
        tracked_external_load=tracked_report,
    )


def execute_fixed_v28(session, part, target_x, max_force, *, tracked_part="carriage",
                      guide_axis=(1., 0., 0.), guide_origin=(0., 0., 0.), speed=.04):
    """Historical command behavior plus diagnostics; used only for pairing."""
    from .library import Result
    from .skills_v12 import control_position
    axis = _unit(guide_axis)
    start = control_position(session, part).copy()
    evaluated_start = session.ctx.obj_pos(part).copy()
    tracked_start = session.ctx.obj_pos(tracked_part).copy()
    target = start.copy(); target[0] = float(target_x)
    offset = session.ctx.eef_pos() - start
    grasp_part = session.held if (session.held == "handle" and part == "carriage") else part
    peak = 0.; peak_event = None; trace = []
    duration = max(1.0, 1.9 * np.linalg.norm(target - start) / float(speed))
    samples = max(2, int(duration / session.ctx.control_dt))
    for step, t in enumerate(np.linspace(0., 1., samples)):
        smooth = t ** 3 * (10. - 15. * t + 6. * t * t)
        command = start + smooth * (target - start)
        session.arm.servo(command + offset)
        driven = contact_load_report(session, part, axis)
        tracked = contact_load_report(session, tracked_part, axis, exclude_bodies=(part,))
        load = float(driven["legacy_unsigned_normal_sum_n"])
        if load > peak:
            peak = load
            peak_event = _trace_row(session, step, float(np.dot(command - start, axis)),
                                    tracked_part, tracked_start, axis, guide_origin, driven, tracked)
        if step % 10 == 0:
            trace.append(_trace_row(session, step, float(np.dot(command - start, axis)),
                                    tracked_part, tracked_start, axis, guide_origin, driven, tracked))
        if peak > max_force:
            return Result(False, dict(
                policy=STROKE_FIXED_V28,
                controller_version=STROKE_MOTION_POLICIES[STROKE_FIXED_V28]["controller_version"],
                peak_force_n=float(peak),
                force_limit_n=float(max_force),
                force_statistic="unsigned sum of all non-gripper contact normal forces on driven part",
                peak_event=peak_event,
                trace=trace,
            ), "contact overload")
        contact = session.ctx.grasp_contacts(grasp_part)
        if not contact["held"]:
            return Result(False, dict(contact=contact, held_part=grasp_part,
                object=session.ctx.obj_pos(part), axis=session.ctx.obj_axis(part),
                eef=session.ctx.eef_pos(), peak_force_n=float(peak), peak_event=peak_event,
                trace=trace), "grasp lost")
    session.hold(.35)
    requested = target - start
    length = float(np.linalg.norm(requested))
    actual_delta = session.ctx.obj_pos(part) - evaluated_start
    progress = float(np.dot(actual_delta, requested / max(length, 1.e-9)))
    actual_tracked = session.ctx.obj_pos(tracked_part).copy()
    return Result(progress >= .7 * length, dict(
        policy=STROKE_FIXED_V28,
        controller_version=STROKE_MOTION_POLICIES[STROKE_FIXED_V28]["controller_version"],
        requested_travel_m=length,
        measured_progress_m=progress,
        tracked_progress_m=float(np.dot(actual_tracked - tracked_start, axis)),
        peak_force_n=float(peak),
        force_limit_n=float(max_force),
        force_statistic="unsigned sum of all non-gripper contact normal forces on driven part",
        peak_event=peak_event,
        trace=trace,
        measurement_source="independent_simulator_motion_evaluator",
    ), "insufficient physical motion progress")


def execute_progress_v29(session, part, target_coordinate, max_force, *, tracked_part="carriage",
                         guide_axis=(1., 0., 0.), guide_origin=(0., 0., 0.), speed=.04):
    """Bounded, force-aware motion driven by measured guide progress."""
    from .library import Result
    axis = _unit(guide_axis)
    origin = np.asarray(guide_origin, float)
    config = STROKE_MOTION_POLICIES[STROKE_PROGRESS_V29]
    start_tracked = session.ctx.obj_pos(tracked_part).copy()
    start_coordinate = guide_coordinate(start_tracked, axis, origin)
    delta = float(target_coordinate) - start_coordinate
    direction = 1. if delta >= 0. else -1.
    distance = abs(delta)
    eef_start = session.ctx.eef_pos().copy()
    driven_start = session.ctx.obj_pos(part).copy()
    grasp_part = session.held if (session.held == "handle" and part == "carriage") else part
    dt = float(session.ctx.control_dt)
    nominal_duration = max(1., 1.9 * distance / float(speed))
    hard_duration = max(3., 3.5 * distance / float(speed))
    nominal_steps = max(2, int(np.ceil(nominal_duration / dt)))
    hard_steps = max(nominal_steps, int(np.ceil(hard_duration / dt)))
    no_progress_steps = max(2, int(np.ceil(float(config["no_progress_window_s"]) / dt)))
    lead = float(config["maximum_command_lead_m"])
    unload_limit = float(config["maximum_unload_m"])
    soft_limit = float(config["soft_load_fraction"]) * float(max_force)
    tolerance = float(config["completion_tolerance_m"])
    command_progress = 0.; peak = 0.; peak_event = None; trace = []
    progress_window = []
    unloading = 0.
    stop_reason = "motion_time_limit"
    for step in range(hard_steps):
        tracked_position = session.ctx.obj_pos(tracked_part).copy()
        signed_progress = direction * float(np.dot(tracked_position - start_tracked, axis))
        progress_window.append(signed_progress)
        progress_window = progress_window[-no_progress_steps:]
        if signed_progress >= distance - tolerance:
            stop_reason = "tracked_target_reached"
            break
        phase = min(1., step / max(1, nominal_steps - 1))
        smooth = phase ** 3 * (10. - 15. * phase + 6. * phase * phase)
        nominal_progress = smooth * distance
        driven_report = contact_load_report(session, part, axis)
        tracked_report = contact_load_report(session, tracked_part, axis, exclude_bodies=(part,))
        hard_load = max(float(driven_report["hard_load_n"]),
                        float(tracked_report["hard_load_n"]))
        if hard_load > peak:
            peak = hard_load
            peak_event = _trace_row(session, step, direction * command_progress,
                                    tracked_part, start_tracked, axis, origin,
                                    driven_report, tracked_report)
        if hard_load > max_force:
            stop_reason = "contact_overload"
            break
        allowed_lead = lead
        if hard_load >= soft_limit:
            allowed_lead = max(.00035, .35 * lead)
            excess = max(0., command_progress - max(0., signed_progress) - allowed_lead)
            unload = min(unload_limit - unloading, excess)
            if unload > 0.:
                command_progress -= unload
                unloading += unload
        desired = min(distance, nominal_progress, max(0., signed_progress) + allowed_lead)
        maximum_increment = float(speed) * dt * (.35 if hard_load >= soft_limit else 1.)
        command_progress = min(distance, max(command_progress,
            min(desired, command_progress + maximum_increment)))
        session.arm.servo(eef_start + direction * command_progress * axis)
        if step % 10 == 0:
            trace.append(_trace_row(session, step, direction * command_progress,
                                    tracked_part, start_tracked, axis, origin,
                                    driven_report, tracked_report))
        contact = session.ctx.grasp_contacts(grasp_part)
        if not contact["held"]:
            stop_reason = "grasp_lost"
            break
        if (len(progress_window) == no_progress_steps
                and max(progress_window) - min(progress_window) < 2.5e-5
                and phase >= 1. and command_progress >= min(distance, lead) - 1.e-6):
            stop_reason = "no_measured_carriage_progress"
            break
    else:
        step = hard_steps
    session.hold(.35)
    final_tracked = session.ctx.obj_pos(tracked_part).copy()
    actual_progress = direction * float(np.dot(final_tracked - start_tracked, axis))
    if (actual_progress >= distance - tolerance
            and stop_reason in ("no_measured_carriage_progress", "motion_time_limit")):
        stop_reason = "tracked_target_reached_after_settle"
    cross_axis = (final_tracked - start_tracked) - np.dot(
        final_tracked - start_tracked, axis) * axis
    metrics = dict(
        policy=STROKE_PROGRESS_V29,
        controller_version=config["controller_version"],
        coordinate_frame="configured guide axis",
        guide_axis_world=axis.tolist(),
        guide_origin_world=origin.tolist(),
        tracked_part=str(tracked_part),
        driven_part=str(part),
        start_tracked_position_world_m=start_tracked.tolist(),
        start_driven_position_world_m=driven_start.tolist(),
        target_coordinate_m=float(target_coordinate),
        requested_travel_m=float(distance),
        commanded_progress_m=float(command_progress),
        measured_progress_m=float(actual_progress),
        final_tracked_position_world_m=final_tracked.tolist(),
        cross_axis_m=float(np.linalg.norm(cross_axis)),
        peak_force_n=float(peak),
        force_limit_n=float(max_force),
        force_statistic=("maximum of resultant contact load and largest single contact; "
                         "driven-interface and tracked-body external loads checked separately"),
        peak_event=peak_event,
        limited_unload_m=float(unloading),
        stop_reason=stop_reason,
        steps=int(step + 1),
        nominal_duration_s=float(nominal_duration),
        hard_duration_s=float(hard_duration),
        trace=trace,
        measurement_source="live simulator tracked-body pose and resolved contact force",
    )
    if stop_reason == "contact_overload":
        return Result(False, metrics, "contact overload")
    if stop_reason == "grasp_lost":
        return Result(False, metrics, "grasp lost")
    if stop_reason == "no_measured_carriage_progress":
        return Result(False, metrics, "no measured carriage progress")
    return Result(actual_progress >= distance - tolerance, metrics,
                  "insufficient physical motion progress")


def execute(session, part, target_x, max_force, *, policy, tracked_part="carriage",
            guide_axis=(1., 0., 0.), guide_origin=(0., 0., 0.), speed=.04):
    if policy == STROKE_FIXED_V28:
        return execute_fixed_v28(session, part, target_x, max_force,
                                 tracked_part=tracked_part, guide_axis=guide_axis,
                                 guide_origin=guide_origin, speed=speed)
    if policy == STROKE_PROGRESS_V29:
        return execute_progress_v29(session, part, target_x, max_force,
                                    tracked_part=tracked_part, guide_axis=guide_axis,
                                    guide_origin=guide_origin, speed=speed)
    from .library import Result
    return Result(False, reason="unknown constrained stroke motion policy")
