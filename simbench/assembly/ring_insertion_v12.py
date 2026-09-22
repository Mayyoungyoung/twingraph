"""White printed ring insertion: sensor BC plus bounded contact recovery.

The learned actor is the existing behavior-cloned sensor policy, not RL.
Control uses a frozen RGB-D grasp registration, encoder FK, and the existing
simulation force/tactile adapter. Simulator geometry is evaluated once after
control has ended, never to select search directions or change the target.
"""
import hashlib
from pathlib import Path
import numpy as np

SCHEMA = "twingraph.ring_sensor_recovery.v12.r3_force_stop_contract"


def search_parameters(observation, cad):
    """Scale the bounded scan using observed uncertainty and declared CAD."""
    sigmas = []
    for part in ("handle", "carriage"):
        row = observation.get("objects", {}).get(part, {})
        if not row.get("valid") or row.get("position_m") is None:
            raise ValueError(f"ring recovery requires valid RGB-D {part}")
        sigma = float(row.get("fit_residual_m") or 0.)
        if not np.isfinite(sigma) or sigma < 0:
            raise ValueError("invalid RGB-D uncertainty")
        sigmas.append(max(.0005, sigma))
    clearance = float(cad["handle_bore_radius_m"]) - float(cad["handle_post_radius_m"])
    height = float(cad["parts"]["handle"]["dimensions_m"][2])
    if not 0 < clearance <= .003 or not 0 < height < .1:
        raise ValueError("ring/post CAD dimensions outside recovery envelope")
    registration_sigma = .0005
    sigma = float(np.sqrt(sum(s*s for s in sigmas) + registration_sigma**2))
    return dict(radius_m=min(.003, 3*sigma), uncertainty_sigma_m=sigma,
        visual_sigma_floor_m=.0005, registration_sigma_m=registration_sigma,
        radial_clearance_m=clearance, spiral_pitch_m=.8*clearance,
        entry_encoder_descent_m=min(.004, height*.25),
        functional_encoder_feed_m=.5*height, ring_height_m=height)


def bounded_target(command, position, center, radius, target_z):
    """Bound desired object coordinates; never command below the CAD seat."""
    command = np.asarray(command, float).copy()
    offset = command[:2] - center
    length = float(np.linalg.norm(offset))
    if length > radius:
        command[:2] = center + offset * radius/length
    # Avoid unbounded compression if the arm cannot realize a set point.
    command[2] = np.clip(command[2], max(target_z, position[2]-.0003),
                         max(target_z, position[2]+.001))
    return command


def execute(session, part, target_z, force_stop):
    """Return metrics; caller wraps metrics['success'] in the skill Result.

    A caller must not interpret force contact or encoder descent as success.
    Only the independent, final functional evaluator owns that label.
    """
    from .skills_v12 import control_position, evaluate_functional_seat
    from .sensor_learning_v12 import load_actor
    if part != "handle" or session.held != part:
        raise ValueError("ring recovery requires the held handle")
    if not getattr(session, "strict_rgbd_v12", False):
        raise ValueError("ring recovery prohibits simulator-pose control fallback")
    registration = getattr(session, "held_visual_transforms_v12", {}).get(part)
    if not registration or registration.get("source") != "rgbd_at_grasp_plus_encoder_forward_kinematics":
        raise ValueError("ring recovery requires RGB-D to FK grasp registration")
    observation = session.decision_observation or {}
    if observation.get("backend") != "rgbd_geometry":
        raise ValueError("ring recovery requires RGB-D observations")
    cad = session.planning_cad
    config = search_parameters(observation, cad)
    target = np.asarray(session.stage_targets[part], float).copy()
    target_z, force_stop_n = float(target_z), float(force_stop)
    if not np.isfinite(target).all() or not np.isfinite(target_z) or not 0 < force_stop_n <= 4.:
        raise ValueError("invalid declared ring target or existing press-force envelope")
    if not np.isclose(target_z, target[2], rtol=0., atol=1.e-9):
        raise ValueError("press target differs from the declared CAD seat target")
    target[2] = target_z
    center = target[:2].copy()
    policy = Path(session.insertion_policy_v12)
    actor = load_actor(policy)
    dt = float(session.ctx.control_dt)
    rotation = session.arm.rotation.copy()  # preserve the planned upright grasp
    previous = control_position(session, part).copy()
    command = previous.copy()
    if np.linalg.norm(previous[:2]-center) > config["radius_m"]:
        raise ValueError("initial registration lies outside visual recovery neighborhood")
    initial_force = float(session.external_force(part))
    peak = initial_force
    trace, contact_z, entry, theta, low_steps = [], None, False, 0., 0
    probe_xy, probe_high_steps = None, 0
    desired_contact = min(.15, .15*force_stop_n)
    # Search must activate before the admittance controller unloads contact.
    contact_threshold = .5*desired_contact
    low_threshold = min(.10, .1*force_stop_n)
    reason, safety_ok = "step_budget_exhausted", True
    # At 50 Hz this is an 80 s bounded recovery, including the CAD descent.
    max_steps = int(np.ceil(80./dt))
    initial_unloading = initial_force >= .6*force_stop_n
    for step in range(max_steps):
        position = control_position(session, part)
        force = float(session.external_force(part))
        peak = max(peak, force)
        if not np.isfinite(force) or force < 0:
            safety_ok, reason = False, "invalid_force_sensor"; break
        if not session.ctx.grasp_contacts(part)["held"]:
            safety_ok, reason = False, "grasp_lost"; break
        # press.force_stop is a contact stopping threshold, matching the
        # parent press skill. Crossing it ends control; the independent final
        # seat predicate decides whether contact was seating or a blockage.
        # No separate hard force limit is declared by this press interface.
        if force >= force_stop_n:
            reason = "force_stop_reached"; break
        # Existing guarded descent can leave residual load. The only command
        # allowed during this first phase is upward unloading, not more force.
        if initial_unloading:
            if force < desired_contact:
                initial_unloading = False
                command = position.copy()
            elif step >= int(1./dt):
                safety_ok, reason = False, "initial_contact_could_not_unload"; break
        if contact_z is None and force >= contact_threshold and not initial_unloading:
            contact_z = float(position[2])
        if contact_z is not None:
            low_steps = low_steps+1 if force < low_threshold else 0
        if contact_z is not None and not entry:
            # Moving a spiral continuously can leave a narrow aperture before
            # enough insertion has accumulated to confirm entry. Pause XY
            # after a small encoder descent plus sustained unloading, then
            # confirm using the stricter depth criterion below.
            if probe_xy is None and contact_z-position[2] >= .0002 and low_steps >= 5:
                probe_xy = position[:2].copy()
                command[:2] = probe_xy
                probe_high_steps = 0
            if probe_xy is not None:
                probe_high_steps = probe_high_steps+1 if force > 2*desired_contact else 0
                if probe_high_steps >= 10:
                    probe_xy = None
                    probe_high_steps = 0
            if contact_z-position[2] >= config["entry_encoder_descent_m"] and low_steps >= 10:
                entry = True
                target[:2] = position[:2]
        if entry and contact_z-position[2] >= config["functional_encoder_feed_m"] and low_steps >= 10:
            # The task requires ring/post engagement, not pressing onto a
            # lower support after engagement is complete. Half the declared
            # ring height is a sensor-space stopping rule; acceptance still
            # independently checks the actual surrounding bore and overlap.
            reason = "functional_encoder_feed_completed"; break
        goal = target.copy()
        mode = "behavior_cloned_feed" if entry or contact_z is None else "bounded_contact_spiral"
        if probe_xy is not None and not entry:
            goal[:2] = probe_xy
            mode = "encoder_descent_probe_hold_xy"
        elif contact_z is not None and not entry:
            r = min(config["radius_m"], config["spiral_pitch_m"]*theta/(2*np.pi))
            goal[:2] = center+r*np.array([np.cos(theta), np.sin(theta)])
            theta += min(.06, .002*dt/max(r, .0003))
            if r >= config["radius_m"]:
                reason = "visual_search_neighborhood_exhausted"; break
        error = goal-position
        sensor_state = np.r_[error/.003, (position-previous)/dt/.01, force/force_stop_n]
        action = np.asarray(actor(sensor_state), float)
        previous = position.copy()
        # The BC actor supplies bounded tracking velocities. Explicit contact
        # admittance replaces only Z near the post rim; its input is force.
        delta = action * .003 * dt
        if initial_unloading:
            mode = "unload_previous_guarded_contact"
            delta = np.array([0., 0., .002*dt])
        elif force >= desired_contact:
            delta[2] = min(.002*dt, max(.00025*dt, (force-desired_contact)*.004*dt))
        else:
            delta[2] = np.clip(delta[2], -.001*dt, .001*dt)
        command += delta
        command = bounded_target(command, position, center, config["radius_m"], target_z)
        if (entry or contact_z is None) and position[2] <= target_z+.0002 and force < force_stop_n:
            reason = "declared_encoder_target_reached"; break
        eef = session.ctx.eef_pos()+command-position
        session.arm.servo(eef, rotation=rotation)
        if step % 10 == 0:
            trace.append(dict(step=step, mode=mode, position_m=position.tolist(),
                desired_object_position_m=command.tolist(), desired_eef_position_m=eef.tolist(),
                force_n=force, actor_observation=sensor_state.tolist(), actor_action=action.tolist(),
                search_angle_rad=theta, entry_inferred=entry))
    final_force = float(session.external_force(part))
    peak = max(peak, final_force)
    if not np.isfinite(final_force) or final_force < 0:
        safety_ok, reason = False, "invalid_force_sensor"
    elif not session.ctx.grasp_contacts(part)["held"]:
        safety_ok, reason = False, "grasp_lost"
    elif safety_ok and final_force >= force_stop_n:
        reason = "force_stop_reached"
    # This is deliberately after the last control action. No evaluator output
    # is read by the policy, search, contact adaptation, or commanded target.
    accepted, geometry = evaluate_functional_seat(session, part, target)
    metrics = dict(schema=SCHEMA, success=bool(accepted and safety_ok),
        controller="behavior_cloned_sensor_actor_plus_explicit_bounded_contact_recovery",
        policy=str(policy), policy_sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),
        input_source="RGB-D grasp registration + encoder FK + existing simulated force/tactile adapter",
        hardware_sensor_validation=False, declared_target_m=[*center.tolist(), target_z],
        adapted_lateral_target_m=target[:2].tolist(), requested_rotation=rotation.tolist(),
        search={**config, "trigger_force_n": contact_threshold, "regulated_contact_force_n": desired_contact},
        entry_inferred_from_encoder_and_force=entry, steps=step+1,
        stop_reason=reason, safety_ok=safety_ok, force_stop_n=force_stop_n,
        force_stop_crossed=bool(peak >= force_stop_n),
        force_stop_overshoot_n=max(0., peak-force_stop_n),
        hard_force_limit_n=None,
        force_contract="press contact-stop threshold; no distinct hard force limit declared by this interface",
        initial_force_n=initial_force, peak_force_n=peak, final_force_n=final_force,
        independent_functional_acceptance=geometry, independent_functional_success=bool(accepted),
        evaluation_source="one final simulator evaluator call; never an online control input",
        trace=trace)
    session.artifacts["ring_insertion_v12"] = metrics
    return metrics
