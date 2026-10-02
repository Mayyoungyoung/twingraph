"""Released-object planar pushing from observed geometry and tactile feedback.

No task names, success labels, object-state writes, or grasp attachment.
The caller performs detection, planning, approach and empty jaw closure.
"""
import math
import numpy as np
from scipy.spatial.transform import Rotation
import mujoco


def finger_contact_force(ctx, part):
    """External fingertip AND finger-shell contact, excluding finger/finger."""
    bid = ctx.body_id(part)
    total = 0.
    for i, c in enumerate(ctx.data.contact):
        bodies = tuple(int(ctx.model.geom_bodyid[g]) for g in (c.geom1, c.geom2))
        if bid not in bodies:
            continue
        other = bodies[1] if bodies[0] == bid else bodies[0]
        if 'finger' not in ctx.model.body(other).name:
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(ctx.model, ctx.data, i, force)
        total += max(0., float(force[0]))
    return total


def estimate(pose, geometry, grasp_region, target, axis, *, clearance=.004):
    position = np.asarray(pose['xyz'], float)
    axis, target = np.asarray(axis, float), np.asarray(target, float)
    if (axis.shape != (3,) or target.shape != (3,) or not np.isfinite(target).all()
            or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-9):
        raise ValueError('finite target and nonzero push axis required')
    axis = axis / np.linalg.norm(axis)
    if abs(axis[2]) > .02:
        raise ValueError('this controller supports horizontal pushing')
    rotation = Rotation.from_quat(np.asarray(pose['quat'])[[1, 2, 3, 0]]).as_matrix()
    lo, hi = grasp_region['body_z_interval_m']
    # Push in the lower portion of an exposed contact region to reduce the
    # overturning moment. The end-effector pad centre is 3.6 mm above its TCP.
    contact_height = lo + .4 * (hi - lo)
    points = []
    for g in geometry:
        if g['type'] != 'box':
            continue
        c, h = np.asarray(g['pos']), np.asarray(g['size'])
        if not c[2]-h[2] <= contact_height <= c[2]+h[2]:
            continue
        local = np.array([[c[0]+x*h[0], c[1]+y*h[1], contact_height]
                          for x in (-1,1) for y in (-1,1)])
        points.extend(position + local @ rotation.T)
    if not points:
        raise ValueError('no declared exposed box face at push height')
    rear_extent = min(float(np.dot(p-position, axis)) for p in points)
    # Original Panda pad half-width along the push axis: 8 mm.
    offset = -rear_extent + .008
    contact = position - axis*(offset + clearance)
    contact[2] = position[2] + contact_height - .0036
    hover = contact + [0., 0., .10]
    if np.dot(target-position, axis) < .01:
        raise ValueError('target must require at least 10 mm of forward pushing')
    return dict(position=position.tolist(), target=target.tolist(), axis=axis.tolist(),
                contact=contact.tolist(), hover=hover.tolist(), rear_offset_m=offset,
                yaw=math.atan2(axis[1], axis[0]), contact_height_m=contact[2]-position[2],
                minimum_progress_m=.01, tolerance_m=.006,
                source='observed pose and declared CAD exposed contact region')


def execute(session, part, plan):
    from .library import Result
    from .skills_v12 import control_position
    ctx = session.ctx
    axis = np.asarray(plan['axis'], float)
    target = np.asarray(plan['target'], float)
    speed, limit = float(plan['speed']), float(plan['force_limit'])
    if session.held is not None or ctx.pad_span() > .020:
        return Result(False, reason='push requires closed empty jaws')
    start = control_position(session, part)
    command = ctx.eef_pos().copy()
    offset = float(plan['rear_offset_m'])
    if np.dot(start-command,axis) < offset-.004:
        return Result(False, reason='pusher must approach the rear face before execution')
    dt = float(ctx.control_dt)
    distance = float(np.dot(target-start, axis))
    if distance < plan['minimum_progress_m']:
        return Result(False, reason='object moved before pushing; replan required')
    budget = min(2500, int((distance+.04)/speed/dt)+200)
    contacts, peak, blocked, over_limit, straddled = 0, 0., 0, 0, False
    previous = start.copy()
    reason = 'motion budget exhausted'
    contact_progress = 0.
    for step in range(budget):
        position = control_position(session, part)
        error = float(np.dot(target-position,axis))
        force = finger_contact_force(ctx, part)
        peak = max(peak, force)
        if force > .05:
            contacts += 1
            contact_progress += max(0., float(np.dot(position-previous, axis)))
        previous = position.copy()
        gap = float(np.dot(position-ctx.eef_pos(),axis))
        straddled |= gap < offset-.010 or ctx.pad_span() > .020
        if straddled:
            reason = 'pusher ceased to be closed and behind the object'; break
        if abs(error) <= plan['tolerance_m']:
            reason = ''; break
        if error < -plan['tolerance_m']:
            reason = 'object overshot push target'; break
        # Cartesian admittance with bounded command lead. A jam may fail;
        # it may never accumulate an arbitrarily large penetration command.
        step_m = speed*dt if force < limit else -.0001
        command += axis*step_m
        lead = float(np.dot(command-ctx.eef_pos(),axis))
        if lead > .001:
            command -= axis*(lead-.001)
        lateral = position-command
        lateral -= axis*np.dot(lateral,axis)
        lateral[2] = 0.
        command += np.clip(lateral, -.0001, .0001)
        session.arm.servo(command)
        moved = float(np.dot(control_position(session,part)-position,axis))
        blocked = blocked+1 if moved < 1e-5 else 0
        over_limit = over_limit+1 if force > 2*limit else 0
        if blocked > 200 or over_limit > 40:
            reason = 'contact stalled or sustained overload'; break
    final = control_position(session,part)
    progress = float(np.dot(final-start,axis))
    error = float(np.dot(target-final,axis))
    lateral = target-final-axis*np.dot(target-final,axis)
    ok = bool(not reason and abs(error) <= plan['tolerance_m']
              and np.linalg.norm(lateral[:2]) <= plan['tolerance_m']
              and progress >= plan['minimum_progress_m'] and contacts >= 3
              and contact_progress >= max(.005, .5*progress) and peak >= plan['press_force']
              and not straddled and session.held is None)
    return Result(ok,dict(start_m=start.tolist(),final_m=final.tolist(), progress_m=progress,
        target_error_m=error,lateral_error_m=float(np.linalg.norm(lateral[:2])),
        contact_progress_m=contact_progress,contact_steps=contacts,peak_finger_contact_n=peak,
        blocked_steps=blocked,over_limit_steps=over_limit,straddled_seen=straddled,
        jaw_span_m=float(ctx.pad_span()),held=session.held,controller='planar_admittance_v32'),
        '' if ok else reason or 'insufficient physical rear-contact progress')
