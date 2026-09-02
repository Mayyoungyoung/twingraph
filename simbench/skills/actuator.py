"""Actuator skills: position-ramp and force-controlled drive for fixture
actuators (pins, clamps, side-pushers).

Ported from taskC_fixture._ramp_act / _drive_until_force as generic,
reusable skills so any task with fixture actuators can drive them through
the executor without task-specific code.
"""
import mujoco

from .settle import settle


def drive_position(ctx, actuator_name, target, dur_s, settle_steps=0,
                   verbose=False):
    """Linearly ramp a fixture actuator to *target* over *dur_s* seconds.

    Uses control-rate steps (ctx.step) so on_control_step hooks (recorder,
    viewer) fire normally.  Optional *settle_steps* after the ramp lets the
    physics quiet down.  Returns True on completion.
    """
    cid = ctx.act_id(actuator_name)
    n = max(1, int(round(dur_s / ctx.control_dt)))
    for k in range(n):
        ctx.data.ctrl[cid] = target * k / max(n - 1, 1)
        ctx.step()
    ctx.data.ctrl[cid] = target
    if settle_steps > 0:
        settle(ctx, max_steps=settle_steps)
    if verbose:
        print(f"  [drive_position {actuator_name}] -> {target:.4f}")
    return True


def drive_force(ctx, actuator_name, q_start, q_min, v, f_stop,
               dur_max, geom_name, ctrl_offset=0.0, hold_s=0.3,
               verbose=False):
    """Force-controlled drive: command the actuator at velocity *v* until the
    contact force on *geom_name* exceeds *f_stop*, or *q_min* is reached, or
    *dur_max* seconds elapse.

    Runs at the physics rate (mj_step) with force checks every 5 substeps
    (~20 µm of travel between checks at the lowered speeds).  The press
    force decays during the hold as the cell plate yields, so the peak
    force is reported.  Returns the peak contact force (N).
    """
    cid = ctx.act_id(actuator_name)
    dt = ctx.dt
    q = q_start
    f_max = 0.0
    hit = False
    n_max = int(dur_max / dt)
    for k in range(n_max):
        q = max(q - v * dt, q_min)
        ctx.data.ctrl[cid] = q - ctrl_offset
        mujoco.mj_step(ctx.model, ctx.data)
        ctx.n_physics_steps += 1
        if k % 5 == 0:
            f = ctx.geom_contact_force(geom_name)
            f_max = max(f_max, f)
            if f > f_stop:
                hit = True
                break
    if not hit:
        # silent no-contact failure is a trap: a too-short dur_max expires
        # BEFORE first contact and the stage reports 0N
        print(f"[warn] {actuator_name} drive exhausted {dur_max}s without "
              f"reaching f_stop={f_stop}N (peak {f_max:.2f}N)")
    for _ in range(int(hold_s / dt)):
        mujoco.mj_step(ctx.model, ctx.data)
        ctx.n_physics_steps += 1
    f_max = max(f_max, ctx.geom_contact_force(geom_name))
    if verbose:
        print(f"  [drive_force {actuator_name}] peak={f_max:.2f}N "
              f"hit={'yes' if hit else 'no'}")
    return f_max
