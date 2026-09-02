"""Motion skills: waypoint moves, alignment, and homing."""
import numpy as np

from .planning import plan_path


def move_eef(arm, target, style="safe_z", tol=0.008, gain=None,
             max_speed=None, smooth=False, **plan_kw):
    """Move the EEF to ``target`` (world xyz) through planned waypoints.

    gain/max_speed are forwarded to every leg (slow carries of held
    parts pass a reduced max_speed).  smooth opt-in applies the
    controller's acceleration limit to the travel legs (robust moves
    only -- see controller.ACCEL_MAX).
    """
    target = np.asarray(target, dtype=float)
    for wp in plan_path(arm.ctx.eef_pos(), target, style=style, **plan_kw):
        if not arm.move_eef(wp, tol=tol, gain=gain, max_speed=max_speed,
                            smooth=smooth):
            return False
    return arm.move_eef(target, tol=tol, gain=gain, max_speed=max_speed,
                        smooth=smooth)


def home(arm, gripper=None):
    """Open the gripper (if given) and servo the arm back to HOME."""
    if gripper is not None:
        gripper.open()
    return arm.home()


def joint_move(arm, q):
    """Joint-space move to an absolute configuration."""
    return arm.joint_move(q)


# ------------------------------------------------------------- alignment
def align_above(arm, ref_xy, tol=0.003, max_steps=80, verbose=False):
    """Align the EEF directly above *ref_xy* (world xy) at the current height.

    A proportional nudge loop on the live EEF xy error (one control step
    per iteration), used to centre a held part over a target before a
    vertical descent.  Returns True when the xy error < tol.
    """
    ref_xy = np.asarray(ref_xy, dtype=float)[:2]
    ctx = arm.ctx
    for _ in range(max_steps):
        eef = ctx.eef_pos()
        err = ref_xy - eef[:2]
        if np.linalg.norm(err) < tol:
            return True
        arm.move_eef(np.array([ref_xy[0], ref_xy[1], eef[2]]),
                     gain=8.0, tol=0.0, max_steps=1, stall=False)
    return False


def goto_ax(arm, target, tol_xy=0.004, tol_z=0.002, gain=8.0,
            max_steps=120):
    """Move EEF to *target* judging xy and z SEPARATELY.

    On a long vertical leg a 3D-norm tolerance lets the xy residual hide
    inside a converged z and vice versa.  This loop commands a single
    step toward the target each iteration and checks the component
    errors independently.  Returns True when both are within tolerance.
    """
    target = np.asarray(target, dtype=float)
    ctx = arm.ctx
    for _ in range(max_steps):
        eef = ctx.eef_pos()
        d = target - eef
        if np.linalg.norm(d[:2]) < tol_xy and abs(d[2]) < tol_z:
            return True
        arm.move_eef(eef + d, gain=gain, tol=0.0, max_steps=1,
                    stall=False)
    eef = ctx.eef_pos()
    d = target - eef
    return bool(np.linalg.norm(d[:2]) < tol_xy and abs(d[2]) < tol_z * 2)
