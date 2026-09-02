"""Cartesian closed-loop controller + gripper servo for simbench.

The arm is driven through 7 joint position actuators (see panda.xml).
Cartesian motion closes the loop at the control rate: the EEF-site
jacobian maps a clipped cartesian error step into joint deltas
(damped least squares), and the joint targets track the current qpos
plus the delta -- the discrete equivalent of the old OSC P-controller,
but entirely inside mujoco position servos (no torque-level impedance).

The gripper is two position-servoed slide joints.  ``close_to`` commands
the finger qpos directly from a calibrated span model and reads the pad
span back for contact stopping -- the robosuite command-integration
pitfalls (squeeze accumulation, pad/parts ejection) do not exist here,
but the light-press semantics are kept.
"""
import numpy as np
import mujoco

# span model: pad-centre span = SPAN_A + SPAN_B * q1, calibrated on the
# bring-up rig (span(0)=9.46 mm, span(0.04)=84.53 mm)
SPAN_A = 0.0094
SPAN_B = 1.877
PAD_HALF_Y = 0.004            # pad half-size along the closing axis

DEFAULT_GAIN = 8.0            # 1/s P gain on the cartesian error
DEFAULT_MAX_SPEED = 0.30      # m/s EEF speed cap
DEFAULT_TOL = 0.008           # convergence tolerance (m)
MAX_JOINT_DELTA = 0.35        # rad per control step (servo clamp)
DLS_LAMBDA = 0.05             # damped-least-squares damping (m/rad scale)
STALL_PATIENCE = 40           # control steps without progress -> stall
ACCEL_MAX = 0.50              # m/s^2 accel ramp of the smooth trapezoidal
                              # velocity profile (move_eef smooth=): the
                              # long TRAVEL legs ride a planned s-curve
                              # profile instead of snapping between 20 Hz
                              # error-chasing commands.  OFF by default --
                              # the delicate descent/align/steering
                              # recipes depend on the exact per-step
                              # commands (measured: smoothing the place
                              # descent flicked the released shaft 18deg)


class _TrapProfile:
    """Trapezoidal velocity profile along a straight segment.

    Position s(t) in [0,1] with constant-accel ramps and a constant
    cruise -- the commanded velocity is C1-continuous, so the 20 Hz
    control steps no longer snap the arm from one velocity to the next
    (the visible target-chasing jitter).  When the segment is too short
    for the cruise the profile degenerates to a triangle.
    """

    def __init__(self, dist, v_max, a_max):
        self.dist = max(dist, 1e-6)
        self.a = a_max
        t_a = v_max / a_max
        d_a = 0.5 * a_max * t_a * t_a
        if 2.0 * d_a >= self.dist:
            self.t_a = (self.dist / a_max) ** 0.5
            self.v_peak = a_max * self.t_a
            self.d_a = self.dist / 2.0
            self.T = 2.0 * self.t_a
        else:
            self.t_a = t_a
            self.v_peak = v_max
            self.d_a = d_a
            self.T = 2.0 * t_a + (self.dist - 2.0 * d_a) / v_max

    def eval(self, t):
        """(s, s_dot) at time t: s in [0,1], s_dot in 1/s."""
        if t <= 0.0:
            return 0.0, 0.0
        if t >= self.T:
            return 1.0, 0.0
        if t < self.t_a:
            d = 0.5 * self.a * t * t
            v = self.a * t
        elif t < self.T - self.t_a:
            d = self.d_a + self.v_peak * (t - self.t_a)
            v = self.v_peak
        else:
            tr = self.T - t
            d = self.dist - 0.5 * self.a * tr * tr
            v = self.a * tr
        return d / self.dist, v / self.dist


class CartesianController:
    """Closed-loop EEF motion on top of :class:`MjContext`.

    A weak orientation servo holds the tool at the HOME rotation
    (captured at construction, tool-down): the position-only IK let the
    wrist tilt up to 29 deg in the nullspace at far reaches, which
    lowered the pad corner ~7 mm below its nominal eef-4.4 mm (the pads
    then hit the table before reaching thin parts) and released parts
    hanging tilted.  Set ``rot_target=None`` for free-wrist moves.
    """

    ROT_GAIN = 0.5               # fraction of the angle error per step
    ROT_MAX_STEP = 0.08          # rad cap per control step

    def __init__(self, ctx):
        self.ctx = ctx
        self.gain = DEFAULT_GAIN
        self.max_speed = DEFAULT_MAX_SPEED
        self.rot_target = ctx.eef_mat().copy()
        self._v_cmd = np.zeros(3)   # last commanded EEF velocity (m/s)
                                    # for the acceleration limiter

    # ------------------------------------------------------------ motion
    def move_eef(self, target, gain=None, tol=DEFAULT_TOL, max_steps=400,
                 gripper=None, stall=True, verbose=False, max_speed=None,
                 smooth=False):
        """Move the EEF site to ``target`` (world xyz).

        gain: P gain (1/s); the step command is ``gain*err`` capped at
              ``max_speed*control_dt`` so a low gain converges smoothly
              without overshoot (the old planner knob semantics).
        max_speed: per-call speed cap override (m/s) -- carries of held
              parts pass a low value: at the default 0.30 m/s the pads'
              static friction cannot hold a heavy part and it slides out
              mid-flight (measured: housing dropped on the carry leg
              with the fingers still closed at span 58 mm).
        gripper: optional (q1, q2) finger command held during the move;
              None keeps the fingers where they are.
        smooth: ride a planned trapezoidal velocity profile (see
              _TrapProfile) to the goal with a small feedback
              correction, instead of the raw error-chasing P step --
              long TRAVEL legs arrive without velocity snaps or
              overshoot.  Opt-in only for robust travel moves; the
              descent/align/steering recipes depend on the exact
              per-step commands (measured: smoothing the place descent
              flicked the released shaft 18deg).
        Returns True when within tol.
        """
        ctx = self.ctx
        gain = self.gain if gain is None else gain
        max_v = self.max_speed if max_speed is None else max_speed
        best = np.inf
        stall_n = 0
        # smooth profile (opt-in travel legs): plan the trapezoidal
        # velocity profile once and track it with a small feedback
        # correction each control step
        profile = None
        start = None
        if smooth and max_steps > 1:
            start = ctx.eef_pos().copy()
            d0 = float(np.linalg.norm(np.asarray(target) - start))
            if d0 > 0.004:
                profile = _TrapProfile(d0, max_v, ACCEL_MAX)
        t = 0.0
        for _ in range(max_steps):
            eef = ctx.eef_pos()
            err = np.asarray(target, dtype=float) - eef
            dist = float(np.linalg.norm(err))
            if dist < tol:
                self._v_cmd[:] = 0.0
                return True
            if stall:
                # progress threshold 3e-4: an IK crawl (5e-5/step
                # progress, e.g. near a reachable-boundary singularity
                # with a heavy held part) never tripped the old 1e-4
                # bar and ground through the full 400-step budget
                # (measured ~20s stalls on the Task A housing carry)
                if dist < best - 3e-4:
                    best = dist
                    stall_n = 0
                else:
                    stall_n += 1
                    if stall_n >= STALL_PATIENCE:
                        if verbose:
                            print(f"  [move] stall at {dist:.4f} m")
                        self._v_cmd[:] = 0.0
                        return False
            if profile is not None:
                # trajectory tracking: feedforward velocity along the
                # profile + a small P correction onto the planned pose
                s, s_dot = profile.eval(t)
                des = start + (np.asarray(target, dtype=float) - start) * s
                v = ((np.asarray(target, dtype=float) - start) * s_dot
                     + gain * (des - eef))
                vn = np.linalg.norm(v)
                if vn > max_v:
                    v *= max_v / vn
                step = v * ctx.control_dt
                t += ctx.control_dt
            else:
                # raw P step (delicate recipes: descents, aligns,
                # single-step steering)
                v = err * gain
                vn = np.linalg.norm(v)
                if vn > max_v:
                    v *= max_v / vn
                step = v * ctx.control_dt
            self._step_ik(eef, step, gripper)
        self._v_cmd[:] = 0.0
        return float(np.linalg.norm(np.asarray(target) - ctx.eef_pos())) < tol

    def _step_ik(self, eef, step, gripper):
        """One IK increment toward eef+step; steps the sim once.

        The rotation error toward ``rot_target`` (if set) rides along
        in the same damped-least-squares solve, as in the old OSC
        quaternion servo.  """
        ctx = self.ctx
        jacp = np.zeros((3, ctx.model.nv))
        mujoco.mj_jacSite(ctx.model, ctx.data, jacp, None, ctx.eef_site_id)
        dofs = [ctx.model.jnt_dofadr[j] for j in ctx.arm_joint_ids]
        if self.rot_target is not None:
            jacr = np.zeros((3, ctx.model.nv))
            mujoco.mj_jacSite(ctx.model, ctx.data, jacp, jacr,
                              ctx.eef_site_id)
            Rerr = np.asarray(self.rot_target) @ ctx.eef_mat().T
            c = float(np.clip((np.trace(Rerr) - 1.0) / 2.0, -1.0, 1.0))
            ang = float(np.arccos(c))
            ax = np.array([Rerr[2, 1] - Rerr[1, 2],
                           Rerr[0, 2] - Rerr[2, 0],
                           Rerr[1, 0] - Rerr[0, 1]])
            nn = np.linalg.norm(ax)
            if nn > 1e-9:
                ax /= nn
            drot = ax * min(ang * self.ROT_GAIN, self.ROT_MAX_STEP)
            J = np.vstack([jacp[:, dofs], jacr[:, dofs]])
            dx = np.concatenate([step, drot])
            JJt = J @ J.T + (DLS_LAMBDA ** 2) * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, dx)
        else:
            J = jacp[:, dofs]
            # damped least squares: dq = J^T (JJ^T + l^2 I)^-1 dx
            JJt = J @ J.T + (DLS_LAMBDA ** 2) * np.eye(3)
            dq = J.T @ np.linalg.solve(JJt, step)
        n = np.linalg.norm(dq)
        if n > MAX_JOINT_DELTA:
            dq *= MAX_JOINT_DELTA / n
        q = ctx.arm_qpos + dq
        if gripper is not None:
            ctx.set_finger_ctrl(gripper[0], gripper[1])
        ctx.set_arm_ctrl(q)
        ctx.step()

    def goto_pose(self, target_pos, target_mat, gain=None, tol_pos=0.0015,
                  tol_rot=0.02, max_steps=300, gripper=None):
        """Move EEF to a full pose (position + orientation).

        Orientation error feeds the rotational jacobian as an axis-angle
        increment (used e.g. to tilt a held part before release).
        """
        ctx = self.ctx
        gain = self.gain if gain is None else gain
        dofs = [ctx.model.jnt_dofadr[j] for j in ctx.arm_joint_ids]
        for _ in range(max_steps):
            eef = ctx.eef_pos()
            dp = np.asarray(target_pos) - eef
            R = ctx.eef_mat()
            Rerr = np.asarray(target_mat) @ R.T
            c = float(np.clip((np.trace(Rerr) - 1.0) / 2.0, -1.0, 1.0))
            ang = float(np.arccos(c))
            ax = np.array([Rerr[2, 1] - Rerr[1, 2],
                           Rerr[0, 2] - Rerr[2, 0],
                           Rerr[1, 0] - Rerr[0, 1]])
            nn = np.linalg.norm(ax)
            if nn > 1e-9:
                ax /= nn
            if np.linalg.norm(dp) < tol_pos and ang < tol_rot:
                return True
            v = dp * gain
            vn = np.linalg.norm(v)
            if vn > self.max_speed:
                v *= self.max_speed / vn
            step_pos = v * ctx.control_dt
            step_rot = ax * min(ang, 0.15)
            jacp = np.zeros((3, ctx.model.nv))
            jacr = np.zeros((3, ctx.model.nv))
            mujoco.mj_jacSite(ctx.model, ctx.data, jacp, jacr,
                              ctx.eef_site_id)
            Jp = jacp[:, dofs]
            Jr = jacr[:, dofs]
            J = np.vstack([Jp, Jr])
            dx = np.concatenate([step_pos, step_rot])
            JJt = J @ J.T + (DLS_LAMBDA ** 2) * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, dx)
            n = np.linalg.norm(dq)
            if n > MAX_JOINT_DELTA:
                dq *= MAX_JOINT_DELTA / n
            if gripper is not None:
                ctx.set_finger_ctrl(gripper[0], gripper[1])
            ctx.set_arm_ctrl(ctx.arm_qpos + dq)
            ctx.step()
        return False

    def joint_move(self, q_target, max_steps=800, tol=0.02):
        """Joint-space servo to a target configuration.

        max_steps 800: the joint servo from a stretched pose takes
        ~1.5s (the 300-step first cut timed out 0.4 rad short and left
        the arm in a pose whose Cartesian moves crawl, measured on the
        Task B lateral_insert retreat)."""
        ctx = self.ctx
        q_target = np.asarray(q_target, dtype=float)
        for _ in range(max_steps):
            q = ctx.arm_qpos
            if np.max(np.abs(q - q_target)) < tol:
                return True
            dq = np.clip(q_target - q, -0.1, 0.1)
            ctx.set_arm_ctrl(q + dq)
            ctx.step()
        return np.max(np.abs(ctx.arm_qpos - q_target)) < tol

    def home(self):
        """Servo back to the home configuration."""
        return self.joint_move(HOME_QPOS_ARR)

    def rotate_eef(self, axis_angle, steps=40, gripper=(0.0, 0.0),
                   gain=None, max_speed=None):
        """Rotate the wrist by an axis-angle increment (world frame),
        holding the eef position (used by the rotation test).

        Same servo shape as move_eef/goto_pose: the position hold is a
        velocity command (gain*err capped at max_speed, scaled by the
        control dt) and dq is clamped.  The original raw ``(hold-eef)*20``
        error fed straight into the DLS solve diverged -- the eef flung
        1.2 m away within 25 steps and the flailing wrist wrenched the
        fingers off the shaft (measured on S8)."""
        ctx = self.ctx
        gain = self.gain if gain is None else gain
        max_v = self.max_speed if max_speed is None else max_speed
        dofs = [ctx.model.jnt_dofadr[j] for j in ctx.arm_joint_ids]
        hold = ctx.eef_pos()
        aa = np.asarray(axis_angle, dtype=float)
        drot = aa / max(steps, 1)
        for _ in range(steps):
            eef = ctx.eef_pos()
            v = (hold - eef) * gain
            vn = np.linalg.norm(v)
            if vn > max_v:
                v *= max_v / vn
            dx = np.concatenate([v * ctx.control_dt, drot])
            jacp = np.zeros((3, ctx.model.nv))
            jacr = np.zeros((3, ctx.model.nv))
            mujoco.mj_jacSite(ctx.model, ctx.data, jacp, jacr,
                              ctx.eef_site_id)
            J = np.vstack([jacp[:, dofs], jacr[:, dofs]])
            JJt = J @ J.T + (DLS_LAMBDA ** 2) * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, dx)
            n = np.linalg.norm(dq)
            if n > MAX_JOINT_DELTA:
                dq *= MAX_JOINT_DELTA / n
            ctx.set_finger_ctrl(gripper[0], gripper[1])
            ctx.set_arm_ctrl(ctx.arm_qpos + dq)
            ctx.step()


HOME_QPOS_ARR = None   # set after import below (avoids circular import)


class Gripper:
    """Finger position-servo wrapper with span-based closing."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.open_q = 0.04

    # ------------------------------------------------------------ queries
    def span_from_q(self, q1):
        return SPAN_A + SPAN_B * q1

    def q_from_span(self, span):
        """Finger command that yields the requested pad-centre span."""
        return float(np.clip((span - SPAN_A) / SPAN_B, 0.0, self.open_q))

    def inner_gap_from_span(self, span):
        """Gap between the pad inner faces for a pad-centre span."""
        return span - 2.0 * PAD_HALF_Y

    def span(self):
        return self.ctx.pad_span()

    # ------------------------------------------------------------ actions
    def open(self, wait=True, max_steps=80):
        """Open the fingers with a SLEWED command (CLOSE_Q_STEP per
        control step).

        The old robosuite gripper integrated its open command, so a
        release opened the pads slowly and the part barely moved; a
        snap-open here spring-loads the pads against the held part and
        ejects it sideways (measured: housing +14 mm on release).
        """
        ctx = self.ctx
        q_goal = self.open_q
        for _ in range(max_steps):
            q = ctx.finger_qpos
            q_now = 0.5 * (q[0] - q[1])
            dq = q_goal - q_now
            if abs(dq) <= self.CLOSE_Q_STEP:
                break
            step = self.CLOSE_Q_STEP * np.sign(dq)
            ctx.set_finger_ctrl(q_now + step, -q_now - step)
            ctx.step()
        ctx.set_finger_ctrl(q_goal, -q_goal)
        if wait:
            for _ in range(max_steps):
                ctx.step()
                if self.span() is None:
                    continue
                if self.span() >= self.span_from_q(q_goal) - 0.002:
                    break
        return True

    def hold(self):
        q = self.ctx.finger_qpos
        self.ctx.set_finger_ctrl(q[0], q[1])

    # finger command slew per control step during a close (5 mm/s of
    # pad travel: a one-shot command closes the pads at ~0.4 m/s and the
    # impact ejects light parts -- measured on the 3 g rings)
    CLOSE_Q_STEP = 0.0025
    # command overshoot past the span-model target (rad of finger q,
    # ~2.3mm of model span).  A close that slews exactly to q_goal ends
    # with zero servo error, so on a polygon part whose FLAT face points
    # at the pads (contact ~1mm DEEPER than the vertex-diameter model)
    # the pads stop just short of the wall with zero grip force
    # (measured: every Task B grasp failed the lift check).  The
    # overshoot keeps kp*(cmd-q) pushing so the pads bite the wall; the
    # +/-20N actuator forcerange bounds the squeeze.
    CLOSE_OVERSHOOT = 0.0012

    def close_to_span(self, target_span, max_steps=400, quiet=3,
                      force_stop=0.8, verbose=False):
        """Close until the pad CONTACT FORCE reaches force_stop (N) or
        the measured pad-centre span reaches target_span.

        The command slews toward the span-model target (+CLOSE_OVERSHOOT)
        in small steps (soft contact); on contact the pads press with
        kp*(cmd-q) force (bounded by the +/-20N actuator forcerange) and
        hold there -- a stable, non-accumulating squeeze.

        Closing is force-closed, not span-closed: the pad-span model
        (calibrated on the level bring-up rig) drifts several mm with
        the ~7deg carried eef tilt, and a polygon part whose FLAT face
        points at the pads contacts ~1mm deeper than the vertex
        diameter.  A span-only stop then quits short of the wall with
        zero grip force (measured: every Task B grasp failed the lift
        check while deep-close produced 2.7N).  force_stop bounds the
        squeeze on light parts; a rigid block (finger on the table)
        trips the 30-flat-step stall watchdog instead.
        Returns the achieved span.
        """
        ctx = self.ctx
        q_goal = self.q_from_span(target_span) + self.CLOSE_OVERSHOOT
        stall = 0
        prev = None
        f_prev = None
        for _ in range(max_steps):
            q = ctx.finger_qpos
            q_now = 0.5 * (q[0] - q[1])          # symmetric closing depth
            dq = q_goal - q_now
            if abs(dq) <= self.CLOSE_Q_STEP:
                ctx.set_finger_ctrl(q_goal, -q_goal)
            else:
                step = self.CLOSE_Q_STEP * np.sign(dq)
                ctx.set_finger_ctrl(q_now + step, -q_now - step)
            ctx.step()
            s = self.span()
            if s is None:
                continue
            f = (ctx.geom_contact_force("finger1_pad_collision")
                 + ctx.geom_contact_force("finger2_pad_collision"))
            if f >= force_stop:
                quiet -= 1
                if quiet <= 0:
                    break
            if s <= target_span:
                quiet -= 1
                if quiet <= 0:
                    break
            if prev is not None and abs(prev - s) < 1e-5:
                # the span reads frozen while the squeeze creeps
                # (kp*(cmd-q) still ramping the bite on a compliant
                # contact -- the pad span model has ~0.01mm readout
                # resolution); only count a stall when the pad FORCE
                # has also stopped growing, else the finger servo
                # quits 2-3mm short of a real bite (measured: span
                # frozen at 26.8mm, padF 0.7N, on the 12mm latch)
                if f_prev is not None and f <= f_prev + 0.02:
                    stall += 1
                    if stall >= 30:      # blocked by a rigid part
                        break
                else:
                    stall = 0
            else:
                stall = 0
            prev = s
            f_prev = f
        if verbose:
            print(f"  [grip] span {self.span()*1000:.1f} mm "
                  f"(target {target_span*1000:.1f})")
        return self.span()

    def close_on_part(self, outer_d, press=0.0015, verbose=False):
        """Close onto a part of outer diameter ``outer_d`` (m), pressing
        ``press`` into its walls (pad inner faces)."""
        target_span = outer_d - press + 2.0 * PAD_HALF_Y
        return self.close_to_span(target_span, verbose=verbose)

    def press_cmd(self, outer_d, press):
        """Finger ctrl pair holding ``press`` of squeeze on a part of
        ``outer_d``.

        Skills that re-command the fingers every step (rotate_eef) must
        hold THIS command, not the current qpos: a qpos hold zeroes the
        position-servo error and the grip force vanishes -- the pads
        then slip at a fraction of the wrist rate (measured on the S8
        rotation: shaft tracked 10.3 of 31 commanded deg with the latch
        pin seated; holding the press command transmits the full
        kp*(cmd-q) squeeze)."""
        q = self.q_from_span(outer_d - press + 2.0 * PAD_HALF_Y)
        return (q, -q)

    def is_grasping(self, threshold_span=None):
        """Crude grasp check: pads closer together than a reference span."""
        s = self.span()
        if s is None:
            return False
        if threshold_span is None:
            threshold_span = SPAN_A + 0.01
        return s < threshold_span


from .sim_context import HOME_QPOS as _HQ   # noqa: E402
HOME_QPOS_ARR = _HQ
