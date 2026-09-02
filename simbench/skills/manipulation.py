"""Manipulation skills: grasp / place / push / insert / rotate.

All skills are closed-loop on the controller, carry optional perception
noise, and return ok booleans (task chains compute their own stage
metrics from the scene state afterwards).

Grasp heights / light-press semantics follow the old gearbox_skills.py
lessons: rings are gripped near their bottom with the pad bottoms almost
touching the table, the pin high enough that the pads fully wrap it, and
the close stops just past first contact so thin parts are never crushed
out of the pads (the finger servo holds a bounded, non-accumulating
squeeze -- see core.controller.Gripper).

insert(mode='thread') ports the bottom-steered descent + anti-jam wiggle
from the old gearbox task chain as a *generic* skill: the reference axis
(where the part bottom should be steered toward) is passed as a callable
or fixed xy, and the target z is perception-derived.  This makes the
threading logic reusable across tasks without hard-coded coordinates.
"""
import numpy as np
import mujoco

from . import perception
from .motion import move_eef
from .settle import settle

DEFAULT_LIFT = 0.07            # retreat/approach height above parts
GRASP_TOL = 0.004
VERIFY_LIFT = 0.03             # part must rise this much for a grasp ok


def _pad_world_z(ctx):
    """World z of the two finger pad geom centres (debug only)."""
    zs = []
    for name in ("finger1_pad_collision", "finger2_pad_collision"):
        try:
            gid = ctx.geom_id(name)
            zs.append(float(ctx.data.geom_xpos[gid][2]))
        except ValueError:
            pass
    return zs


def _eef_tilt(ctx):
    """Tilt of the EEF z-axis from world vertical (deg)."""
    mat = ctx.eef_mat()
    return float(np.degrees(np.arccos(
        np.clip(abs(mat[2, 2]), 0.0, 1.0))))


# ------------------------------------------------------------------ grasp
def grasp(ctx, arm, gripper, name, noise_std=0.0, press=0.0015,
          lift=DEFAULT_LIFT, grasp_tol=GRASP_TOL, grasp_gain=6.0,
          grasp_dz=None, squeeze=0.0, approach_yaw=None, repress=True,
          verify_lift=True, verbose=False, grasp_pose=None,
          approach_direct=False):
    """Detect + estimate + approach + close + lift-verified grasp.

    grasp_pose: a pre-computed estimate dict (the executor's
    ``plan_grasp_pose`` artifact: pos/yaw/outer_d/grasp_dz/...).  When
    given, the internal detect+estimate is SKIPPED -- the grasp is the
    execution half of the plan-grasp-pose -> grasp handshake.  When
    None the skill behaves as before (detect + estimate inline, with
    ``noise_std`` on the detection).

    grasp_tol: EEF descent tolerance for the approach to the grasp pose.
        Thin parts (cover, rings) need a tighter tolerance (~1mm) so the
        pads don't stop 4mm above and fail to wrap the short wall.
    grasp_gain: P gain for the descent (lower than the controller default
        of 8.0 so the EEF doesn't slam into the table when grasping
        thin parts near the surface — gain 6.0 is the old task chain
        value that worked for all parts).
    grasp_dz: override the PART_META grasp_dz (e.g. raise the EEF target
        2mm for thin parts whose pads would otherwise hit the table
        before reaching the part centre — pads sit ~4.4mm below the EEF).
    squeeze: force-feedback deep bite.  After the close-on-part (which
        stops at force_stop on a shallow bite) slew the fingers deeper
        until the pad CONTACT FORCE reaches ``squeeze`` NEWTONS, then
        hold a command overshoot past the achieved q so kp*(cmd-q)
        sustains the bite through lift + carry + press.  The old
        span-model target silently failed on a tilted hang: the
        pad-centre span drifts mm vs the level-calibrated model
        (Task B probe: measured 11.8mm where the model said a DEEPER
        command was already past), so the 'deeper' command came out
        shallower than reality and the fingers let go -- the polished
        rod then crept 4mm during the continuity press and the tip
        skated off the pad.
    approach_yaw: servo the wrist yaw BEFORE the approach so the
        closing axis points along this world angle.  Needed for small
        polygon rings whose hole is wide enough to swallow a pad
        corner (Task A gear: 16mm hole vs 8mm pad): the home pose
        closes along a VERTEX direction, the tilted pad corner slips
        off the rim top and dives diagonally through the hole (span
        20mm vs target 27mm); the wedged pad then carries the part
        +35mm on release.  Closing along a flat-face direction bites
        the wall squarely and releases clean.  Servoing the wrist
        VERTICAL instead is not an option: the home configuration is
        singular for tilt (j5 ~ 0), every tilt servo flails the arm
        (measured: eef flung 20+ cm at all tested configurations).
    repress: allow the squeeze-on-deficit deep re-close.  Must be False
        for parts whose hole is wide enough to swallow a pad face
        (Task A gear: 16mm hole vs 12mm pad gap at the re-close target).
        The first close stops on a tilted-corner tooth-flank bite whose
        contact force reads ~0 an instant later, and the deep re-close
        then spears the pads THROUGH the hole (measured: span 28 ->
        20mm, one pad wedged in the hole, the part carried +35mm on
        release).  The shallow corner bite alone holds a light part.
    """
    gp = (grasp_pose if grasp_pose is not None
          else perception.estimate_grasp_pose(ctx, name,
                                             noise_std=noise_std))
    if grasp_dz is not None:
        gp["pos"][2] = ctx.obj_pos(name)[2] + grasp_dz
    z_start = ctx.obj_pos(name)[2]

    if approach_yaw is not None:
        _approach_yaw(ctx, arm, approach_yaw)

    if verbose:
        print(f"  [grasp {name}] target {np.round(gp['pos'], 4)} "
              f"yaw {gp['yaw']:.2f} outer_d {gp['outer_d']*1000:.1f} mm")

    gripper.open()
    # approach_direct: the caller already travelled to a hover ABOVE the
    # part (the two-class move_to) -- hop straight down instead of the
    # safe_z style, which would LIFT first (max(cur,goal)+0.08) and
    # re-travel: the visible descend-rise-twitch before every grasp
    if not move_eef(arm, gp["pos"] + np.array([0, 0, lift]), smooth=True,
                    style=("direct" if approach_direct else "safe_z")):
        return False
    # descend (don't abort on stall — the EEF may not reach the exact
    # target when pads collide with the table on thin parts, but the
    # pads are still close enough to grip)
    arm.move_eef(gp["pos"], gain=grasp_gain, tol=grasp_tol)
    if verbose:
        eef_after = ctx.eef_pos()
        print(f"  [grasp {name}] eef after descent: "
              f"{np.round(eef_after, 4)} (target {np.round(gp['pos'], 4)}) "
              f"dz_err={eef_after[2]-gp['pos'][2]:+.4f}")
    # re-center if the part slid under the fingers during the approach
    # (an eccentric grip drags the part off-axis during placement).
    # The descent itself can also push the part aside (measured on the
    # Task A gear: pad corner pressure slid the ring 6.6mm during the
    # 1.8mm descent; the eef-vs-gp-pos check below then read <0.6mm
    # and skipped, so the pads closed 1mm off-centre and only ONE pad
    # bit the wall edge -- a weak grip the part slipped out of).
    # Always re-centre on the LIVE part position before closing.
    pnow = ctx.obj_pos(name)
    if np.linalg.norm((ctx.eef_pos() - pnow)[:2]) > 0.0003:
        arm.move_eef(np.array([pnow[0], pnow[1], ctx.eef_pos()[2]]),
                     gain=grasp_gain, tol=0.0004, max_steps=20)
    for _ in range(5):
        pnow = ctx.obj_pos(name)
        if np.linalg.norm((ctx.eef_pos() - pnow)[:2]) < 0.0006:
            break
        arm.move_eef(np.array([pnow[0], pnow[1], ctx.eef_pos()[2]]),
                    gain=grasp_gain, tol=0.0006, max_steps=10)
    gripper.close_on_part(gp["outer_d"], press=press,
                           verbose=verbose)
    # squeeze-on-deficit: the close stops when pad force reaches ~1N or
    # the (tilt-drifted) span model says stop; a polygon part whose
    # corner gap faces the pads only contacts several mm deeper than
    # the model (measured on the Task B housing: close ended at 0.0N;
    # wall bite needed q ~0.024 vs model 0.0262).  If the pads report
    # no contact force, force-close onto the part: command a deep span
    # and let the kp servo bite until the pads load ~1N per side.
    if repress and (ctx.geom_contact_force("finger1_pad_collision")
                    + ctx.geom_contact_force("finger2_pad_collision")) < 0.3:
        gripper.close_to_span(gp["outer_d"] - 0.006,
                              max_steps=250, quiet=2, force_stop=2.0)
        if verbose:
            print(f"  [grasp {name}] re-squeeze span="
                  f"{gripper.span()*1000:.1f}mm")
    if squeeze > 0:
        # force-feedback deep bite (squeeze is in NEWTONS of total pad
        # force).  The close stopped at force_stop on a shallow bite
        # (Task B probe: 12mm span, ~0.4mm bite; the polished rod slid
        # UP through the pads under the continuity press).  Span-based
        # closure is unreliable here: the pad-centre span model is
        # calibrated level and drifts mm on the ~7deg tilted hang, so
        # close on the MEASURED pad force instead, then hold a command
        # overshoot past the achieved q so kp*(cmd-q) sustains the
        # bite through lift + carry + press.
        def _pad_force():
            return (ctx.geom_contact_force("finger1_pad_collision")
                    + ctx.geom_contact_force("finger2_pad_collision"))
        # closing = q DECREASES toward 0 (span model: span grows with q)
        q = 0.5 * (ctx.finger_qpos[0] - ctx.finger_qpos[1])
        # fine ramp: a full CLOSE_Q_STEP per iteration hits the q<=0
        # floor in ONE step from the ~2.3mm shallow bite (measured:
        # the squeeze loop exited after a single step at 2.46N, never
        # reaching the 4N target), so the deep bite never engaged and
        # the rod crept out during the carry.
        step = gripper.CLOSE_Q_STEP * 0.1
        for _ in range(800):
            if _pad_force() >= squeeze:
                break
            if q <= 0.0002:
                break
            q -= step
            ctx.set_finger_ctrl(q, -q)
            ctx.step()
        # hold: command a fixed overshoot DEEPER than the ACHIEVED q so
        # kp*(cmd-q) sustains the bite.  Anchor on the measured qpos,
        # NOT on the commanded q: when the fingers wedge on the part
        # the servo lags the command, and anchoring on the command at
        # the q=0 joint limit gets clamped by ctrlrange back to 0 --
        # zero servo error, zero bite (measured: grip force lost,
        # rod crept again).  TWO overshoots past the achieved q: the
        # single overshoot bite relaxes out during the long carry
        # (measured: padF 3.25N -> 0 over ~940 carry steps, the rod
        # dropped mid-flight); the deeper command keeps kp*(cmd-q)
        # pressing until the pads nearly touch.
        q_ach = 0.5 * (ctx.finger_qpos[0] - ctx.finger_qpos[1])
        q_hold = max(q_ach - 2.0 * gripper.CLOSE_OVERSHOOT, 0.0)
        ctx.set_finger_ctrl(q_hold, -q_hold)
        if verbose:
            print(f"  [grasp {name}] squeeze hold q={q:.4f} "
                  f"F={_pad_force():.2f}N")
    if verbose:
        sp = gripper.span()
        sp_str = f"{sp*1000:.1f} mm" if sp is not None else "None"
        print(f"  [grasp {name}] span after close: {sp_str} "
              f"(target {(gp['outer_d']-press+0.008)*1000:.1f} mm)")
        # contact-force diagnostic: are the pads actually touching
        # the part, or did they stall on something else (table, etc.)?
        for pn in ("finger1_pad_collision",
                   "finger2_pad_collision"):
            try:
                f = ctx.geom_contact_force(pn)
                print(f"          {pn}: {f:.2f} N")
            except ValueError:
                pass
        # full pad/finger contact-pair dump (what did the close stop on?)
        pairs = set()
        for i in range(ctx.data.ncon):
            c = ctx.data.contact[i]
            g1 = ctx.model.geom(c.geom1).name or "?"
            g2 = ctx.model.geom(c.geom2).name or "?"
            if any(k in g1 + g2 for k in ("finger", "pad")):
                pairs.add((g1, g2))
        for a, b in sorted(pairs):
            print(f"          pair: {a} <-> {b}")
    # (the zero-grip squeeze hook lives above as the opt-in `squeeze`
    # param; unconditional squeezing ejected thin parts -- keep it
    # plan-driven)
    # two-leg RELATIVE lift (from the current EEF position) so the
    # lift height is always lift*0.6 + lift*0.4 = lift regardless of
    # where the EEF stalled during descent.  Absolute targets (z_start
    # + offset) under-lifted parts with grasp_dz>0 because z_start is
    # the part centre, not the EEF position (measured: housing/shaft
    # grasp failed with absolute targets after the descent stall left
    # the EEF 1.5 cm above z_start).
    if verbose:
        print(f"  [grasp {name}] pad_zs={np.round(_pad_world_z(ctx), 4)} "
              f"eef_tilt={_eef_tilt(ctx):.1f}deg "
              f"span={gripper.span()*1000:.1f}mm")
    arm.move_eef(ctx.eef_pos() + np.array([0, 0, lift * 0.6]),
                gain=grasp_gain, tol=0.004)
    if verbose:
        print(f"  [grasp {name}] after 1st lift: eef_z={ctx.eef_pos()[2]:.4f} "
              f"part_z={ctx.obj_pos(name)[2]:.4f} "
              f"span={gripper.span()*1000:.1f}mm")
    settle(ctx, max_steps=15)
    arm.move_eef(ctx.eef_pos() + np.array([0, 0, lift * 0.4]), tol=0.01)
    if verbose:
        print(f"  [grasp {name}] after 2nd lift: eef_z={ctx.eef_pos()[2]:.4f} "
              f"part_z={ctx.obj_pos(name)[2]:.4f}")

    z_now = ctx.obj_pos(name)[2]
    # verify_lift=False: a part fixed in the assembly (the Task A
    # shaft seated in the housing hole for the rotation test) cannot
    # be lifted -- grasping it must not fail the lift check.
    ok = (not verify_lift) or z_now > z_start + VERIFY_LIFT
    if verbose:
        print(f"  [grasp {name}] lift check: z {z_start:.4f} -> "
              f"{z_now:.4f} ({'ok' if ok else 'FAIL'})")
    return ok


# ------------------------------------------------------------------ place
def place(ctx, arm, gripper, name, target_xy, target_z=None,
          press_steps=6, retreat=DEFAULT_LIFT, carry_speed=None,
          align=False, release="slew", settle_steps=10, verbose=False,
          carry_style="safe_z", live_align=False, low_carry=False,
          level=False, carry_direct=False):
    """Place the held part so its body lands at (target_xy, target_z).

    The EEF goal is derived from the measured EEF-part offset while
    holding, so detection noise at grasp time does not bias the drop.
    Descends, opens the pads and retreats vertically.

    carry_speed: max EEF speed during the carry (slow for light parts
        that slide out of the pads at default 0.30 m/s).
    carry_style: motion style for the carry leg ('line' or 'safe_z');
        light hanging parts (Task C plate) swing out of the pads on a
        straight hop -- safe_z lifts first, then translates.
    align: nudge the EEF so the PART sits over the target before
        descending (the pads' friction drags the part with the eef).
    live_align: continuous low-gain proportional loop on the LIVE part
        error above the target (Task C plate: the light part slides
        inside the pads during the carry, so the grasp offset is
        stale; 150 x 1-control-step nudges with a quiet-count break),
        followed by a tangential yaw steer and a pendulum damp.
    low_carry: approach at a LOW hover (35mm above the target seat)
        instead of the default high safe_z transfer -- the hanging
        part passes UNDER overhead fixture bars whose z-band the
        default transfer would drag it through (Task C clamp bar at
        z 0.867 clips the plate's -x corner; measured: plate knocked
        out of the pads and stood on edge in the cavity).
    release: 'slew' (default, slewed open), 'burst' (single
        direct ctrl write -- for light plates where staged openings
        let press-in energy push the part sideways), 'lift_first'
        (burst open in place, THEN retreat vertically -- for a part
        whose pads overlap its rim radially: opening in place lets it
        free-fall straight down, and the pads clear as it falls;
        slewing open in place sweeps the pads across the rim edge and
        flips the part (measured on the Task B cover: 0.08 -> 10.8deg
        in 2 physics steps), while retreating with CLOSED pads drags
        the part up and shifts its landing), or 'relax' (hold at zero
        servo error to bleed off the squeeze, settle, THEN slew open
        -- for a part whose pads still overlap its wall at release:
        opening UNDER squeeze flicks the part (the wall-bite force +
        the pad recoil kick it sideways mid-fall; measured on the
        Task B cover: burst/slew/slow-open all flipped it 10-25deg,
        relax+open fell at 2.2deg max).
    settle_steps: physics settle steps after retreat (heavy parts
        like the housing need ~80 to stop bouncing/tipping).
    level: servo the wrist plumb before the descent (the 7.3deg eef
        tilt lands a wide ring on its seat edge-first: the Task B
        cover's down edge snagged the 12.1mm flange and parked 11deg
        tilted, measured).
    """
    rot0 = arm.rot_target
    target_xy = np.asarray(target_xy, dtype=float)[:2]
    obj = ctx.obj_pos(name)
    eef = ctx.eef_pos()
    d_xy = eef[:2] - obj[:2]
    d_z = eef[2] - obj[2]
    tz = obj[2] if target_z is None else float(target_z)
    goal = np.array([target_xy[0] - d_xy[0], target_xy[1] - d_xy[1],
                     tz + d_z])

    move_kw = {}
    if carry_speed is not None:
        move_kw["max_speed"] = carry_speed
    # the carry leg is a robust long travel: smooth its start/stop and
    # direction reversals (opt-in ACCEL_MAX limit).  The descent keeps
    # the exact per-step recipe -- smoothing it flicked the released
    # shaft out of the hole (measured 18deg tilt).
    carry_kw = dict(move_kw, smooth=True)
    # carry_direct: the caller (the two-class transport) already
    # travelled to a hover above the target -- hop straight down
    # instead of the safe_z style, which would LIFT +0.08 first (the
    # visible descend-rise before every placement)
    if carry_direct:
        carry_style = "direct"

    if low_carry:
        # low horizontal approach: hover ~15mm above the target seat so
        # the hanging part passes far UNDER any overhead fixture bar
        # (Task C clamp rests at z 0.867; a 35mm hover left only 7mm of
        # clearance and the plate's pendulum swing clipped the bar,
        # measured: plate knocked back to x=0.10, 18deg tilted).
        hover = goal + np.array([0.0, 0.0, 0.015])
        if not move_eef(arm, hover, style="clearance",
                        safe_z=hover[2], **carry_kw):
            return False
    elif not move_eef(arm, goal + np.array([0, 0, retreat]),
                      style=carry_style, **carry_kw):
        return False
    if live_align:
        # continuous low-gain proportional loop on the live part error
        # (the part slides inside the pads during the carry, so the
        # grasp offset is stale).  A quiet-count break rejects
        # pendulum swings faking convergence.
        quiet = 0
        for _ in range(150):
            err = ctx.obj_pos(name)[:2] - target_xy
            if np.linalg.norm(err) < 0.0025:
                quiet += 1
                if quiet >= 10:
                    break
            else:
                quiet = 0
            e2 = ctx.eef_pos()
            arm.move_eef(e2 - np.array([err[0], err[1], 0.0]),
                         gain=6.0, tol=0.0, max_steps=1, stall=False)
        # re-plumb: the live hang offset changed during the carry +
        # align, so the descent goal must track the CURRENT eef xy or
        # the descent drags the hanging part sideways (measured:
        # Task C plate lands 16deg yawed on the wall tops)
        goal[:2] = ctx.eef_pos()[:2].copy()
        # yaw steer: rotate the eef tangentially about the hanging
        # part's vertical axis so pad friction twists it back to ~0
        for _ in range(80):
            yaw = float(ctx.obj_yaw(name))
            if abs(yaw) < 0.02:
                break
            e2 = ctx.eef_pos()
            p2 = ctx.obj_pos(name)
            r = e2[:2] - p2[:2]
            th = -yaw * 0.3
            c, s = np.cos(th), np.sin(th)
            r2 = np.array([c * r[0] - s * r[1],
                           s * r[0] + c * r[1]])
            arm.move_eef(np.array([p2[0] + r2[0], p2[1] + r2[1], e2[2]]),
                         gain=6.0, tol=0.0, max_steps=1, stall=False)
        # re-plumb again: the yaw steer moved the eef
        goal[:2] = ctx.eef_pos()[:2].copy()
        # damp the pendulum before the descent
        settle(ctx, max_steps=16)
    # Order matters: LEVEL first, then ALIGN.  Levelling the wrist
    # rotates the hanging part about the eef and shifts its xy by up
    # to sin(7.3deg)*hang-depth (~5mm for the Task B cover); aligning
    # first left the part 6.7mm off the housing axis and the descent
    # snagged the connector stack edge-first (measured: 4.8deg wedge).
    if level:
        _level_axis(ctx, arm, name, verbose=verbose)
        goal[:2] = ctx.eef_pos()[:2].copy()
    if align:
        # pre-descent alignment: nudge the part over the target at
        # height (the pads' friction drags the part with the eef).
        # gain=6.0 / 15 iters / 1.5mm tol match the old gearbox task
        # chain — the pad grip has a ~1.6mm steering dead-band, so
        # tighter thresholds just waste iterations without converging.
        for _ in range(15):
            err = ctx.obj_pos(name)[:2] - target_xy
            if np.linalg.norm(err) < 0.0015:
                break
            e2 = ctx.eef_pos()
            arm.move_eef(e2 - np.array([err[0], err[1], 0.0]),
                        gain=6.0, tol=0.0005, max_steps=10,
                        stall=False)
        # Update goal xy to the aligned EEF position so the vertical
        # descent doesn't undo the alignment (the original goal was
        # computed from the pre-align EEF-part offset)
        goal[:2] = ctx.eef_pos()[:2].copy()
    if verbose:
        print(f"  [place {name}] pre-descent tilt="
              f"{ctx.obj_tilt(name):.2f}deg goal_z={goal[2]:.4f}")
    if not arm.move_eef(goal, tol=0.005, **move_kw):
        return False
    # slip-corrected second descent: a light part gripped on a curved
    # wall can climb/sink a few mm inside the pads during the carry +
    # descent (measured on the Task B cover: 3.4mm high), and a
    # free-fall on release wedges it on the seat.  Re-measure the
    # EEF-part offset and drive the part the rest of the way down.
    # Capped at 1.5mm and followed by a small lift: driving the seat
    # contact several mm INTO penetration makes the contact solver
    # snap the part out sideways (measured: the Task B cover flipped
    # 0.4 -> 14.8deg in a single step during the release hold).
    perr = ctx.obj_pos(name)[2] - tz
    if perr > 0.001:
        arm.move_eef(goal - np.array([0.0, 0.0, min(perr, 0.0015)]),
                     tol=0.003, **move_kw)
        arm.move_eef(ctx.eef_pos() + np.array([0.0, 0.0, 0.002]),
                     tol=0.003)
        settle(ctx, max_steps=10)
    if verbose:
        print(f"  [place {name}] post-descent tilt="
              f"{ctx.obj_tilt(name):.2f}deg z={ctx.obj_pos(name)[2]:.4f}")
        # contact dump: what stopped the descent?
        pairs = set()
        for i in range(ctx.data.ncon):
            c = ctx.data.contact[i]
            g1 = ctx.model.geom(c.geom1).name
            g2 = ctx.model.geom(c.geom2).name
            if any(k in g1 + g2 for k in ("finger", "pad", name)):
                pairs.add((g1, g2))
        for a, b in sorted(pairs):
            print(f"          contact: {a} <-> {b}")
    if align:
        # post-descent re-alignment: the part may have shifted during
        # the descent (tilt swing, contact with the tray surface)
        for _ in range(15):
            err = ctx.obj_pos(name)[:2] - target_xy
            if np.linalg.norm(err) < 0.0015:
                break
            e2 = ctx.eef_pos()
            arm.move_eef(e2 - np.array([err[0], err[1], 0.0]),
                        gain=6.0, tol=0.0005, max_steps=10,
                        stall=False)
    for _ in range(press_steps):
        ctx.step()
    if verbose:
        print(f"  [place {name}] pre-release yaw="
              f"{np.degrees(ctx.obj_yaw(name)):.2f}deg "
              f"tilt={ctx.obj_tilt(name):.2f}deg "
              f"swing={np.round(ctx.obj_pos(name), 4)}")
    if release == "lift_first":
        # open IN PLACE then climb away: opening at rest lets the part
        # free-fall straight down onto its seat (the pads never sweep
        # across the rim), and retreating with CLOSED pads drags the
        # part up on pad friction and shifts its landing (measured on
        # the Task B cover: lifted to 0.8718 and landed +10mm x).
        ctx.set_finger_ctrl(gripper.open_q, -gripper.open_q)
        settle(ctx, max_steps=5)
        if not arm.move_eef(ctx.eef_pos() + np.array([0.0, 0.0, retreat]),
                            gain=10.0, tol=0.012):
            return False
    elif release == "burst":
        ctx.set_finger_ctrl(gripper.open_q, -gripper.open_q)
        for _ in range(3):
            ctx.step()
        arm.move_eef(goal + np.array([0.0, 0.0, 0.12]), gain=10.0,
                    tol=0.012)
    elif release == "relax":
        # bleed the squeeze BEFORE opening: hold the current finger q
        # (zero servo error -> grip force ~0), let the part settle on
        # the friction hang, then slew open.  Opening under squeeze
        # flicks the part (measured on the Task B cover, burst/slew/
        # slow all flipped it 10-25deg; relax fell at 2.2deg max).
        gripper.hold()
        settle(ctx, max_steps=20)
        gripper.open()
        settle(ctx, max_steps=10)
        arm.move_eef(ctx.eef_pos() + np.array([0, 0, retreat]), tol=0.01)
    elif release == "two_stage":
        # bleed -> PARTIAL open -> lift clear -> full open.  A seated
        # part with a PROTRUDING head (the Task A cover over its latch
        # pin: the pin top pokes 1.2mm above the cover top at r18.25,
        # inside the pads' open sweep band) gets its head flicked by
        # the full open -- the pads sweep the whole cover radius on
        # the way out (measured: pin ejected 1m, every run).  The
        # partial open (span +1.9mm) frees the cover, the 3cm rise
        # parks the pads ABOVE the pin top, the full open sweeps air.
        gripper.hold()
        settle(ctx, max_steps=20)
        q = ctx.finger_qpos
        q_loose = 0.5 * (q[0] - q[1]) + 0.001
        ctx.set_finger_ctrl(q_loose, -q_loose)
        settle(ctx, max_steps=15)
        arm.move_eef(ctx.eef_pos() + np.array([0, 0, 0.03]), gain=8.0,
                     tol=0.006)
        settle(ctx, max_steps=6)
        gripper.open()
        settle(ctx, max_steps=10)
        arm.move_eef(ctx.eef_pos() + np.array([0, 0, retreat]), tol=0.01)
    else:
        # hold first to release press stress, then slew open --
        # opening under load ejects the part sideways (measured: 11mm)
        gripper.hold()
        settle(ctx, max_steps=5)
        if verbose:
            print(f"  [place {name}] pre-open tilt="
                  f"{ctx.obj_tilt(name):.2f}deg "
                  f"z={ctx.obj_pos(name)[2]:.4f}")
        gripper.open()
        settle(ctx, max_steps=10)
        if verbose:
            print(f"  [place {name}] post-open tilt="
                  f"{ctx.obj_tilt(name):.2f}deg "
                  f"z={ctx.obj_pos(name)[2]:.4f}")
        arm.move_eef(ctx.eef_pos() + np.array([0, 0, retreat]),
                     tol=0.01, smooth=True)
    settle(ctx, max_steps=settle_steps)
    arm.rot_target = rot0
    ok = float(np.linalg.norm(ctx.obj_pos(name)[:2]
                              - target_xy)) < 0.012
    if verbose:
        print(f"  [place {name}] at {np.round(ctx.obj_pos(name), 4)} "
              f"({('ok' if ok else 'FAIL')})")
    return ok


# ------------------------------------------------------------------- push
def push(ctx, arm, gripper, name, delta=None, to_target=None,
         standoff=0.06, overshoot=1.3, push_z=None, speed=0.06,
         stall_patience=40, max_steps=300, verbose=False):
    """Push a part horizontally with the closed fingers (blade push).

    Exactly one of ``delta`` (dx, dy displacement) or ``to_target``
    (absolute target xy) selects the push (to_target wins).

    The pads close to a narrow blade, approach from behind the part
    along the push direction and drive through the target (with
    overshoot).  The drive is CLOSED-LOOP on the live part: progress is
    measured along the push direction every control step and the drive
    quits when the part stops moving (``stall_patience``) instead of
    grinding the blade into a jammed part for hundreds of steps.

    push_z: world z of the push line (default: part-centre height).
        Thin parts slid along a rail are pushed at THEIR surface height,
        which may sit a few mm above or below their centre.
    Returns whether the part moved >= 70% of the requested displacement
    along the direction.
    """
    if to_target is not None:
        delta = (np.asarray(to_target, dtype=float)[:2]
                 - ctx.obj_pos(name)[:2])
    if delta is None:
        raise ValueError("push requires delta or to_target")
    delta = np.asarray(delta, dtype=float)[:2]
    dist = float(np.linalg.norm(delta))
    if dist < 1e-6:
        return True
    d = delta / dist
    obj = ctx.obj_pos(name)
    z = obj[2] if push_z is None else float(push_z)

    start = obj[:2] - d * standoff
    end = obj[:2] + d * (dist * overshoot)
    gripper.close_to_span(0.012)        # fingers together = push blade
    if not move_eef(arm, np.array([start[0], start[1], z + 0.04])):
        return False
    if not arm.move_eef(np.array([start[0], start[1], z]), tol=0.006):
        return False
    best, stall = 0.0, 0
    for _ in range(max_steps):
        eef = ctx.eef_pos()
        arm.move_eef(np.array([end[0], end[1], z]), gain=6.0, tol=0.0,
                     max_steps=1, stall=False, max_speed=speed)
        prog = float((ctx.obj_pos(name)[:2] - obj[:2]) @ d)
        if prog > best + 1e-5:
            best, stall = prog, 0
        else:
            stall += 1
            if stall >= stall_patience:
                break
        if prog >= dist:
            break
    move_eef(arm, ctx.eef_pos() + np.array([0, 0, DEFAULT_LIFT]))

    moved = ctx.obj_pos(name)[:2] - obj[:2]
    ok = float(moved @ d) >= 0.7 * dist
    if verbose:
        print(f"  [push {name}] moved {np.round(moved, 4)} "
              f"(asked {np.round(delta, 4)}) {'ok' if ok else 'FAIL'}")
    return ok


# ----------------------------------------------------------------- insert
def insert(ctx, arm, gripper, name=None, target_eef=None, mode="press",
           tol=0.003, press_steps=15, verbose=False, **kw):
    """Insert a held part via press-fit or threading descent.

    mode='press'  -- descend to *target_eef* and hold for *press_steps*
                     (stalls on contact, that is fine).
    mode='thread' -- bottom-steered descent onto a frozen reference axis
                     with anti-jam wiggle (ported from the old gearbox
                     _bottom_press).  Requires: *name* (part body),
                     *ref_axis* (callable(ctx)->xy or fixed xy), *to_z*
                     (target bottom z), *half* (part half-height; from
                     perception.part_meta if omitted).
    """
    if mode == "press":
        if target_eef is None:
            raise ValueError("insert mode='press' requires target_eef")
        target_eef = np.asarray(target_eef, dtype=float)
        reached = arm.move_eef(target_eef, tol=tol, stall=False)
        for _ in range(press_steps):
            ctx.step()
        if verbose:
            print(f"  [insert press] eef {np.round(ctx.eef_pos(), 4)} "
                  f"(target {np.round(target_eef, 4)}) "
                  f"{'reached' if reached else 'pressed'}")
        return True
    if mode == "thread":
        return _thread_insert(ctx, arm, gripper, name, verbose=verbose,
                             **kw)
    raise ValueError(f"unknown insert mode {mode!r}")


# ---------------------------------------------------- threading descent
def _approach_yaw(ctx, arm, yaw):
    """Servo the wrist yaw to a world angle (closing axis = eef local
    x).  rotate_eef about world z is the one rotation servo that is
    well-conditioned at the home family of configurations (tilt
    servos are singular there -- j5 ~ 0).  Sets arm.rot_target to the
    new pose so subsequent move_eef calls hold the yaw.
    """
    R = ctx.eef_mat()
    cur = float(np.arctan2(R[1, 0], R[0, 0]))
    d = (yaw - cur + np.pi) % (2.0 * np.pi) - np.pi
    if abs(d) < 0.02:
        arm.rot_target = R.copy()
        return
    arm.rotate_eef([0.0, 0.0, d],
                   steps=max(20, int(abs(d) / 0.004)),
                   gripper=tuple(ctx.finger_qpos))
    arm.rot_target = ctx.eef_mat().copy()


def _level_axis(ctx, arm, name, verbose=False):
    """Servo the wrist so the held part's +z axis stands plumb.

    Minimal rotation about the vertical-error axis; sets arm.rot_target
    to the level pose so subsequent move_eef calls hold it.  Returns
    True when a servo was issued.

    The rotation is issued in ~1deg slices with a settle between: a
    one-shot goto_pose turns the wrist ~2.3deg per step and the hanging
    rod's inertia lags the pads -- the pad edge then bites in and rolls
    the rod out (measured: the Task B probe dropped MID-LEVEL at a
    4.6N bite, once in several runs).  Slowing the rotation lets the
    rod track the wrist and the pads hold.
    """
    a = ctx.obj_axis(name)
    c = float(a[2])
    if c >= 0.9998:
        return False
    v = np.array([a[1], -a[0], 0.0])
    nn = float(np.linalg.norm(v))
    if nn < 1e-9:
        return False
    v /= nn
    ang = float(np.arccos(np.clip(c, -1.0, 1.0)))
    K = np.array([[0.0, -v[2], v[1]],
                  [v[2], 0.0, -v[0]],
                  [-v[1], v[0], 0.0]])
    R0 = ctx.eef_mat().copy()
    steps = max(2, int(ang / 0.02))       # ~1.15 deg per slice
    for s in range(1, steps + 1):
        a_s = ang * s / steps
        Rds = (np.eye(3) + np.sin(a_s) * K
               + (1.0 - np.cos(a_s)) * K @ K)
        arm.goto_pose(ctx.eef_pos(), Rds @ R0)
        settle(ctx, max_steps=3)
    R_tgt = (np.eye(3) + np.sin(ang) * K
             + (1.0 - np.cos(ang)) * K @ K) @ R0
    arm.rot_target = R_tgt.copy()
    # gripper=None: keep the CURRENT finger command (a squeeze grasp
    # holds an overshoot command past the real q -- re-issuing the
    # measured q drops the kp*(cmd-q) bite force; measured on the
    # Task B probe continuity press)
    arm.goto_pose(ctx.eef_pos(), R_tgt, gripper=None)
    if verbose:
        print(f"  [level] axis now {ctx.obj_tilt(name):.2f}deg")
    return True


def _press_wiggle(ctx, arm, name, to_z, half, eef0, radius=0.0015,
                  arms=6, per_arm=10, drop=0.0003):
    """Small xy sweep of the eef around *eef0* while pressing.

    A part whose bottom is friction-locked against a guide wall or stuck
    on a hole rim drops in as the sweep walks its bottom around the
    mouth.  Gentle (1.5mm) and always biased downward.  Anchored to the
    press-start eef and returns there after the sweep.
    """
    for k in range(arms):
        ang = k * 2.0 * np.pi / arms
        wxy = np.array([np.cos(ang), np.sin(ang)]) * radius
        for _ in range(per_arm):
            eef = ctx.eef_pos()
            tgt = np.array([eef0[0] + wxy[0], eef0[1] + wxy[1],
                            eef[2] - drop])
            arm.move_eef(tgt, gain=20.0, tol=0.0005, max_steps=1,
                        stall=False)
            if ctx.obj_pos(name)[2] - half <= to_z:
                return
    for _ in range(20):
        eef = ctx.eef_pos()
        if np.linalg.norm(eef[:2] - eef0[:2]) < 0.0005:
            break
        arm.move_eef(np.array([eef0[0], eef0[1], eef[2]]),
                    gain=20.0, tol=0.0005, max_steps=1, stall=False)


def _thread_insert(ctx, arm, gripper, name, ref_axis=None, to_z=None,
                  half=None, max_steps=200, drop=0.004, steer=0.0008,
                  steer_cap=0.001, steer_drop=0.0, max_wiggles=3,
                  align_bottom_z=None, align_tol=0.0015,
                  steer_bottom=False, level=False,
                  retreat=DEFAULT_LIFT,
                  release=True, settle_steps=10, force_seat=True,
                  release_lift=0.006, verbose=True):
    """Bottom-steered descent of a held part onto *to_z*.

    The eef xy steers the part BOTTOM onto the reference axis (frozen at
    press start) while it descends.  Proportional steering (half the
    error, capped).  A descent that makes no progress for 15 steps is
    friction-locked: a circular sweep while pressing walks the bottom
    around the rim until it drops in.  Only a part that survives
    max_wiggles without moving is treated as a true jam.

    ref_axis: callable(ctx)->xy, fixed (x,y) tuple, or None (use the
              part's current xy -- press straight down).
    to_z:     target bottom z of the part (perception-derived).
    half:     part half-height (from perception.part_meta if None).
    align_bottom_z: when given, align the part over the ref axis at this
              height before starting the press descent.
    align_tol: xy convergence tolerance for the pre-align nudge loop
              (rings need 0.5mm — a loose 1.5mm lets the part enter the
              hole wall sideways and jam during the descent).
    steer_bottom: steer on the live bottom point (centre - half*axis)
              instead of the body centre.  A part hanging from the
              7.3deg eef tilt carries its bottom lever*sin(tilt) off
              the centre (3mm for the Task B connector); a tight hole
              needs the BOTTOM entered squarely, not the centre.
    level:    servo the wrist so the held part's +z axis stands plumb
              before the descent (minimal rotation).  A flange-on-rim
              seat cannot level itself from a tilted hang: the flange
              plane stays fixed by the grasp, so a tilted flange wedges
              on the rim edge and RELEASE makes it wedge DEEPER
              (measured on the Task B connector: 5.8 -> 12.1deg over
              400 idle steps).
    """
    if to_z is None:
        raise ValueError("thread_insert requires to_z")
    if half is None:
        meta = perception.part_meta(ctx, name)
        half = meta["half_h"]

    rot0 = arm.rot_target          # restore the wrist servo on exit

    def _level():
        """Servo the wrist so the held part's +z axis stands plumb."""
        _level_axis(ctx, arm, name, verbose=verbose)

    if level:
        _level()

    def ref_xy():
        if ref_axis is None:
            return ctx.obj_pos(name)[:2]
        if callable(ref_axis):
            return np.asarray(ref_axis(ctx), dtype=float)[:2]
        return np.asarray(ref_axis, dtype=float)[:2]

    def part_xy():
        p = ctx.obj_pos(name)
        if not steer_bottom:
            return p[:2]
        return (p - half * ctx.obj_axis(name))[:2]

    # optional pre-align over the reference axis
    if align_bottom_z is not None:
        eef_align = np.array([ref_xy()[0], ref_xy()[1],
                             align_bottom_z + half + (ctx.eef_pos()[2]
                             - ctx.obj_pos(name)[2])])
        move_eef(arm, eef_align, tol=0.003, gain=6.0)
        # nudge the part onto the ref axis
        for _ in range(15):
            err = part_xy() - ref_xy()
            if np.linalg.norm(err) < align_tol:
                break
            eef = ctx.eef_pos()
            arm.move_eef(eef - np.array([err[0], err[1], 0.0]),
                        gain=6.0, tol=0.0005)

    best, stuck, wiggles = np.inf, 0, 0
    best_dn, no_prog = np.inf, 0
    eef0 = ctx.eef_pos().copy()
    frozen_ref = ref_xy()          # freeze at press start
    for i in range(max_steps):
        pnow = ctx.obj_pos(name)
        eef = ctx.eef_pos()
        bot_z = pnow[2] - half
        if bot_z <= to_z:
            break
        if bot_z < best - 0.0001:
            best, stuck = bot_z, 0
        else:
            stuck += 1
            if stuck >= 15:
                if wiggles >= max_wiggles:
                    if verbose:
                        print(f"  [thread] stalled at z={bot_z:.4f} "
                              f"(target {to_z:.4f})")
                    break
                wiggles += 1
                if verbose:
                    print(f"  [thread] wiggle at z={bot_z:.4f}")
                _press_wiggle(ctx, arm, name, to_z, half, eef0.copy())
                stuck = 0
                continue
        d = frozen_ref - part_xy()
        dn = float(np.linalg.norm(d))
        if dn > steer:
            if dn < best_dn - 5e-5:
                best_dn, no_prog = dn, 0
            else:
                no_prog += 1
            if no_prog >= 6:
                tgt = np.array([eef[0], eef[1],
                               eef[2] - steer_drop])
            else:
                k = min(0.5 * dn, 0.0006) / dn
                tgt = np.array([eef[0] + d[0] * k, eef[1] + d[1] * k,
                               eef[2] - steer_drop])
        else:
            best_dn, no_prog = np.inf, 0
            tgt = np.array([eef[0] + d[0], eef[1] + d[1],
                           eef[2] - drop])
        exc = tgt[:2] - eef0[:2]
        en = float(np.linalg.norm(exc))
        if en > steer_cap:
            tgt[:2] = eef0[:2] + exc * (steer_cap / en)
        arm.move_eef(tgt, gain=20.0, tol=0.0005, max_steps=1,
                    stall=False)
    if verbose:
        bot = ctx.obj_pos(name)[2] - half
        print(f"  [thread] bottom at z={bot:.4f} (target {to_z:.4f})")

    # forced seat: a wedge/edge-bitten hang stops the steered descent a
    # few mm short of the seat (measured on the Task A gear: stopped at
    # +3.7mm and the never-seated part rotated inside the pads and was
    # ejected 0.5 m on release).  Drive the last few mm straight down
    # so the part rests on its mechanical stop BEFORE the pads open --
    # a seated part cannot be ejected.  force_seat=False skips this
    # (the Task A latch pin: the high-gain tail rams the pin INTO the
    # soft blind-hole floor, whose stored load ejects it -- measured;
    # the pin instead settles to the floor on release)
    if force_seat:
        for _ in range(60):
            if ctx.obj_pos(name)[2] - half <= to_z + 0.0003:
                break
            eef = ctx.eef_pos()
            arm.move_eef(eef + np.array([0.0, 0.0, -0.0004]), gain=20.0,
                         tol=0.0003, max_steps=1, stall=False)

    # the descent contact torque can rotate the part inside the pads
    # (measured: the connector entered plumb and left the loop 5.8deg
    # tilted); re-level over the seat so a flange-on-rim part lands
    # flat BEFORE the fingers open
    if level:
        _level()

    # release sequence: hold to let the part settle on its seat (the
    # threading loop leaves the part hanging in the gripper; the contact
    # with the collar/stack-top needs a few steps to load before the
    # fingers open, otherwise the part shifts when the pads release)
    gripper.hold()
    settle(ctx, max_steps=20)
    if release:
        # lift the pads CLEAR of a protruding part head BEFORE opening:
        # on a deep part (the Task A latch pin) the part top sits level
        # with the pad tops (eef - 0.4mm), and the pads' fast opening
        # sweep (CLOSE_Q_STEP per control step) clips the protruding
        # head and levers the seated part out of its hole (measured:
        # pin flung out of the 8mm boss, off the table).  A 6mm rise
        # puts the pad tops 5.6mm above the part top; the bled-off hold
        # (zero servo error) means the part stays on its seat.
        # release_lift=0 disables the rise (the latch pin again: the
        # rise DRAGS the gripped pin UP its hole -- 7.5mm of engagement
        # shrinks to 1.5mm and the released pin topples out, measured;
        # with deep engagement the opening pads only clip a 0.1-0.3N
        # side nudge that the hole friction absorbs).
        if release_lift > 0:
            arm.move_eef(ctx.eef_pos() + np.array([0.0, 0.0, release_lift]),
                         gain=8.0, tol=0.004)
            settle(ctx, max_steps=6)
        if release_lift < 0:
            arm.move_eef(ctx.eef_pos() + np.array([0.0, 0.0, release_lift]),
                         gain=8.0, tol=0.004)
            settle(ctx, max_steps=6)
        # two-stage open: first command a ~1.9mm span RELEASE (the pads
        # creep back from the deep bite to a light touch -- the bled-off
        # hold means the retreat is friction-free, so a seated part
        # stays put), THEN the full open.  A single fast open scrapes
        # the pad inner faces along a deep bite and flicks the part
        # sideways out of its hole (measured on the Task A latch pin:
        # ejected 1.5m every run at bite 2mm/side; the two-stage open
        # never ejected it).  Skipped for release_lift<0 (press-down)
        # -- the press already parked the pad tops below the part top
        # and the partial open would let a deep-seated part lean in
        # the slack (measured).
        if release_lift >= 0:
            q = ctx.finger_qpos
            q_loose = 0.5 * (q[0] - q[1]) + 0.001
            ctx.set_finger_ctrl(q_loose, -q_loose)
            settle(ctx, max_steps=25)
        gripper.open()
        settle(ctx, max_steps=settle_steps)
        arm.move_eef(ctx.eef_pos() + np.array([0.0, 0.0, retreat]),
                   gain=10.0, tol=0.012)
        settle(ctx, max_steps=settle_steps)
    arm.rot_target = rot0
    return True


# ------------------------------------------------------------- rotate_eef
def rotate_eef(ctx, arm, gripper, axis_angle, steps=40):
    """Rotate the wrist by an axis-angle increment while holding a part."""
    arm.rotate_eef(axis_angle, steps=steps,
                   gripper=tuple(ctx.finger_qpos))
    return True


# ------------------------------------------------------- horizontal drive
def _drive_along(ctx, arm, name, direction, dist, speed=0.03,
                 stall_patience=30, max_steps=500, tag="drive",
                 verbose=False):
    """Closed-loop horizontal drive of a part along a unit *direction*.

    Commands the EEF toward the far end while watching the LIVE part
    displacement; quits on completion, stall (part stopped moving while
    the eef keeps loading) or step budget.  Returns the achieved
    displacement of the part along the direction (m, may be negative).
    """
    d = np.asarray(direction, dtype=float)[:2]
    d = d / np.linalg.norm(d)
    obj0 = ctx.obj_pos(name)[:2].copy()
    eef0 = ctx.eef_pos()
    # overshoot +0.01 (was +0.05): the drive quits at prog>=dist anyway,
    # and a long overshoot target stretched the arm past the point where
    # the post-release Cartesian lift crawls (measured: the eef stuck at
    # x~0.19 and every subsequent move stalled)
    far = np.array([eef0[0], eef0[1], eef0[2]]) \
        + np.array([d[0], d[1], 0.0]) * (dist + 0.01)
    best, stall = -np.inf, 0
    prog = 0.0
    for _ in range(max_steps):
        arm.move_eef(far, gain=6.0, tol=0.0, max_steps=1, stall=False,
                     max_speed=speed)
        prog = float((ctx.obj_pos(name)[:2] - obj0) @ d)
        if prog > best + 1e-5:
            best, stall = prog, 0
        else:
            stall += 1
            if stall >= stall_patience:
                if verbose:
                    print(f"  [{tag}] STALL dump (prog {prog*1000:.1f}mm)")
                    for ci in range(ctx.data.ncon):
                        c = ctx.data.contact[ci]
                        g1 = ctx.model.geom(c.geom1).name or "?"
                        g2 = ctx.model.geom(c.geom2).name or "?"
                        ff = np.zeros(6)
                        mujoco.mj_contactForce(ctx.model, ctx.data, ci, ff)
                        fn = float(np.linalg.norm(ff[:3]))
                        if fn > 0.05 and name[:3] in (g1 + g2):
                            print(f"      {g1} <-> {g2} {fn:.2f}N "
                                  f"pos={np.round(c.pos, 4)}")
                break
        if prog >= dist:
            break
    if verbose:
        print(f"  [{tag}] displaced {prog * 1000:.1f}mm of "
              f"{dist * 1000:.1f}mm")
    return prog


# --------------------------------------------------------- lateral insert
def lateral_insert(ctx, arm, gripper, name, into, depth, approach=0.02,
                   carry_speed=0.12, speed=0.03, wiggle=0.004,
                   max_wiggles=2, stall_patience=30, max_steps=500,
                   release=True, settle_steps=20, verbose=False):
    """Horizontally insert a held module along its rail into a connector.

    into: world xy of the rail entry / connector axis.  The module is
    carried so its centre sits ``approach`` short of the entry (along
    the rail direction = from the module's current position toward the
    entry), then a closed-loop horizontal drive pushes it in by
    ``depth``.  A stalled drive backs off by ``wiggle`` and retries
    (max_wiggles times) -- a rim catch releases on the reverse stroke.

    The whole skill is tool-down: the pads grip the module's side walls
    and only translate (no wrist reorientation), so it works at the
    home-family configurations where tilt servos are singular.
    Returns (ok, achieved_displacement).
    """
    into = np.asarray(into, dtype=float)[:2]
    obj = ctx.obj_pos(name)
    rail = into - obj[:2]
    rail = rail / np.linalg.norm(rail)
    # obj0 BEFORE the hover: the depth is measured from the part's
    # position at the skill start, so the drive only pushes the
    # remaining distance after the hover carry (capturing obj0 after
    # the hover pushed the FULL depth from the hover point and rammed
    # the part into the rack back plate, measured)
    obj0 = ctx.obj_pos(name)[:2].copy()

    # carry to the entry gap at the module's CURRENT height (the rail
    # defines the line; height errors show up as rail binding, which
    # the wiggle strokes relieve).  style='line': the safe_z default
    # would lift, translate, then descend ON TOP of the rails -- and
    # that final descent is exactly what the pads cannot do (measured:
    # the descent stalled, the part parked high over the rack)
    eef = ctx.eef_pos()
    off = eef - obj
    hover_c = into - rail * approach
    goal = np.array([hover_c[0] + off[0], hover_c[1] + off[1], eef[2]])
    if not move_eef(arm, goal, style="direct", max_speed=carry_speed):
        return False, 0.0
    settle(ctx, max_steps=10)

    for w in range(max_wiggles + 1):
        done = float((ctx.obj_pos(name)[:2] - obj0) @ rail)
        if done >= depth - 1e-4:
            break
        _drive_along(ctx, arm, name, rail, depth - done, speed=speed,
                     stall_patience=stall_patience, max_steps=max_steps,
                     tag=f"lateral w{w}", verbose=verbose)
        if w < max_wiggles:
            done = float((ctx.obj_pos(name)[:2] - obj0) @ rail)
            if done < depth - 1e-4:
                # back off along the rail and retry
                e = ctx.eef_pos()
                arm.move_eef(e - np.array([rail[0], rail[1], 0.0])
                             * wiggle, gain=6.0, tol=0.0, max_steps=10,
                             stall=False)
                settle(ctx, max_steps=5)
    total = float((ctx.obj_pos(name)[:2] - obj0) @ rail)

    if release:
        gripper.hold()
        settle(ctx, max_steps=10)
        if release == "burst":
            # instant open: a SLEWED open drags a light seated part UP
            # out of its seat on the pad friction (measured: the
            # latchbar rode +13mm on the slow open)
            ctx.set_finger_ctrl(gripper.open_q, -gripper.open_q)
        elif release == "relax":
            # bleed the squeeze fully, let the part settle on its
            # seat, then slew open -- the light parts get dragged
            # by an immediate open and flung by a burst (measured:
            # plug +59mm / bolt 57mm ejections); the relaxed open
            # drops them straight (the lid recipe)
            settle(ctx, max_steps=20)
            gripper.open()
            settle(ctx, max_steps=10)
        else:
            gripper.open()
        settle(ctx, max_steps=settle_steps)
        # clear the pads vertically first (the 200mm lift parks the eef
        # at ~0.99, where the horizontal retract is reachable -- at
        # 0.92 the long horizontal stalled mid-way, measured)
        arm.move_eef(ctx.eef_pos() + np.array([0.0, 0.0, 0.07]),
                     gain=10.0, tol=0.012)
        settle(ctx, max_steps=settle_steps)
    if verbose:
        print(f"  [lateral_insert {name}] total {total * 1000:.1f}mm of "
              f"{depth * 1000:.1f}mm")
    return total >= 0.8 * depth, total


# ------------------------------------------------------------------ slide
def slide(ctx, arm, gripper, name, direction, dist, speed=0.05,
          stall_patience=40, max_steps=600, release=False,
          settle_steps=15, verbose=False):
    """Slide a held part along a rail by ``dist`` (long-travel slide-in).

    Pure closed-loop advance of a part already seated in a rail entry
    (the alignment/landing is a previous place step's job).  Stall-aware:
    quits when the part stops tracking the eef (a lid hitting a
    protruding latch stops here, and the recorded displacement feeds the
    stage predicate).  Returns the achieved displacement (m).
    """
    prog = _drive_along(ctx, arm, name, direction, dist, speed=speed,
                        stall_patience=stall_patience, max_steps=max_steps,
                        tag="slide", verbose=verbose)
    if release:
        gripper.hold()
        settle(ctx, max_steps=10)
        gripper.open()
        settle(ctx, max_steps=settle_steps)
        arm.move_eef(ctx.eef_pos() + np.array([0.0, 0.0, DEFAULT_LIFT]),
                     gain=10.0, tol=0.012)
        settle(ctx, max_steps=settle_steps)
    return prog


# -------------------------------------------------------------- snap press
def snap_press(ctx, arm, gripper, at, z_top, press=0.004, f_stop=1.2,
               step=0.0001, max_steps=250, hold_steps=20,
               verbose=False):
    """Press a latch tab down with the closed pads until it clicks.

    at / z_top: world xy and top-face z of the tab.  The closed finger
    pads act as the press head: descend to a 3mm hover, then creep down
    in ``step`` slices until the pad contact force reaches ``f_stop``
    or the ``press`` depth budget is consumed; hold briefly (let the
    wedge settle past its catch), then retreat.
    Returns (peak_force, pressed_depth_mm).
    """
    at = np.asarray(at, dtype=float)[:2]

    def _pad_force():
        return (ctx.geom_contact_force("finger1_pad_collision")
                + ctx.geom_contact_force("finger2_pad_collision"))

    gripper.close_to_span(0.012)        # closed pads = press head
    if not move_eef(arm, np.array([at[0], at[1], z_top + 0.05]),
                    max_speed=0.15):
        return 0.0, 0.0
    arm.move_eef(np.array([at[0], at[1], z_top + 0.003]), gain=8.0,
                 tol=0.0015)
    settle(ctx, max_steps=6)
    f_peak, pressed = 0.0, 0.0
    for _ in range(max_steps):
        f_peak = max(f_peak, _pad_force())
        if f_peak >= f_stop:
            break
        eef = ctx.eef_pos()
        arm.move_eef(eef + np.array([0.0, 0.0, -step]), gain=8.0,
                     tol=0.0, max_steps=1, stall=False)
        pressed += step
        if pressed >= press:
            break
    for _ in range(hold_steps):
        ctx.step()
    f_peak = max(f_peak, _pad_force())
    arm.move_eef(ctx.eef_pos() + np.array([0.0, 0.0, 0.06]), gain=10.0,
                 tol=0.008)
    settle(ctx, max_steps=10)
    if verbose:
        print(f"  [snap_press] peak={f_peak:.2f}N "
              f"pressed={pressed * 1000:.2f}mm")
    return f_peak, pressed * 1000.0


# -------------------------------------------------------------- press fit
def press_fit(ctx, arm, gripper, name, into, to_z, half=None,
              step=0.0003, max_steps=300, align_tol=0.001, level=True,
              contact_geom=None, release=True, settle_steps=20,
              verbose=False):
    """Force-controlled vertical press of a held part into a bore.

    into: bore axis xy (semantic ref resolved by the executor); to_z:
    target bottom z of the part (the mechanical seat).  Levels the part
    plumb, aligns it over the bore, then creeps down in ``step`` slices;
    stops at the seat or the step budget.  contact_geom (optional): a
    geom name on the pressed part whose contact force is tracked as the
    press force (peak reported).  Returns (ok, depth_mm, peak_force).
    """
    if half is None:
        meta = perception.part_meta(ctx, name)
        half = meta["half_h"]
    into = np.asarray(into, dtype=float)[:2]
    rot0 = arm.rot_target

    if level:
        _level_axis(ctx, arm, name, verbose=verbose)

    # hover + pre-align nudge over the bore axis (same pattern as the
    # thread insert pre-align)
    obj = ctx.obj_pos(name)
    off = ctx.eef_pos() - obj
    hover = np.array([into[0] + off[0], into[1] + off[1],
                      ctx.eef_pos()[2] + 0.03])
    if not move_eef(arm, hover, max_speed=0.12):
        return False, 0.0, 0.0
    for _ in range(15):
        err = ctx.obj_pos(name)[:2] - into
        if np.linalg.norm(err) < align_tol:
            break
        eef = ctx.eef_pos()
        arm.move_eef(eef - np.array([err[0], err[1], 0.0]), gain=6.0,
                     tol=0.0005, max_steps=10, stall=False)
    settle(ctx, max_steps=10)

    bot_z0 = ctx.obj_pos(name)[2] - half
    # OVER-TRAVEL guard: pressing more than 3mm PAST the nominal seat
    # gap means the "stop" below is yielding (a friction-hanging stack,
    # not a rigid bore shoulder) -- keep pressing and the servo crushes
    # the whole assembly down (measured on a Task A bearing press:
    # 19.6mm of stack collapse).  A real press stops on resistance.
    z_floor = to_z - 0.003
    f_peak = 0.0
    best, stuck = bot_z0, 0
    for _ in range(max_steps):
        bot_z = ctx.obj_pos(name)[2] - half
        if bot_z <= to_z + 0.0003:
            break
        if bot_z <= z_floor:
            if verbose:
                print(f"  [press_fit {name}] over-travel stop at "
                      f"z={bot_z:.4f} (seat {to_z:.4f} yielding?)")
            break
        if bot_z < best - 0.0001:
            best, stuck = bot_z, 0
        else:
            stuck += 1
            if stuck >= 12:
                if verbose:
                    print(f"  [press_fit {name}] jam stop at z={bot_z:.4f} "
                          f"(target {to_z:.4f})")
                break
        if contact_geom is not None:
            f_peak = max(f_peak, ctx.geom_contact_force(contact_geom))
        eef = ctx.eef_pos()
        arm.move_eef(eef + np.array([0.0, 0.0, -step]), gain=10.0,
                     tol=0.0, max_steps=1, stall=False)
    # forced seat: drive the last stretch straight down so the part
    # rests on its mechanical stop BEFORE the pads open (bounded by
    # the same over-travel floor)
    for _ in range(60):
        bot_z = ctx.obj_pos(name)[2] - half
        if bot_z <= to_z + 0.0003 or bot_z <= z_floor:
            break
        eef = ctx.eef_pos()
        arm.move_eef(eef + np.array([0.0, 0.0, -0.0004]), gain=20.0,
                     tol=0.0003, max_steps=1, stall=False)
    if contact_geom is not None:
        f_peak = max(f_peak, ctx.geom_contact_force(contact_geom))
    depth = max(0.0, bot_z0 - (ctx.obj_pos(name)[2] - half))

    if release:
        gripper.hold()
        settle(ctx, max_steps=15)
        gripper.open()
        settle(ctx, max_steps=settle_steps)
        arm.move_eef(ctx.eef_pos() + np.array([0.0, 0.0, DEFAULT_LIFT]),
                     gain=10.0, tol=0.012)
        settle(ctx, max_steps=settle_steps)
    arm.rot_target = rot0
    if verbose:
        print(f"  [press_fit {name}] depth={depth * 1000:.2f}mm "
              f"f_peak={f_peak:.2f}N "
              f"tilt={ctx.obj_tilt(name):.2f}deg")
    ok = depth >= 0.5 * max(0.0, bot_z0 - to_z)
    return ok, depth * 1000.0, f_peak


# ------------------------------------------------------------ screw drive
def screw_drive(ctx, arm, gripper, name, into, to_z, half=None,
                pitch=0.005, dz_step=0.0002, max_steps=600, level=True,
                release=True, settle_steps=20, verbose=False):
    """Screw a held hex-head bolt into a boss hole (controlled helix).

    The parallel pads grip the hex head ACROSS ITS FLATS and transmit
    torque (a round head measurably slips).  MuJoCo has no thread
    constraint, so the helix is commanded at the CONTROL layer: each
    control step the wrist rotates about world z by d_theta while the
    eef descends by dz_step, with d_theta = -2*pi*dz_step/pitch
    (clockwise = righty-tighty for a downward feed).  The bolt turns
    WITH the pads (flat-to-flat contact, no slip at these torque
    levels) and the boss walls merely steady the shank.

    into: boss axis xy; to_z: target bolt bottom z (seat).  Stops at
    the seat or the step budget.  Returns (depth_mm, tilt_deg).
    """
    if half is None:
        meta = perception.part_meta(ctx, name)
        half = meta["half_h"]
    into = np.asarray(into, dtype=float)[:2]
    rot0 = arm.rot_target

    if level:
        _level_axis(ctx, arm, name, verbose=verbose)

    # hover over the boss axis + pre-align nudge on the live bolt
    obj = ctx.obj_pos(name)
    off = ctx.eef_pos() - obj
    hover = np.array([into[0] + off[0], into[1] + off[1],
                      ctx.eef_pos()[2] + 0.03])
    if not move_eef(arm, hover, max_speed=0.12):
        return 0.0, 99.0
    for _ in range(15):
        err = ctx.obj_pos(name)[:2] - into
        if np.linalg.norm(err) < 0.001:
            break
        eef = ctx.eef_pos()
        arm.move_eef(eef - np.array([err[0], err[1], 0.0]), gain=6.0,
                     tol=0.0005, max_steps=10, stall=False)
    settle(ctx, max_steps=10)

    bot_z0 = ctx.obj_pos(name)[2] - half
    d_theta = -2.0 * np.pi * dz_step / pitch
    c, s = np.cos(d_theta), np.sin(d_theta)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    for _ in range(max_steps):
        bot_z = ctx.obj_pos(name)[2] - half
        if bot_z <= to_z + 0.0003:
            break
        # advance the wrist orientation target about world z while the
        # eef descends -- _step_ik rides the rotation into the same DLS
        # solve, so the pads translate + turn as one rigid press head
        arm.rot_target = Rz @ arm.rot_target
        eef = ctx.eef_pos()
        arm.move_eef(eef + np.array([0.0, 0.0, -dz_step]), gain=10.0,
                     tol=0.0, max_steps=1, stall=False)
    # forced seat: a few straight-down steps so the bolt rests on the
    # hole bottom before the pads open
    for _ in range(60):
        if ctx.obj_pos(name)[2] - half <= to_z + 0.0003:
            break
        eef = ctx.eef_pos()
        arm.move_eef(eef + np.array([0.0, 0.0, -0.0004]), gain=20.0,
                     tol=0.0003, max_steps=1, stall=False)
    depth = max(0.0, bot_z0 - (ctx.obj_pos(name)[2] - half))
    tilt = ctx.obj_tilt(name)

    if release:
        # the hex head sits flush in its socket: open straight up, no
        # lateral sweep (the bolt is a seated part, it cannot be ejected).
        # Re-plumb first: the helix descent leaves the nut a few deg
        # leaning (measured 3.4deg), and a leaning nut on its loose
        # stud pivots off the seat rim the moment the pads let go
        # (measured: ended on the floor 45mm away).
        if level:
            _level_axis(ctx, arm, name, verbose=verbose)
        gripper.hold()
        settle(ctx, max_steps=15)
        gripper.open()
        settle(ctx, max_steps=settle_steps)
        arm.move_eef(ctx.eef_pos() + np.array([0.0, 0.0, DEFAULT_LIFT]),
                     gain=10.0, tol=0.012)
        settle(ctx, max_steps=settle_steps)
    arm.rot_target = rot0
    if verbose:
        print(f"  [screw_drive {name}] depth={depth * 1000:.2f}mm "
              f"tilt={tilt:.2f}deg")
    return depth * 1000.0, tilt
