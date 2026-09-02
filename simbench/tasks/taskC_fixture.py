"""Task C: 9-stage precision-fixture loading chain (transplanted from
assembly/fixture_env.py + fixture_skills.py).

Stages: S1 grasp workpiece -> S2 place into cavity -> S3 guided settle ->
S4 dual locators (FIXTURE) -> S5 side clamp (FIXTURE) -> S6 top clamp
(FIXTURE, force ctrl) -> S7 datum detection (grasped probe) -> S8 tool
move -> S9 pre-machining confirmation.

Design carried over verbatim: every skill executes against FIXED nominal
coordinates (cavity center, datum at (0.022, 0), op point at (0.008,
0.009)); the S2 placement residual propagates through pins/clamps into
the S7 datum read (dimple capture band ~2.2mm) and the S9 operation
error.  Stage predicates are the loose symbolic-success gates of
fixture_env.py (same epsilons).

Physics cost note: this scene runs dt=0.0005 with ~10k contacts, so one
20Hz control step (100 substeps) costs ~1.7s wall -- the skill tolerances
are kept loose and settles short (the old demo did the same: "each
control step costs ~1s wall").

Robosuite->simbench mapping (the measured pitfalls collapse):
  - gripper cmd=0 HOLD  -> finger ctrl simply stays at its last command
    (nothing integrates, nothing drifts open);
  - light close + qpos rewind -> close_to_span leaves the servo command
    AT the target span, which IS the elastic press hold;
  - single-burst release -> one direct ctrl write to open (the slewed
    Gripper.open is a ring-release guard, not needed for the light plate
    grip; the old demo explicitly measured burst-release as correct).
"""
import mujoco
import numpy as np

from ..core.controller import Gripper
from ..skills import perception
from ..skills.motion import move_eef
from ..skills.settle import settle

STAGES = ["grasp", "place", "settle", "pins", "sideclamp", "topclamp",
          "datum", "tool", "confirm"]

# arena references (match gen_sceneC.py)
TABLE_TOP_Z = 0.8
FIXTURE_XY = np.array([0.15, 0.0])
SLEEVE_XY = np.array([-0.28, -0.28])
CAVITY_FLOOR_Z = TABLE_TOP_Z + 0.006          # 0.806
WP_SEATED_ROOT_Z = CAVITY_FLOOR_Z + 0.005     # 0.811
WP_TOP_Z = WP_SEATED_ROOT_Z + 0.005           # 0.816
WP_START_ROOT_Z = TABLE_TOP_Z + 0.015 + 0.005  # 0.82 (pedestal)
PROBE_START_ROOT_Z = TABLE_TOP_Z + 0.030      # 0.83
PROBE_L = 0.030                               # root -> tip bottom

# fixture actuator references (FixtureArena)
PIN_LIFT = 0.011
CLAMP_QMAX = 0.0
SIDE_RANGE = (-0.008, 0.0)   # demo travel (plate push + ~0.5N hold)
CLAMP_SIZE_Z = 0.003
CLAMP_ROOT_Z = 0.870

# loose symbolic thresholds (fixture_env.py, pilot-calibrated)
PLACE_XY_TOL = 0.002
PLACE_YAW_TOL = 5.0
PIN_FULL_DEPTH = 0.009
PIN_PARTIAL_DEPTH = 0.005
SIDE_FORCE_MIN = 0.4
CLAMP_FORCE_MIN = 1.5        # demo; the research gate was 2.0N
DATUM_CAPTURE = 0.0022
TOOL_RESID_TOL = 0.005
OP_ERR_TOL = 0.004           # demo (tip pendulum); research: 0.0015
CAV_HX = 0.039
CAV_HY = 0.023

# grip recipes (fixture_skills.py measurements)
# workpiece: LIGHT close -- pad inner gap 35mm on the 40mm breadth
# (2.5mm press-in per side): no stored ejection energy, the 0.83N plate
# is carried by friction.  A deep squeeze (span 47.6 in the old rig)
# ejects the plate on ANY release.  Pad-centre span in THIS rig =
# inner gap + 2*PAD_HALF_Y = 35 + 8 mm.
WP_LIGHT_SPAN = 0.043
WP_GRIP_DZ = 0.002
# probe: 5mm rod, pin-style tight stop (thinner rod -> do not push out)
PROBE_GRIP_DZ = 0.020

# carry speed cap for the held plate (POC-scaled: the old rig travels
# ~30-40 mm/s saturated; 0.12 m/s swung the plate out of the pads here)
CARRY_MAX_SPEED = 0.05


# ------------------------------------------------------------------ helpers
def _eef_yaw(ctx):
    R = ctx.eef_mat()
    return float(np.arctan2(R[1, 0], R[0, 0]))


def _goto_ax(arm, target, tol_xy=0.004, tol_z=0.002, gain=8.0,
             max_steps=120):
    """goto with xy and z judged SEPARATELY (z-critical legs; on a long
    vertical leg a 3D-norm tolerance lets the xy residual hide inside a
    converged z and vice versa)."""
    target = np.asarray(target, dtype=float)
    for _ in range(max_steps):
        eef = arm.ctx.eef_pos()
        d = target - eef
        if np.linalg.norm(d[:2]) < tol_xy and abs(d[2]) < tol_z:
            return True
        arm.move_eef(eef + d, gain=gain, tol=0.0, max_steps=1,
                     stall=False)
    eef = arm.ctx.eef_pos()
    d = target - eef
    return bool(np.linalg.norm(d[:2]) < tol_xy and abs(d[2]) < tol_z * 2)


def workpiece_err(ctx):
    """(x_err, y_err, yaw_rad) vs the nominal cavity center."""
    pos = ctx.obj_pos("workpiece")
    return (pos[0] - FIXTURE_XY[0], pos[1] - FIXTURE_XY[1],
            ctx.obj_yaw("workpiece"))


def tip_xy(ctx):
    """Probe tip geom center xy (tip is vertical during S7-S9)."""
    gid = ctx.geom_id("probe_tip")
    return np.array(ctx.data.geom_xpos[gid])[:2]


def tip_z(ctx):
    return float(ctx.data.geom_xpos[ctx.geom_id("probe_tip")][2])


def op_point_xy(ctx):
    return ctx.site_pos("op_point")[:2]


def datum_nominal_xy(ctx):
    return ctx.site_pos("datum_nominal")[:2]


def op_nominal_xy(ctx):
    return ctx.site_pos("op_nominal")[:2]


def pin_depth(ctx, tag):
    jid = ctx.joint_id(f"pin{tag}_j")
    return float(ctx.data.qpos[ctx.model.jnt_qposadr[jid]])


def pin_class(depth):
    if depth >= PIN_FULL_DEPTH:
        return "FULL"
    if depth >= PIN_PARTIAL_DEPTH:
        return "PARTIAL"
    return "FAILED"


# -------------------------------------------------- fixture actuators (S4-S6)
# The fixture executes through its own actuators; the robot ctrl stays at
# its last command the whole time (position servos hold it).
def _ramp_act(ctx, name, target, dur_s):
    """Linearly ramp fixture actuator @name to @target over dur_s."""
    cid = ctx.act_id(name)
    n = max(1, int(round(dur_s / ctx.control_dt)))
    for k in range(n):
        ctx.data.ctrl[cid] = target * k / max(n - 1, 1)
        ctx.step()
    ctx.data.ctrl[cid] = target


def _drive_until_force(ctx, ctrl_name, q_start, q_min, v, f_stop,
                       dur_max, geom_name, ctrl_offset=0.0, hold_s=0.3):
    """Substep force-controlled drive (the POC clamp_down loop).

    A position servo commanded lower per control step embeds the tool
    ~0.6mm into the plate before the next force check (measured 16kN).
    Drive on raw mj_step instead, with a force check every 5 substeps
    (~20um of travel between checks at the lowered speeds).
    """
    cid = ctx.act_id(ctrl_name)
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
        # silent no-contact failure is a trap: a too-short dur_max
        # expires BEFORE first contact and the stage reports 0N
        print(f"[warn] {ctrl_name} drive exhausted {dur_max}s without "
              f"reaching f_stop={f_stop}N (peak {f_max:.2f}N)")
    for _ in range(int(hold_s / dt)):
        mujoco.mj_step(ctx.model, ctx.data)
        ctx.n_physics_steps += 1
    # the press force DECAYS during the hold as the cell plate yields,
    # so report the peak: the predicate reads "made contact and pressed"
    f_max = max(f_max, ctx.geom_contact_force(geom_name))
    return f_max


def engage_pins(ctx, verbose=True):
    """S4: raise the dual locator pins (fixture-executed)."""
    _ramp_act(ctx, "pinA_act", PIN_LIFT, 0.4)
    settle(ctx, max_steps=8)
    _ramp_act(ctx, "pinB_act", PIN_LIFT, 0.4)
    settle(ctx, max_steps=16)
    return True


def side_clamp(ctx, verbose=True):
    """S5: push the +x edge toward the -x wall, force-limited, then
    retract (the side+top+wall three-way pinch stores elastic energy in
    the cell network and detonates the top clamp otherwise)."""
    f = _drive_until_force(ctx, "side_act", 0.0, SIDE_RANGE[0],
                           v=0.003, f_stop=1.5, dur_max=1.5,
                           geom_name="side_g")
    _ramp_act(ctx, "side_act", 0.0, 0.3)
    settle(ctx, max_steps=8)
    return f


def clamp_down(ctx, verbose=True):
    """S6: force-controlled press on the -x edge (POC clamp_down).

    The descent target puts the clamp bottom 1mm BELOW the live plate
    top; the command tracks 0.5mm below the descended position so the
    servo keeps a bounded press through its forcerange while the loop
    stops on contact force.  dur_max covers the full ~28mm descent at
    v=0.008 (3.5s + margin; a 3.0s budget expired before contact and
    the press read 0N -- measured pitfall).
    """
    pos = ctx.obj_pos("workpiece")
    z_min = (pos[2] + 0.005 - 0.004 + CLAMP_SIZE_Z - CLAMP_ROOT_Z)
    f = _drive_until_force(ctx, "clamp_act", CLAMP_QMAX, z_min,
                           v=0.008, f_stop=5.0, dur_max=5.0,
                           geom_name="clamp_g", ctrl_offset=0.0005,
                           hold_s=0.5)
    return f


# --------------------------------------------------------------------- S1
def grasp_workpiece(ctx, arm, gripper, noise_std=0.0, verbose=True):
    """S1: top-down grasp of the plate from its pedestal.

    Returns (ok, offset) with offset = eef - plate root.
    """
    meta = perception.part_meta(ctx, "workpiece")
    grip_dz = meta["grasp_dz"]
    true_pos = ctx.obj_pos("workpiece").copy()
    if noise_std > 0.0:
        true_pos = true_pos + np.random.normal(0.0, noise_std, 3)
    x, y, z = true_pos
    if verbose:
        print(f"  [grasp workpiece] at {np.round(true_pos, 4)} "
              f"(dz={grip_dz:+.4f})")

    # hover -> open -> descend (no post-descend re-centre: even ~5mm off
    # centre the sustained close clamps and lifts -- measured in the old
    # rig, where a re-centre loop actually moved the eef to a spot where
    # the pads caught the pedestal lip and never touched the plate)
    # SAFE-Z carry: a single straight hop drags the hanging part down
    # while translating and swings it out of the pads (measured here:
    # the plate dropped mid-carry, found flat on the floor)
    move_eef(arm, np.array([x, y, z + 0.15]), tol=0.004,
             max_speed=0.25, style="safe_z")
    gripper.open()
    _goto_ax(arm, np.array([x, y, z + grip_dz]),
             tol_xy=0.006, tol_z=0.002, gain=8.0, max_steps=100)

    # TWO-PHASE close.  POC closed against the pedestal with one
    # full-close command until the span gate hit -- in THIS rig the
    # spring-loaded servo (kp=1000, no finger damping) slams from 84mm
    # to 38.7mm span in 3 control steps and the deep wedge press
    # ejects the plate on lift (measured).  Instead: slew-press on the
    # pedestal (stalls at ~zero press but leaves the command deep, so
    # the servo keeps ~2.6N of lock force -- enough to lift), lift the
    # first leg, then a HANGING slow press lands exactly on
    # WP_LIGHT_SPAN for the carry.
    gripper.close_to_span(WP_LIGHT_SPAN)
    settle(ctx, max_steps=6)
    arm.move_eef(np.array([x, y, z + 0.05]), gain=8.0, tol=0.008)
    settle(ctx, max_steps=6)
    span = gripper.close_to_span(WP_LIGHT_SPAN)
    settle(ctx, max_steps=10)
    pz = ctx.obj_pos("workpiece")
    if verbose:
        print(f"      * span after close {span if span is not None else gripper.span():.4f} "
              f"plate_z={pz[2]:.4f}")

    # second lift leg
    arm.move_eef(np.array([x, y, z + 0.15]), gain=8.0, tol=0.010)

    # re-center so the carry offset stays purely vertical
    pnow = ctx.obj_pos("workpiece")
    eef = ctx.eef_pos()
    if np.linalg.norm((eef - pnow)[:2]) > 0.004:
        arm.move_eef(np.array([pnow[0], pnow[1], eef[2]]),
                     gain=6.0, tol=0.004)

    pnow = ctx.obj_pos("workpiece")
    if pnow[2] < z + 0.02:
        if verbose:
            print(f"  [grasp workpiece] FAIL: not lifted "
                  f"({pnow[2]:.4f} vs {z:.4f})")
        return False, None
    offset = ctx.eef_pos() - pnow
    if verbose:
        print(f"  [grasp workpiece] OK offset={np.round(offset, 4)}")
    return True, offset


# --------------------------------------------------------------------- S2
def place_workpiece(ctx, arm, gripper, offset, drop_dz=0.040,
                    settle_steps=12, verbose=True):
    """S2: carry + release the plate into the cavity (nominal center).

    HIGH RELEASE + FREE FALL: the gripper cannot enter the wall zone (a
    descent to the wall tops blew the contact buffer in the old rig);
    drop_dz=40mm leaves the plate bottom 14mm clear and it free-falls
    onto the cavity floor.  Accuracy comes from the live-plumb align:
    the carried plate hangs at a FIXED offset from the eef which is NOT
    the grasp offset -- descending on the grasp offset drove the plate
    ~15mm off centre and pitched it out of the pads (measured on every
    variant).  Whatever x/y/yaw the plate lands with IS the terminal
    state S3-S9 live with.
    """
    drop = np.array([FIXTURE_XY[0], FIXTURE_XY[1],
                     WP_SEATED_ROOT_Z + drop_dz])
    # SAFE-Z carry at 0.12 m/s: a straight hop with the 0.4m horizontal
    # leg swings the hanging plate out of the pads (measured: dropped
    # mid-carry onto the floor)
    move_eef(arm, np.array([drop[0], drop[1], 0.9]), tol=0.010,
             gain=6.0, max_speed=CARRY_MAX_SPEED, style="safe_z")
    if verbose:
        pnow = ctx.obj_pos("workpiece")
        print(f"  [place] carry: part_at={np.round(pnow, 4)} "
              f"tilt={ctx.obj_tilt('workpiece'):.1f}deg")

    # ALIGN HIGH, one continuous low-gain proportional loop on the LIVE
    # plate error (burst schemes whipped the hanging plate into pendulum
    # swings in the old rig); break needs quiet steps so a residual
    # swing cannot fake convergence
    nom = FIXTURE_XY[:2]
    quiet = 0
    for i in range(150):
        pnow = ctx.obj_pos("workpiece")
        err = pnow[:2] - nom
        if verbose and i % 40 == 0:
            print(f"      * align i={i} err="
                  f"({err[0] * 1000:.1f},{err[1] * 1000:.1f})mm")
        if np.linalg.norm(err) < 0.0025:
            quiet += 1
            if quiet >= 10:
                break
        else:
            quiet = 0
        eef = ctx.eef_pos()
        arm.move_eef(np.array([eef[0] - err[0], eef[1] - err[1], eef[2]]),
                     gain=6.0, tol=0.0, max_steps=1, stall=False)
    settle(ctx, max_steps=6)

    # PLUMB TARGETING: descend on the live hang offset (a pure rigid
    # translate of the hanging plate; the grasp offset is wrong here)
    pnow = ctx.obj_pos("workpiece")
    plumb = ctx.eef_pos() - pnow
    drop_t = drop + plumb
    if verbose:
        print(f"  [place] plumb offset={np.round(plumb, 4)}")
    # two-leg descent with a mid settle; both legs stay far above the
    # wall zone so no wall contact until the free-fall after release
    eef = ctx.eef_pos()
    mid = eef + np.array([0.0, 0.0, (drop_t[2] - eef[2]) / 2.0])
    _goto_ax(arm, mid, tol_xy=0.003, tol_z=0.004, gain=6.0,
             max_steps=120)
    settle(ctx, max_steps=5)
    _goto_ax(arm, drop_t, tol_xy=0.002, tol_z=0.002, gain=6.0,
             max_steps=120)
    settle(ctx, max_steps=8)
    if verbose:
        pnow = ctx.obj_pos("workpiece")
        print(f"  [place] pre-release: part_at={np.round(pnow, 4)} "
              f"yaw={np.degrees(ctx.obj_yaw('workpiece')):.2f}deg")

    # SINGLE-BURST RELEASE + IMMEDIATE RETREAT: staged openings let the
    # press-in energy push the plate out SIDEWAYS between pauses
    # (measured: ejected 97mm, yaw -151deg).  One burst drops it
    # straight down while the arm gets out of the way at the same time.
    ctx.set_finger_ctrl(gripper.open_q, -gripper.open_q)
    for _ in range(3):
        ctx.step()
    arm.move_eef(drop + np.array([0.0, 0.0, 0.12]), gain=10.0,
                 tol=0.012)
    settle(ctx, max_steps=settle_steps)
    if verbose:
        pnow = ctx.obj_pos("workpiece")
        print(f"  [place] post-release: part_at={np.round(pnow, 4)} "
              f"yaw={np.degrees(ctx.obj_yaw('workpiece')):.2f}deg")
    return True


# --------------------------------------------------------------------- S3
def guided_settle(ctx, arm, verbose=True, press=0.001):
    """S3: robot-assisted seating (pins still down).

    Press the live plate top lightly with the open pads so any landing
    edge/slope is flattened against the cavity floor; the eef target
    sits 1mm below the top so a properly seated plate just gets a
    confirming touch (the bounded servo stalls honestly on contact).
    """
    pos0 = ctx.obj_pos("workpiece")
    top = pos0[2] + 0.005
    target = np.array([pos0[0], pos0[1], top - press])
    move_eef(arm, target + np.array([0.0, 0.0, 0.08]), tol=0.008,
             max_speed=0.2, style="safe_z")
    gripper_open = Gripper(ctx)
    gripper_open.open()
    _goto_ax(arm, target, tol_xy=0.005, tol_z=0.0015, gain=8.0,
             max_steps=60)
    settle(ctx, max_steps=6)
    arm.move_eef(target + np.array([0.0, 0.0, 0.10]), gain=12.0,
                 tol=0.012)
    pos1 = ctx.obj_pos("workpiece")
    slip = (pos1 - pos0)[:2]
    if verbose:
        print(f"  [settle] slip=({slip[0] * 1000:.2f},"
              f"{slip[1] * 1000:.2f})mm")
    return slip


# --------------------------------------------------------------------- S7a
def grasp_probe(ctx, arm, gripper, noise_std=0.0, verbose=True):
    """S7a: dead-centre grasp of the rod probe from its table sleeve.

    The probe is a 5mm rod standing 60mm tall: an off-centre grip closes
    the pads asymmetrically and the lift yanks it sideways out of the
    pads.  Re-seat the probe on its sleeve first (it drifts over the
    long S1-S6 rollout; the datum channel runs through the DETECTION,
    not the pickup).
    """
    ctx.set_obj_pose("probe", SLEEVE_XY, PROBE_START_ROOT_Z)
    settle(ctx, max_steps=6)
    true_pos = ctx.obj_pos("probe").copy()
    if noise_std > 0.0:
        true_pos = true_pos + np.random.normal(0.0, noise_std, 3)
    x, y, z = true_pos
    if verbose:
        print(f"  [grasp probe] at {np.round(true_pos, 4)}")

    move_eef(arm, np.array([x, y, z + 0.15]), tol=0.006,
             max_speed=0.25, style="safe_z")
    gripper.open()
    # live-align the eef over the probe at hover (pads bottom must clear
    # the probe top at z+0.030 -> align 40mm above the root), then a
    # pure vertical descend (a diagonal descend sweeps the rod top)
    align_z = z + 0.040
    for _ in range(80):
        pnow = ctx.obj_pos("probe")
        eef = ctx.eef_pos()
        if np.linalg.norm((pnow - eef)[:2]) < 0.002:
            break
        arm.move_eef(np.array([pnow[0], pnow[1], align_z]),
                     gain=8.0, tol=0.0, max_steps=1, stall=False)
    _goto_ax(arm, np.array([x, y, z + PROBE_GRIP_DZ]),
             tol_xy=0.0015, tol_z=0.0015, gain=6.0, max_steps=100)

    # tight stop span so the thin rod is not pushed out of the pads
    gripper.close_on_part(0.005, press=0.002)
    # vertical (frozen-xy) lifts in two legs
    arm.move_eef(np.array([x, y, z + 0.05]), gain=6.0, tol=0.0012,
                 max_steps=150)
    settle(ctx, max_steps=8)
    arm.move_eef(np.array([x, y, z + 0.15]), gain=6.0, tol=0.0015,
                 max_steps=150)
    settle(ctx, max_steps=12)

    pnow = ctx.obj_pos("probe")
    if pnow[2] < z + 0.02:
        if verbose:
            print(f"  [grasp probe] FAIL: not lifted "
                  f"({pnow[2]:.4f} vs {z:.4f})")
        return False, None
    offset = ctx.eef_pos() - pnow
    if verbose:
        print(f"  [grasp probe] OK offset={np.round(offset, 4)}")
    return True, offset


# --------------------------------------------------------------------- S7b
def detect_datum(ctx, arm, offset, verbose=True, gain=6.0):
    """S7b: dip the probe tip into the datum dimple and read the offset.

    The dimple (O6 x 2mm) self-centres the 0.8mm tip through REAL
    contact: the measured tip xy vs the NOMINAL dimple axis is the
    bounded-accuracy datum read.  A plate whose dimple lies outside the
    capture band leaves the tip resting on the flat top face --
    detection fails and the read is bogus (the early-success/late-failure
    channel).
    """
    nom = datum_nominal_xy(ctx)
    dimple_floor_z = WP_TOP_Z - 0.002
    tip_target = dimple_floor_z + 0.0005        # just above the floor
    eef_target = np.array([nom[0], nom[1],
                           tip_target + PROBE_L + offset[2]])
    move_eef(arm, eef_target + np.array([0.0, 0.0, 0.05]), tol=0.008,
             max_speed=CARRY_MAX_SPEED, style="safe_z")
    settle(ctx, max_steps=40)      # damp the rod swing BEFORE the dip
    _goto_ax(arm, eef_target, tol_xy=0.002, tol_z=0.0015, gain=gain,
             max_steps=120)
    settle(ctx, max_steps=30)      # let the tip rest on the floor

    tip = tip_xy(ctx)
    med = tip - nom
    tz = tip_z(ctx)
    # detected: tip BOTTOM entered the plate top by 0.5mm (tip centre
    # sits half_h above the bottom; the centre-vs-top compare could
    # never fire -- the tip bottoms out only 2mm down on the floor)
    detected = tz - 0.0045 < WP_TOP_Z - 0.0005
    if verbose:
        print(f"  [datum] med=({med[0] * 1000:.2f},"
              f"{med[1] * 1000:.2f})mm tip_z={tz:.4f} "
              f"detected={detected}")
    return med, bool(detected)


# --------------------------------------------------------------------- S8
def move_tool(ctx, arm, offset, med, verbose=True, gain=6.0):
    """S8: compensated move to the machining point.

    Target = nominal op point + the S7 measured offset (the robot
    believes what it measured).  A wrong or bogus datum read aims the
    tool at the wrong point, which S9 exposes against the LIVE
    op_point site.  The tool hovers ~6mm above the plate top.
    """
    tgt_xy = op_nominal_xy(ctx) + med
    zb = WP_TOP_Z + 0.006
    eef_t = np.array([tgt_xy[0], tgt_xy[1], zb + PROBE_L + offset[2]])
    move_eef(arm, eef_t + np.array([0.0, 0.0, 0.05]), tol=0.006,
             max_speed=0.2, style="safe_z")
    _goto_ax(arm, eef_t, tol_xy=0.003, tol_z=0.004, gain=gain,
             max_steps=100)
    settle(ctx, max_steps=8)
    resid = float(np.linalg.norm(tip_xy(ctx) - tgt_xy))
    if verbose:
        print(f"  [tool] tool_resid={resid * 1000:.2f}mm")
    return resid


# ----------------------------------------------------------- stage metrics
def stage_metrics(ctx, k, state):
    """Terminal-state variables of stage k (1-based), meters/deg/N.
    ``state`` carries the chain bookkeeping (grasp_off / slip / forces /
    measured offset / tool residual)."""
    x_err, y_err, yaw = workpiece_err(ctx)
    pos = ctx.obj_pos("workpiece")
    m = {}
    if k == 1:                                   # grasp
        m["z"] = float(pos[2])
        m["tilt"] = ctx.obj_tilt("workpiece")
        m["grasp_dx"] = float(state["grasp_off"][0])
        m["grasp_dy"] = float(state["grasp_off"][1])
        m["grasp_dyaw"] = float(np.degrees(state["grasp_off"][2]))
    elif k == 2:                                 # place
        m["x_err"] = x_err
        m["y_err"] = y_err
        m["yaw"] = np.degrees(yaw)
        m["z"] = float(pos[2])
    elif k == 3:                                 # guided settle
        m["x_err"] = x_err
        m["y_err"] = y_err
        m["z"] = float(pos[2])
        m["slip_x"] = float(state["slip_xy"][0])
        m["slip_y"] = float(state["slip_xy"][1])
    elif k == 4:                                 # dual locators
        m["pinA_depth"] = pin_depth(ctx, "A")
        m["pinB_depth"] = pin_depth(ctx, "B")
        m["pin_class"] = pin_class(pin_depth(ctx, "A"))
        m["tilt"] = ctx.obj_tilt("workpiece")
    elif k == 5:                                 # side clamp
        m["side_force"] = float(state["side_force"])
        m["x_err"] = x_err
        m["yaw"] = np.degrees(yaw)
    elif k == 6:                                 # top clamp (core)
        m["res_x"] = x_err
        m["res_y"] = y_err
        m["res_yaw"] = np.degrees(yaw)
        m["clamp_force"] = float(state["clamp_force"])
        m["z"] = float(pos[2])
    elif k == 7:                                 # datum detection
        m["measured_dx"] = float(state["measured_offset"][0])
        m["measured_dy"] = float(state["measured_offset"][1])
        m["measured_mag"] = float(
            np.linalg.norm(state["measured_offset"]))
        m["detected"] = int(state["datum_detected"])
    elif k == 8:                                 # tool move
        m["tool_resid"] = float(state["tool_resid"])
    elif k == 9:                                 # confirmation
        m["operation_error"] = float(
            np.linalg.norm(tip_xy(ctx) - op_point_xy(ctx)))
    return m


def stage_success(ctx, k, state):
    """LOOSE symbolic success per stage (fixture_env.py epsilons)."""
    x_err, y_err, yaw = workpiece_err(ctx)
    pos = ctx.obj_pos("workpiece")
    if k == 1:
        return (pos[2] > WP_START_ROOT_Z + 0.010
                and ctx.obj_tilt("workpiece") < 30.0)
    if k == 2:
        return (abs(pos[2] - WP_SEATED_ROOT_Z) < 0.004
                and abs(x_err) < PLACE_XY_TOL
                and abs(y_err) < PLACE_XY_TOL
                and abs(np.degrees(yaw)) < PLACE_YAW_TOL)
    if k == 3:
        return (abs(pos[2] - WP_SEATED_ROOT_Z) < 0.003
                and abs(x_err) < CAV_HX - 0.035
                and abs(y_err) < CAV_HY - 0.020)
    if k == 4:
        return (pin_depth(ctx, "A") >= PIN_PARTIAL_DEPTH
                and ctx.obj_tilt("workpiece") < 8.0)
    if k == 5:
        return (state["side_force"] > SIDE_FORCE_MIN
                and abs(x_err) < CAV_HX - 0.035
                and abs(np.degrees(yaw)) < 6.0)
    if k == 6:
        return (state["clamp_force"] > CLAMP_FORCE_MIN
                and abs(pos[2] - WP_SEATED_ROOT_Z) < 0.004)
    if k == 7:
        return (state["datum_detected"]
                and np.linalg.norm(state["measured_offset"])
                < DATUM_CAPTURE)
    if k == 8:
        return state["tool_resid"] < TOOL_RESID_TOL
    if k == 9:
        return float(np.linalg.norm(tip_xy(ctx) - op_point_xy(ctx))) \
            < OP_ERR_TOL
    return False
