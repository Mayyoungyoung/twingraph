"""Task planner: skill-sequence plans with semantic landmarks.

A ``TaskPlan`` is an ordered list of ``(skill_name, params_dict)`` tuples.
Params use **semantic landmarks** (site names, body names, relative offsets)
never absolute world coordinates, so that base-height changes or scene
re-layouts do not invalidate the plan.

``nominal_plan(task)`` returns the hand-authored optimal sequence for each
task.  ``generate_plan(task, llm_output=None)`` is the future interface
for LLM-generated plans (stub: delegates to nominal_plan).
"""
import numpy as np

from .skills import perception


# ---------------------------------------------------------------- helpers
def axis_xy(body_name):
    """Semantic ref: live xy of a body (for insert ref_axis)."""
    return {"ref": body_name, "mode": "axis"}


def top_z(body_name, offset=0.0):
    """Semantic ref: top z of a body (center + half_h + offset)."""
    return {"ref": body_name, "mode": "top", "offset": offset}


def floor_z(body_name, offset=0.0):
    """Semantic ref: floor z of a body (center - half_h + offset)."""
    return {"ref": body_name, "mode": "floor", "offset": offset}


def seated_z(ref_body, part_name):
    """Semantic ref: z where *part_name* seats on *ref_body*'s floor
    (ref floor + part half_h)."""
    return {"ref": ref_body, "mode": "seated", "part": part_name}


def center_z(body_name, offset=0.0):
    """Semantic ref: center z of a body + offset (e.g. shaft collar
    + 2mm for the gear seat bottom).  Use this when the body's
    half_h in PART_META is a reference value, not the geometric
    half-height (e.g. shaft half_h=0.073 is not the bounding-box
    half-height but a root-z offset)."""
    return {"ref": body_name, "mode": "center", "offset": offset}


def site_z(site_name, offset=0.0):
    """Semantic ref: z of a site + offset (e.g. tray floor + half_h)."""
    return {"ref": site_name, "mode": "site", "offset": offset}


def boss_axis(body_name):
    """Semantic ref: live xy of the boss axis of a body (boss sits at
    boss_ring_r in the body's local +x — for insert ref_axis)."""
    return {"ref": body_name, "mode": "boss_axis"}


def offset_axis(body_name, lx, ly):
    """Semantic ref: live xy of a point at local (lx, ly) in a body's
    frame (e.g. a stud bolt offset from the housing centre — for
    screw_drive / insert ref_axis)."""
    return {"ref": body_name, "mode": "offset", "local": (lx, ly)}


# ---------------------------------------------------------------- plan
class TaskPlan:
    """Ordered skill sequence with failure-propagation mode."""

    def __init__(self, steps, fail_mode="continue", name=""):
        self.steps = steps          # list of (skill_name, params_dict)
        self.fail_mode = fail_mode  # "continue" or "abort"
        self.name = name

    def __iter__(self):
        return iter(self.steps)

    def __len__(self):
        return len(self.steps)

    def __repr__(self):
        return f"TaskPlan({self.name!r}, {len(self.steps)} steps, " \
               f"fail={self.fail_mode})"


# ---------------------------------------------------------------- nominal plans
def _A_pick(part, press, yaw=None, grasp_dz=None, tol=0.004,
            repress=True):
    """Task A pick loop (two-class): detect -> plan_grasp_pose ->
    plan_path -> move_to -> grasp.  Planning steps compute and store
    artifacts; execution steps consume them.  tol defaults to the old
    GRASP_TOL (4mm) -- the tight 1.5mm descent combined with the new
    transport leg flicked heavy parts out of the pads at release
    (measured on the shaft: 17deg tilt); only the gear/cover keep the
    proven 1.5mm descent."""
    return [
        ("detect_part", {"part": part}),
        ("plan_grasp_pose", {"part": part, "yaw": yaw,
                             "grasp_dz": grasp_dz}),
        ("plan_path", {"to": {"part": part, "lift": 0.15},
                       "style": "safe_z", "as": f"path_{part}_pick"}),
        ("move_to", {"plan": f"path_{part}_pick", "tol": 0.0015,
                     "max_speed": 0.20}),
        ("grasp", {"part": part, "plan": "grasp", "press": press,
                   "tol": tol, "repress": repress}),
    ]


def _A_carry_place(part, at, z_ref, press, carry_speed=0.12, align=True,
                   settle_steps=60, release="slew"):
    """Task A place loop: plan_path -> transport (carry) -> place
    (descent + release) -> settle."""
    return [
        ("plan_path", {"to": {"at": at, "z": z_ref, "lift": 0.12},
                       "style": "safe_z", "as": f"path_{part}_carry"}),
        ("transport", {"part": part, "plan": f"path_{part}_carry",
                       "carry_speed": carry_speed, "press": press}),
        ("place", {"part": part, "at": at, "z_ref": z_ref,
                   # carry_direct: the transport already brought the part
                   # to a hover above the target -- the place must not
                   # safe_z re-lift (the descend-rise-twitch artifact)
                   "carry_direct": True,
                   "carry_speed": carry_speed, "align": align,
                   "release": release, "settle_steps": settle_steps}),
        ("settle", {"max_steps": 10}),
    ]


def _A_carry_insert(part, into, to_z, align_bottom_z, press, **kw):
    """Task A thread-insert loop: plan_path -> transport -> insert
    (bottom-steered threading) -> settle."""
    return [
        ("plan_path", {"to": {"at": into, "z": align_bottom_z,
                              "lift": 0.12},
                       "style": "safe_z", "as": f"path_{part}_carry"}),
        ("transport", {"part": part, "plan": f"path_{part}_carry",
                       "carry_speed": kw.pop("carry_speed", 0.12),
                       "press": press}),
        ("insert", {"part": part, "mode": "thread", "into": into,
                    "to_z": to_z, "align_bottom_z": align_bottom_z,
                    **kw}),
        ("settle", {"max_steps": 10}),
    ]


def nominal_plan_A():
    """Task A gearbox: 8 assembly stages + 1 quality-inspection stage,
    each assembly stage expanded into the two-class action loop
    (detect -> plan_grasp_pose -> plan_path -> move_to -> grasp ->
    plan_path -> transport -> place/insert).  ~70 atomic steps.

    Stages: housing(place) -> shaft(place into hole) -> gear/spacer/
    bearing(thread) -> pin(thread) -> cover(place) -> retainer(thread)
    -> S9 final quality inspection.  A mid-chain inspection + rework
    nudge follows the S2 shaft placement.

    The execution recipes (press values, thread-insert parameters,
    release sequences, seat heights) are the measured-stable Task A
    baseline; the planning steps around them only add the plan/execute
    handshake and the fault-injection surface (see faults.FailureModel).
    """
    return TaskPlan([
        # S0: seed-controlled incoming-part scatter (fault source:
        # 初始散布 -- zero for the 'none' profile)
        ("scatter_parts", {"parts": ["housing", "shaft", "gear",
                                     "spacer", "bearing", "pin", "cover",
                                     "retainer"]}),

        # S1: housing into the tray (heavy -> slow carry + align +
        # long settle).  yaw 0: the pads close along world x --
        # closing along y would drive the pads into the stud bolts.
        *_A_pick("housing", press=0.003, yaw=0.0),
        *_A_carry_place("housing", "tray_center",
                        site_z("tray_center", 0.0225),
                        press=0.003, carry_speed=0.12, align=True,
                        settle_steps=80),

        # S2: shaft into the housing centre hole (drop, not threading).
        # Align to TRAY_XY (the predicate reference), not the live
        # housing axis.  grasp_dz=0.015 grasps low on the upper section.
        *_A_pick("shaft", press=0.0025, grasp_dz=0.015),
        *_A_carry_place("shaft", "tray_center",
                        top_z("housing", offset=0.002),
                        press=0.0025, carry_speed=0.12, align=True,
                        settle_steps=60),

        # S2b: mid-chain quality inspection + conditional rework (the
        # nudge is skipped when the measured shaft error is in spec;
        # a rework push is the recovery behavior on the failure path).
        # xy_tol 3mm is TIGHTER than the S2 stage gate (3.5mm) so the
        # rework fires BEFORE the stage can fail.
        ("inspect", {"mode": "shaft", "part": "shaft",
                     "ref": "tray_center", "xy_tol": 0.003,
                     "tag": "shaft_ok"}),
        ("nudge", {"part": "shaft", "to_target": "tray_center",
                   "unless": "shaft_ok",
                   "push_z": center_z("shaft", offset=0.05),
                   "speed": 0.04}),

        # S3: gear threads onto the shaft collar (the chain's only deep
        # hole engagement -- plumb-entry servo + generous wiggles).
        # repress=False: the hole is wide enough to swallow a pad face.
        *_A_pick("gear", press=0.003, tol=0.0015, repress=False),
        *_A_carry_insert("gear", axis_xy("shaft"),
                         center_z("shaft", offset=0.0065),
                         center_z("shaft", offset=0.068),
                         press=0.003, half=0.006, drop=0.0015,
                         max_wiggles=10, settle_steps=30,
                         # level: the gear hangs at the 7deg eef tilt --
                         # at 23mm of engagement the tilted hole digs
                         # into the shaft 2.8mm, over the 1.58mm
                         # clearance = jam + 10 wiggles every run
                         # (measured); plumb it before the descent
                         level=True),

        # S4: spacer stacks on the gear top
        *_A_pick("spacer", press=0.001),
        *_A_carry_insert("spacer", axis_xy("shaft"), top_z("gear"),
                         top_z("gear"), press=0.001, half=0.006,
                         drop=0.0015, settle_steps=30, level=True),

        # S5: bearing stacks on the spacer top
        *_A_pick("bearing", press=0.001),
        *_A_carry_insert("bearing", axis_xy("shaft"), top_z("spacer"),
                         top_z("spacer"), press=0.001, half=0.006,
                         drop=0.0015, settle_steps=30, level=True),

        # S6: latch pin BEFORE the cover (pads would collide with the
        # cover boss during the pin thread).  Gentle 0.5mm/step drop +
        # plumb level + press-down release (the 3-PASS recipe).
        *_A_pick("pin", press=0.004),
        *_A_carry_insert("pin", boss_axis("housing"),
                         top_z("housing", offset=0.0005),
                         top_z("bearing", offset=0.002),
                         press=0.004, half=0.026,
                         max_steps=400, steer_cap=0.001,
                         steer_drop=0.0005, max_wiggles=10,
                         drop=0.0005, level=True, force_seat=False,
                         release_lift=-0.001),

        # S7: cover over the pin (two-stage release clears the pads
        # above the protruding pin head before the full open)
        *_A_pick("cover", press=0.003, tol=0.0015, yaw=np.pi / 4),
        *_A_carry_place("cover", axis_xy("shaft"),
                        top_z("bearing", offset=0.005),
                        press=0.003, carry_speed=0.12, align=True,
                        release="two_stage", settle_steps=60),

        # S8: bearing retainer ring threads onto the shaft upper end
        # and seats on the cover top (the stable thread finale; the
        # stud-nut screw drive was retired -- see parts.RETAINER)
        *_A_pick("retainer", press=0.0015, yaw=np.pi / 4),
        *_A_carry_insert("retainer", axis_xy("shaft"), top_z("cover"),
                         top_z("cover", offset=0.020),
                         press=0.0015, half=0.004, drop=0.0015,
                         settle_steps=30,
                         # the retainer's hole clearance is only 0.5mm
                         # -- plumb it or it jams on the shaft side
                         level=True),

        # S9: final quality inspection (retainer seat / cover seat /
        # pin engagement, measured under sensor noise).  The stage
        # predicate reads state["inspect_verdict"].  The retainer is a
        # loose ring that rides up the shaft end (measured resting z
        # 0.905-0.912 vs nominal 0.903), so its z gate is two-sided
        # and 8mm wide.
        ("settle", {"max_steps": 30}),
        ("inspect", {"mode": "final", "checks": [
            {"part": "retainer", "kind": "seat",
             "xy_ref": "tray_center",
             "z_ref": top_z("cover", offset=0.004),
             "xy_tol": 0.006, "z_tol": 0.008, "tilt_tol": 10.0},
            {"part": "cover", "kind": "seat",
             "xy_ref": "tray_center",
             "z_ref": top_z("bearing", offset=0.005),
             "xy_tol": 0.005, "z_tol": 0.005, "tilt_tol": 8.0},
            {"part": "pin", "kind": "pin_engage",
             "axis_ref": boss_axis("housing"),
             "top_ref": top_z("housing", offset=0.008),
             "half": 0.026, "min_depth": 0.006, "bot_off_tol": 0.006},
        ]}),
        ("settle", {"max_steps": 5}),
    ], fail_mode="continue", name="taskA_gearbox")


def nominal_plan_C():
    """Task C fixture: 9 stages → atomic skill steps.

    Stages: grasp workpiece → place cavity → settle → pins → side clamp
    → top clamp → datum detect → tool move → confirm.
    """
    return TaskPlan([
        # S1: grasp workpiece (light press for 0.83N plate)
        ("grasp", {"part": "workpiece", "press": 0.0025,
                   "lift": 0.15}),

        # S2: carry + release the plate into the cavity.  Release the
        # plate ~9mm above the seated root (bottom 0.815, just above
        # the 6mm walls) and let it drop in -- the gripper cannot enter
        # the wall zone.  safe_z carry + live_align: the 71g plate
        # slides inside the pads on a straight hop; the low-gain loop
        # re-centres the LIVE plate over the cavity before the drop,
        # then steers its yaw back via tangential eef nudges.
        ("place", {"part": "workpiece", "at": "cavity_center",
                   "z_ref": 0.820,
                   "carry_speed": 0.05, "carry_style": "safe_z",
                   "live_align": True, "low_carry": True,
                   "release": "burst"}),

        # S3: guided settle (press plate top)
        ("settle_press", {"part": "workpiece", "press": 0.001}),

        # S4: engage dual locator pins (fixture-executed)
        ("drive_position", {"actuator": "pinA_act", "target": 0.011,
                            "dur": 0.4, "settle": 8}),
        ("drive_position", {"actuator": "pinB_act", "target": 0.011,
                            "dur": 0.4, "settle": 16}),

        # S5: side clamp (force-controlled push + retract); 8mm travel
        # pushes the plate to the -x wall and keeps ~0.5N servo contact
        ("drive_force", {"actuator": "side_act", "q_start": 0.0,
                         "q_min": -0.008, "v": 0.005, "f_stop": 1.5,
                         "dur_max": 2.0, "geom": "side_g",
                         "hold": 0.3}),
        ("drive_position", {"actuator": "side_act", "target": 0.0,
                            "dur": 0.3, "settle": 8}),

        # S6: top clamp (force-controlled press on the -x edge;
        # travels from the raised rest ~52mm down onto the edge)
        ("drive_force", {"actuator": "clamp_act", "q_start": 0.0,
                         "q_min": "workpiece_top", "v": 0.008,
                         "f_stop": 5.0, "dur_max": 8.0,
                         "geom": "clamp_g", "ctrl_offset": 0.0005,
                         "hold": 0.5}),

        # S7: grasp probe + detect datum
        ("grasp", {"part": "probe", "press": 0.002,
                   "lift": 0.15}),
        ("detect_datum", {"datum": "datum_nominal",
                          "tip_geom": "probe_tip"}),

        # S8: compensated tool move
        ("move_tool", {"target": "op_nominal",
                       "compensate": "measured_offset"}),

        # S9: confirm
        ("settle", {"max_steps": 5}),
    ], fail_mode="continue", name="taskC_fixture")


def _B_pick(part, press, yaw=None, grasp_dz=None, tol=0.004,
            repress=True, squeeze=None, lift=0.07):
    """Task B pick loop (two-class): detect -> plan_grasp_pose ->
    plan_path -> move_to -> grasp."""
    gp = {"part": part, "plan": "grasp", "press": press, "tol": tol,
          "repress": repress, "lift": lift}
    if squeeze:
        gp["squeeze"] = squeeze
    return [
        ("detect_part", {"part": part}),
        ("plan_grasp_pose", {"part": part, "yaw": yaw,
                             "grasp_dz": grasp_dz}),
        ("plan_path", {"to": {"part": part, "lift": 0.15},
                       "style": "safe_z", "as": f"path_{part}_pick"}),
        ("move_to", {"plan": f"path_{part}_pick", "tol": 0.0015,
                     "max_speed": 0.20}),
        ("grasp", gp),
    ]


def _B_carry(part, to_ref, press, carry_speed=0.12, plan="path_carry",
             lift=0.12):
    """Task B carry loop: plan_path -> transport to a hover above the
    target (the following skill does the fine descent/drive)."""
    return [
        ("plan_path", {"to": to_ref, "style": "safe_z", "as": plan,
                       "lift": lift}),
        ("transport", {"part": part, "plan": plan,
                       "carry_speed": carry_speed, "press": press}),
    ]


def nominal_plan_B():
    """Task B rack-server module assembly: 6 stages expanded into the
    two-class action loop (detect -> plan_grasp_pose -> plan_path ->
    move_to -> grasp -> plan_path -> transport -> insert/place).

    Stages: S1 PSU slide-in (lower rail) -> S2 power plug thread-insert
    (PSU top well, BEFORE the module covers it) -> S3 compute module
    slide-in (upper rail) -> S4 lid place (post tops) -> S5 latchbar
    slide-bolt (lid channel) -> S6 continuity test (probe touch).

    The execution recipes (grasp presses, squeeze bites, slide speeds,
    release styles) are the proven-stable rack pipeline (68-iteration
    tuning); the planning steps add the plan/execute handshake and the
    fault-injection surface (see faults.FailureModel).
    """
    return TaskPlan([
        # S0: seed-controlled incoming-part scatter
        ("scatter_parts", {"parts": ["psu", "module", "powercon", "lid",
                                     "latchbar", "probe"]}),

        # S1: PSU slide into the lower rail.  Squeeze 2N: the deep bite
        # holds the 70mm box rigid through the horizontal drive.
        *_B_pick("psu", press=0.004, squeeze=2.0, yaw=np.pi / 2,
                 lift=0.10),
        *_B_carry("psu", {"at": "rail_psu_drop", "z": 0.90,
                          "lift": 0.0},
                  press=0.004, plan="path_psu_carry"),
        ("lateral_insert", {"part": "psu", "into": "rail_psu_in",
                            "at_z": 0.842,
                            # depth 135mm: the drive overshoots the
                            # target (the part starts 1-2mm inside the
                            # rail mouth and the 150mm budget ends
                            # 16mm past the rack centre -- measured)
                            "depth": 0.135,
                            "approach": 0.02, "speed": 0.03,
                            "max_wiggles": 2}),
        ("home", {}),
        ("home", {}),

        # S2: power plug into the PSU top-face surface socket (MUST
        # precede the module slide -- the module covers the socket).
        # The plug is a light box: place it (align + descend onto the
        # top face + slow release) -- the socket walls funnel it in;
        # the old thread insert could never seat it (the "sunk well"
        # was sealed by the solid box -- see gen_sceneB)
        *_B_pick("powercon", press=0.002, yaw=0.0),
        *_B_carry("powercon",
                  {"at": offset_axis("psu", 0.035, 0.0),
                   "z": center_z("psu", offset=0.040), "lift": 0.10},
                  press=0.002, plan="path_powercon_carry"),
        ("place", {"part": "powercon",
                   "at": offset_axis("psu", 0.035, 0.0),
                   # release HIGH (the plug bottom 4mm above the socket
                   # walls): the plug free-falls ~20mm into the funnel
                   # socket.  Descending to the seat wedges the plug in
                   # the socket and the opening pads drag it back out
                   # (+58mm measured)
                   "z_ref": center_z("psu", offset=0.038),
                   "carry_direct": True,
                   "carry_speed": 0.10, "align": False,
                   "release": "slew", "settle_steps": 30}),

        # S3: compute module slide into the upper rail
        *_B_pick("module", press=0.004, squeeze=2.0, yaw=np.pi / 2,
                 lift=0.10),
        *_B_carry("module", {"at": "rail_mod_drop", "z": 0.94,
                             "lift": 0.0},
                  press=0.004, plan="path_module_carry"),
        ("lateral_insert", {"part": "module", "into": "rail_mod_in",
                            "at_z": 0.895, "depth": 0.15,
                            "approach": 0.02, "speed": 0.03,
                            "max_wiggles": 2}),
        ("home", {}),
        ("home", {}),

        # S4: lid place onto the four post tops (release high +12mm so
        # the swinging edge clears the module fins; relax release)
        *_B_pick("lid", press=0.003, tol=0.0015),
        *_B_carry("lid",
                  {"at": "lid_home", "z": site_z("lid_home", 0.012),
                   "lift": 0.12},
                  press=0.003, plan="path_lid_carry"),
        ("place", {"part": "lid", "at": "lid_home",
                   "z_ref": site_z("lid_home", 0.012),
                   "carry_direct": True,
                   "carry_speed": 0.10, "align": True,
                   "release": "relax", "settle_steps": 60}),
        ("settle", {"max_steps": 10}),

        # S5: latchbar dropped into its channel on the lid top (2026-
        # 09-02: the horizontal slide-bolt drive kept dragging the
        # light lid -- the pads' hang is uncontrolably noisy and the
        # bolt either rode the wall tops or scraped the floor; the
        # vertical drop into the 28mm channel is the reliable recipe).
        # The bolt lands near the channel stop end, between the walls.
        *_B_pick("latchbar", press=0.002, yaw=np.pi / 2),
        *_B_carry("latchbar",
                  {"at": offset_axis("lid", -0.045, 0.0),
                   "z": top_z("lid", offset=0.020), "lift": 0.0},
                  press=0.002, plan="path_latchbar_carry"),
        ("place", {"part": "latchbar",
                   "at": offset_axis("lid", -0.045, 0.0),
                   "z_ref": top_z("lid", offset=0.006),
                   "carry_direct": True,
                   "carry_speed": 0.10, "align": True,
                   "release": "slew", "settle_steps": 30}),
        ("settle", {"max_steps": 10}),

        # S6: continuity test -- probe pick via the two-class loop.
        # Shallow bite (no squeeze): the deep bite rolls the rod out
        # during the tip servo (measured 4.96N bite -> 28 -> 90deg);
        # the shallow bite survives the short 160mm carry and the
        # slice-servo press judges the contact peak (the old taskB
        # proven recipe -- its sleeve is now close enough)
        *_B_pick("probe", press=0.002, grasp_dz=0.010, lift=0.15),
        ("continuity_test", {"tip_geom": "probe_tip",
                             "pad_geom": "lid_pad1",
                             "hover": 0.005, "over_travel": 0.03,
                             "grip_hold": 1.5}),

        # wrap-up: park the probe over a clear region and release in
        # place (the hanging-rod place drifts ~6cm; release is reliable).
        # NO final home: the joint-space sweep knocks the seated lid
        # and bolt off the rack (measured 56/50mm -- see gen_sceneB)
        ("move", {"to": (0.02, -0.12, 0.91), "style": "safe_z"}),
        ("release", {"settle_steps": 25}),
    ], fail_mode="continue", name="taskB_rack")


# ---------------------------------------------------------------- dispatch
def nominal_plan(task):
    """Return the nominal TaskPlan for *task* ('A', 'B', or 'C')."""
    if task == "A":
        return nominal_plan_A()
    if task == "B":
        return nominal_plan_B()
    if task == "C":
        return nominal_plan_C()
    raise ValueError(f"unknown task {task!r}")


def generate_plan(task, llm_output=None):
    """Future LLM interface.  For now, delegates to nominal_plan.

    When *llm_output* is provided, it should be a parsed skill sequence
    (list of (skill, params) tuples) from an LLM planner.  The translation
    layer will validate and convert it into a TaskPlan.
    """
    if llm_output is None:
        return nominal_plan(task)
    # TODO: validate and convert llm_output into a TaskPlan
    raise NotImplementedError("LLM plan generation not yet implemented")
