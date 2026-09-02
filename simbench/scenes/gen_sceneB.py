#!/usr/bin/env python3
"""Generate the Task B (rack-server module assembly) scene XML for simbench.

Scene story: an open aluminium profile rack stands on the bench.  The robot
slides a PSU module and a compute module into two rail levels (horizontal
lateral_insert), drops a lock pin through each module's front ear (snap_press),
plugs the PSU power connector into the PSU's front-face seat (lateral_insert),
places the top lid onto the frame ledge, latches it with a slide-bolt
(lateral_insert) and finally touches the lid test pad with the continuity
probe.

8-stage chain:
  S1 slide in PSU -> S2 pin-lock PSU -> S3 slide in compute module ->
  S4 pin-lock module -> S5 mate power connector -> S6 place lid ->
  S7 slide-bolt the lid -> S8 continuity test.

All motions are proven-stable primitives (horizontal drive, vertical
press, place, tip touch) -- no free spinning, no thread faking.

Physics recipe: identical to Task A (dt=0.002, impratio 20, elliptic
cone, soft 0.004-1 solrefs on the contact-critical geoms).

World-z references (table_top_z = 0.8):
  rack base top 0.806  lower rail floor 0.806  PSU centre 0.827
  upper rail floor 0.852  module centre 0.871  lid seat 0.912
  lock pin seat 0.836 (through the ear hole)  connector axis 0.806+0.020
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))  # repo root (/home/jia/RAL)

import numpy as np  # noqa: E402

from simbench.scenes.parts import (  # noqa: E402
    fmt, geom_xml, site_xml, v3,
)

TABLE_TOP_Z = 0.8
# RACK at x 0.15 (the taskA tray coordinate): at 0.25 the eef is near
# the arm's reach boundary and the Cartesian servo sags ~30mm in z
# during the horizontal drive (measured -- the PSU was dragged into the
# rails)
RACK_XY = (0.15, 0.0)

# rack frame
RACK_BASE = (0.115, 0.085)    # base plate half-extents (230x170)
RACK_BASE_H = 0.002           # base plate top at 0.802: the tilted
                              # module's front-bottom corner sweeps to
                              # ~0.803 and caught the 0.804 plate edge
                              # mid-slide (measured stall at 54mm)
POST_HX = 0.011               # corner posts 22x22
# POST_H 112mm: posts top 0.916 = the lid seat, ~6mm above the compute
# module fins (0.910) and inside the arm's comfortable reach
POST_H = 0.114
# rail floors (module skirt bottom z): the support bars sit on the table
# (their tops at these z); the PSU seats at 0.831 = 0.808 + skirt 5mm +
# half 18mm.  Upper rail 0.868: the inter-level gap is 60mm -- the PSU
# (36mm) needs a WIDE slide window (the pads slip ~8mm and the tilted
# box sweeps +-9mm; a 50mm gap stalled every drive at the mouth).
RAIL_Z_LOWER = TABLE_TOP_Z + 0.008                # 0.808
RAIL_Z_UPPER = TABLE_TOP_Z + 0.080                # 0.880
# rails: support bar (full length) + inner guide wall on each side.
# The rails PROTRUDE 50mm past the frame front: the modules are dropped
# onto the protruding mouth (clear of the frame top beam, which would
# otherwise block the pads' vertical descent -- measured) and then
# pushed home along the rails.
RAIL_LEN = 0.260             # full rail length along x (half 0.130);
                             # centred on rx: the front 110mm protrudes
                             # clear of the frame top beam for the drop
RAIL_SUP_W = 0.012           # support bar half-width
RAIL_SUP_H = 0.004           # support bar half-height
RAIL_GUIDE_W = 0.004         # guide wall half-width
RAIL_GUIDE_H = 0.016         # guide wall half-height

# module box (PSU and compute module share the shell).  70mm across:
# the pads' max span is ~86.8mm and a 90mm box cannot be gripped at all
# (measured 0N contact at full span) -- 70mm leaves a real squeeze
MOD_LEN_H = 0.075            # 150mm along x
MOD_W = 0.070                # 70mm across y
MOD_PSU_H = 0.036            # 36mm tall: the PSU body top (0.849) must
                             # clear the upper support bars (0.850-0.858)
                             # hanging below the upper rail floor (the
                             # 42mm first cut rammed them and stalled)
MOD_COMP_H = 0.030
SKIRT_H = 0.005              # bottom slide skirt height
SKIRT_DX = 0.010             # skirt half-width; the skirt rides the
                             # support bars (y 0.039-0.063) and must
                             # REACH them -- the inboard first cut
                             # (y to 0.035) left a 4mm gap and the
                             # released module fell through between the
                             # bars onto the PSU below (measured)
# rail clearance: the guides must clear the CLOSED PADS around the box
# -- the pads' outer faces sit at span/2 + 4mm = 43mm on the 70mm box,
# so the guide inner faces need >=47mm (SKIRT_GAP 8mm put them at 43mm
# = exactly the pad faces, and the drop descent scraped them, measured)
SKIRT_GAP = 0.012

# PSU front ear (lock pin hole) + top-face power seat (sunk well)
EAR_HX = 0.012
EAR_HY = 0.014
EAR_HZ = 0.010                 # PSU ear height (20mm hole band: the
                             # seated pin needs the deeper guide -- 12mm
                             # let it lean 16deg and the release tipped
                             # it flat, measured)
EAR_HZ_MOD = 0.010              # module ear: 20mm hole band -- pin2 only
                                # bottoms on the flat PSU top, so the
                                # deeper hole is what holds it plumb
EAR_HOLE_R = 0.007           # lock pin hole (14mm: the 7mm square pin's
                             # 9.9mm diagonal + its ~2.4mm hang-swing needs
                             # the slack -- the 10mm first cut caught the
                             # rim every drop, measured)
EAR_ZX = -MOD_LEN_H - 0.008  # ear sits ahead of the front face
EAR_ZY = RAIL_Z_LOWER + MOD_PSU_H / 2.0
# power seat: SURFACE SOCKET on the PSU top face (2026-09-02).  The
# old "sunk well" was fake: the walls + floor were placed INSIDE the
# solid psu_body box, so the box top face sealed the mouth and the
# plug could never enter (it always rested on the box top -- measured
# psu_body <-> powercon_body contacts at the mouth rim).  The socket
# walls now sit ON the top face, open above: the plug drops in and
# seats on the box top itself.  Clearance 7mm/side (42x34 socket vs
# the 28x22 plug): the grasp close yaws the plug up to 5deg (measured
# 0.093 rad).
SEAT_HX, SEAT_HY, SEAT_HZ = 0.021, 0.017, 0.016
SEAT_ZX = MOD_LEN_H - 0.040  # front section of the top face
SEAT_ZY = RAIL_Z_LOWER + 0.005 + MOD_PSU_H - SEAT_HZ   # well floor z

# lock pins: 7x7 square dowels dropped through the ear hole.  The
# square flats give the tilted pads a non-rolling bite (the r4 round
# first cut rolled and its thread-release knocked it flat, measured).
# pin1 (lower) stands on the rack base plate; pin2 (upper) stands on
# the PSU top below the module ear.
PIN_HX = 0.0035
PIN_LEN1 = 0.055   # the pin top must stay below the upper module's
                    # slide path (bottom 0.863): the 70mm first cut
                    # poked to 0.874 and the sliding module knocked it
                    # flat (measured)
PIN_LEN2 = 0.070   # same length as pin1: the 50mm first cut
                             # closed with a weaker bite (0.77N vs 1.14N)
                             # and slipped out of the pads, measured

# power connector plug
CON_HX, CON_HY, CON_HZ = 0.014, 0.011, 0.008
CON_ZY = SEAT_ZY

# top lid + slide bolt (the lid rests on the four post tops -- sized
# 230x170 so its corners fully cover the posts at (+-104, +-74))
LID_HX, LID_HY, LID_HZ = 0.115, 0.085, 0.004
LID_SEAT_Z = TABLE_TOP_Z + RACK_BASE_H + POST_H    # frame top 0.926
LID_PAD_R, LID_PAD_H = 0.003, 0.001               # test pad bump (half-h
                                                   # matches executor's
                                                   # hardcoded pad_top
                                                   # +1mm)
LID_PAD_XY = (-0.055, 0.0)                       # test pad offset on the
                                                 # lid (-x half keeps the
                                                 # probe hover inside the
                                                 # reach envelope)
BOLT_HX, BOLT_HY, BOLT_HZ = 0.016, 0.010, 0.004   # 32x20x8 slide bolt
BOLT_Z = LID_SEAT_Z + LID_HZ + BOLT_HZ            # rides on the lid top

# probe (old taskB probe geometry, 60mm total)
PROBE_MAIN_R = 0.0025
PROBE_MID_R = 0.0015
PROBE_TIP_R = 0.0008
PROBE_MAIN_H = 0.042
PROBE_MID_H = 0.009
PROBE_TIP_H = 0.009
SLEEVE_XY = (0.02, -0.15)    # probe sleeve next to the rack (2026-09-02):
                             # the far corner (-0.28, -0.28) made the
                             # probe carry ~480mm and the shallow-bite
                             # hang dropped mid-carry every run (measured
                             # tilt 11->82deg); the ~160mm carry here is
                             # inside the proven stable envelope (clear
                             # of the rails y +-0.047 and every feed)
SLEEVE_R_OUT = 0.008
SLEEVE_R_HOLE = 0.0055
SLEEVE_H = 0.030
PROBE_START_ROOT_Z = TABLE_TOP_Z + 0.030           # 0.83

# derived part rest positions (on the bench, clear of the rack).  ALL
# feeds sit inside the arm's ~0.45m comfortable reach from the base at
# (-0.56, 0): the first layout (feeds out to x 0.22) put the eef past
# the reach boundary and every grasp approach stalled (measured).
PART_STARTS = {
    "psu":      (-0.14, -0.10, TABLE_TOP_Z + 0.001 + MOD_PSU_H / 2.0),
    # module at (-0.25, -0.10) (2026-09-02): the original (-0.14, 0.10)
    # spawn sits INSIDE the lid's 230x170 footprint now that the lid
    # moved to (-0.25, 0.10) -- the initial overlap ejected the module
    # during the bring-up settle (measured: pushed 46mm to +y); the
    # -0.25 x / -0.10 y corner is clear of every other feed and inside
    # the rotation-held reach envelope
    "module":   (-0.25, -0.10, TABLE_TOP_Z + 0.001 + MOD_COMP_H / 2.0),
    "powercon": (-0.16, 0.20, TABLE_TOP_Z + CON_HZ + 0.0005),
    # lid at x -0.25 (2026-09-02): the -0.35 spawn put the grasp hover
    # (-0.35, 0.1, 0.979) OUTSIDE the rotation-held reachable workspace
    # (the wrist-orientation constraint made the travel leg stall and
    # drift 7cm HIGH; free-wrist reaches it but every grasp carries the
    # orientation hold) -- -0.25 is the measured reachable boundary
    "lid":      (-0.25, 0.10, TABLE_TOP_Z + LID_HZ + 0.0005),
    "latchbar": (-0.22, 0.24, TABLE_TOP_Z + BOLT_HZ + 0.0005),
    "probe":    (SLEEVE_XY[0], SLEEVE_XY[1], PROBE_START_ROOT_Z),
}

# grasp metadata: [type_code, grasp_dz, outer_d, half_h]
PART_META = {
    # psu/module gripped across their 70mm breadth; dz=-3mm centres the
    # 16mm pad band on the box MID-WALL -- the box then swings
    # symmetrically in the pads during the horizontal drive (an upper
    # band grip left 20mm of box below the pads, which swung through
    # the rail gap and rammed the upper bars, measured)
    "psu":      (2, -0.003, MOD_W, MOD_PSU_H / 2.0),
    "module":   (2, -0.003, MOD_W, MOD_COMP_H / 2.0),
    "powercon": (2, 0.006, 0.028, CON_HZ),
    "lid":      (2, 0.012, 0.040, LID_HZ),
    "latchbar": (2, 0.004, 0.020, BOLT_HZ),
    "probe":    (3, 0.020, 0.005, 0.030),
}


def body_open(name, pos, free=True, indent=1):
    pad = " " * indent
    s = f'{pad}<body name="{name}" pos="{v3(*pos)}">\n'
    if free:
        s += f'{pad}  <freejoint name="{name}_j"/>\n'
    return s


def main():
    out = build()
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "taskB_rack.xml")
    with open(path, "w") as f:
        f.write(out)
    print(f"wrote {path} ({len(out)} bytes)")


def build():
    tz = TABLE_TOP_Z
    rx, ry = RACK_XY
    STATIC_CT = dict(contype="0", conaffinity="1")

    # ---- static world ------------------------------------------------
    world = []
    world.append('  <light pos="1.0 1.0 1.5" dir="-0.2 -0.2 -1" '
                 'directional="true" castshadow="true" '
                 'diffuse="0.9 0.9 0.9" specular="0.5 0.5 0.5"/>')
    world.append('  <light pos="-1.2 -0.8 1.8" dir="0.3 0.2 -1" '
                 'directional="true" castshadow="false" '
                 'diffuse="0.35 0.35 0.4" specular="0.2 0.2 0.2"/>')
    world.append(geom_xml("floor", "plane", (0, 0, 0), (2.5, 2.5, 0.1),
                          material="groundplane",
                          friction="1 0.005 0.0001", **STATIC_CT))
    world.append(geom_xml("table_top", "box", (-0.17, 0.0, tz - 0.025),
                          (0.61, 0.4, 0.025), rgba="0.45 0.42 0.38 1",
                          friction="1 0.005 0.0001", **STATIC_CT))
    for i, (lx, ly) in enumerate(((-0.70, -0.32), (0.36, -0.32),
                                  (-0.70, 0.32), (0.36, 0.32))):
        world.append(geom_xml(f"table_leg{i + 1}", "box", (lx, ly, 0.3875),
                              (0.03, 0.03, 0.3875), rgba="0.35 0.32 0.29 1",
                              **STATIC_CT))

    # ---- rack frame ---------------------------------------------------
    fr = [f'  <body name="rack" pos="{v3(rx, ry, 0)}">']
    # base plate
    fr.append(geom_xml("rack_base", "box", (0.0, 0.0, tz + RACK_BASE_H / 2),
                       (RACK_BASE[0], RACK_BASE[1], RACK_BASE_H / 2),
                       rgba="0.32 0.34 0.38 1", friction="1 0.005 0.0001",
                       indent=4, **STATIC_CT))
    # four corner posts
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            fr.append(geom_xml(
                f"rack_post_x{int(sx)}y{int(sy)}", "box",
                (sx * (RACK_BASE[0] - POST_HX),
                 sy * (RACK_BASE[1] - POST_HX),
                 tz + RACK_BASE_H + POST_H / 2.0),
                (POST_HX, POST_HX, POST_H / 2.0),
                rgba="0.50 0.52 0.56 1", friction="1 0.005 0.0001",
                indent=4, **STATIC_CT))
    # rail levels: for each level, two support bars + two guide walls.
    # The slide floor is the support bar top; the guide walls bound the
    # skirt laterally.  Guide inner faces at +-(MOD_W/2 + SKIRT_GAP).
    for lvl, zf in (("lower", RAIL_Z_LOWER), ("upper", RAIL_Z_UPPER)):
        zsup = zf - RAIL_SUP_H                        # bar centre
        for sy in (-1.0, 1.0):
            y_guide = sy * (MOD_W / 2.0 + SKIRT_GAP + RAIL_GUIDE_W)
            fr.append(geom_xml(
                f"rail_{lvl}_sup{int(sy)}", "box",
                (0.0, y_guide, zsup),
                (RAIL_LEN / 2.0, RAIL_SUP_W, RAIL_SUP_H),
                rgba="0.45 0.47 0.52 1", friction="0.3 0.005 0.0001",
                indent=4, **STATIC_CT))
            fr.append(geom_xml(
                f"rail_{lvl}_guide{int(sy)}", "box",
                (0.0, y_guide, zf + RAIL_GUIDE_H),
                (RAIL_LEN / 2.0, RAIL_GUIDE_W, RAIL_GUIDE_H),
                rgba="0.45 0.47 0.52 1", friction="0.3 0.005 0.0001",
                indent=4, **STATIC_CT))
    # (the lock-pin stages were retired with their sockets -- see the
    # task module docstring; the slide depth is drive-controlled)
    # back plate (stops the slide at depth; also the visual backplane)
    fr.append(geom_xml("rack_back", "box",
                       (RACK_BASE[0] - 0.006, 0.0,
                        tz + RACK_BASE_H + 0.040),
                       (0.006, RACK_BASE[1] - 0.012, 0.040),
                       rgba="0.28 0.30 0.34 1", friction="0.5 0.005 0.0001",
                       indent=4, **STATIC_CT))
    # (no frame top beam: it blocked the module drop path and the lid
    # descent -- the lid rests directly on the four post tops, a
    # classic open-frame look.  No staging planks either: the modules
    # slide in HELD via lateral_insert, whose at_z descend happens at
    # x=0 clear of the rails -- the plank prototypes blocked the lower
    # module's descent column and tilted it, measured)
    fr.append("  </body>")
    world.extend(fr)

    # probe sleeve (taskA/B pin sleeve recipe)
    n = 8
    unit_w = SLEEVE_R_OUT * np.sin(np.pi / n)
    unit_h = (SLEEVE_R_OUT - SLEEVE_R_HOLE) * np.cos(np.pi / n) / 2.0
    int_r = SLEEVE_R_HOLE * np.cos(np.pi / n) + unit_h
    world.append('  <body name="probe_sleeve" '
                 f'pos="{v3(SLEEVE_XY[0], SLEEVE_XY[1], 0)}">')
    for i in range(n):
        ang = 2.0 * np.pi * i / n
        world.append(geom_xml(
            f"probe_sleeve_w{i}", "box",
            (int_r * np.cos(ang), int_r * np.sin(ang), tz + SLEEVE_H / 2),
            (unit_h, unit_w, SLEEVE_H / 2),
            quat=f"{np.cos(ang/2):.6f} 0 0 {np.sin(ang/2):.6f}",
            rgba="0.55 0.50 0.45 1", friction="1 0.005 0.0001", indent=4,
            **STATIC_CT))
    world.append("  </body>")


    # ---- reference sites ----------------------------------------------
    # rail_*_drop: the module centre when PLACED onto the rail mouth
    # (rx-0.05: the back end rides the rails so the box does not tip);
    # rail_*_in: the module centre when pushed fully home (= rx).  The
    # push skill drives the blade through this target.
    world.append(site_xml("rail_psu_drop",
                          (rx - 0.15, ry, RAIL_Z_LOWER + MOD_PSU_H / 2.0)))
    world.append(site_xml("rail_mod_drop",
                          (rx - 0.15, ry, RAIL_Z_UPPER + MOD_COMP_H / 2.0)))
    world.append(site_xml("rail_psu_in",
                          (rx, ry, RAIL_Z_LOWER + MOD_PSU_H / 2.0)))
    world.append(site_xml("rail_mod_in",
                          (rx, ry, RAIL_Z_UPPER + MOD_COMP_H / 2.0)))
    world.append(site_xml("lid_home",
                          (rx, ry, LID_SEAT_Z + LID_HZ)))
    world.append(site_xml("probe_start",
                          (SLEEVE_XY[0], SLEEVE_XY[1], tz)))
    world.append(site_xml("bolt_in", (rx - 0.039, ry, BOLT_Z)))

    # ---- cameras --------------------------------------------------------
    world.append('  <camera mode="fixed" name="frontview" pos="1.6 0 1.45" '
                 'quat="0.56 0.43 0.43 0.56"/>')
    world.append('  <camera mode="fixed" name="agentview" pos="0.5 0 1.35" '
                 'quat="0.653 0.271 0.271 0.653"/>')
    world.append('  <camera mode="fixed" name="sideview" '
                 'pos="-0.0565 1.2761 1.488" '
                 'quat="0.0099 0.0069 0.5912 0.8064"/>')

    # ---- parts -----------------------------------------------------------
    parts = []
    custom = []
    for pname, (tcode, dz, od, hh) in PART_META.items():
        custom.append(
            f'    <numeric name="meta_{pname}" '
            f'data="{tcode} {fmt(dz)} {fmt(od)} {fmt(hh)}"/>')

    # PSU: main box + bottom slide skirts + front ear + power seat
    parts.append(body_open("psu", PART_STARTS["psu"]))
    parts.append(geom_xml("psu_body", "box", (0, 0, 0),
                          (MOD_LEN_H, MOD_W / 2.0, MOD_PSU_H / 2.0),
                          rgba="0.25 0.27 0.30 1", density=160.0, condim=4,
                          indent=2))
    for sy in (-1.0, 1.0):
        parts.append(geom_xml(
            f"psu_skirt{int(sy)}", "box",
            (0.0, sy * MOD_W / 2.0,
             -MOD_PSU_H / 2.0 + SKIRT_H / 2.0),
            (MOD_LEN_H - 0.006, SKIRT_DX, SKIRT_H / 2.0),
            rgba="0.35 0.37 0.42 1", density=160.0, condim=4,
            friction="0.3 0.005 0.0001", indent=2))
    # front ear (lock pin hole): ring of 8 boxes around the hole
    top = MOD_PSU_H / 2.0
    for i in range(8):
        ang = 2.0 * np.pi * i / 8
        r_out_b, r_hole_b = 0.009, EAR_HOLE_R
        unit_w = r_out_b * np.sin(np.pi / 8)
        unit_h = (r_out_b - r_hole_b) * np.cos(np.pi / 8) / 2.0
        int_rb = r_hole_b * np.cos(np.pi / 8) + unit_h
        parts.append(geom_xml(
            f"psu_ear{i}", "box",
            (EAR_ZX + int_rb * np.cos(ang),
             int_rb * np.sin(ang), top - 0.001 + EAR_HZ / 2.0),
            (unit_h, unit_w, EAR_HZ / 2.0),
            quat=f"{np.cos(ang/2):.6f} 0 0 {np.sin(ang/2):.6f}",
            rgba="0.35 0.37 0.42 1", density=160.0, condim=4,
            friction="0.5 0.005 0.0001", indent=2))
    # power seat recess (top face, sunk well near the front): four walls
    # + floor, open upward; the plug drops in and stands proud 4mm
    top_p = MOD_PSU_H / 2.0
    # surface socket: the walls rise from the top face
    z_sock = top_p + SEAT_HZ / 2.0
    for sx in (-1.0, 1.0):
        parts.append(geom_xml(
            f"psu_seat_wx{int(sx)}", "box",
            (SEAT_ZX + sx * (SEAT_HX + 0.002), 0.0, z_sock),
            (0.002, SEAT_HY + 0.004, SEAT_HZ / 2.0),
            rgba="0.20 0.22 0.26 1", density=160.0, condim=4,
            friction="0.4 0.005 0.0001", indent=2))
    for sy in (-1.0, 1.0):
        parts.append(geom_xml(
            f"psu_seat_wy{int(sy)}", "box",
            (SEAT_ZX, sy * (SEAT_HY + 0.002), z_sock),
            (SEAT_HX, 0.002, SEAT_HZ / 2.0),
            rgba="0.20 0.22 0.26 1", density=160.0, condim=4,
            friction="0.4 0.005 0.0001", indent=2))
    parts.append("  </body>")

    # compute module: main box + fins + skirts + front ear
    parts.append(body_open("module", PART_STARTS["module"]))
    parts.append(geom_xml("mod_body", "box", (0, 0, 0),
                          (MOD_LEN_H, MOD_W / 2.0, MOD_COMP_H / 2.0),
                          rgba="0.18 0.19 0.22 1", density=160.0, condim=4,
                          indent=2))
    for sy in (-1.0, 1.0):
        parts.append(geom_xml(
            f"mod_skirt{int(sy)}", "box",
            (0.0, sy * MOD_W / 2.0,
             -MOD_COMP_H / 2.0 + SKIRT_H / 2.0),
            (MOD_LEN_H - 0.006, SKIRT_DX, SKIRT_H / 2.0),
            rgba="0.28 0.30 0.35 1", density=160.0, condim=4,
            friction="0.3 0.005 0.0001", indent=2))
    # heatsink fins on top (visual)
    for k in range(5):
        parts.append(geom_xml(
            f"mod_fin{k}", "box",
            (-0.030 + 0.015 * k, 0.0, MOD_COMP_H / 2.0 + 0.004),
            (0.002, MOD_W / 2.0 - 0.008, 0.004),
            rgba="0.45 0.48 0.52 1", density=160.0, condim=4, indent=2))
    top = MOD_COMP_H / 2.0
    for i in range(8):
        ang = 2.0 * np.pi * i / 8
        r_out_b, r_hole_b = 0.009, EAR_HOLE_R
        unit_w = r_out_b * np.sin(np.pi / 8)
        unit_h = (r_out_b - r_hole_b) * np.cos(np.pi / 8) / 2.0
        int_rb = r_hole_b * np.cos(np.pi / 8) + unit_h
        parts.append(geom_xml(
            f"mod_ear{i}", "box",
            (EAR_ZX + int_rb * np.cos(ang),
             0.020 + int_rb * np.sin(ang), top - 0.001 + EAR_HZ_MOD / 2.0),
            (unit_h, unit_w, EAR_HZ_MOD / 2.0),
            quat=f"{np.cos(ang/2):.6f} 0 0 {np.sin(ang/2):.6f}",
            rgba="0.28 0.30 0.35 1", density=160.0, condim=4,
            friction="0.5 0.005 0.0001", indent=2))
    parts.append("  </body>")


    # power connector plug
    parts.append(body_open("powercon", PART_STARTS["powercon"]))
    parts.append(geom_xml("powercon_body", "box", (0, 0, 0),
                          (CON_HX, CON_HY, CON_HZ),
                          rgba="0.80 0.65 0.20 1", density=160.0, condim=4,
                          friction="0.4 0.005 0.0001", indent=2))
    parts.append("  </body>")

    # top lid + test pad + slide-bolt receiver walls
    parts.append(body_open("lid", PART_STARTS["lid"]))
    parts.append(geom_xml("lid_body", "box", (0, 0, 0),
                          (LID_HX, LID_HY, LID_HZ),
                          rgba="0.15 0.45 0.65 1", density=300.0, condim=4,
                          indent=2))
    # centre handle boss: the 190x150 lid is far past the 86mm pad
    # span, so the robot grips this 40x30x16 handle (a classic lid
    # handle look) and carries the lid by it
    parts.append(geom_xml("lid_grip", "box", (0, 0, LID_HZ + 0.008),
                          (0.020, 0.015, 0.008),
                          rgba="0.30 0.55 0.75 1", density=300.0, condim=4,
                          indent=2))
    parts.append(geom_xml("lid_pad1", "cylinder",
                          (LID_PAD_XY[0], LID_PAD_XY[1], LID_HZ),
                          (LID_PAD_R, LID_PAD_H),
                          rgba="0.85 0.75 0.20 1", density=300.0, condim=4,
                          solref="0.004 1", indent=2))
    # slide-bolt receiver walls on the lid top (channel for the bolt)
    for sy in (-1.0, 1.0):
        parts.append(geom_xml(
            f"lid_chan{int(sy)}", "box",
            (-0.010, sy * (BOLT_HY + 0.004), LID_HZ + BOLT_HZ),
            (0.045, 0.004, BOLT_HZ),
            rgba="0.15 0.45 0.65 1", density=300.0, condim=4,
            friction="0.3 0.005 0.0001", indent=2))
    parts.append(geom_xml("lid_chan_stop", "box",
                          (-0.055, 0.0, LID_HZ + BOLT_HZ),
                          (0.004, BOLT_HY, BOLT_HZ),
                          rgba="0.15 0.45 0.65 1", density=300.0, condim=4,
                          friction="0.3 0.005 0.0001", indent=2))
    parts.append("  </body>")

    # slide bolt (the lid latch bar)
    parts.append(body_open("latchbar", PART_STARTS["latchbar"]))
    parts.append(geom_xml("latchbar_body", "box", (0, 0, 0),
                          (BOLT_HX, BOLT_HY, BOLT_HZ),
                          rgba="0.55 0.55 0.60 1", density=800.0, condim=4,
                          friction="0.3 0.005 0.0001", indent=2))
    parts.append("  </body>")

    # continuity probe (taskB probe geometry)
    parts.append(body_open("probe", PART_STARTS["probe"]))
    parts.append(geom_xml("probe_main", "cylinder", (0, 0, 0.009),
                          (PROBE_MAIN_R, PROBE_MAIN_H / 2),
                          rgba="0.20 0.20 0.25 1", density=2700.0, condim=4,
                          solref="0.004 1", indent=2))
    parts.append(geom_xml("probe_mid", "cylinder", (0, 0, -0.0165),
                          (PROBE_MID_R, PROBE_MID_H / 2),
                          rgba="0.20 0.20 0.25 1", density=2700.0, condim=4,
                          friction="0.3 0.005 0.0001", solref="0.004 1",
                          indent=2))
    parts.append(geom_xml("probe_tip", "cylinder", (0, 0, -0.0255),
                          (PROBE_TIP_R, PROBE_TIP_H / 2),
                          rgba="0.20 0.20 0.25 1", density=2700.0, condim=4,
                          friction="0.3 0.005 0.0001", solref="0.004 1",
                          indent=2))
    parts.append("  </body>")

    xml = f"""<mujoco model="taskB_rack">
  <!-- Task B: 8-stage rack-server module assembly (simbench scene 1).
       Generated by gen_sceneB.py -- edit the generator, not this file.
       Physics: dt=0.002 + impratio 20 + elliptic cone + soft 0.004-1
       solrefs on the contact-critical geoms (the taskA recipe). -->
  <compiler angle="radian" meshdir="../assets/panda"
            inertiagrouprange="0 0" autolimits="true"/>
  <option timestep="0.002" impratio="20" cone="elliptic"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.4 0.4 0.4" specular="0.2 0.2 0.2"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <map znear="0.001"/>
    <quality shadowsize="2048" offsamples="4"/>
    <global offwidth="1280" offheight="720"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0"
             width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"
             width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
              texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <include file="../assets/panda/panda.xml"/>
  <custom>
{chr(10).join(custom)}
  </custom>
  <worldbody>
{chr(10).join(world)}
{chr(10).join(parts)}
  </worldbody>
</mujoco>
"""
    return xml


if __name__ == "__main__":
    main()
