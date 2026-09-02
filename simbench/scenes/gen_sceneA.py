#!/usr/bin/env python3
"""Generate the Task A (gearbox) scene XML for simbench.

Transplants the GearboxArena + gearbox part geometry into a standalone
scene file (table, fixture tray, pin sleeve, 7 parts, nominal-target
sites, cameras).  Re-run this script after editing the constants below;
the emitted taskA_gearbox.xml is the committed scene artifact.

Scene variants (taskA_gearbox_s2.xml, ...) can later copy this generator
with different part starts / tray position / clearances.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))  # repo root (/home/jia/RAL)

from simbench.scenes.parts import (  # noqa: E402
    BEARING, BOSS_FRICTION, COVER, GEAR, HOUSING,
    PIN, RETAINER, SHAFT, SPACER, STUD,
    boss_geoms, fmt, geom_xml, hex_geoms, ring_geoms, site_xml, v3,
)

TABLE_TOP_Z = 0.8
TRAY_XY = (0.15, 0.0)
TRAY_INNER = 0.033           # half inner width of the tray cavity
TRAY_WALL_H = 0.008
TRAY_WALL_T = 0.006
SLEEVE_XY = (-0.28, -0.28)
SLEEVE_R_OUT = 0.008
SLEEVE_R_HOLE = 0.0055
SLEEVE_H = 0.030

# (the stud-nut finale was retired 2026-09-01 -- see the RETAINER
# comment in parts.py; the studs stay on the housing as visual detail)

# nominal part starts (same as gearbox_env.part_start_xy + root z).
# All seated-on-table starts carry +0.5mm: a zero-gap start (bottom
# exactly at the tabletop) makes the contact solver eject the part on
# step 1 (measured: pin spun 0.28 rad/s, housing toppled during its
# grasp) -- +0.5mm settles them down 1-2 steps later with no bounce.
PART_STARTS = {
    "housing": (-0.18, -0.14, 0.8230),
    "shaft":   (-0.18, 0.14, 0.8470),
    "gear":    (-0.02, -0.24, 0.8065),
    "spacer":  (-0.02, -0.14, 0.8065),
    "bearing": (-0.02, -0.04, 0.8065),
    "cover":   (-0.18, -0.28, 0.8055),
    "pin":     (-0.28, -0.28, 0.8270),
    "retainer": (0.05, -0.24, 0.8045),
    "test_cube": (-0.02, 0.24, 0.8155),
}

# grasp metadata per part, exposed to the skill library as <custom>
# numerics ``meta_<part>`` = [type_code, grasp_dz, outer_d, half_h].
# type_code: 0 cylinder, 1 ring, 2 box, 3 pin.  Values transplanted from
# assembly/gearbox_skills.py GRASP_DZ / RING_OUTER_D / PART_HALF_H.
#
# 2026-09-01 thin-part dz calibration (measured): the pads sit
# eef+3.2mm at the centre, are 16mm tall, and tilt ~6.2deg with the
# wrist -- so the pad LOW CORNER hangs at eef-5.7mm.  Each part picks
# the dz that clears the table (low corner > 1.5mm) while still
# wrapping its side wall: rings (12mm tall, dz=+2mm -> low corner
# 2.1mm off the table, 10mm of wall contact; dz=0 put the corner ON
# the table and the 6.6N reaction froze the close at full span,
# measured), cover (10mm, dz=+3mm), retainer (8mm, dz=+3mm).
PART_META = {
    "housing": (0, 0.015, 0.052, 0.0225),
    "shaft":   (0, 0.035, 0.013, 0.073),
    "gear":    (1, 0.002, 0.020, 0.006),
    "spacer":  (1, 0.002, 0.036, 0.006),
    "bearing": (1, 0.002, 0.044, 0.006),
    "cover":   (1, 0.003, 0.052, 0.005),
    "pin":     (3, 0.020, 0.008, 0.026),
    "retainer": (1, 0.003, 0.034, 0.004),
    "test_cube": (2, 0.0, 0.030, 0.015),
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
                        "taskA_gearbox.xml")
    with open(path, "w") as f:
        f.write(out)
    print(f"wrote {path} ({len(out)} bytes)")


def build():
    H = HOUSING
    S = SHAFT
    C = COVER
    R = RETAINER
    tz = TABLE_TOP_Z
    tray_x, tray_y = TRAY_XY
    L, W, Hh = TRAY_INNER, TRAY_WALL_T, TRAY_WALL_H
    slab_h = 0.004

    # ---- static world ------------------------------------------------
    # Static geometry uses contype=0 conaffinity=1: under MuJoCo's OR
    # pairing rule the gripper finger collision meshes (robosuite
    # contype=0 conaffinity=1, they extend 4.8mm BELOW the pads) no
    # longer collide with the table/tray/sleeve, while parts, pads, arm
    # links (all default 1/1) still collide with everything.  The old
    # robosuite OSC torque controller pushed through this contact
    # penetration; the bounded position servo here would honestly stall.
    STATIC_CT = dict(contype="0", conaffinity="1")
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
    # table: top surface at z = 0.8; enlarged so the panda base at
    # x=-0.56 sits ON the table (x[-0.78 0.44] y[-0.4 0.4]).  The base
    # rests directly on the tabletop (no plinth) -- see panda.xml.
    world.append(geom_xml("table_top", "box", (-0.17, 0.0, tz - 0.025),
                          (0.61, 0.4, 0.025), rgba="0.45 0.42 0.38 1",
                          friction="1 0.005 0.0001", **STATIC_CT))
    for i, (lx, ly) in enumerate(((-0.70, -0.32), (0.36, -0.32),
                                  (-0.70, 0.32), (0.36, 0.32))):
        world.append(geom_xml(f"table_leg{i + 1}", "box", (lx, ly, 0.3875),
                              (0.03, 0.03, 0.3875), rgba="0.35 0.32 0.29 1",
                              **STATIC_CT))

    # fixture tray: bottom slab + 4 low walls (does not constrain yaw)
    world.append('  <body name="fixture_tray" '
                 f'pos="{v3(tray_x, tray_y, 0)}">')
    world.append(geom_xml("tray_slab", "box", (0.0, 0.0, tz + slab_h / 2),
                          (L + W, L + W, slab_h / 2),
                          rgba="0.25 0.27 0.31 1",
                          friction="1 0.005 0.0001", indent=4, **STATIC_CT))
    z_wall = tz + slab_h + Hh / 2
    for sx in (-1.0, 1.0):
        world.append(geom_xml(f"tray_wall_x{int(sx)}", "box",
                              (sx * (L + W / 2), 0.0, z_wall),
                              (W / 2, L + W, Hh / 2),
                              rgba="0.25 0.27 0.31 1",
                              friction="1 0.005 0.0001", indent=4,
                              **STATIC_CT))
    for sy in (-1.0, 1.0):
        world.append(geom_xml(f"tray_wall_y{int(sy)}", "box",
                              (0.0, sy * (L + W / 2), z_wall),
                              (L + W, W / 2, Hh / 2),
                              rgba="0.25 0.27 0.31 1",
                              friction="1 0.005 0.0001", indent=4,
                              **STATIC_CT))
    world.append("  </body>")

    # pin sleeve: open wall ring on the tabletop
    import numpy as np
    n = 8
    r_out, r_hole, sh = SLEEVE_R_OUT, SLEEVE_R_HOLE, SLEEVE_H
    unit_w = r_out * np.sin(np.pi / n)
    unit_h = (r_out - r_hole) * np.cos(np.pi / n) / 2.0
    int_r = r_hole * np.cos(np.pi / n) + unit_h
    world.append('  <body name="pin_sleeve" '
                 f'pos="{v3(SLEEVE_XY[0], SLEEVE_XY[1], 0)}">')
    for i in range(n):
        ang = 2.0 * np.pi * i / n
        world.append(geom_xml(
            f"pin_sleeve_w{i}", "box",
            (int_r * np.cos(ang), int_r * np.sin(ang), tz + sh / 2),
            (unit_h, unit_w, sh / 2),
            quat=f"{np.cos(ang/2):.6f} 0 0 {np.sin(ang/2):.6f}",
            rgba="0.55 0.50 0.45 1", friction="1 0.005 0.0001", indent=4,
            **STATIC_CT))
    world.append("  </body>")

    # nut feed seats: low ring walls (a bare 4mm nut topples when the
    # pads approach -- same trick as the pin sleeve)
    for k, (nx, ny) in enumerate(()):
        world.append(f'  <body name="nut_seat{k}" '
                     f'pos="{v3(nx, ny, 0)}">')
        for i in range(8):
            ang = 2.0 * np.pi * i / 8
            r_out, r_hole, sh = SEAT_R_OUT, SEAT_R_HOLE, SEAT_H
            unit_w = r_out * np.sin(np.pi / 8)
            unit_h = (r_out - r_hole) * np.cos(np.pi / 8) / 2.0
            int_r = r_hole * np.cos(np.pi / 8) + unit_h
            world.append(geom_xml(
                f"nut_seat{k}_w{i}", "box",
                (int_r * np.cos(ang), int_r * np.sin(ang), tz + sh / 2),
                (unit_h, unit_w, sh / 2),
                quat=f"{np.cos(ang/2):.6f} 0 0 {np.sin(ang/2):.6f}",
                rgba="0.55 0.50 0.45 1", friction="1 0.005 0.0001",
                indent=4, **STATIC_CT))
        world.append("  </body>")
    world.append(site_xml("tray_center", (tray_x, tray_y, tz + slab_h)))
    world.append(site_xml("boss_axis",
                          (tray_x + H["boss_ring_r"], tray_y, tz + slab_h)))
    world.append(site_xml("pin_start", (SLEEVE_XY[0], SLEEVE_XY[1], tz)))

    # cameras (robosuite table_arena definitions -- upright with the
    # mujoco.Renderer readback path)
    world.append('  <camera mode="fixed" name="frontview" pos="1.6 0 1.45" '
                 'quat="0.56 0.43 0.43 0.56"/>')
    world.append('  <camera mode="fixed" name="agentview" pos="0.5 0 1.35" '
                 'quat="0.653 0.271 0.271 0.653"/>')
    world.append('  <camera mode="fixed" name="sideview" '
                 'pos="-0.0565 1.2761 1.488" '
                 'quat="0.0099 0.0069 0.5912 0.8064"/>')

    # ---- parts ---------------------------------------------------------
    parts = []

    custom = []
    for pname, (tcode, dz, od, hh) in PART_META.items():
        custom.append(
            f'    <numeric name="meta_{pname}" '
            f'data="{tcode} {fmt(dz)} {fmt(od)} {fmt(hh)}"/>')

    # housing: ring + boss on the wall top (+x, blind hole)
    parts.append(body_open("housing", PART_STARTS["housing"]))
    parts.append(ring_geoms("housing", H["outer_r"], H["hole_r"],
                            H["height"], H["n"], H["rgba"], H["density"],
                            indent=2))
    top = H["height"] / 2.0
    # boss wall segments sit on the ring top (blind hole 8mm deep).
    # Stiff contacts (timeconst 2*dt): the pin release drop (~5mm, 0.3m/s)
    # punches through the default-soft floor (penetration ~v*0.02 = 6mm,
    # measured -3.3mm that never recovered) -- the bottom then slips
    # BELOW the hole and the pin tips over through the housing wall.
    # These geoms only ever touch the latch pin, so nothing else changes.
    import numpy as np
    for i in range(8):
        ang = 2.0 * np.pi * i / 8
        r_out_b, r_hole_b = H["boss_outer"], H["boss_r"]
        unit_w = r_out_b * np.sin(np.pi / 8)
        unit_h = (r_out_b - r_hole_b) * np.cos(np.pi / 8) / 2.0
        int_rb = r_hole_b * np.cos(np.pi / 8) + unit_h
        z_b = H["boss_depth"] / 2.0 + (top - 0.001)
        parts.append(geom_xml(
            f"housing_boss{i}", "box",
            (H["boss_ring_r"] + int_rb * np.cos(ang),
             int_rb * np.sin(ang), z_b),
            (unit_h, unit_w, H["boss_depth"] / 2.0),
            quat=f"{np.cos(ang/2):.6f} 0 0 {np.sin(ang/2):.6f}",
            rgba=H["rgba"], density=H["density"], friction=BOSS_FRICTION,
            condim=4, solref="0.01 1", indent=2))
    parts.append(geom_xml("housing_floor", "box",
                          (H["boss_ring_r"], 0.0, top - 0.0005),
                          (H["boss_outer"], H["boss_outer"], 0.001),
                          rgba=H["rgba"], density=H["density"], condim=4,
                          # 2026-09-01: floor solref softened to 20ms.
                          # At the old 4ms the thread tail's forced seat
                          # pushes the pin 0.8-1.6mm INTO the stiff floor;
                          # the stored normal load then ejects the tilted
                          # pin sideways out of the 8mm boss (measured:
                          # flung off the table).  20ms caps the penetration
                          # at ~0.3mm with the same gentle 0.5mm release
                          # drop -- no punch-through (that failure needed
                          # the default-soft 30ms+ contacts).
                          solref="0.02 1", indent=2))
    # stud bolts: rise from the housing top face at local +/-y, through
    # the cover notches -- visual classic stud-bolt joint (the nut
    # finale was retired; see parts.RETAINER)
    stud_z0 = top - 0.001
    stud_z1 = stud_z0 + STUD["clearance"]
    for k, sy in enumerate((-1.0, 1.0)):
        parts.append(geom_xml(
            f"housing_stud{k}", "cylinder",
            (0.0, sy * STUD["ring_r"], (stud_z0 + stud_z1) / 2.0),
            (STUD["r"], (stud_z1 - stud_z0) / 2.0),
            rgba=STUD["rgba"], density=H["density"], condim=4,
            friction=BOSS_FRICTION, indent=2))
    parts.append("  </body>")

    # shaft: three coaxial cylinders
    parts.append(body_open("shaft", PART_STARTS["shaft"]))
    parts.append(geom_xml("shaft_lower", "cylinder", (0, 0, -0.0245),
                          (S["lower_r"], S["lower_h"] / 2),
                          rgba=S["rgba"], density=S["density"], condim=4,
                          indent=2))
    parts.append(geom_xml("shaft_collar", "cylinder", (0, 0, 0.0),
                          (S["collar_r"], S["collar_h"] / 2),
                          rgba=S["rgba"], density=S["density"], condim=4,
                          indent=2))
    parts.append(geom_xml("shaft_upper", "cylinder", (0, 0, 0.027),
                          (S["upper_r"], S["upper_h"] / 2),
                          rgba=S["rgba"], density=S["density"], condim=4,
                          indent=2))
    # lead-in dome at the shaft tip (2026-09-02): the flat end caught
    # the gear's hole edge EVERY run -- the thread insert burned all 10
    # anti-jam wiggles walking it over the edge (the visible
    # press-and-shake at step 28).  The r6.5 dome gives the gear and
    # retainer holes a centering lead-in (MuJoCo 2.3 has no cone geom)
    parts.append(geom_xml("shaft_tip", "sphere", (0, 0, 0.057),
                          (S["upper_r"],),
                          rgba=S["rgba"], density=S["density"], condim=4,
                          indent=2))
    parts.append("  </body>")

    # rings: gear / spacer / bearing
    # friction 0.6 (2026-09-01): the ring parts now use a realistic
    # 0.6 sliding friction instead of the MuJoCo default 1.0, so a
    # light grasp + a fast stop can really slip (the grasp-slip fault
    # source is physically visible); the proven thread recipes still
    # hold at this friction (verified on the 'none' profile baseline)
    for name, cfg in (("gear", GEAR), ("spacer", SPACER),
                      ("bearing", BEARING)):
        parts.append(body_open(name, PART_STARTS[name]))
        parts.append(ring_geoms(name, cfg["outer_r"], cfg["hole_r"],
                                cfg["height"], cfg["n"], cfg["rgba"],
                                cfg["density"],
                                friction="0.6 0.005 0.0001",
                                drop_segs=cfg.get("drop_segs", ()),
                                indent=2))
        parts.append("  </body>")

    # cover: thin ring (segs cut for the stud notches +/-y and the old
    # latch channel +x) + through boss that captures the latch-pin top.
    # The inspection-latch channel rails were removed with the latch
    # stage (2026-09-01): orphaned rails caught the diagonal grasp pads
    # and let the cover slip out of the gripper (measured).
    parts.append(body_open("cover", PART_STARTS["cover"]))
    parts.append(ring_geoms("cover", C["outer_r"], C["hole_r"],
                            C["height"], C["n"], C["rgba"], C["density"],
                            drop_segs=C.get("drop_segs", (0,)), indent=2))
    for i in range(8):
        ang = 2.0 * np.pi * i / 8
        r_out_b, r_hole_b = C["boss_outer"], C["boss_r"]
        unit_w = r_out_b * np.sin(np.pi / 8)
        unit_h = (r_out_b - r_hole_b) * np.cos(np.pi / 8) / 2.0
        int_rb = r_hole_b * np.cos(np.pi / 8) + unit_h
        parts.append(geom_xml(
            f"cover_boss{i}", "box",
            (C["boss_ring_r"] + int_rb * np.cos(ang),
             int_rb * np.sin(ang), C["height"] / 2.0),
            (unit_h, unit_w, C["height"] / 2.0),
            quat=f"{np.cos(ang/2):.6f} 0 0 {np.sin(ang/2):.6f}",
            rgba=C["rgba"], density=C["density"], friction=BOSS_FRICTION,
            condim=4, indent=2))
    parts.append("  </body>")

# latch pin: dense rigid cylinder, low-slip
    # (start z +1mm: at 0.826 the pin bottom sat exactly ON the sleeve
    # floor = zero-gap initial penetration; the contact solver kept
    # bouncing the pin, spinning it at 0.28 rad/s for the whole initial
    # settle -- the root of the run-to-run nondeterminism, measured)
    parts.append(body_open("pin", PART_STARTS["pin"]))
    # latch pin: dense rigid cylinder.  friction 0.5 (not the 0.15
    # boss recipe): the low-slip wall let the seated pin micro-slide
    # and its stiff contacts intermittently produced QACC NaN in the
    # boss bore (measured on 2026-09-01; the 0.5 wall friction + the
    # softened wall solref below quiet it)
    parts.append(geom_xml("pin_cyl", "cylinder", (0, 0, 0),
                          (PIN["r"], PIN["length"] / 2),
                          rgba=PIN["rgba"], density=PIN["density"],
                          friction="0.5 0.005 0.0001", condim=4, indent=2))
    # two press-fit collars: r4.0 over the 3.9mm bore = 0.1mm
    # interference that wedges the collars into the boss mouth once
    # the pin seats -- a mechanical lock against vibration ejections
    # (measured: every earlier pin variant was eventually flung out)
    for k, zc in enumerate((-0.006, 0.006)):
        parts.append(geom_xml(
            f"pin_collar{k}", "cylinder", (0, 0, zc),
            (0.004, 0.0012), rgba=PIN["rgba"], density=PIN["density"],
            friction="0.5 0.005 0.0001", condim=4, indent=2))
    parts.append("  </body>")

    # bearing retainer ring (S8): threads onto the shaft upper end and
    # seats on the cover top -- the stable thread-insert finale that
    # replaced the retired stud-nut screw_drive (see parts.RETAINER)
    parts.append(body_open("retainer", PART_STARTS["retainer"]))
    parts.append(ring_geoms("retainer", R["outer_r"], R["hole_r"],
                            R["height"], R["n"], R["rgba"], R["density"],
                            friction="0.6 0.005 0.0001",
                            indent=2))
    parts.append("  </body>")

    # skill-library test cube (30 mm, off the main work area)
    parts.append(body_open("test_cube", PART_STARTS["test_cube"]))
    parts.append(geom_xml("test_cube_box", "box", (0, 0, 0),
                          (0.015, 0.015, 0.015), rgba="0.30 0.60 0.80 1",
                          density=400.0, condim=4, indent=2))
    parts.append("  </body>")

    xml = f"""<mujoco model="taskA_gearbox">
  <!-- Task A: 8-stage gearbox assembly chain (simbench scene 1).
       Generated by gen_sceneA.py -- edit the generator, not this file. -->
  <compiler angle="radian" meshdir="../assets/panda"
            inertiagrouprange="0 0" autolimits="true"/>
  <option timestep="0.002" impratio="20" cone="elliptic"/>
  <size nconmax="800"/>
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
