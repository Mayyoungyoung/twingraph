#!/usr/bin/env python3
"""Generate the Task C (precision-fixture) scene XML for simbench.

Transplants FixtureArena + fixture_objects into a standalone scene file:
table, fixture cavity (base slab + 4 low walls), dual locator pins, side
clamp, top clamp (all slide-jointed with their own position actuators),
workpiece pedestal, probe sleeve, the SIMPLE-GEOMETRY workpiece (12 box
segments: 2 square locator through-holes + 1 square datum blind hole +
op_point site) and the cone-tipped probe.

Physics (2026-08-29 simplified recipe): the original voxel-grid
workpiece (2517 cells / ~10k contacts) cost ~25ms per physics step at
dt=0.0005 with -3000/-1200 solref pairs -- 65x slower than Task A.  The
12-box workpiece needs none of that: dt=0.002, impratio 20, elliptic
cone and soft 0.004-1 solrefs everywhere (identical to Task A, ~0.4ms
per step).

Re-run this script after editing the constants below; the emitted
taskC_fixture.xml is the committed scene artifact.

World-z references (table_top_z = 0.8):
  cavity floor 0.806  workpiece seated root 0.811  top 0.816
  wall top 0.812      pin root 0.797 (rises 11mm)  clamp root 0.842
  pedestal top 0.815  workpiece start root 0.82    probe start root 0.83
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))  # repo root (/home/jia/RAL)

import numpy as np  # noqa: E402

# ------------------------------------------------------------- constants
# (transplanted verbatim from poc_taskc_fixture.py / fixture_arena.py /
#  fixture_objects.py -- see those files for the measurement comments)
TABLE_TOP_Z = 0.8
FIXTURE_XY = (0.15, 0.0)
PEDESTAL_XY = (-0.18, -0.14)
PEDESTAL_H = 0.015
SLEEVE_XY = (-0.28, -0.28)
SLEEVE_R_HOLE = 0.006
SLEEVE_R_OUT = 0.010
SLEEVE_H = 0.030

# workpiece: 70x40x10 simple-geometry plate
WP_HX, WP_HY, WP_HZ = 0.035, 0.020, 0.005
HOLE_A = (-0.025, 0.0)
HOLE_B = (0.025, 0.0)
HOLE_HW = 0.0043         # square locator hole half width
DATUM_XY = (0.012, 0.0)  # moved off HOLE_B (POC: 0.022 overlaps it)
DATUM_HW = 0.005         # square datum hole half width (demo: 10mm
                         # opening; the 6mm POC hole + 0.8mm tip +
                         # ~0.5mm live placement error clips the hole
                         # wall top edge and the tip slides off the
                         # plate face -- measured med=2.7mm detected=0)
DATUM_DEPTH = 0.002
OP_NOM = (0.008, 0.009)

# fixture cavity / pins / clamps (FixtureArena)
CAV_HX = WP_HX + 0.004          # 0.039
CAV_HY = WP_HY + 0.003          # 0.023
WALL_H = 0.006
WALL_T = 0.006
SLAB_H = 0.006
PIN_R = 0.0038
PIN_LIFT = 0.011
# top clamp: 20mm-wide bar pressing ONLY the plate -x edge strip
# (x[0.115,0.125] when seated).  The POC's 28mm bar overhung the
# cavity interior (x[0.118,0.146]) and blocked the robot drop path --
# a vertically falling plate always intersects it, so the plate landed
# ON the clamp and leaned 58deg (measured 2026-08-29).  The bar is
# also retracted ABOVE the drop path at rest and descends only in S6.
#
# NOTE the rest state is PHYSICAL, not joint-ref: MuJoCo resets ctrl=0
# for position servos, so ref=+0.028 left the servo dragging the bar
# down at 12N from t=0 (measured: bar pinned the cavity at qmin).
# Instead CLAMP_ROOT_Z is raised 0.842->0.870 and the joint range
# [-0.056, 0] keeps qpos=0=ctrl=0 force-free at rest (bottom 0.867 vs
# plate top 0.856 during the drop).
CLAMP_SIZE = (0.010, 0.007, 0.003)
CLAMP_XY = (-0.035, 0.0)        # world x 0.115 = plate -x edge
CLAMP_QMAX = 0.0                # rest: raised clear of the drop path
CLAMP_QMIN = -0.056
SIDE_SIZE = (0.004, 0.006, 0.002)
SIDE_Z = 0.8155
# side push travel: the +x edge starts ~3.4mm from the side_g face
# (plate lands at x 0.149-0.150, face 0.1841, side face at 0.1875);
# the 4mm POC travel only embeds the face 0.6mm (0.28N << the 0.37N
# floor friction, plate never moved).  8mm travel pushes the plate
# against the -x wall (3.1mm) and keeps ~1.5mm of servo penetration
# (~0.5N contact).
SIDE_RANGE = (-0.008, 0.0)
SIDE_Q0_X = 0.0415

# probe (60mm total, root = center, tip bottom at -0.030)
PROBE_MAIN_R = 0.0025
PROBE_MID_R = 0.0015
PROBE_TIP_R = 0.0008
PROBE_MAIN_H = 0.042       # z in [-0.012, +0.030]
PROBE_MID_H = 0.009        # z in [-0.021, -0.012]
PROBE_TIP_H = 0.009        # z in [-0.030, -0.021]

# derived world-z references (FixtureArena properties)
CAVITY_FLOOR_Z = TABLE_TOP_Z + SLAB_H             # 0.806
WP_SEATED_ROOT_Z = CAVITY_FLOOR_Z + WP_HZ         # 0.811
WP_TOP_Z = WP_SEATED_ROOT_Z + WP_HZ               # 0.816
PIN_ROOT_Z = 0.797
CLAMP_ROOT_Z = 0.870
WP_START_ROOT_Z = TABLE_TOP_Z + PEDESTAL_H + WP_HZ    # 0.82
PROBE_START_ROOT_Z = TABLE_TOP_Z + 0.030              # 0.83

# grasp metadata exposed as <custom> numerics (taskA convention):
# [type_code, grasp_dz, outer_d, half_h]; workpiece gripped across its
# 40mm breadth, probe = pin-style rod grip
PART_META = {
    "workpiece": (2, 0.002, 0.040, 0.005),
    "probe": (3, 0.020, 0.005, 0.030),
}


# ---------------------------------------------------- workpiece (simple geometry)
def build_workpiece_geoms():
    """12-box plate replacing the 2517-cell voxel grid.

    Layout (local frame, root = plate centre, half height 5mm):
      x[-35,-29.3] and x[29.3,35]      full-width end blocks
      x[+-25] +-4.3mm square locator through-holes (front/back strips)
      x[12] +-3mm square datum blind hole (front/back strips + floor)
      x[-20.7,9] and x[15,20.7]        full-width centre blocks

    The locator pins (d=7.6mm) clear the 8.6mm square holes by 0.5mm
    per side -- same clearance as the original r=4.3mm circles.  The
    datum floor sits 2mm below the plate top, so the 1.6mm probe tip
    bottoms out at wp_top-2mm (S7 detection threshold: wp_top-0.5mm).

    All geoms share the workpiece body, so same-body geom pairs are
    never checked -- the 12 boxes act as one rigid plate.
    """
    HZ = WP_HZ
    geoms = [
        # end blocks
        dict(pos=(-0.03215, 0.0, 0.0), size=(0.00285, 0.020, HZ)),
        dict(pos=(0.03215, 0.0, 0.0), size=(0.00285, 0.020, HZ)),
        # locator-hole strips (front/back) for holes A (-25) and B (+25)
        dict(pos=(-0.025, 0.01215, 0.0), size=(HOLE_HW, 0.00785, HZ)),
        dict(pos=(-0.025, -0.01215, 0.0), size=(HOLE_HW, 0.00785, HZ)),
        dict(pos=(0.025, 0.01215, 0.0), size=(HOLE_HW, 0.00785, HZ)),
        dict(pos=(0.025, -0.01215, 0.0), size=(HOLE_HW, 0.00785, HZ)),
        # datum-hole strips (front/back) + sunken datum floor; the
        # strip outer edge sits flush with the plate face y=0.020
        # (a protruding strip catches the closing pads -- measured:
        # span stuck at 51.9mm and the plate never lifted)
        dict(pos=(0.012, DATUM_HW + (0.020 - DATUM_HW) / 2.0, 0.0),
             size=(DATUM_HW, (0.020 - DATUM_HW) / 2.0, HZ)),
        dict(pos=(0.012, -(DATUM_HW + (0.020 - DATUM_HW) / 2.0), 0.0),
             size=(DATUM_HW, (0.020 - DATUM_HW) / 2.0, HZ)),
    ]
    # datum: 4 tilted boxes form a conical funnel floor.  The probe
    # tip arrives with a ~7deg eef tilt (the probe is a 60mm rod), so
    # a flat blind hole leaves the tip wedged on the +x wall
    # (measured med=3.4mm); the funnel slides the tip to the HOLE
    # CENTRE instead, so med reads the true placement error (research
    # semantic kept).
    ramp_len = float(np.hypot(DATUM_HW, DATUM_DEPTH) / 2.0)  # 0.0054
    th = float(np.arctan2(DATUM_DEPTH, DATUM_HW))
    cq, sq = float(np.cos(th / 2.0)), float(np.sin(th / 2.0))
    off = DATUM_HW / 2.0
    geoms += [
        dict(pos=(0.012 + off, 0.0, -DATUM_DEPTH / 2.0),
             size=(ramp_len, DATUM_HW, 0.0005),
             quat=(cq, 0.0, -sq, 0.0)),
        dict(pos=(0.012 - off, 0.0, -DATUM_DEPTH / 2.0),
             size=(ramp_len, DATUM_HW, 0.0005),
             quat=(cq, 0.0, sq, 0.0)),
        dict(pos=(0.012, off, -DATUM_DEPTH / 2.0),
             size=(DATUM_HW, ramp_len, 0.0005),
             quat=(cq, sq, 0.0, 0.0)),
        dict(pos=(0.012, -off, -DATUM_DEPTH / 2.0),
             size=(DATUM_HW, ramp_len, 0.0005),
             quat=(cq, -sq, 0.0, 0.0)),
    ]
    geoms += [
        # centre blocks either side of the datum hole (span to the
        # 2*DATUM_HW opening edges)
        dict(pos=((0.012 - DATUM_HW - 0.0207) / 2.0, 0.0, 0.0),
             size=((0.012 - DATUM_HW + 0.0207) / 2.0, 0.020, HZ)),
        dict(pos=(0.012 + DATUM_HW + 0.00185, 0.0, 0.0),
             size=(0.00185, 0.020, HZ)),
    ]
    return geoms


def fmt(x):
    return f"{float(x):.6f}".rstrip("0").rstrip(".")


def v3(x, y, z):
    return f"{fmt(x)} {fmt(y)} {fmt(z)}"


# ------------------------------------------------------------- build
def build():
    tz = TABLE_TOP_Z
    fx, fy = FIXTURE_XY

    # static geometry: contype=0 conaffinity=1 (taskA STATIC_CT).  The
    # collision pairings that matter (MuJoCo OR rule):
    #   plate boxes (1,3) vs slab/walls/pedestal (0,1) -> collide (bit 0)
    #   pins (2,2) vs slab (0,1)                 -> pass through (POC
    #     used conaffinity 0 on the slab for the same effect)
    #   pads (1,1) vs static (0,1)               -> collide
    STATIC = 'contype="0" conaffinity="1"'

    world = []
    world.append('  <light pos="1.0 1.0 1.5" dir="-0.2 -0.2 -1" '
                 'directional="true" castshadow="true" '
                 'diffuse="0.9 0.9 0.9" specular="0.5 0.5 0.5"/>')
    world.append(f'  <geom name="floor" type="plane" pos="0 0 0" '
                 f'size="2.5 2.5 0.1" material="groundplane" '
                 f'friction="1 0.005 0.0001" {STATIC}/>')
    # table: top surface at z = 0.8; enlarged so the panda base at
    # x=-0.56 sits ON the table (x[-0.78 0.44] y[-0.4 0.4]).  The base
    # rests directly on the tabletop (no plinth) -- see panda.xml.
    world.append(f'  <geom name="table_top" type="box" '
                 f'pos="-0.17 0 {fmt(tz - 0.025)}" size="0.61 0.4 0.025" '
                 f'rgba="0.45 0.42 0.38 1" friction="1 0.005 0.0001" '
                 f'{STATIC}/>')
    for i, (lx, ly) in enumerate(((-0.70, -0.32), (0.36, -0.32),
                                  (-0.70, 0.32), (0.36, 0.32))):
        world.append(f'  <geom name="table_leg{i + 1}" type="box" '
                     f'pos="{v3(lx, ly, 0.3875)}" size="0.03 0.03 0.3875" '
                     f'rgba="0.35 0.32 0.29 1" {STATIC}/>')

    # ---- fixture: cavity slab + walls + pins + side clamp + top clamp
    fixture = [f'  <body name="fixture" pos="{v3(fx, fy, 0)}">']
    # base slab: pins pass through via the contact bit pairing above;
    # soft solref (taskA recipe) -- the old -1200 cell solref was for
    # the 0.1g voxel cells, meaningless on the 12-box plate
    fixture.append(f'    <geom name="cavity_slab" type="box" '
                   f'pos="0 0 {fmt(tz + SLAB_H / 2)}" '
                   f'size="0.050 0.035 {fmt(SLAB_H / 2)}" '
                   f'rgba="0.25 0.27 0.31 1" friction="0.5 0.005 0.0001" '
                   f'solref="0.004 1" {STATIC}/>')
    # 4 low walls (inner cavity 78x46)
    z_wall = CAVITY_FLOOR_Z + WALL_H / 2.0
    for sx in (-1.0, 1.0):
        fixture.append(f'    <geom name="wall_x{int(sx)}" type="box" '
                       f'pos="{v3(sx * (CAV_HX + WALL_T / 2), 0, z_wall)}" '
                       f'size="{v3(WALL_T / 2, CAV_HY + WALL_T, WALL_H / 2)}" '
                       f'rgba="0.25 0.27 0.31 1" friction="0.15 0.005 0.0001" '
                       f'{STATIC}/>')
    for sy in (-1.0, 1.0):
        fixture.append(f'    <geom name="wall_y{int(sy)}" type="box" '
                       f'pos="{v3(0, sy * (CAV_HY + WALL_T / 2), z_wall)}" '
                       f'size="{v3(CAV_HX + WALL_T, WALL_T / 2, WALL_H / 2)}" '
                       f'rgba="0.25 0.27 0.31 1" friction="0.15 0.005 0.0001" '
                       f'{STATIC}/>')
    # dual locator pins: chamfered tip, slide joints, contype 2 (hit the
    # workpiece cells, pass the base slab)
    for tag, hx in (("A", -0.025), ("B", 0.025)):
        fixture.append(f'    <body name="pin{tag}" '
                       f'pos="{v3(hx, 0.0, PIN_ROOT_Z)}">')
        fixture.append(f'      <joint name="pin{tag}_j" type="slide" '
                       f'axis="0 0 1" range="0 0.02" damping="0.3" '
                       f'frictionloss="0.02"/>')
        fixture.append(f'      <geom name="pin{tag}_cham" type="cylinder" '
                       f'size="{fmt(PIN_R * 0.6)} 0.0005" pos="0 0 0.008" '
                       f'friction="0.2 0.005 0.0001" contype="2" '
                       f'conaffinity="2" condim="4" density="1000" '
                       f'solref="0.004 1"/>')
        fixture.append(f'      <geom name="pin{tag}_g" type="cylinder" '
                       f'size="{fmt(PIN_R)} 0.0065" pos="0 0 0.001" '
                       f'friction="0.25 0.005 0.0001" condim="4" '
                       f'contype="2" conaffinity="2" solref="0.004 1" '
                       f'density="1000"/>')
        fixture.append('    </body>')
    # side clamp: pushes the +x edge toward the -x wall (horizontal);
    # rides 1.5mm above the wall top (a zero-gap wall contact jams it)
    fixture.append(f'    <body name="side_clamp" '
                   f'pos="{v3(SIDE_Q0_X, 0.0, SIDE_Z)}">')
    fixture.append(f'      <joint name="side_j" type="slide" axis="1 0 0" '
                   f'range="{fmt(SIDE_RANGE[0])} {fmt(SIDE_RANGE[1])}" '
                   f'damping="3.0" frictionloss="0.1"/>')
    fixture.append(f'      <geom name="side_g" type="box" pos="0 0 0" '
                   f'size="{v3(*SIDE_SIZE)}" rgba="0.70 0.40 0.20 1" '
                   f'friction="0.4 0.005 0.0001" condim="4" '
                   f'solref="0.004 1" density="2700"/>')
    fixture.append('    </body>')
    # top clamp: force-controlled press on the -x edge (20mm bar;
    # rest position raised physically so the plate drop path is clear;
    # qpos=0=ctrl=0 at rest = no servo force)
    fixture.append(f'    <body name="clamp" '
                   f'pos="{v3(CLAMP_XY[0], CLAMP_XY[1], CLAMP_ROOT_Z)}">')
    fixture.append(f'      <joint name="clamp_j" type="slide" axis="0 0 1" '
                   f'range="{fmt(CLAMP_QMIN)} {fmt(CLAMP_QMAX)}" '
                   f'damping="3.0" frictionloss="0.1"/>')
    fixture.append(f'      <geom name="clamp_g" type="box" pos="0 0 0" '
                   f'size="{v3(*CLAMP_SIZE)}" rgba="0.70 0.40 0.20 1" '
                   f'friction="0.4 0.005 0.0001" condim="4" '
                   f'solref="0.004 1" density="1000"/>')
    fixture.append('    </body>')
    fixture.append('  </body>')
    world.extend(fixture)

    # ---- workpiece pedestal: 80x42mm fully supports the 70x40 plate
    world.append(f'  <body name="pedestal" '
                 f'pos="{v3(PEDESTAL_XY[0], PEDESTAL_XY[1], 0)}">')
    world.append(f'    <geom name="pedestal_g" type="box" '
                 f'pos="0 0 {fmt(tz + PEDESTAL_H / 2)}" '
                 f'size="0.040 0.021 {fmt(PEDESTAL_H / 2)}" '
                 f'rgba="0.55 0.50 0.45 1" friction="1 0.005 0.0001" '
                 f'{STATIC}/>')
    world.append('  </body>')

    # ---- probe sleeve: 8-segment open wall ring (gearbox pin sleeve)
    n = 8
    unit_w = SLEEVE_R_OUT * np.sin(np.pi / n)
    unit_h = (SLEEVE_R_OUT - SLEEVE_R_HOLE) * np.cos(np.pi / n) / 2.0
    int_r = SLEEVE_R_HOLE * np.cos(np.pi / n) + unit_h
    world.append(f'  <body name="probe_sleeve" '
                 f'pos="{v3(SLEEVE_XY[0], SLEEVE_XY[1], 0)}">')
    for i in range(n):
        ang = 2.0 * np.pi * i / n
        world.append(f'    <geom name="probe_sleeve_w{i}" type="box" '
                     f'pos="{v3(int_r * np.cos(ang), int_r * np.sin(ang), tz + SLEEVE_H / 2)}" '
                     f'size="{v3(unit_h, unit_w, SLEEVE_H / 2)}" '
                     f'quat="{np.cos(ang / 2):.6f} 0 0 {np.sin(ang / 2):.6f}" '
                     f'rgba="0.55 0.50 0.45 1" friction="1 0.005 0.0001" '
                     f'{STATIC}/>')
    world.append('  </body>')

    # ---- reference sites (nominal coordinates, taskA convention)
    world.append(f'  <site name="cavity_center" pos="{v3(fx, fy, CAVITY_FLOOR_Z)}"'
                 f' size="0.003" rgba="1 0 0 0"/>')
    world.append(f'  <site name="datum_nominal" '
                 f'pos="{v3(fx + DATUM_XY[0], fy + DATUM_XY[1], WP_TOP_Z)}"'
                 f' size="0.003" rgba="1 0 0 0"/>')
    world.append(f'  <site name="op_nominal" '
                 f'pos="{v3(fx + OP_NOM[0], fy + OP_NOM[1], WP_TOP_Z)}"'
                 f' size="0.003" rgba="1 0 0 0"/>')
    world.append(f'  <site name="workpiece_start" '
                 f'pos="{v3(PEDESTAL_XY[0], PEDESTAL_XY[1], WP_START_ROOT_Z)}"'
                 f' size="0.003" rgba="1 0 0 0"/>')
    world.append(f'  <site name="probe_start" '
                 f'pos="{v3(SLEEVE_XY[0], SLEEVE_XY[1], tz)}"'
                 f' size="0.003" rgba="1 0 0 0"/>')

    world.append('  <camera mode="fixed" name="frontview" pos="1.6 0 1.45" '
                 'quat="0.56 0.43 0.43 0.56"/>')
    world.append('  <camera mode="fixed" name="agentview" pos="0.5 0 1.35" '
                 'quat="0.653 0.271 0.271 0.653"/>')
    world.append('  <camera mode="fixed" name="sideview" '
                 'pos="-0.0565 1.2761 1.488" '
                 'quat="0.0099 0.0069 0.5912 0.8064"/>')

    # ---- workpiece: 12-box simple-geometry plate + op_point site
    wp = [f'  <body name="workpiece" '
          f'pos="{v3(PEDESTAL_XY[0], PEDESTAL_XY[1], WP_START_ROOT_Z)}">',
          '    <freejoint name="workpiece_j"/>']
    n_cells = 0
    for k, g in enumerate(build_workpiece_geoms()):
        quat = g.get("quat")
        qs = f' quat="{v3(*quat[:3])} {fmt(quat[3])}"' if quat else ""
        wp.append(f'    <geom name="cell{k}" class="wp_box" '
                  f'pos="{v3(*g["pos"])}" size="{v3(*g["size"])}"{qs}/>')
        n_cells += 1
    # op_point site rides with the plate (live operation point for S9)
    wp.append(f'    <site name="op_point" pos="{v3(OP_NOM[0], OP_NOM[1], 0)}"'
              f' size="0.002" rgba="0 1 0 1"/>')
    wp.append('  </body>')

    # ---- probe: cone-tipped rod, low-friction mid/tip so the dimple
    # walls self-centre the tip (SLIDE_FRICTION mu=0.3 < tan(45deg))
    probe = [f'  <body name="probe" '
             f'pos="{v3(SLEEVE_XY[0], SLEEVE_XY[1], PROBE_START_ROOT_Z)}">',
             '    <freejoint name="probe_j"/>',
             f'    <geom name="probe_main" class="probe_rod" '
             f'pos="0 0 0.009" size="{fmt(PROBE_MAIN_R)} '
             f'{fmt(PROBE_MAIN_H / 2)}" friction="1 0.005 0.0001"/>',
             f'    <geom name="probe_mid" class="probe_rod" '
             f'pos="0 0 -0.0165" size="{fmt(PROBE_MID_R)} '
             f'{fmt(PROBE_MID_H / 2)}" friction="0.3 0.005 0.0001"/>',
             f'    <geom name="probe_tip" class="probe_rod" '
             f'pos="0 0 -0.0255" size="{fmt(PROBE_TIP_R)} '
             f'{fmt(PROBE_TIP_H / 2)}" friction="0.3 0.005 0.0001"/>',
             '  </body>']

    custom = []
    for pname, (tcode, dz, od, hh) in PART_META.items():
        custom.append(f'    <numeric name="meta_{pname}" '
                      f'data="{tcode} {fmt(dz)} {fmt(od)} {fmt(hh)}"/>')

    xml = f"""<mujoco model="taskC_fixture">
  <!-- Task C: 9-stage precision-fixture loading chain (simbench scene 1).
       Generated by gen_sceneC.py -- edit the generator, not this file.
       Physics (2026-08-29): dt=0.002 + impratio 20 + elliptic cone +
       soft 0.004-1 solrefs -- the taskA recipe, valid now that the
       workpiece is a 12-box plate instead of a ~2600-cell voxel grid. -->
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
  <default>
    <!-- workpiece plate boxes: kept the POC pairing contype=1
         conaffinity=3 (collide with the fixture base, walls, clamps
         and pins bit-1); same-body box pairs are never checked, so
         the 12 segments act as one rigid plate -->
    <default class="wp_box">
      <geom type="box" rgba="0.75 0.75 0.80 1" density="2700"
            friction="0.35 0.005 0.0001" condim="4" solref="0.004 1"
            contype="1" conaffinity="3"/>
    </default>
    <default class="probe_rod">
      <geom type="cylinder" rgba="0.20 0.20 0.25 1" density="2700"
            condim="4" solref="0.004 1" contype="1" conaffinity="1"/>
    </default>
  </default>
  <include file="../assets/panda/panda.xml"/>
  <custom>
{chr(10).join(custom)}
  </custom>
  <worldbody>
{chr(10).join(world)}
{chr(10).join(wp)}
{chr(10).join(probe)}
  </worldbody>
  <actuator>
    <!-- fixture executors (POC parameters): force limits below the
         0.74N plate weight on the pins so a misaligned pin COMPLIES -->
    <position name="pinA_act" joint="pinA_j" kp="250"
              ctrlrange="0 0.02" forcerange="-0.2 0.2"/>
    <position name="pinB_act" joint="pinB_j" kp="250"
              ctrlrange="0 0.02" forcerange="-0.2 0.2"/>
    <position name="side_act" joint="side_j" kp="300"
              ctrlrange="{fmt(SIDE_RANGE[0])} {fmt(SIDE_RANGE[1])}"
              forcerange="-2 2"/>
    <position name="clamp_act" joint="clamp_j" kp="600"
              ctrlrange="{fmt(CLAMP_QMIN)} {fmt(CLAMP_QMAX)}"
              forcerange="-12 12"/>
  </actuator>
</mujoco>
"""
    return xml, n_cells


def main():
    out, n_cells = build()
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "taskC_fixture.xml")
    with open(path, "w") as f:
        f.write(out)
    print(f"wrote {path} ({len(out)} bytes, {n_cells} plate geoms)")


if __name__ == "__main__":
    main()
