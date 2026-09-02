#!/usr/bin/env python3
"""Generate the minimal peg-in-hole scene used to TRAIN the insertion
policy (simbench.skills.learned).

A deliberately small scene -- one Panda, a blind bore (a base block + an
8-segment round sleeve wall) and a free-jointed cylindrical peg -- so RL
rollouts are cheap and resets are clean.  The peg is gripped and hung
above the bore at reset; the policy drives the EEF to insert it.  The
observation / action contract (``build_obs`` / ``action_to_delta``) is
scene-agnostic, so the SAME trained policy deploys through
``extension.peg_insert(mode='policy')`` in any scene that has a peg body
+ a hole xy + a seat z (e.g. the Task A latch pin / gearbox rings).

Follows the gen_sceneC.py conventions: ``<include>`` the Panda asset,
expose grasp metadata as ``<custom><numeric name="meta_peg" .../>``,
soft ``solref="0.004 1"`` contacts, dt=0.002, elliptic cone.

World-z references (table_top_z = 0.8):
  bore floor (seat) 0.808   sleeve mouth 0.818   peg start bottom 0.824
Re-run this script after editing the constants; the emitted
peg_in_hole.xml is the committed scene artifact.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))  # repo root

import numpy as np  # noqa: E402

# ------------------------------------------------------------- constants
TABLE_TOP_Z = 0.8
HOLE_XY = (0.15, 0.0)

# bore: a solid base block (its top is the blind-hole floor = the seat)
BASE_HX, BASE_HY, BASE_HZ = 0.028, 0.028, 0.008
BASE_TOP_Z = TABLE_TOP_Z + BASE_HZ                # 0.808 = seat (to_z)

# sleeve wall ring: round bore, inner r 6mm, outer r 10mm, 10mm tall
SLEEVE_R_HOLE = 0.006
SLEEVE_R_OUT = 0.010
SLEEVE_H = 0.010
SLEEVE_Z0 = BASE_TOP_Z                            # 0.808
SLEEVE_MOUTH_Z = SLEEVE_Z0 + SLEEVE_H             # 0.818

# peg: cylinder r 4mm, length 20mm (2mm radial clearance in the bore)
PEG_R = 0.004
PEG_HALF_H = 0.010
PEG_START_BOTTOM_Z = BASE_TOP_Z + 0.016           # 0.824 (16mm of travel)

# grasp metadata (taskA convention): [type_code, grasp_dz, outer_d, half_h]
PART_META = {"peg": (3, 0.0, 2.0 * PEG_R, PEG_HALF_H)}


def fmt(x):
    return f"{float(x):.6f}".rstrip("0").rstrip(".")


def v3(x, y, z):
    return f"{fmt(x)} {fmt(y)} {fmt(z)}"


def build():
    tz = TABLE_TOP_Z
    hx, hy = HOLE_XY
    STATIC = 'contype="0" conaffinity="1"'

    world = []
    world.append('  <light pos="1.0 1.0 1.5" dir="-0.2 -0.2 -1" '
                 'directional="true" castshadow="false" '
                 'diffuse="0.9 0.9 0.9" specular="0.5 0.5 0.5"/>')
    world.append(f'  <geom name="floor" type="plane" pos="0 0 0" '
                 f'size="2.5 2.5 0.1" material="groundplane" '
                 f'friction="1 0.005 0.0001" {STATIC}/>')
    # table: top at z=0.8, sized so the panda base (x=-0.56) rests on it
    world.append(f'  <geom name="table_top" type="box" '
                 f'pos="-0.17 0 {fmt(tz - 0.025)}" size="0.61 0.4 0.025" '
                 f'rgba="0.45 0.42 0.38 1" friction="1 0.005 0.0001" '
                 f'{STATIC}/>')

    # ---- bore: base block (blind-hole floor) + 8-segment round sleeve
    world.append(f'  <body name="hole_block" pos="{v3(hx, hy, 0)}">')
    world.append(f'    <geom name="hole_base" type="box" '
                 f'pos="0 0 {fmt(tz + BASE_HZ / 2)}" '
                 f'size="{v3(BASE_HX, BASE_HY, BASE_HZ / 2)}" '
                 f'rgba="0.30 0.34 0.40 1" friction="0.4 0.005 0.0001" '
                 f'condim="4" solref="0.004 1" contype="1" conaffinity="1"/>')
    n = 8
    unit_w = SLEEVE_R_OUT * np.sin(np.pi / n)
    unit_h = (SLEEVE_R_OUT - SLEEVE_R_HOLE) * np.cos(np.pi / n) / 2.0
    int_r = SLEEVE_R_HOLE * np.cos(np.pi / n) + unit_h
    zc = SLEEVE_Z0 + SLEEVE_H / 2.0
    for i in range(n):
        ang = 2.0 * np.pi * i / n
        world.append(
            f'    <geom name="hole_w{i}" type="box" '
            f'pos="{v3(int_r * np.cos(ang), int_r * np.sin(ang), zc)}" '
            f'size="{v3(unit_h, unit_w, SLEEVE_H / 2)}" '
            f'quat="{np.cos(ang / 2):.6f} 0 0 {np.sin(ang / 2):.6f}" '
            f'rgba="0.36 0.40 0.47 1" friction="0.3 0.005 0.0001" '
            f'condim="4" solref="0.004 1" contype="1" conaffinity="1"/>')
    world.append('  </body>')

    # ---- reference sites (nominal coordinates, taskA convention)
    world.append(f'  <site name="hole_center" pos="{v3(hx, hy, BASE_TOP_Z)}"'
                 f' size="0.003" rgba="1 0 0 0"/>')
    world.append(f'  <site name="hole_mouth" pos="{v3(hx, hy, SLEEVE_MOUTH_Z)}"'
                 f' size="0.003" rgba="0 1 0 0"/>')
    world.append(f'  <site name="peg_start" '
                 f'pos="{v3(hx, hy, PEG_START_BOTTOM_Z + PEG_HALF_H)}"'
                 f' size="0.003" rgba="0 0 1 0"/>')

    world.append('  <camera mode="fixed" name="frontview" pos="1.4 0 1.35" '
                 'quat="0.56 0.43 0.43 0.56"/>')
    world.append('  <camera mode="fixed" name="agentview" pos="0.5 0 1.3" '
                 'quat="0.653 0.271 0.271 0.653"/>')

    # ---- peg: free-jointed cylinder gripped + hung above the bore
    peg = [f'  <body name="peg" '
           f'pos="{v3(hx, hy, PEG_START_BOTTOM_Z + PEG_HALF_H)}">',
           '    <freejoint name="peg_j"/>',
           f'    <geom name="peg_g" type="cylinder" size="{fmt(PEG_R)} '
           f'{fmt(PEG_HALF_H)}" rgba="0.80 0.55 0.20 1" density="2700" '
           f'friction="0.4 0.005 0.0001" condim="4" solref="0.004 1" '
           f'contype="1" conaffinity="1"/>',
           '  </body>']

    custom = []
    for pname, (tcode, dz, od, hh) in PART_META.items():
        custom.append(f'    <numeric name="meta_{pname}" '
                      f'data="{tcode} {fmt(dz)} {fmt(od)} {fmt(hh)}"/>')

    xml = f"""<mujoco model="peg_in_hole">
  <!-- Minimal peg-in-hole scene for TRAINING the learned insertion
       policy (simbench.skills.learned).  Generated by gen_insert_scene.py
       -- edit the generator, not this file.  Physics mirrors taskA/C:
       dt=0.002 + impratio 20 + elliptic cone + soft 0.004-1 solrefs. -->
  <compiler angle="radian" meshdir="../assets/panda"
            inertiagrouprange="0 0" autolimits="true"/>
  <option timestep="0.002" impratio="20" cone="elliptic"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.4 0.4 0.4" specular="0.2 0.2 0.2"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <map znear="0.001"/>
    <quality shadowsize="2048" offsamples="4"/>
    <global offwidth="640" offheight="480"/>
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
{chr(10).join(peg)}
  </worldbody>
</mujoco>
"""
    return xml


def main():
    out = build()
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "peg_in_hole.xml")
    with open(path, "w") as f:
        f.write(out)
    print(f"wrote {path} ({len(out)} bytes)")


if __name__ == "__main__":
    main()
