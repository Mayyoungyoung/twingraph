"""Shared primitive-geometry builders for simbench scene XMLs.

These mirror the trigonometry of the old robosuite CompositeObject
builders (gearbox_objects.py) so the transplanted scenes are physically
identical: a hollow cylinder is a ring of n boxes, a boss is a small
vertical wall ring with a hole.  All sizes are half-extents semantics of
MuJoCo boxes; ``height`` is the FULL wall height.
"""
import numpy as np


def fmt(x):
    """Compact float formatting for XML attributes."""
    return f"{float(x):.6f}".rstrip("0").rstrip(".")


def v3(x, y, z):
    return f"{fmt(x)} {fmt(y)} {fmt(z)}"


def quat_z(ang):
    """Quaternion for a rotation about z by ``ang`` (w x y z)."""
    return f"{fmt(np.cos(ang / 2.0))} 0 0 {fmt(np.sin(ang / 2.0))}"


def ring_segs(r_out, r_hole, height, n):
    """Box segments of a hollow-cylinder wall, centred at z=0.

    Returns a list of dicts {pos, quat, size} (all in the body frame).
    """
    unit_w = r_out * np.sin(np.pi / n)
    unit_h = (r_out - r_hole) * np.cos(np.pi / n) / 2.0
    int_r = r_hole * np.cos(np.pi / n) + unit_h
    segs = []
    for i in range(n):
        ang = 2.0 * np.pi * i / n
        segs.append(dict(
            pos=(int_r * np.cos(ang), int_r * np.sin(ang), 0.0),
            quat=quat_z(ang),
            size=(unit_h, unit_w, height / 2.0)))
    return segs


def boss_segs(r_out, r_hole, height, n):
    """Wall segments of a small boss; wall spans local z 0..height
    (segment centres at z=height/2)."""
    unit_w = r_out * np.sin(np.pi / n)
    unit_h = (r_out - r_hole) * np.cos(np.pi / n) / 2.0
    int_r = r_hole * np.cos(np.pi / n) + unit_h
    segs = []
    for i in range(n):
        ang = 2.0 * np.pi * i / n
        segs.append(dict(
            pos=(int_r * np.cos(ang), int_r * np.sin(ang), height / 2.0),
            quat=quat_z(ang),
            size=(unit_h, unit_w, height / 2.0)))
    return segs


def geom_xml(name, gtype, pos, size, quat=None, rgba=None, material=None,
             density=None,
             friction=None, condim=None, solref=None, contype=None,
             conaffinity=None, indent=2):
    """One <geom> element."""
    a = [f'type="{gtype}"', f'name="{name}"', f'pos="{v3(*pos)}"']
    if gtype == "box":
        a.append(f'size="{v3(*size)}"')
    elif gtype == "cylinder":
        a.append(f'size="{fmt(size[0])} {fmt(size[1])}"')
    elif gtype == "sphere":
        a.append(f'size="{fmt(size[0])}"')
    elif gtype == "plane":
        a.append(f'size="{v3(*size)}"')
    if quat is not None:
        a.append(f'quat="{quat}"')
    if rgba is not None:
        a.append(f'rgba="{rgba}"')
    if material is not None:
        a.append(f'material="{material}"')
    if density is not None:
        a.append(f'density="{fmt(density)}"')
    if friction is not None:
        a.append(f'friction="{friction}"')
    if condim is not None:
        a.append(f'condim="{condim}"')
    if solref is not None:
        a.append(f'solref="{solref}"')
    if contype is not None:
        a.append(f'contype="{contype}"')
    if conaffinity is not None:
        a.append(f'conaffinity="{conaffinity}"')
    pad = " " * indent
    return f"{pad}<geom " + " ".join(a) + "/>"


def ring_geoms(prefix, r_out, r_hole, height, n, rgba, density,
               friction=None, drop_segs=(), z_off=0.0, indent=2):
    """Geoms of a full ring wall (segments centred at z_off)."""
    out = []
    for i, s in enumerate(ring_segs(r_out, r_hole, height, n)):
        if i in drop_segs:
            continue
        pos = (s["pos"][0], s["pos"][1], s["pos"][2] + z_off)
        out.append(geom_xml(f"{prefix}_seg{i}", "box", pos, s["size"],
                            quat=s["quat"], rgba=rgba, density=density,
                            friction=friction, condim=4, indent=indent))
    return "\n".join(out)


def boss_geoms(prefix, r_out, r_hole, height, n, rgba, density,
               friction=None, indent=2):
    """Geoms of a boss wall (spans local z 0..height)."""
    out = []
    for i, s in enumerate(boss_segs(r_out, r_hole, height, n)):
        out.append(geom_xml(f"{prefix}_boss{i}", "box", s["pos"], s["size"],
                            quat=s["quat"], rgba=rgba, density=density,
                            friction=friction, condim=4, indent=indent))
    return "\n".join(out)


def hex_segs(r_out, r_hole, height, ang0=0.0):
    """Box segments of a HEXAGONAL ring wall (nut), centred at z=0.

    With ang0=0 the segment face normals (flats) point along local
    +/-x and +/-y, so the parallel pads grip across the flats from
    either world axis direction (torque transmission -- a round nut
    slips).  Note the polygon flat-face reduction: the effective hole
    radius is r_hole*cos(pi/6) and the effective outer across-flats is
    2*r_out*cos(pi/6).
    """
    n = 6
    unit_w = r_out * np.sin(np.pi / n)
    unit_h = (r_out - r_hole) * np.cos(np.pi / n) / 2.0
    int_r = r_hole * np.cos(np.pi / n) + unit_h
    segs = []
    for i in range(n):
        ang = ang0 + 2.0 * np.pi * i / n
        segs.append(dict(
            pos=(int_r * np.cos(ang), int_r * np.sin(ang), 0.0),
            quat=quat_z(ang),
            size=(unit_h, unit_w, height / 2.0)))
    return segs


def hex_geoms(prefix, r_out, r_hole, height, rgba, density,
              friction=None, z_off=0.0, indent=2):
    """Geoms of a hexagonal nut ring (segments centred at z_off)."""
    out = []
    for i, s in enumerate(hex_segs(r_out, r_hole, height)):
        pos = (s["pos"][0], s["pos"][1], s["pos"][2] + z_off)
        out.append(geom_xml(f"{prefix}_hex{i}", "box", pos, s["size"],
                            quat=s["quat"], rgba=rgba, density=density,
                            friction=friction, condim=4, indent=indent))
    return "\n".join(out)


def site_xml(name, pos, indent=2):
    pad = " " * indent
    return (f'{pad}<site name="{name}" pos="{v3(*pos)}" size="0.003" '
            f'rgba="1 0 0 0"/>')


# ---------------------------------------------------------------------------
# gearbox part constants (identical to assembly/gearbox_objects.py)
# ---------------------------------------------------------------------------
HOUSING = dict(outer_r=0.026, hole_r=0.012, height=0.045, n=12,
               # boss_depth 16mm (2026-09-01, up from 8mm): the seated
               # pin is an inverted pendulum in its 1mm-clearance bore
               # -- 8mm of engagement admits a geometric lean up to
               # ~14deg (and measured up to 41deg after the release
               # bounce), which the cover descend then caught and flung
               # off the table; 16mm caps the geometric lean at ~8deg,
               # well inside the cover boss-hole capture band
               boss_outer=0.0065, boss_r=0.0042, boss_depth=0.016,
               boss_ring_r=0.01825, density=500.0,
               rgba="0.55 0.55 0.60 1")
SHAFT = dict(lower_r=0.010, lower_h=0.045, collar_r=0.013, collar_h=0.004,
             upper_r=0.0065, upper_h=0.060, density=500.0,
             rgba="0.35 0.40 0.85 1")
GEAR = dict(outer_r=0.010, hole_r=0.0085, height=0.012, n=10,
            density=300.0, rgba="0.75 0.30 0.30 1")
SPACER = dict(outer_r=0.018, hole_r=0.008, height=0.012, n=10,
              density=300.0, rgba="0.35 0.75 0.35 1")
BEARING = dict(outer_r=0.022, hole_r=0.0085, height=0.012, n=10,
               density=300.0, rgba="0.85 0.85 0.35 1",
               drop_segs=(0,))
COVER = dict(outer_r=0.026, hole_r=0.014, height=0.010, n=16,
             boss_outer=0.0065, boss_r=0.0042, boss_ring_r=0.01825,
             density=400.0, rgba="0.90 0.60 0.20 1",
             drop_segs=(0, 4, 12))   # seg0: latch channel (+x);
                                     # seg4/12 (+/-y): stud notches
PIN = dict(r=0.0032, length=0.052, density=5000.0,
           rgba="0.15 0.15 0.15 1")   # r3.2 x 52mm: the 3-PASS
                                       # 2026-09-01 baseline (the cover
                                       # annulus CLAMPS the seated pin
                                       # top once it lands -- that
                                       # press is what holds the pin;
                                       # 3.6x40/56 variants destabilised
                                       # the carry or the thread)
# bearing retainer ring (S8): the stable thread-insert finale that
# replaced the retired stud-nut screw_drive (MuJoCo has no thread
# constraint, a loose nut spun on the smooth stud and was flung off on
# release in every measured variant).  Threads onto the shaft upper end
# (r6.5) through its 7mm hole and seats on the cover top -- classic
# shaft-end clamp ring look.  outer_r 17mm: the band r14-17 lands on
# the cover top face, and 17 < 18.25 keeps it clear of the latch pin.
RETAINER = dict(outer_r=0.017, hole_r=0.007, height=0.008, n=10,
                density=300.0, rgba="0.30 0.60 0.80 1")

# ---------------------------------------------------------------------------
# Task A additions: stud-bolt end-cap fastening + inspection latch slider
# ---------------------------------------------------------------------------
# hex nut (lock-nut profile): effective across-flats 2*outer_r*cos(30deg);
# effective hole 2*hole_r*cos(30deg) = 5.37mm over the stud r=2.5mm.
# height 12mm (2026-09-01, up from the original 4mm): the ~7deg
# carried-wrist tilt offsets the 16mm-tall pads over the part, and any
# wall shorter than ~10mm leaves the tilted pads nothing to bite while
# their low corners foul the table (measured: the 4-6mm nuts never
# lifted; the empty screw helix swept the whole stack off the table).
# 12mm flat-on-table grasps cleanly (measured 17mm lift).
# outer_r 9mm (up from 6.5mm): the nut bridges the cover's one-segment
# stud notch (~9.8mm wide at the 25mm stud radius) -- at the old 13mm
# across-flats the rim support was only ~1.6mm per side, and a nut
# leaning 3deg from the helix pivoted off the notch rim and fell to the
# floor the moment the pads opened (measured); 18mm across-flats gives
# ~4mm of rim support per side.
NUT = dict(outer_r=0.009, hole_r=0.0031, height=0.012, density=5000.0,
           rgba="0.20 0.20 0.25 1")
# stud bolt: rises from the housing top face OUTSIDE the bearing radius
# (ring_r 25mm clears the bearing outer 22mm by 0.5mm -- at 19mm the
# studs jammed the bearing 8mm sideways and toppled the stack, measured)
STUD = dict(r=0.0025, ring_r=0.025, clearance=0.064,
            rgba="0.70 0.70 0.75 1")
# inspection latch slider: slides in a channel on the cover top face
LATCH = dict(hx=0.006, hy=0.010, hz=0.004, density=800.0,
             rgba="0.60 0.20 0.20 1")
# cover-top slider channel: two rails at channel centre (cx, 0).
# rail_gap 24.4mm: the latch (12x20) is grasped across its 12mm short
# side (the stable bite) so its 20mm LONG side lands ACROSS the
# channel -- a 13.4mm gap wedged it at the mouth (measured push
# stalled 7 of 12mm); 24.4mm lets the long side ride freely.
# cx -10mm (2026-09-01): rail0 sits at cx-(gap/2+th) = -23.8mm, INBOARD
# of the ring wall outer face (-25.5mm) -- at the old cx=-13 rail0
# poked to -26.4mm, past the wall, and the 45deg-diagonal grasp pads
# caught rail0's corner with one pad and the wall with the other
# (asymmetric bite, span 61.7 vs 57; the cover slipped out of the
# pads on the 2nd lift leg, measured 3/3 seeds).  The cover grasp
# now closes along 112.5deg instead (planner), so cx is back to the
# latch-proven -13mm.
CHANNEL = dict(cx=-0.013, rail_gap=0.0244, rail_hy=0.015, rail_hz=0.004,
               rail_th=0.0012, rgba="0.90 0.60 0.20 1")

# low-slip friction for the latch channel (pin + boss walls), see
# gearbox_objects.py BOSS_FRICTION for the physics rationale
BOSS_FRICTION = "0.15 0.005 0.0001"
