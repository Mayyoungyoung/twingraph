"""Geometry-based pin insertion predicates.

The fixture CAD contains an 8 mm annular guide at the end-stop hole.  A pin
passes when its shaft occupies that cylindrical guide for a meaningful depth
and remains there after the gripper is opened.  Overall body-centre error and
roll/tilt are deliberately not acceptance criteria: a pin can be slightly
inclined while its shaft is still seated in the hole.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PinInsertionConfig:
    shaft_radius_m: float = 0.0033
    guide_inner_radius_m: float = 0.0055
    plate_hole_half_width_m: float = 0.004  # scene.holed_plate square aperture
    guide_length_m: float = 0.008  # v7_bore ring: centre z=.014, half=.004
    required_depth_m: float = 0.006  # 75% of the physical guide length
    shaft_tip_offset_m: float = -0.047
    shaft_head_offset_m: float = 0.006
    radial_clearance_m: float = 0.0002
    source: str = "stage_v7._add_pin_guides and scene.holed_plate CAD"

    def manifest(self):
        return {
            "shaft_radius_m": self.shaft_radius_m,
            "guide_inner_radius_m": self.guide_inner_radius_m,
            "plate_hole_half_width_m": self.plate_hole_half_width_m,
            "guide_length_m": self.guide_length_m,
            "required_depth_m": self.required_depth_m,
            "shaft_tip_offset_m": self.shaft_tip_offset_m,
            "shaft_head_offset_m": self.shaft_head_offset_m,
            "radial_clearance_m": self.radial_clearance_m,
            "source": self.source,
        }


def _unit(vec):
    v = np.asarray(vec, dtype=float)
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        raise ValueError("zero axis")
    return v / n


def insertion_geometry(pin_origin, pin_axis, hole_entry, hole_axis,
                       config: PinInsertionConfig = PinInsertionConfig(), samples=257):
    """Evaluate shaft/guide overlap using only supplied geometric quantities."""
    origin = np.asarray(pin_origin, dtype=float)
    paxis = _unit(pin_axis)
    entry = np.asarray(hole_entry, dtype=float)
    haxis = _unit(hole_axis)
    if samples < 8:
        raise ValueError("samples must be >= 8")
    offsets = np.linspace(config.shaft_tip_offset_m, config.shaft_head_offset_m, int(samples))
    points = origin[None, :] + offsets[:, None] * paxis[None, :]
    rel = points - entry[None, :]
    depth = -rel @ haxis
    radial_vec = rel + depth[:, None] * haxis[None, :]
    radial = np.linalg.norm(radial_vec, axis=1)
    # The square plate aperture surrounds the same axial region as the added
    # annular guide. Its inscribed circle is a conservative bound independent
    # of the fixture's unprovided transverse frame. At a tilted shaft's
    # constant-depth section, its circular radius projects by 1/|cos(theta)|.
    axial = abs(float(np.dot(paxis, haxis)))
    bore_radius = min(config.guide_inner_radius_m, config.plate_hole_half_width_m)
    permitted = bore_radius - config.radial_clearance_m - config.shaft_radius_m / max(axial, 1e-12)
    allowed = (depth >= 0.0) & (depth <= config.guide_length_m) & (radial <= permitted)
    valid_depths = depth[allowed]
    max_depth = float(valid_depths.max(initial=0.0))
    if valid_depths.size:
        ordered = np.sort(valid_depths)
        # Largest contiguous span in the sampled axial interval.  This avoids
        # counting a single tip contact as a deep insertion.
        gaps = np.diff(ordered)
        span = float(ordered[-1] - ordered[0]) if not gaps.size else float(ordered[-1] - ordered[0])
    else:
        span = 0.0
    # Require an uninterrupted solid shaft from the entry plane through the
    # required depth. Endpoint checks suffice because radial offset is convex
    # along a straight shaft and the bore bound is constant over this interval.
    along = float(np.dot(paxis, haxis))
    full_depth = abs(along) > 1e-9 and permitted >= 0
    boundary_offsets = []
    for boundary in (0., config.required_depth_m):
        offset = float(np.dot(entry - origin, haxis) - boundary) / along if abs(along) > 1e-9 else float('inf')
        point = origin + offset * paxis
        transverse = point - entry + boundary * haxis
        boundary_offsets.append(transverse.tolist())
        full_depth = full_depth and (config.shaft_tip_offset_m <= offset <= config.shaft_head_offset_m) and (np.linalg.norm(transverse) <= permitted + 1e-12)
    inserted = bool(full_depth and config.required_depth_m <= config.guide_length_m)
    return {
        "inserted": inserted,
        "max_insertion_depth_m": max_depth,
        "valid_depth_span_m": span,
        "minimum_required_depth_m": config.required_depth_m,
        "limiting_bore_radius_m": bore_radius,
        "permitted_center_offset_m": permitted,
        "center_offsets_entry_required_m": boundary_offsets,
        "max_radial_error_m": float(radial[allowed].max()) if allowed.any() else float("inf"),
        "samples": int(samples),
    }


def evaluate_pin_state(pin_origin, pin_axis, hole_entry, hole_axis,
                       released: bool, touching_finger: bool,
                       phase="inserted_after_release", config=PinInsertionConfig()):
    geom = insertion_geometry(pin_origin, pin_axis, hole_entry, hole_axis, config)
    held_ok = bool(geom["inserted"])
    released_ok = bool(released and not touching_finger)
    if phase == "inserted_while_held":
        success = held_ok
    elif phase in ("inserted_after_release", "retained_after_stroke"):
        success = held_ok and released_ok
    else:
        raise ValueError(f"unknown pin phase: {phase}")
    return dict(success=bool(success), phase=phase,
                inserted_while_held=held_ok,
                inserted_after_release=bool(held_ok and released_ok),
                retained_after_stroke=bool(held_ok and released_ok and phase == "retained_after_stroke"),
                released=bool(released), touching_finger=bool(touching_finger), **geom)


def evaluate_pin_context(ctx, pin_part, fixture_part="end_stop", hole_offset_m=(0., 0., 0.),
                         phase="inserted_after_release", released=True, touching_finger=False,
                         config=PinInsertionConfig()):
    """Independent evaluator adapter; this is never imported by perception."""
    fpos, fquat = ctx.obj_pose(fixture_part)
    R = np.zeros(9)
    import mujoco
    mujoco.mju_quat2Mat(R, fquat)
    R = R.reshape(3, 3)
    entry = np.asarray(fpos) + R @ np.asarray(hole_offset_m, dtype=float) + R[:, 2] * .018
    origin = np.asarray(ctx.obj_pos(pin_part), dtype=float)
    axis = np.asarray(ctx.obj_axis(pin_part), dtype=float)
    return evaluate_pin_state(origin, axis, entry, R[:, 2], released, touching_finger, phase, config)
