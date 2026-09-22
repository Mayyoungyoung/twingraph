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
    # Optional physical entrance bands, (end depth from entry, bore radius).
    # An empty tuple preserves the V7/V8 cylindrical aperture.
    bore_profile: tuple = ()
    aperture_shape: str = "circular"
    acceptance_mode: str = "mouth_continuity"
    # Explicit V12 numerical-contact policy. These do not resize collision
    # geometry. Legacy scenes keep ideal zero-interpenetration acceptance.
    contact_robustness: bool = False
    retention_window_s: float = 0.2
    retention_samples: int = 5

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
            "bore_profile_depth_radius_m": [list(row) for row in self.bore_profile],
            "aperture_shape": self.aperture_shape,
            "acceptance_mode": self.acceptance_mode,
            "contact_robustness": self.contact_robustness,
            "contact_guard_formula": "min(0.05 * shaft_radius, 0.25 * nominal_radial_clearance)" if self.contact_robustness else None,
            "contact_guard_m": min(.05*self.shaft_radius_m,.25*(self.plate_hole_half_width_m-self.shaft_radius_m)) if self.contact_robustness else 0.,
            "retention_window_s": self.retention_window_s if self.contact_robustness else 0.,
            "retention_samples": self.retention_samples if self.contact_robustness else 0,
        }


def _unit(vec):
    v = np.asarray(vec, dtype=float)
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        raise ValueError("zero axis")
    return v / n


def insertion_geometry(pin_origin, pin_axis, hole_entry, hole_axis,
                       config: PinInsertionConfig = PinInsertionConfig(), samples=257, hole_axes=None):
    """Evaluate shaft/guide overlap using only supplied geometric quantities."""
    origin = np.asarray(pin_origin, dtype=float)
    paxis = _unit(pin_axis)
    entry = np.asarray(hole_entry, dtype=float)
    haxis = _unit(hole_axis)
    if config.aperture_shape not in ("circular", "square"):
        raise ValueError("unsupported aperture shape")
    if config.acceptance_mode not in ("mouth_continuity", "functional_contiguous"):
        raise ValueError("unsupported pin acceptance mode")
    transverse_axes = None
    if config.aperture_shape == "square":
        if config.bore_profile:
            raise ValueError("square aperture cannot use a circular bore profile")
        if hole_axes is None:
            if not np.allclose(np.abs(haxis), [0., 0., 1.]):
                raise ValueError("square hole requires its transverse CAD axes")
            transverse_axes = np.eye(3)[:2]
        else:
            transverse_axes = np.asarray(hole_axes, float)
        if (transverse_axes.shape != (2, 3)
                or not np.allclose(transverse_axes @ transverse_axes.T, np.eye(2), atol=1e-6)
                or not np.allclose(transverse_axes @ haxis, 0., atol=1e-6)):
            raise ValueError("invalid square-hole CAD axes")
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
    if config.bore_profile:
        ends = np.array([float(row[0]) for row in config.bore_profile])
        radii = np.array([float(row[1]) for row in config.bore_profile])
        if not (np.all(np.diff(ends) > 0) and ends[-1] >= config.guide_length_m and np.all(radii > config.shaft_radius_m)):
            raise ValueError("invalid bore profile")
        indices = np.clip(np.searchsorted(ends, depth, side="right"), 0, len(radii) - 1)
        local_bore = radii[indices]
        bore_radius = float(radii.min())
    else:
        local_bore = np.full_like(depth, bore_radius)
    shaft_cross_section = config.shaft_radius_m / max(axial, 1e-12)
    permitted = bore_radius - config.radial_clearance_m - shaft_cross_section
    permitted_xy = None
    if transverse_axes is not None:
        # The plane section of a tilted cylinder is an ellipse. Its support
        # radius along each square-hole axis is r*sqrt(1+(axis_i/axis_z)^2).
        support = config.shaft_radius_m * np.sqrt(1. + ((transverse_axes @ paxis) / max(axial, 1e-12)) ** 2)
        permitted_xy = config.plate_hole_half_width_m - config.radial_clearance_m - support
        cross_section_ok = np.all(np.abs(radial_vec @ transverse_axes.T) <= permitted_xy, axis=1)
        permitted = float(permitted_xy.min())
    else:
        cross_section_ok = radial <= local_bore - config.radial_clearance_m - shaft_cross_section
    allowed = (depth >= 0.0) & (depth <= config.guide_length_m) & cross_section_ok
    valid_depths = depth[allowed]
    max_depth = float(valid_depths.max(initial=0.0))
    valid_indices = np.flatnonzero(allowed)
    runs = np.split(valid_indices, np.flatnonzero(np.diff(valid_indices) > 1) + 1) if valid_indices.size else []
    segments = [[float(depth[run].min()), float(depth[run].max())] for run in runs if run.size]
    # Never bridge invalid gaps: each counted interval contains consecutive
    # samples whose full cylinder cross-sections fit the real aperture.
    span = max((hi-lo for lo, hi in segments), default=0.)
    # Require an uninterrupted solid shaft from the entry plane through the
    # required depth. Endpoint checks suffice because radial offset is convex
    # along a straight shaft and the bore bound is constant over this interval.
    along = float(np.dot(paxis, haxis))
    full_depth = abs(along) > 1e-9 and permitted >= 0
    boundary_offsets = []
    boundaries = [0.] + [float(row[0]) for row in config.bore_profile
                         if 0. < float(row[0]) < config.required_depth_m] + [config.required_depth_m]
    profile_offsets = []
    for boundary in boundaries:
        offset = float(np.dot(entry - origin, haxis) - boundary) / along if abs(along) > 1e-9 else float('inf')
        point = origin + offset * paxis
        transverse = point - entry + boundary * haxis
        if config.bore_profile:
            index = min(int(np.searchsorted(ends, boundary, side="right")), len(radii) - 1)
            boundary_bore = float(radii[index])
        else:
            boundary_bore = bore_radius
        boundary_permitted = boundary_bore - config.radial_clearance_m - shaft_cross_section
        profile_offsets.append(dict(depth_m=boundary, transverse_m=transverse.tolist(),
                                    bore_radius_m=boundary_bore, permitted_center_offset_m=boundary_permitted))
        if boundary in (0., config.required_depth_m):
            boundary_offsets.append(transverse.tolist())
        inside = (np.all(np.abs(transverse_axes @ transverse) <= permitted_xy + 1e-12)
                  if transverse_axes is not None else np.linalg.norm(transverse) <= boundary_permitted + 1e-12)
        full_depth = full_depth and (config.shaft_tip_offset_m <= offset <= config.shaft_head_offset_m) and inside
    mouth_continuity = bool(full_depth and config.required_depth_m <= config.guide_length_m)
    contiguous = bool(span >= config.required_depth_m)
    inserted = contiguous if config.acceptance_mode == "functional_contiguous" else mouth_continuity
    return {
        "inserted": inserted,
        "max_insertion_depth_m": max_depth,
        "valid_depth_span_m": span,
        "longest_contiguous_depth_span_m": span,
        "valid_contiguous_depth_intervals_m": segments,
        "mouth_continuity_pass": mouth_continuity,
        "functional_contiguous_pass": contiguous,
        "acceptance_mode": config.acceptance_mode,
        "minimum_required_depth_m": config.required_depth_m,
        "limiting_bore_radius_m": bore_radius,
        "permitted_center_offset_m": permitted,
        "aperture_shape": config.aperture_shape,
        "permitted_center_offset_xy_m": permitted_xy.tolist() if permitted_xy is not None else None,
        "center_offsets_entry_required_m": boundary_offsets,
        "center_offsets_by_depth": profile_offsets,
        "max_radial_error_m": float(radial[allowed].max()) if allowed.any() else float("inf"),
        "samples": int(samples),
    }


def evaluate_pin_state(pin_origin, pin_axis, hole_entry, hole_axis,
                       released: bool, touching_finger: bool,
                       phase="inserted_after_release", config=PinInsertionConfig(), hole_axes=None):
    geom = insertion_geometry(pin_origin, pin_axis, hole_entry, hole_axis, config, hole_axes=hole_axes)
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
    if config.contact_robustness:
        from .pin_contact_v12 import evaluate_context_contact
        return evaluate_context_contact(ctx, pin_part, fixture_part, origin, axis, entry, R,
            released=released, touching_finger=touching_finger, phase=phase, config=config)
    return evaluate_pin_state(origin, axis, entry, R[:, 2], released, touching_finger, phase, config,
                              hole_axes=R[:, :2].T)
