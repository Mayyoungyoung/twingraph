"""Closed-set RGB-D geometry perception.

The detector in this module is intentionally independent of MuJoCo.  It only
accepts image arrays, camera calibration, and static CAD/template metadata.
The simulator is allowed to render the arrays, but no physics/session object
is passed across this boundary.  It is a lightweight deterministic detector
for the small, known set of tabletop parts used by value-v7; it is not an
RGB-D neural detector and its limitations are recorded in the evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import hashlib
import json
from typing import Mapping, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraCalibration:
    """Pinhole intrinsics and a world-from-camera rigid transform."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    world_from_camera: tuple
    version: str = "mujoco-pinhole-v1"

    def matrix(self) -> np.ndarray:
        arr = np.asarray(self.world_from_camera, dtype=float)
        if arr.size == 16:
            arr = arr.reshape(4, 4)
        elif arr.size == 12:
            arr = arr.reshape(3, 4)
        if arr.shape == (3, 4):
            out = np.eye(4); out[:3] = arr; return out
        if arr.shape != (4, 4):
            raise ValueError("world_from_camera must be 4x4")
        return arr

    def manifest(self) -> dict:
        return {
            "width": int(self.width), "height": int(self.height),
            "fx": float(self.fx), "fy": float(self.fy),
            "cx": float(self.cx), "cy": float(self.cy),
            "world_from_camera": self.matrix().tolist(),
            "version": self.version,
        }


def _as_calibration(row) -> CameraCalibration:
    if isinstance(row, CameraCalibration):
        return row
    return CameraCalibration(**dict(row))


def _depth_m(depth: np.ndarray) -> np.ndarray:
    d = np.asarray(depth)
    if d.ndim != 2:
        raise ValueError("depth must be HxW")
    if np.issubdtype(d.dtype, np.integer):
        return d.astype(np.float32) / 1000.0
    return d.astype(np.float32)


def _rgb(rgb: np.ndarray) -> np.ndarray:
    a = np.asarray(rgb)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError("rgb must be HxWx3")
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return a


def _backproject(u, v, depth, calibration: CameraCalibration):
    d = np.asarray(depth, dtype=float)
    x = (np.asarray(u, dtype=float) - calibration.cx) * d / calibration.fx
    y = -(np.asarray(v, dtype=float) - calibration.cy) * d / calibration.fy
    local = np.stack([x, y, -d], axis=-1)
    tf = calibration.matrix()
    return local @ tf[:3, :3].T + tf[:3, 3]


def _prototype_rgb(template: Mapping) -> np.ndarray:
    value = template.get("color_rgb", template.get("rgba", [0.5, 0.5, 0.5]))
    value = np.asarray(value, dtype=float)
    if value.max(initial=0) <= 1.0:
        value = value * 255.0
    return np.clip(value[:3], 0, 255).astype(np.float32)


def _candidate_mask(rgb: np.ndarray, depth: np.ndarray, template: Mapping) -> np.ndarray:
    """Robust colour/depth foreground mask without instance IDs.

    Colour is used as a proposal cue and depth validity removes the sky.  A
    broad chroma tolerance handles shadows and renderer lighting changes; the
    final component is still checked against CAD size and point-cloud quality.
    """
    proto = _prototype_rgb(template)
    dist = np.linalg.norm(rgb.astype(np.float32) - proto[None, None, :], axis=2)
    # RGB is a cue rather than an encoded instance ID.  A calibrated broad
    # tolerance keeps illumination/shadow variants while rejecting the tan
    # tabletop and the orange accents on the fixed rail.
    tolerance = float(template.get("rgb_tolerance", 72.0))
    mask = dist < tolerance
    if template.get("neutral", False):
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask &= hsv[..., 1] < 120
    mask &= np.isfinite(depth) & (depth > 0.08) & (depth < 5.0)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)


def _components(mask: np.ndarray):
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if int(area) < 8:
            continue
        ys, xs = np.where(labels == i)
        out.append(dict(label=i, x=int(x), y=int(y), w=int(w), h=int(h), area=int(area), xs=xs, ys=ys))
    return out


def _component_score(component, rgb, depth, template):
    proto = _prototype_rgb(template)
    pixels = rgb[component["ys"], component["xs"]].astype(float)
    colour = float(np.linalg.norm(np.median(pixels, axis=0) - proto) / 255.0)
    nominal = np.asarray(template.get("size_m", [0.02, 0.02, 0.01]), dtype=float)
    area_hint = float(template.get("pixel_area_hint", max(12.0, nominal[0] * nominal[1] * 50000)))
    size_penalty = abs(math.log(max(component["area"], 1) / area_hint))
    return colour + 0.12 * min(size_penalty, 8.0)


def _quat_from_yaw(yaw):
    return [float(math.cos(yaw / 2.0)), 0.0, 0.0, float(math.sin(yaw / 2.0))]


def _estimate_component(component, rgb, depth, calibration, template, source_view):
    xs, ys = component["xs"], component["ys"]
    z = depth[ys, xs]
    good = np.isfinite(z) & (z > 0.08) & (z < 5.0)
    if int(good.sum()) < 8:
        return None
    xs, ys, z = xs[good], ys[good], z[good]
    points = _backproject(xs, ys, z, calibration)
    # Raised bosses are deliberately included in the closed-set CAD model,
    # but their visible pixels bias a raw median away from the body origin.
    # For a declared plate-like template, estimate the reference from the
    # lower depth layer (the broad plate top) and keep the boss for quality/
    # orientation diagnostics.  This is image geometry, not simulator state.
    fit_points = points
    if template.get("reference_mode") == "lower_plane" and len(points) >= 12:
        cutoff = float(np.quantile(points[:, 2], 0.60))
        plane = points[points[:, 2] <= cutoff]
        if len(plane) >= 8:
            fit_points = plane
    position = np.median(fit_points, axis=0)
    if template.get("center_mode") == "obb_midpoint" and len(fit_points) >= 12:
        # A boss can add many pixels on one side of a plate.  The midpoint of
        # the oriented point-cloud extents is a better CAD-origin estimate
        # than the pixel median and remains purely image-derived.
        cloud = fit_points - np.mean(fit_points, axis=0)
        try:
            _, _, basis = np.linalg.svd(cloud, full_matrices=False)
            axes = basis[:2]
            proj = cloud @ axes.T
            midpoint = 0.5 * (proj.min(axis=0) + proj.max(axis=0))
            position = np.mean(fit_points, axis=0) + midpoint @ axes
        except np.linalg.LinAlgError:
            pass
    # The visible point-cloud reference is converted to the static CAD body
    # origin.  Offsets are declared per template (surface/shaft geometry),
    # never inferred from a hidden simulator pose.
    reference_offset = np.asarray(template.get("reference_offset_m", [0.0, 0.0, 0.0]), dtype=float)
    if reference_offset.shape == (3,):
        position = position + reference_offset
    # A robust PCA gives a task-relevant yaw for elongated parts.  A normal
    # from the smallest eigenvector is recorded, but symmetric pins do not
    # claim an unobservable roll angle.
    centered = fit_points - np.mean(fit_points, axis=0)
    try:
        _, values, vectors = np.linalg.svd(centered, full_matrices=False)
        major = vectors[0]
        if abs(major[2]) > 0.85:
            major = vectors[1] if vectors.shape[0] > 1 else major
        yaw = float(math.atan2(major[1], major[0])) + float(template.get("yaw_offset_rad", 0.0))
        if template.get("symmetric_axis", False):
            yaw = 0.0
        residual = float(np.median(np.abs(centered @ vectors[-1])))
    except np.linalg.LinAlgError:
        yaw, residual = 0.0, float("inf")
    spread = float(np.median(np.linalg.norm(centered, axis=1)))
    quality = float(np.clip(1.0 / (1.0 + 40.0 * residual + 8.0 * spread), 0.0, 1.0))
    return {
        "position_m": position.tolist(), "quat_wxyz": _quat_from_yaw(yaw),
        "bbox_xyxy": [component["x"], component["y"], component["x"] + component["w"], component["y"] + component["h"]],
        "mask_area_px": int(component["area"]), "quality": quality,
        "fit_residual_m": residual, "valid": bool(quality >= float(template.get("min_quality", 0.02))),
        "occluded": bool(component["area"] < float(template.get("min_area_px", 8))),
        "source_view": source_view, "timestamp": None,
    }


def estimate_scene(rgbd_frames: Mapping, camera_calibration: Mapping,
                   object_templates: Mapping, previous_estimates: Optional[Mapping] = None) -> dict:
    """Estimate known objects from RGB-D only.

    ``rgbd_frames`` maps view names to ``{"rgb": ..., "depth_m": ...}`` (or
    ``depth_mm``).  The result intentionally keeps missing objects as invalid
    rows; callers must not silently fill those rows from a simulator state.
    """
    if not rgbd_frames or not object_templates:
        raise ValueError("frames and object templates are required")
    detections = {name: [] for name in object_templates}
    for view, frame in rgbd_frames.items():
        rgb = _rgb(frame["rgb"])
        depth = _depth_m(frame.get("depth_m", frame.get("depth_mm")))
        if depth.shape != rgb.shape[:2]:
            raise ValueError("RGB and depth shape mismatch")
        cal = _as_calibration(camera_calibration[view])
        for part, template in object_templates.items():
            comps = _components(_candidate_mask(rgb, depth, template))
            eligible = []
            bounds = template.get("world_bounds")
            for comp in comps:
                row = _estimate_component(comp, rgb, depth, cal, template, view)
                if row is not None:
                    if bounds is not None:
                        xyz = np.asarray(row["position_m"], dtype=float)
                        lo, hi = np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float)
                        if np.any(xyz[:2] < lo[:2]) or np.any(xyz[:2] > hi[:2]):
                            continue
                    z_bounds = template.get("world_z_bounds", [0.74, 0.95])
                    if not (float(z_bounds[0]) <= float(row["position_m"][2]) <= float(z_bounds[1])):
                        continue
                    row["score"] = float(1.0 - _component_score(comp, rgb, depth, template) / 5.0)
                    eligible.append((comp, row))
            # At most the declared number of visual instances is associated
            # with a class.  Identical pins remain separate anonymous tracks.
            count = int(template.get("instances", 1))
            # Keep a small candidate set for an installed object whose CAD
            # relation to another detected part is known; the final choice is
            # made below from image geometry, not from a body ID.
            limit = 12 if template.get("anchor_part") else count
            for comp, row in sorted(eligible, key=lambda cr: (_component_score(cr[0], rgb, depth, template), -float(cr[1].get("quality", 0.0))))[:limit]:
                detections[part].append(row)
    objects = {}
    for part, rows in detections.items():
        rows.sort(key=lambda r: (-(float(r.get("score", 0.0))), -float(r.get("quality", 0.0)), r.get("source_view", "")))
        template = object_templates[part]
        if rows and template.get("anchor_part") in objects:
            anchor = objects[template["anchor_part"]]
            anchor_xyz = anchor.get("position_m") if anchor.get("valid") else None
            if anchor_xyz is not None:
                expected = np.asarray(anchor_xyz, dtype=float) + np.asarray(template.get("anchor_offset_m", [0., 0., 0.]), dtype=float)
                chosen = min(rows, key=lambda r: float(np.linalg.norm(np.asarray(r["position_m"], dtype=float) - expected)))
                if float(np.linalg.norm(np.asarray(chosen["position_m"], dtype=float) - expected)) <= float(template.get("anchor_max_distance_m", .08)):
                    rows = [chosen]
                else:
                    rows = []
        row = rows[0] if rows else {
            "position_m": None, "quat_wxyz": None, "bbox_xyxy": None,
            "mask_area_px": 0, "quality": 0.0, "fit_residual_m": None,
            "valid": False, "occluded": True, "source_view": None, "timestamp": None,
        }
        row = dict(row); row["track_id"] = f"rgbd:{part}:0"; row["category"] = part
        objects[part] = row
    payload = {
        "schema": "twingraph.rgbd_geometry.v1", "backend": "rgbd_geometry",
        "objects": objects, "views": list(rgbd_frames),
        "detector_resolution": {v: [int(_rgb(f["rgb"]).shape[1]), int(_rgb(f["rgb"]).shape[0])] for v, f in rgbd_frames.items()},
        "source": "rgb_pixels_depth_pixels_calibration_static_cad_templates",
        "history_used": previous_estimates is not None,
    }
    return payload


def to_sensor_observation(result: Mapping, parts) -> dict:
    """Convert detector rows to the common frozen observation envelope."""
    objects = {}
    for part in parts:
        row = dict(result["objects"].get(part, {}))
        objects[part] = {
            "position_m": row.get("position_m"), "quat_wxyz": row.get("quat_wxyz"),
            "valid": bool(row.get("valid", False)), "quality": float(row.get("quality", 0.0)),
            "source_view": row.get("source_view"), "track_id": row.get("track_id"),
            "bbox_xyxy": row.get("bbox_xyxy"), "fit_residual_m": row.get("fit_residual_m"),
        }
    payload = dict(result)
    payload["objects"] = objects
    payload["decision_boundary"] = "pre-twin-or-target-execution"
    unsigned = dict(payload)
    payload["sha256"] = hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return payload
