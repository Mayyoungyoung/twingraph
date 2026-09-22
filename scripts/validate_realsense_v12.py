"""SDK projection conformance using synthetic arrays; never opens a camera."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from unittest.mock import patch

import cv2
import numpy as np

from simbench.value import realsense_v12 as adapter
from simbench.value.rgbd_perception import _backproject

MODELS = ("brown_conrady", "modified_brown_conrady", "inverse_brown_conrady")


def intrinsics(model):
    return dict(width=96, height=72, fx=67.3, fy=71.7, ppx=41.3, ppy=37.1,
                model=model, coeffs=[-.19, .052, .003, -.004, .007])


def native_intrinsics(spec):
    import pyrealsense2 as rs
    native = rs.intrinsics()
    for key in ("width", "height", "fx", "fy", "ppx", "ppy"):
        setattr(native, key, spec[key])
    native.model = getattr(rs.distortion, spec["model"])
    native.coeffs = spec["coeffs"]
    return native


def validate_model(model):
    import pyrealsense2 as rs
    spec = intrinsics(model)
    native = native_intrinsics(spec)
    h, w = spec["height"], spec["width"]
    v, u = np.indices((h, w))
    # Independent oracle: invoke the installed vendor projection at every
    # ideal pinhole output pixel, not a locally reimplemented lens equation.
    expected = np.asarray([rs.rs2_project_point_to_pixel(native,
        [(float(x)-spec["ppx"])/spec["fx"], (float(y)-spec["ppy"])/spec["fy"], 1.])
        for y, x in zip(v.ravel(), u.ravel())], np.float32).reshape(h, w, 2)
    helper_x, helper_y = adapter.sdk_rectification_maps(spec)
    helper_error = float(np.max(abs(np.stack((helper_x, helper_y), -1)-expected)))
    assert helper_error < 2.e-5, helper_error
    rgb = np.stack(((u*2) % 256, (v*3) % 256, (u+v) % 256), -1).astype(np.uint8)
    depth = np.where(u < w//2, 2000, 8000).astype(np.uint16)
    depth[19:42, 23:49] = 0
    depth[(u+2*v) % 13 == 0] = 0
    scale = .00025  # Deliberately not the familiar 1 mm default.
    angle = .31
    world = np.array([[np.cos(angle), -np.sin(angle), 0., .13],
                      [np.sin(angle), np.cos(angle), 0., -.08],
                      [0., 0., 1., .7], [0., 0., 0., 1.]])
    remap = cv2.remap
    observed_maps = []
    def capture_remap(src, map_x, map_y, interpolation, **kwargs):
        observed_maps.append((map_x.copy(), map_y.copy(), interpolation))
        return remap(src, map_x, map_y, interpolation, **kwargs)
    with patch.object(adapter.cv2, "remap", side_effect=capture_remap):
        frame, calibration, metadata = adapter.adapt_aligned(rgb, depth, spec, world, depth_scale_m=scale,
            source="synthetic_arrays_sdk_conformance_only")
    assert len(observed_maps) == 2
    map_x, map_y, interpolation = observed_maps[-1]
    map_error = float(np.max(abs(np.stack((map_x, map_y), -1)-expected)))
    assert map_error < 2.e-5, map_error
    assert interpolation == cv2.INTER_NEAREST
    # An independent nearest-neighbour index lookup proves zeros are not
    # bilinearly averaged into positive invented depths at occlusion edges.
    sx, sy = np.rint(map_x).astype(int), np.rint(map_y).astype(int)
    inside = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
    expected_depth = np.zeros((h, w), np.float32)
    expected_depth[inside] = depth[sy[inside], sx[inside]].astype(np.float32)*scale
    np.testing.assert_array_equal(frame["depth_m"], expected_depth)
    zeros = expected_depth == 0
    assert np.count_nonzero(zeros) > 100
    assert np.all(frame["depth_m"][zeros] == 0)
    assert set(np.unique(frame["depth_m"])) <= {0., .5, 2.}
    # Rectification now has pinhole intrinsics; deproject with vendor NONE,
    # then apply the externally declared optical-to-world transform.
    pinhole = native_intrinsics({**spec, "model": "none", "coeffs": [0.]*5})
    pixels = [(8., 9.), (spec["ppx"], spec["ppy"]), (82., 57.), (51., 65.)]
    optical_points = np.asarray([rs.rs2_deproject_pixel_to_point(pinhole, p, 1.25) for p in pixels])
    expected_world = optical_points@world[:3, :3].T+world[:3, 3]
    actual_world = _backproject(np.asarray([p[0] for p in pixels]), np.asarray([p[1] for p in pixels]),
                               np.full(len(pixels), 1.25), calibration)
    world_error = float(np.max(np.abs(actual_world-expected_world)))
    assert world_error < 1.e-6, world_error
    assert metadata["rectified"] and metadata["depth_scale_m"] == scale
    return dict(model=model, image_size=[w, h], coefficients=spec["coeffs"], sdk_helper_max_pixel_error=helper_error,
        adapt_map_max_pixel_error=map_error, world_deprojection_max_error_m=world_error,
        depth_interpolation="nearest", depth_scale_m=scale, unknown_output_pixels=int(np.count_nonzero(zeros)),
        invented_depth_values=0, camera_opened=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    import pyrealsense2 as rs
    sdk_root = Path(rs.__file__).resolve().parent
    root = Path(__file__).resolve().parents[1]
    files = [Path(__file__), root/"simbench/value/realsense_v12.py", root/"simbench/value/rgbd_perception.py",
             root/"scripts/capture_realsense_v12.py"]
    record = dict(status="passed", hardware_connected=False, real_camera_frames_used=False,
        fixture="synthetic distorted RGB-D arrays with installed SDK projection oracle",
        versions=dict(python=platform.python_version(), numpy=np.__version__, opencv=cv2.__version__,
                      pyrealsense2=importlib.metadata.version("pyrealsense2")),
        checks=[validate_model(model) for model in MODELS],
        sdk_installation=str(sdk_root),
        sdk_binary_sha256={str(path.relative_to(sdk_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in sorted(sdk_root.rglob("*.so"))},
        sources={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        references=["https://github.com/realsenseai/librealsense/wiki/Projection-in-RealSense-SDK-2.0",
                    "https://github.com/realsenseai/librealsense/blob/master/src/rs.cpp"],
        limitation="No device/firmware, RGB-depth alignment capture, exposure, white-part recognition, or extrinsic calibration accuracy is validated.")
    (args.out/"validation.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
