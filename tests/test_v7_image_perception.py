import inspect

import numpy as np

from simbench.value.pin_geometry import evaluate_pin_state
from simbench.value.rgbd_perception import CameraCalibration, estimate_scene


def _calibration():
    return CameraCalibration(100, 100, 100., 100., 49.5, 49.5,
                             tuple(np.eye(4).ravel()))


def _frame(x0=45):
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    rgb[:] = [30, 30, 30]
    rgb[45:56, x0:x0 + 11] = [220, 80, 40]
    depth = np.full((100, 100), .9, dtype=np.float32)
    return {"rgb": rgb, "depth_m": depth}


def test_rgbd_estimate_moves_with_pixels_and_has_no_physics_argument():
    template = {"color_rgb": [220, 80, 40], "pixel_area_hint": 100,
                "world_z_bounds": [-1., 1.]}
    args = list(inspect.signature(estimate_scene).parameters)
    assert args == ["rgbd_frames", "camera_calibration", "object_templates", "previous_estimates"]
    first = estimate_scene({"view": _frame(40)}, {"view": _calibration()}, {"part": template})
    second = estimate_scene({"view": _frame(60)}, {"view": _calibration()}, {"part": template})
    assert first["objects"]["part"]["valid"]
    assert second["objects"]["part"]["valid"]
    assert second["objects"]["part"]["position_m"][0] > first["objects"]["part"]["position_m"][0]


def test_blank_rgbd_returns_unknown_instead_of_filling_pose():
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    depth = np.zeros((100, 100), dtype=np.float32)
    result = estimate_scene({"view": {"rgb": rgb, "depth_m": depth}},
                            {"view": _calibration()},
                            {"part": {"color_rgb": [220, 80, 40]}})
    assert not result["objects"]["part"]["valid"]
    assert result["objects"]["part"]["position_m"] is None


def test_pin_geometry_accepts_tilted_shaft_but_rejects_edge_and_low_contact():
    base = dict(pin_origin=[0., 0., .04], hole_entry=[0., 0., 0.],
                hole_axis=[0., 0., 1.], released=True, touching_finger=False)
    good = evaluate_pin_state(pin_axis=[.05, 0., .999], **base)
    edge = evaluate_pin_state(pin_axis=[.0, .35, .999], **base)
    low = evaluate_pin_state(pin_origin=[0., 0., .055], pin_axis=[0., 0., 1.], **{k: v for k, v in base.items() if k != "pin_origin"})
    assert good["success"]
    assert not edge["success"]
    assert not low["success"]


def test_pin_release_and_retention_phases_are_separate():
    args = dict(pin_origin=[0., 0., .04], pin_axis=[0., 0., 1.],
                hole_entry=[0., 0., 0.], hole_axis=[0., 0., 1.])
    held = evaluate_pin_state(**args, released=False, touching_finger=True,
                              phase="inserted_while_held")
    released = evaluate_pin_state(**args, released=True, touching_finger=False,
                                  phase="inserted_after_release")
    retained = evaluate_pin_state(**args, released=True, touching_finger=False,
                                  phase="retained_after_stroke")
    assert held["success"] and not held["inserted_after_release"]
    assert released["success"] and retained["success"]

