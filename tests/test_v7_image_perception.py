import inspect

import numpy as np

from simbench.value.pin_geometry import evaluate_pin_state
from simbench.value.pin_geometry import PinInsertionConfig
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
    good = evaluate_pin_state(pin_axis=[.005, 0., .999], **base)
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


def test_pin_radius_and_original_plate_limit_both_apply():
    args = dict(pin_origin=[0., 0., .04], pin_axis=[0., 0., 1.],
                hole_entry=[0., 0., 0.], hole_axis=[0., 0., 1.],
                released=True, touching_finger=False)
    assert evaluate_pin_state(**args)["success"]
    wall = evaluate_pin_state(**{**args, "pin_origin": [.001, 0., .04]})
    assert not wall["success"]
    assert np.isclose(wall["permitted_center_offset_m"], .0005)
    wider_plate = PinInsertionConfig(plate_hole_half_width_m=.006)
    assert evaluate_pin_state(**{**args, "pin_origin": [.001, 0., .04]}, config=wider_plate)["success"]


def test_v7_trial_does_not_advertise_unapplied_perturbations():
    from simbench.value.collect_v7 import trial_spec
    row = trial_spec(1200, 1, "development")
    assert set(row) == {"domain", "repeat", "friction_scale", "actuator_gain_scale"}


def test_v7_task_requirement_is_fixed_for_all_candidates():
    from simbench.value import stage_v7
    import tempfile
    from pathlib import Path
    import pytest
    with tempfile.TemporaryDirectory(dir=Path(stage_v7.__file__).parents[2]) as directory:
        _, session, _, targets = stage_v7.make_scene(1200, Path(directory), level="L1")
        plans, _ = stage_v7.build_pool(session, targets, 1200, n=12)
        assert {p.prefix["stroke_minimum"] for p in plans} == {stage_v7.TASK_STROKE_MINIMUM_M}
        with pytest.raises(ValueError, match="cannot change"):
            stage_v7._full_plan(session, targets, plans[0].prefix["order"],
                                plans[0].prefix["choices"], 0, 1.5, 14., .09)


def test_session_checkpoint_restores_execution_targets():
    from simbench.value import stage_v7
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory(dir=Path(stage_v7.__file__).parents[2]) as directory:
        _, session, _, _ = stage_v7.make_scene(1200, Path(directory), level="L1")
        snapshot = session.snapshot()
        original = snapshot["stage_targets"]
        session.stage_targets["pin_left"][0] += .1
        session.execution_relocalizations = [{"part": "pin_left"}]
        session.restore(snapshot)
        assert session.stage_targets == original
        assert not hasattr(session, "execution_relocalizations")
