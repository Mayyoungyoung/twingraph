"""White geometry perception, without color identity or simulator-pose inputs."""
from copy import deepcopy

import numpy as np
import pytest

from simbench.value import cad_rgbd_v12, stage_v7, stage_v12
from simbench.value.rgbd_perception import _as_calibration


@pytest.fixture(scope="module")
def white_installed(tmp_path_factory):
    session = stage_v12.make_scene(1620, tmp_path_factory.mktemp("white_installed"), preinstalled_end_stop=True)[1]
    frames, calibration = stage_v7.capture_detector(session)
    # Truth is reserved for the assertion and creation of an occlusion mask.
    return frames, calibration, session.ctx.obj_pos("end_stop").copy()


def test_white_installed_stop_uses_visible_cad_instead_of_a_plate_fragment(white_installed):
    frames, calibration, truth = white_installed
    result = cad_rgbd_v12.estimate_scene(frames, calibration)
    row = result["objects"]["end_stop"]
    assert result["color_identity_used"] is False
    assert result["simulator_pose_used"] is False
    assert row["valid"]
    assert np.linalg.norm(np.asarray(row["position_m"]) - truth) < .004
    assert row["geometry_agreement"]["matched_samples"] >= 35


def test_randomized_rgb_cannot_change_depth_cad_pose(white_installed):
    frames, calibration, _ = white_installed
    original = cad_rgbd_v12.estimate_scene(frames, calibration, {"end_stop": cad_rgbd_v12.templates()["end_stop"]})
    changed = deepcopy(frames)
    for frame in changed.values():
        frame["rgb"] = np.random.default_rng(901).integers(0, 256, frame["rgb"].shape, dtype=np.uint8)
    result = cad_rgbd_v12.estimate_scene(changed, calibration, {"end_stop": cad_rgbd_v12.templates()["end_stop"]})
    assert result["objects"]["end_stop"] == original["objects"]["end_stop"]


def test_depth_loss_returns_unknown_without_expected_pose_fill(white_installed):
    frames, calibration, _ = white_installed
    changed = deepcopy(frames)
    for frame in changed.values(): frame.get("depth_m", frame.get("depth_mm"))[:] = 0
    result = cad_rgbd_v12.estimate_scene(changed, calibration)
    assert all(not r["valid"] and r["position_m"] is None for r in result["objects"].values())
    assert not result["fixtures"]["guide_base"]["valid"]
    assert result["assembly_targets"] == {}
    assert not result["fixture_relations"]["end_stop_to_base"]["observable"]


def test_partial_boss_occlusion_cannot_produce_a_valid_shifted_wing_pose(white_installed):
    frames, calibration, truth = white_installed
    changed = deepcopy(frames)
    # Controlled missing depth over the observed upper boss, synthesized by
    # the evaluator. The detector receives only the resulting image arrays.
    corners = np.array([[x,y,z] for x in (-.017,.017) for y in (-.020,.020) for z in (.020,.040)]) + truth
    for view, frame in changed.items():
        cal = _as_calibration(calibration[view]); tf = cal.matrix()
        points = (corners - tf[:3,3]) @ tf[:3,:3]; depth = -points[:,2]
        u = cal.fx * points[:,0]/depth + cal.cx; v = -cal.fy * points[:,1]/depth + cal.cy
        x0,x1 = max(0,int(u.min())), min(cal.width,int(u.max())+1)
        y0,y1 = max(0,int(v.min())), min(cal.height,int(v.max())+1)
        frame.get("depth_m", frame.get("depth_mm"))[y0:y1,x0:x1] = 0
    result = cad_rgbd_v12.estimate_scene(changed, calibration, {"end_stop": cad_rgbd_v12.templates()["end_stop"]})
    row = result["objects"]["end_stop"]
    assert not row["valid"] or np.linalg.norm(np.asarray(row["position_m"]) - truth) < .004
