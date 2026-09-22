"""Eye-in-hand acquisition: calibration, provenance and safe motion boundaries."""
from copy import deepcopy

import mujoco
import numpy as np
import pytest

from simbench.value import stage_v7, stage_v12, wrist_rgbd_v12 as wrist


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    return stage_v12.make_scene(1600, tmp_path_factory.mktemp("wrist_rgbd"))[1]


def test_camera_is_fixed_to_hand_and_fk_matches_render_transform(session):
    ctx = session.ctx
    cid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_CAMERA, wrist.CAMERA_NAME)
    bid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, wrist.MOUNT_BODY)
    assert ctx.model.cam_bodyid[cid] == bid
    transform = wrist.calibration_from_encoders(ctx).matrix()
    assert np.allclose(transform[:3, 3], ctx.data.cam_xpos[cid], atol=1e-12)
    assert np.allclose(transform[:3, :3], ctx.data.cam_xmat[cid].reshape(3, 3), atol=1e-12)


def test_home_capture_uses_only_wrist_and_does_not_advance_physics(session, monkeypatch):
    monkeypatch.setattr(stage_v7, "capture_vision", lambda *a, **k: pytest.fail("external camera used"))
    before_time = float(session.ctx.data.time)
    before_qpos = session.ctx.data.qpos.copy()
    frames, calibrations = stage_v7.capture_detector(session)
    assert list(frames) == [wrist.CAMERA_NAME] == list(calibrations)
    assert float(session.ctx.data.time) == before_time
    assert np.array_equal(session.ctx.data.qpos, before_qpos)
    assert not session.last_rgbd_acquisition["scan_motion"]
    assert frames[wrist.CAMERA_NAME]["rgb"].shape == (720, 960, 3)


def test_fk_calibration_ignores_object_poses(session):
    ctx = session.ctx
    previous = ctx.data.qpos.copy()
    baseline = wrist.calibration_from_encoders(ctx).matrix()
    try:
        for part in session.parts:
            bid = ctx.body_id(part); jid = ctx.model.body_jntadr[bid]
            qa = ctx.model.jnt_qposadr[jid]
            ctx.data.qpos[qa:qa+3] += [.4, -.3, .7]
        assert np.array_equal(wrist.calibration_from_encoders(ctx).matrix(), baseline)
    finally:
        ctx.data.qpos[:] = previous
        mujoco.mj_forward(ctx.model, ctx.data)


def test_held_object_never_triggers_view_motion(session, monkeypatch):
    before = session.ctx.data.qpos.copy()
    previous_held = session.held
    try:
        session.ctx.data.qpos[session.ctx.arm_qadr[0]] += .04
        mujoco.mj_forward(session.ctx.model, session.ctx.data)
        session.held = "handle"
        monkeypatch.setattr(session.arm, "execute_joint", lambda *a, **k: pytest.fail("held camera moved robot"))
        info = wrist._prepare_view(session)
        assert info["scan_status"] == "held_object_current_view_only"
        assert not info["scan_motion"]
    finally:
        session.held = previous_held
        session.ctx.data.qpos[:] = before
        mujoco.mj_forward(session.ctx.model, session.ctx.data)


def test_initial_objects_localized_without_external_view(session):
    observation = session.decision_observation
    assert observation["views"] == [wrist.CAMERA_NAME]
    assert observation["acquisition"]["external_camera_used"] is False
    assert observation["acquisition"]["object_pose_used"] is False
    for part, row in observation["objects"].items():
        assert row["source_view"] == wrist.CAMERA_NAME
        assert row["valid"], part
        # Ground truth only scores this independent sensor test.
        assert np.linalg.norm(np.asarray(row["position_m"])-session.ctx.obj_pos(part)) < .004


def test_observation_hash_excludes_wall_clock_and_scan_diagnostics(session, monkeypatch):
    frames, calibrations = stage_v7.capture_detector(session)
    acquisition = deepcopy(session.last_rgbd_acquisition)
    original_observation = deepcopy(session.decision_observation)
    clock = [1.]
    def fixed_frame(s, size):
        full = dict(acquisition, wall_seconds=clock[0],
                    collision_check={"valid": True, "wall_seconds": clock[0]*2., "samples": 40})
        s.last_rgbd_acquisition = full
        s.artifacts.setdefault("perception_acquisitions", []).append(deepcopy(full))
        return frames, calibrations
    monkeypatch.setattr(session, "capture_detector_fn", fixed_frame)
    try:
        first = stage_v7.install_visual(session, parts=session.parts)
        clock[0] = 900.
        second = stage_v7.install_visual(session, parts=session.parts)
        assert first["sha256"] == second["sha256"]
        assert first["acquisition"] == second["acquisition"]
        assert "wall_seconds" not in second["acquisition"]
        assert "collision_check" not in second["acquisition"]
        assert session.last_rgbd_acquisition["wall_seconds"] == 900.
        assert session.artifacts["perception_acquisitions"][-1]["collision_check"]["wall_seconds"] == 1800.
    finally:
        session.set_decision_observation(original_observation)
        session.last_rgbd_acquisition = acquisition


def test_observation_return_uses_registered_move_atom(session, monkeypatch):
    previous = session.ctx.data.qpos.copy()
    calls = []
    try:
        session.ctx.data.qpos[session.ctx.arm_qadr[0]] += .04
        mujoco.mj_forward(session.ctx.model, session.ctx.data)
        monkeypatch.setattr(session.arm, "check_joint_path", lambda *a, **k: {"valid": True})
        def registered(name, **kwargs):
            calls.append((name, kwargs))
            from simbench.assembly.library import Result
            return Result(True)
        monkeypatch.setattr(session, "call", registered)
        info = wrist._prepare_view(session)
        assert calls == [("move", {"target": "home"})]
        assert info["scan_motion"]
    finally:
        session.ctx.data.qpos[:] = previous
        mujoco.mj_forward(session.ctx.model, session.ctx.data)


def test_assembled_handle_and_carriage_are_visible(tmp_path):
    # Initialize an independent CAD assembly fixture, not an execution result.
    s = stage_v12.make_scene(1600, tmp_path / "assembled_visibility", preinstalled_end_stop=True)[1]
    center = np.asarray(s.stage_targets["carriage"], float)
    s.ctx.set_obj_pose("carriage", center[:2], center[2], yaw=0.)
    handle = center + np.array([0., 0., .048])
    s.ctx.set_obj_pose("handle", handle[:2], handle[2], yaw=0.)
    for _ in range(30): s.ctx.step()
    observation = stage_v7.refresh_visual_observation(s, parts=s.parts)
    for part in ("carriage", "handle", "end_stop"):
        row = observation["objects"][part]
        assert row["valid"], part
        assert np.linalg.norm(np.asarray(row["position_m"])-s.ctx.obj_pos(part)) < .004
