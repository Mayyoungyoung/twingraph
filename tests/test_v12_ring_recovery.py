"""Unit safety/observation tests; fake sessions are not physical evidence."""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from simbench.assembly import ring_insertion_v12 as ring


def observation(sigma=.0001):
    return dict(backend="rgbd_geometry", objects={part:dict(valid=True, position_m=[0,0,0],
        fit_residual_m=sigma) for part in ("handle", "carriage")})


def cad():
    return dict(handle_bore_radius_m=.0061, handle_post_radius_m=.0055,
                parts=dict(handle=dict(dimensions_m=[.042,.042,.016])))


def test_search_radius_depends_on_uncertainty_and_never_exceeds_three_mm():
    small = ring.search_parameters(observation(), cad())
    large = ring.search_parameters(observation(.02), cad())
    assert 0 < small["radius_m"] < large["radius_m"] == .003
    assert small["entry_encoder_descent_m"] == .004


def test_command_bounds_preserve_cad_height_and_visual_neighborhood():
    desired = ring.bounded_target([.02, -.02, -.1], np.array([0.,0.,.005]), np.zeros(2), .002, .001)
    assert np.linalg.norm(desired[:2]) <= .002+1.e-12
    assert desired[2] >= .001


def test_no_simulator_object_pose_reads_in_control_module():
    tree = ast.parse(Path(ring.__file__).read_text())
    forbidden = {"obj_pos", "obj_pose", "obj_axis", "obj_quat", "xpos", "xquat", "xmat", "qpos"}
    assert not [node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute) and node.attr in forbidden]


def fake_session(tmp_path):
    policy = tmp_path/"policy.npz"; policy.write_bytes(b"unit fixture only")
    position = np.array([0.,0.,.01])
    actions = []
    ctx = SimpleNamespace(control_dt=.02, eef_pos=lambda:position.copy(),
        grasp_contacts=lambda part:dict(held=True))
    def servo(command, rotation):
        actions.append(np.asarray(command).copy())
        position[:] = command
    session = SimpleNamespace(held="handle", strict_rgbd_v12=True,
        held_visual_transforms_v12=dict(handle=dict(source="rgbd_at_grasp_plus_encoder_forward_kinematics")),
        decision_observation=observation(), planning_cad=cad(), stage_targets=dict(handle=[0.,0.,0.]),
        insertion_policy_v12=str(policy), ctx=ctx, arm=SimpleNamespace(rotation=np.eye(3), servo=servo),
        artifacts={}, external_force=lambda part:0.)
    return session, position, actions


def test_nonvisual_session_fails_before_control(tmp_path):
    session, _, actions = fake_session(tmp_path)
    session.strict_rgbd_v12 = False
    with pytest.raises(ValueError, match="fallback"):
        ring.execute(session, "handle", 0., 1.)
    assert not actions


@pytest.mark.parametrize("accepted", [True,False])
def test_force_stop_ends_control_and_independent_seating_decides_success(tmp_path, monkeypatch,accepted):
    from simbench.assembly import skills_v12, sensor_learning_v12
    session, position, actions = fake_session(tmp_path)
    evaluation_calls = []
    monkeypatch.setattr(skills_v12, "control_position", lambda *args:position.copy())
    monkeypatch.setattr(sensor_learning_v12, "load_actor", lambda path:lambda obs:np.array([0.,0.,-1.]))
    def evaluate(*args):
        evaluation_calls.append(len(actions))
        return accepted, {"unit_fixture": True}
    monkeypatch.setattr(skills_v12, "evaluate_functional_seat", evaluate)
    session.external_force = lambda part:1.5 if actions else 0.
    result = ring.execute(session, "handle", 0., 1.)
    assert len(actions) == 1 and evaluation_calls == [1]
    assert result["success"] is accepted and result["stop_reason"] == "force_stop_reached"
    assert result["independent_functional_success"] is accepted
    assert result["peak_force_n"]==1.5 and result["force_stop_n"]==1.
    assert result["force_stop_overshoot_n"]==.5 and result["force_stop_crossed"] is True
    assert result["hard_force_limit_n"] is None


def test_declared_target_cannot_be_silently_lowered(tmp_path):
    session, _, actions = fake_session(tmp_path)
    with pytest.raises(ValueError, match="declared CAD"):
        ring.execute(session, "handle", -.002, 1.)
    assert not actions


def test_search_starts_before_contact_admittance_unloads(tmp_path, monkeypatch):
    from simbench.assembly import skills_v12, sensor_learning_v12
    session, position, actions = fake_session(tmp_path)
    monkeypatch.setattr(skills_v12, "control_position", lambda *args:position.copy())
    monkeypatch.setattr(sensor_learning_v12, "load_actor", lambda path:lambda obs:np.clip(obs[:3], -1., 1.))
    monkeypatch.setattr(skills_v12, "evaluate_functional_seat", lambda *args:(False, {"unit_fixture": True}))
    session.external_force = lambda part:.12
    result = ring.execute(session, "handle", 0., 1.)
    assert result["trace"][0]["mode"] == "bounded_contact_spiral"
    assert result["search"]["trigger_force_n"] < result["search"]["regulated_contact_force_n"]
    assert result["success"] is False


def test_sensor_engagement_stops_before_unnecessary_bottom_press(tmp_path, monkeypatch):
    from simbench.assembly import skills_v12, sensor_learning_v12
    session, position, actions = fake_session(tmp_path)
    monkeypatch.setattr(skills_v12, "control_position", lambda *args:position.copy())
    monkeypatch.setattr(sensor_learning_v12, "load_actor", lambda path:lambda obs:np.array([0.,0.,-1.]))
    monkeypatch.setattr(skills_v12, "evaluate_functional_seat", lambda *args:(True, {"unit_fixture": True}))
    session.external_force = lambda part:.3 if not actions else 0.
    result = ring.execute(session, "handle", 0., 1.)
    assert result["stop_reason"] == "functional_encoder_feed_completed"
    assert result["entry_inferred_from_encoder_and_force"] is True
    assert 0. < position[2] <= .002
    assert result["peak_force_n"] <= result["force_stop_n"]


@pytest.mark.parametrize("fault", ["grasp","sensor"])
def test_invalid_sensor_or_lost_grasp_cannot_be_overridden_by_seat_label(tmp_path,monkeypatch,fault):
    from simbench.assembly import skills_v12,sensor_learning_v12
    session,position,actions=fake_session(tmp_path)
    monkeypatch.setattr(skills_v12,"control_position",lambda *a:position.copy())
    monkeypatch.setattr(sensor_learning_v12,"load_actor",lambda p:lambda o:np.zeros(3))
    monkeypatch.setattr(skills_v12,"evaluate_functional_seat",lambda *a:(True,{}))
    if fault=="grasp": session.ctx.grasp_contacts=lambda p:dict(held=False)
    else: session.external_force=lambda p:float("nan")
    result=ring.execute(session,"handle",0.,1.)
    assert result["success"] is False and result["safety_ok"] is False and not actions
