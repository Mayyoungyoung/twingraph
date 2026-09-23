"""V17-A stable-supported end-stop placement: acceptance and binding tests.

Artificial states here verify only what the local acceptance predicate means;
they never count as robot placement success rates.
"""
from types import SimpleNamespace
import numpy as np
import pytest
import mujoco

from simbench.assembly.skills_v12 import (EndStopStablePlacementConfig,
    evaluate_end_stop_stable_support)
from simbench.assembly.library import Session
from simbench.value.stage_v5 import stage_calls
from simbench.value import system_v12
from simbench.assembly.library import SkillFailure

STOP_BODY, BASE_BODY, FINGER_BODY = 5, 6, 7


def stable_session(position, contacts, forces, *, drop_after=None, held=None):
    """Fake physics state; contacts/forces are software fixtures for unit tests."""
    state = dict(position=np.asarray(position, float).copy(), steps=0)

    def step():
        state["steps"] += 1
        if drop_after is not None and state["steps"] > drop_after:
            state["position"] = state["position"] + np.array([0., 0., -.05])

    geom_bodyid = np.zeros(2 * len(contacts) + 1, dtype=int)
    rows = []
    for i, contact in enumerate(contacts):
        geom_bodyid[2 * i] = contact["a"]
        geom_bodyid[2 * i + 1] = contact["b"]
        rows.append(SimpleNamespace(geom1=2 * i, geom2=2 * i + 1,
            frame=np.array([0., 0., 1., 0., 1., 0., 1., 0., 0.]),
            pos=np.asarray(contact["pos"], float), dist=float(contact["dist"])))
    names = {STOP_BODY: "end_stop", BASE_BODY: "guide_base", FINGER_BODY: "panda_finger_l"}
    model = SimpleNamespace(geom_bodyid=geom_bodyid,
        body=lambda bid: SimpleNamespace(name=names[int(bid)]))
    ctx = SimpleNamespace(model=model, data=SimpleNamespace(contact=rows),
        body_id=lambda part: {"end_stop": STOP_BODY, "guide_base": BASE_BODY}[part],
        obj_pos=lambda part: state["position"].copy(), control_dt=.02, step=step)
    session = SimpleNamespace(ctx=ctx, held=held,
        end_stop_stable_placement_config_v17a=EndStopStablePlacementConfig())
    original = mujoco.mj_contactForce

    def fake_contact_force(model_arg, data_arg, index, force):
        force[0] = float(forces[index])
    mujoco.mj_contactForce = fake_contact_force
    return session, state, original


@pytest.fixture(autouse=True)
def restore_contact_force(request):
    yield


def contact_rows(position):
    return [dict(a=STOP_BODY, b=BASE_BODY, pos=[position[0], position[1], position[2] - .01], dist=-.0001)], [.8]


def run(position, contacts, forces, *, after_retreat=True, **kwargs):
    session, state, original = stable_session(position, contacts, forces, **kwargs)
    try:
        return evaluate_end_stop_stable_support(session, "end_stop", [0., 0., 0.],
                                                after_retreat=after_retreat)
    finally:
        mujoco.mj_contactForce = original


def test_supported_inside_region_above_old_seating_band_passes():
    contacts, forces = contact_rows([.012, .015, .011])
    ok, metrics = run([.012, .015, .011], contacts, forces)
    assert ok and metrics["success"]
    assert metrics["supporting_bodies"] == ["guide_base"]
    assert metrics["gripper_released"] and not metrics["gripper_still_supporting"]
    assert metrics["retained_entire_window"] and metrics["window_samples"] == 10
    assert metrics["observation_window_s"] == pytest.approx(.5)


def test_pose_residual_alone_is_not_a_gate():
    # 16/15/11 mm off the nominal target, resting on the locator tops:
    # outside every legacy residual gate, inside the declared region.
    contacts, forces = contact_rows([.012, .015, .011])
    ok, metrics = run([.012, .015, .011], contacts, forces)
    assert ok
    assert metrics["region_ok"]


def test_floating_or_gripper_carried_fails():
    ok, metrics = run([.012, .015, .011], [], [])
    assert not ok and not metrics["supported"]
    finger = [dict(a=FINGER_BODY, b=STOP_BODY, pos=[0., 0., .02], dist=0.)]
    ok, metrics = run([.012, .015, .011], finger, [.6])
    assert not ok and not metrics["supported"] and metrics["finger_support_force_n"] > 0.


def test_gripper_still_supporting_after_retreat_fails():
    both = [dict(a=STOP_BODY, b=BASE_BODY, pos=[0., 0., .001], dist=-.0001),
            dict(a=FINGER_BODY, b=STOP_BODY, pos=[0., 0., .02], dist=0.)]
    ok, metrics = run([.012, .015, .011], both, [.8, .6], after_retreat=True)
    assert not ok and metrics["gripper_still_supporting"]
    ok, metrics = run([.012, .015, .011], both, [.8, .6], after_retreat=False)
    assert ok


def test_outside_declared_region_fails():
    contacts, forces = contact_rows([.05, 0., .011])
    ok, metrics = run([.05, 0., .011], contacts, forces)
    assert not ok and not metrics["region_ok"]


def test_loss_of_support_inside_window_fails():
    contacts, forces = contact_rows([.012, .015, .011])
    ok, metrics = run([.012, .015, .011], contacts, forces, drop_after=3)
    assert not ok and not metrics["retained_entire_window"]
    assert any(not row["region_ok"] for row in metrics["window_rows"][1:])


def test_gross_penetration_fails():
    contacts = [dict(a=STOP_BODY, b=BASE_BODY, pos=[0., 0., .001], dist=-.003)]
    ok, metrics = run([.012, .015, .011], contacts, [.8])
    assert not ok and metrics["gross_penetration"]


def test_place_acceptance_guard():
    assert not Session.place_object(None, "end_stop", [0., 0., 0.], acceptance="bogus").ok
    assert "only defined for end_stop" in Session.place_object(
        SimpleNamespace(), "carriage", [0., 0., 0.], acceptance="stable_supported").reason
    assert "requires the explicit" in Session.place_object(
        SimpleNamespace(), "end_stop", [0., 0., 0.], acceptance="stable_supported").reason


def stop_choice():
    return dict(yaw=0., height=.004, clearance=1., force=5., speed=.004,
                press_force=2., placement_yaw=0.)


def test_stable_mode_compiles_local_flow_and_skips_corridor_precheck():
    calls = stage_calls("end_stop", [0., 0., .836], stop_choice(), 1, v7=True, v12=True,
                        assembly_target=dict(position_m=[0., 0., .836]), end_stop_stable=True)
    place = next(c for c in calls if c.skill == "place")
    assert place.arguments["acceptance"].value == "stable_supported"
    final = [c for c in calls if c.skill == "inspect"][-1]
    assert final.arguments["what"].value == "stable_supported"
    assert not any(c.skill == "move" and c.arguments.get("target") is not None
                   and c.arguments["target"].value == "home" for c in calls)
    assert not any("what" in c.arguments and c.arguments["what"].value == "receiver_relation"
                   for c in calls)


def test_legacy_mode_keeps_pose_acceptance_home_and_corridor_precheck():
    calls = stage_calls("end_stop", [0., 0., .836], stop_choice(), 1, v7=True, v12=True,
                        assembly_target=dict(position_m=[0., 0., .836]))
    place = next(c for c in calls if c.skill == "place")
    assert place.arguments["acceptance"].value == "pose"
    final = [c for c in calls if c.skill == "inspect"][-1]
    assert final.arguments["what"].value == "receiver_relation"
    assert any(c.skill == "move" and c.arguments.get("target") is not None
               and c.arguments["target"].value == "home" for c in calls)


def prefix_session(tmp_path, *, stable):
    return SimpleNamespace(decision_observation={"sha256": "test"},
        planning_cad={"functional_stroke_minimum_m": .02},
        out=tmp_path, receiver_binding_history=[{}],
        **({"end_stop_place_acceptance_v12": "stable_supported"} if stable else {}))


def run_prefix(monkeypatch, tmp_path, *, stable):
    monkeypatch.setattr(system_v12, "_prepare_receiver_targets", lambda *a, **k: {})
    monkeypatch.setattr(system_v12, "assembly_program",
        lambda s, p: SimpleNamespace(calls=[], to_dict=lambda: dict(calls=[])))
    monkeypatch.setattr(system_v12, "execute_calls", lambda s, p, c: None)
    return system_v12.run_mechanism_prefix(prefix_session(tmp_path, stable=stable),
        stop_after="end_stop", order=["carriage", "end_stop", "pin_left", "pin_right", "handle"],
        choices={}, stroke_minimum=.02)


def test_mechanism_prefix_skips_corridor_only_in_stable_mode(monkeypatch, tmp_path):
    result = run_prefix(monkeypatch, tmp_path, stable=True)
    assert result.ok and result.metrics["stop_after"] == "end_stop"
    with pytest.raises(SkillFailure):
        run_prefix(monkeypatch, tmp_path, stable=False)
