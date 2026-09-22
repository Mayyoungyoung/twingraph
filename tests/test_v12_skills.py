from types import SimpleNamespace
import numpy as np
import pytest

from simbench.assembly.skills_v12 import control_position, functional_geometry, visual_position, functional_stroke_targets
from simbench.value.stage_v5 import stage_calls


def test_old_video_local_handle_is_engaged_but_requires_functional_test():
    carriage = np.array([.1060092763, .0850864435, .8239811481])
    handle = np.array([.1066018268, .0851285937, .8740819164])
    ok, metrics = functional_geometry("handle", handle, carriage + [0, 0, .048], handle-carriage)
    assert ok and metrics["post_ring_overlap_m"] > .004
    # Moving the ring outside the real printed post clearance or raising it
    # above the post must fail even if a nominal target is supplied.
    for shift in ([.002, 0, 0], [0, 0, .030]):
        p = handle + shift
        assert not functional_geometry("handle", p, p, p-carriage)[0]


def test_visual_control_never_falls_back_to_body_truth():
    class Trap:
        def obj_pos(self, part):
            raise AssertionError("oracle body position accessed")
        def eef_pos(self):
            return np.array([.2, .3, .9])
        def eef_mat(self):
            return np.eye(3)
    s = SimpleNamespace(strict_rgbd_v12=True, ctx=Trap(), held="pin_left",
        held_visual_transforms_v12={"pin_left": {"local_position": np.array([0., 0., -.04]),
            "local_rotation": np.eye(3), "source": "rgbd_at_grasp_plus_encoder_forward_kinematics"}},
        decision_observation={"backend": "rgbd_geometry", "objects": {}})
    np.testing.assert_allclose(control_position(s,"pin_left"), [.2,.3,.86])
    s.held = None
    with pytest.raises(ValueError, match="no valid RGB-D"):
        visual_position(s, "pin_left")


def test_direct_strict_pose_call_cannot_bypass_detector():
    from simbench.assembly.library import Session
    class Trap:
        def obj_pose(self, part):
            raise AssertionError("oracle pose accessed")
    s = SimpleNamespace(strict_rgbd_v12=True, ctx=Trap(), decision_observation=None,
                        observations={"pin_left": [np.zeros(3)]})
    result = Session.estimate_pose(s, "pin_left")
    assert not result.ok and "RGB-D" in result.reason
    s.decision_observation = {"backend": "rgbd_geometry", "objects": {
        "pin_left": {"position_m": [0,0,0], "valid": True, "quat_wxyz": None}}}
    assert not Session.estimate_pose(s,"pin_left").ok


def test_v12_plan_has_explicit_learning_and_reasoned_limits():
    choice = dict(yaw=0., height=0., clearance=1., force=5., speed=.0043,
                  force_limit=9.1, press_force=1.9)
    calls = stage_calls("pin_left", [-.01,.05,.855], choice, 0, v7=True, v12=True)
    insert = next(c for c in calls if c.skill == "insert")
    assert insert.arguments["strategy"].value == "learned"
    contact = next(c for c in calls if c.skill == "plan_path" and "force_limit" in c.arguments)
    assert contact.arguments["force_limit"].value == 9.1
    assert contact.arguments["speed"].value == .0043
    assert next(c for c in calls if c.skill == "press").arguments["force_stop"].value == 1.9
    assert not any(c.skill == "orient_wrist" for c in calls)
    assert calls[-1].skill == "move" and calls[-1].arguments["target"].value == "home"
    legacy = stage_calls("pin_left", [-.01,.05,.855], choice, 0, v7=True)
    assert legacy[-1].skill == "inspect"


def test_functional_push_requires_real_bidirectional_progress():
    from simbench.assembly.library import Session
    for x in (.035, .06, .106, .135):
        a,b = functional_stroke_targets(x, .020, (.035,.135))
        assert .035 <= a <= .135 and .035 <= b <= .135
        assert (a-x)*(b-a) < 0 and min(abs(a-x),abs(b-a)) >= .022
    s = SimpleNamespace(functional_acceptance_v12=True, stroke=[0,.08,.075],
        stroke_runs=[dict(start_x=0,end_x=.08),dict(start_x=.08,end_x=.075)], stroke_peak_forces=[2.])
    assert not Session.verify_stroke(s, minimum=.020).ok
    s.stroke_runs[1]["end_x"] = .05
    assert Session.verify_stroke(s, minimum=.020).ok


def test_real_square_hole_accepts_corner_but_rejects_wall_and_excessive_tilt():
    from dataclasses import replace
    from simbench.value.pin_geometry import PinInsertionConfig, insertion_geometry
    square = PinInsertionConfig(guide_inner_radius_m=.004, plate_hole_half_width_m=.004,
        guide_length_m=.036, required_depth_m=.006, radial_clearance_m=0., aperture_shape="square")
    def check(origin, axis=(0,0,1), config=square):
        return insertion_geometry(origin, axis, [0,0,0], [0,0,1], config)["inserted"]
    assert check([.00065,.00065,.02])
    assert not check([.00065,.00065,.02], config=replace(square,aperture_shape="circular"))
    assert not check([.00071,0,.02])
    for angle, expected in ((5,True),(15,False)):
        a=np.deg2rad(angle); axis=np.array([np.sin(a),0,np.cos(a)])
        assert check(axis*.02, axis) == expected


def test_functional_pin_requires_real_contiguous_internal_occupancy():
    from dataclasses import replace
    from simbench.value.pin_geometry import PinInsertionConfig, insertion_geometry
    cfg = PinInsertionConfig(guide_inner_radius_m=.004, plate_hole_half_width_m=.004,
        guide_length_m=.036, required_depth_m=.006, radial_clearance_m=0.,
        aperture_shape="square", acceptance_mode="functional_contiguous")
    def verdict(origin, axis=(0,0,1), config=cfg):
        return insertion_geometry(origin, axis, [0,0,0], [0,0,1], config)
    assert not verdict([0,0,.047])["inserted"]  # Tip only touches the plate.
    assert not verdict([0,0,.044])["inserted"]  # Three millimeters is shallow.
    assert not verdict([.002,0,.02])["inserted"]  # Shaft outside real hole.
    angle=np.deg2rad(3); axis=np.array([np.sin(angle),0,np.cos(angle)])
    row=verdict(np.array([.0009,0,0])+axis*.02,axis)
    assert row["inserted"] and not row["mouth_continuity_pass"]
    assert row["longest_contiguous_depth_span_m"] >= .006
    from simbench.value.pin_geometry import evaluate_pin_state
    for released, touching, expected in ((False, True, False), (True, True, False), (True, False, True)):
        state = evaluate_pin_state([0,0,.02], [0,0,1], [0,0,0], [0,0,1],
                                   released, touching, config=cfg)
        assert state["inserted_after_release"] == expected
    # Two legal short regions separated by an illegal neck cannot add up
    # to a functional insertion interval.
    gap=replace(cfg, aperture_shape="circular", guide_length_m=.008,
                bore_profile=((.003,.005),(.005,.0035),(.008,.005)))
    row=verdict([.001,0,.02],config=gap)
    assert len(row["valid_contiguous_depth_intervals_m"]) == 2
    assert not row["inserted"]


def test_split_withdrawal_consumes_same_branch_without_repeating_initial_descent():
    from simbench.assembly.grasp_v12 import withdrawal_slice
    grasp=dict(approach_joints_v12=[np.zeros(7)], lift_joints_v12=[np.full(7,v) for v in np.linspace(.025,.1,4)],
               withdrawal_height_m=.1)
    first,progress=withdrawal_slice(grasp,.017)
    np.testing.assert_allclose(first[-1],.017)
    grasp["withdrawal_progress_m"]=progress
    second,progress=withdrawal_slice(grasp,.083)
    assert all(np.all(q>.017) for q in second)
    np.testing.assert_allclose(second[-1],.1)
    assert np.isclose(progress,.1)
    with pytest.raises(ValueError,match="exceeds"):
        withdrawal_slice(grasp,.1)


def test_ring_release_requires_real_engagement_and_rechecks_after_settling():
    from simbench.assembly.library import Session,Result
    def trial(position, after=None, v12=True):
        current=np.asarray(position,float); opened=[]
        def inspect(part,target,tol):
            ok,metrics=functional_geometry("handle",current,[0,0,.048],current)
            return Result(ok,metrics)
        def hold(seconds):
            if after is not None: current[:]=after
        s=SimpleNamespace(functional_acceptance_v12=v12,inspect_seat=inspect,
            external_force=lambda part:0.,held="handle",stage_passes={},
            ctx=SimpleNamespace(body_id=lambda part:1,data=SimpleNamespace(contact=[]),pad_span=lambda:.08),
            call=lambda skill:opened.append(skill),hold=hold)
        result=Session.place_object(s,"handle",[0,0,.048])
        return result,opened
    for bad in ([.002,0,.048],[0,0,.1]):
        result,opened=trial(bad)
        assert not result.ok and not opened
    result,opened=trial([0,0,.048])
    assert result.ok and opened==["open_gripper"] and result.metrics["support_force_n"]==0.
    result,opened=trial([0,0,.048],after=[.002,0,.048])
    assert not result.ok and opened==["open_gripper"]
    result,opened=trial([0,0,.048],v12=False)
    assert not result.ok and not opened  # Legacy support rule remains intact.
