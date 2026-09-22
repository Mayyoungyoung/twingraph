import math
from types import SimpleNamespace

import mujoco
import numpy as np

from simbench.assembly.skills_v12 import end_stop_locator_capture, evaluate_end_stop_fixture
from simbench.value.stage_v12 import PIN_CONFIG, write_scene
from simbench.value.stage_v7 import StageV7Spec
from simbench.core.sim_context import MjContext
from simbench.assembly.control import HOME


def footprint(center, yaw=0.):
    c,s = math.cos(yaw),math.sin(yaw)
    R = np.array([[c,-s,0],[s,c,0],[0,0,1]])
    return np.array([[x,y,-.018] for x in (-.012,.012) for y in (-.043,.043)]) @ R.T + center


def test_printed_posts_capture_supported_stop_without_ideal_pose_gate():
    center = np.array([-.092,0.,.024])
    result = end_stop_locator_capture(center,footprint(center),.4)
    assert result["success"] and result["locator_vertical_overlap_m"] > .010
    assert not end_stop_locator_capture(center,footprint(center),0.)["success"]


def test_nearby_or_lifted_or_crosswise_stop_is_not_functionally_captured():
    for center,yaw in ((np.array([-.13,0.,.024]),0.),
                       (np.array([-.092,.060,.024]),0.),
                       (np.array([-.092,0.,.042]),0.),
                       (np.array([-.092,0.,.024]),math.pi/2)):
        assert not end_stop_locator_capture(center,footprint(center,yaw),.4)["success"]


def test_tilted_stop_can_be_captured_by_two_posts_at_its_supported_end():
    # A rigid 86 mm plate tilted about X: one bottom end touches the base
    # at 6 mm; its opposite end is 0.29 mm above the 17 mm locator tops.
    angle = math.asin(.01129/.086)
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([[1.,0.,0.],[0.,c,-s],[0.,s,c]])
    center = np.array([-.092, -.018*s, .006+.018*c+.043*s])
    corners = footprint(np.zeros(3)) @ rotation.T + center
    result = end_stop_locator_capture(center, corners, .4)
    assert result["success"]
    assert np.isclose(result["highest_bottom_corner_clearance_m"], -.00029)
    blockers = [row for row in result["locator_post_checks"] if row["blocks_translation"]]
    assert len(blockers) == 2
    assert {row["direction_x"] for row in blockers} == {-1, 1}
    assert all(row["post_xy_m"][1] < 0 for row in blockers)
    # Keeping a synthetic support signal cannot rescue absent local overlap.
    lifted = end_stop_locator_capture(center+[0,0,.012], corners+[0,0,.012], .4)
    assert not lifted["success"]
    assert not any(row["blocks_translation"] for row in lifted["locator_post_checks"])


def test_low_corner_on_only_one_x_side_does_not_imply_bidirectional_capture():
    angle = math.asin(.01129/.024)
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([[c,0.,s],[0.,1.,0.],[-s,0.,c]])
    center = np.array([-.092+.018*s, 0., .006+.018*c+.012*s])
    corners = footprint(np.zeros(3)) @ rotation.T + center
    result = end_stop_locator_capture(center, corners, .4)
    assert result["blocked_positive_rail_direction"]
    assert not result["blocked_negative_rail_direction"]
    assert not result["success"]


def test_physical_base_support_is_measured_and_shallow_pins_do_not_claim_base_bridge(tmp_path):
    spec = StageV7Spec.sample(1600)
    spec.supply_yaws_rad["end_stop"] = 0.
    ctx = MjContext(write_scene(spec,tmp_path,preinstalled_end_stop=True),control_freq=50)
    ctx.reset(); ctx.data.qpos[ctx.arm_qadr] = HOME; mujoco.mj_forward(ctx.model,ctx.data)
    ctx.hold_arm(); ctx.set_finger_ctrl(.04)
    for _ in range(50): ctx.step()
    session = SimpleNamespace(ctx=ctx,held=None,pin_insertion_config=PIN_CONFIG)
    ok, metrics = evaluate_end_stop_fixture(session)
    assert ok and metrics["functional_locator_capture"]["supporting_contact_count"] > 0
    assert not metrics["pin_base_bridge_verified"]  # pins remain in supply holders
    assert metrics["pin_base_bridge_is_required"] is False


def test_base_inserted_pins_bridge_only_when_the_same_shafts_engage_the_stop(tmp_path):
    """Initialized evaluator states are unit tests, never robot rollout evidence."""
    spec = StageV7Spec.sample(1600)
    ctx = MjContext(write_scene(spec, tmp_path), control_freq=50)
    ctx.reset()
    ctx.data.qpos[ctx.arm_qadr] = HOME
    mujoco.mj_forward(ctx.model, ctx.data)
    base = ctx.obj_pos("guide_base")
    # Printed CAD: stop bottom meets base top at +6 mm; the stop is 36 mm
    # thick. These 53 mm shafts span both that stop and the 12 mm base plate.
    stop = base + np.array([-.092, 0., .024])
    ctx.set_obj_pose("end_stop", stop[:2], stop[2])
    for pin, y in (("pin_left", -.032), ("pin_right", .032)):
        origin = base + np.array([-.092, y, .039])
        ctx.set_obj_pose(pin, origin[:2], origin[2])
    session = SimpleNamespace(ctx=ctx, held=None, pin_insertion_config=PIN_CONFIG)

    # Preserve the original (invalid) fixture as a negative regression: at
    # base+.039 the shaft spans both holes, but its head is buried 4 mm into
    # the stop plate. The new contact evaluator correctly rejects this state.
    _, buried = evaluate_end_stop_fixture(session)
    assert not buried["pin_base_bridge_verified"]
    for pin in ("pin_left", "pin_right"):
        check = buried["pin_stop_engagement"][pin]
        assert check["strict_zero_interpenetration"]["success"]
        assert not check["success"]
        assert check["maximum_receiver_contact_penetration_m"] > .003
        assert any(f"{pin}_head" in contact["geoms"] for contact in check["receiver_contacts"])

    # Use the actual head geometry to initialize the valid positive fixture:
    # head underside is body-.001, so it meets stop top (base+.042) at
    # body=base+.043. The shaft still occupies 10 mm of the 12 mm base bore.
    stop_plate_top = stop[2] + .018
    for pin, y in (("pin_left", -.032), ("pin_right", .032)):
        head = ctx.model.geom(f"{pin}_head")
        head_bottom_offset = float(head.pos[2] - head.size[1])
        origin = base + np.array([-.092, y, 0.])
        origin[2] = stop_plate_top - head_bottom_offset
        ctx.set_obj_pose(pin, origin[:2], origin[2])

    _, connected = evaluate_end_stop_fixture(session)
    assert connected["pin_base_bridge_verified"]
    for pin, hole in (("pin_left", 0), ("pin_right", 1)):
        assert connected["pin_stop_engagement"][pin]["success"]
        assert connected["pin_stop_engagement"][pin]["strict_zero_interpenetration"]["success"]
        assert connected["pin_stop_engagement"][pin]["maximum_receiver_contact_penetration_m"] < 1e-10
        assert connected["pin_base_bridge_diagnostic"][pin][hole]["success"]

    # Preserve both pins in their base holes but move the stop clear of them.
    # A test of the base alone would falsely claim the components are joined.
    displaced = stop + np.array([.025, 0., 0.])
    ctx.set_obj_pose("end_stop", displaced[:2], displaced[2])
    _, disconnected = evaluate_end_stop_fixture(session)
    assert not disconnected["pin_base_bridge_verified"]
    for pin, hole in (("pin_left", 0), ("pin_right", 1)):
        assert disconnected["pin_base_bridge_diagnostic"][pin][hole]["success"]
        assert not disconnected["pin_stop_engagement"][pin]["success"]
