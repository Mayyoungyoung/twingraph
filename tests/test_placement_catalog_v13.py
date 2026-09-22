"""Pure declared geometry contracts; no controller, training, or rollout."""
import copy
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from simbench.assembly.placement_catalog_v13 import _down, _pad_corners, placement_clearance_catalog


def pose(xyz, yaw=0.):
    q = Rotation.from_euler("z", yaw).as_quat()
    return dict(valid=True, position_m=list(xyz), quat_wxyz=q[[3, 0, 1, 2]].tolist(), fit_residual_m=.0005)


def fixture():
    objects = {p: pose([-.4+i*.07, -.25, .85]) for i, p in enumerate(("carriage", "end_stop", "pin_left", "pin_right", "handle"))}
    targets = {p: pose([.1+i*.03, .08, .84]) for i, p in enumerate(objects)}
    cad = dict(collision_primitives={p: [dict(name=f"{p}_solid", type="box", pos=[0, 0, 0], size=[.01, .01, .01])]
                for p in (*objects, "guide_base")}, parts={})
    for part, dz, width in (("carriage", .027, .028), ("end_stop", .023, .026), ("handle", 0., .042)):
        cad["parts"][part] = dict(grasp_reference=dict(height_offset_m=dz, width_m=width),
            grasp_region=dict(body_z_interval_m=[-.05, .06], rotational_symmetry=part == "handle",
                closing_axes_body=[dict(axis=[1, 0, 0], width_m=.044 if part == "carriage" else .022 if part == "end_stop" else .042),
                                   dict(axis=[0, 1, 0], width_m=width)]))
    obs = dict(objects=objects, fixtures=dict(guide_base=pose([0., 0., .812])), assembly_targets=targets,
        receiver_geometry=dict(rail=dict(entry_approach_m=[-.12, .08, .87], entry_m=[-.12, .08, .84], axis=[1., 0., 0.])))
    return obs, cad


class Query:
    def __init__(self, primitives, clearance=.02):
        self.primitives = primitives
        self.provenance = dict(software_fixture=True)
        self.clearance = clearance
        self.calls = []

    def query(self, xyz, rotation, gap, receivers):
        self.calls.append((np.asarray(xyz), np.asarray(rotation), gap, copy.deepcopy(receivers)))
        return dict(min_clearance_m=self.clearance, collision=self.clearance < 0,
                    distance_is_lower_bound=False, pairs=[])


def test_down_and_pad_extent_match_unmodified_robot_frames():
    from simbench.assembly.control import down
    for yaw in (0., .3, math.pi/2):
        np.testing.assert_allclose(_down(yaw), down(yaw), atol=1.e-12)
    for corners, direction in _pad_corners():
        z = (_down(0)@corners.T)[2]
        assert z.min() == pytest.approx(-.0044)
        assert z.max() == pytest.approx(.0116)
        assert np.linalg.norm(direction) == pytest.approx(1.)


def test_scene_relative_pickup_placement_and_face_widths():
    obs, cad = fixture()
    obs["objects"]["carriage"] = pose([-.3, -.2, .85], .6)
    obs["assembly_targets"]["carriage"] = pose([.1, .08, .84], .9)
    rows = placement_clearance_catalog(obs, cad, "carriage", height_offsets=(0.,), query_factory=Query)
    assert len(rows) == 2
    assert rows[0]["yaw"] == pytest.approx(.6)
    assert rows[0]["placement_yaw"] == pytest.approx(.9)
    assert rows[1]["grasp_width_m"] == pytest.approx(.044)
    assert rows[0]["grasp_width_m"] == pytest.approx(.028)
    assert all(r["status"] == "necessary_pass" and not r["source_grasp_ik_checked"] for r in rows)
    assert all(r["goal_orientation_residual_rad"] < 1.e-8 for r in rows)


def test_source_body_checked_before_rigid_lift_without_inventing_later_stop():
    obs, cad = fixture()
    queries = []
    def factory(primitives):
        result = Query(primitives); queries.append(result); return result
    rows = placement_clearance_catalog(obs, cad, "carriage", source_axis_offsets=(0.,), height_offsets=(0.,), query_factory=factory)
    assert len(queries) == 3
    assert all("carriage" not in q.primitives for q in queries[:2])
    assert set(queries[2].primitives)=={"carriage"}
    np.testing.assert_allclose(queries[2].calls[-1][0],rows[0]["source_eef_m"])
    assert rows[0]["source_body_checked_phases"]==["source_approach","source_close"]
    future_stop = queries[1].calls[0][3]["end_stop"]
    assert future_stop == obs["objects"]["end_stop"]
    assert rows[0]["future_receiver_pose_modes"]["end_stop"] == "observed"
    assert {"source_approach", "source_close", "source_withdrawal", "descent", "rail_entry", "rail_push", "seat", "opening", "retraction"} <= set(rows[0]["sampled_phases"])


def test_prior_carriage_changes_future_stop_scene_but_not_source_scene():
    obs, cad = fixture()
    queries = []
    def factory(primitives):
        result = Query(primitives); queries.append(result); return result
    row = placement_clearance_catalog(obs, cad, "end_stop", source_axis_offsets=(0.,), height_offsets=(0.,), query_factory=factory)[0]
    assert queries[0].calls[0][3]["carriage"] == obs["objects"]["carriage"]
    assert queries[1].calls[0][3]["carriage"]["position_m"] == obs["assembly_targets"]["carriage"]["position_m"]
    assert row["future_receiver_pose_modes"]["carriage"] == "predicted_mated"


@pytest.mark.parametrize("clearance,status", [(-.001, "rejected"), (.001, "unknown"), (.01, "necessary_pass")])
def test_penetration_and_uncertainty_reserve_are_different(clearance, status):
    obs, cad = fixture()
    row = placement_clearance_catalog(obs, cad, "end_stop", source_axis_offsets=(0.,), height_offsets=(0.,),
        required_clearance_m=.003, query_factory=lambda p: Query(p, clearance))[0]
    assert row["status"] == status
    assert row["min_clearance_m"] == clearance


def test_missing_prior_geometry_is_unknown_but_known_base_collision_still_rejects():
    obs, cad = fixture(); del cad["collision_primitives"]["carriage"]
    for clearance, expected in ((.02, "unknown"), (-.001, "rejected")):
        row = placement_clearance_catalog(obs, cad, "handle", source_axis_offsets=(0.,), height_offsets=(0.,),
            query_factory=lambda p: Query(p, clearance))[0]
        assert row["status"] == expected
        if expected == "unknown":
            assert "receiver CAD missing: carriage" in row["reason"]


def test_pad_face_overlap_uses_extent_instead_of_tcp_center():
    obs, cad = fixture()
    cad["parts"]["handle"]["grasp_region"]["body_z_interval_m"] = [-.008, .008]
    # TCP is outside the ring, but the original pad still overlaps its side.
    row = placement_clearance_catalog(obs, cad, "handle", source_axis_offsets=(0.,), height_offsets=(.010,), query_factory=Query)[0]
    assert row["status"] == "necessary_pass"
    assert row["pad_face_axial_overlap_m"] == pytest.approx(.0024)
    row = placement_clearance_catalog(obs, cad, "handle", source_axis_offsets=(0.,), height_offsets=(.020,), query_factory=Query)[0]
    assert row["status"] == "rejected" and row["samples"] == 0


def test_unknown_pose_does_not_invent_fixed_target_or_executable_angles():
    obs, cad = fixture(); obs["objects"]["carriage"]["valid"] = False
    row = placement_clearance_catalog(obs, cad, "carriage", query_factory=Query)[0]
    assert row["status"] == "unknown" and row["yaw"] is None
    assert not row["executable_parameters_available"]


def test_static_actual_full_gripper_query_can_reject_known_block():
    obs, cad = fixture()
    # Static CAD counterexample, not a robot execution or claimed task result.
    cad["collision_primitives"]["guide_base"] = [dict(name="declared_block", type="box", pos=[0., 0., 0.], size=[.5, .5, .5])]
    row = placement_clearance_catalog(obs, cad, "end_stop", source_axis_offsets=(0.,), height_offsets=(0.,))[0]
    assert row["status"] == "rejected" and row["min_clearance_m"] < 0
    assert all(p["no_simulation_steps"] for p in row["geometry_provenance"].values())


def test_actual_carriage_post_rejects_nominal_source_palm_collision():
    from simbench.assembly.printed_kit import planning_metadata
    from simbench.assembly.gripper_clearance_v12 import GripperClearance
    obs,_=fixture();cad=planning_metadata()
    # Isolate the new source-body necessary condition from receiver geometry.
    def factory(primitives):
        return GripperClearance(primitives) if set(primitives)=={"carriage"} else Query(primitives)
    row=placement_clearance_catalog(obs,cad,"carriage",source_axis_offsets=(0.,),
        height_offsets=(0.,),query_factory=factory)[0]
    assert row["status"]=="rejected" and row["min_clearance_m"]<-.003
    assert row["sampled_worst_phase"] in ("source_approach","source_close")
    assert any(set(p["geoms"])=={"handle_post","hand_collision"} for p in row["worst_pairs"])
    assert row["geometry_provenance"]["source_body"]["no_simulation_steps"]
    assert row["source_body_min_clearance_m"] < 0
    assert row["environment_min_clearance_m"] >= 0


def test_legitimate_closing_pad_touch_is_not_a_stale_source_lift_collision():
    obs,cad=fixture();source=np.asarray(obs["objects"]["carriage"]["position_m"])
    class PadTouch(Query):
        def query(self,xyz,rotation,gap,receivers):
            result=super().query(xyz,rotation,gap,receivers)
            if set(self.primitives)=={"carriage"} and gap<.039:
                result.update(min_clearance_m=-5.e-18,collision=True,pairs=[dict(
                    geoms=["finger1_pad_collision","carriage_solid"],distance_m=-5.e-18,
                    shell_involved=False,contact_world_m=(source+[0.,.014,.03]).tolist())])
            return result
    row=placement_clearance_catalog(obs,cad,"carriage",source_axis_offsets=(0.,),
        height_offsets=(0.,),query_factory=PadTouch)[0]
    assert row["status"]=="necessary_pass"
    assert row["source_body_intentional_pad_pair_count"]>0
    assert row["source_body_min_raw_clearance_m"]<0
    assert row["source_body_checked_phases"]==["source_approach","source_close"]


@pytest.mark.parametrize("geom,depth,z",[("finger1_collision",-.001,.03),
                                        ("finger1_pad_collision",-.001,.03),
                                        ("finger1_pad_collision",0.,.20)])
def test_source_closure_never_exempts_shell_penetration_pad_penetration_or_wrong_face(geom,depth,z):
    obs,cad=fixture();source=np.asarray(obs["objects"]["carriage"]["position_m"])
    class InvalidClosure(Query):
        def query(self,xyz,rotation,gap,receivers):
            result=super().query(xyz,rotation,gap,receivers)
            if set(self.primitives)=={"carriage"} and gap<.039:
                result.update(min_clearance_m=depth,pairs=[dict(geoms=[geom,"carriage_solid"],
                    distance_m=depth,contact_world_m=(source+[0.,.014,z]).tolist())])
            return result
    row=placement_clearance_catalog(obs,cad,"carriage",source_axis_offsets=(0.,),
        height_offsets=(0.,),query_factory=InvalidClosure)[0]
    assert row["status"]=="rejected"
    assert row["source_body_intentional_pad_pair_count"]==0


def test_missing_handle_self_geometry_is_unknown_even_with_other_clearance():
    obs,cad=fixture();del cad["collision_primitives"]["handle"]
    row=placement_clearance_catalog(obs,cad,"handle",source_axis_offsets=(0.,),
        height_offsets=(0.,),query_factory=Query)[0]
    assert row["status"]=="unknown"
    assert "source body CAD missing: handle" in row["reason"]
    assert not row["source_body_checked"] and not row["source_body_geometry_available"]
