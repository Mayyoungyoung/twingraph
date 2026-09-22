"""Pure observed-pose/CAD functional receiver checks; no scene is simulated."""
import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from simbench.assembly.printed_kit import apply_printed_kit, planning_metadata
from simbench.assembly.scene import SCENE
from simbench.value.cad_rgbd_v12 import _hypotheses, _rotate, receiver_interface, shared_shaft_corridor, templates


def row(position, yaw=0.):
    return dict(valid=True,position_m=list(position),quat_wxyz=[math.cos(yaw/2),0.,0.,math.sin(yaw/2)],
                fit_residual_m=0.,geometry_agreement=dict(projected_pixel_size_m=0.))


def installed(yaw=0., translation=(.075,.085,.812)):
    cad=planning_metadata(); R=_rotate(yaw); base=np.asarray(translation)
    stop=base+R@np.asarray(cad["end_stop_mating_pose_in_base"]["position_m"])
    return {"end_stop":row(stop,yaw)}, {"guide_base":row(base,yaw)},cad


def test_two_true_layers_have_a_shared_route_and_sufficient_shaft():
    objects,fixtures,cad=installed()
    result=receiver_interface(objects,fixtures,cad)
    relation=result["fixture_relations"]["end_stop_to_base"]
    assert relation["observable"] and relation["geometric_route_exists"]
    assert relation["contact_or_capture_verified"] is False
    assert relation["requires_guarded_verification"] is True
    for hole in relation["holes"].values():
        assert hole["minimum_total_depth_m"] == pytest.approx(.042)
        assert hole["head_seated_depth_limit_m"] == pytest.approx(.046)
        assert hole["geometric_margin_m"] == pytest.approx(.0007)
        assert hole["stop_to_base_gap_m"] == pytest.approx(0.)
    assert set(result["assembly_targets"])=={"carriage","end_stop","pin_left","pin_right","handle"}


def test_shared_route_is_not_the_stop_hole_centre_or_an_ideal_pose_gate():
    # A shaft centred on the stop would miss the base, yet an off-centre
    # common axis passes both unchanged 8 mm apertures. Use their intersection.
    result=shared_shaft_corridor([.001,0,.036],np.eye(3),[0,0,0],np.eye(3))
    assert result["geometric_route_exists"]
    assert result["common_axis_point_m"] == pytest.approx([.0005,0.,0.])
    assert result["geometric_margin_m"] == pytest.approx(.0002)


def test_disjoint_apertures_cannot_be_called_downstream_feasible():
    result=shared_shaft_corridor([.0015,0,.036],np.eye(3),[0,0,0],np.eye(3))
    assert not result["geometric_route_exists"]


def test_tilt_is_checked_through_the_complete_bore_not_only_at_entry():
    a=math.radians(5.)
    R=np.array([[math.cos(a),0,math.sin(a)],[0,1,0],[-math.sin(a),0,math.cos(a)]])
    result=shared_shaft_corridor([0,0,.036],R,[0,0,0],np.eye(3))
    assert not result["geometric_route_exists"]


def test_unknown_receiver_never_uses_nominal_world_target():
    objects,fixtures,cad=installed()
    fixtures["guide_base"]["valid"]=False
    result=receiver_interface(objects,fixtures,cad)
    assert result["assembly_targets"]=={}
    assert not result["fixture_relations"]["end_stop_to_base"]["observable"]
    objects,fixtures,cad=installed()
    objects["end_stop"]["valid"]=False
    result=receiver_interface(objects,fixtures,cad)
    assert len(result["assembly_targets"])==5  # Goals, not false claims of an observed assembled stop.
    assert not result["fixture_relations"]["end_stop_to_base"]["observable"]


def test_plan_targets_use_the_measured_receiver_not_source_stop_pose():
    objects,fixtures,cad=installed()
    first=receiver_interface(objects,fixtures,cad)
    objects["end_stop"]["position_m"]=[-.24,-.31,.818]
    source=receiver_interface(objects,fixtures,cad)
    assert source["assembly_targets"]==first["assembly_targets"]
    assert not source["fixture_relations"]["end_stop_to_base"]["geometric_route_exists"]


def test_targets_and_two_hole_routes_transform_with_observed_base():
    objects,fixtures,cad=installed(.4,(.12,-.06,.81))
    result=receiver_interface(objects,fixtures,cad)
    R=_rotate(.4);base=np.asarray(fixtures["guide_base"]["position_m"])
    assert result["assembly_targets"]["end_stop"]["position_m"] == pytest.approx(objects["end_stop"]["position_m"])
    assert result["receiver_geometry"]["rail"]["entry_m"] == pytest.approx(base+R@[-.155,0,.012])
    for name,offset in zip(("pin_left","pin_right"),cad["base_hole_offsets_m"]):
        assert result["fixture_relations"]["end_stop_to_base"]["holes"][name]["common_axis_point_m"]==pytest.approx(base+R@offset)


def test_coaxial_but_excessive_layer_separation_exceeds_shaft_reach():
    objects,fixtures,cad=installed()
    objects["end_stop"]["position_m"][2]+=.02
    relation=receiver_interface(objects,fixtures,cad)["fixture_relations"]["end_stop_to_base"]
    assert not relation["geometric_route_exists"]
    assert all(not h["shaft_length_sufficient"] for h in relation["holes"].values())


def test_uncertainty_is_reported_separately_from_nominal_corridor():
    objects,fixtures,cad=installed()
    objects["end_stop"]["fit_residual_m"] = .001
    relation=receiver_interface(objects,fixtures,cad)["fixture_relations"]["end_stop_to_base"]
    assert relation["geometric_route_exists"]
    assert all(not h["uncertainty_clearance_certified"] for h in relation["holes"].values())
    assert relation["requires_guarded_verification"]


def test_base_feature_hypothesis_removes_its_static_body_offset():
    spec=templates()["guide_base"]
    patch=dict(center=np.array([.21,.08]),z=.84,size=np.array([.018,.052]),yaw=.2)
    hypotheses=list(_hypotheses("guide_base",spec,patch,[patch]))
    assert len(hypotheses)==2
    for position,yaw,_ in hypotheses:
        assert position+_rotate(yaw)@spec["feature_center_body_m"]==pytest.approx([.21,.08,.84])


def test_static_receiver_primitives_match_scene_without_closing_holes():
    cad=planning_metadata(); root=ET.parse(SCENE).getroot()
    ET.SubElement(root.find("worldbody"),"body",name="wipe_tool",pos="0 0 .804")
    apply_printed_kit(root)
    for part,primitives in cad["collision_primitives"].items():
        body=root.find(f".//body[@name='{part}']")
        for item in primitives:
            geom=body.find(f"geom[@name='{item['name']}']")
            assert geom is not None
            assert np.fromstring(geom.get("pos"),sep=" ")==pytest.approx(item["pos"])
            if item["type"]=="box" and part in ("guide_base","end_stop"):
                assert geom.get("type")=="mesh"
                mesh=root.find(f"asset/mesh[@name='{geom.get('mesh')}']")
                vertices=np.fromstring(mesh.get("vertex"),sep=" ").reshape(-1,3)
                assert len(vertices)==8
                assert vertices.min(axis=0)==pytest.approx(-np.asarray(item["size"]))
                assert vertices.max(axis=0)==pytest.approx(item["size"])
            else:
                assert geom.get("type")==item["type"]
                assert np.fromstring(geom.get("size"),sep=" ")==pytest.approx(item["size"])
        if part=="carriage":
            continue
        hole_points=cad["base_hole_offsets_m"] if part=="guide_base" else cad["pin_hole_offsets_m"]
        prefix="base_" if part=="guide_base" else "stop_"
        for point in hole_points:
            point=np.asarray(point).copy();point[2]=0.
            assert not any(np.all(np.abs(point-np.asarray(p["pos"]))<np.asarray(p["size"])-1.e-10)
                for p in primitives if p["type"]=="box" and p["name"].startswith(prefix))
