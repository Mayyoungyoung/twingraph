import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from simbench.assembly.printed_kit import KIT, body_triangles, mass_properties, planning_metadata, source_manifest
from simbench.value.stage_v7 import StageV7Spec
from simbench.value.stage_v12 import write_scene
from scripts.audit_v12_printed_kit import collision_membership, inside_stl


def test_original_stl_bytes_are_versioned_and_pin_orientation_is_restored():
    frozen = json.loads((KIT / "manifest.json").read_text(encoding="utf-8"))
    assert {r["file"]: r["sha256"] for r in source_manifest()["parts"]} == frozen["sha256"]
    pin = body_triangles("pin_left")
    np.testing.assert_allclose([pin[:, :, 2].min(), pin[:, :, 2].max()], [-.048, .013], atol=1e-9)
    # The wide head belongs above the shaft after undoing print orientation.
    head_vertices = pin[np.linalg.norm(pin[:, :, :2], axis=2) > .008]
    assert head_vertices[:, 2].min() >= -.001000001


def test_holes_and_actual_carriage_bridge_survive_collision_conversion(tmp_path):
    root = ET.parse(write_scene(StageV7Spec.sample(1), tmp_path)).getroot()
    holes = np.array([[x, -.032 + y, z] for x in (-.0038, 0, .0038)
                      for y in (-.0038, 0, .0038) for z in (-.017, 0, .017)])
    assert not collision_membership(root, "end_stop", holes).any()
    assert collision_membership(root, "end_stop", np.array([[.005, -.032, 0.]])).all()
    assert collision_membership(root, "carriage", np.array([[0., 0., .008]])).all()
    assert not collision_membership(root, "handle", np.array([[.006, 0., 0.]])).any()
    assert not collision_membership(root, "pin_left_holder", np.array([[.0043, 0., 0.]])).any()
    assert not any("cradle" in g.get("name", "") or "v9_bore" in g.get("name", "") for g in root.findall(".//geom"))


def test_annular_convex_sectors_match_source_stl_including_inner_wall(tmp_path):
    root = ET.parse(write_scene(StageV7Spec.sample(1), tmp_path)).getroot()
    # Dense near-wall probes would detect a full-mesh convex hull or widened bore.
    theta = np.linspace(0., 2 * np.pi, 1001, endpoint=False)
    points = np.array([[radius * np.cos(a), radius * np.sin(a), 0.]
                       for radius in (.00608, .006099, .006101, .020999, .021001) for a in theta])
    np.testing.assert_array_equal(collision_membership(root, "handle", points), inside_stl(points, body_triangles("handle")))


def test_print_geometry_compiles_with_visual_meshes_disabled_for_collision(tmp_path):
    path = write_scene(StageV7Spec.sample(2), tmp_path)
    model = mujoco.MjModel.from_xml_path(str(path))
    for part in planning_metadata()["parts"]:
        visual = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "print_visual_" + part)
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, part)
        assert model.geom_contype[visual] == model.geom_conaffinity[visual] == 0
        assert model.body_mass[body] > 0
    assert not any("v9_bore" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "") for i in range(model.ngeom))


def test_static_metadata_and_mesh_mass_integrals_preserve_design_dimensions():
    cad = planning_metadata()
    assert cad["pin_hole_shape"] == "square" and cad["pin_hole_width_m"] == .008
    assert cad["pin_shaft_radius_m"] == .0033
    assert cad["parts"]["carriage"]["dimensions_m"][2] == .068
    volume, center, inertia = mass_properties(body_triangles("end_stop"))
    assert abs(volume * 1e9 - 77132.) < .01
    assert np.linalg.eigvalsh(inertia).min() > 0
