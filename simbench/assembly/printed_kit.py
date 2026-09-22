"""Printed-kit CAD in SI units, with open-hole collision solids.

Source STL bytes are the user's htzp files, not the widened V9 fixture.
MuJoCo takes convex hulls of mesh collision geoms: therefore full STL meshes
are visual only; named boxes/cylinders and convex annular sectors implement
the CAD union while retaining holes. No dynamic simulation poses enter CAD.
"""
from __future__ import annotations

import hashlib
import math
import itertools
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np

from .scene import fmt, geom, holed_plate

KIT = Path(__file__).resolve().parents[1] / "assets" / "printed_kit_v0_1"
FILES = {
    "guide_base": "01_guide_base_nominal.stl",
    "carriage": "02_carriage_bridged.stl",
    "end_stop": "03_end_stop.stl",
    "pin_left": "04_pin.stl", "pin_right": "04_pin.stl",
    "handle": "05_handle_ring.stl",
    "pin_left_holder": "06_pin_supply_holder.stl",
    "pin_right_holder": "06_pin_supply_holder.stl",
    "wipe_tool": "07_wipe_tool_rigid.stl",
}
BODY_Z = {"guide_base": -.006, "carriage": -.006, "end_stop": -.018,
          "pin_left": .013, "pin_right": .013, "handle": -.008,
          "pin_left_holder": -.019, "pin_right_holder": -.019, "wipe_tool": .004}
COLORS = {"guide_base": ".28 .31 .35 1", "carriage": ".26 .49 .70 1",
          "end_stop": ".78 .48 .24 1", "handle": ".21 .23 .26 1",
          "pin_left": ".95 .88 .20 1", "pin_right": ".95 .88 .20 1",
          "pin_left_holder": ".44 .44 .43 1", "pin_right_holder": ".44 .44 .43 1",
          "wipe_tool": ".85 .65 .25 1"}
# User confirmed all printed pieces are white (2026-09-21). Earlier colored
# renderings remain development evidence, not the user's actual appearance.
COLORS = {part: ".92 .92 .92 1" for part in FILES}
ANNULUS_SEGMENTS = 210  # exact circular tessellation in the supplied ring STLs


def stl_triangles(path):
    raw = Path(path).read_bytes()
    count = struct.unpack_from("<I", raw, 80)[0]
    if len(raw) != 84 + 50 * count:
        raise ValueError("expected binary source STL")
    dtype = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
    return np.frombuffer(raw, dtype=dtype, offset=84)["vertices"].astype(float)


def body_triangles(part):
    triangles = stl_triangles(KIT / "stl" / FILES[part]) * .001
    if part in ("pin_left", "pin_right"):
        triangles *= np.array([1., -1., -1.])  # undo head-down print orientation
    triangles[:, :, 2] += BODY_Z[part]
    return triangles


def mass_properties(triangles, density=700.):
    """Signed tetrahedron integral; avoids mass double-counting CAD unions."""
    v = np.asarray(triangles, float)
    signed_volume = np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])) / 6
    volume = float(signed_volume.sum())
    if volume <= 0:
        raise ValueError("source mesh must have outward winding and positive volume")
    total = v.sum(axis=1)
    center = np.sum(signed_volume[:, None] * total / 4, axis=0) / volume
    second = np.sum(signed_volume[:, None, None] * (
        np.einsum("nki,nkj->nij", v, v) + np.einsum("ni,nj->nij", total, total)) / 20, axis=0)
    inertia = density * (np.eye(3) * np.trace(second) - second)
    inertia -= density * volume * (np.eye(3) * np.dot(center, center) - np.outer(center, center))
    return volume, center, inertia


def fixture_collision_primitives():
    """Static body-frame CAD solids, matching the open-hole scene union.

    These descriptors contain no world pose. Consumers must transform them
    using an observed receiver pose, never a simulator object transform.
    """
    output = {}
    for part in ("guide_base", "end_stop", "carriage"):
        body = ET.Element("body")
        if part == "guide_base":
            holed_plate(body, "base", (.118,.049), 0., .006,
                        [(-.092,-.032),(-.092,.032)], "0 0 0 0", holehalf=.004)
            for sign in (-1,1):
                geom(body, f"rail_{sign}", [.020,sign*.034,.012], [.089,.007,.012], "0 0 0 0")
                geom(body, f"lip_{sign}", [.020,sign*.027,.023], [.089,.006,.003], "0 0 0 0")
            geom(body,"rear_stop",[.108,0,.014],[.009,.026,.014],"0 0 0 0")
            for i,x in enumerate((-.10723,-.07677)):
                for j,y in enumerate((-.04623,.04623)):
                    geom(body,f"stop_locator_{i}_{j}",[x,y,.008],[.004,.009],"0 0 0 0","cylinder")
        elif part == "end_stop":
            holed_plate(body,"stop",(.012,.043),0.,.018,[(0,-.032),(0,.032)],"0 0 0 0",holehalf=.004)
            geom(body,"stop_boss",[0,0,.023],[.011,.013,.008],"0 0 0 0")
        else:
            geom(body,"carriage_shoe",[0,0,0],[.027,.023,.006],"0 0 0 0")
            geom(body,"carriage_bridge",[0,0,.008],[.022,.014,.002],"0 0 0 0")
            geom(body,"carriage_boss",[0,0,.025],[.022,.014,.015],"0 0 0 0")
            geom(body,"handle_post",[0,0,.050],[.0055,.012],"0 0 0 0","cylinder")
        output[part] = [dict(name=g.get("name"),type=g.get("type","box"),
            pos=np.fromstring(g.get("pos","0 0 0"),sep=" ").tolist(),
            quat_wxyz=np.fromstring(g.get("quat","1 0 0 0"),sep=" ").tolist(),
            size=np.fromstring(g.get("size"),sep=" ").tolist()) for g in body.findall("geom")]
    return output


def planning_metadata():
    parts = {}
    for part in FILES:
        tri = body_triangles(part)
        volume, center, _ = mass_properties(tri)
        bounds = np.array([tri.min(axis=(0, 1)), tri.max(axis=(0, 1))])
        parts[part] = dict(stl_file=FILES[part], dimensions_m=(bounds[1] - bounds[0]).tolist(),
                           body_bounds_m=bounds.tolist(), solid_volume_m3=volume,
                           center_of_mass_body_m=center.tolist(),
                           assumed_mass_kg=volume * 700., effective_density_kg_m3=700.,
                           stl_to_body_z_m=BODY_Z[part])
    for part, dz, width in (("carriage", .027, .028), ("end_stop", .023, .026),
                            ("pin_left", .006, .018), ("pin_right", .006, .018),
                            ("handle", 0., .042), ("wipe_tool", .047, .028)):
        parts[part]["grasp_reference"] = dict(height_offset_m=dz, width_m=width)
    for part, height, widths in (("carriage", [.010,.040], [.044,.028]),
                                 ("end_stop", [.018,.031], [.022,.026]),
                                 ("handle", [-.008,.008], [.042,.042])):
        parts[part]["grasp_region"] = dict(body_z_interval_m=height,
            closing_axes_body=[dict(axis=[1.,0.,0.],width_m=widths[0]),
                               dict(axis=[0.,1.,0.],width_m=widths[1])],
            rotational_symmetry=part=="handle",
            scope="CAD side surface only; gripper fit, pad overlap, contact and clearance require separate query")
    # Object-frame grasp centres are task CAD, not value-model features or
    # outcome-derived answers.  The generic catalogue checks every declared
    # centre against the same complete gripper/body/receiver geometry.
    parts["carriage"]["grasp_region"]["center_offsets_body_m"] = [
        [0.,0.,0.], [.012,0.,0.], [-.012,0.,0.], [0.,.012,0.], [0.,-.012,0.]]
    parts["end_stop"]["grasp_region"]["center_offsets_body_m"] = [
        [0.,0.,0.], [.008,0.,0.], [-.008,0.,0.], [0.,.008,0.], [0.,-.008,0.]]
    parts["handle"]["grasp_region"]["center_offsets_body_m"] = [[0.,0.,0.]]
    return dict(schema="twingraph.printed-kit.v12", source="user htzp STL / accompanying CAD source",
                units="m", parts=parts, pin_hole_shape="square", pin_hole_width_m=.008,
                pin_hole_radius_m=.004, pin_shaft_radius_m=.0033,
                pin_shaft_length_m=.053, pin_shaft_offsets_m=[-.047, .006],
                pin_hole_offsets_m=[[0., -.032, .018], [0., .032, .018]],
                base_hole_offsets_m=[[-.092,-.032,.006],[-.092,.032,.006]],
                base_receiver_depth_m=.012, pin_base_minimum_depth_m=.006,
                pin_head_underside_body_z_m=-.001,
                pin_bridge_minimum_total_depth_m=.042,
                pin_head_seated_total_depth_m=.046,
                legacy_isolated_stop_command_depth_m=.008,
                end_stop_mating_pose_in_base=dict(position_m=[-.092,0.,.024],quat_wxyz=[1.,0.,0.,0.]),
                carriage_mating_pose_in_base=dict(position_m=[.030,0.,.012],quat_wxyz=[1.,0.,0.,0.]),
                rail_geometry=dict(axis_local=[1.,0.,0.],entry_face_local_m=[-.118,0.,.006],
                    entry_center_local_m=[-.155,0.,.012],entry_approach_local_m=[-.155,0.,.042],
                    entry_clearance_m=.010,carriage_center_z_local_m=.012),
                collision_primitives=fixture_collision_primitives(),
                pin_receiver_depth_m=.036, handle_bore_radius_m=.0061,
                handle_post_radius_m=.0055, handle_post_top_body_m=.062,
                handle_seat_center_offset_from_carriage_m=.048,
                carriage_rail_height_from_base_origin_m=.006,
                carriage_permitted_x_range_body_m=[-.040, .060],
                guide_stroke_x_bounds_m=[.035, .135],
                guide_stroke_available_m=.100, functional_stroke_minimum_m=.020, foam_thickness_m=.008,
                wipe_surface_height_m=.818, wipe_pad_half_thickness_m=.004,
                fixture_calibration_source="fixed guide_base mounting: body z .812 m + CAD plate top .006 m",
                assumptions={"mass": "effective density 700 kg/m3 placeholder; measure printed masses before hardware transfer",
                             "material": "friction/compliance are uncalibrated; no claim of print-material identification",
                             "foam": "8 mm pad required by kit CAD; separate non-STL accessory",
                             "appearance": "all printed parts white per user; foam remains unmeasured neutral accessory"})


def _annulus(asset, body, name, inner, outer, halfheight, z=0.):
    source_part = "handle" if name == "handle" else name
    triangles = body_triangles(source_part)
    xy = np.unique(np.round(triangles[:, :, :2].reshape(-1, 2), 11), axis=0)
    radius = np.linalg.norm(xy, axis=1)
    inner_ring, outer_ring = xy[radius < (inner + outer) / 2], xy[radius >= (inner + outer) / 2]
    outer_ring = outer_ring[np.argsort(np.arctan2(outer_ring[:, 1], outer_ring[:, 0]))]
    inner_angles = np.arctan2(inner_ring[:, 1], inner_ring[:, 0])
    aligned_inner = []
    for point in outer_ring:
        angle = math.atan2(point[1], point[0])
        distances = np.abs(np.angle(np.exp(1j * (inner_angles - angle))))
        aligned_inner.append(inner_ring[np.argmin(distances)])
    inner_ring = np.asarray(aligned_inner)
    if len(outer_ring) != ANNULUS_SEGMENTS or len(inner_ring) != ANNULUS_SEGMENTS:
        raise ValueError("source ring tessellation changed; re-audit convex sector decomposition")
    for i in range(len(outer_ring)):
        key = f"print_{name}_sector_{i}"
        nxt = (i + 1) % len(outer_ring)
        polygon = [inner_ring[i], outer_ring[i], outer_ring[nxt], inner_ring[nxt]]
        vertices = [[x, y, h] for h in (-halfheight, halfheight) for x, y in polygon]
        ET.SubElement(asset, "mesh", name=key, vertex=fmt(np.ravel(vertices)))
        ET.SubElement(body, "geom", name=key, type="mesh", mesh=key, pos=fmt([0, 0, z]),
                      group="3", rgba="0 0 0 0", mass="0", friction="1 .02 .001",
                      condim="4", solref=".008 1", solimp=".95 .99 .001")


def _exact_cuboid_collisions(asset, body):
    """Use identical convex cuboids through MuJoCo's mesh narrow phase.

    MuJoCo 2.3.7 native box/box contacts can emit impossible deep contacts
    for exactly coplanar subdivided plates. Eight vertices preserve each
    solid, including all open-hole boundaries; no hull spans multiple boxes.
    """
    for item in body.findall("geom"):
        if item.get("type", "box") != "box":
            continue
        half = np.fromstring(item.get("size"), sep=" ")
        if half.shape != (3,):
            raise ValueError("expected a three-dimensional CAD cuboid")
        vertices = list(itertools.product(*[(-h, h) for h in half]))
        name = "exact_cuboid_" + item.get("name")
        ET.SubElement(asset, "mesh", name=name, vertex=fmt(np.ravel(vertices)))
        item.set("type", "mesh"); item.set("mesh", name)
        del item.attrib["size"]


def apply_printed_kit(root):
    """Replace all product/holder/tool geometry on a generated V7 scene."""
    asset = root.find("asset")
    for part in FILES:
        body = root.find(f".//body[@name='{part}']")
        if body is None:
            raise ValueError(f"scene lacks printed body {part}")
        for child in list(body):
            if child.tag in {"geom", "inertial"}:
                body.remove(child)
        name = "print_visual_" + part
        # Absolute paths avoid inheriting Panda's meshdir for the part STL.
        ET.SubElement(asset, "mesh", name=name, file=str((KIT / "stl" / FILES[part]).resolve()), scale=".001 .001 .001")
        ET.SubElement(body, "geom", name=name, type="mesh", mesh=name,
                      pos=fmt([0, 0, BODY_Z[part]]), quat="0 1 0 0" if part in ("pin_left", "pin_right") else "1 0 0 0",
                      rgba=COLORS[part], contype="0", conaffinity="0", group="1", mass="0")
        volume, center, inertia = mass_properties(body_triangles(part))
        if part == "wipe_tool":
            # Add external 36x24x8 foam, assumed 8 g, without altering STL.
            foam_mass, pad_center = .008, np.zeros(3)
            rigid_mass = volume * 700.
            combined = (rigid_mass * center + foam_mass * pad_center) / (rigid_mass + foam_mass)
            inertia += rigid_mass * (np.eye(3) * np.dot(center - combined, center - combined) - np.outer(center - combined, center - combined))
            inertia += foam_mass * (np.eye(3) * np.dot(pad_center - combined, pad_center - combined) - np.outer(pad_center - combined, pad_center - combined))
            inertia += np.diag(foam_mass / 12 * np.array([.024**2 + .008**2, .036**2 + .008**2, .036**2 + .024**2]))
            center, mass = combined, rigid_mass + foam_mass
        else:
            mass = volume * 700.
        # Fixed parts also need nonzero inertial mass for mesh compilation.
        ET.SubElement(body, "inertial", mass=str(mass), pos=fmt(center),
                      fullinertia=fmt([inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2]]))
        collision = dict(group="3", rgba="0 0 0 0", mass="0")
        def box(key, pos, size, **kw):
            return geom(body, key, pos, size, "0 0 0 0", **{**collision, **kw})
        def cylinder(key, pos, radius, halfheight, **kw):
            return geom(body, key, pos, [radius, halfheight], "0 0 0 0", "cylinder", **{**collision, **kw})
        if part == "guide_base":
            holed_plate(body, "base", (.118, .049), 0., .006, [(-.092, -.032), (-.092, .032)], "0 0 0 0", holehalf=.004)
            for g in body.findall("geom"):
                if g.get("name", "").startswith("base_"): g.attrib.update(collision)
            for sign in (-1, 1):
                box(f"rail_{sign}", [.020, sign * .034, .012], [.089, .007, .012], friction=".2 .005 .0001")
                box(f"lip_{sign}", [.020, sign * .027, .023], [.089, .006, .003], friction=".2 .005 .0001")
            box("rear_stop", [.108, 0, .014], [.009, .026, .014])
            for i, x in enumerate((-.10723, -.07677)):
                for j, y in enumerate((-.04623, .04623)):
                    cylinder(f"stop_locator_{i}_{j}", [x, y, .008], .004, .009)
        elif part == "carriage":
            box("carriage_shoe", [0, 0, 0], [.027, .023, .006], friction=".25 .005 .0001")
            box("carriage_bridge", [0, 0, .008], [.022, .014, .002])
            box("carriage_boss", [0, 0, .025], [.022, .014, .015], friction="1 .02 .001")
            cylinder("handle_post", [0, 0, .050], .0055, .012)
        elif part == "end_stop":
            holed_plate(body, "stop", (.012, .043), 0., .018, [(0, -.032), (0, .032)], "0 0 0 0", holehalf=.004)
            for g in body.findall("geom"):
                if g.get("name", "").startswith("stop_"): g.attrib.update(collision)
            box("stop_boss", [0, 0, .023], [.011, .013, .008], friction="1 .02 .001")
        elif part in ("pin_left", "pin_right"):
            cylinder(part + "_shaft", [0, 0, -.0205], .0033, .0265)
            cylinder(part + "_tip", [0, 0, -.047], .0027, .001)
            cylinder(part + "_head", [0, 0, .006], .009, .007, friction="1 .02 .001")
        elif part == "handle":
            _annulus(asset, body, "handle", .0061, .021, .008)
        elif part.endswith("_holder"):
            _annulus(asset, body, part, .0044, .010, .019)
        elif part == "wipe_tool":
            # Body ref is center of external foam, rigid STL bottom is +4 mm.
            box("wipe_pad", [0, 0, 0], [.018, .012, .004], rgba=".45 .45 .45 1", group="1", solref=".03 1", solimp=".8 .95 .002", friction=".25 .01 .001")
            box("wipe_backing", [0, 0, .0055], [.018, .012, .0015])
            box("wipe_stem", [0, 0, .010], [.006, .007, .006])
            taper = [[x * width, y * depth, z] for width, depth, z in ((.006, .007, .016), (.012, .014, .025)) for x, y in ((-1, -1), (-1, 1), (1, -1), (1, 1))]
            ET.SubElement(asset, "mesh", name="printed_wipe_taper", vertex=fmt(np.ravel(taper)))
            ET.SubElement(body, "geom", name="wipe_taper", type="mesh", mesh="printed_wipe_taper", **collision)
            box("wipe_handle", [0, 0, .047], [.012, .014, .022], friction="1 .02 .001")
            xyz = np.fromstring(body.get("pos"), sep=" "); xyz[2] = .804
            body.set("pos", fmt(xyz))
        if part in ("guide_base", "end_stop"):
            # Scope to the subdivided receiver plates and their rigid solids.
            # Other components retain their established primitive backend.
            _exact_cuboid_collisions(asset, body)
    return root


def source_manifest():
    entries = []
    for name in sorted(set(FILES.values())):
        path = KIT / "stl" / name
        tri = stl_triangles(path)
        entries.append(dict(file=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            triangles=len(tri), source_bounds_mm=[tri.min(axis=(0, 1)).tolist(), tri.max(axis=(0, 1)).tolist()]))
    return dict(schema="twingraph.printed-kit-source.v12", source_directory=r"F:\RAL\3D\TwinGraph_print_kit_v0_1\TwinGraph_print_kit_v0_1\stl\htzp",
                source_units="mm", simulation_units="m", stl_scale=.001,
                duplicate_instances={"04_pin_2.stl": "04_pin.stl", "06_pin_supply_holder_2.stl": "06_pin_supply_holder.stl"},
                parts=entries, annulus_collision_segments=ANNULUS_SEGMENTS,
                maximum_annulus_chord_error_m=.021 * (1 - math.cos(math.pi / ANNULUS_SEGMENTS)),
                annulus_source_match="sectors reuse exact STL inner/outer XY vertices; analytical-circle chord error is already present in source STL",
                collision_method="CAD box/cylinder union; base/end-stop cuboids use exact 8-vertex convex meshes; open convex annular sectors, never full-STL convex hull",
                cuboid_backend="Base/end-stop use generic convex mesh narrow phase; other boxes retain native primitives; mathematical box descriptors remain dimensionally equivalent",
                planning_cad=planning_metadata())
