#!/usr/bin/env python3
"""Independent STL / collision occupancy audit and initial rendered evidence."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.spatial import ConvexHull

from simbench.assembly.printed_kit import FILES, body_triangles, source_manifest
from simbench.value.stage_v7 import StageV7Spec
from simbench.value.stage_v12 import write_scene


def inside_stl(points, triangles):
    direction = np.array([1., .237113, .137921]); direction /= np.linalg.norm(direction)
    a = triangles[:, 0]; e1 = triangles[:, 1] - a; e2 = triangles[:, 2] - a
    h = np.cross(direction, e2); determinant = np.einsum("ij,ij->i", e1, h)
    usable = np.abs(determinant) > 1e-14
    a, e1, e2, h, determinant = [x[usable] for x in (a, e1, e2, h, determinant)]
    inv = 1 / determinant
    result = []
    for start in range(0, len(points), 64):
        s = points[start:start+64, None] - a[None]
        u = np.einsum("nki,ki->nk", s, h) * inv
        q = np.cross(s, e1)
        v = np.einsum("i,nki->nk", direction, q) * inv
        distance = np.einsum("ki,nki->nk", e2, q) * inv
        hits = (u >= 0) & (v >= 0) & (u + v <= 1) & (distance > 1e-12)
        result.append((hits.sum(axis=1) % 2).astype(bool))
    return np.concatenate(result)


def collision_membership(root, part, points):
    body = root.find(f".//body[@name='{part}']")
    occupied = np.zeros(len(points), dtype=bool)
    meshes = {m.get("name"): m for m in root.findall("./asset/mesh")}
    for g in body.findall("geom"):
        if g.get("contype") == "0" and g.get("conaffinity") == "0": continue
        if part == "wipe_tool" and g.get("name") == "wipe_pad": continue  # source is rigid STL only
        relative = points - np.fromstring(g.get("pos", "0 0 0"), sep=" ")
        kind = g.get("type", "sphere")
        if kind == "box":
            inside = (np.abs(relative) <= np.fromstring(g.get("size"), sep=" ") + 1e-12).all(axis=1)
        elif kind == "cylinder":
            size = np.fromstring(g.get("size"), sep=" ")
            inside = (np.linalg.norm(relative[:, :2], axis=1) <= size[0]) & (np.abs(relative[:, 2]) <= size[1])
        elif kind == "mesh":
            vertices = np.fromstring(meshes[g.get("mesh")].get("vertex"), sep=" ").reshape(-1, 3)
            equations = ConvexHull(vertices).equations
            inside = (relative @ equations[:, :3].T + equations[:, 3] <= 1e-10).all(axis=1)
        else: raise ValueError(kind)
        occupied |= inside
    return occupied


def audit(path, count=8192):
    root = ET.parse(path).getroot(); results = {}
    for index, part in enumerate(FILES):
        tri = body_triangles(part)
        bounds = np.array([tri.min(axis=(0, 1)), tri.max(axis=(0, 1))])
        rng = np.random.default_rng(42000 + index)
        points = rng.uniform(bounds[0] - .0001, bounds[1] + .0001, (count, 3))
        actual, expected = collision_membership(root, part, points), inside_stl(points, tri)
        mismatch = actual != expected
        results[part] = dict(samples=count, mismatches=int(mismatch.sum()),
            disagreement_fraction=float(mismatch.mean()),
            mismatch_points_body_m=points[mismatch].tolist(),
            note="Annular chords and analytical cylinders differ from tessellated STL within stated discretization bound")
    stop_holes = np.array([[x, y0 + y, z] for y0 in (-.032, .032) for x in (-.0038, 0, .0038)
                           for y in (-.0038, 0, .0038) for z in (-.017, 0, .017)])
    checks = dict(square_stop_holes_open=not bool(collision_membership(root, "end_stop", stop_holes).any()),
        stop_has_material_outside_8mm_hole=bool(collision_membership(root, "end_stop", np.array([[.005, -.032, 0.]])).all()),
        handle_bore_open=not bool(collision_membership(root, "handle", np.array([[0., 0., 0.], [.006, 0., 0.]])).any()),
        holder_bore_open=not bool(collision_membership(root, "pin_left_holder", np.array([[0., 0., 0.], [.0043, 0., 0.]])).any()),
        bridge_is_solid=bool(collision_membership(root, "carriage", np.array([[0., 0., .008]])).all()),
        no_unprinted_cradle=not any("cradle" in g.get("name", "") for g in root.findall(".//geom")),
        no_unprinted_funnel=not any("v9_bore" in g.get("name", "") for g in root.findall(".//geom")))
    return dict(schema="twingraph.printed-kit-audit.v12", results=results, structural_checks=checks,
                passed=all(checks.values()) and all(r["disagreement_fraction"] < .001 for r in results.values()),
                sampling_limitation="Random occupancy is supplementary, not a proof of Hausdorff distance; source CAD primitives and open-hole probes are also checked.")


def main():
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1600); p.add_argument("--samples", type=int, default=8192)
    p.add_argument("--render", action="store_true"); args = p.parse_args()
    path = write_scene(StageV7Spec.sample(args.seed), args.out)
    result = audit(path, args.samples)
    (args.out / "collision_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.render:
        from simbench.value.stage_v12 import make_scene
        from simbench.value.stage_v7 import capture_vision
        from PIL import Image, ImageDraw
        _, session, _, _ = make_scene(args.seed, args.out)
        views = capture_vision(session, (640, 800))
        canvas = Image.new("RGB", (1600, 675), "#16202b")
        for i, camera in enumerate(("task_view", "top_view")):
            canvas.paste(Image.fromarray(views[camera + "_rgb"]), (800 * i, 35))
        ImageDraw.Draw(canvas).text((12, 10), "V12 | Original htzp STL visuals + audited open-hole collision geometry | seed " + str(args.seed), fill="white")
        canvas.save(args.out / "printed_kit_scene.png")
        (args.out / "initial_rgbd_observation.json").write_text(json.dumps(session.decision_observation, indent=2), encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "mismatches": {k: v["mismatches"] for k, v in result["results"].items()}, "structural_checks": result["structural_checks"]}))
    if not result["passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
