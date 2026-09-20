"""V9 CAD, perception metadata and insertion predicate agree."""
import xml.etree.ElementTree as ET
from pathlib import Path
import tempfile

import numpy as np

from simbench.value import stage_v7, stage_v9
from simbench.value.pin_geometry import insertion_geometry


def test_v9_physical_bore_holder_and_templates():
    # MuJoCo mesh paths are relative to the scene directory on the same drive.
    scratch = Path("results/value_v9_functional_assembly/test_scenes")
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as directory:
        path = stage_v9.write_scene(stage_v7.StageV7Spec.sample(1200), directory)
        root = ET.parse(path).getroot()
    stop = root.find(".//body[@name='end_stop']")
    assert not any(g.get("name", "").startswith("v7_bore_") for g in stop.findall("geom"))
    assert sum(g.get("name", "").startswith("v9_bore_") for g in stop.findall("geom")) == 32
    assert sum(g.get("name", "").startswith("v9_stop_") for g in stop.findall("geom")) > 0
    for part in ("pin_left", "pin_right"):
        holder = root.find(f".//body[@name='{part}_holder']")
        assert len(holder.findall("geom")) == 12
        assert all(float(g.get("size").split()[2]) == stage_v9.HOLDER_HALF_HEIGHT_M
                   for g in holder.findall("geom"))
    templates = stage_v9.visual_templates()
    assert templates["end_stop"]["through_hole_half_width_m"] == stage_v9.PIN_CONFIG.plate_hole_half_width_m
    assert templates["end_stop"]["guide_inner_radius_m"] == stage_v9.PIN_CONFIG.guide_inner_radius_m
    assert templates["pin_left"]["shaft_radius_m"] == stage_v9.PIN_CONFIG.shaft_radius_m


def test_functional_acceptance_allows_tilt_but_rejects_missed_bore():
    entry = np.array([0., 0., .0])
    axis = np.array([.15, 0., 1.]); axis /= np.linalg.norm(axis)
    inside = insertion_geometry(np.array([.006, 0., .039]), axis, entry, [0., 0., 1.], stage_v9.PIN_CONFIG)
    outside = insertion_geometry(np.array([.015, 0., .039]), axis, entry, [0., 0., 1.], stage_v9.PIN_CONFIG)
    assert inside["inserted"]
    assert not outside["inserted"]
