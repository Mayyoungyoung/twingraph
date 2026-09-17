import xml.etree.ElementTree as ET

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from simbench.value.collect_v6 import trial_spec
from simbench.value.stage_v6 import PARTS, StageV6Spec, write_scene
from simbench.value.value_v6 import RobustProgramNet, _vision_array


def test_v6_scene_randomization_is_bounded_and_reproducible(tmp_path):
    left = StageV6Spec.sample(70100)
    right = StageV6Spec.sample(70100)
    assert left == right
    assert set(left.supply_shifts) == set(PARTS)
    assert max(abs(x) for shift in left.supply_shifts.values() for x in shift) <= .012
    path = write_scene(left, tmp_path)
    root = ET.parse(path).getroot()
    assert root.find(".//camera[@name='task_view']") is not None
    assert root.find(".//camera[@name='top_view']") is not None


def test_v6_trials_pair_noise_and_keep_nonperfect_nominal_perception():
    nominal = trial_spec(70100, 0, "train")
    perturbed = trial_spec(70100, 1, "train")
    assert nominal["friction_scale"] == nominal["mass_scale"] == 1.
    assert nominal["position_noise_std_m"] > 0
    assert .92 <= perturbed["friction_scale"] <= 1.08
    assert .00025 <= perturbed["position_noise_std_m"] <= .00125
    assert perturbed == trial_spec(70100, 1, "train")


def test_v6_view_ablation_channels_and_network_shapes():
    arrays = {}
    for name in ("task_view", "top_view"):
        arrays[f"{name}_rgb"] = np.zeros((80, 80, 3), np.uint8)
        arrays[f"{name}_depth_mm"] = np.full((80, 80), 900, np.uint16)
    assert _vision_array(arrays, "none").shape == (0, 1, 1)
    assert _vision_array(arrays, "task").shape == (4, 80, 80)
    assert _vision_array(arrays, "top").shape == (4, 80, 80)
    assert _vision_array(arrays, "both").shape == (8, 80, 80)
    x = torch.zeros(3, 11)
    assert RobustProgramNet(11, "none")(x).shape == (3,)
    assert RobustProgramNet(11, "both")(x, torch.zeros(3, 8, 80, 80)).shape == (3,)
