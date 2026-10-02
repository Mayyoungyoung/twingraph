import numpy as np
import pytest

from simbench.assembly.constrained_stroke_v29 import (
    STROKE_PROGRESS_V29,
    guide_coordinate,
    live_stroke_plan,
)
from simbench.assembly.library import HANDLERS
from simbench.assembly.ports import validate_ports


def test_live_plan_uses_post_grasp_tracked_carriage_coordinate():
    plan = live_stroke_plan(np.array([.104, .085, .85]), .020, (.035, .135),
                            preferred_first_direction=1.)
    assert plan["tracked_start_coordinate_m"] == pytest.approx(.104)
    first, second = plan["targets_m"]
    assert .0365 <= first <= .1335 and .0365 <= second <= .1335
    assert abs(first - .104) >= .022
    assert abs(second - first) >= .022
    assert first > .104


def test_guide_coordinate_supports_configured_axis_and_origin():
    assert guide_coordinate([2., 5., 9.], [0., 1., 0.], [2., 1., 9.]) == pytest.approx(4.)


def test_v23_program_declares_versioned_stroke_policy_port():
    spec = HANDLERS["run_sliding_assembly_v23"]
    port = next(port for port in spec.ports if port.name == "stroke_motion_policy")
    assert port.kind == "category"
    validate_ports(spec, dict(order=[], choices={}, stroke_minimum=.02,
                              pin_press_budget_policy="remaining_distance_v28",
                              stroke_motion_policy=STROKE_PROGRESS_V29))
