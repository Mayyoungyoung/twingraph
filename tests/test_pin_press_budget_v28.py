import pytest

from simbench.assembly.library import HANDLERS
from simbench.assembly.ports import validate_ports
from simbench.assembly.sensor_learning_v12 import (
    PIN_PRESS_FIXED_V27,
    PIN_PRESS_REMAINING_V28,
    pin_press_budget,
)


def test_v27_budget_is_the_recorded_fixed_six_millimetre_capacity():
    budget = pin_press_budget(.033, .043, .004, .02, PIN_PRESS_FIXED_V27)
    assert budget["step_budget"] == 241
    assert budget["command_capacity_m"] == pytest.approx(.006025)
    assert budget["remaining_distance_m"] == pytest.approx(.010)
    assert not budget["budget_limited_by_total_time"]


def test_v28_budget_uses_remaining_distance_speed_and_period():
    slow = pin_press_budget(.033, .043, .001, .02, PIN_PRESS_REMAINING_V28)
    faster = pin_press_budget(.033, .043, .00125, .02, PIN_PRESS_REMAINING_V28)
    less_remaining = pin_press_budget(.038, .043, .00125, .02, PIN_PRESS_REMAINING_V28)
    assert slow["step_budget"] > faster["step_budget"] > less_remaining["step_budget"]
    assert slow["time_budget_s"] > faster["time_budget_s"]
    assert faster["expected_executable_travel_m"] == pytest.approx(.010)
    assert faster["controller_version"] == "pin_press.remaining_distance.v28"


def test_v28_total_time_cap_is_explicit_instead_of_silent_truncation():
    budget = pin_press_budget(-.050, .043, .001, .02, PIN_PRESS_REMAINING_V28)
    assert budget["budget_limited_by_total_time"]
    assert budget["time_budget_s"] == pytest.approx(budget["total_time_limit_s"])
    assert budget["derived_step_budget"] > budget["step_budget"]


def test_complete_program_exposes_the_versioned_press_policy_port():
    spec = HANDLERS["run_sliding_assembly_v23"]
    port = next(port for port in spec.ports if port.name == "pin_press_budget_policy")
    assert port.kind == "category"
    validate_ports(spec, dict(order=[], choices={}, stroke_minimum=.02,
                              pin_press_budget_policy=PIN_PRESS_REMAINING_V28))
