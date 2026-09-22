import numpy as np
from scripts.check_v12_pin_contact_tolerance import CONFIG,evaluate,numerical_guard


def contact(depth):
    return [dict(distance_m=-depth,normal_force_n=.03,shaft_receiver=True)]


def test_guard_comes_from_dimensionless_cad_ratios():
    assert np.isclose(numerical_guard(),.000165)


def test_true_square_corner_and_tilted_shaft_remain_functional():
    assert evaluate([.00065,.00065,.012],[0,0,1])['success']
    axis=np.array([.01,0,1.]);axis/=np.linalg.norm(axis)
    assert evaluate([0,0,.012],axis)['success']


def test_soft_overlap_requires_both_cad_and_real_contact_bounds():
    guard=numerical_guard();clearance=.004-.0033
    inside=[clearance+.5*guard,0,.012]
    row=evaluate(inside,[0,0,1],contacts=contact(.5*guard))
    assert row['success'] and not row['strict_inserted']
    assert not evaluate(inside,[0,0,1])['success']  # No contact explanation.
    assert not evaluate(inside,[0,0,1],contacts=contact(1.01*guard))['success']
    assert not evaluate([clearance+1.01*guard,0,.012],[0,0,1],contacts=contact(.5*guard))['success']


def test_released_retained_and_no_finger_contact_are_mandatory():
    for changes in [dict(released=False),dict(touching_finger=True),dict(retained=False)]:
        assert not evaluate([0,0,.012],[0,0,1],**changes)['success']
    assert evaluate([0,0,.012],[0,0,1],stable=False)['success']


def test_plate_top_shallow_outside_and_crossing_shaft_fail():
    # Shaft tip at/beyond entry, only 3 mm insertion, entirely outside bore,
    # and a nearly transverse shaft cannot provide a 6 mm occupied interval.
    for origin,axis in [([0,0,.048],[0,0,1]),([0,0,.044],[0,0,1]),
        ([.008,0,.012],[0,0,1]),([0,0,-.015],[1,0,.01])]:
        assert not evaluate(origin,axis,contacts=contact(.5*numerical_guard()))['success']


def test_guard_is_not_allowed_to_bridge_invalid_depth_gaps():
    # A strongly tilted shaft crosses the receiver but only fits laterally
    # over less than 6 mm, despite broad axial intersection.
    axis=np.array([.4,0,1.]);axis/=np.linalg.norm(axis)
    assert not evaluate([0,0,.012],axis,contacts=contact(.5*numerical_guard()))['success']
