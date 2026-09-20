"""Targeted script grasp proposals for the unchanged functional V9 task.

These plans use the existing PlanIR and controller. Their names describe the
first intended control difference; no outcome labels enter construction.
"""

from copy import deepcopy
from math import pi

from .v9_candidates import bind, proposals as v9_proposals

SOURCE = "script_targeted_grasp_v10_r1"


def proposals():
    reference = deepcopy(v9_proposals()[0])
    reference["source"] = SOURCE
    variants = [
        ("carriage_yaw90", "carriage", {"yaw": pi / 2}),
        ("carriage_grasp_low", "carriage", {"height": -.002}),
        ("pin_left_yaw90", "pin_left", {"yaw": pi / 2}),
        ("pin_left_grasp_high", "pin_left", {"height": .003}),
        ("pin_left_yaw90_high", "pin_left", {"yaw": pi / 2, "height": .003}),
        ("handle_yaw90", "handle", {"yaw": pi / 2}),
        ("handle_grasp_high", "handle", {"height": .004}),
        ("handle_yaw90_high", "handle", {"yaw": pi / 2, "height": .004}),
    ]
    result = [reference]
    for name, part, changes in variants:
        row = deepcopy(reference)
        row["name"] = name
        row["choices"][part].update(changes)
        result.append(row)
    return result
