"""Curated, executable script proposals for the frozen V9 task.

This is a screening fixture, not an LLM planner.  Every row changes at least
one parameter consumed by a physical atom in the complete-task controller.
"""

from copy import deepcopy
from . import stage_v5, stage_v7

SOURCE = "script_curated_v9_r1"


def reference_choices():
    return {part: dict(yaw=0., height=.001, clearance=.98,
                       force=8. if part.startswith("pin_") else 3.,
                       **({} if part == "carriage" else {"speed": .006}))
            for part in stage_v5.PARTS}


def proposals():
    order = tuple(stage_v5.legal_orders()[0])
    reversed_pins = tuple(stage_v5.legal_orders()[1])
    # The same task acceptance and stroke requirement apply to every proposal.
    variants = [
        ("reference", order, {}, 0, 1.5, 14.),
        ("pin_order", reversed_pins, {}, 0, 1.5, 14.),
        ("pin_slow", order, {"pin_left": {"speed": .0045}, "pin_right": {"speed": .0045}}, 0, 1.5, 14.),
        ("pin_firm", order, {"pin_left": {"force": 9.}, "pin_right": {"force": 9.}}, 0, 1.5, 14.),
        ("pin_gentle", order, {"pin_left": {"force": 7.}, "pin_right": {"force": 7.}}, 0, 1.5, 14.),
        ("pin_high_grasp", order, {"pin_left": {"height": .002}, "pin_right": {"height": .002}}, 0, 1.5, 14.),
        ("pin_low_grasp", order, {"pin_left": {"height": 0.}, "pin_right": {"height": 0.}}, 0, 1.5, 14.),
        ("transfer_high", order, {p: {"clearance": 1.01} for p in stage_v5.PARTS}, 0, 1.5, 14.),
        ("transfer_low", order, {p: {"clearance": .96} for p in stage_v5.PARTS}, 0, 1.5, 14.),
        ("stop_slow", order, {"end_stop": {"speed": .0045}}, 0, 1.5, 14.),
        ("wipe_reverse", order, {}, 1, 1.5, 14.),
        ("wipe_long", order, {}, 0, 1.5, 16.),
    ]
    rows = []
    for name, sequence, edits, variant, force, duration in variants:
        choices = reference_choices()
        for part, fields in edits.items():
            choices[part].update(fields)
        rows.append(dict(name=name, source=SOURCE, order=list(sequence),
                         choices=deepcopy(choices), wipe_variant=variant,
                         wipe_force=force, wipe_duration=duration,
                         stroke_minimum=stage_v7.TASK_STROKE_MINIMUM_M))
    return rows


def bind(session, targets, proposal):
    return stage_v7._full_plan(
        session, targets, proposal["order"], proposal["choices"],
        proposal["wipe_variant"], proposal["wipe_force"],
        proposal["wipe_duration"], proposal["stroke_minimum"])
