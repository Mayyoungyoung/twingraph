"""Public parameterized interfaces; legacy component names remain compatible.

Modes deliberately dispatch to the original handlers, preserving their distinct
joint/Cartesian goals, contact guards and success checks.
"""

INTERFACES = {
    "observe": {"scene": "observe_parts"},
    "move": {
        "joint_path": "execute_joint_path",
        "cartesian_path": "execute_cartesian_path",
        "lift": "lift",
        "lower": "lower",
        "home": "home",
        "retreat": "retreat",
        "approach": "approach",
        "orient": "orient_wrist",
        "align": "align_axis",
    },
    "gripper": {"open": "open_gripper", "close": "close_gripper"},
    "guarded_move": {
        "descend": "guarded_descent",
        "seat": "press_seat",
        "retract": "retract_contact",
    },
    "insert": {
        "axis": "slide_insert",
        "spiral": "spiral_search",
        "learned": "learned_insert",
    },
    "actuate": {"slide": "move_constrained"},
}


def resolve(name, params):
    params = dict(params)
    if name in INTERFACES:
        mode = params.pop("mode", None)
        modes = INTERFACES[name]
        if mode is None and len(modes) == 1:
            mode = next(iter(modes))
        if mode not in modes:
            raise ValueError(f"{name}: mode must be one of {tuple(modes)}")
        name = modes[mode]
    return name, params


def family(component):
    return next(
        (name for name, modes in INTERFACES.items() if component in modes.values()),
        component,
    )
