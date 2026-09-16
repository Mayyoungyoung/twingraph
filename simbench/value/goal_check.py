"""Independent final task predicates, separate from candidate-authored checkers."""
import numpy as np


def validate_goals(goals):
    if not goals:
        raise ValueError("physical value labels require declared task goals")
    for goal in goals:
        if goal.get("predicate") not in {"seated_released_retracted", "seated"}:
            raise ValueError("unsupported independent task goal")
        if not goal.get("manipulated") or np.asarray(goal.get("position")).shape != (3,):
            raise ValueError("goal requires object and world position")
        scalars = [goal.get("position_tolerance", .0015), goal.get("tilt_tolerance_deg", 3.),
                   goal.get("minimum_eef_clearance_m", 0.)]
        if (not np.isfinite(goal["position"]).all() or not np.isfinite(scalars).all()
                or scalars[0] <= 0 or scalars[1] <= 0 or scalars[2] < 0):
            raise ValueError("invalid task acceptance tolerance")


def evaluate_goals(session, goals):
    """Pose, tilt, released ownership and actual finger separation at final state.

    Any optional EEF clearance is a declared task acceptance predicate in metres,
    not a weighted reward. Unsupported predicates fail loudly before trials.
    """
    validate_goals(goals)
    rows = []
    ctx = session.ctx
    for goal in goals:
        part = goal["manipulated"]
        error = float(np.linalg.norm(ctx.obj_pos(part) - np.asarray(goal["position"])))
        tilt = float(np.degrees(np.arccos(np.clip(ctx.obj_axis(part)[2], -1, 1))))
        released = session.held != part
        touching_finger = False
        bid = ctx.body_id(part)
        for contact in ctx.data.contact:
            b1, b2 = map(int, ctx.model.geom_bodyid[[contact.geom1, contact.geom2]])
            if bid not in (b1, b2):
                continue
            other = b2 if b1 == bid else b1
            if "finger" in ctx.model.body(other).name and contact.dist <= 0:
                touching_finger = True
        clearance = float(ctx.eef_pos()[2] - ctx.obj_pos(part)[2])
        ok = error < goal.get("position_tolerance", .0015) and tilt < goal.get("tilt_tolerance_deg", 3.)
        if goal["predicate"] == "seated_released_retracted":
            ok = ok and released and not touching_finger and clearance >= goal.get("minimum_eef_clearance_m", 0.)
        rows.append(dict(part=part, success=bool(ok), position_error_m=error, tilt_deg=tilt,
                         released=released, touching_finger=touching_finger, eef_clearance_m=clearance))
    return dict(success=all(r["success"] for r in rows), goals=rows)
