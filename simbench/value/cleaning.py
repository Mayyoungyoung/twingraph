"""Stateful, contact-gated stain model used by value-v7.

This is a deliberately modest contact-condition model, not a material or
particle simulation.  Stain amount changes only when the wipe pad is in
contact with the named surface and the tool has made tangential progress.
"""

import copy
import numpy as np
import mujoco


def initialize(session, points, site_names, seed=0):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("dirty points must be a nonempty Nx3 array")
    if len(site_names) != len(points):
        raise ValueError("dirty site count mismatch")
    ids = [mujoco.mj_name2id(session.ctx.model, mujoco.mjtObj.mjOBJ_SITE, n)
           for n in site_names]
    if any(i < 0 for i in ids):
        raise ValueError("dirty site missing from compiled scene")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 8301]))
    weights = rng.uniform(.75, 1.25, len(points))
    state = dict(
        surface="guide_base",
        points=points.tolist(),
        site_names=list(site_names),
        site_ids=[int(i) for i in ids],
        weights=weights.tolist(),
        initial_dirty_amount=float(weights.sum()),
        remaining_dirty_amount=float(weights.sum()),
        cleaning_ratio=0.0,
        seed=int(seed),
    )
    session.dirty_state = state
    for site_id in ids:
        session.ctx.model.site_rgba[site_id, 3] = 1.0
    return state


def _sync_visual(session):
    state = session.dirty_state
    if not state:
        return
    initial = max(float(state["initial_dirty_amount"]), 1e-12)
    state["remaining_dirty_amount"] = float(sum(state["weights"]))
    state["cleaning_ratio"] = float(1.0 - state["remaining_dirty_amount"] / initial)
    for site_id, weight in zip(state["site_ids"], state["weights"]):
        session.ctx.model.site_rgba[int(site_id), 3] = float(np.clip(weight, 0.0, 1.0))


def apply_contact(session, actual_points, forces, footprint, surface_ok=True):
    state = session.dirty_state
    if not state:
        return dict(initial_dirty_amount=0.0, remaining_dirty_amount=0.0,
                    cleaning_ratio=1.0, cleaned_sites=0)
    points = np.asarray(state["points"], dtype=float)
    weights = np.asarray(state["weights"], dtype=float)
    actual = np.asarray(actual_points, dtype=float)
    forces = np.asarray(forces, dtype=float)
    footprint = np.asarray(footprint, dtype=float)
    # A stationary press is intentionally not a cleaning action.  Each
    # segment must contain tangential motion and valid pad/surface force.
    for i in range(1, len(actual)):
        moved = float(np.linalg.norm(actual[i, :2] - actual[i - 1, :2]))
        valid = bool(surface_ok and forces[i] > 0.15 and moved >= 0.0008)
        if not valid:
            continue
        hit = np.all(np.abs(points[:, :2] - actual[i, :2]) <= footprint, axis=1)
        weights[hit] *= 0.08
    state["weights"] = weights.tolist()
    _sync_visual(session)
    return dict(
        initial_dirty_amount=float(state["initial_dirty_amount"]),
        remaining_dirty_amount=float(state["remaining_dirty_amount"]),
        cleaning_ratio=float(state["cleaning_ratio"]),
        cleaned_sites=int(np.sum(weights <= 0.1)),
    )


def verify(session, threshold=0.05):
    state = session.dirty_state
    if not state:
        return False, dict(cleaning_ratio=0.0, remaining_dirty_amount=None), "no dirty state"
    remaining = float(state["remaining_dirty_amount"])
    initial = max(float(state["initial_dirty_amount"]), 1e-12)
    ratio = remaining / initial
    surface_z = max(float(p[2]) for p in state["points"])
    tool_z = float(session.ctx.obj_pos("wipe_tool")[2]) if "wipe_tool" in session.parts else 0.0
    tool_xy = np.asarray(session.ctx.obj_pos("wipe_tool")[:2], dtype=float) if "wipe_tool" in session.parts else np.zeros(2)
    dirty_xy = np.asarray(state["points"], dtype=float)[:, :2]
    moved_away = bool(tool_z > surface_z + 0.025 or
                      np.min(np.linalg.norm(dirty_xy - tool_xy[None, :], axis=1)) > 0.04)
    ok = ratio <= threshold and moved_away
    metrics = dict(
        initial_dirty_amount=float(state["initial_dirty_amount"]),
        remaining_dirty_amount=remaining,
        cleaning_ratio=float(1.0 - ratio),
        residual_fraction=ratio,
        tool_moved_away=moved_away,
        threshold=float(threshold),
    )
    return ok, metrics, "cleaning threshold or unobstructed inspection failed" if not ok else ""


def restore_visual(session):
    _sync_visual(session)
