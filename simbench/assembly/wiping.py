"""Learn a normalized wiping trajectory, then track it with physical contact feedback.

The teacher is a procedural raster, not human demonstrations. Ridge regression
learns smooth radial-basis weights; no policy training or material removal is faked.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import mujoco

POLICY = Path(__file__).parent / "checkpoints" / "wipe_imitation.npz"


def features(phase, centers):
    x = np.exp(-0.5 * ((np.asarray(phase)[:, None] - centers) / 0.014) ** 2)
    return x / np.maximum(x.sum(axis=1, keepdims=True), 1.0e-12)


def teacher(phase):
    knots = np.array([[-1, -1], [1, -1], [1, 0], [-1, 0], [-1, 1], [1, 1]])
    t = np.asarray(phase) * 5
    i = np.minimum(t.astype(int), 4)
    u = t - i
    u = u * u * (3 - 2 * u)
    return knots[i] * (1 - u[:, None]) + knots[i + 1] * u[:, None]


def train(out=POLICY):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(173)
    phase = np.linspace(0, 1, 501)
    centers = np.linspace(0, 1, 101)
    base = teacher(phase)
    demos = np.array(
        [
            base + np.sin(np.pi * phase)[:, None] * rng.normal(0, 0.009, (1, 2))
            for _ in range(24)
        ]
    )
    X = features(phase, centers)
    weights = np.linalg.solve(
        X.T @ X + np.eye(len(centers)) * 1.0e-5, X.T @ demos.mean(axis=0)
    )
    heldout_phase = np.linspace(0, 1, 800)
    prediction = features(heldout_phase, centers) @ weights
    error = np.linalg.norm(prediction - teacher(heldout_phase), axis=1)
    np.savez(out, centers=centers, weights=weights)
    np.savez(out.with_name("wipe_expert_demos.npz"), phase=phase, normalized_xy=demos)
    report = dict(
        method="normalized RBF trajectory imitation with ridge regression",
        teacher="procedural three-pass raster, 24 perturbed trajectories",
        seed=173,
        training_trajectories=24,
        evaluation="800 unseen phase samples; not unseen task shapes",
        normalized_rmse=float(np.sqrt(np.mean(error**2))),
        max_normalized_error=float(error.max()),
        physical_robustness="must be evaluated separately in MuJoCo",
    )
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    return report


def trajectory(n):
    with np.load(POLICY, allow_pickle=False) as p:
        return features(np.linspace(0, 1, n), p["centers"]) @ p["weights"]


def plan(s, part, center, halfspan, surface, height, duration, as_):
    from .library import Result

    center, halfspan = np.asarray(center, float), np.asarray(halfspan, float)
    if (
        center.shape != (2,)
        or halfspan.shape != (2,)
        or not np.all(np.isfinite(np.r_[center, halfspan, height, duration]))
    ):
        return Result(False, reason="finite XY surface parameters required")
    if np.any(halfspan <= 0) or duration < 4 or duration > 120:
        return Result(False, reason="invalid wipe extent or duration")
    if mujoco.mj_name2id(s.ctx.model, mujoco.mjtObj.mjOBJ_BODY, surface) < 0:
        return Result(False, reason="unknown wipe surface")
    xy = center + trajectory(int(duration / s.ctx.control_dt)) * halfspan
    points = np.c_[xy, np.full(len(xy), height)]
    s.artifact(
        as_,
        "wipe_path",
        part,
        points=points.tolist(),
        center=center.tolist(),
        halfspan=halfspan.tolist(),
        surface=surface,
        duration=duration,
        policy=str(POLICY),
        footprint=[0.018, 0.012],
    )
    return Result(
        True, dict(samples=len(points), duration_s=duration, policy=str(POLICY))
    )


def surface_force(s, part, surface):
    a, b = s.ctx.body_id(part), s.ctx.body_id(surface)
    total = 0.0
    for i, c in enumerate(s.ctx.data.contact):
        bodies = set(map(int, s.ctx.model.geom_bodyid[[c.geom1, c.geom2]]))
        names = {s.ctx.model.geom(c.geom1).name, s.ctx.model.geom(c.geom2).name}
        pad_only = part != "wipe_tool" or "wipe_pad" in names
        if bodies == {a, b} and pad_only:
            f = np.zeros(6)
            mujoco.mj_contactForce(s.ctx.model, s.ctx.data, i, f)
            total += max(0.0, f[0])
    return total


def execute(s, part, artifact, target_force, force_limit, minimum_coverage):
    from .library import Result

    if not (0 < target_force < force_limit <= 30 and 0 < minimum_coverage <= 1):
        return Result(False, reason="invalid wiping force or coverage parameters")
    p = s.artifacts[artifact]
    points = np.asarray(p["points"])
    if np.linalg.norm(s.ctx.obj_pos(part) - points[0]) > 0.008:
        return Result(False, reason="move the tool to the bound surface start first")
    center, span = np.asarray(p["center"]), np.asarray(p["halfspan"])
    gx, gy = np.meshgrid(
        np.linspace(-span[0] - 0.015, span[0] + 0.015, 40),
        np.linspace(-span[1] - 0.010, span[1] + 0.010, 12),
    )
    grid = np.c_[gx.ravel(), gy.ravel()] + center
    covered = np.zeros(len(grid), bool)
    forces = []
    errors = []
    trace = []
    correction = 0.0
    last_z = float(s.ctx.obj_pos(part)[2])
    # Live telemetry is consumed by the recorder, not used to alter any geometry.
    s.wipe_telemetry = dict(
        grid=grid,
        covered=covered,
        surface_z=points[0, 2] - 0.004,
        force=0.0,
        coverage=0.0,
        trace=trace,
    )
    for point in points:
        force = surface_force(s, part, p["surface"])
        if force > force_limit:
            return Result(
                False, dict(peak_force_n=force), "surface force limit exceeded"
            )
        if not s.ctx.grasp_contacts(part)["held"]:
            return Result(False, reason="wiping tool grasp lost")
        correction = np.clip(
            correction + (force - target_force) * 0.000001, -0.002, 0.002
        )
        goal = point.copy()
        goal[2] = np.clip(goal[2] + correction, last_z - 0.00001, last_z + 0.00001)
        last_z = float(goal[2])
        s.arm.servo(s.arm.part_target(part, goal))
        actual = s.ctx.obj_pos(part)
        force = surface_force(s, part, p["surface"])
        forces.append(force)
        errors.append(np.linalg.norm(actual[:2] - point[:2]))
        trace.append(actual.tolist())
        if force > 0.15:
            covered |= np.all(
                np.abs(grid - actual[:2]) <= np.asarray(p["footprint"]), axis=1
            )
        s.wipe_telemetry.update(force=force, coverage=float(covered.mean()))
    dirty = {}
    if getattr(s, "dirty_state", None) is not None:
        from simbench.value.cleaning import apply_contact
        dirty = apply_contact(s, trace, forces, p["footprint"], surface_ok=True)
    metrics = dict(
        coverage=float(covered.mean()),
        contact_fraction=float(np.mean(np.asarray(forces) > 0.15)),
        peak_force_n=float(max(forces)),
        mean_force_n=float(np.mean(forces)),
        xy_rmse_m=float(np.sqrt(np.mean(np.square(errors)))),
        policy=str(POLICY),
        **dirty,
    )
    ok = (
        metrics["coverage"] >= minimum_coverage
        and metrics["contact_fraction"] >= 0.55
        and metrics["peak_force_n"] <= force_limit
        and metrics["xy_rmse_m"] < 0.003
    )
    s.artifacts["wipe_result"] = dict(type="feedback", part=part, **metrics)
    return Result(
        ok, metrics, "insufficient physical surface contact, coverage or tracking"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(POLICY))
    a = p.parse_args()
    print(json.dumps(train(a.out), indent=2))
