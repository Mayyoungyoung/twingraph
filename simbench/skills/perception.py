"""Perception skills: noisy part detection + grasp-pose estimation.

Part metadata (type / grasp height / outer diameter) lives in the scene
XML as ``<custom><numeric name="meta_<part>" data="..."/>`` entries
written by the scene generators, so skills never hard-code geometry.
"""
import mujoco
import numpy as np

TYPE_NAMES = {0: "cylinder", 1: "ring", 2: "box", 3: "pin"}


def part_meta(ctx, name):
    """Grasp metadata dict for a scene part (from the custom numerics).

    Raises ValueError when the scene forgot to define ``meta_<name>``.
    """
    nid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_NUMERIC,
                            f"meta_{name}")
    if nid < 0:
        raise ValueError(f"scene {ctx.scene_path} lacks grasp metadata "
                         f"'meta_{name}'")
    adr = ctx.model.numeric_adr[nid]
    size = ctx.model.numeric_size[nid]
    t, dz, od, hh = ctx.model.numeric_data[adr:adr + size]
    return dict(name=name, type=TYPE_NAMES.get(int(t), "?"),
                type_code=int(t), grasp_dz=float(dz),
                outer_d=float(od), half_h=float(hh))


def detect(ctx, name, noise_std=0.0, rng=None):
    """Detect a part: (pos, yaw) with optional Gaussian noise.

    noise_std (m) perturbs the position; the yaw noise is scaled x10
    (rad) so a 3 mm detection error also means ~1.7 deg of heading error.
    """
    rng = np.random if rng is None else rng
    pos, _ = ctx.obj_pose(name)
    yaw = ctx.obj_yaw(name)
    if noise_std > 0.0:
        pos = pos + rng.normal(0.0, noise_std, 3)
        yaw = float(yaw + rng.normal(0.0, 10.0 * noise_std))
    return np.asarray(pos, dtype=float), float(yaw)


def detect_part(ctx, name, faults=None, step=None):
    """Detect a part under the failure model: (found, pos, yaw, outcome).

    outcome is 'ok' | 'miss' | 'false' (see faults.FailureModel).  On a
    miss the reported pose is None; on a false feature lock the position
    carries the random wrong-feature offset on top of the estimation
    noise, so the planner consumes a systematically wrong pose.
    """
    pos, _ = ctx.obj_pose(name)
    yaw = ctx.obj_yaw(name)
    outcome = "ok"
    if faults is not None:
        outcome = faults.detect_outcome(step=step)
        if outcome == "miss":
            return False, None, None, outcome
        pn, yn = faults.detect_noise()
        pos = pos + pn
        yaw = float(yaw + yn)
        if outcome == "false":
            pos = pos + faults.false_offset()
    return True, np.asarray(pos, dtype=float), float(yaw), outcome


def estimate_grasp_pose(ctx, name, noise_std=0.0, rng=None,
                       detect_result=None):
    """Estimate the grasp pose of a part from (noisy) detection + metadata.

    Generates the two axis-aligned approach candidates (gripper closing
    axis along world x or y) and scores them by proximity to the current
    EEF plus a small yaw-mismatch penalty, then returns the best one as
    a dict:  pos (EEF target), yaw (approach), outer_d, type, grasp_dz.

    detect_result: an external observation from ``detect_part`` as
    (found, pos, yaw, outcome); when given, the internal detect() is
    skipped and the estimate is computed from the observation -- this is
    the plan-grasp-pose handshake of the two-class action taxonomy.
    Returns None when the observation is a miss.
    """
    meta = part_meta(ctx, name)
    if detect_result is None:
        pos, yaw = detect(ctx, name, noise_std=noise_std, rng=rng)
    else:
        found, pos, yaw, _ = detect_result
        if not found:
            return None
    pos = np.asarray(pos, dtype=float)
    eef = ctx.eef_pos()
    cands = []
    for ay in (0.0, np.pi / 2.0):
        gp = np.array([pos[0], pos[1], pos[2] + meta["grasp_dz"]])
        score = (-float(np.linalg.norm(gp[:2] - eef[:2]))
                 - 0.05 * abs(_wrap_pi(ay - yaw)))
        cands.append((score, ay, gp))
    _, best_yaw, best_pos = max(cands, key=lambda c: c[0])
    return dict(pos=best_pos, yaw=float(best_yaw), outer_d=meta["outer_d"],
                type=meta["type"], grasp_dz=meta["grasp_dz"],
                half_h=meta["half_h"], confidence=1.0)


def _wrap_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi
