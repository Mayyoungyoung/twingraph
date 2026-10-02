"""Declared MuJoCo-pose observation for the V20 perception ablation.

Only this observation adapter reads body poses. The independent physical
acceptance evaluators continue to read the actual simulated assembly.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
from scipy.spatial.transform import Rotation

from .cad_rgbd_v12 import receiver_interface


BACKEND = "mujoco_state_pose"


def observe(session, parts):
    config = session.state_observation_config
    position_std = float(config["position_noise_std_m"])
    yaw_std = float(config["yaw_noise_std_rad"])
    if not all(math.isfinite(x) and x >= 0 for x in (position_std, yaw_std)):
        raise ValueError("state observation noise must be finite and nonnegative")
    index = int(getattr(session, "state_observation_index", 0))
    seed = int(config["seed"])
    rng = np.random.default_rng(np.random.SeedSequence([seed, 20260924, index]))
    names = tuple(dict.fromkeys((*parts, "guide_base")))
    rows = {}
    deltas = {}
    yaw_deltas = {}
    for name in names:
        true_position, true_quat = session.ctx.obj_pose(name)
        delta = rng.normal(0., position_std, 3) if position_std else np.zeros(3)
        yaw = float(rng.normal(0., yaw_std)) if yaw_std else 0.
        quat = np.asarray(true_quat, float)
        if yaw:
            rotated = Rotation.from_euler("z", yaw) * Rotation.from_quat(quat[[1, 2, 3, 0]])
            xyzw = rotated.as_quat()
            quat = xyzw[[3, 0, 1, 2]]
        rows[name] = dict(position_m=(np.asarray(true_position) + delta).tolist(),
            quat_wxyz=quat.tolist(), valid=True, quality=1.,
            fit_residual_m=position_std, source_view="mujoco_state", category=name,
            track_id=f"sim-body:{name}", bbox_xyxy=None, mask_area_px=0,
            geometry_agreement={"projected_pixel_size_m": 0.})
        deltas[name] = delta.tolist()
        yaw_deltas[name] = yaw
    fixtures = {"guide_base": rows.pop("guide_base")}
    interface = receiver_interface(rows, fixtures, session.planning_cad)
    payload = dict(schema="twingraph.state_observation.v20.r1", backend=BACKEND,
        objects=rows, fixtures=fixtures, **interface,
        source="MuJoCo body poses plus declared independent position/yaw noise",
        simulator_pose_used=True, color_identity_used=False,
        decision_boundary="pre-twin-or-target-execution",
        config=dict(backend=BACKEND, position_noise_std_m=position_std,
                    yaw_noise_std_rad=yaw_std, observation_index=index))
    payload["sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True,
        allow_nan=False).encode()).hexdigest()
    session.state_observation_index = index + 1
    session.state_position_offsets_m = deltas
    session.state_observation_random_state = dict(seed=seed, observation_index=index,
        random_scheme="numpy SeedSequence(seed,20260924,observation_index)",
        position_offsets_m=deltas, yaw_offsets_rad=yaw_deltas)
    session.set_decision_observation(payload)
    return payload
