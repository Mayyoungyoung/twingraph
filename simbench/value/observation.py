"""Explicit simulated sensing boundary for value-v7.

The backend is deliberately named a simulated sensor proxy.  It uses hidden
MuJoCo state only to generate a noisy observation at one decision boundary;
the value module and candidate generator consume the frozen observation, not
the target rollout state.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np


@dataclass(frozen=True)
class SensorConfig:
    position_noise_std_m: float = 0.001
    yaw_noise_std_rad: float = math.radians(1.0)
    seed: int = 0
    backend: str = "simulated_pose_sensor_proxy"
    version: str = "v7.sensor.1"

    def manifest(self):
        return asdict(self)


def _yaw_from_quat(quat):
    w, x, y, z = np.asarray(quat, dtype=float)
    return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _yaw_quat(yaw):
    return np.asarray([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])


def observe(session, parts, config: SensorConfig):
    if config.position_noise_std_m < 0 or config.yaw_noise_std_rad < 0:
        raise ValueError("sensor noise must be nonnegative")
    rng = np.random.default_rng(np.random.SeedSequence([int(config.seed), 7201]))
    objects = {}
    for part in parts:
        true_pos = np.asarray(session.ctx.obj_pos(part), dtype=float)
        true_quat = np.asarray(session.ctx.obj_pose(part)[1], dtype=float)
        position = true_pos + rng.normal(0.0, config.position_noise_std_m, 3)
        yaw = _yaw_from_quat(true_quat) + float(rng.normal(0.0, config.yaw_noise_std_rad))
        objects[part] = {
            "position_m": position.tolist(),
            "quat_wxyz": _yaw_quat(yaw).tolist(),
            "position_noise_std_m": float(config.position_noise_std_m),
            "yaw_noise_std_rad": float(config.yaw_noise_std_rad),
        }
    payload = {
        "schema": "twingraph.observation.v7",
        "backend": config.backend,
        "config": config.manifest(),
        "objects": objects,
        "source": "simulator_truth_through_declared_sensor_model",
        "decision_boundary": "before_twin_or_target_execution",
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    return payload


def install(session, parts, config: SensorConfig):
    observation = observe(session, parts, config)
    session.set_decision_observation(observation)
    return observation

