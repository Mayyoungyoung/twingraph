"""Finite, measured Cartesian Panda commands using the observed panda_py 0.8.1 API.

Nothing connects to a robot at import time. Dry-run is the default. Simulation
joint trajectories and simulated contact forces are deliberately not executable.
The software force/workspace guards supplement the robot's own safety controls;
they are not a certified collision monitor or an emergency stop.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import threading
import time
from typing import Any, Dict, Optional

import numpy as np


class SafetyError(RuntimeError):
    """A preflight condition, measured guard, or task postcondition failed."""


DEFAULT_LIMITS = {
    "max_speed_m_s": 0.03,
    "max_angular_speed_rad_s": 0.15,
    "max_force_n": 15.0,
    "max_torque_nm": 2.0,
    "max_step_translation_m": 0.20,
    "max_step_rotation_rad": 0.50,
    "max_reference_tracking_error_m": 0.015,
    "max_reference_tracking_error_rad": 0.15,
    "max_gripper_width_m": 0.08,
    "max_gripper_speed_m_s": 0.03,
    "max_gripper_force_n": 15.0,
    "max_command_timeout_s": 90.0,
    "controller_frequency_hz": 200.0,
    "feedback_stale_timeout_s": 0.20,
    "observation_max_age_s": 2.0,
    "calibration_max_age_s": 86400.0,
    "position_tolerance_m": 0.003,
    "orientation_tolerance_rad": 0.05,
    "start_position_tolerance_m": 0.005,
    "start_orientation_tolerance_rad": 0.05,
    "max_linear_acceleration_m_s2": 0.05,
    "max_angular_acceleration_rad_s2": 0.25,
    "max_program_timeout_s": 1800.0,
    "start_joint_tolerance_rad": 0.02,
    "start_joint_speed_rad_s": 0.01,
    "start_gripper_tolerance_m": 0.002,
}
# Deliberate hard ceilings for this initial commissioning adapter.
HARD_CEILINGS = {
    "max_speed_m_s": 0.05, "max_angular_speed_rad_s": 0.30,
    "max_force_n": 20.0, "max_torque_nm": 3.0,
    "max_gripper_width_m": 0.08, "max_gripper_speed_m_s": 0.05,
    "max_gripper_force_n": 20.0, "max_command_timeout_s": 120.0,
    "max_reference_tracking_error_m": 0.02,
    "max_reference_tracking_error_rad": 0.20,
    "feedback_stale_timeout_s": 0.5, "observation_max_age_s": 10.0,
    "max_linear_acceleration_m_s2": 0.10,
    "max_angular_acceleration_rad_s2": 0.50,
    "max_program_timeout_s": 3600.0,
    "start_joint_tolerance_rad": 0.05,
    "start_joint_speed_rad_s": 0.03,
    "start_gripper_tolerance_m": 0.003,
}
SUPPORTED = {"move_tcp", "move_path_tcp", "contact_path_tcp", "guarded_move_tcp", "contact_move_tcp", "press_tcp", "gripper_open", "gripper_grasp",
             "verify_pose", "verify_observation"}
EVIDENCE_KEYS = ("request", "layout", "candidate", "twin_result")


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def _frozen_digest(value):
    # Match simbench.value.plan.digest without importing the Python 3.10 twin.
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def required_arm_token(schedule, calibration):
    return "EXECUTE_PANDA:" + canonical_sha256({"schedule": schedule,
                                               "calibration": calibration})[:16]


def _number(value, name, lower=0.0, upper=float("inf"), allow_zero=False):
    if isinstance(value, bool):
        raise SafetyError("{} must be numeric".format(name))
    try:
        value = float(value)
    except (ValueError, TypeError) as exc:
        raise SafetyError("{} must be numeric".format(name)) from exc
    if not math.isfinite(value) or value > upper or value < lower or (
            not allow_zero and value == lower):
        raise SafetyError("{} is outside its allowed range".format(name))
    return value


def transform(value, name="transform"):
    try:
        matrix = np.asarray(value, dtype=float)
    except (ValueError, TypeError) as exc:
        raise SafetyError("{} is not a 4x4 matrix".format(name)) from exc
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise SafetyError("{} must be a finite 4x4 matrix".format(name))
    rotation = matrix[:3, :3]
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8) or not np.allclose(
            rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
                np.linalg.det(rotation), 1.0, atol=1e-5):
        raise SafetyError("{} is not a rigid SE(3) transform".format(name))
    return matrix.copy()


def _rotation_angle(a, b):
    return math.acos(float(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1)))


def _quaternion(rotation):
    """Stable XYZW representation, matching Panda.get_orientation(False)."""
    r = np.asarray(rotation, dtype=float)
    trace = np.trace(r)
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s,
             (r[1, 0] - r[0, 1]) / s, s / 4]
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(max(0, 1 + r[i, i] - r[j, j] - r[k, k])) * 2
        q = [0.0] * 4
        q[i] = s / 4
        q[j] = (r[i, j] + r[j, i]) / s
        q[k] = (r[i, k] + r[k, i]) / s
        q[3] = (r[k, j] - r[j, k]) / s
    q = np.asarray(q)
    return q / np.linalg.norm(q)


def _rotation_from_quaternion(q):
    x, y, z, w = np.asarray(q) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                     [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                     [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]])


def interpolate_pose(start, goal, fraction):
    out = np.eye(4)
    out[:3, 3] = start[:3, 3] + fraction * (goal[:3, 3] - start[:3, 3])
    q0, q1 = _quaternion(start[:3, :3]), _quaternion(goal[:3, :3])
    dot = float(q0 @ q1)
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > .9995:
        q = q0 + fraction * (q1 - q0)
    else:
        angle = math.acos(float(np.clip(dot, -1, 1)))
        q = (math.sin((1-fraction)*angle)*q0 + math.sin(fraction*angle)*q1) / math.sin(angle)
    out[:3, :3] = _rotation_from_quaternion(q)
    return out


def _limits(calibration):
    limits = dict(DEFAULT_LIMITS)
    extras = calibration.get("limits", {})
    unknown = set(extras) - set(limits)
    if unknown:
        raise SafetyError("Unknown limits: {}".format(sorted(unknown)))
    limits.update(extras)
    for key, value in limits.items():
        limits[key] = _number(value, key, upper=HARD_CEILINGS.get(key, float("inf")))
    if not 100 <= limits["controller_frequency_hz"] <= 500:
        raise SafetyError("controller_frequency_hz must be between 100 and 500")
    return limits


def _workspace(calibration, pose_base_ee):
    box = calibration.get("workspace_base", {})
    lo, hi = np.asarray(box.get("min", []), dtype=float), np.asarray(box.get("max", []), dtype=float)
    if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo, hi]).all() or not (lo < hi).all():
        raise SafetyError("A measured workspace_base min/max box is required")
    # Check both controller EE and physical TCP, including rotational TCP offset.
    tcp = pose_base_ee @ transform(calibration["T_ee_tcp"], "T_ee_tcp")
    for name, point in (("EE", pose_base_ee[:3, 3]), ("TCP", tcp[:3, 3])):
        if ((point < lo) | (point > hi)).any():
            raise SafetyError("{} is outside the configured robot-base workspace".format(name))


def _base_ee(pose_world_tcp, calibration):
    return transform(calibration["T_base_world"], "T_base_world") @ transform(
        pose_world_tcp, "pose_world_tcp") @ np.linalg.inv(transform(calibration["T_ee_tcp"], "T_ee_tcp"))


def _tolerances(command, limits):
    return (_number(command.get("position_tolerance_m", limits["position_tolerance_m"]),
                    "position_tolerance_m", upper=0.01),
            _number(command.get("orientation_tolerance_rad", limits["orientation_tolerance_rad"]),
                    "orientation_tolerance_rad", upper=0.15))


def _pose_reached(measured, target, tolerances):
    return (np.linalg.norm(measured[:3, 3] - target[:3, 3]) <= tolerances[0] and
            _rotation_angle(measured[:3, :3], target[:3, :3]) <= tolerances[1])


def _trajectory_duration(start, goal, command, limits):
    distance = float(np.linalg.norm(goal[:3, 3] - start[:3, 3]))
    angle = _rotation_angle(start[:3, :3], goal[:3, :3])
    return max(1.875 * distance / command["speed_m_s"],
               1.875 * angle / command["angular_speed_rad_s"],
               math.sqrt(5.78 * distance / limits["max_linear_acceleration_m_s2"]),
               math.sqrt(5.78 * angle / limits["max_angular_acceleration_rad_s2"]), .05)


def validate_schedule(schedule, calibration):
    """Validate finite commands and return their calibrated EE goals.

    Validation does not certify calibration, collisions, task success, or
    provenance. Execution additionally checks those declared preconditions.
    """
    if schedule.get("schema_version") != 1 or schedule.get("frame") != "twin_world":
        raise SafetyError("Schedule schema_version=1 and frame=twin_world required")
    if calibration.get("schema_version") != 1:
        raise SafetyError("Calibration schema_version=1 required")
    if not schedule.get("task_id") or not schedule.get("candidate_id"):
        raise SafetyError("Schedule must identify task_id and candidate_id")
    limits = _limits(calibration)
    previous = _base_ee(schedule.get("start_pose_world_tcp"), calibration)
    _workspace(calibration, previous)
    commands = schedule.get("commands")
    if not isinstance(commands, list) or not 0 < len(commands) <= 10000:
        raise SafetyError("A nonempty finite command list is required")
    ids, output = set(), []
    acceptance_count = 0
    for index, original in enumerate(commands):
        command = dict(original)
        kind, identifier = command.get("kind"), command.get("id")
        if kind not in SUPPORTED or not isinstance(identifier, str) or not identifier or identifier in ids:
            raise SafetyError("Unsupported command or invalid/duplicate id at {}".format(index))
        ids.add(identifier)
        command["timeout_s"] = _number(command.get("timeout_s", 30), "timeout_s",
                                            upper=limits["max_command_timeout_s"])
        if kind in ("move_path_tcp", "contact_path_tcp"):
            poses = command.get("waypoints_world_tcp", [])
            if not isinstance(poses, list) or not 1 <= len(poses) <= 1000:
                raise SafetyError("Cartesian paths require 1 to 1000 explicit finite waypoints")
            command["speed_m_s"] = _number(command.get("speed_m_s", .003 if kind == "contact_path_tcp" else .01),
                "speed_m_s", upper=limits["max_speed_m_s"])
            command["angular_speed_rad_s"] = _number(command.get("angular_speed_rad_s", .03),
                "angular_speed_rad_s", upper=limits["max_angular_speed_rad_s"])
            command["tolerances"] = _tolerances(command, limits)
            command["waypoint_tolerances"] = (
                _number(command.get("waypoint_position_tolerance_m", .0005), "waypoint_position_tolerance_m", upper=.001),
                _number(command.get("waypoint_orientation_tolerance_rad", .005), "waypoint_orientation_tolerance_rad", upper=.01))
            base_poses, duration, path_start = [], 0., previous.copy()
            for waypoint in poses:
                goal = _base_ee(waypoint, calibration)
                _workspace(calibration, goal)
                if np.linalg.norm(goal[:3, 3] - previous[:3, 3]) > limits["max_step_translation_m"] or _rotation_angle(
                        previous[:3, :3], goal[:3, :3]) > limits["max_step_rotation_rad"]:
                    raise SafetyError("Cartesian path contains an oversized segment")
                for fraction in np.linspace(0, 1, 101):
                    _workspace(calibration, interpolate_pose(previous, goal, float(fraction)))
                duration += _trajectory_duration(previous, goal, command, limits)
                base_poses.append(goal)
                previous = goal
            command["pose_base_ee_waypoints"] = base_poses
            command["duration_s"] = duration
            if command["timeout_s"] <= duration + .1:
                raise SafetyError("Cartesian path timeout is shorter than its bounded complete curve")
            command["min_progress_m"] = _number(command.get("min_progress_m", 0), "min_progress_m", allow_zero=True)
            axis = command.get("progress_axis_world")
            if axis is not None or command["min_progress_m"] > 0:
                axis = np.asarray(axis, dtype=float)
                if axis.shape != (3,) or not np.isfinite(axis).all() or not np.isclose(np.linalg.norm(axis), 1, atol=1e-6):
                    raise SafetyError("Path progress_axis_world must be a finite unit vector")
                command["progress_axis_base"] = transform(calibration["T_base_world"])[:3, :3] @ axis
                advance = float((previous[:3, 3] - path_start[:3, 3]) @ command["progress_axis_base"])
                if advance < command["min_progress_m"]:
                    raise SafetyError("Requested path insertion depth exceeds its final geometric progress")
        elif kind in ("move_tcp", "guarded_move_tcp", "contact_move_tcp", "press_tcp", "verify_pose"):
            goal = _base_ee(command.get("pose_world_tcp"), calibration)
            _workspace(calibration, goal)
            command["pose_base_ee"] = goal
            command["tolerances"] = _tolerances(command, limits)
            if kind != "verify_pose":
                distance = float(np.linalg.norm(previous[:3, 3] - goal[:3, 3]))
                angle = _rotation_angle(previous[:3, :3], goal[:3, :3])
                if distance > limits["max_step_translation_m"] or angle > limits["max_step_rotation_rad"]:
                    raise SafetyError("{} exceeds the maximum single segment".format(identifier))
                command["speed_m_s"] = _number(command.get("speed_m_s", .01), "speed_m_s",
                                                   upper=limits["max_speed_m_s"])
                command["angular_speed_rad_s"] = _number(command.get("angular_speed_rad_s", .05),
                        "angular_speed_rad_s", upper=limits["max_angular_speed_rad_s"])
                # Smooth quintic ramp with bounded speed AND acceleration.
                duration = max(1.875 * distance / command["speed_m_s"],
                               1.875 * angle / command["angular_speed_rad_s"],
                               math.sqrt(5.78 * distance / limits["max_linear_acceleration_m_s2"]),
                               math.sqrt(5.78 * angle / limits["max_angular_acceleration_rad_s2"]), .05)
                if command["timeout_s"] <= duration + .1:
                    raise SafetyError("{} timeout is shorter than the bounded trajectory".format(identifier))
                command["duration_s"] = duration
                for fraction in np.linspace(0, 1, 101):
                    _workspace(calibration, interpolate_pose(previous, goal, float(fraction)))
                if kind in ("guarded_move_tcp", "contact_move_tcp", "press_tcp"):
                    axis = np.asarray(command.get("contact_axis_world", []), dtype=float)
                    if axis.shape != (3,) or not np.isfinite(axis).all() or not np.isclose(np.linalg.norm(axis), 1, atol=1e-6):
                        raise SafetyError("contact_axis_world must be a finite unit vector")
                    axis_base = transform(calibration["T_base_world"] )[:3, :3] @ axis
                    advance = float((goal[:3, 3] - previous[:3, 3]) @ axis_base)
                    if advance <= 0 or angle > 1e-6 or np.linalg.norm(
                            (goal[:3, 3] - previous[:3, 3]) - advance * axis_base) > 1e-5:
                        raise SafetyError("Guarded motion must advance straight along its contact axis")
                    command["contact_axis_base"] = axis_base
                    command["min_progress_m"] = _number(command.get("min_progress_m", 0), "min_progress_m",
                                                          upper=advance, allow_zero=True)
                    if kind != "contact_move_tcp":
                        command["contact_force_n"] = _number(command.get("contact_force_n"), "contact_force_n",
                                                           upper=limits["max_force_n"] * .8)
                    samples = command.get("consecutive_contact_samples", 5)
                    if isinstance(samples, bool) or not isinstance(samples, int) or not 2 <= samples <= 50:
                        raise SafetyError("consecutive_contact_samples must be an integer from 2 to 50")
                    command["consecutive_contact_samples"] = samples
                    if kind == "press_tcp":
                        command["hold_s"] = _number(command.get("hold_s"), "hold_s", upper=5.0)
                        band = command.get("contact_force_band_n", [])
                        if len(band) != 2 or not np.isfinite(band).all() or not 0 < band[0] < band[1] <= limits["max_force_n"] * .9:
                            raise SafetyError("press_tcp requires a finite positive contact force band below the abort limit")
                        if not band[0] <= command["contact_force_n"] <= band[1]:
                            raise SafetyError("Contact trigger must lie inside the press acceptance force band")
                        if command["timeout_s"] <= duration + command["hold_s"] + .1:
                            raise SafetyError("Press timeout is too short for motion plus hold")
                previous = goal
        elif kind in ("gripper_open", "gripper_grasp"):
            command["width_m"] = _number(command.get("width_m"), "width_m",
                                             upper=limits["max_gripper_width_m"], allow_zero=True)
            command["speed_m_s"] = _number(command.get("speed_m_s", .01), "gripper speed_m_s",
                                                  upper=limits["max_gripper_speed_m_s"])
            if kind == "gripper_grasp":
                command["force_n"] = _number(command.get("force_n"), "gripper force_n",
                                                   upper=limits["max_gripper_force_n"])
            command["width_tolerance_m"] = _number(command.get("width_tolerance_m", .003),
                                                         "width_tolerance_m", upper=.005)
        else:
            if not command.get("observation_file"):
                raise SafetyError("verify_observation requires an external measured observation_file")
            if not any(command.get(key) for key in ("expected_objects", "expected_scalars", "expected_flags")):
                raise SafetyError("verify_observation requires measured postconditions")
            for name, expected in command.get("expected_objects", {}).items():
                transform(expected.get("pose_world"), "expected object " + name)
                _tolerances(expected, limits)
            for name, interval in command.get("expected_scalars", {}).items():
                if len(interval) != 2 or not np.isfinite(interval).all() or interval[0] > interval[1]:
                    raise SafetyError("Invalid expected scalar interval: " + name)
            if not all(type(value) is bool for value in command.get("expected_flags", {}).values()):
                raise SafetyError("Expected flags must be boolean")
            if command.get("task_acceptance") is True:
                acceptance_count += 1
                if index != len(commands) - 1:
                    raise SafetyError("Task acceptance must be the last command")
        output.append(command)
    if acceptance_count != 1:
        raise SafetyError("Exactly one final measured task_acceptance checkpoint is required")
    if sum(command["timeout_s"] for command in output) > limits["max_program_timeout_s"]:
        raise SafetyError("Total finite command timeouts exceed max_program_timeout_s")
    return output


def _utc_epoch(value):
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("timezone missing")
        return parsed.timestamp()
    except (ValueError, TypeError) as exc:
        raise SafetyError("An explicit timezone is required for measurement timestamps") from exc


def _inside_root(root, relative):
    path = Path(relative)
    root = Path(root).resolve()
    resolved = (root / path).resolve()
    if path.is_absolute() or resolved == root or root not in resolved.parents:
        raise SafetyError("Artifact/observation paths must remain inside evidence_root")
    return resolved


def _verify_evidence(schedule, evidence_root):
    if evidence_root is None:
        raise SafetyError("Execution requires evidence_root to verify source artifact contents")
    files, expected = schedule.get("artifact_files", {}), schedule.get("evidence", {})
    parsed = {}
    for name in EVIDENCE_KEYS:
        digest = expected.get(name + "_sha256", "")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise SafetyError("Missing/invalid {} evidence hash".format(name))
        path = _inside_root(evidence_root, files.get(name, ""))
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise SafetyError("Cannot read evidence file: {}".format(name)) from exc
        if hashlib.sha256(raw).hexdigest() != digest:
            raise SafetyError("{} evidence contents changed".format(name))
        try:
            parsed[name] = json.loads(raw)
        except (ValueError, UnicodeDecodeError) as exc:
            raise SafetyError("{} evidence is not JSON".format(name)) from exc
    result = parsed["twin_result"]
    if result.get("valid") is not True or result.get("success") is not True or result.get("timeout"):
        raise SafetyError("The bound independent twin result did not validly pass")
    candidate = parsed["candidate"]
    if candidate.get("name", candidate.get("candidate_id")) != schedule["candidate_id"]:
        raise SafetyError("Candidate identity does not match the bound evidence")
    request, layout = parsed["request"], parsed["layout"]
    layout_digest, request_digest = _frozen_digest(layout), _frozen_digest(request)
    if request.get("explicit_layout_sha256") != layout_digest or result.get("explicit_layout_sha256") != layout_digest:
        raise SafetyError("Twin/request layout binding is inconsistent")
    if result.get("request_sha256") != request_digest:
        raise SafetyError("Twin result belongs to a different request")
    if result.get("candidate_name") != schedule["candidate_id"] or result.get("plan_sha256") != candidate.get("plan_sha256"):
        raise SafetyError("Twin result belongs to a different candidate plan")
    if _frozen_digest(candidate.get("complete_candidate_plan_ir")) != candidate.get("plan_sha256"):
        raise SafetyError("Candidate executable plan contents changed")
    if not request.get("runtime_sha256") or result.get("runtime_sha256") != request["runtime_sha256"]:
        raise SafetyError("Twin/request runtime binding is inconsistent")
    pool = [row for row in request.get("candidates", []) if row.get("name") == schedule["candidate_id"]]
    if len(pool) != 1 or _frozen_digest(pool[0]) != _frozen_digest(candidate):
        raise SafetyError("Candidate is not exactly the selected member of the frozen request")
    if layout.get("measured") is not True or layout.get("robot", {}).get("state_measured") is not True:
        raise SafetyError("A nominal layout cannot be used as measured hardware-scene evidence")
    return parsed


def _execution_preflight(schedule, calibration, evidence_root, arm_token):
    if schedule.get("hardware_executable") is not True:
        raise SafetyError("This reference program is not hardware executable: {}".format(
            schedule.get("blocked_stages", schedule.get("blocking_reasons", []))))
    if arm_token != required_arm_token(schedule, calibration):
        raise SafetyError("Use the exact arm token produced by dry-run for these files")
    for name in ("verified", "tcp_verified", "payload_verified", "workspace_verified", "layout_verified"):
        if calibration.get(name) is not True:
            raise SafetyError("Measured calibration flag {} must be true".format(name))
    limits = _limits(calibration)
    age = time.time() - _utc_epoch(calibration.get("measured_at_utc"))
    if age < -1 or age > limits["calibration_max_age_s"]:
        raise SafetyError("Calibration measurement is stale or in the future")
    if calibration.get("collision_review_schedule_sha256") != canonical_sha256(schedule):
        raise SafetyError("Collision clearance must be reviewed for this exact finite schedule")
    if not calibration.get("allowed_observation_sources"):
        raise SafetyError("Configure the trusted measurement sources for task checkpoints")
    transform(calibration.get("expected_F_T_EE"), "expected_F_T_EE")
    _number(calibration.get("expected_total_payload_mass_kg"), "expected_total_payload_mass_kg",
            upper=3.0, allow_zero=True)
    bias = np.asarray(calibration.get("wrench_bias_base", [0] * 6), dtype=float)
    if bias.shape != (6,) or not np.isfinite(bias).all() or np.linalg.norm(bias[:3]) > 5:
        raise SafetyError("wrench_bias_base must be a measured finite six-vector, <=5 N bias")
    evidence = _verify_evidence(schedule, evidence_root)
    declared = transform(evidence["layout"].get("robot", {}).get("T_world_base"), "layout.T_world_base")
    if not np.allclose(np.linalg.inv(declared), transform(calibration["T_base_world"]), atol=1e-6):
        raise SafetyError("Robot base/world transform differs between calibration and twin layout")
    return evidence


class PandaPyBackend:
    """Native Panda adapter; construction connects but never unlocks or recovers.

    This uses APIs inspected on SSH 901: Panda, libfranka.Gripper,
    CartesianImpedance, create_context, stop_controller and read_once. A
    software timeout cannot interrupt a blocked native network call instantly.
    """
    def __init__(self, hostname="192.168.1.2"):
        from panda_py import Panda, controllers, libfranka
        self.panda = Panda(hostname)
        self.gripper = libfranka.Gripper(hostname)
        self.controllers = controllers
        self.libfranka = libfranka
        self.controller = None
        self.nullspace = None
        self.context_handle = None

    def feedback(self):
        state = self.panda.get_state()
        return {
            "pose_base_ee": np.asarray(state.O_T_EE, dtype=float).reshape(4, 4, order="F"),
            "wrench_base": np.asarray(state.O_F_ext_hat_K, dtype=float),
            "state_time_s": float(state.time.to_sec()),
            "q": np.asarray(state.q, dtype=float),
            "dq": np.asarray(state.dq, dtype=float),
            "F_T_EE": np.asarray(state.F_T_EE, dtype=float).reshape(4, 4, order="F"),
            "total_payload_mass_kg": float(state.m_total),
            "errors": bool(state.current_errors),
            "collision": bool(np.any(state.cartesian_collision) or np.any(state.joint_collision)),
            "robot_mode": state.robot_mode,
        }

    def begin(self, pose_base_ee, frequency, timeout_s):
        state = self.feedback()
        if state["robot_mode"] != self.libfranka.RobotMode.kIdle:
            raise SafetyError("Robot must be manually enabled and idle before control")
        self.nullspace = state["q"].copy()
        self.controller = self.controllers.CartesianImpedance(
            impedance=np.diag(getattr(self, "impedance_diagonal", [120., 120., 120., 8., 8., 8.])),
            damping_ratio=1.0, nullspace_stiffness=.2, filter_coeff=1.0)
        self.set_target(pose_base_ee)
        self.panda.start_controller(self.controller)
        self.set_target(pose_base_ee)
        self.context_handle = self.panda.create_context(frequency=frequency, max_runtime=timeout_s)
        return self.context_handle

    def tick(self):
        return self.context_handle.ok()

    def set_target(self, pose_base_ee):
        self.controller.set_control(pose_base_ee[:3, 3],
                                    _quaternion(pose_base_ee[:3, :3]), self.nullspace)

    def stop(self):
        # Hold the current measured pose before stopping the torque controller.
        if self.controller is not None:
            try:
                self.set_target(self.feedback()["pose_base_ee"])
            finally:
                self.panda.stop_controller()
                self.controller = None

    def gripper_action(self, command):
        if command["kind"] == "gripper_grasp":
            ok = self.gripper.grasp(command["width_m"], command["speed_m_s"],
                                    command["force_n"], epsilon_inner=command["width_tolerance_m"],
                                    epsilon_outer=command["width_tolerance_m"])
        else:
            ok = self.gripper.move(command["width_m"], command["speed_m_s"])
        state = self.gripper.read_once()
        return {"ok": bool(ok), "width_m": float(state.width),
                "is_grasped": bool(state.is_grasped)}

    def gripper_feedback(self):
        state = self.gripper.read_once()
        return {"width_m": float(state.width), "is_grasped": bool(state.is_grasped)}

    def stop_gripper(self):
        self.gripper.stop()

    def capture_state(self):
        state = self.feedback()
        grip = self.gripper.read_once()
        state["robot_mode"] = str(state["robot_mode"])
        state["gripper"] = {"width_m": float(grip.width), "max_width_m": float(grip.max_width),
                            "is_grasped": bool(grip.is_grasped)}
        return _json_safe(state)


class _FeedbackGuard:
    def __init__(self, calibration):
        self.calibration, self.limits = calibration, _limits(calibration)
        self.last_robot_time, self.last_change = None, time.monotonic()
        self.commanded_reference = None

    def check(self, state, reference=None):
        pose = transform(state["pose_base_ee"], "measured robot pose")
        _workspace(self.calibration, pose)
        if state.get("errors") or state.get("collision"):
            raise SafetyError("Robot reported an error or collision")
        raw = np.asarray(state["wrench_base"], dtype=float)
        if raw.shape != (6,) or not np.isfinite(raw).all():
            raise SafetyError("Invalid measured force/torque feedback")
        adjusted = raw - np.asarray(self.calibration.get("wrench_bias_base", [0]*6))
        # Bias cannot conceal a large absolute wrench.
        for wrench in (raw, adjusted):
            if np.linalg.norm(wrench[:3]) > self.limits["max_force_n"] or np.linalg.norm(
                    wrench[3:]) > self.limits["max_torque_nm"]:
                raise SafetyError("Measured force/torque guard exceeded")
        robot_time = float(state["state_time_s"])
        if not math.isfinite(robot_time):
            raise SafetyError("Invalid robot feedback timestamp")
        if self.last_robot_time is not None and robot_time < self.last_robot_time:
            raise SafetyError("Robot feedback timestamp moved backwards")
        now = time.monotonic()
        if self.last_robot_time is None or robot_time > self.last_robot_time:
            self.last_robot_time, self.last_change = robot_time, now
        elif now - self.last_change > self.limits["feedback_stale_timeout_s"]:
            raise SafetyError("Robot feedback stopped updating")
        if reference is not None and not _pose_reached(pose, reference, (
                self.limits["max_reference_tracking_error_m"],
                self.limits["max_reference_tracking_error_rad"])):
            raise SafetyError("Reference tracking error guard exceeded")
        return pose, adjusted


def verify_observation(command, schedule, calibration, evidence_root, not_before_epoch):
    path = _inside_root(evidence_root, command["observation_file"])
    try:
        raw = path.read_bytes()
        observation = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise SafetyError("Fresh external observation unavailable: " + str(path)) from exc
    if observation.get("task_id") != schedule["task_id"] or observation.get("command_id") != command["id"]:
        raise SafetyError("Observation is bound to a different task/checkpoint")
    if observation.get("source") not in calibration.get("allowed_observation_sources", []):
        raise SafetyError("Observation source is not configured as a trusted measurement source")
    timestamp = _utc_epoch(observation.get("measured_at_utc"))
    age = time.time() - timestamp
    if timestamp < not_before_epoch or age < -1 or age > _limits(calibration)["observation_max_age_s"]:
        raise SafetyError("Observation must be fresh and measured after this checkpoint started")
    for name, expected in command.get("expected_objects", {}).items():
        measured = transform(observation.get("objects", {}).get(name, {}).get("pose_world"),
                             "observed object " + name)
        target = transform(expected["pose_world"])
        if not _pose_reached(measured, target, _tolerances(expected, _limits(calibration))):
            raise SafetyError("Measured object pose did not pass: " + name)
    for name, bounds in command.get("expected_scalars", {}).items():
        value = observation.get("scalars", {}).get(name)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not bounds[0] <= value <= bounds[1]:
            raise SafetyError("Measured task scalar did not pass: " + name)
    for name, expected in command.get("expected_flags", {}).items():
        if observation.get("flags", {}).get(name) is not expected:
            raise SafetyError("Measured task flag did not pass: " + name)
    return {"measurement_sha256": hashlib.sha256(raw).hexdigest(),
            "source": observation["source"], "measured_at_utc": observation["measured_at_utc"]}


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _measured_tick(backend, guard, reference=None):
    if not backend.tick():
        raise SafetyError("Panda control context ended before the program completed")
    state = backend.feedback()
    pose, wrench = guard.check(state, reference if reference is not None else guard.commanded_reference)
    return pose, wrench


def _set_reference(backend, guard, pose):
    backend.set_target(pose)
    guard.commanded_reference = pose.copy()


def _tcp_progress(actual, measured_origin, axis_base, calibration):
    tcp_transform = transform(calibration["T_ee_tcp"])
    return float(((actual @ tcp_transform)[:3, 3] - (measured_origin @ tcp_transform)[:3, 3]) @ axis_base)


def _motion(command, backend, guard, calibration):
    limits = guard.limits
    measured_start, _ = _measured_tick(backend, guard)
    # Never reset a moving/loaded reference to lagging measured feedback between
    # waypoints; that reset creates an unbounded reference position jump.
    start = guard.commanded_reference.copy() if guard.commanded_reference is not None else measured_start.copy()
    goal = command["pose_base_ee"]
    distance = float(np.linalg.norm(start[:3, 3] - goal[:3, 3]))
    angle = _rotation_angle(start[:3, :3], goal[:3, :3])
    if distance > limits["max_step_translation_m"] or angle > limits["max_step_rotation_rad"]:
        raise SafetyError("Measured start makes this segment exceed its bound")
    duration = max(1.875 * distance / command["speed_m_s"],
                   1.875 * angle / command["angular_speed_rad_s"],
                   math.sqrt(5.78 * distance / limits["max_linear_acceleration_m_s2"]),
                   math.sqrt(5.78 * angle / limits["max_angular_acceleration_rad_s2"]), .05)
    if duration + command.get("hold_s", 0) + .1 >= command["timeout_s"]:
        raise SafetyError("Measured start requires a longer trajectory than the command timeout")
    kind = command["kind"]
    if kind in ("guarded_move_tcp", "contact_move_tcp", "press_tcp"):
        axis = command["contact_axis_base"]
        delta = goal[:3, 3] - start[:3, 3]
        advance = float(delta @ axis)
        if advance <= 0 or np.linalg.norm(delta - advance * axis) > command["tolerances"][0] or angle > .01:
            raise SafetyError("Measured contact approach is not aligned with its calibrated axis")
    began, deadline = time.monotonic(), time.monotonic() + command["timeout_s"]
    reference, contact_count = start.copy(), 0
    contacted, hold_started, peak = False, None, 0.0
    while time.monotonic() < deadline:
        actual, wrench = _measured_tick(backend, guard, reference)
        peak = max(peak, float(np.linalg.norm(wrench[:3])))
        elapsed = time.monotonic() - began
        if kind in ("guarded_move_tcp", "press_tcp"):
            resisting = -float(wrench[:3] @ command["contact_axis_base"])
            progress = _tcp_progress(actual, measured_start, command["contact_axis_base"], calibration)
            if not contacted:
                contact_count = contact_count + 1 if resisting >= command["contact_force_n"] else 0
                if contact_count >= command["consecutive_contact_samples"]:
                    if progress < command["min_progress_m"]:
                        raise SafetyError("Contact was detected before the required insertion depth")
                    contacted = True
                    if kind == "guarded_move_tcp":
                        # End this approach at measured contact, with no automatic further push.
                        _set_reference(backend, guard, reference)
                        return {"contact_detected": True, "measured_progress_m": progress,
                                "resisting_force_n": resisting, "peak_force_n": peak,
                                "pose_base_ee": actual}
                    # press_tcp keeps the last bounded reference; it never advances farther
                    # to chase a lost contact or a missing task acceptance measurement.
                    hold_started = time.monotonic()
            if contacted:
                low, high = command["contact_force_band_n"]
                if not low <= resisting <= high:
                    raise SafetyError("Measured press force left the configured acceptance band")
                _set_reference(backend, guard, reference)
                if time.monotonic() - hold_started >= command["hold_s"]:
                    _set_reference(backend, guard, reference)
                    return {"contact_detected": True, "measured_progress_m": progress,
                            "resisting_force_n": resisting, "hold_s": command["hold_s"],
                            "peak_force_n": peak, "pose_base_ee": actual}
                continue
        if elapsed >= duration:
            if kind in ("guarded_move_tcp", "press_tcp") and _pose_reached(actual, goal, command["tolerances"]):
                raise SafetyError("Contact approach reached its limit without measured contact")
            if kind in ("move_tcp", "contact_move_tcp") and _pose_reached(actual, goal, command["tolerances"]):
                progress = _tcp_progress(actual, measured_start, command["contact_axis_base"], calibration) if kind == "contact_move_tcp" else None
                if progress is not None and progress < command["min_progress_m"]:
                    raise SafetyError("Measured insertion did not reach the required depth")
                _set_reference(backend, guard, goal)
                return {"pose_verified": True, "peak_force_n": peak, "pose_base_ee": actual,
                        "contact_insertion": kind == "contact_move_tcp", "measured_progress_m": progress}
        fraction = min(1.0, max(0.0, elapsed / duration))
        fraction = fraction**3 * (10 - 15*fraction + 6*fraction**2)
        reference = interpolate_pose(start, goal, fraction)
        _workspace(calibration, reference)
        _set_reference(backend, guard, reference)
    raise SafetyError("Measured motion/contact command timed out: " + command["id"])


def _gripper_command(command, backend, guard):
    result, errors = [], []
    actual, _ = _measured_tick(backend, guard)
    hold = guard.commanded_reference.copy() if guard.commanded_reference is not None else actual

    def invoke():
        try:
            result.append(backend.gripper_action(command))
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=invoke, daemon=True)
    worker.start()
    deadline = time.monotonic() + command["timeout_s"]
    try:
        while worker.is_alive():
            _measured_tick(backend, guard, hold)
            _set_reference(backend, guard, hold)
            if time.monotonic() > deadline:
                raise SafetyError("Gripper command timed out")
        if errors:
            raise SafetyError("Gripper native call failed: " + str(errors[0]))
        feedback = result[0]
        if feedback.get("ok") is not True:
            raise SafetyError("Gripper command reported failure")
        if command["kind"] == "gripper_grasp":
            if feedback.get("is_grasped") is not True:
                raise SafetyError("Gripper did not measure a held object")
            if abs(feedback["width_m"] - command["width_m"]) > command["width_tolerance_m"]:
                raise SafetyError("Measured grasp width does not match the configured object width")
        elif abs(feedback["width_m"] - command["width_m"]) > command["width_tolerance_m"]:
            raise SafetyError("Measured gripper opening did not reach the target")
        return feedback
    except BaseException:
        backend.stop_gripper()
        raise


def _motion_path(command, backend, guard, calibration):
    start, _ = _measured_tick(backend, guard)
    deadline, peak = time.monotonic() + command["timeout_s"], 0.
    for goal in command["pose_base_ee_waypoints"]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SafetyError("Complete Cartesian path timed out")
        subcommand = dict(command, kind="move_tcp", pose_base_ee=goal,
                          timeout_s=remaining, tolerances=command["waypoint_tolerances"])
        detail = _motion(subcommand, backend, guard, calibration)
        peak = max(peak, detail["peak_force_n"])
    actual, _ = _measured_tick(backend, guard)
    if not _pose_reached(actual, command["pose_base_ee_waypoints"][-1], command["tolerances"]):
        raise SafetyError("Actual final Cartesian path pose did not pass")
    progress = None
    if "progress_axis_base" in command:
        progress = _tcp_progress(actual, start, command["progress_axis_base"], calibration)
        if progress < command["min_progress_m"]:
            raise SafetyError("Actual Cartesian path insertion depth did not pass")
    return {"pose_verified": True, "pose_base_ee": actual, "peak_force_n": peak,
            "waypoints": len(command["pose_base_ee_waypoints"]), "measured_progress_m": progress,
            "contact_path": command["kind"] == "contact_path_tcp"}


def execute_schedule(schedule, calibration, execute=False, arm_token=None,
                     backend=None, evidence_root=None, journal_path=None):
    """Dry-run by default; actual completion and task acceptance are distinct."""
    digest = canonical_sha256(schedule)
    result = {"schema_version": 1, "task_id": schedule.get("task_id"),
              "candidate_id": schedule.get("candidate_id"), "schedule_sha256": digest,
              "calibration_sha256": canonical_sha256(calibration),
              "dry_run": not execute, "task_accepted": False, "commands": []}
    if not execute and schedule.get("hardware_executable") is not True:
        result.update(status="blocked_reference", blocking_reasons=schedule.get(
            "blocking_reasons", schedule.get("blocked_stages", ["Reference has not been physically bound"])),
            proposed_commands=_json_safe(schedule.get("commands", [])))
        return result
    if execute:
        evidence = _execution_preflight(schedule, calibration, evidence_root, arm_token)
    commands = validate_schedule(schedule, calibration)
    if not execute:
        result.update(status="dry_run", proposed_commands=_json_safe(commands),
                      arm_token=required_arm_token(schedule, calibration),
                      execution_preconditions={key: calibration.get(key, False) for key in
                            ("verified", "tcp_verified", "payload_verified", "workspace_verified", "layout_verified")})
        return result
    limits = _limits(calibration)
    impedance = np.asarray(calibration.get("cartesian_impedance_diagonal", [120, 120, 120, 8, 8, 8]), dtype=float)
    if impedance.shape != (6,) or not np.isfinite(impedance).all() or (impedance <= 0).any() or (
            impedance[:3] > 600).any() or (impedance[3:] > 30).any():
        raise SafetyError("Cartesian stiffness must be positive and <=600 N/m, <=30 Nm/rad")
    # All file/configuration checks happen before creating a hardware connection.
    backend = backend if backend is not None else PandaPyBackend(calibration.get("robot_hostname", "192.168.1.2"))
    guard = _FeedbackGuard(calibration)
    initial = backend.feedback()
    initial_pose, _ = guard.check(initial)
    joints = np.asarray(initial.get("q", []), dtype=float)
    reviewed_joints = np.asarray(evidence["layout"]["robot"].get("joints_rad", []), dtype=float)
    if joints.shape != (7,) or reviewed_joints.shape != (7,) or not np.isfinite([joints, reviewed_joints]).all() or np.max(np.abs(joints-reviewed_joints)) > limits["start_joint_tolerance_rad"]:
        raise SafetyError("Measured robot joints differ from the reviewed twin initial state")
    joint_speed = np.asarray(initial.get("dq", []), dtype=float)
    if joint_speed.shape != (7,) or not np.isfinite(joint_speed).all() or np.max(np.abs(joint_speed)) > limits["start_joint_speed_rad_s"]:
        raise SafetyError("Robot must be stationary before the reviewed task begins")
    grip = backend.gripper_feedback()
    actual_width = float(grip.get("width_m", float("nan")))
    expected_width = float(evidence["layout"]["robot"].get("finger_width_m", float("nan")))
    if not math.isfinite(actual_width) or not math.isfinite(expected_width) or abs(actual_width-expected_width) > limits["start_gripper_tolerance_m"] or grip.get("is_grasped") is not False:
        raise SafetyError("Initial gripper opening/held-object state differs from the reviewed twin")
    expected_start = _base_ee(schedule["start_pose_world_tcp"], calibration)
    if not _pose_reached(initial_pose, expected_start, (limits["start_position_tolerance_m"],
                                                      limits["start_orientation_tolerance_rad"])):
        raise SafetyError("Measured starting pose differs from the reviewed start; no automatic homing")
    if not np.allclose(initial.get("F_T_EE"), transform(calibration["expected_F_T_EE"]), atol=1e-5):
        raise SafetyError("FCI configured tool frame differs from the measured calibration")
    payload = float(initial.get("total_payload_mass_kg", float("nan")))
    if not math.isfinite(payload) or abs(payload - calibration["expected_total_payload_mass_kg"]) > .05:
        raise SafetyError("FCI configured total payload differs from the reviewed payload")
    backend.impedance_diagonal = impedance
    started = False
    try:
        context = backend.begin(initial_pose, limits["controller_frequency_hz"],
                                sum(command["timeout_s"] for command in commands) + 5)
        guard.commanded_reference = initial_pose.copy()
        started = True
        with context:
            for command in commands:
                began = time.time()
                kind = command["kind"]
                if kind in ("move_path_tcp", "contact_path_tcp"):
                    detail = _motion_path(command, backend, guard, calibration)
                elif kind in ("move_tcp", "guarded_move_tcp", "contact_move_tcp", "press_tcp"):
                    detail = _motion(command, backend, guard, calibration)
                elif kind.startswith("gripper_"):
                    detail = _gripper_command(command, backend, guard)
                elif kind == "verify_pose":
                    actual, _ = _measured_tick(backend, guard)
                    if not _pose_reached(actual, command["pose_base_ee"], command["tolerances"]):
                        raise SafetyError("Measured robot pose checkpoint failed")
                    detail = {"pose_verified": True, "pose_base_ee": actual}
                else:
                    deadline = time.monotonic() + command["timeout_s"]
                    actual, _ = _measured_tick(backend, guard)
                    hold = guard.commanded_reference.copy() if guard.commanded_reference is not None else actual
                    last_error, detail, next_read = None, None, 0.0
                    while time.monotonic() < deadline:
                        _measured_tick(backend, guard, hold)
                        _set_reference(backend, guard, hold)
                        if time.monotonic() >= next_read:
                            try:
                                detail = verify_observation(command, schedule, calibration, evidence_root, began)
                                break
                            except SafetyError as exc:
                                last_error = exc
                                next_read = time.monotonic() + .05
                    if detail is None:
                        raise SafetyError("Task measurement checkpoint timed out: {}".format(last_error))
                    if command.get("task_acceptance") is True:
                        result["task_accepted"] = True
                result["commands"].append(_json_safe({"id": command["id"], "kind": kind,
                    "started_at_epoch": began, "ended_at_epoch": time.time(), "detail": detail}))
                if journal_path is not None:
                    Path(journal_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(journal_path).write_text(json.dumps(_json_safe(result), indent=2, allow_nan=False), encoding="utf-8")
        result["status"] = "completed"
    except BaseException as exc:
        result.update(status="failed", task_accepted=False, error="{}: {}".format(type(exc).__name__, exc))
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        # begin() may start the controller before raising (e.g. context setup).
        if started or getattr(backend, "controller", None) is not None:
            backend.stop()
        if journal_path is not None:
            Path(journal_path).parent.mkdir(parents=True, exist_ok=True)
            Path(journal_path).write_text(json.dumps(_json_safe(result), indent=2, allow_nan=False), encoding="utf-8")
    return _json_safe(result)


def _review_waypoints(start, goal, identifier):
    """Small Cartesian segments for review, never a substitute for a joint path."""
    distance = np.linalg.norm(goal[:3, 3] - start[:3, 3])
    angle = _rotation_angle(start[:3, :3], goal[:3, :3])
    count = max(1, int(math.ceil(distance / .15)), int(math.ceil(angle / .35)))
    commands, previous = [], start
    for index in range(1, count + 1):
        next_pose = interpolate_pose(start, goal, index / count)
        step = np.linalg.norm(next_pose[:3, 3] - previous[:3, 3])
        turn = _rotation_angle(previous[:3, :3], next_pose[:3, :3])
        expected_duration = max(1.875 * step / .01, 1.875 * turn / .05,
                                math.sqrt(5.78 * step / .05), math.sqrt(5.78 * turn / .25), .05)
        commands.append({"id": "{}_{:03d}".format(identifier, index), "kind": "move_tcp",
                         "pose_world_tcp": next_pose.tolist(), "speed_m_s": .01,
                         "angular_speed_rad_s": .05, "timeout_s": max(5, expected_duration + 3)})
        previous = next_pose
    return commands


def _trace_cartesian_route(reference, event, mapping):
    """Preserve the full sampled curve, not just a joint path's final pose.

    Approximation tolerance is geometric; timing is recomputed using bounded
    Cartesian ramps. Hence the proposed path needs a new collision review.
    """
    primitive_ids = set(event.get("source_event_ids", [event["id"]]))
    samples = [row for row in reference.get("control_samples", []) if row.get("primitive_id") in primitive_ids]
    if not samples:
        return None
    poses = [transform(event["pose_before_world_eef"]) @ mapping]
    poses.extend(transform(row["pose_world_eef"]) @ mapping for row in samples)
    poses.append(transform(event["pose_after_world_eef"]) @ mapping)
    # A chord-only simplifier erases collinear advance/retract motions. Build
    # noise-scale anchors, then retain every meaningful direction reversal.
    # Timing will change, but wiping/probing/retract geometry must survive.
    positions = np.asarray([point[:3, 3] for point in poses])
    anchors = [0]
    for index in range(1, len(poses)-1):
        previous = anchors[-1]
        if np.linalg.norm(positions[index] - positions[previous]) >= .0005 or _rotation_angle(
                poses[index][:3, :3], poses[previous][:3, :3]) >= .005:
            anchors.append(index)
    anchors.append(len(poses)-1)
    keep = {0, len(poses)-1}
    for first, middle, last in zip(anchors, anchors[1:], anchors[2:]):
        incoming, outgoing = positions[middle]-positions[first], positions[last]-positions[middle]
        if np.linalg.norm(incoming) >= .0005 and np.linalg.norm(outgoing) >= .0005 and float(incoming @ outgoing) < -.1 * np.linalg.norm(incoming)*np.linalg.norm(outgoing):
            direction = incoming / np.linalg.norm(incoming)
            extremum = first + int(np.argmax((positions[first:last+1]-positions[first]) @ direction))
            keep.add(extremum)
    mandatory = sorted(keep)
    stack = list(zip(mandatory, mandatory[1:]))
    while stack:
        first, last = stack.pop()
        if last - first < 2:
            continue
        delta = poses[last][:3, 3] - poses[first][:3, 3]
        squared = float(delta @ delta)
        worst_error, worst_index = 0.0, None
        for index in range(first+1, last):
            fraction = float(np.clip((poses[index][:3, 3] - poses[first][:3, 3]) @ delta / squared, 0, 1)) if squared > 1e-12 else (index-first)/(last-first)
            chord = interpolate_pose(poses[first], poses[last], fraction)
            error = max(float(np.linalg.norm(chord[:3, 3] - poses[index][:3, 3])) / .0005,
                        _rotation_angle(chord[:3, :3], poses[index][:3, :3]) / .005)
            if error > worst_error:
                worst_error, worst_index = error, index
        if worst_error > 1:
            keep.add(worst_index)
            stack.extend(((first, worst_index), (worst_index, last)))
    route, previous = [], poses[0]
    for index, retained in enumerate(sorted(keep)[1:]):
        goal = poses[retained]
        if np.linalg.norm(goal[:3, 3] - previous[:3, 3]) < 1e-7 and _rotation_angle(goal[:3, :3], previous[:3, :3]) < 1e-6:
            continue
        route.extend(_review_waypoints(previous, goal, "{}_curve_{:03d}".format(event["id"], index)))
        previous = goal
    return route


def _coalesce_servo_streams(events):
    grouped, index = [], 0
    while index < len(events):
        first = events[index]
        if first.get("primitive") != "servo":
            grouped.append(first)
            index += 1
            continue
        group = [first]
        key = (first.get("stage"), first.get("skill"), canonical_sha256(first.get("call_stack", [])))
        end = index + 1
        while end < len(events):
            candidate = events[end]
            candidate_key = (candidate.get("stage"), candidate.get("skill"), canonical_sha256(candidate.get("call_stack", [])))
            if candidate.get("primitive") != "servo" or candidate_key != key:
                break
            group.append(candidate)
            end += 1
        if len(group) == 1:
            grouped.append(first)
        else:
            stream = dict(first)
            stream.update(id=first["id"] + "_to_" + group[-1]["id"], primitive="servo_stream",
                          source_event_ids=[event["id"] for event in group], coalesced_event_count=len(group),
                          pose_after_world_eef=group[-1]["pose_after_world_eef"],
                          destination_world_eef=group[-1]["destination_world_eef"],
                          end_sim_time_s=group[-1].get("end_sim_time_s"))
            grouped.append(stream)
        index = end
    return grouped


def _path_proposal(route, event, kind, calibration, mapping):
    command = {"id": event["id"], "kind": kind,
               "waypoints_world_tcp": [item["pose_world_tcp"] for item in route],
               "speed_m_s": .003 if kind == "contact_path_tcp" else .01,
               "angular_speed_rad_s": .03 if kind == "contact_path_tcp" else .05,
               "curve_position_tolerance_m": .0005, "curve_orientation_tolerance_rad": .005}
    current = _base_ee((transform(event["pose_before_world_eef"]) @ mapping).tolist(), calibration)
    duration = 0.
    for point in command["waypoints_world_tcp"]:
        goal = _base_ee(point, calibration)
        duration += _trajectory_duration(current, goal, command, _limits(calibration))
        current = goal
    command["timeout_s"] = max(5, duration + 3)
    return command


def _split_free_path(command, event, calibration, mapping):
    """Finite free-path chunks; contact depth always keeps its whole origin."""
    budget = min(_limits(calibration)["max_command_timeout_s"], 90.) - 3
    current = _base_ee((transform(event["pose_before_world_eef"]) @ mapping).tolist(), calibration)
    chunks, waypoints, duration = [], [], 0.
    for point in command["waypoints_world_tcp"]:
        goal = _base_ee(point, calibration)
        step_time = _trajectory_duration(current, goal, command, _limits(calibration))
        if step_time > budget:
            raise SafetyError("A free-path segment requires further reviewed geometric subdivision")
        if waypoints and duration + step_time > budget:
            chunks.append(dict(command, id=command["id"] + "_chunk_{:03d}".format(len(chunks)),
                               waypoints_world_tcp=waypoints, timeout_s=max(5, duration+3)))
            waypoints, duration = [], 0.
        waypoints.append(point)
        duration += step_time
        current = goal
    if waypoints:
        chunks.append(dict(command, id=command["id"] if not chunks else command["id"] + "_chunk_{:03d}".format(len(chunks)),
                           waypoints_world_tcp=waypoints, timeout_s=max(5, duration+3)))
    return chunks


def _stationary_joint_checkpoint(event, mapping):
    return {"id": event["id"], "kind": "verify_pose", "timeout_s": 1,
            "pose_world_tcp": (transform(event["pose_after_world_eef"]) @ mapping).tolist(),
            "position_tolerance_m": .0005, "orientation_tolerance_rad": .005,
            "stationary_sim_joint_posture_change_discarded": True,
            "requires_route_review": True}


def compile_reference_schedule(reference, calibration):
    """Produce a complete editable stage program from a measured twin trace.

    Cartesian endpoints are inspectable proposals. Joint/feedback paths require
    explicit physical stage/route bindings; physical contact and gripper settings
    are never inferred from simulated forces. Only reviewed bindings can become
    hardware_executable. This compiler does not certify robot link collisions.
    """
    blockers, commands, stages = [], [], []
    mapping = transform(calibration.get("T_sim_eef_tcp", np.eye(4)), "T_sim_eef_tcp")
    if calibration.get("sim_eef_tcp_verified") is not True:
        blockers.append("Measure and verify T_sim_eef_tcp; grip_site is not automatically the FCI EE")
    start = transform(reference.get("start_pose_world_eef"), "start_pose_world_eef") @ mapping
    events = reference.get("events", [])
    # Wrapper move/servo events may contain an actual executed joint primitive.
    # Keep the inner primitive, so the compiler never executes a path twice or
    # silently replaces that joint path with the wrapper's endpoint.
    parent_ids = {event.get("parent_primitive_id") for event in events if event.get("parent_primitive_id")}
    events = [event for event in events if event.get("id") not in parent_ids]
    events = _coalesce_servo_streams(events)
    if not events:
        blockers.append("No instrumented twin control events are available")
    if reference.get("twin_validated") is not True:
        blockers.append("Reference twin validation did not pass")
    groups = []
    for event in events:
        stage = str(event.get("stage", "unassigned"))
        if not groups or groups[-1][0] != stage:
            groups.append((stage, []))
        groups[-1][1].append(event)
    reviewed = set(calibration.get("reviewed_event_ids", []))
    reviewed_stages = set(calibration.get("reviewed_stage_ids", []))
    bindings = calibration.get("stage_bindings", {})
    route_bindings = calibration.get("joint_route_bindings", {})
    contact_profiles = calibration.get("contact_profiles", {})
    grip_profiles = calibration.get("gripper_profiles", {})
    observation_profiles = calibration.get("observation_checkpoints", {})
    for group_index, (stage, group) in enumerate(groups):
        stage_id = "stage_{:03d}_{}".format(group_index, stage)
        start_index = len(commands)
        binding = bindings.get(stage_id, bindings.get(stage))
        if binding is not None:
            if binding.get("reviewed") is not True:
                blockers.append("Review the measured task-specific stage binding: " + stage_id)
            bound = binding.get("commands", [])
            if not bound:
                blockers.append("Stage binding has no finite commands: " + stage_id)
            for index, original in enumerate(bound):
                command = dict(original)
                command.setdefault("id", "{}_bound_{:03d}".format(stage_id, index))
                command["source_stage"] = stage_id
                commands.append(command)
        else:
            for event in group:
                primitive, event_id = event.get("primitive"), str(event.get("id"))
                if event_id not in reviewed and stage not in reviewed_stages and stage_id not in reviewed_stages:
                    blockers.append("Review hardware motion/contact binding: " + event_id)
                if primitive == "move":
                    before = transform(event["pose_before_world_eef"]) @ mapping
                    goal = transform(event["destination_world_eef"]) @ mapping
                    for command in _review_waypoints(before, goal, event_id):
                        command.update(source_event_id=event_id, source_stage=stage_id)
                        commands.append(command)
                elif primitive in ("execute_joint", "execute_joint_waypoints"):
                    route = route_bindings.get(event_id)
                    trace_route = _trace_cartesian_route(reference, event, mapping)
                    if not route or route.get("reviewed") is not True:
                        blockers.append("Supply a reviewed Cartesian route for the simulated joint path: " + event_id)
                        if trace_route is not None:
                            if not trace_route:
                                command = _stationary_joint_checkpoint(event, mapping)
                                command.update(source_event_id=event_id, source_stage=stage_id)
                                commands.append(command)
                                continue
                            command = _path_proposal(trace_route, event, "move_path_tcp", calibration, mapping)
                            command.update(source_event_id=event_id, source_stage=stage_id, requires_route_review=True)
                            commands.extend(_split_free_path(command, event, calibration, mapping))
                        else:
                            commands.append({"id": event_id, "kind": "unbound_joint_route",
                                "source_event_id": event_id, "source_stage": stage_id,
                                "pose_before_world_tcp": (transform(event["pose_before_world_eef"]) @ mapping).tolist(),
                                "pose_after_world_tcp": (transform(event["pose_after_world_eef"]) @ mapping).tolist(),
                                "required": "Full sampled curve or explicit reviewed measured waypoints; no endpoint/joint replay"})
                    else:
                        if route.get("use_trace_cartesian_proposal") is True and trace_route is not None:
                            originals = _split_free_path(_path_proposal(trace_route, event, "move_path_tcp", calibration, mapping), event, calibration, mapping) if trace_route else [_stationary_joint_checkpoint(event, mapping)]
                        else:
                            originals = route.get("commands", [])
                        if not originals:
                            blockers.append("Reviewed route is missing finite commands/control samples: " + event_id)
                        for index, original in enumerate(originals or []):
                            command = dict(original)
                            command.setdefault("id", "{}_route_{:03d}".format(event_id, index))
                            command.update(source_event_id=event_id, source_stage=stage_id, requires_route_review=False)
                            commands.append(command)
                elif primitive in ("servo", "servo_stream"):
                    profile = contact_profiles.get(event_id, contact_profiles.get(stage))
                    if not profile or profile.get("reviewed") is not True:
                        blockers.append("Supply a measured contact/feedback profile for: " + event_id)
                    goal = transform(event["destination_world_eef"]) @ mapping
                    before = transform(event["pose_before_world_eef"]) @ mapping
                    delta = goal[:3, 3] - before[:3, 3]
                    direction = delta / np.linalg.norm(delta) if np.linalg.norm(delta) > 1e-8 else np.zeros(3)
                    command = {"id": event_id, "kind": "contact_move_tcp", "pose_world_tcp": goal.tolist(),
                        "contact_axis_world": direction.tolist(), "min_progress_m": max(0, float(np.linalg.norm(delta)) - .001),
                        "speed_m_s": .003, "angular_speed_rad_s": .03, "timeout_s": 90,
                        "source_event_id": event_id, "source_stage": stage_id}
                    if primitive == "servo_stream":
                        curve = _trace_cartesian_route(reference, event, mapping)
                        if curve:
                            command = _path_proposal(curve, event, "contact_path_tcp", calibration, mapping)
                            command.update(source_event_id=event_id, source_stage=stage_id,
                                           source_event_ids=event["source_event_ids"])
                        else:
                            blockers.append("Servo stream has no complete TCP curve: " + event_id)
                    if profile:
                        command.update({key: value for key, value in profile.items() if key != "reviewed"})
                    if command["kind"] in ("press_tcp", "guarded_move_tcp", "contact_move_tcp"):
                        command.setdefault("pose_world_tcp", goal.tolist())
                        command.setdefault("contact_axis_world", direction.tolist())
                    commands.append(command)
                elif primitive in ("open", "close"):
                    profile = grip_profiles.get(event_id, grip_profiles.get(stage))
                    if not profile or profile.get("reviewed") is not True:
                        blockers.append("Set measured gripper width/force profile: " + event_id)
                    command = {"id": event_id, "kind": "gripper_open" if primitive == "open" else "gripper_grasp",
                        "width_m": None, "speed_m_s": .01, "timeout_s": 10,
                        "source_event_id": event_id, "source_stage": stage_id}
                    if primitive == "close":
                        command["force_n"] = None
                    if profile:
                        command.update({key: value for key, value in profile.items() if key != "reviewed"})
                    commands.append(command)
                else:
                    blockers.append("Unsupported physical primitive: " + str(primitive))
        checkpoint = dict(observation_profiles.get(stage_id, observation_profiles.get(stage, {})))
        if not checkpoint:
            blockers.append("Set fresh measured stage postconditions: " + stage_id)
            checkpoint["expected_flags"] = {stage + "_stage_passed": True}
        checkpoint.update(id=stage_id + "_observed", kind="verify_observation",
                          observation_file="observations/{}.json".format(stage_id),
                          timeout_s=checkpoint.get("timeout_s", 60), source_stage=stage_id)
        commands.append(checkpoint)
        stages.append({"id": stage_id, "part": stage, "command_start": start_index,
                       "command_end": len(commands), "source_event_ids": [event["id"] for event in group]})
    final = dict(calibration.get("final_postconditions", {}))
    if not final:
        blockers.append("Configure independent measured final assembly postconditions")
        final["expected_flags"] = {"assembly_accepted": True}
    final.update(id="final_assembly_observed", kind="verify_observation", task_acceptance=True,
                 observation_file="observations/final_assembly.json", timeout_s=final.get("timeout_s", 60))
    commands.append(final)
    schedule = {"schema_version": 1, "frame": "twin_world",
                "task_id": reference.get("task_id", "assembly_v34"),
                "candidate_id": reference.get("candidate_id"),
                "start_pose_world_tcp": start.tolist(), "commands": commands,
                "stages": stages, "evidence": reference.get("evidence", {}),
                "artifact_files": reference.get("artifact_files", {}),
                "reference_sha256": canonical_sha256(reference),
                "blocking_reasons": list(dict.fromkeys(blockers)), "hardware_executable": False}
    total_time = sum(command.get("timeout_s", 0) for command in commands)
    longest = max((command.get("timeout_s", 0) for command in commands), default=0)
    schedule["required_limits"] = {"max_program_timeout_s": math.ceil(total_time),
                                    "max_command_timeout_s": math.ceil(longest)}
    for name, needed in schedule["required_limits"].items():
        if needed > HARD_CEILINGS[name]:
            schedule["blocking_reasons"].append("Generated task exceeds hard {} ceiling; redesign/revalidate the stage program".format(name))
        elif needed > _limits(calibration)[name]:
            schedule["blocking_reasons"].append("Review/configure finite {} >= {} for this complete low-speed program".format(name, needed))
    if not schedule["blocking_reasons"]:
        try:
            validate_schedule(schedule, calibration)
        except (SafetyError, ValueError, TypeError) as exc:
            schedule["blocking_reasons"].append("Finite-command validation failed: " + str(exc))
        else:
            schedule["hardware_executable"] = True
    return schedule


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(_json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    capture = sub.add_parser("capture-state", help="Read current FCI state/gripper only; no controller/motion/recovery")
    capture.add_argument("--hostname", default="192.168.1.2")
    capture.add_argument("--output", required=True)
    compile_parser = sub.add_parser("compile", help="Prepare an editable, blocked-by-default task command schedule")
    compile_parser.add_argument("--reference", required=True)
    compile_parser.add_argument("--calibration", required=True)
    compile_parser.add_argument("--output", required=True)
    run = sub.add_parser("run", help="Validate/dry-run; --execute and exact --arm token required for hardware")
    run.add_argument("--schedule", required=True)
    run.add_argument("--calibration", required=True)
    run.add_argument("--evidence-root")
    run.add_argument("--report", required=True)
    run.add_argument("--execute", action="store_true")
    run.add_argument("--arm")
    args = parser.parse_args(argv)
    try:
        if args.command == "capture-state":
            result = {"schema_version": 1, "measurement_source": "panda_py_fci_read_only",
                      "measured_at_utc": datetime.now(timezone.utc).isoformat(),
                      "hostname": args.hostname, "state": PandaPyBackend(args.hostname).capture_state(),
                      "motion_started": False}
            _write_json(args.output, result)
        elif args.command == "compile":
            result = compile_reference_schedule(_read_json(args.reference), _read_json(args.calibration))
            _write_json(args.output, result)
        else:
            result = execute_schedule(_read_json(args.schedule), _read_json(args.calibration),
                        execute=args.execute, arm_token=args.arm, evidence_root=args.evidence_root,
                        journal_path=args.report if args.execute else None)
            _write_json(args.report, result)
        print(json.dumps({key: result[key] for key in ("status", "hardware_executable", "blocking_reasons",
                            "arm_token", "task_accepted", "motion_started") if key in result}, ensure_ascii=False, indent=2))
        return 2 if result.get("status") == "failed" else 0
    except (SafetyError, OSError, ValueError, TypeError, KeyError) as exc:
        failure = {"status": "blocked", "task_accepted": False,
                   "error": "{}: {}".format(type(exc).__name__, exc)}
        if args.command == "run":
            # A failed new attempt must not leave an older successful report.
            try:
                _write_json(args.report, failure)
            except OSError as report_error:
                failure["report_error"] = str(report_error)
        print(json.dumps(failure, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
