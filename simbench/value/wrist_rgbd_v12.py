"""Eye-in-hand RGB-D acquisition with an explicit simulated mounting transform.

Only the robot joint encoders, fixed robot kinematics and declared camera
mount enter calibration. Object poses are not read. The fixed scene cameras
are retained for recordings, never used by this acquisition backend.
"""
from pathlib import Path
import math
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .rgbd_perception import CameraCalibration
from simbench.assembly.control import HOME

CAMERA_NAME = "wrist_rgbd"
MOUNT_BODY = "right_hand"
HAND_FROM_CAMERA = np.array([[1., 0., 0., .08],
                             [0., -1., 0., .08],
                             [0., 0., -1., .02],
                             [0., 0., 0., 1.]])
FOVY_DEG = 70.
DETECTOR_SIZE = (720, 960)


def install_camera(root, directory):
    """Create a per-scene Panda include; the shared robot asset is unchanged."""
    directory = Path(directory).resolve()
    include = root.find("include")
    panda_path = (directory / include.get("file")).resolve()
    panda = ET.parse(panda_path).getroot()
    hand = panda.find(f".//body[@name='{MOUNT_BODY}']")
    if hand is None:
        raise ValueError("Panda hand body is missing")
    ET.SubElement(hand, "camera", name=CAMERA_NAME, mode="fixed",
                  pos=".08 .08 .02", xyaxes="1 0 0 0 -1 0", fovy=str(FOVY_DEG))
    mounted = directory / "panda_with_wrist_rgbd.xml"
    ET.indent(panda, space="  ")
    ET.ElementTree(panda).write(mounted, encoding="unicode")
    include.set("file", mounted.name)
    visual = root.find("visual")
    if visual is None: visual = ET.SubElement(root, "visual")
    global_visual = visual.find("global")
    if global_visual is None: global_visual = ET.SubElement(visual, "global")
    global_visual.set("offheight", str(DETECTOR_SIZE[0]))
    global_visual.set("offwidth", str(DETECTOR_SIZE[1]))


def calibration_from_encoders(ctx, size=(480, 640)):
    height, width = map(int, size) if isinstance(size, (tuple, list)) else (int(size), int(size))
    # Scratch FK contains default free-body coordinates, which are never
    # queried. Only measured arm joints determine the camera transform.
    fk = mujoco.MjData(ctx.model)
    fk.qpos[ctx.arm_qadr] = ctx.arm_qpos
    mujoco.mj_kinematics(ctx.model, fk)
    hand_id = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, MOUNT_BODY)
    world_from_hand = np.eye(4)
    world_from_hand[:3, :3] = fk.xmat[hand_id].reshape(3, 3)
    world_from_hand[:3, 3] = fk.xpos[hand_id]
    transform = world_from_hand @ HAND_FROM_CAMERA
    focal = .5 * height / math.tan(math.radians(FOVY_DEG) / 2.)
    return CameraCalibration(width, height, focal, focal, (width-1)/2., (height-1)/2.,
                             tuple(transform.ravel()), version="simulated-wrist-FK-explicit-mount-v1")


def _prepare_view(session):
    """An empty robot may return to HOME using the normal checked controller.

    A held object never causes a viewpoint motion. HOME captures do not step
    physics. A rejected scan retains the current view, hence can be Unknown.
    """
    ctx = session.ctx
    info = dict(scan_motion=False, scan_status="already_at_observation_home")
    if np.max(np.abs(ctx.arm_qpos-HOME)) <= .012:
        return info
    if session.held is not None:
        return dict(scan_motion=False, scan_status="held_object_current_view_only")
    if any(ctx.grasp_contacts(part)["held"] for part in session.parts):
        return dict(scan_motion=False, scan_status="physical_grasp_current_view_only")
    started = time.perf_counter(); sim_start = float(ctx.data.time)
    check = session.arm.check_joint_path([HOME.copy()], step=.02)
    ok = bool(check["valid"])
    if ok:
        # Use the existing atomic skill contract, logging and recorder.
        # Do not invent a camera-specific executable atom outside the graph.
        ok = bool(session.call("move", target="home").ok)
    info = dict(scan_motion=bool(check["valid"]),
                scan_status="returned_to_observation_home" if ok else "observation_home_unreachable_current_view_only",
                collision_check=check, collision_source="simulator_scene_geometry_oracle",
                sim_seconds=float(ctx.data.time)-sim_start, wall_seconds=time.perf_counter()-started)
    return info


def capture_detector(session, size=(480, 640)):
    """Capture the attached camera at the current robot configuration."""
    ctx = session.ctx
    viewpoint = _prepare_view(session)
    calibration = calibration_from_encoders(ctx, size)
    renderer = mujoco.Renderer(ctx.model, height=calibration.height, width=calibration.width)
    try:
        renderer.disable_depth_rendering(); renderer.update_scene(ctx.data, camera=CAMERA_NAME)
        rgb = renderer.render().copy().astype(np.uint8)
        renderer.enable_depth_rendering(); renderer.update_scene(ctx.data, camera=CAMERA_NAME)
        depth = renderer.render().copy().astype(np.float32)
    finally:
        close = getattr(renderer, "close", None) or getattr(renderer, "free", None)
        if close: close()
    session.last_rgbd_acquisition = dict(
        backend="eye_in_hand_rgbd", camera_names=[CAMERA_NAME],
        mount_body=MOUNT_BODY, hand_from_camera=HAND_FROM_CAMERA.tolist(),
        fovy_deg=FOVY_DEG, mounting_status="declared simulation mount; real hand-eye calibration unavailable",
        calibration_source="robot_joint_encoders_forward_kinematics_and_declared_mount",
        arm_joints_rad=ctx.arm_qpos.tolist(), simulation_time_s=float(ctx.data.time),
        external_camera_used=False, object_pose_used=False, **viewpoint)
    session.artifacts.setdefault("perception_acquisitions", []).append(dict(session.last_rgbd_acquisition))
    return {CAMERA_NAME: dict(rgb=rgb, depth_m=depth)}, {CAMERA_NAME: calibration}
