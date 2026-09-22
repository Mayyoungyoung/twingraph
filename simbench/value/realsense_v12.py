"""RealSense RGB-D adapter with explicit, measured camera-to-world calibration.

No simulator import or object pose is used. Frames are aligned to RGB, depth
units come from the device, distortion is rectified, and optical (+Z forward,
+Y down) coordinates are converted to the detector's OpenGL convention.
"""
from pathlib import Path
import json
import numpy as np
import cv2
from .rgbd_perception import CameraCalibration


def sdk_rectification_maps(intrinsics):
    """Use the vendor's projection for non-OpenCV distortion conventions.

    In particular, modified Brown applies tangential terms after the radial
    transform. Treating it as ordinary OpenCV Brown would silently bias pose.
    """
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError("this RealSense distortion requires pyrealsense2 for exact rectification") from exc
    native=rs.intrinsics()
    for key in ("width","height","fx","fy","ppx","ppy"):
        setattr(native,key,intrinsics[key])
    model=str(intrinsics["model"]).split(".")[-1].lower()
    native.model=getattr(rs.distortion,model);native.coeffs=list(intrinsics["coeffs"])
    v,u=np.indices((native.height,native.width))
    xy=np.c_[((u-native.ppx)/native.fx).ravel(),((v-native.ppy)/native.fy).ravel()]
    projected=np.asarray([rs.rs2_project_point_to_pixel(native,[float(x),float(y),1.]) for x,y in xy],np.float32)
    return projected[:,0].reshape(u.shape),projected[:,1].reshape(u.shape)


def rigid_transform(value):
    transform=np.asarray(value,dtype=float)
    if transform.shape != (4,4) or not np.isfinite(transform).all():
        raise ValueError("world_from_optical must be a finite 4x4 measured transform")
    if not np.allclose(transform[3],[0,0,0,1],atol=1e-8):
        raise ValueError("invalid homogeneous transform")
    R=transform[:3,:3]
    if not np.allclose(R.T@R,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(R),1.,atol=1e-5):
        raise ValueError("extrinsic rotation must be a proper orthonormal matrix")
    return transform


def adapt_aligned(rgb, depth, intrinsics, world_from_optical, *, depth_scale_m,
                  source="realsense_aligned_capture"):
    rgb=np.asarray(rgb); depth=np.asarray(depth)
    width,height=int(intrinsics["width"]),int(intrinsics["height"])
    if rgb.shape != (height,width,3) or depth.shape != (height,width):
        raise ValueError("RGB and aligned depth must share the declared color intrinsics")
    if rgb.dtype != np.uint8: raise ValueError("RGB must be uint8 in RGB channel order")
    if not np.isfinite(depth_scale_m) or not 0 < depth_scale_m <= 1:
        raise ValueError("explicit positive depth scale in metres is required")
    if not np.isfinite(depth).all() or np.any(depth < 0):
        raise ValueError("depth contains invalid values; missing samples must be zero")
    depth_m=depth.astype(np.float32)*float(depth_scale_m)
    fx,fy,cx,cy=[float(intrinsics[k]) for k in ("fx","fy","ppx","ppy")]
    if not np.isfinite([fx,fy,cx,cy]).all() or min(fx,fy)<=0:
        raise ValueError("invalid camera intrinsics")
    K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
    coefficients=np.asarray(intrinsics.get("coeffs",[0]*5),float)
    model=str(intrinsics.get("model","none")).split(".")[-1].lower()
    if model not in ("none","brown_conrady","modified_brown_conrady","inverse_brown_conrady"):
        raise ValueError(f"unsupported distortion {model}; rectify using the camera SDK first")
    if coefficients.shape != (5,) or not np.isfinite(coefficients).all():
        raise ValueError("expected five finite Brown-Conrady coefficients")
    rectified=bool(np.any(np.abs(coefficients)>1e-12))
    if rectified:
        if model == "none": raise ValueError("nonzero coefficients with distortion model none")
        if model=="brown_conrady":
            map_x,map_y=cv2.initUndistortRectifyMap(K,coefficients,np.eye(3),K,(width,height),cv2.CV_32FC1)
        else:
            map_x,map_y=sdk_rectification_maps(intrinsics)
        rgb=cv2.remap(rgb,map_x,map_y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        # Nearest-neighbour avoids invented depth at object/background edges.
        depth_m=cv2.remap(depth_m,map_x,map_y,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
    optical=rigid_transform(world_from_optical)
    detector_transform=optical@np.diag([1.,-1.,-1.,1.])
    calibration=CameraCalibration(width,height,fx,fy,cx,cy,tuple(detector_transform.ravel()),
        version="realsense_rgb_aligned_rectified_optical_to_world_v12")
    metadata=dict(source=source,depth_scale_m=float(depth_scale_m),rgb_order="RGB",
        aligned_to="color",input_intrinsics=intrinsics,rectified=rectified,
        world_from_optical=optical.tolist(),calibration_source="external measured extrinsics",
        note="zero depth remains unknown; no simulated pose or per-object coordinates")
    return dict(rgb=rgb,depth_m=depth_m),calibration,metadata


def save_frame(directory,frame,calibration,metadata):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(directory/"frame.npz",**frame)
    (directory/"calibration.json").write_text(json.dumps(calibration.manifest(),indent=2),encoding="utf-8")
    (directory/"capture.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")


def load_frame(directory):
    directory=Path(directory)
    with np.load(directory/"frame.npz",allow_pickle=False) as data:
        frame={k:data[k].copy() for k in ("rgb","depth_m")}
    saved=json.loads((directory/"calibration.json").read_text(encoding="utf-8"))
    calibration=CameraCalibration(**saved)
    rigid_transform(calibration.matrix())
    if frame["rgb"].shape != (calibration.height,calibration.width,3) or frame["depth_m"].shape != frame["rgb"].shape[:2]:
        raise ValueError("recorded frame/calibration dimensions disagree")
    return frame,calibration


def capture(world_from_optical, *, serial=None, bag=None, width=640, height=480, warmup=15):
    # Optional SDK is imported only when a real capture is requested.
    import pyrealsense2 as rs
    rigid_transform(world_from_optical)
    pipeline=rs.pipeline();config=rs.config()
    if bag: config.enable_device_from_file(str(bag),repeat_playback=False)
    else:
        if serial: config.enable_device(str(serial))
        config.enable_stream(rs.stream.depth,width,height,rs.format.z16,30)
        config.enable_stream(rs.stream.color,width,height,rs.format.rgb8,30)
    profile=pipeline.start(config)
    try:
        if bag: profile.get_device().as_playback().set_real_time(False)
        scale=profile.get_device().first_depth_sensor().get_depth_scale()
        align=rs.align(rs.stream.color)
        frames=None
        for _ in range(max(1,int(warmup))): frames=align.process(pipeline.wait_for_frames(5000))
        color=frames.get_color_frame();depth=frames.get_depth_frame()
        if not color or not depth: raise ValueError("RealSense returned no aligned RGB-D pair")
        if depth.profile.format() != rs.format.z16:
            raise ValueError("RealSense depth must be z16 before applying the device depth scale")
        rgb=np.asanyarray(color.get_data()).copy()
        fmt=color.profile.format()
        if fmt==rs.format.bgr8: rgb=rgb[:,:,::-1].copy()
        elif fmt!=rs.format.rgb8: raise ValueError(f"unsupported RGB recording format: {fmt}")
        raw_depth=np.asanyarray(depth.get_data()).copy()
        intr=color.profile.as_video_stream_profile().intrinsics
        intrinsics=dict(width=intr.width,height=intr.height,fx=intr.fx,fy=intr.fy,
            ppx=intr.ppx,ppy=intr.ppy,model=str(intr.model),coeffs=list(intr.coeffs))
        frame,calibration,metadata=adapt_aligned(rgb,raw_depth,intrinsics,world_from_optical,
            depth_scale_m=scale,source="realsense_bag" if bag else "realsense_live")
        metadata.update(timestamp_ms=float(color.get_timestamp()),frame_number=int(color.get_frame_number()),
            device_serial=profile.get_device().get_info(rs.camera_info.serial_number))
        return frame,calibration,metadata
    finally: pipeline.stop()
