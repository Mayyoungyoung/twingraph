import numpy as np
import pytest
from simbench.value.realsense_v12 import adapt_aligned,save_frame,load_frame
from simbench.value.rgbd_perception import _backproject


def inputs():
    return np.zeros((3,5,3),np.uint8),np.full((3,5),1000,np.uint16),dict(
        width=5,height=3,fx=100.,fy=100.,ppx=2.,ppy=1.,model="none",coeffs=[0]*5)


def test_optical_depth_direction_scale_and_image_y(tmp_path):
    rgb,depth,intr=inputs();transform=np.eye(4);transform[:3,3]=[.2,.1,.3]
    frame,calibration,metadata=adapt_aligned(rgb,depth,intr,transform,depth_scale_m=.001)
    xyz=_backproject(np.array([2,3]),np.array([1,2]),np.array([1.,1.]),calibration)
    np.testing.assert_allclose(xyz,[[.2,.1,1.3],[.21,.11,1.3]],atol=1e-7)
    save_frame(tmp_path,frame,calibration,metadata)
    restored,cal=load_frame(tmp_path)
    np.testing.assert_array_equal(restored["depth_m"],np.ones((3,5)))
    np.testing.assert_array_equal(cal.matrix(),calibration.matrix())


def test_missing_depth_stays_unknown_and_invalid_calibration_is_rejected():
    rgb,depth,intr=inputs();depth[1,2]=0
    frame,_,_=adapt_aligned(rgb,depth,intr,np.eye(4),depth_scale_m=.001)
    assert frame["depth_m"][1,2]==0
    bad=np.eye(4);bad[0,0]=2
    with pytest.raises(ValueError,match="orthonormal"):
        adapt_aligned(rgb,depth,intr,bad,depth_scale_m=.001)
    intr["model"]="unrecognized_lens"
    with pytest.raises(ValueError,match="unsupported distortion"):
        adapt_aligned(rgb,depth,intr,np.eye(4),depth_scale_m=.001)


def test_unaligned_arrays_and_unknown_scale_are_not_silently_accepted():
    rgb,depth,intr=inputs()
    with pytest.raises(ValueError,match="aligned depth"):
        adapt_aligned(rgb,depth[:,:4],intr,np.eye(4),depth_scale_m=.001)
    with pytest.raises(ValueError,match="positive depth scale"):
        adapt_aligned(rgb,depth,intr,np.eye(4),depth_scale_m=0)
