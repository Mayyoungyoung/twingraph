"""Optional installed-SDK conformance; no physical RealSense is accessed."""
import pytest

pytest.importorskip("pyrealsense2", reason="optional SDK is isolated in vendor/realsense_sdk")
from scripts.validate_realsense_v12 import MODELS, validate_model


@pytest.mark.parametrize("model", MODELS)
def test_nonzero_distortion_with_vendor_projection_and_missing_depth(model):
    result = validate_model(model)
    assert result["camera_opened"] is False
    assert result["adapt_map_max_pixel_error"] < 2.e-5
    assert result["invented_depth_values"] == 0
