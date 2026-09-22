import numpy as np
import pytest

from simbench.assembly.candidates import transfer_route_points


def test_high_home_descends_before_lateral_transfer():
    points=transfer_route_points([.3,0.,1.27],[-.1,-.3,.93],.98)
    np.testing.assert_allclose(points[0],[.3,0.,.98])
    np.testing.assert_allclose(points[1],[-.1,-.3,.98])
    np.testing.assert_allclose(points[2],[-.1,-.3,.93])


def test_route_plane_never_drops_below_target():
    points=transfer_route_points([0.,0.,.8],[.1,.2,1.0],.95)
    assert points[0][2]==pytest.approx(1.)
    assert points[1][2]==pytest.approx(1.)
