import numpy as np
import pytest
from applied_ai.calibration.intrinsics import DepthProjectionCalibration
from applied_ai.domain.models import Intrinsics
from applied_ai.reconstruction.poses import T_world_cvcamera, invert_transform, pose_matrix, transform_points
from applied_ai.reconstruction.projection import depth_to_camera_points


def test_intrinsics_scaling():
    rgb = Intrinsics(1600.0, 1600.0, 960.0, 720.0, 1920, 1440)
    d = DepthProjectionCalibration("scaled_rgb_intrinsics", (256, 192)).depth_intrinsics(rgb)
    assert d.fx == pytest.approx(1600 / 7.5) and d.cx == pytest.approx(128.0) and d.cy == pytest.approx(96.0)
    with pytest.raises(ValueError):
        DepthProjectionCalibration("bogus")
    with pytest.raises(NotImplementedError):
        DepthProjectionCalibration("cropped_scaled").depth_intrinsics(rgb)


def test_depth_projection_principal_point_and_known_pixels():
    K = Intrinsics(100.0, 200.0, 8.0, 6.0, 16, 12)
    depth = np.full((12, 16), 2000, np.uint16)
    pts, px = depth_to_camera_points(depth, K, 0.001, stride=1, return_pixels=True)
    assert pts.shape == (192, 3)
    i = int(np.flatnonzero((px[:, 0] == 8) & (px[:, 1] == 6))[0])
    assert np.allclose(pts[i], [0, 0, 2.0])
    j = int(np.flatnonzero((px[:, 0] == 10) & (px[:, 1] == 9))[0])
    assert np.allclose(pts[j], [(10 - 8) * 2.0 / 100, (9 - 6) * 2.0 / 200, 2.0], atol=1e-6)


def test_projection_filters_depth_range_and_confidence():
    K = Intrinsics(100.0, 100.0, 2.0, 2.0, 4, 4)
    d = np.array([[0, 50, 1000, 20000]] * 4, np.uint16)  # 0 m, .05 m, 1 m, 20 m
    p = depth_to_camera_points(d, K, 0.001, stride=1, min_depth=0.1, max_depth=10.0)
    assert p.shape[0] == 4 and np.allclose(p[:, 2], 1.0)
    c = np.array([[0, 0, 2, 2]] * 2 + [[0, 0, 0, 0]] * 2)
    p2 = depth_to_camera_points(d, K, 0.001, confidence=c, stride=1, min_confidence=2)
    assert p2.shape[0] == 2
    assert depth_to_camera_points(d, K, 0.001, stride=2).shape[0] == 2  # stride


def test_pose_transform_and_inverse():
    q = (0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4))  # 90 deg about z
    T = pose_matrix((1, 2, 3), q)
    assert np.allclose(transform_points(T, np.array([[1.0, 0, 0]])), [[1, 3, 3]], atol=1e-9)
    assert np.allclose(invert_transform(T) @ T, np.eye(4), atol=1e-9)
    Tw = T_world_cvcamera((1, 2, 3), q, "world_to_camera", "opencv")
    assert np.allclose(Tw, invert_transform(T), atol=1e-9)


def test_camera_axes_convention_flips_y_and_z():
    T = T_world_cvcamera((0, 0, 0), (0, 0, 0, 1), "camera_to_world", "arkit")
    assert np.allclose(transform_points(T, np.array([[0.0, 1.0, 2.0]])), [[0, -1, -2]])  # CV forward(z) -> ARKit -z
    with pytest.raises(ValueError):
        T_world_cvcamera((0, 0, 0), (0, 0, 0, 1), "sideways", "arkit")
