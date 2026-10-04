"""Pose -> 4x4 transforms. Both pose directions and both camera-axis conventions are supported;
which one is right is established empirically (see calibration/conventions.py), never assumed silently."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

DIRECTIONS = ("camera_to_world", "world_to_camera")
AXES = {"opencv": np.eye(3), "arkit": np.diag([1.0, -1.0, -1.0])}  # arkit: x right, y up, z backward


def pose_matrix(position, quaternion_xyzw) -> np.ndarray:
    """Raw 4x4 built from the pose fields as stored (rotation R, translation t)."""
    q = np.asarray(quaternion_xyzw, float)
    n = np.linalg.norm(q)
    if n < 1e-8:
        raise ValueError("degenerate quaternion")
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(q / n).as_matrix()  # scipy order is x,y,z,w
    T[:3, 3] = position
    return T


def T_world_cvcamera(position, quaternion_xyzw, direction: str, camera_axes: str) -> np.ndarray:
    """Transform taking points in the OpenCV camera frame (x right, y down, z forward - the frame depth
    back-projection produces) to world coordinates."""
    if direction not in DIRECTIONS:
        raise ValueError(f"pose direction must be one of {DIRECTIONS}, got {direction!r}")
    if camera_axes not in AXES:
        raise ValueError(f"camera_axes must be one of {tuple(AXES)}, got {camera_axes!r}")
    T = pose_matrix(position, quaternion_xyzw)
    if direction == "world_to_camera":
        T = invert_transform(T)
    A = np.eye(4)
    A[:3, :3] = AXES[camera_axes]  # OpenCV cam coords -> native camera coords
    return T @ A


def invert_transform(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ T[:3, :3].T + T[:3, 3]


def relative_motion(T_a: np.ndarray, T_b: np.ndarray) -> tuple[float, float]:
    """(translation distance, rotation angle in degrees) between two camera-to-world transforms."""
    d = np.linalg.norm(T_a[:3, 3] - T_b[:3, 3])
    ang = np.degrees(Rotation.from_matrix(T_a[:3, :3].T @ T_b[:3, :3]).magnitude())
    return float(d), float(ang)
