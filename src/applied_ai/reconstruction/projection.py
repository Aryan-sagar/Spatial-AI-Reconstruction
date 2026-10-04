"""Vectorized depth -> 3D back-projection (OpenCV camera frame: x right, y down, z forward)."""
from __future__ import annotations

import numpy as np

from ..domain.models import Intrinsics


def depth_to_camera_points(depth, intrinsics: Intrinsics, depth_scale: float, confidence=None, stride: int = 2,
                           min_depth: float = 0.1, max_depth: float = 10.0, min_confidence: int | None = None,
                           return_pixels: bool = False):
    """depth: raw array (h,w). intrinsics must be in the depth image's pixel grid.
    Returns N x 3 float32 metres (and N x 2 [u,v] pixel coords if return_pixels)."""
    d = np.asarray(depth)
    if d.ndim != 2:
        raise ValueError(f"depth must be 2-D, got {d.shape}")
    sub = d[::stride, ::stride].astype(np.float64) * depth_scale
    v, u = np.mgrid[0:d.shape[0]:stride, 0:d.shape[1]:stride]
    keep = np.isfinite(sub) & (sub >= min_depth) & (sub <= max_depth)
    if confidence is not None and min_confidence is not None:
        c = np.asarray(confidence)
        if c.shape != d.shape:
            raise ValueError(f"confidence shape {c.shape} != depth shape {d.shape}")
        keep &= c[::stride, ::stride] >= min_confidence
    z, uu, vv = sub[keep], u[keep].astype(np.float64), v[keep].astype(np.float64)
    pts = np.column_stack(((uu - intrinsics.cx) * z / intrinsics.fx, (vv - intrinsics.cy) * z / intrinsics.fy, z)).astype(np.float32)
    return (pts, np.column_stack((uu, vv))) if return_pixels else pts
