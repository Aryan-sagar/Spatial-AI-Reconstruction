"""Typed domain objects (Phase 1 subset: Intrinsics, Pose, DepthScaleConfig, DepthFrame, FrameRecord)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion_center: Optional[tuple[float, float]] = None
    distortion_params: Optional[tuple[float, ...]] = None
    source: str = "unknown"  # e.g. "odometry_per_frame", "camera_matrix_csv"

    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])


@dataclass(frozen=True)
class Pose:
    timestamp: float
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]  # qx, qy, qz, qw
    convention: str  # taken from config; NOT verified against data yet
    quality: Optional[float] = None


@dataclass(frozen=True)
class DepthScaleConfig:
    scale: float
    unit: str
    status: str  # provisional | empirically_calibrated | proven

    @classmethod
    def from_config(cls, cfg: dict) -> "DepthScaleConfig":
        d = cfg["calibration"]["depth_scale"]
        return cls(float(d["value"]), d["unit"], d["status"])


@dataclass
class DepthFrame:
    frame_id: int
    path: Path
    width: int
    height: int
    dtype: str
    depth_scale: DepthScaleConfig
    raw_values: np.ndarray  # uint16, untouched
    confidence: Optional[np.ndarray] = None

    def to_meters(self) -> np.ndarray:
        """raw * scale (float32). Zero stays zero (invalid). Scale may be provisional - check depth_scale.status."""
        return self.raw_values.astype(np.float32) * np.float32(self.depth_scale.scale)


@dataclass
class FrameRecord:
    frame_id: int
    timestamp: float
    rgb_frame_index: int
    depth_frame_index: int
    pose: Pose
    intrinsics: Intrinsics
    depth: DepthFrame
    rgb: Optional[np.ndarray] = None  # BGR uint8
    imu_window: Optional[np.ndarray] = None  # (k, 7): timestamp, a_xyz, alpha_xyz
    warnings: list[str] = field(default_factory=list)
