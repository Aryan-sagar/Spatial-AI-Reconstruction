"""Frame and keyframe reconstruction (LiDAR tier)."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from ..calibration.intrinsics import DepthProjectionCalibration
from ..io.dataset import ScanDataset
from ..logging_utils import stage
from .keyframes import select_keyframes
from .pointcloud import PointCloudAccumulator, write_ply
from .poses import T_world_cvcamera, transform_points
from .projection import depth_to_camera_points

log = logging.getLogger(__name__)


def frame_world_points(ds: ScanDataset, fid: int, cfg: dict, stride: int | None = None):
    """(camera_points, world_points, diagnostics) for one frame under the configured conventions."""
    rc = cfg["reconstruction"]
    fr = ds.get_frame(fid, load_rgb=False, load_imu=False, load_confidence=True)
    cal = DepthProjectionCalibration.from_config(cfg)
    Kd = cal.depth_intrinsics(fr.intrinsics)
    scale = fr.depth.depth_scale
    cam, = (depth_to_camera_points(fr.depth.raw_values, Kd, scale.scale, fr.depth.confidence, stride or rc["stride"], rc["min_depth_m"],
                                   rc["max_depth_m"], rc["confidence_min"] if rc["confidence_min"] else None),)
    T = T_world_cvcamera(fr.pose.position, fr.pose.quaternion, cfg["pose"]["convention"], cfg["pose"]["camera_axes"])
    world = transform_points(T, cam)
    diag = {
        "frame_id": fid, "depth_scale": {"value": scale.scale, "unit": scale.unit, "status": scale.status},
        "intrinsics_rgb": vars(fr.intrinsics) | {"distortion_center": list(fr.intrinsics.distortion_center) if fr.intrinsics.distortion_center else None},
        "depth_projection": cal.diagnostics(fr.intrinsics),
        "pose": {"timestamp": fr.pose.timestamp, "position": list(fr.pose.position), "quaternion_xyzw": list(fr.pose.quaternion),
                 "direction": cfg["pose"]["convention"], "camera_axes": cfg["pose"]["camera_axes"]},
        "points": {"count": int(len(cam)), "pixels_total": int(fr.depth.raw_values.size // ((stride or rc["stride"]) ** 2)),
                   "filtered_out": int(fr.depth.raw_values.size // ((stride or rc["stride"]) ** 2) - len(cam))},
        "camera_bounds": {"min": cam.min(0).tolist(), "max": cam.max(0).tolist()} if len(cam) else None,
        "world_bounds": {"min": world.min(0).tolist(), "max": world.max(0).tolist()} if len(cam) else None,
        "reconstruction_config": rc, "warnings": fr.warnings,
    }
    return cam, world, diag


def reconstruct_frame(ds: ScanDataset, fid: int, cfg: dict, out_dir: str | Path) -> dict:
    cam, world, diag = frame_world_points(ds, fid, cfg, stride=1)
    out = Path(out_dir) / "frames"
    out.mkdir(parents=True, exist_ok=True)
    write_ply(out / f"frame_{fid:06d}_camera.ply", cam)
    write_ply(out / f"frame_{fid:06d}_world.ply", world)
    (out / f"frame_{fid:06d}.json").write_text(json.dumps(diag, indent=2, default=str))
    return diag


def keyframe_ids(ds: ScanDataset, cfg: dict) -> list[int]:
    k = cfg["keyframes"]
    df = ds.odometry.df
    return [int(df["frame"].iloc[i]) for i in select_keyframes(df[["x", "y", "z"]].to_numpy(), df[["qx", "qy", "qz", "qw"]].to_numpy(),
                                                                k["min_frame_gap"], k["translation_threshold"], k["rotation_threshold_deg"], k["max_frames"])]


def reconstruct_scene_cloud(ds: ScanDataset, cfg: dict, timings: list | None = None):
    ids = keyframe_ids(ds, cfg)
    rc = cfg["reconstruction"]
    acc = PointCloudAccumulator(rc["voxel_size_m"], rc["max_points_per_batch"])
    with stage("pointcloud", timings, keyframes=len(ids)) as info:
        for fid in ids:
            _, world, _ = frame_world_points(ds, fid, cfg)
            acc.add_frame(world)
        cloud = acc.merge()
        info.update(input_points=acc.raw_points, output_points=int(len(cloud)))
    return cloud, ids, acc.statistics()
