"""ScanDataset: random access to a synchronized FrameRecord for any frame id (read-only)."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..calibration.synchronization import FrameSynchronizer
from ..domain.models import DepthFrame, DepthScaleConfig, FrameRecord, Intrinsics, Pose
from .depth import list_depth_frames, read_depth_raw
from .imu import load_camera_matrix, load_imu
from .inventory import DatasetInventory
from .odometry import load_odometry
from .rgb import RGBVideoReader

log = logging.getLogger(__name__)


class ScanDataset:
    def __init__(self, path: str | Path, cfg: dict, check_sync: bool = True):
        self.root, self.cfg = Path(path), cfg
        self.files = cfg["dataset"]["files"]
        self.check_sync = check_sync
        self.warnings: list[str] = []
        self._rgb: RGBVideoReader | None = None

    def open(self) -> "ScanDataset":
        inv = DatasetInventory(self.cfg)
        inv.scan(self.root)
        if not inv.validate():
            errs = "; ".join(f"{i['code']}: {i['message']}" for i in inv.issues if i["severity"] == "error")
            raise RuntimeError(f"cannot open scan {self.root}: {errs}")
        self.inventory = inv.report()
        if self.check_sync:  # raises SynchronizationError when streams disagree (if strict)
            self.sync_report = FrameSynchronizer(self.cfg).check(self.inventory)
            if self.sync_report["status"] != "consistent":
                from ..calibration.synchronization import SynchronizationError
                failed = "; ".join(c["detail"] for c in self.sync_report["checks"] if not c["pass"])
                if self.cfg["synchronization"]["strict"]:
                    raise SynchronizationError(f"streams disagree: {failed}")
                self.warnings.append(f"sync inconsistent: {failed}")
        self.odometry = load_odometry(self.root / self.files["odometry"], self.cfg["inventory"]["quaternion_norm_tolerance"])
        self._odo = self.odometry.df.set_index("frame")
        _, self._depth_paths, _ = list_depth_frames(self.root / self.files["depth_dir"])
        _, self._conf_paths, _ = list_depth_frames(self.root / self.files["confidence_dir"])
        self.imu = load_imu(self.root / self.files["imu"])
        self.camera_matrix = load_camera_matrix(self.root / self.files["camera_matrix"])
        self.depth_scale = DepthScaleConfig.from_config(self.cfg)
        o_ts = self._odo["timestamp"]
        if self.imu["timestamp"].max() < o_ts.min() or self.imu["timestamp"].min() > o_ts.max():
            self.warnings.append("IMU and odometry timestamp ranges do not overlap; clocks may differ")
        self._rgb = RGBVideoReader(self.root / self.files["rgb"]).open()
        self.rgb_meta = self._rgb.metadata()
        return self

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        if self._rgb:
            self._rgb.close()
            self._rgb = None

    @property
    def frame_ids(self) -> list[int]:
        return self._odo.index.tolist()

    def __len__(self) -> int:
        return len(self._odo)

    def _check_id(self, fid: int) -> None:
        if fid not in self._odo.index:
            raise IndexError(f"frame {fid} not in odometry (valid {self._odo.index.min()}..{self._odo.index.max()})")
        if fid not in self._depth_paths:
            raise IndexError(f"frame {fid}: depth file missing")

    def get_pose(self, fid: int) -> Pose:
        self._check_id(fid)
        r = self._odo.loc[fid]
        return Pose(float(r["timestamp"]), (float(r["x"]), float(r["y"]), float(r["z"])),
                    (float(r["qx"]), float(r["qy"]), float(r["qz"]), float(r["qw"])), self.cfg["pose"]["convention"])

    def get_intrinsics(self, fid: int) -> Intrinsics:
        """Per-frame RGB-space intrinsics from odometry (primary runtime source)."""
        self._check_id(fid)
        r = self._odo.loc[fid]
        dc = (float(r["distortion_center_x"]), float(r["distortion_center_y"])) if "distortion_center_x" in r.index else None
        return Intrinsics(float(r["fx"]), float(r["fy"]), float(r["cx"]), float(r["cy"]),
                          self.rgb_meta["width"], self.rgb_meta["height"], dc, None, "odometry_per_frame")

    def get_depth(self, fid: int, load_confidence: bool = True) -> DepthFrame:
        self._check_id(fid)
        dcfg = self.cfg["dataset"]["depth"]
        raw = read_depth_raw(self._depth_paths[fid], (dcfg["height"], dcfg["width"]), f"{fid:06d}")
        conf = None
        if load_confidence and fid in self._conf_paths:
            import cv2
            conf = cv2.imread(str(self._conf_paths[fid]), cv2.IMREAD_UNCHANGED)  # semantics of values not assumed
        return DepthFrame(fid, self._depth_paths[fid], raw.shape[1], raw.shape[0], str(raw.dtype), self.depth_scale, raw, conf)

    def get_imu_window(self, timestamp: float) -> np.ndarray:
        w = float(self.cfg["frames"]["imu_window_sec"])
        t = self.imu["timestamp"].to_numpy()
        sel = self.imu[(t >= timestamp - w) & (t <= timestamp + w)]
        return sel.to_numpy(dtype=float)

    def get_frame(self, fid: int, load_rgb: bool = True, load_imu: bool = True, load_confidence: bool = True) -> FrameRecord:
        """FrameBundle for frame `fid` under the index-aligned hypothesis (RGB[i], depth[i], odometry[i])."""
        pose = self.get_pose(fid)
        rgb = self._rgb.get_frame(fid) if load_rgb else None
        return FrameRecord(
            frame_id=fid, timestamp=pose.timestamp, rgb_frame_index=fid, depth_frame_index=fid,
            pose=pose, intrinsics=self.get_intrinsics(fid), depth=self.get_depth(fid, load_confidence),
            rgb=rgb, imu_window=self.get_imu_window(pose.timestamp) if load_imu else None,
            warnings=list(self.warnings),
        )

    def iter_frames(self, frame_ids=None, **kw):
        for fid in (self.frame_ids if frame_ids is None else sorted(frame_ids)):
            yield self.get_frame(fid, **kw)
