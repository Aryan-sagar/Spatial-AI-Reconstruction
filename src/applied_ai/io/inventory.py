"""DatasetInventory: scan -> validate -> report. Never modifies raw data."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .depth import DepthFormatError, list_depth_frames, read_depth_raw
from .imu import load_camera_matrix, load_imu
from .odometry import OdometryError, load_odometry
from .rgb import RGBVideoReader

log = logging.getLogger(__name__)


class DatasetInventory:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.files = cfg["dataset"]["files"]
        self.data: dict = {}
        self.issues: list[dict] = []
        self.odometry = None

    def _issue(self, severity: str, code: str, msg: str) -> None:
        self.issues.append({"severity": severity, "code": code, "message": msg})
        {"error": log.error, "warning": log.warning}.get(severity, log.info)("%s: %s", code, msg)

    def scan(self, path: str | Path) -> dict:
        root = Path(path)
        self.data, self.issues, self.odometry = {"scan": root.name, "path": str(root)}, [], None
        if not root.is_dir():
            self._issue("error", "scan_missing", f"scan directory not found: {root}")
            return self.data
        self._scan_depth(root / self.files["depth_dir"])
        self._scan_confidence(root / self.files["confidence_dir"])
        self._scan_rgb(root / self.files["rgb"])
        self._scan_odometry(root / self.files["odometry"])
        self._scan_camera_matrix(root / self.files["camera_matrix"])
        self._scan_imu(root / self.files["imu"])
        return self.data

    # ---- sections -------------------------------------------------------
    def _scan_depth(self, d: Path) -> None:
        if not d.is_dir():
            self._issue("error", "depth_dir_missing", f"missing depth directory: {d}")
            return
        ids, paths, odd = list_depth_frames(d)
        dd = self.data["depth"] = {"frames": len(ids), "non_conforming_files": odd[:10]}
        if not ids:
            self._issue("error", "depth_empty", f"no NNNNNN.png frames in {d}")
            return
        dd["range"] = [ids[0], ids[-1]]
        dd["first_file"], dd["last_file"] = paths[ids[0]].name, paths[ids[-1]].name
        missing = sorted(set(range(ids[0], ids[-1] + 1)) - set(ids))
        dd["missing_ids"] = missing[:20]
        if missing:
            self._issue("error", "depth_ids_missing", f"{len(missing)} missing depth frame IDs (first: {missing[:5]})")
        if ids[0] != 0:
            self._issue("warning", "depth_not_zero_based", f"first depth frame id is {ids[0]}")
        n = int(self.cfg["inventory"]["depth_sample_count"])
        sample = sorted({ids[int(i)] for i in np.linspace(0, len(ids) - 1, min(n, len(ids)))})
        want = (self.cfg["dataset"]["depth"]["height"], self.cfg["dataset"]["depth"]["width"])
        shapes, dtypes, stats = set(), set(), {}
        for fid in sample:
            try:
                a = read_depth_raw(paths[fid], frame_label=f"{fid:06d}")
            except DepthFormatError as e:
                self._issue("error", "depth_malformed", str(e))
                continue
            shapes.add(a.shape); dtypes.add(str(a.dtype))
            stats[fid] = {"min": int(a.min()), "max": int(a.max()), "zero_pixels": int((a == 0).sum()), "pixels": int(a.size)}
        dd["sampled_frames"] = stats
        if shapes:
            h, w = next(iter(shapes))
            dd["resolution"] = [w, h]
            dd["dtype"] = sorted(dtypes)[0]
            if len(shapes) > 1:
                self._issue("error", "depth_shape_inconsistent", f"sampled shapes differ: {sorted(shapes)}")
            elif (h, w) != want:
                self._issue("error", "depth_shape_mismatch", f"depth is (h,w)=({h},{w}), config expects {want}")

    def _scan_confidence(self, d: Path) -> None:
        if not d.is_dir():
            self._issue("error", "confidence_dir_missing", f"missing confidence directory: {d}")
            return
        ids, _, _ = list_depth_frames(d)
        self.data["confidence"] = {"frames": len(ids)}
        dn = self.data.get("depth", {}).get("frames")
        if dn is not None and dn != len(ids):
            self._issue("warning", "confidence_count_mismatch", f"confidence frames={len(ids)} vs depth frames={dn}")

    def _scan_rgb(self, p: Path) -> None:
        if not p.exists():
            self._issue("error", "rgb_missing", f"missing RGB video: {p}")
            return
        try:
            with RGBVideoReader(p) as r:
                m = r.metadata()
        except IOError as e:
            self._issue("error", "rgb_unreadable", str(e))
            return
        self.data["rgb"] = {"resolution": [m["width"], m["height"]], "fps": m["fps"], "frames": m["frame_count"], "duration_sec": m["duration_sec"]}
        want = [self.cfg["dataset"]["rgb"]["width"], self.cfg["dataset"]["rgb"]["height"]]
        if [m["width"], m["height"]] != want:
            self._issue("warning", "rgb_resolution_mismatch", f"RGB is {[m['width'], m['height']]}, config expects {want}")

    def _scan_odometry(self, p: Path) -> None:
        if not p.exists():
            self._issue("error", "odometry_missing", f"missing odometry file: {p}")
            return
        try:
            self.odometry = load_odometry(p, self.cfg["inventory"]["quaternion_norm_tolerance"])
        except OdometryError as e:
            self._issue("error", "odometry_invalid", str(e))
            return
        s = self.odometry.summary()
        self.data["odometry"] = s
        if s["malformed_rows"]:
            self._issue("error", "odometry_malformed_rows", f"{s['malformed_rows']} malformed rows")
        if s["duplicate_frame_ids"]:
            self._issue("error", "odometry_duplicate_ids", f"duplicate frame IDs {s['duplicate_frame_ids'][:5]}")
        if s["n_missing_frame_ids"]:
            self._issue("error", "odometry_missing_ids", f"{s['n_missing_frame_ids']} missing frame IDs")
        if not s["timestamps_monotonic"]:
            self._issue("error", "odometry_timestamps", "timestamps not strictly increasing")

    def _scan_camera_matrix(self, p: Path) -> None:
        if not p.exists():
            self._issue("error", "camera_matrix_missing", f"missing camera matrix: {p}")
            return
        try:
            m = load_camera_matrix(p)
        except ValueError as e:
            self._issue("error", "camera_matrix_invalid", str(e))
            return
        self.data["camera_matrix"] = m.tolist()
        if self.odometry is not None and self.odometry.n_rows:
            from ..calibration.intrinsics import intrinsics_stats
            st = intrinsics_stats(self.odometry.df, m)
            self.data["intrinsics_stats"] = st
            for name, v in st.items():
                if not v["fixed_within_range"]:
                    self._issue("warning", "intrinsics_disagree", f"camera_matrix {name}={v['fixed_matrix_value']} lies outside the per-frame odometry range [{v['min']}, {v['max']}]")
            spread = (st["fx"]["max"] - st["fx"]["min"]) / st["fx"]["median"]
            if spread > self.cfg["inventory"]["intrinsics_variation_note_frac"]:
                self._issue("info", "intrinsics_vary", f"per-frame fx varies {spread:.1%} (min {st['fx']['min']:.1f}, max {st['fx']['max']:.1f}); a 1% focal error is ~1% length error")

    def _scan_imu(self, p: Path) -> None:
        if not p.exists():
            self._issue("error", "imu_missing", f"missing IMU file: {p}")
            return
        try:
            df = load_imu(p)
        except ValueError as e:
            self._issue("error", "imu_invalid", str(e))
            return
        self.data["imu"] = {"rows": len(df), "timestamp_range": [float(df["timestamp"].min()), float(df["timestamp"].max())]}

    # ---- outputs --------------------------------------------------------
    def validate(self) -> bool:
        """True iff no error-severity issues."""
        return not any(i["severity"] == "error" for i in self.issues)

    def report(self) -> dict:
        r = dict(self.data)
        r["issues"] = self.issues
        r["valid"] = self.validate()
        return r

    def write(self, out_dir: str | Path) -> Path:
        out = Path(out_dir) / "inventory"
        out.mkdir(parents=True, exist_ok=True)
        p = out / f"{self.data['scan']}.json"
        p.write_text(json.dumps(self.report(), indent=2))
        return p
