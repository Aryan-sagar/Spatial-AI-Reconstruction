"""Odometry loader. Column names are whitespace-stripped (real headers have leading spaces)."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["timestamp", "frame", "x", "y", "z", "qx", "qy", "qz", "qw", "fx", "fy", "cx", "cy"]
OPTIONAL_COLUMNS = ["distortion_center_x", "distortion_center_y"]


class OdometryError(ValueError):
    """Raised when odometry cannot be used (missing columns, degenerate quaternion, ...)."""


@dataclass
class OdometryTable:
    df: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    n_malformed_rows: int = 0
    duplicate_frame_ids: list[int] = field(default_factory=list)
    missing_frame_ids: list[int] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return len(self.df)

    def summary(self) -> dict:
        d = self.df
        ts = d["timestamp"].to_numpy()
        dt = np.diff(ts) if len(ts) > 1 else np.array([])
        return {
            "rows": self.n_rows,
            "malformed_rows": self.n_malformed_rows,
            "frame_range": [int(d["frame"].min()), int(d["frame"].max())] if self.n_rows else None,
            "duplicate_frame_ids": self.duplicate_frame_ids[:20],
            "missing_frame_ids": self.missing_frame_ids[:20],
            "n_missing_frame_ids": len(self.missing_frame_ids),
            "timestamp_range": [float(ts.min()), float(ts.max())] if len(ts) else None,
            "duration_sec": float(ts.max() - ts.min()) if len(ts) else None,
            "dt_median": float(np.median(dt)) if len(dt) else None,
            "dt_mean": float(dt.mean()) if len(dt) else None,
            "dt_max": float(dt.max()) if len(dt) else None,
            "timestamps_monotonic": bool(np.all(dt > 0)) if len(dt) else True,
            "warnings": list(self.warnings),
        }


def load_odometry(path: str | Path, quat_tol: float = 1e-3) -> OdometryTable:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"odometry file missing: {path}")
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise OdometryError(f"{path}: missing required columns {missing}; found {list(df.columns)}")

    keep = REQUIRED_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in df.columns]
    num = df[keep].apply(pd.to_numeric, errors="coerce")
    bad = ~np.isfinite(num[REQUIRED_COLUMNS].to_numpy(dtype=float)).all(axis=1)  # optional columns never drop rows
    warnings: list[str] = []
    n_bad = int(bad.sum())
    if n_bad:
        warnings.append(f"{n_bad} malformed/non-finite rows dropped (first row indices: {np.flatnonzero(bad)[:5].tolist()})")
    for c in OPTIONAL_COLUMNS:
        if c in num.columns and not np.isfinite(num[c].to_numpy(dtype=float)).all():
            n_nf = int((~np.isfinite(num[c].to_numpy(dtype=float))).sum())
            warnings.append(f"optional column {c} has {n_nf} non-finite values (rows kept)")
    if n_bad:
        cols = {c: int((~np.isfinite(num[c].to_numpy(dtype=float))).sum()) for c in REQUIRED_COLUMNS if (~np.isfinite(num[c].to_numpy(dtype=float))).any()}
        warnings.append(f"non-finite values per required column: {cols}")
    num = num[~bad].reset_index(drop=True)
    if len(num) == 0:
        raise OdometryError(f"{path}: no valid rows remain ({n_bad} malformed of {len(df)})")

    frame_f = num["frame"].to_numpy()
    if not np.all(frame_f == np.round(frame_f)):
        raise OdometryError(f"{path}: 'frame' column contains non-integer values")
    num["frame"] = frame_f.astype(np.int64)

    counts = num["frame"].value_counts()
    dups = sorted(int(i) for i in counts[counts > 1].index)
    if dups:
        warnings.append(f"duplicate frame IDs: {dups[:10]}")
    if len(num):
        full = set(range(0, int(num["frame"].max()) + 1))
        missing_ids = sorted(full - set(num["frame"].tolist()))
    else:
        missing_ids = []
    if missing_ids:
        warnings.append(f"frame IDs not contiguous from 0: {len(missing_ids)} missing (first: {missing_ids[:10]})")

    ts = num["timestamp"].to_numpy()
    if len(ts) > 1 and not np.all(np.diff(ts) > 0):
        warnings.append("timestamps are not strictly increasing")

    for c in ("fx", "fy"):
        if not (num[c] > 0).all():
            raise OdometryError(f"{path}: non-positive {c}")
    if not (num[["cx", "cy"]] > 0).all().all():
        warnings.append("non-positive principal point values present")

    q = num[["qx", "qy", "qz", "qw"]].to_numpy()
    norms = np.linalg.norm(q, axis=1)
    if len(norms) and norms.min() < 1e-8:
        raise OdometryError(f"{path}: degenerate (zero-norm) quaternion at rows {np.flatnonzero(norms < 1e-8)[:5].tolist()}")
    off = np.abs(norms - 1.0) > quat_tol
    if off.any():
        warnings.append(
            f"{int(off.sum())} quaternions had |norm-1| > {quat_tol} (range {norms.min():.6f}..{norms.max():.6f}); normalized"
        )
        num.loc[:, ["qx", "qy", "qz", "qw"]] = q / norms[:, None]

    for w in warnings:
        log.warning("%s: %s", path.name, w)
    return OdometryTable(num, warnings, n_bad, dups, missing_ids)
