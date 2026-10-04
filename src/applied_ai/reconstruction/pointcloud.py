"""Point-cloud utilities (numpy only): PLY writer, voxel downsampling, memory-bounded accumulator."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def write_ply(path: str | Path, xyz: np.ndarray, rgb: np.ndarray | None = None) -> None:
    xyz = np.asarray(xyz, np.float32)
    n = len(xyz)
    props = "property float x\nproperty float y\nproperty float z\n"
    if rgb is None:
        dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
        arr = np.empty(n, dt)
    else:
        props += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
        arr = np.empty(n, dt)
        arr["r"], arr["g"], arr["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n{props}end_header\n".encode())
        f.write(arr.tobytes())


def read_ply_xyz(path: str | Path) -> np.ndarray:
    raw = Path(path).read_bytes()
    hdr_end = raw.index(b"end_header\n") + len(b"end_header\n")
    hdr = raw[:hdr_end].decode().splitlines()
    n = int(next(l for l in hdr if l.startswith("element vertex")).split()[-1])
    has_rgb = any("uchar red" in l for l in hdr)
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")] + ([("r", "u1"), ("g", "u1"), ("b", "u1")] if has_rgb else []))
    a = np.frombuffer(raw[hdr_end:], dt, count=n)
    return np.column_stack((a["x"], a["y"], a["z"]))


def voxel_downsample(xyz: np.ndarray, voxel: float, weights: np.ndarray | None = None):
    """Mean point per voxel. Returns (points, counts)."""
    if len(xyz) == 0:
        return xyz.astype(np.float32), np.zeros(0, np.int64)
    p = np.asarray(xyz, np.float64)
    idx = np.floor((p - p.min(0)) / voxel).astype(np.int64)
    dims = idx.max(0) + 1
    if np.prod(dims.astype(float)) > 2**62:
        raise ValueError(f"voxel grid too large ({dims}); check for outliers / wrong units")
    key = (idx[:, 0] * dims[1] + idx[:, 1]) * dims[2] + idx[:, 2]
    uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    out = np.column_stack([np.bincount(inv, p[:, k]) / cnt for k in range(3)])
    return out.astype(np.float32), cnt


class PointCloudAccumulator:
    """Adds per-frame world points; merges/downsamples whenever the buffer exceeds a cap so RAM stays bounded."""

    def __init__(self, voxel: float, max_buffer_points: int = 4_000_000):
        self.voxel, self.cap = voxel, max_buffer_points
        self._merged = np.zeros((0, 3), np.float32)
        self._buf: list[np.ndarray] = []
        self._buf_n = 0
        self.frames = 0
        self.raw_points = 0

    def add_frame(self, xyz: np.ndarray) -> None:
        self._buf.append(np.asarray(xyz, np.float32))
        self._buf_n += len(xyz)
        self.frames += 1
        self.raw_points += len(xyz)
        if self._buf_n > self.cap:
            self.merge()

    def merge(self) -> np.ndarray:
        if self._buf:
            allp = np.concatenate([self._merged] + self._buf)
            self._merged, _ = voxel_downsample(allp, self.voxel)
            self._buf, self._buf_n = [], 0
        return self._merged

    def voxel_downsample(self) -> np.ndarray:
        return self.merge()

    def save(self, path) -> None:
        write_ply(path, self.merge())

    def statistics(self) -> dict:
        m = self.merge()
        return {"frames": self.frames, "raw_points": int(self.raw_points), "voxel_points": int(len(m)), "voxel_size_m": self.voxel,
                "bounds_min": m.min(0).tolist() if len(m) else None, "bounds_max": m.max(0).tolist() if len(m) else None}
