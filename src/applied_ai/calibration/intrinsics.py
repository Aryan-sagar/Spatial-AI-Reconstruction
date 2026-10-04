"""RGB intrinsics -> depth-grid intrinsics. The mapping is a configurable calibration mode, not a hardcode."""
from __future__ import annotations

import numpy as np

from ..domain.models import Intrinsics

MODES = ("scaled_rgb_intrinsics", "direct_depth_intrinsics", "cropped_scaled", "custom_calibration")


class DepthProjectionCalibration:
    def __init__(self, mode: str = "scaled_rgb_intrinsics", depth_wh: tuple[int, int] = (256, 192), direct: Intrinsics | None = None):
        if mode not in MODES:
            raise ValueError(f"unknown depth projection mode {mode!r}; options: {MODES}")
        self.mode, self.depth_wh, self.direct = mode, depth_wh, direct

    @classmethod
    def from_config(cls, cfg: dict) -> "DepthProjectionCalibration":
        d = cfg["dataset"]["depth"]
        return cls(cfg["calibration"]["depth_intrinsics"]["mode"], (d["width"], d["height"]))

    def depth_intrinsics(self, rgb: Intrinsics) -> Intrinsics:
        w, h = self.depth_wh
        if self.mode == "scaled_rgb_intrinsics":
            sx, sy = w / rgb.width, h / rgb.height
            return Intrinsics(rgb.fx * sx, rgb.fy * sy, rgb.cx * sx, rgb.cy * sy, w, h, None, None, f"{rgb.source}|scaled_rgb({sx:.5f},{sy:.5f})")
        if self.mode == "direct_depth_intrinsics":
            if self.direct is None:
                raise ValueError("direct_depth_intrinsics requires depth intrinsics to be supplied")
            return self.direct
        raise NotImplementedError(f"depth projection mode {self.mode!r} is not implemented yet")

    def diagnostics(self, rgb: Intrinsics) -> dict:
        d = self.depth_intrinsics(rgb)
        return {"mode": self.mode, "rgb_wh": [rgb.width, rgb.height], "depth_wh": list(self.depth_wh),
                "scale_x": self.depth_wh[0] / rgb.width, "scale_y": self.depth_wh[1] / rgb.height,
                "depth_intrinsics": {"fx": d.fx, "fy": d.fy, "cx": d.cx, "cy": d.cy}}


def intrinsics_stats(df, camera_matrix: np.ndarray) -> dict:
    """Spread of per-frame intrinsics and where the fixed camera_matrix.csv sits within it."""
    out = {}
    for name, fixed in (("fx", camera_matrix[0, 0]), ("fy", camera_matrix[1, 1]), ("cx", camera_matrix[0, 2]), ("cy", camera_matrix[1, 2])):
        v = df[name].to_numpy(float)
        out[name] = {"min": float(v.min()), "max": float(v.max()), "mean": float(v.mean()), "median": float(np.median(v)),
                     "std": float(v.std()), "fixed_matrix_value": float(fixed), "fixed_within_range": bool(v.min() <= fixed <= v.max())}
    return out
