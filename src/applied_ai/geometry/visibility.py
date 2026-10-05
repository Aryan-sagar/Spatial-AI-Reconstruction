"""Free space by visibility carving. A depth frame says more than where surfaces are: every point it measured was reached by a ray that passed
through empty space. Projecting each frame's wall-band points into the floor plane and filling the polygon from the camera to the nearest hit in
each angular bin gives the space that frame SAW to be free. Unlike floor evidence (which a horizontally held phone barely captures), this covers
the whole room the camera looked into.

Per frame: points with height in `h_band_m` -> (range, bearing) from the camera; per `angle_bin_deg` bin with >= `min_points_per_bin` points take the
`range_percentile` range (the nearest surface, robust to a few outliers) minus `margin_m`; contiguous bins (gaps up to `gap_bins`) form a fan polygon.
Bins with no points are NOT carved (unknown, not free). Polygons are returned; rasterising/union is the caller's job."""
from __future__ import annotations

import numpy as np


def frame_free_polygons(cam_uv: np.ndarray, pts_local: np.ndarray, cfg: dict) -> list[np.ndarray]:
    vc = cfg["visibility"]
    h = pts_local[:, 2]
    sel = (h >= vc["h_band_m"][0]) & (h <= vc["h_band_m"][1])
    if sel.sum() < vc["min_points_per_bin"]:
        return []
    d = pts_local[sel, :2] - np.asarray(cam_uv, float)
    r = np.hypot(d[:, 0], d[:, 1])
    ok = (r > 0.15) & (r <= vc["max_range_m"])
    d, r = d[ok], r[ok]
    if len(r) < vc["min_points_per_bin"]:
        return []
    ang = np.arctan2(d[:, 1], d[:, 0])
    nb = int(round(360.0 / vc["angle_bin_deg"]))
    b = ((ang + np.pi) / (2 * np.pi) * nb).astype(int) % nb
    order = np.argsort(b, kind="stable")
    b, r, ang = b[order], r[order], ang[order]
    starts = np.flatnonzero(np.r_[True, b[1:] != b[:-1]])
    ends = np.r_[starts[1:], len(b)]
    bins, rng, bearing = [], [], []
    for s, e in zip(starts, ends):
        if e - s < vc["min_points_per_bin"]:
            continue
        rr = r[s:e]
        bins.append(int(b[s]))
        rng.append(max(float(np.percentile(rr, vc["range_percentile"])) - vc["margin_m"], 0.0))
        bearing.append(float(np.median(ang[s:e])))
    if not bins:
        return []
    bins, rng, bearing = np.array(bins), np.array(rng), np.array(bearing)
    # split into contiguous runs of bins (wrapping around is ignored: a phone never sees 360 degrees in one frame)
    cut = np.flatnonzero(np.diff(bins) > vc["gap_bins"]) + 1
    polys = []
    c = np.asarray(cam_uv, float)
    for idx in np.split(np.arange(len(bins)), cut):
        if len(idx) < 2:
            continue
        ends_xy = c + rng[idx, None] * np.column_stack((np.cos(bearing[idx]), np.sin(bearing[idx])))
        polys.append(np.vstack([c, ends_xy]))
    return polys
