from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

from .walls import endpoints


def wall_enclosed_footprint(walls: list[dict], cfg: dict, cam_uv: np.ndarray | None = None) -> dict | None:
    """Rasterize wall segments as barriers and take the enclosed region that holds the camera path (so a second enclosed space,
    e.g. a corridor seen through a door, is not merged in). With no camera positions the largest enclosed region is used.
    None if the walls enclose nothing. Area is taken at the wall centre-lines (barrier pixels adjacent to the region are included)."""
    if len(walls) < 3:
        return None
    res = cfg["floorplan"]["grid_resolution_m"]
    E = np.array([e for w in walls for e in endpoints(w)])
    lo, hi = E.min(0) - 0.3, E.max(0) + 0.3
    shape = np.ceil((hi - lo) / res).astype(int)[::-1]
    g = np.zeros(shape, np.uint8)
    for w in walls:
        a, b = endpoints(w)
        pa, pb = np.round((a - lo) / res).astype(int), np.round((b - lo) / res).astype(int)
        cv2.line(g, tuple(pa), tuple(pb), 1, 1)
    n, lab = cv2.connectedComponents((g == 0).astype(np.uint8), connectivity=4)  # 8-connected barrier lines cannot be crossed diagonally
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist())
    sizes = np.bincount(lab.ravel(), minlength=n)
    enclosed = [k for k in range(1, n) if k not in border and sizes[k] >= 4]
    if not enclosed:
        return None
    pick, cam_frac = None, None
    if cam_uv is not None and len(cam_uv):
        ij = np.round((np.asarray(cam_uv, float) - lo) / res).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < shape[1]) & (ij[:, 1] < shape[0])
        counts = np.bincount(lab[ij[ok, 1], ij[ok, 0]], minlength=n)
        best = max(enclosed, key=lambda k: counts[k])
        if counts[best] > 0:
            pick, cam_frac = best, float(counts[best] / len(ij))
    if pick is None:
        pick = max(enclosed, key=lambda k: sizes[k])
        cam_frac = 0.0 if cam_uv is not None and len(cam_uv) else None
    region = (lab == pick).astype(np.uint8)
    filled = (cv2.dilate(region, np.ones((3, 3), np.uint8)) & (region | g)).astype(np.uint8)
    cnts, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)
    poly = cv2.approxPolyDP(cnt, cfg["floorplan"]["polygon_epsilon_m"] / res, True).reshape(-1, 2).astype(float) * res + lo + res / 2
    return {"area_m2": float(cv2.contourArea(cnt) * res * res), "polygon": poly, "mask": filled, "grid_origin": lo, "res": res,
            "n_enclosed_regions": len(enclosed), "camera_fraction_inside": cam_frac}


def floor_footprint(L: np.ndarray, walls: list[dict], cfg: dict, cam_uv: np.ndarray | None = None) -> dict:
    """Floor evidence grid -> closed/filled mask -> area + polygon (floor-plane coords). Clipped to the wall hull."""
    fc = cfg["floorplan"]
    res = fc["grid_resolution_m"]
    sel = np.abs(L[:, 2]) < fc["floor_band_m"]
    uv = L[sel, :2]
    if len(uv) == 0:
        raise RuntimeError("no floor evidence points for the footprint")
    if walls:
        E = np.array([e for w in walls for e in endpoints(w)])
        hull = cv2.convexHull(E.astype(np.float32))
        # expand hull slightly so floor cells touching walls survive
        c = hull.reshape(-1, 2).mean(0)
        hull = (c + (hull.reshape(-1, 2) - c) * 1.0).astype(np.float32)
    lo = uv.min(0) - 0.2
    hi = uv.max(0) + 0.2
    shape = np.ceil((hi - lo) / res).astype(int)[::-1]
    g = np.zeros(shape, np.uint8)
    ij = np.floor((uv - lo) / res).astype(int)
    g[ij[:, 1], ij[:, 0]] = 1
    if walls:
        mask = np.zeros(shape, np.uint8)
        pts = ((hull.reshape(-1, 2) - lo) / res).astype(np.int32)
        cv2.fillConvexPoly(mask, pts, 1)
        pad = int(round(0.10 / res))
        mask = cv2.dilate(mask, np.ones((2 * pad + 1, 2 * pad + 1), np.uint8))
        g &= mask
    r = max(int(round(fc["closing_radius_m"] / res)), 1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, k)
    g = ndi.binary_fill_holes(g).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(g, connectivity=8)
    if n <= 1:
        raise RuntimeError("floor footprint empty after cleanup")
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    g = (lab == big).astype(np.uint8)
    area = float(g.sum() * res * res)
    cnts, _ = cv2.findContours(g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)
    eps = fc["polygon_epsilon_m"] / res
    poly = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2).astype(float) * res + lo + res / 2
    warn = [] if area >= fc["minimum_room_area_m2"] else [f"footprint area {area:.2f} m2 below minimum {fc['minimum_room_area_m2']}"]
    out = {"area_m2": area, "polygon": poly, "grid_origin": lo, "res": res, "mask": g, "floor_points": int(sel.sum()), "warnings": warn,
           "method": "floor_evidence_grid_closed_filled", "floor_evidence_area_m2": area}
    enc = wall_enclosed_footprint(walls, cfg, cam_uv)
    out["enclosure"] = {"enclosed": enc is not None}
    if enc is not None and (enc["area_m2"] >= area or fc.get("prefer_wall_enclosed", False)):
        out.update(area_m2=enc["area_m2"], polygon=enc["polygon"], method="wall_enclosed_region")
        # how much of the enclosed region is actually backed by floor evidence (honest quality number; the area itself is a wall-based extrapolation)
        Eg = np.zeros(enc["mask"].shape, np.uint8)
        ij2 = np.floor((uv - enc["grid_origin"]) / enc["res"]).astype(int)
        inb = (ij2[:, 0] >= 0) & (ij2[:, 1] >= 0) & (ij2[:, 0] < Eg.shape[1]) & (ij2[:, 1] < Eg.shape[0])
        Eg[ij2[inb, 1], ij2[inb, 0]] = 1
        Eg = cv2.morphologyEx(Eg, cv2.MORPH_CLOSE, k)
        cov = float((Eg & enc["mask"]).sum() / max(enc["mask"].sum(), 1))
        outside = float(Eg.sum() - (Eg & enc["mask"]).sum()) * enc["res"] ** 2
        out["floor_coverage_frac"] = cov
        out["enclosure"].update(n_enclosed_regions=enc["n_enclosed_regions"], camera_fraction_inside=enc["camera_fraction_inside"],
                                floor_coverage_frac=cov, floor_evidence_outside_region_m2=outside)
        if cov < fc.get("min_floor_coverage", 0.5):
            out["warnings"].append(f"floor evidence backs only {cov:.0%} of the wall-enclosed region (sparse floor coverage; area is a wall-based extrapolation)")
        if outside > fc.get("outside_evidence_warn_m2", 0.5):
            out["warnings"].append(f"{outside:.1f} m2 of floor evidence lies outside the enclosed region (space seen through openings, or a wall is missing/mis-bounded)")
        if enc["camera_fraction_inside"] is not None and enc["camera_fraction_inside"] < 0.5:
            out["warnings"].append(f"only {enc['camera_fraction_inside']:.0%} of camera positions lie inside the enclosed region (walls do not bound the scanned space)")
    else:
        out["warnings"].append("walls do not enclose a region; area comes from observed floor evidence only (may under-estimate)")
    # Reliability verdict: a footprint that fails these checks is NOT a measurement and must not be presented as one (confident garbage).
    reasons = []
    if enc is None:
        reasons.append("walls do not enclose a region")
    else:
        if out.get("floor_coverage_frac", 1.0) < fc.get("min_floor_coverage", 0.5):
            reasons.append(f"floor evidence backs only {out['floor_coverage_frac']:.0%} of the enclosed region")
        cf = enc["camera_fraction_inside"]
        if cf is not None and cf < fc.get("min_camera_inside_frac", 0.5):
            reasons.append(f"only {cf:.0%} of camera positions lie inside the enclosed region")
        if out["enclosure"].get("floor_evidence_outside_region_m2", 0.0) > fc.get("outside_evidence_warn_m2", 0.5):
            reasons.append(f"{out['enclosure']['floor_evidence_outside_region_m2']:.1f} m2 of floor evidence lies outside the region")
    out["area_reliable"], out["unreliable_reasons"] = not reasons, reasons
    return out
