"""Wall stage of the pipeline as one reusable function (used by lidar_run and by the synthetic furnished-room tests, so the tests
exercise the real code path)."""
from __future__ import annotations

import logging

import numpy as np

from .walls import (close_corners, dedupe_walls, detect_walls, detect_walls_axes, dominant_axes, drop_camera_crossed, manhattan_regularize,
                    prune_non_boundary, snap_corners, wall_band_cells)

log = logging.getLogger(__name__)


def extract_walls(L: np.ndarray, cam_uv: np.ndarray, ceiling_h: float | None, cfg: dict) -> dict:
    """Floor-local points + camera path (2-D) -> walls.
    Returns dict(cells, band, all_raw_walls, raw_walls, final, minfo, prune_info). `raw_walls` and `final` are aligned 1:1."""
    col_stats: dict = {}
    cells, band = wall_band_cells(L, None, cfg, ceiling_h, stats=col_stats)
    method_req = cfg["walls"].get("method", "legacy_ransac")
    method_used, axes_info, winfo = method_req, None, {}
    if method_req == "manhattan_axes":
        axes_info = dominant_axes(cells, cfg)
        if axes_info["reliable"]:
            raw_walls, reg, winfo = detect_walls_axes(cells, axes_info["phi_rad"], cfg)
            dup_dropped = [r for r in winfo["rejected"] if r["reason"] == "weak_vs_best"]
            minfo = {"enabled": True, "dominant_angle_deg": axes_info["phi_deg"],
                     "snapped": [{"id": w["id"], "deviation_deg": w["snapped_deg"]} for w in reg],
                     "non_conforming": [r for r in winfo["rejected"] if r["reason"] == "non_conforming_angle"]}
        else:
            method_used = "legacy_ransac"
            log.warning("walls.method=manhattan_axes requested but the axis frame is unreliable (%s); FALLING BACK to legacy_ransac", axes_info["reason"])
    if method_used == "legacy_ransac":
        raw_walls = detect_walls(cells, cfg)
        if not raw_walls:
            raise RuntimeError("no walls detected (band %s, %d cells); see debug artifacts" % (band, len(cells)))
        raw_walls, dup_dropped = dedupe_walls(raw_walls, cfg)
        reg, minfo = manhattan_regularize(raw_walls, cfg)
    if not raw_walls:
        raise RuntimeError("no walls detected (band %s, %d cells); see debug artifacts" % (band, len(cells)))
    reg, crossing_info = drop_camera_crossed(reg, cam_uv, cells, cfg)  # interior lines the camera walked through are not walls
    reg, prune_info = prune_non_boundary(reg, cam_uv, cfg)
    prune_info["camera_crossing"] = crossing_info
    keep_ids = {w["id"] for w in reg}
    all_raw_walls = raw_walls
    raw_walls = [w for w in raw_walls if w["id"] in keep_ids]
    final = close_corners(snap_corners(reg, cfg), cfg)
    minfo.update(duplicates_dropped=dup_dropped, boundary_pruning=prune_info, method_requested=method_req, method_used=method_used,
                 axes=axes_info, axis_rejected=winfo.get("rejected", []), column_filter=col_stats, camera_crossing=crossing_info)
    log.info("walls: method=%s, %d kept (%d candidates), non-conforming: %d", method_used, len(final), len(all_raw_walls), len(minfo.get("non_conforming", [])))
    return {"cells": cells, "band": band, "all_raw_walls": all_raw_walls, "raw_walls": raw_walls, "final": final, "minfo": minfo, "prune_info": prune_info}
