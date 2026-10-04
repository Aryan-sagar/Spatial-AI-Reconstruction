"""Assemble a Room (internal schema) from detected geometry, attaching an interval to every measurement."""
from __future__ import annotations

import numpy as np

from ..uncertainty import UncertaintyModel
from .walls import endpoints, wall_length


def _geom(w):
    s, e = endpoints(w)
    return {"start": s.tolist(), "end": e.tolist(), "centroid": w["centroid"].tolist(), "direction": w["direction"].tolist(),
            "normal": w["normal"].tolist(), "d": float(w["d"]), "s_min": float(w["s_min"]), "s_max": float(w["s_max"])}


def build_room(room_id, raw_walls, final_walls, openings, footprint, fc, plan_rotation_deg, um: UncertaintyModel, artifacts: list[str]):
    walls = []
    for raw, fin in zip(raw_walls, final_walls):
        L = wall_length(fin)
        walls.append({
            "id": raw["id"], "surface_id": f"surface_{raw['id']}", "orientation_deg": float(np.degrees(np.arctan2(fin["direction"][1], fin["direction"][0])) % 180.0),
            "geometry": {"raw": _geom(raw), "regularized": _geom(fin) if fin.get("regularized") else None, "final": _geom(fin),
                         "manhattan_snapped": bool(fin.get("regularized")), "corner_snapped": bool(fin.get("corner_snapped"))},
            "length": um.measurement("wall_length", L, "m", "lidar_line_fit+corner_snap", "observed", fin["residual_std"], fin["n_cells"], artifacts),
            "raw_length_m": wall_length(raw), "support_cells": fin["n_cells"], "fit_residual_std_m": fin["residual_std"]})
    ops = []
    for o in openings:
        if o["status"] != "observed":
            continue
        ops.append({"id": o["id"], "type": o["type"], "parent_wall": o["wall_id"], "position": {"start_s": o["start"], "end_s": o["end"]},
                    "width": um.measurement("opening_width", o["width"], "m", "mid_band_density_gap+see_through", "observed",
                                            o["bin_resolution_m"] / 2, o["see_through_points"], artifacts),
                    "evidence": {"see_through_points": o["see_through_points"], "low_band_fill": o["low_band_fill"], "bin_resolution_m": o["bin_resolution_m"]}})
    area = um.measurement("floor_area", footprint["area_m2"], "m2", footprint.get("method", "floor_evidence_grid_closed_filled"), "observed", 0.0, footprint["floor_points"], artifacts)
    if fc.get("ceiling") is not None:
        c, f = fc["ceiling"], fc["floor"]
        ch = um.measurement("ceiling_height", fc["ceiling_height"], "m", "floor_ceiling_plane_distance", "observed",
                            float(np.hypot(c["residual_std"], f["residual_std"])), min(c["support_count"], f["support_count"]), artifacts)
    else:
        ch = um.unobserved("m", "floor_ceiling_plane_distance", "no ceiling plane observed in the scan; height is not invented")
    return {"id": room_id, "polygon": footprint["polygon"].tolist(), "plan_rotation_deg": plan_rotation_deg, "floor_area": area, "ceiling_height": ch,
            "walls": walls, "openings": ops,
            "surfaces": [{"id": "surface_floor", "label": "floor", "normal_world": fc["floor"]["normal"].tolist(), "d_world": float(fc["floor"]["d"]),
                          "support_points": fc["floor"]["support_count"]}] +
                        ([{"id": "surface_ceiling", "label": "ceiling", "normal_world": fc["ceiling"]["normal"].tolist(), "d_world": float(fc["ceiling"]["d"]),
                           "support_points": fc["ceiling"]["support_count"]}] if fc.get("ceiling") is not None else []) +
                        [{"id": w["surface_id"], "label": "wall", "wall_id": w["id"]} for w in walls]}
