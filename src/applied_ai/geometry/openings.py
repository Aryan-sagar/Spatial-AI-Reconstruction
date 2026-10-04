"""Geometry-first opening detection: persistent gaps in mid-height wall-surface point density, validated by see-through evidence."""
from __future__ import annotations

import numpy as np


def detect_openings(L: np.ndarray, walls: list[dict], interior_point: np.ndarray, ceiling_h: float | None, cfg: dict) -> list[dict]:
    oc = cfg["openings"]
    out = []
    top = (ceiling_h - 0.05) if ceiling_h is not None else cfg["walls"]["no_ceiling_band_height_m"]
    for w in walls:
        sd = L[:, :2] @ w["normal"] + w["d"]
        s = (L[:, :2] - w["centroid"]) @ w["direction"]
        h = L[:, 2]
        side = np.sign(w["normal"] @ interior_point + w["d"]) or 1.0
        out_dist = -side * sd  # >0 on the far (outside) side
        on = (np.abs(sd) <= oc["wall_band_m"]) & (s >= w["s_min"] - 0.05) & (s <= w["s_max"] + 0.05)
        mid = on & (h >= oc["mid_band_m"][0]) & (h <= min(oc["mid_band_m"][1], top))
        if mid.sum() < 50:
            continue
        b = oc["bin_m"]
        edges = np.arange(w["s_min"], w["s_max"] + b, b)
        cnt, _ = np.histogram(s[mid], bins=edges)
        ref = np.median(cnt[cnt > 0]) if (cnt > 0).any() else 0
        if ref == 0:
            continue
        gap = cnt < oc["gap_density_frac"] * ref
        i = 0
        while i < len(gap):
            if not gap[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(gap) and gap[j + 1]:
                j += 1
            g0, g1 = edges[i], edges[j + 1]
            i = j + 1
            if g1 - g0 < oc["min_width_m"]:
                continue
            at_edge = g0 - w["s_min"] < oc["edge_margin_m"] or w["s_max"] - g1 < oc["edge_margin_m"]
            sm = s[mid]
            left = sm[sm < g0].max() if (sm < g0).any() else None
            right = sm[sm > g1].min() if (sm > g1).any() else None
            if left is None or right is None or at_edge:
                continue
            width = float(right - left)
            see = (np.abs(s - (left + right) / 2) < width / 2) & (out_dist > oc["see_through_min_m"]) & (h >= oc["mid_band_m"][0]) & (h <= oc["mid_band_m"][1])
            n_see = int(see.sum())
            low = on & (h >= oc["low_band_m"][0]) & (h <= oc["low_band_m"][1]) & (s > left) & (s < right)
            low_bins = np.histogram(s[low], bins=np.linspace(left, right, max(int(width / b), 1) + 1))[0] if low.any() else np.zeros(1)
            low_fill = float((low_bins > 0).mean())
            kind = "door" if low_fill < 0.2 else ("window" if low_fill > 0.6 else "unknown")
            observed = n_see >= oc["see_through_min_points"]
            out.append({"wall_id": w["id"], "start": float(left), "end": float(right), "width": width, "type": kind if observed else "unknown",
                        "observed": observed, "see_through_points": n_see, "low_band_fill": low_fill,
                        "bin_resolution_m": b, "status": "observed" if observed else "rejected_unobserved_gap"})
    for k, o in enumerate(out):
        o["id"] = f"opening_{k:02d}"
    return out
