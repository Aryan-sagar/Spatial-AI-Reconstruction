"""Wall detection in floor-plane coordinates: sequential 2-D RANSAC lines on occupied cells, Manhattan regularization
(raw geometry always retained), and corner snapping."""
from __future__ import annotations

import numpy as np


def wall_band_cells(L: np.ndarray, floor_h_top: float | None, cfg: dict, ceiling_h: float | None, stats: dict | None = None):
    """Unique occupied 2-D cells (u,v) from points in the wall height band.

    If walls.min_column_extent_m > 0, only points in 'tall' columns are kept: a column (column_m x column_m) must contain points
    spanning at least that vertical extent (metres, capped at 80% of the band), so horizontal surfaces and low furniture (table tops,
    beds, chair backs) cannot become wall evidence. Tall furniture (wardrobes) still passes - height alone cannot separate it from a
    wall - and a wall seen only low (camera pitched down) loses its low-seen stretch if the extent is set too high. `stats` (optional
    dict) is filled with counts."""
    wc = cfg["walls"]
    lo = wc["band_floor_offset_m"]
    hi = (ceiling_h - wc["band_ceiling_offset_m"]) if ceiling_h is not None else wc["no_ceiling_band_height_m"]
    if hi <= lo:
        raise RuntimeError(f"empty wall height band ({lo}..{hi})")
    sel = (L[:, 2] >= lo) & (L[:, 2] <= hi)
    P = L[sel]
    need = min(float(wc.get("min_column_extent_m", 0.0)), 0.8 * (hi - lo))
    if stats is not None:
        stats.update(band_points=int(len(P)), column_filter_enabled=need > 0)
    if need > 0 and len(P):
        cm = float(wc.get("column_m", 0.10))
        ci = np.floor(P[:, :2] / cm).astype(np.int64)
        _, inv, cnt = np.unique(ci, axis=0, return_inverse=True, return_counts=True)
        inv = inv.ravel()
        zmin = np.full(len(cnt), np.inf)
        zmax = np.full(len(cnt), -np.inf)
        np.minimum.at(zmin, inv, P[:, 2])
        np.maximum.at(zmax, inv, P[:, 2])
        tall = ((zmax - zmin) >= need) & (cnt >= int(wc.get("column_min_points", 3)))
        keep = tall[inv]
        if stats is not None:
            stats.update(columns_total=int(len(cnt)), columns_tall=int(tall.sum()), points_rejected_low_profile=int((~keep).sum()),
                         extent_required_m=float(need))
        P = P[keep]
        if stats is not None and stats.get("band_points"):
            stats["fraction_rejected"] = float(stats["points_rejected_low_profile"] / stats["band_points"])
            if stats["fraction_rejected"] > 0.7:
                import logging
                logging.getLogger(__name__).warning("tall-column filter removed %.0f%% of wall-band points (walls.min_column_extent_m=%.2f): "
                                                    "walls seen only low (camera pitched down?) may be shortened", 100 * stats["fraction_rejected"], need)
    c = wc["cell_m"]
    if len(P) == 0:
        raise RuntimeError("no wall-band points left after the tall-column filter; lower walls.min_column_extent_m or check the band")
    ij = np.unique(np.floor(P[:, :2] / c).astype(np.int64), axis=0)
    return (ij + 0.5) * c, (lo, hi)


def _line_ransac(P, thr, iters, rng, subsample=20000):
    sub = P if len(P) <= subsample else P[rng.choice(len(P), subsample, replace=False)]
    best, bc = None, -1
    for s in range(0, iters, 250):
        c = min(250, iters - s)
        i = rng.integers(0, len(sub), (c, 2))
        a, b = sub[i[:, 0]], sub[i[:, 1]]
        t = b - a
        ln = np.linalg.norm(t, axis=1)
        ok = ln > 0.3
        n = np.column_stack((-t[:, 1], t[:, 0])) / np.where(ok, ln, 1)[:, None]
        d = -np.einsum("ij,ij->i", n, a)
        cnt = (np.abs(sub @ n.T + d) < thr).sum(0)
        cnt[~ok] = -1
        k = int(np.argmax(cnt))
        if cnt[k] > bc:
            bc, best = int(cnt[k]), (n[k].copy(), float(d[k]))
    if best is None:
        return None
    n, d = best
    m = np.abs(P @ n + d) < thr
    for _ in range(2):
        Q = P[m]
        c = Q.mean(0)
        _, _, vt = np.linalg.svd(Q - c, full_matrices=False)
        t = vt[0]
        n = np.array([-t[1], t[0]])
        d = float(-n @ c)
        m = np.abs(P @ n + d) < thr
    return n, d, m


def _trim_sparse_tails(s: np.ndarray, bin_m: float = 0.1, win: int = 3, need: int = 2) -> np.ndarray:
    """Cut stray cells off the ends of a run: the run starts/ends at the first/last window of `win` bins (10 cm each) with at least
    `need` occupied bins. A wall end is solid; clutter within the gap tolerance of a wall end is sparse and is dropped."""
    b = np.floor((s - s[0]) / bin_m).astype(int)
    occ = np.zeros(b.max() + 1, bool)
    occ[b] = True
    if len(occ) < win:
        return s
    k = np.convolve(occ, np.ones(win), "valid") >= need
    if not k.any():
        return s
    i0 = int(np.argmax(k))
    i1 = len(k) - 1 - int(np.argmax(k[::-1])) + win - 1
    return s[(b >= i0) & (b <= i1)]


def _extent(pts, direction, centroid, trim, gap_split=1.5, min_run_m=None, min_occupancy=0.5):
    """Extent along the wall. Runs are split at gaps > gap_split metres so stray clutter/see-through cells near the line cannot
    stretch the wall; door-sized gaps stay inside one wall. Default (min_run_m=None): the largest run only (legacy behaviour).
    With min_run_m: every run at least that long AND at least min_occupancy dense (share of its 10 cm bins holding cells; sparse
    scatter that merely lies near the line does not count) is kept, and the extent spans first..last kept run, so a wall hidden
    behind furniture for more than gap_split is bridged (the hidden stretch is later reported as an unobserved gap, not as wall evidence)."""
    s = np.sort((pts - centroid) @ direction)
    if min_run_m is not None and len(s) > 3:
        s = _trim_sparse_tails(s)
    cut = np.flatnonzero(np.diff(s) > gap_split)
    if len(cut):
        groups = np.split(s, cut + 1)
        if min_run_m is None:
            s = max(groups, key=len)
        else:
            def _dense(g):
                span = g[-1] - g[0]
                if span < min_run_m:
                    return False
                nb = max(int(np.ceil(span / 0.1)), 1)
                return len(np.unique(np.floor((g - g[0]) / 0.1))) / nb >= min_occupancy
            good = [g for g in groups if _dense(g)]
            s = np.concatenate(good) if good else max(groups, key=len)
    return float(np.quantile(s, trim)), float(np.quantile(s, 1 - trim))


def detect_walls(cells: np.ndarray, cfg: dict) -> list[dict]:
    wc, pc = cfg["walls"], cfg["planes"]
    rng = np.random.default_rng(pc["seed"] + 1)
    rem, walls = cells.copy(), []
    min_cells = int(wc["min_wall_length_m"] / wc["cell_m"] * 0.6)
    for _ in range(wc["max_walls"]):
        if len(rem) < min_cells:
            break
        r = _line_ransac(rem, wc["line_inlier_m"], 1500, rng)
        if r is None:
            break
        n, d, m = r
        if m.sum() < min_cells:
            break
        pts = rem[m]
        t = np.array([n[1], -n[0]])
        c = pts.mean(0)
        s0, s1 = _extent(pts, t, c, wc["endpoint_trim_frac"], wc["extent_gap_split_m"])
        rem = rem[~m]
        if s1 - s0 < wc["min_wall_length_m"]:
            continue
        walls.append({"normal": n, "d": d, "direction": t, "centroid": c, "s_min": s0, "s_max": s1, "n_cells": int(m.sum()),
                      "residual_std": float(np.std(pts @ n + d)), "_pts": pts})
    for i, w in enumerate(walls):
        w["id"] = f"wall_{i:02d}"
    return walls


def _angle(t):  # direction angle in [0, pi)
    return float(np.arctan2(t[1], t[0]) % np.pi)


def manhattan_regularize(walls: list[dict], cfg: dict):
    """Returns (regularized_walls, info). Raw wall dicts are left untouched."""
    mc = cfg["walls"]["manhattan"]
    info = {"enabled": bool(mc["enabled"]), "dominant_angle_deg": None, "snapped": [], "non_conforming": []}
    if not mc["enabled"] or not walls:
        return [dict(w, regularized=False) for w in walls], info
    wts = np.array([w["s_max"] - w["s_min"] for w in walls])
    th = np.array([_angle(w["direction"]) for w in walls])
    phi = float(np.angle(np.sum(wts * np.exp(4j * th))) / 4.0)
    info["dominant_angle_deg"] = float(np.degrees(phi) % 90.0)
    tol = np.radians(mc["snap_tolerance_deg"])
    out = []
    for w, a in zip(walls, th):
        k = np.round((a - phi) / (np.pi / 2))
        target = phi + k * np.pi / 2
        dev = abs(a - target)
        if dev <= tol:
            t = np.array([np.cos(target), np.sin(target)])
            n = np.array([-t[1], t[0]])
            pts = w["_pts"]
            d = float(-n @ pts.mean(0))
            c = pts.mean(0)
            s0, s1 = _extent(pts, t, c, cfg["walls"]["endpoint_trim_frac"], cfg["walls"]["extent_gap_split_m"])
            out.append(dict(w, normal=n, d=d, direction=t, centroid=c, s_min=s0, s_max=s1, regularized=True, snapped_deg=float(np.degrees(dev))))
            info["snapped"].append({"id": w["id"], "deviation_deg": float(np.degrees(dev))})
        elif mc.get("drop_non_conforming", False):
            # A wall off the dominant axes by more than the tolerance is NOT kept silently: it is dropped and logged here.
            info["non_conforming"].append({"id": w["id"], "deviation_deg": float(np.degrees(dev)), "n_cells": w["n_cells"],
                                           "length_m": float(w["s_max"] - w["s_min"]), "action": "dropped"})
        else:
            info["non_conforming"].append({"id": w["id"], "deviation_deg": float(np.degrees(dev)), "n_cells": w["n_cells"],
                                           "length_m": float(w["s_max"] - w["s_min"]), "action": "kept_unregularized"})
            out.append(dict(w, regularized=False))
    return out, info


def endpoints(w):
    return w["centroid"] + w["s_min"] * w["direction"], w["centroid"] + w["s_max"] * w["direction"]


def snap_corners(walls: list[dict], cfg: dict) -> list[dict]:
    """Snap wall endpoints to the intersection with a non-parallel wall when within corner_snap_m."""
    lim = cfg["walls"]["corner_snap_m"]
    res = []
    for i, w in enumerate(walls):
        s = [w["s_min"], w["s_max"]]
        ends = endpoints(w)
        for k in (0, 1):
            best = None
            for j, o in enumerate(walls):
                if j == i or abs(w["direction"] @ o["direction"]) > np.cos(np.radians(60)):
                    continue
                A = np.array([w["normal"], o["normal"]])
                try:
                    X = np.linalg.solve(A, -np.array([w["d"], o["d"]]))
                except np.linalg.LinAlgError:
                    continue
                dist = np.linalg.norm(X - ends[k])
                if dist <= lim and (best is None or dist < best[0]):
                    best = (dist, X)
            if best is not None:
                s[k] = float((best[1] - w["centroid"]) @ w["direction"])
        res.append(dict(w, s_min=s[0], s_max=s[1], corner_snapped=bool(s != [w["s_min"], w["s_max"]])))
    return res


def wall_length(w) -> float:
    return float(w["s_max"] - w["s_min"])


def dedupe_walls(walls: list[dict], cfg: dict) -> tuple[list[dict], list[dict]]:
    """Drop weaker walls that duplicate a stronger parallel wall (same offset, overlapping extent) or are negligible vs the best wall."""
    wc = cfg["walls"]
    keep, dropped = [], []
    for w in sorted(walls, key=lambda x: -x["n_cells"]):
        reason = None
        if keep and w["n_cells"] < wc["min_support_frac_of_best"] * keep[0]["n_cells"]:
            reason = "weak_vs_best"
        else:
            for k in keep:
                if abs(w["direction"] @ k["direction"]) < np.cos(np.radians(wc["duplicate_angle_deg"])):
                    continue
                off = abs(k["normal"] @ w["centroid"] + k["d"])  # perpendicular offset of w from k's line
                pa = ((w["centroid"] + np.array([w["s_min"], w["s_max"]])[:, None] * w["direction"]) - k["centroid"]) @ k["direction"]
                lo, hi = max(pa.min(), k["s_min"]), min(pa.max(), k["s_max"])
                overlap = max(hi - lo, 0.0) / max(pa.max() - pa.min(), 1e-9)
                if off < wc["duplicate_offset_m"] and overlap > 0.5:
                    reason = f"duplicate_of_{k.get('id', '?')}"
                    break
        (dropped if reason else keep).append(w if not reason else dict(w, drop_reason=reason))
    for i, w in enumerate(keep):
        w["id"] = f"wall_{i:02d}"
    return keep, [{"drop_reason": d["drop_reason"], "n_cells": d["n_cells"]} for d in dropped]


def prune_non_boundary(walls: list[dict], origins: np.ndarray, cfg: dict, n_rays: int = 360, max_origins: int = 40) -> tuple[list[dict], dict]:
    """Keep walls that are first-hit by rays cast from the camera positions (finite segments). `origins` is (K,2) or (2,).
    Walls seen only through openings (outside the room) are rarely first-hit, so they fall below min_boundary_ray_frac."""
    origins = np.asarray(origins, float).reshape(-1, 2)
    if len(origins) > max_origins:
        origins = origins[np.linspace(0, len(origins) - 1, max_origins).astype(int)]
    if len(walls) < 3:
        return walls, {"pruned": [], "note": "fewer than 3 walls; nothing pruned"}
    ang = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    dirs = np.column_stack((np.cos(ang), np.sin(ang)))  # (R,2)
    K = len(origins)
    best_t = np.full((K, n_rays), np.inf)
    best_i = np.full((K, n_rays), -1)
    for i, w in enumerate(walls):
        a, b = endpoints(w)
        e = b - a
        denom = dirs[:, 0] * (-e[1]) + dirs[:, 1] * e[0]  # (R,)
        ao = a - origins  # (K,2)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (ao[:, 0:1] * (-e[1]) + ao[:, 1:2] * e[0]) / denom
            u = (ao[:, 1:2] * dirs[:, 0] - ao[:, 0:1] * dirs[:, 1]) / denom
        ok = (np.abs(denom) > 1e-9) & (t > 0) & (u >= -0.02) & (u <= 1.02) & (t < best_t)
        best_t[ok], best_i[ok] = t[ok], i
    cover = np.array([(best_i == i).mean() for i in range(len(walls))])
    lim = cfg["walls"]["min_boundary_ray_frac"]
    keep = [w for w, c in zip(walls, cover) if c >= lim]
    pruned = [{"id": w["id"], "ray_fraction": float(c)} for w, c in zip(walls, cover) if c < lim]
    return keep, {"pruned": pruned, "ray_fractions": {w["id"]: float(c) for w, c in zip(walls, cover)}, "origins": int(K)}


# --------------------------------------------------------------------------------------------------------------------
# Axis-based wall extraction (walls.method: manhattan_axes). The room frame is estimated FIRST from the cells themselves; walls are
# then peaks of the 1-D offset histograms along the two axes. Duplicates are removed structurally (non-maximum suppression in
# offset) and a line that does not conform to the axes is rejected and logged instead of being kept.
# --------------------------------------------------------------------------------------------------------------------

def _axis_score(P: np.ndarray, theta: float, bin_m: float) -> float:
    """Sharpness of the offset histograms along the two directions of a frame rotated by theta: sum of squared bin counts / N^2
    (probability that two random cells share a bin), averaged over two bin phases so a wall is not split by a bin edge."""
    t = np.array([np.cos(theta), np.sin(theta)])
    n = np.array([-t[1], t[0]])
    tot = 0.0
    for v in (P @ n, P @ t):
        v = v - v.min()
        for ph in (0.0, 0.5):
            b = np.floor(v / bin_m + ph).astype(np.int64)
            h = np.bincount(b)
            tot += 0.5 * float((h.astype(float) ** 2).sum())
    return tot / len(P) ** 2


def dominant_axes(cells: np.ndarray, cfg: dict) -> dict:
    """Estimate the Manhattan frame angle phi in [0, 90) deg. `contrast` = best score / median score over all angles (1 ~ no
    structure). `reliable` is False when contrast < walls.min_axis_contrast (clutter / non-rectilinear room): callers must then
    fall back and say so, not trust the axes."""
    wc = cfg["walls"]
    step = float(wc.get("axis_search_step_deg", 0.25))
    bin_m = float(wc.get("axis_hist_bin_m", 0.04))
    if len(cells) < 20:
        return {"reliable": False, "phi_deg": None, "phi_rad": None, "contrast": 0.0, "reason": f"only {len(cells)} wall cells"}
    rng = np.random.default_rng(cfg["planes"]["seed"] + 2)
    P = cells if len(cells) <= 60000 else cells[rng.choice(len(cells), 60000, replace=False)]
    angs = np.arange(0.0, 90.0, step)
    sc = np.array([_axis_score(P, np.radians(a), bin_m) for a in angs])
    k = int(np.argmax(sc))
    fine = np.arange(angs[k] - step, angs[k] + step + 1e-9, step / 5.0)
    fsc = np.array([_axis_score(P, np.radians(a), bin_m) for a in fine])
    j = int(np.argmax(fsc))
    phi_deg = float(fine[j] % 90.0)
    contrast = float(fsc[j] / max(np.median(sc), 1e-12))
    ok = contrast >= float(wc.get("min_axis_contrast", 1.5))
    return {"reliable": bool(ok), "phi_deg": phi_deg, "phi_rad": float(np.radians(phi_deg)), "contrast": contrast, "score": float(fsc[j]),
            "median_score": float(np.median(sc)), "bin_m": bin_m, "step_deg": step,
            "reason": None if ok else f"axis contrast {contrast:.2f} < min_axis_contrast {wc.get('min_axis_contrast', 1.5)}"}


def _free_fit(pts: np.ndarray, cfg: dict) -> dict:
    """Unconstrained least-squares line through the inlier cells (the 'raw' geometry that regularization never overwrites)."""
    wc = cfg["walls"]
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    t = vt[0]
    n = np.array([-t[1], t[0]])
    d = float(-n @ c)
    s0, s1 = _extent(pts, t, c, wc["endpoint_trim_frac"], wc["extent_gap_split_m"], wc.get("bridge_min_run_m"), wc.get("bridge_min_occupancy", 0.5))
    return {"normal": n, "d": d, "direction": t, "centroid": c, "s_min": s0, "s_max": s1, "n_cells": int(len(pts)),
            "residual_std": float(np.std(pts @ n + d)), "_pts": pts}


def detect_walls_axes(cells: np.ndarray, phi_rad: float, cfg: dict) -> tuple[list[dict], list[dict], dict]:
    """Walls as peaks of the offset histograms along the two Manhattan axes.
    Returns (raw_walls, regularized_walls, info), aligned 1:1 by id. raw = free-angle fit of the inlier cells (kept as measured);
    regularized = exactly axis-aligned. A peak whose free fit deviates from its axis by more than manhattan.snap_tolerance_deg is
    rejected as non_conforming (logged), never kept."""
    wc = cfg["walls"]
    c = float(wc["cell_m"])
    capture = float(wc.get("axis_capture_m", 0.05))
    tol = np.radians(wc["manhattan"]["snap_tolerance_deg"])
    min_cells = int(wc["min_wall_length_m"] / c * 0.6)
    sup = max(int(round(wc["duplicate_offset_m"] / c)), 1)
    w_bins = max(int(round(capture / c)), 1)
    cands, rejected, suppressed = [], [], 0
    for ax in (0, 1):
        a = phi_rad + ax * np.pi / 2
        t = np.array([np.cos(a), np.sin(a)])
        n = np.array([-t[1], t[0]])
        u, s = cells @ n, cells @ t
        lo = float(u.min())
        nb = int(np.ceil((u.max() - lo) / c)) + 1
        h, _ = np.histogram(u, bins=nb, range=(lo, lo + nb * c))
        sm = np.convolve(h, np.ones(2 * w_bins + 1), mode="same").astype(float)
        used = np.zeros(nb, bool)
        for _ in range(int(wc["max_walls"])):
            masked = np.where(used, -1.0, sm)
            k = int(np.argmax(masked))
            if masked[k] < min_cells:
                break
            used[max(k - sup, 0):k + sup + 1] = True  # non-maximum suppression: one wall per offset neighbourhood
            u0 = lo + (k + 0.5) * c
            for _it in range(2):
                m = np.abs(u - u0) < capture
                if not m.any():
                    break
                u0 = float(np.median(u[m]))
            m = np.abs(u - u0) < capture
            if m.sum() < min_cells:
                continue
            pts = cells[m]
            raw = _free_fit(pts, cfg)
            dev = float(np.arccos(np.clip(abs(raw["direction"] @ t), -1, 1)))
            if dev > tol:
                rejected.append({"axis": ax, "offset_m": u0, "reason": "non_conforming_angle", "deviation_deg": float(np.degrees(dev)), "n_cells": int(m.sum())})
                continue
            cen = pts.mean(0)
            cen = cen - (n @ cen - u0) * n  # on the axis-aligned line n.x = u0
            s0, s1 = _extent(pts, t, cen, wc["endpoint_trim_frac"], wc["extent_gap_split_m"], wc.get("bridge_min_run_m"), wc.get("bridge_min_occupancy", 0.5))
            if s1 - s0 < wc["min_wall_length_m"]:
                rejected.append({"axis": ax, "offset_m": u0, "reason": "too_short", "length_m": float(s1 - s0), "n_cells": int(m.sum())})
                continue
            reg = {"normal": n.copy(), "d": float(-u0), "direction": t.copy(), "centroid": cen, "s_min": s0, "s_max": s1, "n_cells": int(m.sum()),
                   "residual_std": float(np.std(u[m] - u0)), "_pts": pts, "regularized": True, "snapped_deg": float(np.degrees(dev)), "axis": ax}
            cands.append((raw, reg))
    cands.sort(key=lambda rr: -rr[1]["n_cells"])
    best = cands[0][1]["n_cells"] if cands else 0
    raws, regs = [], []
    for raw, reg in cands:
        if reg["n_cells"] < wc["min_support_frac_of_best"] * best:
            rejected.append({"axis": reg["axis"], "offset_m": float(-reg["d"]), "reason": "weak_vs_best", "n_cells": reg["n_cells"]})
            continue
        wid = f"wall_{len(raws):02d}"
        raw["id"] = reg["id"] = wid
        raws.append(raw)
        regs.append(reg)
    info = {"rejected": rejected, "n_candidates": len(cands), "suppression_offset_m": wc["duplicate_offset_m"]}
    return raws, regs, info


def close_corners(walls: list[dict], cfg: dict) -> list[dict]:
    """Extend an open wall end to the nearest perpendicular wall line when that is within walls.corner_extend_m and the meeting
    point lies on (or within corner_extend_m of) that other wall's extent. Every such extension is INFERRED, not observed: it is
    recorded in `inferred_ext_m` = [at s_min, at s_max] so the report can say how much of the outline is not measured."""
    ext = float(cfg["walls"].get("corner_extend_m", 0.0))
    res = []
    for i, w in enumerate(walls):
        s = [w["s_min"], w["s_max"]]
        inferred = [0.0, 0.0]
        if ext > 0:
            ends = endpoints(w)
            for k in (0, 1):
                best = None
                sign = 1.0 if k == 1 else -1.0
                for j, o in enumerate(walls):
                    if j == i or abs(w["direction"] @ o["direction"]) > np.cos(np.radians(60)):
                        continue
                    try:
                        X = np.linalg.solve(np.array([w["normal"], o["normal"]]), -np.array([w["d"], o["d"]]))
                    except np.linalg.LinAlgError:
                        continue
                    outward = float((X - ends[k]) @ (sign * w["direction"]))
                    if not (1e-6 < outward <= ext):
                        continue
                    so = float((X - o["centroid"]) @ o["direction"])
                    if so < o["s_min"] - ext or so > o["s_max"] + ext:
                        continue
                    if best is None or outward < best[0]:
                        best = (outward, X)
                if best is not None:
                    s[k] = float((best[1] - w["centroid"]) @ w["direction"])
                    inferred[k] = best[0]
        res.append(dict(w, s_min=s[0], s_max=s[1], inferred_ext_m=inferred))
    return res


def drop_camera_crossed(walls: list[dict], cam_uv: np.ndarray, cells: np.ndarray, cfg: dict) -> tuple[list[dict], dict]:
    """A camera cannot walk through a wall. A wall segment is DROPPED when the camera path (consecutive camera positions, in time order)
    crosses it at a point backed by wall-band cells. A crossing where the wall has no cell support (a doorway / gap) is allowed, so the
    camera walking from room to room does not delete the wall it passed through. Runs before ray-based pruning, because interior lines
    (furniture fronts, doubled surfaces) otherwise win the first-hit contest and push the true perimeter below the pruning threshold.
    Config (walls.camera_crossing): enabled, support_radius_m, min_support_cells. Every drop is recorded with its crossing count."""
    cc = cfg["walls"].get("camera_crossing", {})
    cam = np.asarray(cam_uv, float).reshape(-1, 2)
    if not cc.get("enabled", False) or len(cam) < 2 or not walls:
        return walls, {"enabled": bool(cc.get("enabled", False)), "dropped": []}
    rad, need = float(cc.get("support_radius_m", 0.25)), int(cc.get("min_support_cells", 8))
    p0, p1 = cam[:-1], cam[1:]
    keep, dropped = [], []
    for w in walls:
        a, b = endpoints(w)
        e = b - a
        d = p1 - p0
        den = d[:, 0] * e[1] - d[:, 1] * e[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((a[0] - p0[:, 0]) * e[1] - (a[1] - p0[:, 1]) * e[0]) / den   # along the camera step
            u = ((a[0] - p0[:, 0]) * d[:, 1] - (a[1] - p0[:, 1]) * d[:, 0]) / den  # along the wall
        hit = (np.abs(den) > 1e-12) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
        n_supported = n_gap = 0
        # cells that belong to this wall's line (within 0.10 m of it), used to test support around each crossing point
        near = cells[np.abs(cells @ w["normal"] + w["d"]) < 0.10]
        for k in np.flatnonzero(hit):
            X = p0[k] + t[k] * d[k]
            sup = int((np.linalg.norm(near - X, axis=1) < rad).sum()) if len(near) else 0
            if sup >= need:
                n_supported += 1
            else:
                n_gap += 1
        if n_supported:
            dropped.append({"id": w["id"], "crossings_on_wall_material": n_supported, "crossings_in_gaps": n_gap, "length_m": float(wall_length(w))})
        else:
            keep.append(w)
    return keep, {"enabled": True, "dropped": dropped, "support_radius_m": rad, "min_support_cells": need}


def high_band_cells(L: np.ndarray, ceiling_h: float | None, cfg: dict) -> np.ndarray | None:
    """Occupied 2-D cells from points ABOVE furniture height (walls.high_band.min_h_m .. ceiling - margin), no column filter. In that band almost only the
    room's perimeter walls exist, so it tells a wall that runs to the ceiling from the face of a wardrobe / shelf / bed headboard that merely stands in front
    of it. Needs an observed ceiling and enough points; returns None (filter skipped, loudly) otherwise."""
    hb = cfg["walls"].get("high_band", {})
    if not hb.get("enabled", False) or ceiling_h is None:
        return None
    lo, hi = float(hb["min_h_m"]), ceiling_h - float(hb["ceiling_margin_m"])
    if hi - lo < 0.25:
        return None
    P = L[(L[:, 2] >= lo) & (L[:, 2] <= hi), :2]
    c = cfg["walls"]["cell_m"]
    if len(P) == 0:
        return None
    cells = (np.unique(np.floor(P / c).astype(np.int64), axis=0) + 0.5) * c
    return cells if len(cells) >= int(hb.get("min_cells", 200)) else None


def high_band_support(w: dict, hb_cells: np.ndarray, cfg: dict) -> float:
    """Fraction of the wall's length (0.1 m bins) that has high-band cells within support_tol_m of its line."""
    hb = cfg["walls"]["high_band"]
    sd = hb_cells @ w["normal"] + w["d"]
    s = (hb_cells - w["centroid"]) @ w["direction"]
    on = (np.abs(sd) <= hb["support_tol_m"]) & (s >= w["s_min"]) & (s <= w["s_max"])
    n_bins = max(int(np.ceil((w["s_max"] - w["s_min"]) / 0.1)), 1)
    if not on.any():
        return 0.0
    occ = np.unique(np.floor((s[on] - w["s_min"]) / 0.1).astype(int).clip(0, n_bins - 1))
    return float(len(occ) / n_bins)


def filter_by_high_band(walls: list[dict], hb_cells: np.ndarray | None, cfg: dict) -> tuple[list[dict], dict]:
    """Drop candidate walls that exist only at furniture height. Skipped (and said so) when the high band is unavailable or when fewer than
    walls.high_band.min_supported_walls candidates are supported, because then the evidence is too thin to decide and dropping would be a guess."""
    hb = cfg["walls"].get("high_band", {})
    if not hb.get("enabled", False):
        return walls, {"enabled": False}
    if hb_cells is None:
        return walls, {"enabled": True, "applied": False, "reason": "no usable high band (ceiling unobserved, too few points, or band empty)"}
    sup = {w["id"]: high_band_support(w, hb_cells, cfg) for w in walls}
    ok = [w for w in walls if sup[w["id"]] >= hb["min_support"]]
    info = {"enabled": True, "applied": True, "high_band_cells": int(len(hb_cells)), "support": {k: round(v, 3) for k, v in sup.items()},
            "dropped": [{"id": w["id"], "support": round(sup[w["id"]], 3), "length_m": float(wall_length(w))} for w in walls if sup[w["id"]] < hb["min_support"]]}
    if len(ok) < int(hb.get("min_supported_walls", 2)):
        info.update(applied=False, reason=f"only {len(ok)} walls have high-band support (< {hb.get('min_supported_walls', 2)}); filter not applied", dropped=[])
        return walls, info
    return ok, info


def trim_overshoot(walls: list[dict], cfg: dict, interior_uv: np.ndarray | None = None) -> tuple[list[dict], list[dict]]:
    """Room walls are measured corner to corner. A fitted line often runs on past a perpendicular wall (the cells continue behind the corner: a neighbouring
    space seen through a door, the next room's wall on the same line). Cut each wall end at the nearest perpendicular wall that (a) crosses its line within
    [min_m, max_m] of that end, (b) covers the crossing (+/- tol_m) and (c) reaches at least `arm_m` from the crossing towards the room interior (the camera
    side). (c) separates a room corner from, say, a corridor wall that touches the line from the outside at a doorway. Longer runs than max_m are left alone
    (more likely a through wall / interior partition). Every cut is recorded; geometry is never extended here."""
    tc = cfg["walls"].get("trim_overshoot", {})
    if not tc.get("enabled", False):
        return walls, []
    lo_m, hi_m, tol, arm_m = float(tc["min_m"]), float(tc["max_m"]), float(tc["tol_m"]), float(tc.get("arm_m", 0.5))
    keep_min = float(cfg["walls"].get("min_wall_length_m", 0.6))
    inside = None if interior_uv is None or len(interior_uv) == 0 else np.asarray(interior_uv, float).reshape(-1, 2).mean(0)
    out, log_ = [], []
    for i, w in enumerate(walls):
        s = [w["s_min"], w["s_max"]]
        cuts = [0.0, 0.0]
        side = 1.0 if inside is None else (np.sign(w["normal"] @ inside + w["d"]) or 1.0)
        for k in (0, 1):
            best = None
            for j, o in enumerate(walls):
                if j == i or abs(w["direction"] @ o["direction"]) > np.cos(np.radians(60)):
                    continue
                try:
                    X = np.linalg.solve(np.array([w["normal"], o["normal"]]), -np.array([w["d"], o["d"]]))
                except np.linalg.LinAlgError:
                    continue
                sx = float((X - w["centroid"]) @ w["direction"])
                so = float((X - o["centroid"]) @ o["direction"])
                over = (w["s_max"] - sx) if k == 1 else (sx - w["s_min"])          # how far the wall runs past the crossing at this end
                if not (lo_m <= over <= hi_m and o["s_min"] - tol <= so <= o["s_max"] + tol):
                    continue
                kept_m = (sx - w["s_min"]) if k == 1 else (w["s_max"] - sx)               # what would remain of this wall after the cut
                if kept_m < max(over, keep_min):                                          # never cut away most of a wall (a stub meeting a wall at its far end)
                    continue
                if inside is not None:
                    arm = max(side * float(w["normal"] @ e + w["d"]) for e in endpoints(o))   # reach of the other wall towards the interior side
                    if arm < arm_m:
                        continue
                if best is None or over < best[0]:
                    best = (over, sx)
            if best is not None:
                s[k], cuts[k] = best[1], best[0]
        if any(cuts):
            log_.append({"id": w["id"], "trimmed_m": cuts})
        out.append(dict(w, s_min=s[0], s_max=s[1], trimmed_overshoot_m=cuts))
    return out, log_
