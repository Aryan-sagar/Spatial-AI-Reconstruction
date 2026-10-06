"""Wall extraction / footprint closure: unit tests per mechanism + a furnished synthetic room (logic only, NOT real-data accuracy)."""
import copy

import numpy as np
import pytest
from applied_ai.config import load_config
from applied_ai.geometry.floorplan import floor_footprint, wall_enclosed_footprint
from applied_ai.geometry.room_layout import extract_walls
from applied_ai.geometry.walls import (_extent, close_corners, detect_walls_axes, dominant_axes, manhattan_regularize, wall_band_cells,
                                       wall_length)
from synth_cloud import furnished_room


def _rot(a_deg):
    a = np.radians(a_deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def _rect_cells(W=4.0, D=3.0, rot=0.0, c=0.02, rng=None):
    xs, ys = np.arange(0, W + c, c), np.arange(0, D + c, c)
    P = np.vstack([np.column_stack((xs, np.zeros_like(xs))), np.column_stack((xs, np.full_like(xs, D))),
                   np.column_stack((np.zeros_like(ys), ys)), np.column_stack((np.full_like(ys, W), ys))])
    if rng is not None:
        P = P + rng.normal(0, 0.006, P.shape)
    return P @ _rot(rot).T


def test_dominant_axes_recovers_rotation_and_flags_unstructured():
    cfg = load_config()
    ax = dominant_axes(_rect_cells(rot=23.0, rng=np.random.default_rng(0)), cfg)
    assert ax["reliable"] and ax["phi_deg"] == pytest.approx(23.0, abs=0.5) and ax["contrast"] > 3
    th = np.random.default_rng(1).uniform(0, 2 * np.pi, 3000)  # a round room has no preferred axes
    ring = 2.5 * np.column_stack((np.cos(th), np.sin(th)))
    bad = dominant_axes(ring, cfg)
    assert not bad["reliable"] and bad["reason"]


def test_axis_walls_merge_thick_band_and_reject_off_axis_line():
    cfg = load_config()
    rng = np.random.default_rng(2)
    cells = _rect_cells(5.9, 3.7, 0.0, rng=rng)
    band = np.vstack([np.column_stack((np.arange(1.0, 4.0, 0.02), np.full(150, 3.7 - d))) for d in (0.04, 0.08, 0.12)])  # furniture front ~ thick band
    t = np.radians(-35.0)
    diag = np.array([2.0, 1.5]) + np.arange(0, 1.5, 0.02)[:, None] * np.array([np.cos(t), np.sin(t)])               # off-axis free-standing panel
    cells = np.vstack([cells, band, diag])
    ax = dominant_axes(cells, cfg)
    raw, reg, info = detect_walls_axes(cells, ax["phi_rad"], cfg)
    assert len(reg) == 4 and sorted(round(wall_length(w)) for w in reg) == [4, 4, 6, 6]
    for w in reg:  # regularized walls are exactly axis-aligned, raw (free-angle) fit is retained and differs slightly
        assert abs(abs(w["direction"] @ np.array([np.cos(ax["phi_rad"]), np.sin(ax["phi_rad"])])) - 1) < 1e-9 or abs(w["direction"] @ np.array([-np.sin(ax["phi_rad"]), np.cos(ax["phi_rad"])])) > 0.999999
    assert all(r["id"] == g["id"] for r, g in zip(raw, reg)) and all("_pts" in r for r in raw)


def test_hidden_wall_stretch_is_bridged_but_sparse_scatter_is_not():
    d = np.array([1.0, 0.0])
    wall = np.column_stack((np.concatenate([np.linspace(0, 2.0, 100), np.linspace(3.9, 5.9, 100)]), np.zeros(200)))   # 1.9 m hidden behind a wardrobe
    ext_legacy = _extent(wall, d, np.zeros(2), 0.0, 1.5)
    ext_bridge = _extent(wall, d, np.zeros(2), 0.0, 1.5, 0.4)
    assert ext_legacy[1] - ext_legacy[0] < 2.1 and ext_bridge[1] - ext_bridge[0] == pytest.approx(5.88, abs=0.05)
    scatter = np.array([[7.0, 0], [7.3, 0], [8.1, 0]])  # three stray cells 1.1 m beyond the wall end: a 1.1 m 'span' but only 3 occupied bins of 11
    ext = _extent(np.vstack([wall, scatter]), d, np.zeros(2), 0.0, 1.5, 0.4)
    assert ext[1] < 6.0


def test_column_filter_rejects_table_top_and_low_bed_keeps_wall():
    cfg = load_config()
    rng = np.random.default_rng(3)
    wall = np.column_stack((rng.uniform(0, 4, 3000), rng.normal(0, 0.005, 3000), rng.uniform(0, 2.6, 3000)))
    table = np.column_stack((rng.uniform(1, 2.2, 3000), rng.uniform(1, 1.8, 3000), np.full(3000, 0.75)))        # horizontal top surface
    bed = np.column_stack((rng.uniform(2.5, 3.5, 3000), rng.uniform(0.5, 2.0, 3000), rng.uniform(0.0, 0.55, 3000)))  # 0.55 m tall
    L = np.vstack([wall, table, bed])
    st: dict = {}
    on, _ = wall_band_cells(L, None, cfg, 2.6, stats=st)
    off_cfg = copy.deepcopy(cfg); off_cfg["walls"]["min_column_extent_m"] = 0.0
    off, _ = wall_band_cells(L, None, off_cfg, 2.6)
    assert len(off) > 4 * len(on) and st["points_rejected_low_profile"] > 3000
    assert (on[:, 1] > 0.3).sum() == 0 or (on[:, 1] > 0.3).sum() < 0.05 * len(on)   # (almost) nothing left away from the wall line
    assert (np.abs(on[:, 1]) < 0.05).sum() > 150  # the wall line survives


def test_non_conforming_wall_is_dropped_and_logged_not_kept_silently():
    def w(i, ang_deg, n_cells=300):
        t = np.array([np.cos(np.radians(ang_deg)), np.sin(np.radians(ang_deg))])
        pts = np.arange(0, 3, 0.02)[:, None] * t
        return {"id": f"w{i}", "normal": np.array([-t[1], t[0]]), "d": 0.0, "direction": t, "centroid": pts.mean(0), "s_min": -1.5, "s_max": 1.5,
                "n_cells": n_cells, "residual_std": 0.01, "_pts": pts}
    walls = [w(0, 0.5), w(1, 90.2), w(2, 0.0), w(3, -35.0, 120)]
    cfg = {"walls": {"endpoint_trim_frac": 0.0, "extent_gap_split_m": 1.5, "manhattan": {"enabled": True, "snap_tolerance_deg": 6.0, "drop_non_conforming": True}}}
    reg, info = manhattan_regularize(walls, cfg)
    assert [x["id"] for x in reg] == ["w0", "w1", "w2"] and info["non_conforming"][0]["id"] == "w3" and info["non_conforming"][0]["action"] == "dropped"
    cfg["walls"]["manhattan"]["drop_non_conforming"] = False  # legacy behaviour: kept un-regularized, but still logged
    reg2, info2 = manhattan_regularize(walls, cfg)
    assert len(reg2) == 4 and info2["non_conforming"][0]["action"] == "kept_unregularized"


def _axis_wall(i, p0, p1):
    p0, p1 = np.array(p0, float), np.array(p1, float)
    t = (p1 - p0) / np.linalg.norm(p1 - p0)
    n = np.array([-t[1], t[0]])
    c = (p0 + p1) / 2
    h = np.linalg.norm(p1 - p0) / 2
    return {"id": f"w{i}", "normal": n, "d": float(-n @ c), "direction": t, "centroid": c, "s_min": -h, "s_max": h, "n_cells": 200, "residual_std": 0.01}


def test_close_corners_extends_open_ends_and_records_inferred_length():
    cfg = {"walls": {"corner_extend_m": 1.0}}
    a, b = _axis_wall(0, (0, 0), (3.5, 0)), _axis_wall(1, (4.0, 0), (4.0, 3))          # 0.5 m short of the perpendicular wall
    out = close_corners([a, b], cfg)
    assert out[0]["s_max"] - out[0]["s_min"] == pytest.approx(4.0, abs=1e-6) and out[0]["inferred_ext_m"][1] == pytest.approx(0.5, abs=1e-6)
    far = close_corners([_axis_wall(0, (0, 0), (2.5, 0)), b], cfg)                      # 1.5 m gap > corner_extend_m: NOT invented
    assert far[0]["inferred_ext_m"] == [0.0, 0.0] and far[0]["s_max"] - far[0]["s_min"] == pytest.approx(2.5)
    miss = close_corners([a, _axis_wall(1, (4.0, 2.0), (4.0, 5.0))], cfg)               # the other wall does not reach the line: no extension
    assert miss[0]["inferred_ext_m"] == [0.0, 0.0]
    off = close_corners([a, b], {"walls": {"corner_extend_m": 0.0}})
    assert off[0]["inferred_ext_m"] == [0.0, 0.0]


def _box(x0, y0, x1, y1, i0=0):
    return [_axis_wall(i0, (x0, y0), (x1, y0)), _axis_wall(i0 + 1, (x1, y0), (x1, y1)), _axis_wall(i0 + 2, (x1, y1), (x0, y1)), _axis_wall(i0 + 3, (x0, y1), (x0, y0))]


def test_enclosure_takes_the_region_holding_the_camera_path_and_reports_outside_evidence():
    cfg = load_config()
    walls = _box(0, 0, 4, 3) + _box(5, 0, 8, 3, 4)                 # room A (scanned) and room B (seen through a door, never visited)
    cam = np.column_stack((np.linspace(0.5, 3.5, 20), np.full(20, 1.5)))
    enc = wall_enclosed_footprint(walls, cfg, cam)
    assert enc["area_m2"] == pytest.approx(12.0, rel=0.03) and enc["n_enclosed_regions"] == 2 and enc["camera_fraction_inside"] == 1.0
    rng = np.random.default_rng(0)
    A = np.column_stack((rng.uniform(0.05, 3.95, 20000), rng.uniform(0.05, 2.95, 20000), rng.normal(0, 0.005, 20000)))
    B = np.column_stack((rng.uniform(5.05, 7.95, 20000), rng.uniform(0.05, 2.95, 20000), rng.normal(0, 0.005, 20000)))
    fp = floor_footprint(np.vstack([A, B]), walls, cfg, cam)
    assert fp["method"] == "wall_enclosed_region" and fp["area_m2"] == pytest.approx(12.0, rel=0.03)    # NOT 21 m2 (evidence of both rooms)
    assert any("outside the enclosed region" in w for w in fp["warnings"])
    open_walls = walls[:2] + walls[3:4]                                                                  # 3 walls only: nothing enclosed -> loud fallback
    fp2 = floor_footprint(A, open_walls, cfg, cam)
    assert fp2["method"] != "wall_enclosed_region" and any("do not enclose" in w for w in fp2["warnings"])


def _run_layout(cfg, **kw):
    L, cam, gt = furnished_room(**kw)
    wl = extract_walls(L, cam, 2.6, cfg)
    return wl, floor_footprint(L, wl["final"], cfg, cam), gt


def _legacy(cfg):
    c = copy.deepcopy(cfg)
    c["walls"].update(method="legacy_ransac", corner_extend_m=0.0, min_column_extent_m=0.0)
    c["walls"]["manhattan"]["drop_non_conforming"] = False
    c["floorplan"]["prefer_wall_enclosed"] = False
    return c


def test_furnished_rotated_room_outline_is_closed_and_accurate_synthetic():
    cfg = load_config()
    wl, fp, gt = _run_layout(cfg)
    assert fp["method"] == "wall_enclosed_region" and abs(fp["area_m2"] / gt["area"] - 1) < 0.03
    lens = [wall_length(w) for w in wl["final"]]
    for g in gt["wall_lengths"]:  # every true wall is found at its true length (the wardrobe-hidden north wall is bridged)
        assert min(abs(x - g) for x in lens) < 0.05
    assert all(w.get("regularized") for w in wl["final"]) and wl["minfo"]["method_used"] == "manhattan_axes"
    assert wl["minfo"]["dominant_angle_deg"] == pytest.approx(gt["rot_deg"], abs=0.7)
    assert len(wl["final"]) <= 7                                  # no pile of duplicates / table rows / bed rows
    # the corridor seen through the door is neither merged into the room nor kept as a wall of it
    assert not any(w["id"] for w in wl["final"] if wall_length(w) > 6.0)
    # honest about what is NOT measured: any inferred corner extension is recorded
    assert all("inferred_ext_m" in w for w in wl["final"])


def test_new_geometry_beats_legacy_on_the_furnished_room_and_is_stable_across_seeds_and_rotations():
    cfg = load_config()
    new_err, old_err = [], []
    for seed, rot in ((0, 18.0), (1, 5.0), (2, 37.0), (3, 63.0)):
        _, fpn, gt = _run_layout(cfg, seed=seed, rot_deg=rot)
        _, fpo, _ = _run_layout(_legacy(cfg), seed=seed, rot_deg=rot)
        new_err.append(abs(fpn["area_m2"] / gt["area"] - 1))
        old_err.append(abs(fpo["area_m2"] / gt["area"] - 1))
    assert max(new_err) < 0.03 and min(old_err) > 0.05 and max(new_err) < min(old_err)


def test_unreliable_axis_frame_falls_back_to_legacy_and_says_so():
    cfg = load_config()
    cfg["walls"]["min_axis_contrast"] = 1e9                       # force 'unreliable'
    wl, _, _ = _run_layout(cfg)
    assert wl["minfo"]["method_requested"] == "manhattan_axes" and wl["minfo"]["method_used"] == "legacy_ransac" and wl["minfo"]["axes"]["reliable"] is False


def test_turning_the_column_filter_off_exposes_the_failure_it_prevents():
    cfg = load_config()
    cfg["walls"]["min_column_extent_m"] = 0.0
    cfg["walls"]["camera_crossing"]["enabled"] = False   # isolate the column filter: the crossing filter also removes some of these ghost walls
    cfg["walls"]["high_band"]["enabled"] = False         # ...and so does the high-band filter (table/bed rows have no support above furniture height)
    cfg["walls"]["trim_overshoot"]["enabled"] = False    # ...and corner trimming changes the outline a little
    wl, fp, gt = _run_layout(cfg)
    assert len(wl["final"]) > 10 and abs(fp["area_m2"] / gt["area"] - 1) > 0.2      # table/bed rows become 'walls' and shred the outline (documented ablation)


def test_ablation_overlays_load_and_switch_the_intended_knobs():
    from pathlib import Path
    root = Path(__file__).resolve().parents[2] / "configs"
    d, old, nofilt = load_config(), load_config(root / "ablation_legacy_walls.yaml"), load_config(root / "ablation_no_column_filter.yaml")
    assert d["walls"]["method"] == "manhattan_axes" and d["walls"]["min_column_extent_m"] > 0 and d["floorplan"]["prefer_wall_enclosed"]
    assert old["walls"]["method"] == "legacy_ransac" and old["walls"]["min_column_extent_m"] == 0 and old["walls"]["corner_extend_m"] == 0
    assert old["walls"]["manhattan"]["drop_non_conforming"] is False and old["floorplan"]["prefer_wall_enclosed"] is False
    assert nofilt["walls"]["min_column_extent_m"] == 0 and nofilt["walls"]["method"] == "manhattan_axes"
    assert (root / "lidar.yaml").read_text() == (root / "default.yaml").read_text()
