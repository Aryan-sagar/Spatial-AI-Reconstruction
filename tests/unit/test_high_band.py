"""High-band perimeter filter. SYNTHETIC: a deep full-height wardrobe stands along a whole wall. Its FACE is a strong, well-supported line 0.5 m inside the real
wall, and it blocks every ray to the real wall, so the ray-based boundary pruning keeps the wardrobe face and drops the wall (the failure seen on the first real
benchmark room). Above the wardrobe the real wall is visible, which is what the filter uses."""
import numpy as np
from applied_ai.config import load_config
from applied_ai.geometry.floorplan import floor_footprint
from applied_ai.geometry.room_layout import extract_walls
from applied_ai.geometry.walls import wall_length
from synth_cloud import _rect

W, D, H, DEEP = 5.9, 3.7, 2.8, 0.5


def _room(rot_deg=18.0, seed=0, wardrobe_h=2.0):
    rng = np.random.default_rng(seed)
    P = []

    def vwall(p0, p1, z0, z1, cut=None):
        p0, p1 = np.array(p0, float), np.array(p1, float)
        d = p1 - p0
        pts = _rect(rng, [p0[0], p0[1], z0], [d[0], d[1], 0], [0, 0, z1 - z0])
        nrm = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        pts[:, :2] += rng.normal(0, 0.008, (len(pts), 1)) * nrm
        return pts if cut is None else pts[~cut(pts)]

    behind = lambda p: (p[:, 2] < wardrobe_h) & (p[:, 1] > 0.3) & (p[:, 1] < D - 0.3)      # the wall below wardrobe height is hidden behind it
    P += [vwall((0, 0), (W, 0), 0, H), vwall((0, D), (W, D), 0, H), vwall((0, 0), (0, D), 0, H, cut=behind), vwall((W, 0), (W, D), 0, H)]
    P += [vwall((DEEP, 0.3), (DEEP, D - 0.3), 0, wardrobe_h), vwall((0, 0.3), (DEEP, 0.3), 0, wardrobe_h), vwall((0, D - 0.3), (DEEP, D - 0.3), 0, wardrobe_h)]
    F = _rect(rng, [DEEP, 0, 0], [W - DEEP, 0, 0], [0, D, 0], 1 / 0.03 ** 2 / 2)
    F[:, 2] = rng.normal(0, 0.005, len(F))
    P.append(F)
    ceil = _rect(rng, [0, 0, H], [W, 0, 0], [0, D, 0], 1 / 0.05 ** 2)
    ceil[:, 2] = H + rng.normal(0, 0.005, len(ceil))
    P.append(ceil)
    L = np.vstack(P)
    a = np.radians(rot_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    shift = np.array([-1.0, -1.5])
    L[:, :2] = L[:, :2] @ R.T + shift
    th = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    cam = np.column_stack((3.1 + 1.7 * np.cos(th), 1.85 + 1.0 * np.sin(th))) @ R.T + shift
    return L, cam


def _extent(wl):
    """Distance between the two long-wall lines (the room width as the walls say it)."""
    longs = [w for w in wl["final"] if wall_length(w) > 4.0]
    return longs


def test_without_the_high_band_the_wardrobe_face_replaces_the_real_wall():
    cfg = load_config()
    cfg["walls"]["high_band"]["enabled"] = False
    L, cam = _room()
    wl = extract_walls(L, cam, H, cfg)
    fp = floor_footprint(L, wl["final"], cfg, cam)
    assert abs(fp["area_m2"] / (W * D) - 1) > 0.05            # documented failure: the wall-to-wall area is wrong by the wardrobe depth


def test_with_the_high_band_the_real_perimeter_is_recovered():
    cfg = load_config()
    L, cam = _room()
    wl = extract_walls(L, cam, H, cfg)
    hb = wl["prune_info"]["high_band"]
    assert hb["enabled"] and hb["applied"] and any(d["length_m"] > 3 for d in hb["dropped"])    # the wardrobe face was dropped as furniture-height-only
    fp = floor_footprint(L, wl["final"], cfg, cam)
    assert abs(fp["area_m2"] / (W * D) - 1) < 0.03
    lens = sorted(round(wall_length(w), 2) for w in wl["final"])
    assert min(abs(x - W) for x in lens) < 0.1 and min(abs(x - D) for x in lens) < 0.1


def test_filter_is_skipped_and_says_so_when_the_ceiling_is_unobserved():
    cfg = load_config()
    L, cam = _room()
    wl = extract_walls(L, cam, None, cfg)
    hb = wl["prune_info"]["high_band"]
    assert hb["enabled"] and hb["applied"] is False and "ceiling unobserved" in hb["reason"]


def test_a_wall_that_only_reaches_wardrobe_height_everywhere_is_not_wiped_out():
    """If almost nothing has high-band support (e.g. a low ceiling scan), fewer than min_supported_walls survive and the filter must not apply."""
    cfg = load_config()
    L, cam = _room()
    cfg["walls"]["high_band"]["min_h_m"] = 2.75                 # band ~empty: nothing has support
    cfg["walls"]["high_band"]["ceiling_margin_m"] = 0.0
    wl = extract_walls(L, cam, H, cfg)
    assert wl["prune_info"]["high_band"]["applied"] is False and len(wl["final"]) >= 2
